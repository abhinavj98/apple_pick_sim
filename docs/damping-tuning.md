# Damping tuning: cable bend/stretch vs. FIXED-joint kd

## Document status

| Field | Value |
| ----- | ----- |
| **Last updated** | 2026-07-02 |
| **Roadmap slice** | [V].2.1.3 (material-parameter sampling) / settle stability |
| **Owner** | Abhinav |

## Summary

The fruiting system has **two decoupled dissipation layers**, on two different joint
types, addressing two different problems:

| Layer | Joint type | Knob | Purpose |
| ----- | ---------- | ---- | ------- |
| **Material bend/stretch damping** | `CABLE` (within each rod) | `damping_ratio` (ζ) → `bend_damping` / `stretch_damping` | Wood-like energy loss **along** a branch as it bends/stretches |
| **Joint (weld) damping** | `FIXED` (and other non-cable joints) | `rigid_joint_linear_kd` / `rigid_joint_angular_kd` | Numerical stability at discrete supports/welds (world anchor, rod↔rod, T-junction, stem→apple, apple→gripper proxy) |

They are **not interchangeable**. Cable ζ governs how realistic branch bending looks
(and is the knob that matters for sys-ID / hardware transfer). Joint `kd` governs how
fast the VBD penalty solver settles discretization artifacts at rigid welds. Conflating
them either fails to damp real ringing (ζ too weak on stiff segments) or corrupts wood
realism and wrench readouts (joint `kd` too large).

Stretch and bend are also independently controllable per rod segment via the optional
`vbd_stretch_fixed` fixture block (see `docs/material-parameter-sampling.md`) — stretch
can be pinned to stable VBD constants while bend stays derived from `(E, ζ)`.

## 1. Cable bend/stretch damping (`damping_ratio`)

Set per rod segment (`primary`, `spur`, `stem`) in the range JSON and consumed by
`apple_pick_sim/fruiting_system/params.py::rod_params_from_material`:

\[
c_{\text{bend}} = 2\zeta\sqrt{k_{\text{bend}} \cdot J_{\text{seg}}}, \qquad
c_{\text{stretch}} = 2\zeta\sqrt{k_{\text{stretch}} \cdot m_{\text{seg}}}
\]

where \(k_{\text{bend}} = EI/L_{\text{seg}}\), \(k_{\text{stretch}} = EA/L_{\text{seg}}\)
(see `docs/material-parameter-sampling.md` for the full geometry derivation). These
values are written into `RodParams.bend_damping` / `stretch_damping` and passed straight
into `builder.add_rod(...)` in `apple_pick_sim/fruiting_system/build.py`, which Newton
stores as `model.joint_target_kd` on the `CABLE` joints.

**Current fixture** (`fruiting_system_ranges_real_world_proxy_variance.json`) uses
`damping_ratio: {"min": 0.3, "max": 0.3}` uniformly across primary, spur, and stem — a
wood-like band rather than the far-above-critical values (5–10) used earlier purely for
stability. Stretch is pinned separately via `vbd_stretch_fixed` per segment, so ζ in
this fixture only drives **bend** damping in practice.

**Known limitation — soft segments can't reach meaningful absolute damping via ζ
alone.** Because \(c_{\text{bend}} \propto \sqrt{k_{\text{bend}}}\), and spur/stem
`k_bend` is intentionally tiny (compliant shoot / interim torsion proxy), even ζ = 0.2
yields `c_bend` on the order of 10⁻⁴–10⁻³ N·m·s/rad for stem — dynamically negligible.
**Do not chase settle-time targets by raising ζ on spur/stem past a wood-plausible
value** (e.g. 0.1–0.2, or whatever literature/sys-ID range is adopted); if a segment
needs more damping than that gives, the fix is joint `kd`, not ζ.

## 2. FIXED-joint damping (`rigid_joint_*_kd`)

Set once, globally, in `apple_pick_sim/fruiting_system/build.py`:

```python
FRUITING_VBD_RIGID_JOINT_LINEAR_KD = 5.0e-4   # N·s/m, absolute (no Rayleigh scaling)
FRUITING_VBD_RIGID_JOINT_ANGULAR_KD = 5.0e-4  # N·m·s/rad, absolute (no Rayleigh scaling)
```

passed into every `SolverVBD` constructed via `make_fruiting_solver_vbd`. Newton's
`SolverVBD._init_joint_penalty_k` copies these two scalars into the `[linear, angular]`
constraint slots of **every** `FIXED` (and `BALL`/`REVOLUTE`/`PRISMATIC`/`D6`) joint —
i.e. one number for the world anchor, every rod↔rod weld, the T-junction, stem→apple,
and apple→gripper proxy alike.

### Mechanism

VBD treats each constraint as a spring-damper on constraint error `C` (position/angle
mismatch). The damping term adds force proportional to `dC/dt` **and** stiffens the
local linearization by `kd/Δt`:

\[
K_{\text{eff}} = k + \frac{kd}{\Delta t}
\]

(see `evaluate_linear_constraint_force_hessian` / `evaluate_angular_constraint_force_hessian`
in `newton/newton/_src/solvers/vbd/rigid_vbd_kernels.py`). This is why joint `kd` is not
"free" dissipation: too large a value doesn't just remove energy faster, it also makes
the constraint numerically stiffer than the AVBD iteration count can resolve, producing
bounce/jitter rather than smooth settling.

**`solver.joint_penalty_kd` stores the raw `kd`, not `kd/Δt`.** The `1/Δt` factor is
applied fresh, every substep, inside the force/Hessian evaluator (`k_damp = damping *
inv_dt`) — it is never baked into the stored array. Concretely:

```python
# _apply_batched_joint_angular_kd_kernel (apple_pick_sim/fruiting_system/build.py)
joint_penalty_kd[c0 + angular_slot] = kd_values[k]   # raw kd, verbatim

# evaluate_angular_constraint_force_hessian (newton/_src/solvers/vbd/rigid_vbd_kernels.py)
k_damp = damping * inv_dt                            # divided live, per substep
```

So `kd` (and any override you pass to `set_fruiting_joint_angular_kd[_batched]`) is a
`Δt`-independent physical quantity — units N·m·s/rad, comparable directly against
`kd_crit = 2√(k·I)` below without any manual `× dt` or `/ dt` conversion. If the
per-substep `Δt` changes (different `--hz` / substep count), the same `kd` value keeps
the same physical damping ratio; only its numerical-stiffening margin (`kd/Δt` vs. `k`,
see above) shifts.

### Why one global scalar is a poor fit for this chain

Critical joint damping scales with the **child body's rotational inertia** at that
constraint, roughly `kd_crit ≈ 2√(k · I_child)`, with `k` fixed at
`rigid_joint_angular_ke = 1e5` for every `FIXED` joint today. Child-body inertia in a
representative seed-42 build spans **~1000×** across the chain:

| Joint (child body) | `I_child` (kg·m²) | `kd_crit` at `k = 1e5` |
| ------------------- | ------------------ | ---------------------- |
| Primary segment weld | ~1.1×10⁻⁴ | ~6.6 N·m·s/rad |
| T-junction (spur base) | ~1.8×10⁻⁶ | ~0.85 |
| Stem→apple | ~1.0×10⁻⁶ | ~0.63 |
| Stem segment weld | ~1.3×10⁻⁷ | ~0.23 |

A single `FRUITING_VBD_RIGID_JOINT_ANGULAR_KD` therefore cannot be simultaneously
correct everywhere: a value that critically damps the primary welds (~6–7) is ~10–30×
past critical for stem/apple (bouncy, over-stiff there), while a value tuned for
stem/apple (~0.2–0.6) leaves primary badly underdamped (rings for a long time). This
is the direct explanation for "small `kd` never settles, large `kd` gets bouncy."

### `k_start` caveat (currently inert)

`make_fruiting_solver_vbd` also passes `rigid_joint_linear_k_start=1.0e8` and
`rigid_joint_angular_k_start=1.0e6`. These only take effect when AVBD penalty ramping
is enabled (`rigid_avbd_linear_beta` / `rigid_avbd_angular_beta` > 0 in `SolverVBD`).
Neither is set anywhere in this codebase (default `0.0`), so **these `k_start` values
are currently dead configuration** — every `FIXED` joint uses the fixed penalty ceiling
`rigid_joint_linear_ke` / `rigid_joint_angular_ke` (default `1e5`) directly. Worth a
follow-up cleanup (remove the misleading kwargs, or actually enable ramping with a
`k_start` below `ke`), but out of scope for damping tuning itself.

## 3. Per-joint `kd` (angular and linear)

Preferred fixture path: set **`sim_build.joint_damping_ratio`** (ζ in `[0, 1]`) and
optional `joint_*_kp_overrides`. At build time
`joint_kd_from_damping_ratio` expands

\[
k_{d,\mathrm{ang}} = \zeta\,2\sqrt{k_{\mathrm{ang}} I_{\mathrm{child}}},\quad
k_{d,\mathrm{lin}} = \zeta\,2\sqrt{k_{\mathrm{lin}} m_{\mathrm{child}}}
\]

using intended `kp` per role (else Newton `ke=1e5`), then distal roles scale
`kd ∝ √(E/E_ref)` via `scale_joint_kd_overrides`. Absolute
`joint_angular_kd_overrides` / `joint_linear_kd_overrides` remain supported but are
**mutually exclusive** with `joint_damping_ratio`. Variance fixture ships
`joint_damping_ratio: 1.0` (critical weld damping for settle stability; raise/lower
ζ in JSON only — do not retune cable `damping_ratio` for weld ringing). Distal
roles then scale `kd ∝ √(E/E_ref)` so joint ζ holds approximately while Young's
modulus candidates vary (CMA/grid replay amendment; see
`docs/youngs-modulus-cmaes-implementation.md`).

Given the inertia spread above, a **single global `rigid_joint_angular_kd` will
generally not settle the whole chain without either under-damping primary or
over-damping stem/apple.** `SolverVBD`'s constructor has no per-joint `kd` parameter,
but the underlying state is a plain per-constraint-scalar array,
`solver.joint_penalty_kd`, indexed via `solver.joint_constraint_start[joint_index]`
(`[linear, angular]` offsets for `FIXED` joints). Patch it after solver construction
via `apple_pick_sim.fruiting_system.set_fruiting_joint_angular_kd`:

```python
from apple_pick_sim.fruiting_system import (
    make_fruiting_solver_vbd,
    set_fruiting_joint_angular_kd,
)

solver = make_fruiting_solver_vbd(model)

# Substring keys match fruiting_fixed_joints labels (e.g. "joint_stem_apple").
matched = set_fruiting_joint_angular_kd(
    solver,
    scene.fruiting_fixed_joints,
    {
        "support": 5.0,       # T-junction world anchors (both ends)
        "primary_spur": 3.0,  # T-junction branch (primary -> spur base)
        "stem_apple": 0.3,    # light apple hang
    },
)
# matched == {"support": [j_left, j_right], "primary_spur": [...], ...}
```

Role-name substrings are **topology-dependent** (they come straight from the
`f"joint_{prev_name}_{name}"` labels `build.py` assigns while walking the chain):
T-junction (`DEFAULT_TOPOLOGY`) produces `support` (×2), `primary_spur`, `spur_stem`,
`stem_apple`; a linear chain without a spur would instead produce e.g.
`primary_secondary`, `secondary_stem`. Check `scene.fruiting_fixed_joints` for the
actual labels of the topology you built before choosing keys.

Behavior:

- Each `label_kd` key is a **substring** of the joint label (same convention as
  `fruiting_fixed_joints` entries from `build.py`).
- One key may match multiple joints (e.g. `"support"` → both T-junction world supports).
- Joints not matched by any key keep the global `FRUITING_VBD_RIGID_JOINT_ANGULAR_KD`
  from `make_fruiting_solver_vbd`.
- Raises `ValueError` on negative `kd`, ambiguous multi-key match on one joint, or a
  key that matches no joint.
- **Both slots.** `set_fruiting_joint_angular_kd` / `set_fruiting_joint_linear_kd`
  (and batched variants) patch the angular and linear slots of `joint_penalty_kd`
  respectively. Linear tiering matters for translational ringing at supports and the
  apple hang (shear/bounce), not only rotational nod.

Suggested tiering (starting point, to validate against settle-time and bounce
observations, not a final answer):

| Joint | Suggested `kd_ang` range | Suggested `kd_lin` range |
| ----- | ------------------------ | ------------------------ |
| World anchor / T-junction | ~2–10 N·m·s/rad | ~2–10 N·s/m |
| Primary segment welds | ~1–6 | ~1–6 |
| Stem→apple | ~0.1–1 | ~0.1–1 |

**Status: implemented** (angular and linear). `make_fruiting_solver_vbd` still seeds one
global default per slot; call the per-role helpers after construction to tier by role.

### Linear slot (`set_fruiting_joint_linear_kd`)

Same substring matching and validation as angular. Units are N·s/m (linear slot).

```python
from apple_pick_sim.fruiting_system import (
    set_fruiting_joint_linear_kd,
    set_fruiting_joint_linear_kd_batched,
)

set_fruiting_joint_linear_kd(
    solver,
    scene.fruiting_fixed_joints,
    {
        "support": FRUITING_VBD_RIGID_JOINT_LINEAR_KD,
        "primary_spur": FRUITING_VBD_RIGID_JOINT_LINEAR_KD,
        "stem_apple": FRUITING_VBD_RIGID_JOINT_LINEAR_KD,
    },
)

set_fruiting_joint_linear_kd_batched(
    scene.cable.solver,
    scene.cable.fruiting_fixed_joints,
    {"stem_apple": FRUITING_VBD_RIGID_JOINT_LINEAR_KD},
    num_envs=scene.layout.num_envs,
    joints_per_world=scene.layout.joints_per_world,
)
```

Config defaults mirror angular numerically in `_DEFAULT_JOINT_LINEAR_KD_OVERRIDES`
(`batched_heterogeneous_config.py`).

### Batched envs (`set_fruiting_joint_angular_kd_batched`)

For heterogeneous/replicated batched scenes (one shared `SolverVBD`, `model.world_count > 1`),
use `set_fruiting_joint_angular_kd_batched` instead of looping the single-world helper.
It applies the same substring label matching on **world-0 template** `fruiting_fixed_joints`
and broadcasts each matched role to every env with one `wp.launch` — no full
`joint_penalty_kd` host round trip, no per-env Python loop.

Precondition: **uniform topology** across envs (same joint count/layout per world; already
asserted by `build_heterogeneous_coupled_cable_scene` / replicate builds).

```python
from apple_pick_sim.fruiting_system import set_fruiting_joint_angular_kd_batched

set_fruiting_joint_angular_kd_batched(
    scene.cable.solver,
    scene.cable.fruiting_fixed_joints,  # template (world-0) labels
    {
        "support": 5.0,
        "primary_spur": 3.0,
        "stem_apple": 0.3,
    },
    num_envs=scene.layout.num_envs,
    joints_per_world=scene.layout.joints_per_world,
)
```

`batched_heterogeneous_build.py` calls `_apply_joint_penalty_overrides` (angular + linear
`kd` and optional `kp`) **before** `_run_vbd_settle` and again on the final post-weld scene.
Kd defaults come from `_DEFAULT_JOINT_*_KD_OVERRIDES` in `batched_heterogeneous_config.py`
(roles: `support`, `primary_spur`, `spur_stem`, `stem_apple`). Kp overrides default to
empty dicts — only roles listed in `FruitingSystemConfig.joint_*_kp_overrides` are patched.
Returns global joint indices (`world * joints_per_world + template_index`) per matched key.

Manifest keys: `joint_angular_kd_overrides`, `joint_angular_kd_applied`,
`joint_linear_kd_overrides`, `joint_linear_kd_applied`, `joint_angular_kp_overrides`,
`joint_angular_kp_applied`, `joint_linear_kp_overrides`, `joint_linear_kp_applied`
(see `manifest_sim_config.py`).

### Damping-ratio check against the current script values

`_DEFAULT_JOINT_ANGULAR_KD_OVERRIDES` in `batched_heterogeneous_config.py` (`FruitingSystemConfig`)
sets uniform `FRUITING_VBD_RIGID_JOINT_ANGULAR_KD` for
`support`, `primary_spur`, `spur_stem`, and `stem_apple`, with **no**
`kp` override applied (see §4) — so every matched joint's `k` is Newton's default
`rigid_joint_angular_ke = 1e5`, matching the §2 inertia table exactly (no rescaling
needed). Plugging into `ζ = kd / kd_crit = kd / (2√(k·I_child))`:

| Joint | `I_child` (§2 estimate) | `kd_crit` at `k=1e5` | current `kd` | ζ |
| ----- | ------------------------ | --------------------- | ------------- | --- |
| support | ~1.1×10⁻⁴ | ~6.6 | 1.0 | **~0.15** (underdamped) |
| primary_spur | ~1.8×10⁻⁶ | ~0.85 | 1.0 | **~1.18** (slightly overdamped) |
| spur_stem | ~1.3×10⁻⁷ | ~0.23 | 0.0 (global default) | **~0.0** (underdamped) |
| stem_apple | ~1.0×10⁻⁶ | ~0.63 | 0.05 | **~0.08** (underdamped) |

Read with the same caveat as §2: `I_child` is from a representative build and may not
exactly match `fruiting_system_ranges_real_world_proxy_variance.json`'s geometry, so
treat ζ as order-of-magnitude, not exact. Numerical over-stiffening (`kd/Δt` vs. `k`) is
a non-issue for all three (`≤1.8%` of `k=1e5` at this script's `sim_dt ≈ 5.56e-4 s`), so
there's headroom to raise `support` and `stem_apple` well before hitting that failure
mode — `support` toward `~4–7` and `stem_apple` toward `~0.3–0.6` would bring both closer
to `primary_spur`'s current (healthy) ζ. Validate any change with the KE-decay diagnostic
in Verification below rather than trusting ζ alone.

## 4. Per-joint `kp` (`set_fruiting_joint_angular_kp`)

Angular weld **stiffness** uses the same per-constraint-slot array layout as `kd`, but
patches `solver.joint_penalty_k` (and widens `joint_penalty_k_min` / `joint_penalty_k_max`
so AVBD per-step decay does not clamp the new value back to the constructor ceiling).

Same substring label matching as §3. Global default ceiling is Newton's
`rigid_joint_angular_ke` (`1e5` N·m/rad) unless overridden in `make_fruiting_solver_vbd`.

```python
from apple_pick_sim.fruiting_system import (
    set_fruiting_joint_angular_kp,
    set_fruiting_joint_angular_kp_batched,
)

set_fruiting_joint_angular_kp(
    solver,
    scene.fruiting_fixed_joints,
    {
        "support": 2.0e5,
        "primary_spur": 1.0e5,
        "stem_apple": 5.0e4,
    },
)

# Batched (same preconditions as §3 batched kd):
set_fruiting_joint_angular_kp_batched(
    scene.cable.solver,
    scene.cable.fruiting_fixed_joints,
    {"stem_apple": 5.0e4},
    num_envs=scene.layout.num_envs,
    joints_per_world=scene.layout.joints_per_world,
)
```

**Note:** `joint_penalty_k` decays each step by `rigid_avbd_gamma` (default `0.999`).
After patching, expect `k ≈ kp × gamma^N` after `N` substeps — unlike `kd`, which is
constant. Raising `kp` above the default `1e5` requires the helper to bump
`joint_penalty_k_max`; lowering below the initial `k_min` widens `k_min` accordingly.

**Applied at build** via `build_batched_heterogeneous_scene` → `_apply_joint_penalty_overrides`,
using optional `FruitingSystemConfig.joint_angular_kp_overrides` /
`joint_linear_kp_overrides` (empty by default). Batched examples read shared
`EXAMPLE_JOINT_*` fallbacks from `batched_heterogeneous_config.py`, but the
canonical copy for the default variance fixture lives under optional top-level
`sim_build` in `fruiting_system_ranges_real_world_proxy_variance.json`
(`"support": 2000.0` angular + linear kp). Other roles keep Newton's default
`1e5` unless listed. See `parse_sim_build` in `fruiting_system/params.py`.

Linear kp uses `set_fruiting_joint_linear_kp_batched` with the same label matching.

## Diagnosing which knob to change

| Symptom | Likely cause | Knob |
| ------- | ------------ | ---- |
| Branch sways/bends and slowly loses energy | Cable bend under-damped | ↑ `damping_ratio` on that segment (primary first) |
| Jitter/ringing at a support, junction, or the apple hang, not decaying | FIXED-joint weld under-damped | ↑ joint `kd` (ideally per-joint, see §3) |
| `|v|_max` / KE oscillating with roughly constant or growing amplitude across checkpoints (limit cycle) | Joint `kd` too large relative to `Δt` (`kd/Δt` over-stiffening) | ↓ joint `kd`, or move to per-joint tiering instead of raising further |
| Spur/stem never settles even at ζ near the wood-plausible ceiling | ζ has hit the `c_bend ∝ √k_bend` wall — physically expected | Joint `kd` on the affected welds, not more ζ |
| `branch_path>nominal` (settled path length exceeds nominal by more than tolerance) | **Static geometric/stiffness issue, not damping** | Stretch/bend stiffness or `settle_quasi_static.py` path tolerance — neither ζ nor joint `kd` will fix this |

The last row matters: `settle_stability_reports_from_cable` in
`apple_pick_sim/coupled_fruiting/settle_quasi_static.py` reports `branch_path>nominal`
and `residual_motion` as **separate** issues. Only `residual_motion` (and KE) are
damping-responsive; don't spend a damping sweep chasing `branch_path>nominal`.

## Current configuration snapshot

| Item | Value | Location |
| ---- | ----- | -------- |
| `damping_ratio` (primary/spur/stem) | fixed `0.3` (JSON band) | `fruiting_system_ranges_real_world_proxy_variance.json` |
| `vbd_stretch_fixed` (primary) | `stretch_stiffness≈1e7`, `stretch_damping≈1492` | same fixture |
| `FRUITING_VBD_RIGID_JOINT_LINEAR_KD` | `0.0` (Newton ``SolverVBD`` default) | `apple_pick_sim/fruiting_system/build.py` |
| `FRUITING_VBD_RIGID_JOINT_ANGULAR_KD` | `0.0` (Newton ``SolverVBD`` default) | `apple_pick_sim/fruiting_system/build.py` |
| `rigid_joint_linear_ke` / `rigid_joint_angular_ke` | `1e5` (Newton default, not overridden) | `newton/newton/_src/solvers/vbd/solver_vbd.py` |
| `rigid_joint_*_k_start` | `1e8` / `1e6` passed but **inert** (ramping disabled) | `make_fruiting_solver_vbd` |
| VBD `iterations` | 25 | `make_fruiting_solver_vbd` |
| `_DEFAULT_JOINT_ANGULAR_KD_OVERRIDES` (batched config) | uniform `0.0` per role (`support`, `primary_spur`, `spur_stem`, `stem_apple`; Newton default via `FRUITING_VBD_RIGID_JOINT_ANGULAR_KD`) | `batched_heterogeneous_config.py` |
| `_DEFAULT_JOINT_LINEAR_KD_OVERRIDES` (batched config) | uniform `0.0` per role (`support`, `primary_spur`, `spur_stem`, `stem_apple`; Newton default via `FRUITING_VBD_RIGID_JOINT_LINEAR_KD`) | `batched_heterogeneous_config.py` |
| `EXAMPLE_JOINT_*_KD_OVERRIDES` (Python fallback) | uniform `0.3` per role | `batched_heterogeneous_config.py` (used when ranges omit `sim_build`) |
| `EXAMPLE_JOINT_*_KP_OVERRIDES` (Python fallback) | `"support": 2000.0` angular + linear | `batched_heterogeneous_config.py` |
| `sim_build` VIC + joint (canonical variance) | VIC `100/20/10/3`; `joint_damping_ratio: 1.0`; kp `"support": 10000` | `fruiting_system_ranges_real_world_proxy_variance.json` via `parse_sim_build` |
| `joint_*_kp_overrides` (batched config default) | empty dict | `batched_heterogeneous_config.py` (`FruitingSystemConfig`) |

Update this table when any of these values change so it stays a reliable snapshot.

## Code map

| Module | Role |
| ------ | ---- |
| `apple_pick_sim/fruiting_system/params.py` | `RodParams`, `rod_params_from_material` — derives `bend_damping`/`stretch_damping` from `(E, ζ, geometry)`; `parse_sim_build` / `joint_damping_ratio` |
| `apple_pick_sim/fruiting_system/joint_kd_scaling.py` | `joint_kd_from_damping_ratio`, `scale_joint_kd_overrides` (√(E/E_ref)) |
| `apple_pick_sim/fruiting_system/build.py` | `FRUITING_VBD_RIGID_JOINT_{LINEAR,ANGULAR}_KD`, `make_fruiting_solver_vbd`, `set_fruiting_joint_angular_kd`, `set_fruiting_joint_angular_kd_batched`, `set_fruiting_joint_angular_kp`, `set_fruiting_joint_angular_kp_batched`, `set_fruiting_joint_linear_kp`, `set_fruiting_joint_linear_kp_batched`, `fruiting_fixed_joints` |
| `newton/newton/_src/solvers/vbd/solver_vbd.py` | `SolverVBD._init_joint_penalty_k` (global kd/k → per-constraint-slot arrays), `joint_penalty_kd`, `joint_penalty_k`, `joint_constraint_start` |
| `newton/newton/_src/solvers/vbd/rigid_vbd_kernels.py` | `evaluate_linear_constraint_force_hessian` / `evaluate_angular_constraint_force_hessian` — where `kd` enters the AVBD force/Hessian |
| `apple_pick_sim/coupled_fruiting/settle_quasi_static.py` | `settle_stability_reports_from_cable` — `residual_motion` / `branch_path>nominal` classification |
| `apple_pick_sim/coupled_fruiting/settle_ke_decay.py` | Branch KE peak-envelope decay diagnostics (better than eyeballing `|v|_max` snapshots) |

## Tests

| Test | Intent |
| ---- | ------ |
| `apple_pick_sim/tests/test_wrench_equilibrium.py` (see "NOTE ON JOINT DAMPING") | Documents why joint `kd` is kept small relative to `rigid_joint_*_k_start ~ 1e8`; equilibrium wrench checks are sensitive to joint damping magnitude |
| `apple_pick_sim/tests/test_fruiting_system.py` | `test_set_fruiting_joint_angular_kd_*`, `test_set_fruiting_joint_angular_kp_*` — single-world per-joint angular kd/kp patching |
| `apple_pick_sim/tests/test_heterogeneous_coupled_fruiting.py` | `test_set_fruiting_joint_angular_kd_batched_*`, `test_set_fruiting_joint_angular_kp_batched_*` — batched all-env kernel patch |
| `apple_pick_sim/tests/test_settle_ke_decay.py` | KE envelope decay analysis correctness |
| `apple_pick_sim/tests/test_sweep_settle_weld_stability.py` | Settle-duration sweep → stability rate after settle/weld/hold |
| `apple_pick_sim/tests/test_real_world_proxy_fixture.py` | Fixture `vbd_stretch_fixed` / `damping_ratio` bands validate as expected |

## Verification

```bash
# Per-joint angular kd patching (single-world)
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_fruiting_system.py -q -k joint_angular_kd

# Batched all-env angular kd patching
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_heterogeneous_coupled_fruiting.py -q -k joint_angular_kd

# Per-joint angular kp patching (single-world + batched)
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_fruiting_system.py apple_pick_sim/tests/test_heterogeneous_coupled_fruiting.py -q -k joint_angular_kp

# Fast gate for material damping derivation
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_fruiting_system.py -q -k "damping"

# Wrench equilibrium (sensitive to joint kd magnitude)
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_wrench_equilibrium.py -q

# KE envelope decay diagnostic (objective settle-time measurement, not eyeballing)
uv run python apple_pick_sim/diagnostics/log_settle_ke_decay.py \
  --num-envs 4 --seed 42 --settle-substeps 15000 --settle-gravity-ramp \
  --output-dir /tmp/settle_ke_seed42

# Settle → weld → hold sweep across settle durations
uv run python apple_pick_sim/diagnostics/sweep_settle_weld_stability.py \
  --num-envs 4 --seed 42 --settle-substeps 1000,5000,10000,30000 --robot fr3
```

Always sweep with `--settle-gravity-ramp` when comparing soft fixtures unless
specifically testing instant-gravity shock response — the ramp is **opt-in**
(`settle_gravity_ramp=False` by default in `BatchedHeterogeneousCoupledSimConfig`).
Instant full gravity injects much more initial energy and makes early checkpoints
look worse than a ramped settle would.

## Related docs

- `docs/material-parameter-sampling.md` — `(E, ζ)` → `bend_stiffness`/`bend_damping`/`stretch_stiffness` derivation, `vbd_stretch_fixed` override
- `docs/WRENCH_READOUT.md` — joint damping's effect on fixed-joint wrench readout tolerances
- `docs/real-world-proxy.md` — fixture stiffness tiers and placement this damping config applies to
