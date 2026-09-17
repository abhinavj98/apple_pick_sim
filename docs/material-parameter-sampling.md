# Material-parameter sampling (E, ζ)

## Document status

| Field | Value |
| ----- | ----- |
| **Last updated** | 2026-08-30 |
| **Roadmap slice** | [V].2.1.3 |
| **Owner** | Abhinav |

## Summary

Rod parameters split into three roles:

| Role | Fixture keys | Tuned for |
| ---- | ------------ | --------- |
| **Bend** | `flexural_modulus_pa`, `damping_ratio` | Real wood / peduncle bending (sys-ID / literature) |
| **Stretch (axial)** | `youngs_modulus_pa`, shared `damping_ratio` | Beam \(EA/L_{\mathrm{seg}}\) axial stiffness from sampled axial \(E\) |
| **Joints** | `sim_build.joint_damping_ratio` (etc.) | Numerical weld settle — see `docs/damping-tuning.md` |

Sample **flexural** \(E_{\mathrm{flex}}\) and **axial** \(E_{\mathrm{ax}}\) (`youngs_modulus_pa`) independently per segment, plus one **`damping_ratio`** \(\zeta\) that drives both bend and stretch damping at `sample_params` time. Derive VBD knobs from geometry; do not sample raw `bend_stiffness` / `stretch_stiffness` bands in range JSON.

Legacy **`vbd_stretch_force`** max-load budgets are **rejected** by `load_ranges` (removed from shipped fixtures). The helper `stretch_knobs_from_max_force` remains for unit tests only.

Geometry (`length`, `radius`, `density`, directions, `num_segments`) continues to be sampled (or fixed) as today. The VBD build path in `apple_pick_sim/fruiting_system/build.py` consumes derived `RodParams` stiffness/damping fields — only the **sampler** and **range JSON contract** change.

## Motivation

| Today | Problem |
| ----- | ------- |
| Independent min/max bands on `bend_stiffness`, `stretch_stiffness`, `bend_damping` | Stiffness and damping can be physically inconsistent with sampled `radius` / `length` / `density` |
| Sys-ID / CEM over raw VBD knobs | Hard to interpret identified values; weak transfer across geometry changes |
| Fixture tiers in N/m (`docs/real-world-proxy.md`) | Useful as **targets**, but should map through \(E\) and geometry rather than direct `bend_stiffness` bands |

Sampling \(E\) and \(\zeta\) keeps domain randomization and calibration tied to interpretable material parameters while preserving the existing Newton `add_rod` API.

## Sampling contract

### Range JSON (per rod segment)

**New primary keys** (replace `bend_stiffness`, `stretch_stiffness`, `bend_damping` ranges):

| Key | Unit | Meaning |
| --- | ---- | ------- |
| `flexural_modulus_pa` | Pa | Flexural / bend Young's modulus \(E_{\mathrm{flex}}\) |
| `youngs_modulus_pa` | Pa | Axial Young's modulus \(E_{\mathrm{ax}}\) (stretch \(EA/L_{\mathrm{seg}}\)) |
| `damping_ratio` | – | Modal damping ratio \(\zeta\) (fraction of critical; shared bend + stretch) |

**Unchanged** (still sampled or fixed per segment): `num_segments`, `length`, `radius`, `density`, angle keys (`azimuth_deg`, `elevation_deg`, deltas).

**Apple** segment unchanged (`radius`, `density` only).

### Derived VBD knobs (at `sample_params` time)

For a circular rod cross-section with sampled `radius` \(r\), `density` \(\rho\), total `length` \(L\), and `num_segments` \(N\):

\[
A = \pi r^2,\quad I = \frac{\pi r^4}{4},\quad L_{\mathrm{seg}} = \frac{L}{N},\quad m_{\mathrm{seg}} = \rho A L_{\mathrm{seg}},\quad J_{\mathrm{seg}} = \frac{m_{\mathrm{seg}}(3r^2 + L_{\mathrm{seg}}^2)}{12}
\]

Default constitutive mapping (segment-local):

| VBD field | Formula | Units |
| --------- | ------- | ----- |
| `stretch_stiffness` | \(E_{\mathrm{ax}} A / L_{\mathrm{seg}}\) | N/m |
| `stretch_damping` | \(2 \zeta \sqrt{k_{\mathrm{stretch}}\, m_{\mathrm{seg}}}\) | N·s/m |
| `bend_stiffness` | \(E_{\mathrm{flex}} I / L_{\mathrm{seg}}\) | N·m/rad |
| `bend_damping` | \(2 \zeta \sqrt{k_{\mathrm{bend}}\, J_{\mathrm{seg}}}\) | N·m·s/rad |

\(J_{\mathrm{seg}}\) is the solid-cylinder segment moment of inertia about a transverse axis through the midpoint. One sampled \(\zeta\) governs both axial and bending damping.

`rod_params_from_material(flexural_modulus_pa, youngs_modulus_pa, damping_ratio, …)` is the canonical builder. **`set_rod_flexural_modulus`** retargets bend only; **`set_rod_youngs_modulus`** retargets axial stretch only.

### Rejected legacy keys

`load_ranges` rejects `vbd_stretch_force`, legacy `vbd_stretch_fixed`, and direct VBD stiffness bands. Shipped fixtures duplicate pre-split \(E\) bands onto both modulus keys (axial may be retuned independently later).

### Optional top-level `sim_build` (VIC + joint overrides)

Ranges JSON may include an optional **file-level** (not per-segment) `sim_build` block so batched examples share one source of truth for TCP VIC gains and FIXED-joint damping / kp:

```json
"sim_build": {
  "vic_gains": {
    "linear_k": 100.0,
    "linear_d": 20.0,
    "angular_k": 10.0,
    "angular_d": 3.0
  },
  "joint_damping_ratio": 1.0,
  "joint_angular_kp_overrides": { "support": 1000.0 },
  "joint_linear_kp_overrides": { "support": 1000.0 }
}
```

- **Optional:** omit the key entirely; `load_ranges` still succeeds. `parse_sim_build(ranges)` returns `None`.
- **When present:** `vic_gains` (all four keys) is required. Prefer **`joint_damping_ratio`** (ζ ≥ 0; critical at 1, >1 overdamped) to derive absolute kd at build (`kd = ζ·2·√(k·I/m)`); absolute `joint_*_kd_overrides` remain supported but are **mutually exclusive** with `joint_damping_ratio`. Joint roles are `support`, `primary_spur`, `spur_stem`, `stem_apple`. Support **angular** \(k_p\) is overwritten from linear as \(\tfrac{3}{4} L_{\mathrm{dowel}}^{2} k_{\mathrm{lin}}\) (`docs/handbook-youngs-cma.md`, `docs/damping-tuning.md`).
- **Canonical ship:** `fruiting_system_ranges_real_world_proxy_variance.json`. Consumed by `example_batched_heterogeneous_coupled_sim.py`, `example_batched_collect_sysid_data.py`, and `example_batched_sysid_mmd_grid.py` (Python `EXAMPLE_JOINT_*` / VIC constants remain fallbacks when `sim_build` is absent).
- **API:** `parse_sim_build` in `apple_pick_sim/fruiting_system/params.py`. Tuning notes: `docs/damping-tuning.md`. Design: `docs/specs/2026-07-10-fixture-sim-build-knobs-design.md`.

### Tier constraints

When both **primary** and **secondary** are enabled, enforce **`primary.flexural_modulus_pa >= secondary.flexural_modulus_pa`** (replaces the current `primary.bend_stiffness >= secondary.bend_stiffness` check on derived values). No ordering constraint on axial `youngs_modulus_pa`.

### Serialization

- Current schema: `fruiting_system_params_v3` with both `flexural_modulus_pa` and `youngs_modulus_pa` per rod row.
- v2 read: legacy `youngs_modulus_pa` → flexural; axial back-computed from stored `stretch_stiffness`.
- v1 read-only: derived stiffness only; moduli back-computed where possible.
- `params_fingerprint` adds `*_flexural_modulus_pa` and `*_youngs_modulus_pa` (axial meaning).

## Code map

| Module | Change |
| ------ | ------ |
| `apple_pick_sim/fruiting_system/params.py` | `RodParams.flexural_modulus_pa` + axial `youngs_modulus_pa`; `sample_params` / `_validate_ranges`; `rod_params_from_material(E_flex, E_ax, ζ, geometry)`; split setters |
| `apple_pick_sim/fixtures/*.json` | Both modulus keys + `damping_ratio` bands (shipped fixtures duplicate bands at migration) |
| `apple_pick_sim/fruiting_system/build.py` | **No change** — still passes derived `RodParams` into `add_rod` |
| `apple_pick_gym/examples/run_system_identification.py` | Follow-up: grid over \(E\) or derived stiffness (legacy CLI flags remain until [S]) |

## Tests (TDD)

Add / extend in `apple_pick_sim/tests/test_fruiting_system.py`:

| Test | Intent |
| ---- | ------ |
| `test_sample_params_material_mode_derives_stiffness_from_E_zeta` | Known \(E\), \(\zeta\), \(r\), \(L\), \(\rho\) → expected `stretch_stiffness`, `bend_stiffness`, `bend_damping` |
| `test_sample_params_material_mode_primary_E_ge_secondary` | Tier ordering on \(E\) when both segments enabled |
| `test_load_ranges_material_keys_validate` | Fixture JSON with `youngs_modulus_pa` / `damping_ratio` passes validation |
| *(behavior)* legacy stiffness keys | `load_ranges` **rejects** deprecated `bend_stiffness` / similar keys — use material keys only |
| `test_fruiting_params_v2_roundtrip` | Serialize / deserialize includes \(E\), \(\zeta\) |
| `test_parse_sim_build_*` / `test_load_ranges_*sim_build*` | Optional top-level `sim_build` validate / parse |
| `test_variance_sim_build_knobs` (`test_real_world_proxy_fixture.py`) | Variance fixture ships expected VIC + joint overrides |

Regression: existing deterministic-seed tests updated to material-mode fixtures or legacy path smoke.

## Verification

```bash
# Fast gate after implementation
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_fruiting_system.py -q -k "material or youngs or damping_ratio or sample_params"

# Fixture + proxy policy
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_real_world_proxy_fixture.py -q

# Heterogeneous batched DR still samples per env
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_heterogeneous_coupled_fruiting.py -q
```

## Related docs

- `docs/real-world-proxy.md` — map proxy N/m tier **targets** to \(E\) bands via nominal geometry
- `docs/damping-tuning.md` — joint kd/kp + variance fixture `sim_build` snapshot
- `docs/handbook-coupled-simulation.md` — batched DR still varies per-env θ; material keys replace raw stiffness bands
- `docs/system_identification.md` — CEM search space moves to \(E\), \(\zeta\) (plus geometry when fixed)
- `docs/handbook-sysid-scoring.md` — episode `fruiting_system_params` schema v2
- `docs/specs/2026-07-10-fixture-sim-build-knobs-design.md` — optional `sim_build` design (Implemented)

## Derivation: why sample E and ζ instead of raw stiffness/damping

This section records the reasoning behind the contract above — why independent sampling of `bend_stiffness` / `bend_damping` / `radius` / `length` / `density` is unstable, and why deriving those VBD knobs from material properties (`E`, `ζ`) plus geometry fixes it. The scheme below is **implemented** (`sample_params` in `apple_pick_sim/fruiting_system/params.py`); this section is the "why," not a proposal.

### Problem statement

`apple_pick_sim/fixtures/*.json` specifies independent `[min, max]` ranges for each rod segment's stiffness, damping, radius, length, density, and `num_segments`. Sampling each of these independently via `rng.uniform`/`rng.integers` is what produces unstable draws (apple drift/sag, non-convergent settle, blown-up velocities — see `apple_pick_sim/diagnostics/sweep_zero_vic_stability.py` and `apple_pick_sim/coupled_fruiting/settle_quasi_static.py`).

The root cause: these quantities are **not physically independent**. Whether a given damping value is "enough" depends on stiffness and on the segment's mass/inertia (set by radius, length, density, `num_segments`). Sampling them from independent boxes covers a hyperrectangle in parameter space, but the physically stable region is a curved manifold that cuts diagonally through that box — many corners of the box are guaranteed-unstable regardless of how "reasonable" each individual range looks in isolation.

### Rod-level params vs. joint-level params

`RodParams` is **one scalar tuple per rod segment** (`primary`/`secondary`/`spur`/`stem`). `build.py` passes that tuple straight into `newton.ModelBuilder.add_rod`, which creates `num_segments` capsule bodies and `num_segments - 1` internal cable joints, and broadcasts the *same* stiffness/damping/radius/density to **every** joint in that segment. There is no per-joint variation within one sampled segment — homogeneity within a segment is structural, not sampled.

This means `num_segments` is itself a hidden multiplier on stability, not just a topology knob: more segments in series over the same physical `length` requires each joint to be *stiffer* to represent the same overall compliance (springs in series), and it also changes each joint's *effective inertia* far more steeply (see below).

### Deriving damping from a target damping ratio

Model each joint as a torsional spring-damper acting on its segment's rotational inertia. For one rod segment (radius \(r\), density \(\rho\), `length`, split into `num_segments` \(N\)):

- Segment length: \(L_{seg} = \text{length}/N\)
- Segment mass: \(m_{seg} = \rho \cdot \pi r^2 \cdot L_{seg}\)
- Effective bending inertia (solid circular cylinder about a diameter through the CoM — shipped in `_segment_material_geometry`): \(J_{\mathrm{seg}} = m_{seg}(3r^2 + L_{seg}^2)/12\)
- Natural frequency: \(\omega_n = \sqrt{k_{bend}/J_{\mathrm{seg}}}\)
- Damping ratio: \(\zeta = \dfrac{c_{bend}}{2\sqrt{k_{bend}\cdot J_{\mathrm{seg}}}}\)

Older notes sometimes used a cantilever slender-segment approximation \(I_{\mathrm{eff}} \approx \tfrac13 m L^2\); that is **not** what the code uses — prefer \(J_{\mathrm{seg}}\) above.

Sampling a dimensionless damping ratio \(\zeta\) and deriving \(c_{bend} = \zeta \cdot 2\sqrt{k_{bend}\cdot J_{\mathrm{seg}}}\) from that env's already-sampled stiffness/radius/density/length/`num_segments` makes every draw land at a controlled, geometry-consistent damping ratio instead of an absolute number that means something different for every geometry combination.

### Deriving stiffness from Young's modulus

Bending stiffness is not a free material parameter — it decomposes as \(k_{bend} \propto E \cdot I_{area}\), where \(E\) is an intrinsic material property and \(I_{area} = \pi r^4/4\) is the cross-section's second moment of area. The per-joint discretized bending stiffness is:

$$k_{bend,joint} = \frac{E\, I_{area}}{L_{seg}} = \frac{E\cdot \pi r^4/4 \cdot N}{\text{length}}$$

Combining with the damping-ratio derivation above:

$$\omega_n = \sqrt{\frac{3E}{4\rho}}\cdot \frac{r\cdot N^2}{\text{length}^2}$$

Practical implication: sample `E` (and derive per-env `bend_stiffness` from that env's own geometry), not `bend_stiffness` directly — the same physical material needs different `bend_stiffness` at different radii (by \(r^4\)!). Sampling `E` keeps material and geometry properly decoupled and matches the CEM sys-ID target directly (θ becomes lower-dimensional and geometry-invariant).

**Units caveat:** the proxy stiffness table in `docs/real-world-proxy.md` (210–736 "N/m") is a **cantilever tip force/deflection** measurement (\(k_{cantilever} = 3EI_{area}/L^3\)), not the solver's per-joint torque/radian quantity. Converting requires the bench geometry used for that measurement: \(E = \frac{3\,k_{cantilever}\cdot L^3}{\pi r^4}\).

### The ω_n · dt numerical guard

Even with damping correctly derived from a target \(\zeta\), a draw can still be **numerically** unstable: if the fixed simulation substep `dt` isn't small relative to the joint's oscillation period \(T = 2\pi/\omega_n\), the discrete-time integrator can't represent the continuous-time system the \(\zeta\) formula assumed.

- **Aliasing:** rule of thumb, want \(T/dt \gtrsim 10\text{–}20\) samples per period, i.e. \(\omega_n \cdot dt \lesssim 0.3\text{–}0.6\).
- **Iterative solver non-convergence:** `make_fruiting_solver_vbd` runs a fixed iteration budget per substep; a joint whose local stiffness is very high relative to `dt` can fail to converge regardless of its analytical damping.
- **Why `num_segments` matters more than expected:** \(\omega_n \propto N^2\) — natural frequency scales with the *square* of segment count, independent of whether `E`/`ρ` are "correct." A max-segment draw can be far faster in \(\omega_n\) than a min-segment draw from the same fixture, purely from discretization.

Fixture authors should validate new ranges against their worst corner (max `num_segments`, max `E`, min `ρ`, min `radius`) using the settle-stability sweep tooling (`apple_pick_sim/diagnostics/sweep_zero_vic_stability.py`, `apple_pick_sim/coupled_fruiting/settle_quasi_static.py`) before locking them in — the closed-form formulas above are single-joint/SDOF approximations of a coupled multi-body VBD system.

### If density can't be measured directly

- Don't fix `ρ` to a single point — real branches vary in density (species, moisture, age); collapsing to one value understates real branch-to-branch variance.
- Source a range from literature matching the correct regime: real wood-species density (green/living-wood, not seasoned-lumber) if the physical rig is literal wood; the material spec sheet if the bench proxy is a mechanical rig with rigid links (confirm which applies via `docs/real-world-proxy.md`).
- `E` and `ρ` are physically linked through moisture content in real wood — an independently chosen `ρ` range could combine with a sys-ID'd `E` to produce an unphysical combination. Measure `ρ` on the same specimens used for the `E` sys-ID when possible.
- Document assumed/placeholder density in the fixture `_comment` field so it can be revisited if CEM/MMD replay validation shows a systematic mismatch consistent with a wrong density assumption.
