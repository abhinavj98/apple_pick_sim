# Explicit apple inertia on the coupling channel

| Field | Value |
| ----- | ----- |
| **Status** | Implemented (2026-08-29) |
| **Canonical living docs:** | `docs/handbook-coupled-simulation.md` §6, `docs/explicit-apple-load-tcp-harvest.md`, `docs/handbook-sysid-scoring.md` |
| **Date** | 2026-08-29 |
| **Roadmap** | M4.0 real replay / CMA — sim `ft_wrist` closer to env-on-robot apple load |
| **Extends** | Explicit apple weight (`explicit_load.py`, `docs/explicit-apple-load-tcp-harvest.md`) |

## Purpose

Unify TCP apple **weight and inertia** on the coupling channel so MuJoCo `body_f` and gym `ft_wrist` see the same external fruit load.

Today welded builds split the apple: MuJoCo `apple_payload` puts mass in `M(q)` (motion / VIC), while harvest writes only stem + explicit `m g` into `proxy_forces` / `ft_wrist`. Real FT is one env-on-robot wrench at TCP. This spec moves inertial apple load onto the same harvest → `body_f` path as `m g`, and leaves the payload body at **mass 0** so `M(q)` does not double-count.

This does **not** merge VBD and MuJoCo models. The VBD apple stays prescribed (`fix_to_apple`). Stem mechanics stay in VBD gather. Only the TCP apple load (weight + rigid-body inertia of the fruit) is unified.

## Locked decisions (do not reopen)

| Topic | Choice |
| ----- | ------ |
| Architecture | One apple-load channel at coupling: stem + `m g` + inertial reaction → `proxy_forces` → lagged `body_f` → `ft_wrist` |
| Welded default | Inertia **on** when `fix_to_apple=True` (same resolver pattern as explicit weight) |
| Payload body | **Keep** the FIXED `apple_payload` child for topology / A/B; production welded build leaves **mass 0** |
| Acceleration | Finite-difference TCP twist, then rigid map through grasp offset. Not MuJoCo `cacc`. |
| Sign | Env-on-robot companion of existing `F = m g`, **not** textbook force-on-apple `F = m a` |
| Apply target | Coupling (`proxy_forces` / `body_f`) every substep, **not** obs-only logging |
| Caps | Unchanged (`DEFAULT_STEM_FORCE_CAP_N = 40`, `DEFAULT_STEM_TORQUE_CAP_NM = 10`). Inertia is added **before** gain/caps. |
| CUDA graphs | Device `qd_prev` snapshot; no `body_qd.numpy()` per substep |

## Non-goals

- Sim F/T low-pass (`ft_wrist_lpf` stays convert-only on real bags)
- Removing the `apple_payload` body from the FR3 builder / replicated-robot cache
- `fix_to_apple=False` (free apple): VBD already integrates fruit mass; explicit inertia would double-count
- Series-elastic joint K/D from `dynamics.yaml`
- Changing harvest caps, stem gather, or `stem_coupling_gain` defaults
- Taring simulated `ft_wrist`
- EE / tool inertia in `ft_wrist` (still excluded; H3 “do not tare sim” remains)
- Wiring `FruitingSystemConfig.stem_harvest_explicit_apple_weight` (still inert). Do **not** add a second inert inertia field on that config.

---

## 1. Frames, packing, signs

### World frame

Same as `docs/WRENCH_READOUT.md` and `apple_explicit_wrench_about_tcp`:

| Axis | Direction |
| --- | --- |
| X | right |
| Y | forward |
| Z | **up** |

`gravity_vec = (0, 0, −9.81)` m/s². All of `g`, `F`, `τ`, `v`, `ω`, `a`, `α`, and `r` are **world**.

### Application point

Wrench is about the **TCP origin**: `body_q` translation of `/ee/tcp` (same as today’s explicit weight). Not apple COM, not flange, not `apple_payload` COM.

Lever arm:

\[
\mathbf{r} = \mathbf{p}_{\mathrm{apple}} - \mathbf{p}_{\mathrm{tcp}}
\]

With `gripper_proxy_offset_in_apple_frame` set (`fix_to_apple`), `p_apple` comes from

\[
X_{\mathrm{apple}} = X_{\mathrm{tcp}} \cdot X_{\mathrm{offset}}^{-1}
\]

(`apple_com_from_tcp_grasp_offset` / harvest kernel). Same `r` as weight.

### Spatial packing (this repo)

Newton/Warp `body_qd` and harvest wrenches in this codebase are **linear then angular**, not Featherstone `[ω, v]`:

| Accessor | Quantity |
| --- | --- |
| `wp.spatial_top(qd)` | linear velocity of TCP origin, world [m/s] |
| `wp.spatial_bottom(qd)` | angular velocity, world [rad/s] |
| numpy `body_qd` row | `[vx, vy, vz, wx, wy, wz]` |
| harvest `spatial_vector` | `[Fx, Fy, Fz, τx, τy, τz]` (N, N·m) |

Confirmed by `vic_wrench.compute_vic_spatial_wrench` (`v_act = spatial_top`, `w_act = spatial_bottom`) and `_limit_and_write_tcp_stem_wrench_kernel` (writes `spatial_vector(f, τ)`).

`I` is the solid-sphere inertia already used by `solid_sphere_inertia_diag`: \(I = \tfrac{2}{5} m R^{2}\) about apple COM. Isotropic ⇒ the rotational reaction is \(−I\boldsymbol{\alpha}\) in world with **no** `R I Rᵀ`.

### Kinematics

From current and previous TCP `body_qd` (post-MuJoCo, same lag as harvest):

\[
\mathbf{v},\boldsymbol{\omega} \leftarrow \mathrm{qd},\qquad
\mathbf{a}_{\mathrm{tcp}},\boldsymbol{\alpha} = (\mathrm{qd} - \mathrm{qd}_{\mathrm{prev}}) / \mathrm{dt}
\]

Use **current-step** `ω` in the centripetal term (not average). Rigid weld:

\[
\mathbf{a}_{\mathrm{com}} = \mathbf{a}_{\mathrm{tcp}} + \boldsymbol{\alpha}\times\mathbf{r} + \boldsymbol{\omega}\times(\boldsymbol{\omega}\times\mathbf{r})
\]

Matches co-teleport: `v_apple = v_tcp + ω × r_world` in `mirror_robot_tcp_to_proxy_and_apple_kernel`.

If `dt ≤ 0` or `mass ≤ 0`, the inertial wrench is **zero** (no divide-by-zero).

### Sign: env-on-robot

Existing weight (must not change):

\[
\mathbf{F}_{\mathrm{weight}} = m\mathbf{g},\qquad
\boldsymbol{\tau}_{\mathrm{weight}} = \mathbf{r}\times\mathbf{F}_{\mathrm{weight}}
\]

With `g_z < 0`, `F_z = −mg` (downward load on the robot).

Inertia is the D’Alembert **reaction on the TCP**, not force-on-apple. Combined explicit apple load:

\[
\mathbf{F} = m\mathbf{g} - m\mathbf{a}_{\mathrm{com}},\qquad
\boldsymbol{\tau} = \mathbf{r}\times\mathbf{F} - I\boldsymbol{\alpha}
\]

Sphere ⇒ `ω × (I ω) = 0`. Equivalently: keep today’s weight term and add

\[
\mathbf{F}_{\mathrm{inertia}} = -m\mathbf{a}_{\mathrm{com}},\qquad
\boldsymbol{\tau}_{\mathrm{inertia}} = \mathbf{r}\times\mathbf{F}_{\mathrm{inertia}} - I\boldsymbol{\alpha}.
\]

**Do not** use velocity-delta harvest `f = m(a − g)` (`compute_proxy_reaction_wrench_kernel`). That is force **on a free proxy**. Explicit load is the opposite sign family (reaction on the robot). Mixing them would flip inertia relative to `m g`.

| Case | Required result |
| --- | --- |
| Hold, `qd = qd_prev` | `a = 0`, `α = 0` ⇒ inertia 0; `F_z = −mg`; `τ = r × mg` |
| Accelerate apple **up** (`a_z > 0`), `r = 0` | `F_z` **more negative** than weight |
| Free-fall `a_com = g` | explicit apple **force** → 0 |
| Constant `ω`, `α = 0` | `−m ω×(ω×r)` is **centrifugal on the robot** (away from TCP) |
| Pure `α`, `r = 0` | `τ = −I α` on the robot |

---

## 2. Data flow

`coupled_substep` order is unchanged:

1. Copy lagged `proxy_forces` → `coupling_forces_cache` → TCP `body_f`
2. MuJoCo step (updates `robot_state_0.body_q` / `body_qd`)
3. Mirror TCP → proxy / prescribed apple
4. VBD step
5. Stem harvest into `proxy_forces` (stem gather + explicit apple load + gain/caps)
6. Snapshot TCP `body_qd` → `robot_tcp_qd_prev` for the **next** harvest

Inertia uses post-MuJoCo `qd` vs the previous snapshot, then feeds the **next** MuJoCo step. Same one-substep lag as stem harvest.

Gym `ft_wrist` remains a copy of `coupling_forces_cache` / `tcp_coupling_force`. No obs-only inertia path.

### Payload mass

| Flag | `apple_payload` mass | Harvest apple load |
| --- | --- | --- |
| Welded default (this spec) | **0** | `m g − m a`, `r×F − Iα` |
| `explicit_apple_inertia=False` (A/B) | AVBD apple mass / sphere `I` via `apply_mujoco_apple_payload_inertias` | weight only (today) |
| Both mass > 0 and inertia on | **Forbidden** | — |

`append_apple_payload_link` still runs for `fix_to_apple` (cache keys / `BatchedEnvLayout.mj_apple_payload_body_indices` unchanged). Build **skips** `apply_mujoco_apple_payload_inertias` when inertia is on. `apply_mujoco_apple_payload_inertias(scene)` **raises** if `scene.stem_harvest_explicit_apple_inertia` is True.

---

## 3. API and flags

### Resolver

Add `_resolve_stem_harvest_explicit_apple_inertia` next to `_resolve_stem_harvest_explicit_apple_weight` in `builders.py`:

| `override` | `fix_to_apple` | Result |
| --- | --- | --- |
| `None` (default) | True | `True` |
| `None` | False | `False` |
| `True` | False | **raise** `ValueError` (VBD already integrates apple mass) |
| `True` | True | `True` |
| `False` | any | `False` (restores payload-mass A/B) |

Builder kwargs: `stem_harvest_explicit_apple_inertia: bool | None = None` on `build_coupled_fruiting_scene` / `build_coupled_fruiting_fr3` / batched equivalents, passed through the same path as `stem_harvest_explicit_apple_weight`.

Do **not** add `FruitingSystemConfig.stem_harvest_explicit_apple_inertia` until that config is actually passed through `_builder_kwargs` (weight is still inert there).

### Scene fields

On `CoupledFruitingScene`:

- `stem_harvest_explicit_apple_inertia: bool = False`
- `apple_inertia_kgm2: float = 0.0` — scalar \(I = \tfrac{2}{5} m R^{2}\) cached at build (CUDA-graph safe, like `apple_mass_kg`)
- `robot_tcp_qd_prev: wp.array | None` — `dtype=wp.spatial_vector`, same length/device as `robot_state_0.body_qd`

Batched: per-env `stem_harvest_apple_inertias_wp` (float) and `stem_harvest_use_explicit_inertia_wp` (int), filled in `prepare_batched_stem_harvest_arrays`. Snapshot remains the full robot `body_qd` buffer (TCP slots via `layout.tcp_body_indices`).

### `explicit_load.py`

Keep `apple_explicit_wrench_about_tcp` as **weight only** (existing tests stay valid). Add:

```python
def apple_com_acceleration_world(
    v_tcp_world: np.ndarray,
    w_tcp_world: np.ndarray,
    a_tcp_world: np.ndarray,
    alpha_world: np.ndarray,
    r_tcp_to_apple_world: np.ndarray,
) -> np.ndarray:
    """a_com = a_tcp + α × r + ω × (ω × r), world frame."""

def apple_inertial_reaction_wrench_about_tcp(
    mass_kg: float,
    inertia_kgm2: float,
    a_com_world: np.ndarray,
    alpha_world: np.ndarray,
    r_tcp_to_apple_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Env-on-robot: F = −m a_com, τ = r × F − I α."""

def tcp_twist_finite_difference(
    qd: Any,
    qd_prev: Any,
    tcp_body_index: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (v, ω, a_tcp, α) from spatial_top/bottom; zeros if dt ≤ 0."""
```

`explicit_apple_wrench_for_stem_harvest` gains optional `robot_body_qd`, `robot_body_qd_prev`, `dt`, `inertia_kgm2`, `explicit_apple_inertia`. When inertia is on it **adds** the inertial reaction to the weight wrench (weight flag still independent).

Host CPU harvest (`_harvest_stem_tension_for_tcp_cpu`) must call the same composition.

### Harvest kernels

Extend `_limit_and_write_tcp_stem_wrench_kernel` and `_batched_limit_and_write_tcp_stem_wrench_kernel`:

- Inputs: `use_explicit_apple_inertia`, `apple_inertia_kgm2`, `robot_body_qd`, `robot_body_qd_prev`, `dt`
- After the weight block, if inertia is on and `mass > 0` and `dt > 0`: compute `a_com`, add `F += −m a_com`, `τ += r × (−m a_com) − I α`
- Then existing gain/caps

`harvest_stem_tension_for_tcp` / `harvest_batched_stem_tension` (the `fix_to_apple` path in `scene.py`) pass the new fields. Do **not** add inertia to velocity-delta `harvest_proxy_wrenches` (free proxy). `coupled_substep` copies `robot_state_0.body_qd` → `robot_tcp_qd_prev` **after** harvest (device `wp.copy`).

### Snapshot init

At scene build, after robot state exists:

```text
wp.copy(scene.robot_tcp_qd_prev, scene.robot_state_0.body_qd)
```

After `settle_then_weld` / any path that zeros robot `body_qd`, copy again so the first harvest is not `(qd − 0) / dt`. Never initialize `qd_prev` to a silent all-zero buffer if current `qd` is nonzero.

Allocate `robot_tcp_qd_prev` for every coupled FR3 scene that has `proxy_forces` (single-env and batched). If inertia is off, skip the kernel math; still `wp.copy` the snapshot so CUDA graph capture does not depend on the flag.

### `apple_inertia_kgm2` source

`I = 0.4 * m * R²` with `m` from AVBD `body_mass[apple]` (same as `apple_mass_kg`) and `R` from `FruitingSystemParams.apple_radius` (0 if missing ⇒ `I = 0` ⇒ force-only inertia). Heterogeneous worlds: per-env `m` and `R` as payload code already does.

---

## 4. Tests

TDD: formula tests first (no FR3), then harvest, then default payload mass 0, then CUDA graph.

### Unit (`test_explicit_apple_load.py`)

World frame, packing `[vx,vy,vz, wx,wy,wz]`:

| Test | Assert |
| --- | --- |
| `qd = qd_prev` | inertial wrench is 0; weight unchanged `F_z = −mg` |
| `a_z > 0`, `r = 0` | `F_z` more negative than weight by `−m a_z` |
| `a_com = g` | explicit force → 0 when weight+inertia both on |
| Pure `α`, `r = 0` | `τ = −I α` |
| Offset `r`, `α = 0` | `τ = r × (mg − m a)` |
| Constant `ω`, `α = 0` | `−m ω×(ω×r)` points **away** from TCP |
| `mass ≤ 0` or `dt ≤ 0` | inertial zeros |

### Harvest / scene (FR3, gain=1, caps off)

- Harvest inertia on vs off: TCP `proxy_forces` delta matches `apple_inertial_reaction_wrench_about_tcp` from `body_qd` vs `qd_prev`.
- Welded default: `stem_harvest_explicit_apple_inertia is True`; `body_mass[apple_payload] == 0`; payload body index still set.
- `fix_to_apple=False`: inertia False; override `True` raises.
- `apply_mujoco_apple_payload_inertias` while inertia True raises.
- `inertia=False`: `apply_mujoco_apple_payload_inertias` still sets mass = AVBD apple mass (existing A/B).
- CPU/GPU harvest parity including inertia (`test_proxy_coupling` / stem harvest parity).
- CUDA graph: `test_cuda_graph.py` still captures; finite wrench; `qd_prev` is a device buffer.
- Motion: payload mass 0, short VIC burst, inertia on moves TCP **less** than inertia off (arm feels apple load through `body_f`).

### Payload tests to retarget (`test_mujoco_apple_payload.py`)

Tests that currently expect build-time payload mass > 0:

- Default welded build: mass **0**, index still present.
- `apply_mujoco_apple_payload_inertias` after building with `stem_harvest_explicit_apple_inertia=False` still matches sphere `I`.
- `test_vic_tcp_motion_differs_with_and_without_payload_inertia` stays valid only on the `inertia=False` A/B path (payload in `M(q)`). Add a sibling test for harvest inertia vs no inertia with mass 0.

---

## 5. Documentation (required)

The payload body remaining in the model is easy to misread as “apple still in `M(q)`”. Docs must state the split in one place each:

| File | Change |
| --- | --- |
| `docs/handbook-coupled-simulation.md` §6 | Welded default: harvest `F = mg − ma`, `τ = r×F − Iα`; `apple_payload` is a mass-0 stub; never enable payload mass and harvest inertia together. |
| `docs/explicit-apple-load-tcp-harvest.md` | Formulas, packing, `qd_prev` init, resolver table for **weight and inertia**, CUDA-graph note. |
| `docs/handbook-sysid-scoring.md` warning | Sim `ft_wrist` = stem + explicit apple **weight and inertia**. Still no EE/tool/VIC. Still do not tare sim. |
| `mujoco_apple_payload.py` module docstring | Production welded path leaves mass 0; `apply_…` is A/B only and refuses when harvest inertia is on. |

Optional one-liner in `docs/gpu-coupling-optimization.md` (stem harvest already mentions explicit weight).

Do not claim `ft_wrist` is a full libfranka external wrench.

---

## 6. Error handling

| Condition | Behavior |
| --- | --- |
| `explicit_apple_inertia=True` and `fix_to_apple=False` | `ValueError` at resolve/build |
| `apply_mujoco_apple_payload_inertias` while inertia on | `ValueError` |
| `dt ≤ 0` or `mass ≤ 0` | inertial term 0 |
| Missing `robot_tcp_qd_prev` on a welded dynamic scene | allocate at build; harvest treats missing prev as `qd` (a = 0) only as a defensive fallback, then copies |
| `apple_radius` missing | `I = 0` (force `−m a_com` still applied; no `Iα`) |

---

## 7. Implementation order

1. Unit tests for kinematics + env-on-robot wrench (red).
2. `explicit_load.py` functions (green).
3. Harvest CPU + GPU kernels + `qd_prev` snapshot; CPU/GPU parity (red then green).
4. Resolver + skip `apply_mujoco_apple_payload_inertias` on welded default; mutual-exclusion raise.
5. Retarget payload tests; motion consistency test.
6. CUDA graph smoke.
7. Handbook + `explicit-apple-load` + H3 + payload module docstring.

---

## 8. Success criteria

- Welded `vic_pose` / real-replay builds: payload mass 0, harvest includes weight + inertia, `ft_wrist` is that 6-vector.
- Resting hold still matches today’s `F_z = −mg` (inertia 0).
- Accelerating the fruit changes `ft_wrist` and arm motion in the same direction (heavier effective load).
- CUDA graph capture still works.
- Free-apple builds unchanged (both explicit flags off).
