# Sys-ID §2.1 Data Collection — Deep-Dive Review

**Date:** 2026-06-15  
**Scope:** `apple_pick_gym/examples/example_gym_sysid.py`, `apple_pick_sim/system_id/`, `apple_pick_gym/envs/apple_pick_sysid_env.py`, supporting env stack.  
**Purpose:** Verify whether the current implementation correctly supports quasi-static stiffness data collection (M3.0 §2.1), and identify what must change before data can be stored to file for the CEM/MMD tuning stage.

---

## 1. What is working correctly

### 1.1 Fibonacci hemisphere sampling (`fibonacci_hemisphere.py`)

The implementation is correct:

- `_fibonacci_sphere(n)` generates `(n, 3)` unit vectors on the full sphere using the golden-ratio Fibonacci lattice (indices offset by 0.5, azimuth via `theta = 2π i / φ`, elevation via `phi = arccos(1 - 2i/n)`). This is the standard formulation and avoids polar over-sampling.
- `sample_fibonacci_hemisphere(n, stem_dir)` builds a pool 4× larger than needed, keeps only the forward-facing half (`dot(d, stem_dir) >= 0`), then draws the first `n` from that forward set with cycling if the pool runs short. All outputs are re-normalised.
- Tests confirm unit norms, forward-facing invariant, exact count, and ≥15° pairwise separation for 10 samples.

**Verdict:** faithful to §2.1 spec.

### 1.2 Trajectory phase machine (`quasi_static_trajectory.py`)

`QuasiStaticTrajectory.iter_frames()` yields `(phase, EEVelocity)` tuples in the correct sequence:

```
move_out (continuous) → hold (zero velocity) → return (continuous) → … (repeat per direction)
```

Default config: `movement_per_step_m=0.05`, `total_movement_m=0.10` (2 increments by default; must be an integer multiple), `hold_duration_s=1.5`, `move_speed_mps=0.2`, `control_hz=60.0`.

- Frame counts are computed with `math.ceil`, so fractional durations round up correctly.
- `current_amplitude_m` is updated per-frame throughout move_out and decremented during return; it is accessible at any point in the trajectory via property.
- `current_direction` returns a copy of the active direction.
- Return speed matches move speed, and the net displacement per direction integrates to zero (tested).
- Hold phase emits exactly `ceil(1.5 × 60) = 90` zero-velocity frames (tested).

**Verdict:** correct and well-tested.

### 1.3 `ExcitationContext` (`excitation_state.py`)

Frozen dataclass with `type` (string), `f_inst` (float), `direction` (3-vector, normalised in `__post_init__`). The type tokens match the `_EXCITATION_TYPE_TO_INT` table in the env. This is sufficient metadata for §3.1's `φ_exc` feature.

### 1.4 `ApplePickSysIdEnv` observation contract

The observation space is a proper superset of the VIC env:

| Key | Shape | Source |
|-----|-------|--------|
| `woody_part_start_pos` | `(N×3,)` | parent fixed-joint anchors |
| `woody_part_end_pos` | `(N×3,)` | child fixed-joint anchors |
| `woody_part_force` | `(N×6,)` | per-junction `[F, τ]` |
| `apple_pos` | `(3,)` | apple body CoM |
| `tcp_force` | `(6,)` | harvested coupling wrench |
| `tcp_velocity` | `(6,)` | TCP spatial velocity |
| `ft_wrist` | `(6,)` | plant wrench at TCP (VIC-subtracted for wrench-only mode) |
| `excitation_type` | scalar int | 0=quasi_static, 1=trans_chirp, 2=torsional |
| `excitation_f_inst` | scalar float | 0.0 for quasi-static |
| `excitation_direction` | `(3,)` | current push direction |
| `tcp_pos` | `(3,)` | **actual** TCP body position from `robot_state_0`, not the VIC target |

`tcp_pos` reading from `robot_state_0.body_q` rather than `controller.target_tf` is the right choice: stiffness estimates from `K = ||ft_wrist|| / ||Δtcp_pos||` are unbiased by VIC compliance lag (tested by `test_sysid_env_tcp_pos_is_actual`).

### 1.5 Action space

`Box(6)` with symmetric `±max_linear_vel` and `±max_angular_vel` bounds, clipped in `_action_to_command`. The conversion from `EEVelocity` to flat `np.ndarray` in `_action_from_velocity` and back is consistent. The example script uses the trajectory's velocity objects directly, which is correct.

### 1.6 Weld direction infrastructure (beyond original plan)

`ApplePickSysIdEnv` adds `robot_facing_weld=True` by default: at `reset()` time it samples a Fibonacci hemisphere of robot-facing weld directions and cycles through them across resets. This means successive data-collection episodes automatically vary the gripper orientation, which is useful for multi-weld-pose stiffness identification. The `info["weld_direction"]` return is tested. This is a clean extension that does not interfere with data collection.

### 1.7 Tests

`test_quasi_static_sysid.py` (CPU-only) covers all trajectory utilities. `test_sysid_env.py` covers action/observation contracts, VIC gains defaults, `tcp_pos` fidelity, context roundtrip, and weld cycling. Coverage is adequate for the units tested.

---

## 2. Gaps: what is missing before data can be saved to file

### 2.1 No data persistence

**This is the most critical gap.** `example_gym_sysid.py` accumulates `hold_forces` in a `defaultdict` and prints mean values at the end. Nothing is written to disk. There is no `--output` argument, no `np.savez`, no pickle, no JSON, and no structured dataset format.

Before moving to the tuning stage, the script needs to collect the full observation stream (or at minimum the fields needed for CEM/MMD) and save them. The natural format for a single simulation run is a compressed NumPy archive (`.npz`) or a list of dicts serialised to pickle. The CEM/MMD pipeline in §3 needs transition pairs `[s_t, s_{t+1}]`, which requires the complete per-frame timeseries, not just hold-phase summaries.

### 2.2 Narrow data capture during hold phase

Currently only `obs["ft_wrist"][:3]` (3D force only) is saved per hold frame:

```python
hold_forces[(dir_idx, traj.current_step_index)].append(
    np.asarray(obs["ft_wrist"][:3], dtype=np.float64)
)
```

The following fields are silently discarded:

| Missing field | Why it matters |
|---------------|----------------|
| `ft_wrist[3:6]` (torque) | Full 6-DOF wrench is the unit of measurement for impedance identification |
| `tcp_pos` | Displacement from rest; required for `K = F / Δx` |
| `excitation_direction` | Needed to decompose `F` into axial and shear components |
| `woody_part_force` | Per-junction loads — the state variables that CEM matches |
| `apple_pos` | Node position observable; part of `P_nodes` in §3.1 |
| `tcp_velocity` | Should be near zero in hold (useful as a quality filter) |
| current amplitude | `traj.current_amplitude_m` is available but not recorded |

Saving only `F[:3]` supports printing a mean force summary but does not produce a dataset usable for stiffness identification or CEM.

### 2.3 No initial TCP position reference

The script reads `tcp_target` at reset time:

```python
tcp_target = np.asarray(env.unwrapped._controller.target_tf[:3], dtype=np.float64)
```

This is used only to infer `stem_dir`. It is not stored as a reference for computing displacement `Δx = obs["tcp_pos"] - tcp_target`. Without a stored initial position, the saved `tcp_pos` values have no displacement reference, so stiffness cannot be computed from the saved data in isolation.

### 2.4 No logging during move_out and return phases

The §3.1 state definition covers the full trajectory, not only hold snapshots:

> Observable state $s_t$: $P_\text{nodes}$, $v_{ee}$, $W_{ee}$, $\phi_\text{exc}$, $\hat{u}$.

The CEM/MMD pipeline matches *distributions of transitions* `[s_t, Δs_t]`. This requires data from the entire timeseries including move_out and return phases. Hold-only sampling:
- Provides at most one displacement level per direction (at maximum 10 cm)
- Misses the K(x) profile across the 0–10 cm range
- Produces a sparse dataset that may be insufficient for MMD to distinguish parameter differences

Logging every frame (or at least every move_out frame) would capture the full K(x) curve per direction since the 0.05 m/s motion is slow enough to be quasi-static.

### 2.5 Multi-amplitude holds exist; persistence is still hold-only summaries

`QuasiStaticTrajectory` now performs a **move_out → hold** cycle at each increment (`movement_per_step_m` up to `total_movement_m`). With `--movement-per-step-m 0.02 --total-movement-m 0.10` the example prints mean steady-state force at 2, 4, 6, 8, and 10 cm per direction. The trajectory phase machine is correct for §2.1 stepped mapping.

The gap is **collection**, not trajectory shape: only hold-phase `ft_wrist[:3]` samples are aggregated into printed means. Raw per-frame observations across move_out, hold, and return are still discarded (see §2.4).

---

## 3. Divergences from the original plan (non-blocking)

### 3.1 VIC linear stiffness default

The plan (`m3.0_§2.1_quasi-static_sysid_39e08160.plan.md`) specified `vic_linear_k=3000.0` to reduce compliance undershoot during holds. `ApplePickSysIdEnv` defaults to `vic_linear_k=2000.0` N/m (`test_sysid_env_default_vic_stiffness_and_no_stem_caps`).

At 2000 N/m with a 1 N stem force, VIC lag introduces ~0.5 mm position error. At full-range 30 N, that is ~15 mm — smaller than the 10 cm excursion but still visible in `tcp_pos` during holds.

This is a tunable parameter and the plan's 3000 N/m was itself advisory. Because `tcp_pos` reads the actual body state (not the VIC target), `K = F / Δx` using measured displacement is not biased by compliance lag; commanded vs achieved amplitude can still differ at low stiffness.

### 3.2 Wrench safety guard not implemented

The plan specified:
```python
compute_terminated → True if ||ft_wrist[:3]|| > max_tcp_force_n (default 30.0 N)
```

The current `ApplePickSysIdEnv` does not override `compute_terminated`. It inherits the always-`False` implementation from `ApplePickCoupledEnv`. The test `test_sysid_env_no_force_termination` confirms termination never fires.

For sim-only use this is safe (no physical robot at risk), but the safety guard is part of the protocol for field deployment (§2 "Safety" clause: abort if force exceeds ~80% of stem-break estimate). For pure sim data collection the guard is not strictly required, but it should be noted that the sim can drive arbitrarily large forces without stopping.

### 3.3 `run_quasi_static.py` duplicates example script logic

`apple_pick_sim/system_id/run_quasi_static.py` is a near-duplicate of the gym example script without viewer support. Both accumulate `hold_forces` and print summaries. There is no shared helper, so data collection logic would need to be kept in sync across both files. The smoke script is useful for headless CI, but the duplication creates a maintenance burden once data persistence is added.

---

## 4. What the observation stream looks like right now

At each env step during a hold phase, the observation dict has:

```
ft_wrist[0:3]   → stem force at TCP, world frame [N]   ← currently saved
ft_wrist[3:6]   → stem torque at TCP, world frame [N·m] ← discarded
tcp_pos[0:3]    → actual TCP body position [m]           ← discarded
tcp_velocity    → TCP spatial velocity [m/s, rad/s]      ← discarded
apple_pos       → apple COM position [m]                 ← discarded
woody_part_force → per-junction [F, τ] [N, N·m]         ← discarded
excitation_*    → trajectory metadata                    ← discarded
```

The discarded fields are available at every step. Saving them costs nothing extra computationally. The data is being thrown away at the collection stage.

---

## 5. What needs to happen before saving data to file

Listed in priority order for the data collection goal:

1. **Define a data record structure** — at minimum, per-frame records containing: `sim_time`, `phase`, `dir_idx`, `amplitude_m`, `direction` (3-vector), `ft_wrist` (6), `tcp_pos` (3), `tcp_velocity` (6), `apple_pos` (3), `woody_part_force` (N×6). The initial `tcp_pos` at reset should be stored as a metadata field.

2. **Log every frame, not only hold frames** — the full timeseries from move_out, hold, and return phases is needed for the transition-feature representation `[s_t, Δs_t]` required by CEM/MMD. Hold-only data is too sparse.

3. **Add `--output` argument and save at end of episode** — `np.savez_compressed` with arrays for each field plus metadata (seed, n_directions, config params, weld_direction, initial_tcp_pos) is sufficient. The filename convention could be `sysid_run_<seed>_<n_dirs>dirs.npz`.

4. **Store initial TCP position** — save `tcp_target` (from `controller.target_tf[:3]` at reset) as metadata so the replay-side can compute `Δx = tcp_pos - tcp_pos_initial`.

5. **Consider intermediate holds** — if K(x) profiling at multiple amplitudes is needed (likely for detecting nonlinear stiffness), modify `QuasiStaticTrajectory` to support an optional stepped mode with holds at each `step_size_m` increment. This is a trajectory config change, not an env change.

---

## 6. Relation to the CEM/MMD replay path

The downstream M3.1 step (`ApplePickReplayEnv`) expects to receive recorded EE velocity telemetry (`v_ee(t)`) and replay it in simulation with varied `FruitingSystemParams` `θ`. The current data collection does not save the action sequence (the velocity commands fed to `env.step`). Without recording `vel.linear` and `vel.angular` per frame, the replay env cannot be driven by the recorded trajectory.

The action sequence is deterministic from the trajectory config and `directions` array, so it can be reconstructed — but this creates a coupling between the data file and the trajectory generator. Saving the action sequence alongside observations is cleaner and avoids replay fidelity issues if the trajectory code changes.

---

## 7. Summary table

| Capability | Status | Notes |
|---|---|---|
| Fibonacci hemisphere sampling | ✓ Correct | Well tested |
| Trajectory phase machine | ✓ Correct | Phase sequence, hold frame count, zero net displacement all tested |
| ExcitationContext metadata | ✓ Correct | Frozen, normalised, type-safe |
| Observation space (env) | ✓ Correct | All sysid keys present; `tcp_pos` reads actual body state |
| Action space (env) | ✓ Correct | Box(6), clipped |
| VIC integration | ✓ Correct | Joint-torque VIC default; ft_wrist excludes VIC contribution |
| Data persistence | ✗ Missing | Nothing saved to disk |
| Full observation logging | ✗ Missing | Only `ft_wrist[:3]` saved; 8+ fields discarded |
| Initial TCP reference position | ✗ Missing | Needed for Δx → K computation |
| Full timeseries (not only hold) | ✗ Missing | CEM/MMD needs transitions across all phases |
| Action sequence saved | ✗ Missing | Needed for deterministic replay in M3.2 |
| Intermediate holds at each increment | ✓ Trajectory | Per-increment hold phases; example prints mean force per step |
| Wrench safety guard | ✗ Missing (intentional) | Plan specified force guard; not needed for sim-only use |
| VIC stiffness 3000 N/m | ✗ Diverges from plan | 2000 N/m default; K estimate still unbiased if `tcp_pos` used for Δx |
| Smoke script | ✓ Present | `run_quasi_static.py` headless, but duplicates example logic |
