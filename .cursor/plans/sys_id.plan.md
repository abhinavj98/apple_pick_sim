---
name: ""
overview: ""
todos: []
isProject: false
---

# M3.0 §2.1 — Quasi-Static Stepped Stiffness Mapping

## Step 0 — Worktree setup

All work happens in `../apple_pick_sim-sysid` on branch `feature/sysid`. Commands (from repo root):

## What we are building

§2.1 drives the EE through ~10 Fibonacci-hemisphere directions, stepping 2 cm at a time to 10 cm and back, holding 1–2 s per pose to let transients decay, and logging steady-state F/T. Results define amplitude bounds for §2.2/2.3.

## New files

### `apple_pick_sim/system_id/` (new module)

- `__init__.py` — exports `sample_fibonacci_hemisphere`, `QuasiStaticTrajectory`, `ExcitationContext`
- `fibonacci_hemisphere.py` — `sample_fibonacci_hemisphere(n, stem_dir) -> ndarray (n,3)`:
  - Uses golden-ratio Fibonacci lattice on the unit sphere, then selects the `n` points whose dot product with `stem_dir` ≥ 0 (forward hemisphere). If fewer than `n` survive, reflect duplicates. All output vectors are unit-norm.
- `quasi_static_trajectory.py` — `QuasiStaticStepConfig` (dataclass) + `QuasiStaticTrajectory`:
  - `QuasiStaticStepConfig`: `step_size_m=0.02`, `n_steps=5` (0→10 cm), `hold_duration_s=1.5`, `move_speed_mps=0.05`, `control_hz=60.0`
  - `QuasiStaticTrajectory(directions, config)`: generates a sequence of `(phase, EEVelocity)` frames via a Python generator `iter_frames()`:
    - Phase = `"move_out"` | `"hold"` | `"return"` per direction
    - Each move phase emits velocity frames for `step_size / move_speed` seconds
    - Each hold phase emits zero-velocity frames for `hold_duration * control_hz` frames
    - Return phase drives EE back to center at same speed
  - `current_direction()` and `current_amplitude_m()` properties for the env to read as excitation context
- `excitation_state.py` — `ExcitationContext` frozen dataclass: `type: str` (`"quasi_static"`/`"translational_chirp"`/`"torsional"`), `f_inst: float` (0.0 for quasi-static), `direction: ndarray (3,)`

### `apple_pick_gym/envs/apple_pick_sysid_env.py` (new)

`ApplePickSysIdEnv(ApplePickVicEnv)`:

- **Action space**: `Box(-max_lin, max_lin, (3,)) × Box(-max_ang, max_ang, (3,))` → flat `Box(6)` (same as `ApplePickReplayEnv`); `max_linear_vel=0.2`, `max_angular_vel=1.0` constructor params
- **VIC stiffness default override**: `vic_linear_k=3000.0` (up from base 800 N/m) to reduce compliance undershoot during quasi-static holds — still passed to `ApplePickVicEnv` as a constructor kwarg, so it remains tunable
- **Extra obs keys** added on top of VIC obs:
  - `excitation_type`: `Discrete(3)` → int scalar (0=quasi_static, 1=trans_chirp, 2=torsional)
  - `excitation_f_inst`: `Box` scalar float (instantaneous frequency; 0.0 for quasi-static)
  - `excitation_direction`: `Box(3)` unit direction of current push
  - `tcp_pos`: `Box(3)` actual TCP world position read from `robot_state_0.body_q[tcp_body_index][:3]` — **not** the commanded `target_tf`, so `K = ft_wrist_force / tcp_pos_displacement` is unbiased by VIC compliance
- **New constructor params**:
  - `max_tcp_force_n: float = 30.0` — wrench safety guard threshold
  - `max_linear_vel: float = 0.2`, `max_angular_vel: float = 1.0`
- `_action_to_command(action)`: converts `ndarray (6,)` to `EEVelocity(linear=action[:3], angular=action[3:])`
- `set_excitation_context(ctx: ExcitationContext)`: stores context; called by the trajectory runner between steps
- `compute_terminated(obs, info)`: returns `True` if `np.linalg.norm(obs["ft_wrist"][:3]) > max_tcp_force_n`; otherwise `False`
- **Observation space**: extend `_observation_space_for` with the four new keys

## Modified files

### `apple_pick_gym/envs/__init__.py`

Add `from apple_pick_gym.envs.apple_pick_sysid_env import ApplePickSysIdEnv` and include in `__all_`_.

### `apple_pick_gym/__init__.py`

Register `"ApplePickSysId-v0"` → `ApplePickSysIdEnv`.

## Tests (written first — TDD)

`apple_pick_sim/tests/test_quasi_static_sysid.py` (CPU-only, no full sim needed):


| Test                                             | Asserts                                                                |
| ------------------------------------------------ | ---------------------------------------------------------------------- |
| `test_fibonacci_hemisphere_unit_norms`           | All directions have `norm ≈ 1.0`                                       |
| `test_fibonacci_hemisphere_forward_facing`       | All dot products with `stem_dir ≥ 0`                                   |
| `test_fibonacci_hemisphere_count`                | Returns exactly `n` directions                                         |
| `test_fibonacci_hemisphere_approx_uniform`       | No two directions within 15° of each other (10-pt lattice)             |
| `test_quasi_static_trajectory_phase_sequence`    | For 2 directions, phases alternate: move→hold→return, move→hold→return |
| `test_quasi_static_trajectory_returns_to_center` | Net displacement after one full direction cycle sums to zero           |
| `test_quasi_static_trajectory_hold_frame_count`  | Hold emits exactly `ceil(hold_s * control_hz)` zero-velocity frames    |
| `test_quasi_static_trajectory_move_speed`        | Move frames have velocity norm ≤ `move_speed_mps + ε`                  |


`apple_pick_gym/tests/test_sysid_env.py` (uses existing mocked/fast env patterns):


| Test                               | Asserts                                                                                          |
| ---------------------------------- | ------------------------------------------------------------------------------------------------ |
| `test_sysid_env_action_space`      | Action space is `Box(6)`, clipped to `max_lin/ang` bounds                                        |
| `test_sysid_env_obs_keys`          | `reset()` obs contains `excitation_type`, `excitation_f_inst`, `excitation_direction`, `tcp_pos` |
| `test_sysid_env_tcp_pos_is_actual` | `obs["tcp_pos"]` matches `robot_state_0.body_q[tcp][0:3]`, not `controller.target_tf`            |
| `test_sysid_env_wrench_guard`      | Manually set `max_tcp_force_n=0.001`; after step with nonzero plant force, `terminated=True`     |
| `test_sysid_env_context_roundtrip` | `set_excitation_context(ctx)` reflects in next `_make_obs()`                                     |


## Smoke script

`apple_pick_sim/system_id/run_quasi_static.py` (runnable via `uv run python apple_pick_sim/system_id/run_quasi_static.py --viewer null`):

- Builds `ApplePickSysIdEnv(render_mode=None)`, calls `reset()`
- Constructs `QuasiStaticTrajectory` from Fibonacci hemisphere (stem_dir inferred from initial obs: direction from TCP to `apple_pos`)
- Runs `iter_frames()` loop, feeding each `EEVelocity` to `env.step()`, calling `set_excitation_context`
- Prints per-direction mean steady-state `ft_wrist` force during hold phases

## Stem direction inference

At `reset()` time in the smoke script (not the env itself): `stem_dir = normalize(obs["apple_pos"] - tcp_pos)`, where `tcp_pos` is read from the controller's current target transform. This approximates "toward the apple" at grasp time.

## Implementation doc

`docs/system-id-quasi-static-implementation.md` — math for Fibonacci lattice, trajectory phase machine, wrench guard invariants, test list, validation command.

## Validation command (after implementation)

```bash
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_quasi_static_sysid.py apple_pick_gym/tests/test_sysid_env.py -q
```

