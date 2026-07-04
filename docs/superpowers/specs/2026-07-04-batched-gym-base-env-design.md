# Batched gym base env (V.3.3) — design spec

**Date:** 2026-07-04  
**Slice:** V.3.3 — `ApplePickBatchedBaseEnv`  
**Status:** Approved for implementation planning  

## Summary

Introduce a SKRL-native, GPU-resident Gymnasium adapter over `BatchedHeterogeneousCoupledSim`. Build once in `__init__`; cheap episode reset via device-side `EpisodeStateSnapshot`. Default controller is VIC with `fix_to_apple=True`. Supports `num_envs >= 1` with batched Torch observations and actions.

Concrete env `ApplePickBatchedVicEnv` ships in this slice for parity tests only; migration of legacy gym envs is V.3.4.

## Goals

- Wire gym to the batched heterogeneous sim API (no single-world `build_coupled_fruiting_fr3` in the new path).
- Keep the simulation hot path on GPU (`gather_batched_obs` → `wp.to_torch` at the gym boundary).
- Expose `num_envs`, `device`, and SKRL batch conventions (`(N, …)` obs, `(N, 1)` reward/done).
- `num_envs=1` observation semantics match legacy `ApplePickVicEnv` v3 fields via a documented mapping (not identical key layout).
- Multi-env smoke at `N > 1`.

## Non-goals (V.3.3)

- Migrating `ApplePickCoupledEnv`, `ApplePickSysIdEnv`, `ApplePickReplayEnv`.
- Gym registration (`ApplePickBatchedVic-v0`), examples, SKRL training scripts (M2.2c).
- Full rebuild or per-episode θ DR on `reset()` (per-env params fixed for env lifetime).
- `gymnasium.vector.VectorEnv` wrapper.

## Architecture

```
ApplePickBatchedBaseEnv (gym.Env, abstract)
  ├── BatchedHeterogeneousCoupledSim     # built once in __init__
  ├── EpisodeStateSnapshot               # capture after build; restore on reset()
  ├── batched_envs.obs_torch.obs_dict_from_bufs()
  └── abstract: reward, terminated, action/obs hooks

ApplePickBatchedVicEnv (concrete)
  └── VIC obs layout; stub reward=0, terminated=False
```

### Lifecycle

| Phase | Behavior |
| ----- | -------- |
| `__init__` | Sample per-env params, build sim (`gym_defaults(num_envs=N)`), `EpisodeStateSnapshot.capture(sim)` |
| `reset()` | `snapshot.restore(sim)`, reset step counter / RNG, return batched obs |
| `step(action)` | Validate `Tensor(N, 6)` → `sim.step(actions)` → `gather_obs()` → torch obs, `(N, 1)` scalars |

### Defaults

| Setting | Value |
| ------- | ----- |
| Controller | `vic` (joint-torque VIC) |
| `fix_to_apple` | `True` |
| Actions | `Box(6)` EE velocity; runtime `(N, 6)` float32 on `self.device` |
| `obs.allocate_buffers` | `True` |
| Reset | Build-once + snapshot restore (not full rebuild) |

### SKRL contract

- `self.num_envs: int`
- `self.device: torch.device`
- Obs: nested `spaces.Dict`; each leaf tensor has leading batch dim `(N, …)`
- `reward`, `terminated`, `truncated`: `Tensor(N, 1)`
- Optional `wrap_env(env, wrapper="gymnasium")` for rollout memory flattening

## Sim API — `EpisodeStateSnapshot`

**Module:** `apple_pick_sim/coupled_fruiting/episode_state_snapshot.py`

```python
snapshot = EpisodeStateSnapshot.capture(sim)
snapshot.restore(sim)
```

Thin wrappers on `BatchedHeterogeneousCoupledSim`: `capture_episode_snapshot()`, `restore_episode_snapshot()`.

### Captured buffers (device `wp.copy` clones)

- Robot: `robot_state_0.body_q`, `body_qd`, `joint_q`, `joint_qd`; `robot_model.joint_q`, `joint_qd`
- Cable: `cable.state_0` and `state_1` `body_q`, `body_qd`
- Batched VIC IK targets (`Fr3BatchedEEImpedanceController` device target buffers)
- `vic_jt_default_dof_pos` when joint-torque VIC is configured

### Restore side effects

Mirror single-env `ApplePickSysIdEnv.restore_grasp_pose()` (batched):

- Copy all captured buffers back
- `init_robot_mujoco_step_buffers`, hold actuator targets
- `sync_solver_body_q_prev_from_state`
- Zero `proxy_forces`, `coupling_forces_cache`
- Reset `vic_target_twist` per env
- `ee_ctrl.sync_target_from_state`

**Capture timing:** immediately after post-weld IK bootstrap completes — not the pre-weld `SettledCheckpoint` (that remains a build-time disk cache only).

## Observation layout — `ApplePickBatchedVicEnv`

Batched env uses a **junction-grouped** layout. Legacy v3 uses flat / split keys; map between them in parity tests only.

### Top-level keys

| Key | Per-env leaf shape | Batched tensor | Source (`gather_batched_obs`) |
| --- | ------------------ | -------------- | ----------------------------- |
| `woody_part_info` | nested dict (below) | same structure, each leaf `(N, …)` | woody + wrench buffers |
| `apple_pos` | `(3,)` | `(N, 3)` | `apple_pos` |
| `tcp_force` | `(6,)` | `(N, 6)` | `tcp_force` |
| `tcp_velocity` | `(6,)` | `(N, 6)` | `tcp_velocity` |
| `ft_wrist` | `(6,)` | `(N, 6)` | `tcp_coupling_force` |

### `woody_part_info` (per junction name)

One nested `spaces.Dict` entry per junction (same names as `env.junction_names`):

| Sub-key | Per-env shape | Batched | Semantics |
| ------- | ------------- | ------- | --------- |
| `anchors_pos` | `(6,)` | `(N, 6)` | `[parent_xyz(3), child_xyz(3)]` world-frame fixed-joint anchors |
| `anchor_force` | `(6,)` | `(N, 6)` | `[Fx, Fy, Fz, τx, τy, τz]` on child body at child COM |

Example:

```python
obs["woody_part_info"]["stem_apple"]["anchors_pos"]   # Tensor(N, 6)
obs["woody_part_info"]["stem_apple"]["anchor_force"]  # Tensor(N, 6)
```

### Buffer reshaping (`apple_pick_gym/batched_envs/obs_torch.py`)

From `BatchedObsBuffers` with `num_envs=N`, `num_junctions=J`:

- `woody_parent_pos`, `woody_child_pos`: `(N*J, 3)` → reshape `(N, J, 3)` → per-junction `anchors_pos = cat(parent, child, dim=-1)`
- `woody_force`, `woody_torque`: `(N*J, 3)` each → `anchor_force = cat(force, torque, dim=-1)`

Junction order matches world-0 `fruiting_tree_fixed_joints` labels (topology fixed per batch).

### v3 parity mapping (`num_envs=1`)

Legacy `ApplePickVicEnv` obs → batched env (compare after `.cpu().numpy()`):

| v3 key | Batched key |
| ------ | ----------- |
| `woody_part_start_pos[name]` | `woody_part_info[name]["anchors_pos"][:3]` |
| `woody_part_end_pos[name]` | `woody_part_info[name]["anchors_pos"][3:]` |
| `woody_part_force[i*6:(i+1)*6]` | `woody_part_info[junction_names[i]]["anchor_force"]` |
| `apple_pos`, `tcp_force`, `tcp_velocity`, `ft_wrist` | unchanged top-level keys |

`info["obs_schema"]` remains `"v3"` (semantic contract). Add `info["obs_layout"] = "batched_vic"` to distinguish key nesting from legacy envs.

### Gym `observation_space`

Unbatched leaf shapes (no `N` in space definition); env returns batched tensors:

```python
spaces.Dict({
    "woody_part_info": spaces.Dict({
        name: spaces.Dict({
            "anchors_pos": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "anchor_force": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
        })
        for name in junction_names
    }),
    "apple_pos": spaces.Box(..., shape=(3,), ...),
    "tcp_force": spaces.Box(..., shape=(6,), ...),
    "tcp_velocity": spaces.Box(..., shape=(6,), ...),
    "ft_wrist": spaces.Box(..., shape=(6,), ...),
})
```

## Config changes

Update `BatchedHeterogeneousCoupledSimConfig.gym_defaults()`:

- `controller.mode = "vic"`
- `robot.fix_to_apple = True`
- `obs.allocate_buffers = True` (unchanged)

## Package layout (`apple_pick_gym/batched_envs/`)

Batched GPU gym envs live in a dedicated subpackage, separate from legacy single-world envs under `apple_pick_gym/envs/`.

```
apple_pick_gym/
  batched_envs/
    __init__.py                    # export ApplePickBatchedBaseEnv, ApplePickBatchedVicEnv
    apple_pick_batched_base_env.py
    apple_pick_batched_vic_env.py
    obs_torch.py                   # obs_dict_from_bufs() (was batched_obs_torch.py)
  envs/                            # legacy single-world envs (unchanged until V.3.4)
  tests/
    test_batched_vic_env.py
```

No Gymnasium imports in `apple_pick_sim/`. Shared obs reshaping stays in `apple_pick_gym/batched_envs/` (not `apple_pick_sim/`).

## Files

| Path | Role |
| ---- | ---- |
| `apple_pick_sim/coupled_fruiting/episode_state_snapshot.py` | GPU snapshot capture/restore |
| `apple_pick_sim/coupled_fruiting/batched_heterogeneous_coupled_sim.py` | `capture_episode_snapshot` / `restore_episode_snapshot` |
| `apple_pick_sim/coupled_fruiting/batched_heterogeneous_config.py` | `gym_defaults()` VIC preset |
| `apple_pick_sim/tests/test_episode_state_snapshot.py` | Snapshot round-trip after N steps |
| `apple_pick_gym/batched_envs/obs_torch.py` | `obs_dict_from_bufs()` |
| `apple_pick_gym/batched_envs/apple_pick_batched_base_env.py` | Abstract base |
| `apple_pick_gym/batched_envs/apple_pick_batched_vic_env.py` | Concrete VIC env |
| `apple_pick_gym/tests/test_batched_vic_env.py` | Parity + multi-env smoke |

## Testing

### TDD order

1. `test_episode_state_snapshot.py` — capture → step → restore → state matches baseline
2. `test_batched_obs_torch.py` — buffer reshape matches CPU reference
3. `test_batched_vic_env.py`:
   - `num_envs=1` v3 parity vs `ApplePickVicEnv` (via mapping table)
   - `num_envs=4` shape / `num_envs` / `device` / reset-restore smoke

### Validation

```bash
uv sync --extra gym --extra vic --extra dev

uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_episode_state_snapshot.py \
  apple_pick_gym/tests/test_batched_obs_torch.py \
  apple_pick_gym/tests/test_batched_vic_env.py -q
```

## Error handling

- `reset()` / `step()` before build: `RuntimeError`
- Action wrong shape/device/dtype: `ValueError` from `ControllerConfig.validate_actions`
- `num_junctions > max_woody_parts`: `ValueError` at space setup (same as legacy base)
- Snapshot restore without prior capture: `RuntimeError`

## Follow-ups (V.3.4+)

- Migrate legacy envs; adapters from `woody_part_info` ↔ v3 flat keys for replay Parquet
- Optional `obs_schema` v4 if single-env envs adopt `woody_part_info` layout
- SKRL smoke test (M2.2c) on `ApplePickBatchedVicEnv`
- Hybrid reset with new per-env params (backlog)
