# SysID trajectory storage

Parquet persistence for quasi-static (and future) sysID rollouts, plus dataset-backed replay for CEM/MMD parameter tuning.

## Status

Recording and replaying are complete for sim-to-sim datasets. Collection writes per-frame observations/actions, episode metadata, and an optional privileged initial-state snapshot. Replay loads the dataset, restores the saved Newton state when available, and applies the recorded EE velocity actions open-loop while recomputing observations from the live simulation.

The next sysID step is **observation-only replay initialization**: replay should initialize from recorded observations and calibration metadata, not from privileged simulator arrays such as body poses, velocities, solver previous-state buffers, or saved controller internals. The first validation target is sim-to-sim replay with the `.npz` snapshot withheld, so we can measure how close an observation-derived initial state gets before using real-world data.

## Layout

```
<output_dir>/
  metadata.parquet              # one row per episode (appended across runs)
  frames/
    <episode_id>.parquet        # per-frame timeseries for one episode
```

## Required frame columns

| Column | Shape | Description |
|--------|-------|-------------|
| `episode_id` | str | FK to metadata |
| `step_idx` | int | Frame index within episode |
| `phase` | int8 | 0=move_out, 1=hold, 2=return |
| `excitation_type` | int8 | 0=quasi_static, 1=chirp, 2=torsional |
| `excitation_direction` | list[f32]×3 | Pull direction unit vector (world) |
| `action` | list[f32]×6 | EE velocity command `[vx,vy,vz,wx,wy,wz]` — **used for replay** |
| `tcp_velocity` | list[f32]×6 | EE velocity observation |
| `woody_start__<junction>` | list[f32]×3 | Parent-side anchor for one junction (e.g. `woody_start__stem_apple`) |
| `woody_end__<junction>` | list[f32]×3 | Child-side anchor for the same junction |
| `ft_wrist` | list[f32]×6 | Plant F/T at TCP `[F, τ]` |

Per-junction woody columns are dynamic (one start/end pair per entry in `junction_names` metadata). There are no flat `woody_part_start_pos` / `woody_part_end_pos` frame columns.

## Bonus frame columns (written by default)

`sim_time`, `dir_idx`, `amplitude_m`, `tcp_pos`, `apple_pos`, `woody_part_force`

## Required metadata columns

| Column | Description |
|--------|-------------|
| `episode_id` | Primary key |
| `weld_direction` | Gripper weld orientation at reset |
| `excitation_type` | `"quasi_static"` \| `"translational_chirp"` \| `"torsional"` |
| `n_woody_parts` | Number of woody fixed joints |
| `junction_names` | Ordered junction labels for woody parquet columns |
| `params_fingerprint` | JSON stiffness/damping summary of `FruitingSystemParams` θ |
| `control_hz` | Env step rate |

## Bonus metadata columns

`timestamp`, `seed`, `n_directions`, `initial_tcp_pos`, `fixture_path`, trajectory config (`movement_per_step_m`, `total_movement_m`, `hold_duration_s`, `move_speed_mps`, `skip_return`)

## Displacement convention

Steady-state stiffness from a recorded episode:

\[
\Delta x = \texttt{tcp\_pos} - \texttt{initial\_tcp\_pos}
\]

`initial_tcp_pos` is stored in episode metadata (TCP VIC target at reset). Use hold-phase rows (`phase == 1`) for quasi-static K estimates.

## Collection

From the repository root:

```bash
uv sync --extra gym --extra vic --extra dev
uv run python apple_pick_gym/examples/example_gym_sysid.py \
  --viewer null --n-directions 1 --max-steps 200 \
  --output /tmp/sysid_dataset
```

Implementation: [`apple_pick_sim/system_id/trajectory_store.py`](../apple_pick_sim/system_id/trajectory_store.py), wired in [`apple_pick_gym/examples/example_gym_sysid.py`](../apple_pick_gym/examples/example_gym_sysid.py).

## Replay example script

[`apple_pick_gym/examples/example_gym_replay.py`](../apple_pick_gym/examples/example_gym_replay.py) replays a collected dataset headless or with the Newton viewer:

```bash
uv run python apple_pick_gym/examples/example_gym_replay.py \
  --dataset tmp/sysid_dataset --viewer null

uv run python apple_pick_gym/examples/example_gym_replay.py \
  --dataset tmp/sysid_dataset --list-episodes
```

## Replay env

`ApplePickReplay-v0` (`ApplePickReplayEnv`) loads a dataset and applies stored `action` rows open-loop. The Gym `action` argument is ignored during replay.

Current replay is state-initialized when `initial_states/<episode_id>.npz` exists. That privileged snapshot is useful for proving storage, action replay, and live-vs-recorded observation comparison, but it is not the real-data sysID path. Real-world replay must instead infer the initial Newton state from observable quantities such as TCP pose/velocity, F/T bias-corrected wrench, apple pose, woody marker positions, grasp/weld direction, and known calibration transforms.

```python
from apple_pick_gym.envs import ApplePickReplayEnv

env = ApplePickReplayEnv(
    fix_to_apple=True,
    fix_to_apple_warmup_substeps=0,
    robot_facing_weld=False,
)
env.load_dataset("/tmp/sysid_dataset", episode_id="<uuid>")
obs, info = env.reset(seed=3)
while True:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if truncated or terminated:
        break
env.close()
```

Inject candidate parameters for CEM rollouts:

```python
from apple_pick_sim.fruiting_system import FruitingSystemParams

obs, info = env.reset(options={"params": candidate_theta})
```

## Tests

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_trajectory_store.py \
  apple_pick_gym/tests/test_replay_env.py -q
```

Replay tests require `uv sync --extra vic` (PyTorch for VIC joint torques).

## Code map

| Module | Role |
|--------|------|
| `apple_pick_sim/system_id/episode_meta.py` | `EpisodeMeta` dataclass |
| `apple_pick_sim/system_id/trajectory_store.py` | `TrajectoryWriter`, `TrajectoryDataset` |
| `apple_pick_gym/envs/apple_pick_replay_env.py` | Dataset-backed open-loop replay |
