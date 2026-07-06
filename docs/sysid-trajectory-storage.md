# SysID trajectory storage

Parquet persistence for quasi-static (and future) sysID rollouts, plus dataset-backed replay for CEM/MMD parameter tuning.

## Status

Recording and replaying are complete for sim-to-sim datasets, **and observation-only replay initialization is the shipped default** (M3.0.3 — see `docs/ROADMAP.md`). Collection writes per-frame observations/actions, reset observable state in episode metadata, and an optional privileged initial-state snapshot. Replay loads the dataset and, by default, rebuilds its initial state from recorded observations and calibration metadata rather than privileged simulator arrays (body poses, velocities, solver previous-state buffers, saved controller internals). It applies the recorded EE velocity actions open-loop while recomputing observations from the live simulation. Pass `--use-snapshot` only to opt into the privileged `.npz` path for sim-to-sim debugging. Observation and digital-twin requirements are specified in `docs/digital-twin.md`.

Digital-twin fixture-catalog reconstruction (M3.0.4) is a separate, still-partially-blocked slice — see `docs/digital-twin.md` ("Known gap").

**Batched parallel collection** (`example_batched_collect_sysid_data.py`) uses a separate **batched_sysid_v1** layout (`manifest.json` + `episodes/s{s}_d{d}.parquet`). See [`batched-sysid-dataset.md`](batched-sysid-dataset.md).

## Layout (single-env / legacy)

```
<output_dir>/
  metadata.parquet              # one row per episode (appended across runs)
  frames/
    <episode_id>.parquet        # per-frame timeseries for one episode
  initial_states/
    <episode_id>.npz            # optional privileged baseline (--save-snapshot)
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
| `ft_wrist` | list[f32]×6 | Applied plant F/T feedback at TCP `[F, τ]`; sys-ID defaults cap force/torque norms at 100 N / 100 N·m before applying them to the robot |

Per-junction woody columns are dynamic (one start/end pair per entry in `junction_names` metadata). There are no flat `woody_part_start_pos` / `woody_part_end_pos` frame columns.

## Bonus frame columns (written by default)

`sim_time`, `dir_idx`, `amplitude_m`, `raw_ft_wrist` (uncapped stem-harvest TCP wrench; legacy loaders fall back to `ft_wrist`), `tcp_pos`, `tcp_quat`, `apple_pos`,
`apple_quat`, `robot_joint_q`, `woody_part_force`

## Required metadata columns

| Column | Description |
|--------|-------------|
| `episode_id` | Primary key |
| `weld_direction` | Gripper weld orientation at reset |
| `excitation_type` | `"quasi_static"` \| `"translational_chirp"` \| `"torsional"` |
| `n_woody_parts` | Number of woody fixed joints |
| `junction_names` | Ordered junction labels for woody parquet columns |
| `params_fingerprint` | JSON material/geometry summary of `FruitingSystemParams` θ (includes derived stiffness and, from schema v2, \(E\)/\(\zeta\); see `docs/material-parameter-sampling.md`) |
| `fruiting_system_params` | Lossless JSON for the sampled `FruitingSystemParams` used to build the episode; nullable for legacy or real-data rows |
| `control_hz` | Env step rate |

## Bonus metadata columns

`timestamp`, `seed`, `n_directions`, reset observable state (`initial_tcp_pos`, `initial_tcp_quat`, `initial_apple_pos`, `initial_apple_quat`, `initial_robot_joint_q`), `fixture_path`, trajectory config (`movement_per_step_m`, `total_movement_m`, `hold_duration_s`, `move_speed_mps`, `skip_return`)

## Observation-only initialization metadata

For M3.0.3, datasets must be usable when `initial_states/<episode_id>.npz` is absent. The required real-world equivalent is:

| Item | Storage location today | Required evolution |
|------|------------------------|--------------------|
| TCP pose/twist at reset | metadata `initial_tcp_pos`, `initial_tcp_quat`; per-step `tcp_velocity`; optional snapshot `obs_tcp_pos`, `obs_tcp_velocity` | Keep reset-frame TCP position/orientation/velocity available even when no privileged snapshot is written |
| F/T bias-corrected wrench | first-frame `ft_wrist`; optional snapshot `obs_ft_wrist` | Record bias metadata or bias-corrected convention in episode metadata |
| Apple pose | metadata `initial_apple_pos`, `initial_apple_quat`; optional snapshot `obs_apple_pos` | Keep reset-frame apple position/orientation in the v3 pose bundle |
| Robot joint positions | metadata `initial_robot_joint_q` | Restore the dynamic robot state and MuJoCo/VIC buffers from observed joint positions |
| Woody endpoints | per-frame `woody_start__<junction>`, `woody_end__<junction>`; optional snapshot `obs_woody_start`, `obs_woody_end` | Keep `junction_names` stable and map each key to the fixture topology |
| Sampled fruiting parameters | metadata `fruiting_system_params`; summary in `params_fingerprint` | Rebuild the same sim-to-sim θ without depending on the original procedural seed; for real data this becomes calibrated or candidate θ |
| Grasp/weld transform | `weld_direction`, metadata `weld_reference_pos`, `weld_reference_quat`; optional snapshot copies | Store real grasp transform/calibration rather than deriving only from sim body state |
| Calibration transforms | not represented directly | Add fixture/world/robot/camera/F/T transforms when field data collection starts |
| Digital-twin fixture identity | `fixture_path` | Point to a named fixture catalog entry with topology, base poses, and geometry ranges |

Per-step frame 0 remains the observation after replay action 0 has been applied, so replay can compare action 0 against recorded frame 0 after one `env.step(action_0)`. Observation-only reset initialization prefers the metadata reset fields above and falls back to frame-0 `robot_joint_q`, `tcp_pos`, and `tcp_quat` only for legacy datasets that predate reset metadata.

`fruiting_system_params` is not a privileged dynamic state snapshot: it stores the episode's realized physical parameters (geometry, sampled \(E\)/\(\zeta\) and derived stiffness/damping, density, segment counts, apple scalars), not Newton body transforms, velocities, solver buffers, or controller internals. No-snapshot replay prefers this exact θ when present, then falls back to observation-derived geometry plus fixture midpoint dynamics for legacy datasets. The `.npz` snapshot may continue to be written for privileged replay baselines, but it must be optional. Replay code should treat the absence of `.npz` as the normal real-data path, not as a corrupted dataset.

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

The command above does **not** write `initial_states/*.npz`. Add
`--save-snapshot` only when collecting a privileged sim-to-sim baseline for
comparison while observation-only replay is being validated:

```bash
uv run python apple_pick_gym/examples/example_gym_sysid.py \
  --viewer null --n-directions 1 --max-steps 200 --save-snapshot \
  --output /tmp/sysid_dataset_with_snapshot
```

Implementation: [`apple_pick_sim/system_id/trajectory_store.py`](../apple_pick_sim/system_id/trajectory_store.py), wired in [`apple_pick_gym/examples/example_gym_sysid.py`](../apple_pick_gym/examples/example_gym_sysid.py).

## Replay example script

[`apple_pick_gym/examples/example_gym_replay.py`](../apple_pick_gym/examples/example_gym_replay.py) replays a collected dataset headless or with the Newton viewer:

```bash
uv run python apple_pick_gym/examples/example_gym_replay.py \
  --dataset /tmp/sysid_dataset --viewer null

uv run python apple_pick_gym/examples/example_gym_replay.py \
  --dataset /tmp/sysid_dataset --list-episodes
```

## Diagnostic stiffness grid

[`apple_pick_gym/examples/run_system_identification.py`](../apple_pick_gym/examples/run_system_identification.py)
uses the same dataset format and replay env, but loops over candidate
`primary`, `secondary`, `spur`, and `stem` `bend_stiffness` values. It
recreates the replay digital twin from observable Parquet metadata by default,
drives the sim with recorded EE velocity actions, and prints replay loss
summaries for each candidate in grid-search order. Add `--mmd-output <dir>` to
rank candidates by hold-phase biased MMD² and write `mmd_results.csv` plus
`mmd_ranked_loss.png`.

List episodes:

```bash
uv run python apple_pick_gym/examples/run_system_identification.py \
  --dataset /tmp/sysid_dataset --viewer null --list-episodes
```

Run one candidate as a smoke check:

```bash
uv run python apple_pick_gym/examples/run_system_identification.py \
  --dataset /tmp/sysid_dataset --viewer null \
  --primary-bend-stiffness-values 10 \
  --secondary-bend-stiffness-values 10 \
  --spur-bend-stiffness-values 10 \
  --stem-bend-stiffness-values 10 \
  --max-candidates 1
```

Run a larger grid:

```bash
uv run python apple_pick_gym/examples/run_system_identification.py \
  --dataset /tmp/sysid_dataset --viewer null \
  --primary-bend-stiffness-values 10,25,50 \
  --secondary-bend-stiffness-values 10,25,50 \
  --spur-bend-stiffness-values 10,25,50 \
  --stem-bend-stiffness-values 10,25,50 \
  --mmd-output /tmp/apple_pick_mmd_grid
```

Use `--episode-id <uuid>` to restrict evaluation to one recorded episode, or
repeat `--episode-id` to evaluate a subset. `--use-snapshot` is available for
privileged sim-to-sim debugging only; leave it off for observation-only replay.

## Dataset dashboard

[`apple_pick_gym/examples/dashboard_sysid_dataset.py`](../apple_pick_gym/examples/dashboard_sysid_dataset.py)
opens a local Plotly Dash app for raw dataset sanity checks before replay or
MMD tuning. It reads the same Parquet dataset directory and plots 3D TCP, apple,
and woody endpoint trajectories, linked motion/wrench time series, and
per-hold force/displacement summaries.

```bash
uv run python apple_pick_gym/examples/dashboard_sysid_dataset.py \
  --dataset /tmp/sysid_dataset
```

## Replay env

`ApplePickReplay-v0` (`ApplePickReplayEnv`) loads a dataset and applies stored `action` rows open-loop. The Gym `action` argument is ignored during replay.

Current replay is state-initialized when `initial_states/<episode_id>.npz` exists and snapshot loading is enabled. That privileged snapshot remains useful for proving storage, action replay, and live-vs-recorded observation comparison, but it is not the default collection path and it is not the real-data sysID path. Real-world replay must instead infer the initial Newton state from observable v3 quantities such as TCP position/orientation/velocity, robot joint positions, F/T bias-corrected wrench, apple position/orientation, woody marker positions, grasp/weld direction, digital-twin fixture geometry, and known calibration transforms.

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
