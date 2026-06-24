# Sys-ID Dataset Dashboard Design

## Purpose

Build a local Plotly Dash dashboard for sanity-checking sys-ID datasets collected
by `apple_pick_gym/examples/example_gym_sysid.py`. The first version reads the
existing Parquet dataset layout directly and focuses on raw observations before
MMD, CEM, or replay comparison diagnostics.

## Data Flow

The dashboard takes a dataset directory containing `metadata.parquet` and
`frames/<episode_id>.parquet`. It uses
`apple_pick_sim.system_id.TrajectoryDataset` to load one episode at a time as
the same array contract used by replay and MMD helpers.

```text
dataset dir -> TrajectoryDataset -> episode arrays -> dashboard data helpers -> Dash figures
```

The initial scope does not convert data to rosbags or Foxglove. A future
ROS/Foxglove reader or exporter can be added once the real-world data collection
contract is clearer.

## Dashboard Layout

The app exposes controls for dataset inspection:

- Episode selector from `TrajectoryDataset.episode_ids()`.
- Direction selector from `dir_idx`, including an "all directions" option.
- Phase filter for `move_out`, `hold`, and `return`.

The main panel shows an interactive 3D trajectory plot with:

- TCP position trajectory.
- Apple position trajectory.
- Woody endpoint trajectories for each recorded junction.
- Phase-colored markers so trajectory generation mistakes are visible.

Linked time-series panels show:

- `amplitude_m` when available.
- Linear action command and TCP linear velocity.
- `ft_wrist` force/torque.
- `raw_ft_wrist` when available and distinct from `ft_wrist`.

## Hold Summary

For each direction and amplitude represented by hold frames, the dashboard
computes a compact summary:

- Mean TCP force vector and force norm.
- Mean TCP displacement from the episode reset TCP position when metadata
  provides `initial_tcp_pos`; otherwise displacement from the first frame.
- A stiffness-like diagnostic `|F| / |dx|`, reported only when displacement is
  nonzero.

This summary is a sanity check, not a calibrated stiffness estimate.

## Error Handling

The command should fail early with clear messages when the dataset directory is
missing, `metadata.parquet` is missing, or an episode has no frame data. Optional
columns should degrade gracefully: missing `amplitude_m` or `raw_ft_wrist`
removes only the corresponding plot traces.

## Testing

Implementation should keep Dash callbacks thin and put data preparation in
testable helpers. Tests should use small synthetic arrays shaped like
`TrajectoryDataset.load_episode_obs_arrays()` output to cover:

- Phase and direction filtering.
- Woody endpoint flattening for 3D plotting.
- Hold summary displacement and force calculations.
- CLI parser defaults without starting a live Dash server.

## Verification

Targeted checks:

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_sysid_dashboard_data.py \
  apple_pick_gym/tests/test_dashboard_sysid_dataset.py -q

uv run python apple_pick_gym/examples/dashboard_sysid_dataset.py --help
```
