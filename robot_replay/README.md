# `robot_replay/` CLI and folder cheat-sheet

Real replay contracts and rationale live in
[`docs/handbook-real-replay.md`](../docs/handbook-real-replay.md). This file
only lists local data layout and copy-paste commands. Grid/ranking/CMA status
belongs to H5 and [`docs/ROADMAP.md`](../docs/ROADMAP.md).

## Folder layout

```text
robot_replay/
  new_data/<session>/
    <episode>.parquet           # compiled converter input
    <episode>_robot.parquet     # upstream robot source
    <episode>_tracking.parquet  # upstream tracking source
  convert_real_to_batched_sysid_metadata.py
  example_view_pre_grasp_settle.py
  example_view_batched_episode_meta.py
  example_replay_real_batched.py
  pack_vic_pose_actions.py      # legacy datasets only
```

Fresh conversion writes:

```text
<dataset>/
  manifest.json
  episodes/s00_d00.parquet
```

Run commands from the repository root.

## Inspect native pre-grasp settle/weld

```bash
uv run python robot_replay/example_view_pre_grasp_settle.py \
  --parquet robot_replay/new_data/s09/s09-d00.parquet \
  --grasp-after-settle \
  --settle-substeps 80 \
  --post-grasp-settle-substeps 40 \
  --viewer null --num-frames 8
```

For interactive inspection, use `--viewer gl` and increase settle counts.

## Convert real parquet

Metadata JSON only:

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input robot_replay/new_data/s09/s09-d00.parquet \
  --fixture apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json \
  --out /tmp/s09_d00_episode_meta.json
```

Full 1×1 `batched_sysid_v1` dataset:

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input robot_replay/new_data/s09/s09-d00.parquet \
  --dataset-out /tmp/real_batched_s09_d00 \
  --overwrite
```

Fresh pose-controller logs are converted directly to 19D `vic_pose_v1`.
Do not run `pack_vic_pose_actions.py` on a fresh conversion.

## Replay

Short headless smoke:

```bash
uv run python robot_replay/example_replay_real_batched.py \
  --dataset /tmp/real_batched_s09_d00 \
  --viewer null --max-frames 24 \
  --settle-substeps 80 \
  --post-grasp-settle-substeps 0
```

Full GL replay:

```bash
uv run python robot_replay/example_replay_real_batched.py \
  --dataset /tmp/real_batched_s09_d00 \
  --viewer gl --max-frames 0 \
  --settle-substeps 5000 \
  --settle-quiet-every 300 \
  --post-grasp-settle-substeps 500
```

`vic_pose` is the default and shipped real drive. Use `--controller-mode vic`
only for a genuine legacy 6D twist dataset. A real pose-control wrench is not
an EE twist.

If converted metadata contains `camera_to_base_4x4`, GL replay places the
viewer at the recording camera translation and +Z look direction.

## Record MP4

```bash
uv run python robot_replay/example_replay_real_batched.py \
  --dataset /tmp/real_batched_s09_d00 \
  --viewer gl --headless --max-frames 0 \
  --record-video /tmp/real_batched_s09_d00.mp4
```

`--record-video` requires a GL viewer; `--headless` is supported. FPS follows
the episode control rate. The writer uses `imageio-ffmpeg` from the `gym`
extra (`uv sync --extra gym`).

## Legacy action repair

Only for a pre-`vic_pose` dataset:

```bash
uv run python robot_replay/pack_vic_pose_actions.py \
  --dataset-in /tmp/old_6d_batched \
  --dataset-out /tmp/old_6d_batched_vic_pose \
  --kp 800 800 800 40 40 40 \
  --kd 80 80 80 4 4 4 \
  --overwrite
```

`fill_actions_from_tcp_velocity.py` is only for older zero-action files. It
does not reproduce the real pose-PD drive.

## Focused verification

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_real_to_batched_sysid.py \
  apple_pick_gym/tests/test_real_batched_replay_build.py \
  apple_pick_gym/tests/test_real_batched_replay_cli.py \
  robot_replay/tests/test_gl_video_recorder.py \
  robot_replay/tests/test_pack_vic_pose_actions.py \
  -q -p no:launch_testing
```

Drive = `vic_pose` **Done**. Trusted real-data ranking and CMA remain open;
follow H5 and `docs/ROADMAP.md`, not successful replay alone.
