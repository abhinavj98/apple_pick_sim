# Real recording camera → GL viewer pose

| Field | Value |
| ----- | ----- |
| **Status** | Implemented |
| **Date** | 2026-08-12 |
| **Code** | `real_to_batched_sysid.build_episode_metadata_from_real`, `robot_replay/example_replay_real_batched.py` |

## Purpose

Place the Newton GL viewer camera at the real recording-camera pose relative to
`franka_base_o` during `example_replay_real_batched.py --viewer gl`.

## Contract

### Convert

`build_episode_metadata_from_real` copies into episode metadata:

- **Key:** `camera_to_base_4x4` — nested 4×4 `list[list[float]]` (camera → base)
- **Source (preferred):** `dataset_metadata.camera_to_base_4x4_used`
- **Fallback:** first found `pre_grasp_geometry.*.camera_to_base_4x4` (settled / snapshot / …)
- **Missing:** omit key; convert still succeeds

Written into the batched episode parquet via existing `traj.save(..., episode_meta)`.

### Replay

On first GL viewer init in `make_replay_on_step`:

1. Read `camera_to_base_4x4` from episode metadata passed into the callback.
2. `pos` = translation; look = camera **+Z** column (OpenCV optical axis).
3. Derive Newton Z-up `pitch` / `yaw` (same formulas as `Camera._set_orientation_from_direction`).
4. `viewer.set_camera(pos, pitch, yaw)` once if the viewer supports it.
5. Missing / malformed → keep default camera.

**Limit:** Newton GL has no roll; match position + look axis only (world-Z up).

## Non-goals

- Manifest/collection mirroring
- CLI overrides for camera pose
- Matching intrinsic FOV
- Reading live from `source_real_parquet` at replay time
