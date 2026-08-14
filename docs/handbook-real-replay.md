# Real replay pipeline handbook

This is the canonical living reference for converting a compiled real-robot
episode into the batched replay format and driving the reconstructed twin.
Sequencing, ranking acceptance, and CMA status belong in `docs/ROADMAP.md`.

## Document status

| Field | Value |
| ----- | ----- |
| Last reviewed | 2026-08-14 |
| Code owners | `robot_replay/`; `apple_pick_sim/system_id/real_to_batched_sysid.py`; `apple_pick_sim/system_id/batched_digital_twin_init.py`; `apple_pick_gym/batched_envs/real_batched_replay_build.py` |
| Status | Living handbook — defer sequencing to `docs/ROADMAP.md` |
| Related handbooks | H1 `docs/handbook-coupled-simulation.md`; H2 `docs/handbook-variable-impedance.md`; H3 `docs/handbook-sysid-scoring.md`; H5 `docs/handbook-youngs-cma.md` |
| Archive specs | **Partial:** `docs/superpowers/specs/2026-08-07-real-to-batched-metadata-parity-design.md`; **Implemented:** `docs/superpowers/specs/2026-08-10-real-batched-gl-replay-design.md`, `2026-08-11-batched-real-replay-post-grasp-se3-design.md`, `2026-08-12-real-camera-gl-viewer-design.md`, `2026-08-12-gl-video-record-design.md`; **Partial:** `2026-08-12-real-replay-cmaes-plumbing-design.md` |

Related boundaries:

- H1 owns the coupled rebuild, settle, and weld mechanics. Its planned
  canonical path is linked above even if that handbook has not landed yet.
- H2 owns `vic_pose` action and controller semantics.
- H3 owns the `batched_sysid_v1` bag and score-vector contract.
- H5 owns real-data grid/ranking/CMA use of the shared builder.

## 1. End-to-end flow

```text
compiled real parquet
  → convert metadata and trajectory
  → 1×1 batched_sysid_v1 dataset
  → rebuild pre-grasp digital twin
  → free settle
  → logged post-grasp apple/TCP weld
  → optional welded settle
  → replay converted 19D vic_pose actions
  → grid/ranking/CMA (H5 and ROADMAP)
```

The shipped drive path is complete: converted pose-controller logs are driven
through `vic_pose`. In particular, a logged 6D pose-control wrench is not an EE
twist. `real_to_batched_sysid.export_real_episode_to_batched_dataset` replaces
that source column with a 19D `vic_pose_v1` action, and
`real_batched_replay_build.check_action_semantics` rejects an incompatible
`vic` replay unless a legacy smoke-only escape hatch is explicitly requested.

**Status boundary:** Drive = `vic_pose` **Done**. Trusted Cartesian ranking and
real-data CMA remain open; see H5 and `docs/ROADMAP.md`. A successful replay or
environment build does not establish a trustworthy ranking.

## 2. Episode inputs

Current local bench data lives under:

```text
robot_replay/new_data/
  <session>/
    <episode>.parquet
    <episode>_robot.parquet
    <episode>_tracking.parquet
```

The compiled `<episode>.parquet` is the converter input. Robot and tracking
files are source logs used by the upstream compiler; replay does not combine
them at runtime. A usable compiled episode provides:

- `pre_grasp_geometry` and `post_grasp_geometry` in Parquet schema metadata;
- per-frame robot joints, TCP/tag poses, target pose, velocity, F/T, phase, and
  hold data;
- `dump.controller_gains` and pose-control action semantics; and
- a stable episode id and control rate, with converter fallbacks for older
  compiler layouts.

The converter writes one real episode as one structure and one direction:

```text
<dataset>/
  manifest.json
  episodes/
    s00_d00.parquet
```

`manifest.json` carries collection provenance and action declarations. Episode
schema metadata carries twin geometry, initial robot/apple/TCP poses, weld
fields, controller metadata, and optional camera calibration. See H3 for the
general bag schema and the runtime-obs/bag/score-vector distinction.

## 3. Convert contract

The CLI `robot_replay/convert_real_to_batched_sysid_metadata.py` is a thin
front end to
`real_to_batched_sysid.export_real_episode_to_batched_dataset`.

### Action packing

For a pose-controller source, each output action is:

```text
[target_pos(3), target_quat_wxyz(4), Kp(6), Kd(6)]
```

The target comes from per-frame `target_pose_4x4`; gains come from
`dump.controller_gains`. Conversion stamps:

```text
action_dim = 19
action_layout = vic_pose_v1
action_compatible_with_vic_twist = false
```

Fresh conversions need no `pack_vic_pose_actions.py` pass. That script is only
for legacy datasets or deliberately replacing gains with `--force`.

### F/T, woody points, and holds

Conversion aligns real bags with H3:

- `world_wrench_from_ee_logged` rotates force and torque with
  \(R_{W,TCP}\): \(F_W=R_{W,TCP}F_{EE}\) and
  \(\tau_W=R_{W,TCP}\tau_{EE}\). There is no second sign flip, lever-arm
  transport, or simulated EMA/LPF.
- `tag_poses_to_cma_woody` reads Branch, Spur, and Apple pose translations.
  It emits the two woody starts `primary_spur` and `spur_stem`, plus
  `apple_pos`. Trajectory bags do not carry woody ends.
- `_scalar_hold_number` prefers scalar `hold_index`; one-hot expansion occurs
  only in H3 score construction.
- `action` remains required to replay the episode but is excluded from
  `STATE_VECTOR_FIELDS` and Sinkhorn features.

Missing target/TCP/tag poses required by these contracts raise during convert
instead of silently producing a differently framed bag.

## 4. Pre-grasp rebuild, settle, and post-grasp weld

The two geometry blocks have different jobs:

1. **Pre-grasp** is the non-bending construction reference. The native mapping
   prefers `pre_grasp_geometry.rest_snapshot_during_run`, falls back to legacy
   `snapshot`, derives `fruiting_base_pos`, and rebuilds the plant. The apple's
   pre-grasp orientation seeds the free settle.
2. **Post-grasp** is the measured grasped state. After free settle, replay
   places the apple at its logged post-grasp SE(3) and places the proxy from the
   logged apple-to-TCP transform. It must not be used to rebuild rods from bent
   post-grasp chords.

`batched_digital_twin_init.gripper_proxy_for_real_batched_replay` computes
\(X_{\text{apple}}^{-1}X_{\text{TCP}}\) from the converted initial apple/TCP
poses. `apply_logged_post_grasp_se3_to_cable` then writes the logged apple pose,
realigns the proxy, zeros their twists, synchronizes both cable states, aligns
VBD history, and updates rest state. It is called after the normal
settle→weld seed so the free settle still starts from pre-grasp geometry.

`example_replay_real_batched.py` uses free-settle defaults of 5000 VBD
substeps with twists quieted every 300 substeps, followed by 500 welded
post-grasp settle substeps. Shorter values are useful for CI smoke tests, but
they are not equivalent settling evidence.

## 5. Digital-twin and arm initialization

`real_batched_replay_build.real_replay_sim_config` builds the FR3 coupled
configuration from:

- converted `fruiting_system_params` and episode `fruiting_base_pos`;
- fixture `sim_build` VIC/support-joint settings;
- recorded `initial_robot_joint_q`, applied open-loop with IK skipped;
- episode/collection `control_hz`; and
- the real replay gripper/weld fields.

The reconstructed episode geometry and the H3 bag are separate layers.
Runtime/digital-twin observations may need woody ends to infer rods; real
trajectory bags persist only the two starts and apple position used by H3.
See `docs/digital-twin.md` for observation-only reconstruction and fixture
catalog details.

## 6. `vic_pose` replay

`robot_replay/example_replay_real_batched.py` loads the converted dataset,
constructs the shared real builder, and calls
`batched_sysid_mmd_grid.replay_batched_sysid_structure`. Its default is
`--controller-mode vic_pose`, so each 19D row directly commands absolute target
pose and anisotropic gains as defined by H2.

Legacy `--controller-mode vic` is valid only for a true 6D twist-compatible
dataset. `--allow-wrench-as-twist` exists for old format/GL smoke and represents
incorrect physics; it is rejected for an already packed `vic_pose_v1` dataset.

The GL viewer renders trajectory frames after off-screen settle/weld. The CLI
reports TCP motion and fails if the trajectory is effectively stationary.

### Camera and MP4

`real_to_batched_sysid.camera_to_base_4x4_from_dataset_metadata` copies the
calibration into episode metadata. On the first GL frame,
`example_replay_real_batched.gl_camera_from_camera_to_base` places the eye at
its translation and uses the OpenCV camera +Z axis for pitch/yaw; Newton GL
does not preserve camera roll.

`--record-video PATH` requires `--viewer gl` (`--headless` is allowed).
`gl_video_recorder.GlVideoRecorder` captures `viewer.get_frame()` after each
rendered control frame, uses simulation `control_hz` as FPS, and writes H.264
MP4 through `imageio-ffmpeg` from the `gym` extra.

## 7. Shared gym builder

`apple_pick_gym/batched_envs/real_batched_replay_build.py` keeps standalone
replay and optimization callers on the same initialization path:

- `dataset_declares_vic_pose` detects 19D metadata;
- metadata helpers load base position, open-loop joints, and control rate;
- `real_replay_sim_config` selects FR3 coupled stepping and the requested
  action dimension; and
- `make_real_replay_build_env_fn` creates `ApplePickBatchedSysIdEnv`, disables
  the settle cache, supplies per-environment candidate params/grippers, and
  applies the logged post-grasp SE(3) with the batched layout.

The Young's grid can opt into this builder from real dataset metadata. That
plumbing being present is not the same as accepting its ranking. Trusted
ranking and CMA execution remain H5/ROADMAP work.

## 8. CLI cheat-sheet

Copy-paste commands and folder examples live in
`robot_replay/README.md`. Keep that file short; contracts and rationale belong
here.

## 9. Code map

| Responsibility | Module / symbol |
| -------------- | --------------- |
| Native real metadata mapping and batched export | `apple_pick_sim/system_id/real_to_batched_sysid.py` — `build_episode_metadata_from_real`, `export_real_episode_to_batched_dataset` |
| F/T, woody, hold, camera conversion | same module — `world_wrench_from_ee_logged`, `tag_poses_to_cma_woody`, `_scalar_hold_number`, `camera_to_base_4x4_from_dataset_metadata` |
| Twin init and logged weld pose | `apple_pick_sim/system_id/batched_digital_twin_init.py` — `gripper_proxy_for_real_batched_replay`, `apply_logged_post_grasp_se3_to_cable` |
| Shared gym build path | `apple_pick_gym/batched_envs/real_batched_replay_build.py` — `real_replay_sim_config`, `make_real_replay_build_env_fn` |
| Standalone replay and GL camera | `robot_replay/example_replay_real_batched.py` — `_run`, `make_replay_on_step`, `gl_camera_from_camera_to_base` |
| MP4 writer | `robot_replay/gl_video_recorder.py` — `GlVideoRecorder` |

## 10. Tests and verification

Key regression coverage:

- `apple_pick_sim/tests/test_real_to_batched_sysid.py` — metadata parity,
  camera propagation, 19D packing, F/T rotation, two woody starts, scalar
  holds, and no trajectory woody ends.
- `apple_pick_sim/tests/test_batched_digital_twin_init.py` — twin initialization
  and post-grasp SE(3).
- `apple_pick_gym/tests/test_real_batched_replay_build.py` — shared builder,
  per-env grippers, and batched logged-pose application.
- `apple_pick_gym/tests/test_real_batched_replay_cli.py` — action semantics,
  camera placement, replay callbacks, and video flag parsing.
- `robot_replay/tests/test_gl_video_recorder.py` and
  `test_pack_vic_pose_actions.py` — MP4 writer and legacy packer.

Focused checks from the repository root:

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_real_to_batched_sysid.py \
  apple_pick_gym/tests/test_real_batched_replay_build.py \
  apple_pick_gym/tests/test_real_batched_replay_cli.py \
  robot_replay/tests/test_gl_video_recorder.py \
  robot_replay/tests/test_pack_vic_pose_actions.py \
  -q -p no:launch_testing
```

Parquet-local replay smoke:

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input robot_replay/new_data/s09/s09-d00.parquet \
  --dataset-out /tmp/real_batched_s09_d00 --overwrite
uv run python robot_replay/example_replay_real_batched.py \
  --dataset /tmp/real_batched_s09_d00 --viewer null --max-frames 24 \
  --settle-substeps 80 --post-grasp-settle-substeps 0
```

The smoke proves conversion/build/19D drive, not ranking quality. Use the
current `docs/ROADMAP.md` validation block for the next real grid/CMA gate.
