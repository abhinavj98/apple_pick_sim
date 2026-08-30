# Real replay pipeline handbook

This is the canonical living reference for converting a compiled real-robot
episode into the batched replay format and driving the reconstructed twin.
Sequencing, ranking acceptance, and CMA status belong in `docs/ROADMAP.md`.

## Document status

| Field | Value |
| ----- | ----- |
| Last reviewed | 2026-08-27 |
| Code owners | `robot_replay/`; `apple_pick_sim/system_id/real_to_batched_sysid.py`; `apple_pick_sim/system_id/real_pre_grasp_params.py`; `apple_pick_sim/system_id/batched_digital_twin_init.py`; `apple_pick_gym/batched_envs/real_batched_replay_build.py` |
| Status | Living handbook — defer sequencing to `docs/ROADMAP.md` |
| Related handbooks | H1 `docs/handbook-coupled-simulation.md`; H2 `docs/handbook-variable-impedance.md`; H3 `docs/handbook-sysid-scoring.md`; H5 `docs/handbook-youngs-cma.md` |
| Archive specs | **Partial:** `docs/superpowers/specs/2026-08-07-real-to-batched-metadata-parity-design.md`; **Implemented:** `docs/superpowers/specs/2026-08-10-real-batched-gl-replay-design.md`, `2026-08-11-batched-real-replay-post-grasp-se3-design.md`, `2026-08-12-real-camera-gl-viewer-design.md`, `2026-08-12-gl-video-record-design.md`, `2026-08-14-real-rod-mass-density-override-design.md`; **Partial:** `2026-08-12-real-replay-cmaes-plumbing-design.md`; **Implemented (convert/replay; Task 9 science gate failed on torque):** `2026-08-17-one-structure-multidir-holdout-cmaes-design.md` |

> **Warning — tare real F/T, never simulated F/T.** Convert the compiled
> episode parquet whose `ft_wrist` is already loaded EMA minus unloaded EMA.
> Convert then rotates to world, block-means unfiltered F/T and TCP velocity
> to `--control-hz` (default 30 Hz), and writes a separate `ft_wrist_lpf`
> column (10 Hz zero-phase Butterworth, then the same block-mean). Scoring
> uses `ft_wrist_lpf`. Commands, poses, phase, hold, joints, and woody geometry
> take the last sample of each window. Do not subtract a robot-only sim replay
> from candidate `ft_wrist`: on `vic_pose` that signal is plant harvest only,
> so the unload would be zero. Full contract: H3
> `docs/handbook-sysid-scoring.md` (warning at top).

Related boundaries:

- H1 owns the coupled rebuild, settle, and weld mechanics. Its planned
  canonical path is linked above even if that handbook has not landed yet.
- H2 owns `vic_pose` action and controller semantics.
- H3 owns the `batched_sysid_v1` bag and score-vector contract.
- H5 owns real-data grid/ranking/CMA use of the shared builder.

## 1. End-to-end flow

```text
compiled real parquet(s)
  → convert --input (1×1) or --input-dir (1×N)
  → batched_sysid_v1 dataset (episodes/s00_dNN)
  → rebuild pre-grasp digital twin
  → free settle
  → per-direction logged post-grasp apple/TCP weld, gripper, arm joints
  → optional welded settle
  → replay converted 19D vic_pose actions (pad last action; truncate before features)
  → grid/ranking/CMA (H5 and ROADMAP)
```

The shipped drive path is complete: converted pose-controller logs are driven
through `vic_pose`. In particular, a logged 6D pose-control wrench is not an EE
twist. `real_to_batched_sysid.export_real_episode_to_batched_dataset` replaces
that source column with a 19D `vic_pose_v1` action, and
`real_batched_replay_build.check_action_semantics` rejects an incompatible
`vic` replay unless a legacy smoke-only escape hatch is explicitly requested.

**Status boundary:** Drive = `vic_pose` **Done**. Folder convert and
per-direction replay for one tree are **Done**. Opt-in 5/3 holdout CMA is
wired (H5); Task 9 GPU science gate **ran and failed** on val torque
magnitude (ROADMAP) and is **not** claimed as a pass here. A successful replay or environment build does not establish a
trustworthy ranking.

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

Single-file `--input` still writes one episode as one structure and one
direction (`episodes/s00_d00.parquet`). Folder `--input-dir` writes one
structure and one episode per compiled pull:

```text
<dataset>/
  manifest.json
  episodes/
    s00_d00.parquet
    s00_d01.parquet
    ...
```

`manifest.json` carries collection provenance and action declarations. Episode
schema metadata carries twin geometry, initial robot/apple/TCP poses, weld
fields, controller metadata, and optional camera calibration. See H3 for the
general bag schema and the runtime-obs/bag/score-vector distinction.

## 3. Convert contract

The CLI `robot_replay/convert_real_to_batched_sysid_metadata.py` is a thin
front end. `--input` and `--input-dir` are a required mutually exclusive
group:

- `--input` → `export_real_episode_to_batched_dataset` (1×1; unchanged);
- `--input-dir` → `export_real_tree_folder_to_batched_dataset` (1×N).

Passing both is an argparse error. `--out` JSON metadata remains single-file
only.

### Folder convert (one tree, N directions)

`--input-dir` collects compiled files matching `sXX-dNN.parquet`. It ignores
`*_robot` / `*_tracking` siblings, PNGs, and videos. The integer `NN` from
the filename is `direction_idx`. If parquet `dump.direction_index` is
present, it **must** equal `NN` or convert fails. Duplicate `dNN`, mixed tree
prefixes, or an empty folder fail loudly.

Canonical geometry (one tree):

- Every direction must share rod geometry (`junction_names` and rebuilt
  `fruiting_system_params` lengths/radii/segment counts).
- `fruiting_base_pos` is the mean across directions. Convert fails if any
  axis spread exceeds `--base-pos-tolerance-m` (default 5 mm).
- After the assert, every episode gets the same mean base pose and the same
  canonical `fruiting_system_params` / `params_fingerprint` (copied from the
  lowest `direction_idx`). `true_params_for_structure` still reads direction
  0.

Manifest for the folder path:

- `collection.num_structures = 1`, `structure_idx = 0`;
- `collection.num_directions = max(NN)+1` (sparse `d03`+`d05` ⇒ width 6 with
  two episode rows);
- `env_idx = direction_idx`;
- `collection.n_holds = max(hold_number)+1` (4 for the s09 logs);
- `collection.control_hz` from `--control-hz` (default 30);
- `collection.sim_config` via `sim_config_to_manifest_dict` built in
  `apple_pick_sim` (`controller.mode = "vic_pose"`; no gym import);
- `collection.topology_seed` present;
- `collection.source_real_parquets` is the list of compiled inputs (the 1×1
  path keeps singular `source_real_parquet`).

Holdout CMA still requires eight usable disk dirs; sparse convert is legal
for the bag. Scoring one-hot width is this collection width, not
`len(selected)` — see H3.

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
  \(\tau_W=R_{W,TCP}\tau_{EE}\). Rotation happens **before** filtering.
  **Current real collections** (`robot_replay/final_data_correct_torque/`, e.g.
  s09) already store compiled `ft_wrist` in the correct world frame at TCP:
  env-on-robot, force and torque about TCP (libfranka external wrench estimate
  after collection-time frame correction). Default convert applies rotation only
  when needed; do **not** use `--transport-torque-to-tcp` on those parquets.
  There is no second sign flip or simulated tare.
  The source `ft_wrist` must already be compiled EMA−EMA (loaded EMA minus
  unloaded EMA). The **unloaded replay is without the apple**, so apple weight
  remains in the tared signal. Convert then `filtfilt`s a Butterworth (default
  cutoff 10 Hz, order 4) on world `ft_wrist` only, writing the result as
  `ft_wrist_lpf`, and block-means unfiltered `ft_wrist`, `raw_ft_wrist`,
  `tcp_velocity`, and `ft_wrist_lpf` to `--control-hz` (default 30 Hz).
  Action / `vic_pose`, TCP and tag poses, `phase`, hold index, joint `q`,
  and woody starts copy the **last sample** of each window so labels and
  commands are not averaged. Sim harvest is not filtered. Provenance is
  `collection.ft_filter` (`column: ft_wrist_lpf`, `applied`) and episode
  `ft_filter`.
- `tag_poses_to_cma_woody` reads Branch, Spur, and Apple pose translations.
  It emits the two woody starts `primary_spur` and `spur_stem`, plus
  `apple_pos`. Trajectory bags do not carry woody ends.
- `_scalar_hold_number` prefers scalar `hold_index`; one-hot expansion occurs
  only in H3 score construction.
- `action` remains required to replay the episode but is excluded from
  `STATE_VECTOR_FIELDS` and Sinkhorn features.
- Rod `mass_kg` (today `parts.spur.mass_kg`) overrides that rod's
  `density_kg_m3` as \(\rho = m/(\pi r^{2} L)\). Catalog `radius_m` is
  unchanged so bending \(I \propto r^{4}\) stays geometric. Missing
  `mass_kg` keeps catalog density. `map_pre_grasp_geometry` owns this;
  `build_fruiting_params_from_real` still reads `rod_geometry` density.

Missing target/TCP/tag poses required by these contracts raise during convert
instead of silently producing a differently framed bag.

## 4. Pre-grasp rebuild, settle, and post-grasp weld

The two geometry blocks have different jobs:

1. **Pre-grasp** is the non-bending construction reference. The native mapping
   prefers `pre_grasp_geometry.rest_snapshot_during_run`, falls back to legacy
   `snapshot`, derives `fruiting_base_pos`, and rebuilds the plant. The apple's
   pre-grasp orientation seeds the free settle. When
   `parts.spur.manual_spur_angle_deg` and `parts.stem.manual_stem_angle_deg`
   are both set, rod directions come from those catalog connection angles
   (not woody marker chords). See [Checking connection angles](#checking-connection-angles).
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
`ApplePickBatchedBaseEnv` snapshots physics at the end of construct, before
that write. `make_real_replay_build_env_fn` recaptures after SE(3) so the
fused/scalar `reset()` restore keeps the grasped apple/proxy pose.

`example_replay_real_batched.py` uses free-settle defaults of 5000 VBD
substeps with twists quieted every 300 substeps, followed by 500 welded
post-grasp settle substeps. Shorter values are useful for CI smoke tests, but
they are not equivalent settling evidence.

### Checking connection angles

Catalog `manual_*_angle_deg` is the plant-rebuild ground truth when both keys
are present. Frames and Rodrigues steps:
`docs/connection-angles-implementation.md`. Do **not** treat woody marker
chords or `connection_rpy_deg` as the kit pose: chords are a rest snapshot
(often nearly vertical) and RPY is a compiler snapshot frame.

Proxy world: primary **+X**, robot reach **+Y**, hang **−Z**. Fruiting→robot is
**−Y**.

| Catalog field | Rotation axis | Rest / result |
|---------------|---------------|----------------|
| `manual_spur_angle_deg` | primary (proxy +X) | Rest = +Y (horizontal T). **90°** hangs to −Z. |
| `manual_stem_angle_deg` | fruiting→robot (proxy −Y) | Applied **after** the spur hang. **60°** leans the stem in the XZ plane to `(sin 60, 0, −cos 60)`. |

World **Z** is gravity. After a 90° hang the spur already lies on −Z, so a
rotation about world Z cannot produce a 60° stem. The old sequential elevation
path (`_deflect_direction`) did that and collapsed spur and stem to collinear
−Z.

**How to check a converted episode**

1. Print pre-grasp diagnostics (settle viewer already does):

   ```bash
   uv run python robot_replay/example_view_pre_grasp_settle.py \
     --parquet robot_replay/final_data_correct_torque/s09/s09-d00.parquet \
     --viewer null --num-frames 1
   ```

   Look for the `connection angles:` block:
   - `source=manual_catalog_angles`
   - `manual_spur_angle_deg` / `manual_stem_angle_deg` match the parquet parts
   - `built spur–stem angle` equals the stem catalog angle (90/60 → **60.0°**)
   - `chord spur–stem angle` may differ (s09 chords are ~5°); that is expected

2. In GL, the spur should hang down and the stem should lean **along the
   primary** (X), not toward the robot (Y).

3. Unit check:

   ```bash
   uv run --env-file pytest.env python -m pytest \
     apple_pick_sim/tests/test_real_pre_grasp_params.py \
     -k "catalog_angles or connection_angles or smoke" -q
   ```

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
  applies the logged post-grasp SE(3) with the batched layout. Env construct
  snapshots physics before that write; the builder recaptures afterward so
  the fused/scalar `reset()` restore keeps the grasped apple/proxy pose.

The Young's grid can opt into this builder from real dataset metadata. That
plumbing being present is not the same as accepting its ranking. Trusted
ranking remains H5/ROADMAP work (Task 9 science gate failed on val torque).

### Per-direction weld, gripper, and arm joints

A 1×N real batch must not reuse direction 0's grasp for every env. When
`ReplayStructureRequest.meta_by_direction` is set:

- each `ReplaySlot` stores that direction's episode metadata;
- `ReplaySlot.gripper` is `gripper_proxy_for_real_batched_replay` of **that**
  meta (the driver already prefers `per_env_grippers=[slot.gripper, …]`);
- `make_real_replay_build_env_fn` advertises `wants_per_env_meta = True` and
  accepts `per_env_episode_meta`;
- logged post-grasp apple/TCP SE(3) is applied per env
  (`apply_logged_post_grasp_se3_to_cable(..., per_env_meta=…)`);
- open-loop FR3 joints use `per_world_bootstrap_joint_q` from each meta's
  `initial_robot_joint_q` (`apply_open_loop_fr3_joint_q_per_world`; no
  broadcast from world 0).

Sim-sim (no `meta_by_direction`) is unchanged: one `request.gripper` and the
scalar bootstrap path. Per-direction controller gains stay inside the 19D
`vic_pose` action.

### Unequal lengths: pad the drive, truncate before features

Directions in one batch may differ in frame count. Replay allocates
`(num_slots, T_max, A)` and pads each shorter drive with the **last logged
action** (not zeros). After stepping, every collected array is sliced to that
slot's recorded `n_frames` **before** feature extraction / Sinkhorn, and
padded control frames are excluded from the collector `record_mask`. Padded
tails must not enter scores.

## 8. CLI cheat-sheet

Copy-paste commands and folder examples live in
`robot_replay/README.md`. Keep that file short; contracts and rationale belong
here.

## 9. Code map

| Responsibility | Module / symbol |
| -------------- | --------------- |
| Native real metadata mapping and batched export | `apple_pick_sim/system_id/real_to_batched_sysid.py` — `build_episode_metadata_from_real`, `export_real_episode_to_batched_dataset`, `export_real_tree_folder_to_batched_dataset` |
| Pre-grasp rod geometry, `mass_kg` → density | `apple_pick_sim/system_id/real_pre_grasp_params.py` — `map_pre_grasp_geometry`, `fruiting_params_from_pre_grasp_meta` |
| F/T, woody, hold, camera conversion | same module — `world_wrench_from_ee_logged`, `tag_poses_to_cma_woody`, `_scalar_hold_number`, `camera_to_base_4x4_from_dataset_metadata`, `zero_phase_lowpass`, `zero_phase_lowpass_with_status`, `block_mean_downsample` |
| Twin init and logged weld pose | `apple_pick_sim/system_id/batched_digital_twin_init.py` — `gripper_proxy_for_real_batched_replay`, `apply_logged_post_grasp_se3_to_cable` (optional `per_env_meta`) |
| Per-world open-loop joints | `apple_pick_sim/coupled_fruiting/settle_then_weld.py` — `apply_open_loop_fr3_joint_q_per_world` |
| Shared gym build path | `apple_pick_gym/batched_envs/real_batched_replay_build.py` — `real_replay_sim_config`, `make_real_replay_build_env_fn` (`wants_per_env_meta`) |
| Multi-direction replay slots, last-action pad, truncate | `apple_pick_gym/batched_envs/batched_sysid_multi_replay.py` — `ReplaySlot.episode_meta`, `_pad_actions_with_last`, `_truncate_replay_arrays` |
| Standalone replay and GL camera | `robot_replay/example_replay_real_batched.py` — `_run`, `make_replay_on_step`, `gl_camera_from_camera_to_base` |
| MP4 writer | `robot_replay/gl_video_recorder.py` — `GlVideoRecorder` |

## 10. Tests and verification

Key regression coverage:

- `apple_pick_sim/tests/test_real_to_batched_sysid.py` — metadata parity,
  camera propagation, 19D packing, F/T rotation, two woody starts, scalar
  holds, no trajectory woody ends, folder `--input-dir` (filename
  `direction_idx`, canonical geometry, `n_holds` / `sim_config` /
  `topology_seed`), and 1×1 `s00_d00` regression.
- `apple_pick_sim/tests/test_real_pre_grasp_params.py` — Branch T-junction,
  rest-snapshot preference, rod `mass_kg` → density override, and catalog
  connection angles (spur about primary, stem about fruiting→robot).
- `apple_pick_sim/tests/test_batched_digital_twin_init.py` — twin initialization
  and post-grasp SE(3), including per-env logged poses.
- `apple_pick_sim/tests/test_open_loop_joint_bootstrap.py` — per-world
  `joint_q` (no broadcast).
- `apple_pick_gym/tests/test_real_batched_replay_build.py` — shared builder,
  per-env grippers, batched logged-pose application, and snapshot recapture
  after post-grasp SE(3) so `reset()` keeps the grasped pose.
- `apple_pick_gym/tests/test_batched_sysid_multi_replay.py` — distinct
  per-direction weld/gripper metadata, last-action drive padding, and
  truncate-before-features.
- `apple_pick_gym/tests/test_real_batched_replay_cli.py` — action semantics,
  camera placement, replay callbacks, and video flag parsing.
- `robot_replay/tests/test_gl_video_recorder.py` and
  `test_pack_vic_pose_actions.py` — MP4 writer and legacy packer.

Focused checks from the repository root:

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_real_to_batched_sysid.py \
  apple_pick_sim/tests/test_real_pre_grasp_params.py \
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

The smoke proves conversion/build/19D drive, not ranking quality. Folder
convert of one tree (`s09`, eight compiled pulls):

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input-dir robot_replay/new_data/s09 \
  --dataset-out tmp/real_batched_s09 --overwrite
```

Expect `collection.num_structures=1`, `num_directions=8`, `control_hz=30`,
`n_holds=4`, and `episodes/s00_d00` … `s00_d07`. Holdout CMA on that bag is
H5 / `docs/ROADMAP.md` (Task 9 science gate failed on val torque).
