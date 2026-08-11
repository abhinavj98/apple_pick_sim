# `robot_replay/` — real-robot sys-ID episodes

Working directory for **real-robot** static sys-ID parquet episodes and the
metadata conversion CLI toward `batched_sysid_v1`-style episode JSON.

Sim-collected batched datasets live elsewhere (`docs/batched-sysid-dataset.md`).
This folder is the bridge from bench logs → twin init / replay tooling.

## Pre-grasp vs post-grasp (contract)

Real episodes carry (or should carry) two geometry blocks. They are used in
sequence:

1. **`pre_grasp_geometry`** — Apple in a **correct position** with **branches
   not bending**. Supporting / opposing forces largely cancel gravity so the
   plant is held without visible deflection. Use this to **rebuild
   `fruiting_system` geometry** (rod directions, lengths, apple placement).
2. **`post_grasp_geometry`** — **Settled apple under grasp**: the fruit after the
   robot has grasped from a known **direction and position**. Use this to
   **weld / attach the robot** to the plant that was built from pre-grasp and
   settled in simulation.

```text
pre_grasp  →  build fruiting_system (non-bending)  →  sim settle
post_grasp →  weld proxy at logged TCP SE(3) and apple at measured post-grasp pose
```

Full fix list, quirks (including string-encoded `apple_pos`), and compile
asserts: **`docs/real-sysid-pre-post-grasp-fixes.md`**.

**Woody snapshot selection** (`select_pre_grasp_woody_snapshot`): prefer
`pre_grasp_geometry.rest_snapshot_during_run` when it has `woody_part_*` +
`apple_pos`; fall back to legacy `snapshot`; never use `settled_snapshot` for
rebuild.

## Files

| Path | Role |
| ---- | ---- |
| `s00-d00.parquet` | Example compiled real episode (`real_static_sysid_episode` v1.0.0) |
| `s02-d00_action.parquet` | Preferred bit-2 source: real episode with non-zero `action` |
| `funified.parquet` | Additional real / fused episode artifact |
| `manifest.json` | Collection/run manifest (sim-side batch metadata companion) |
| `example_view_pre_grasp_settle.py` | Plant-only VBD: pre-grasp settle → optional post-grasp weld → viewer |
| `example_view_batched_episode_meta.py` | Same settle/weld view from **converted** episode metadata JSON |
| `example_replay_real_batched.py` | FR3+VIC real replay test (settle→weld→post-grasp settle→GL/actions) via `replay_batched_sysid_structure` |
| `convert_real_to_batched_sysid_metadata.py` | CLI → metadata JSON and/or 1×1 `batched_sysid_v1` (`--dataset-out`) |

## Pre-grasp settle viewer

Rebuild `FruitingSystemParams` from `dataset_metadata.pre_grasp_geometry`
(Branch = T-junction / `fruiting_base_pos`), including apple **orientation** from
the preferred woody snapshot (`apple_pose_4x4` / `apple_quat_xyzw`) so the
stem–apple joint is baked in the tracker frame. Build a plant-only coupled cable
scene (free gripper proxy, no FR3), settle under gravity in the viewer,
optionally apply a post-grasp true TCP SE(3) weld (logged TCP + apple poses;
no catalog surface snap), then keep simulating.

From repo root:

```bash
uv run python robot_replay/example_view_pre_grasp_settle.py \
  --parquet robot_replay/s00-d00.parquet \
  --fixture apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json \
  --settle-substeps 5000 \
  --viewer gl
```

Post-grasp weld (true logged TCP + apple SE(3); follow measured poses):

```bash
uv run python robot_replay/example_view_pre_grasp_settle.py \
  --parquet robot_replay/s00-d00.parquet \
  --grasp-after-settle \
  --post-grasp-settle-substeps 500 \
  --tcp-radius-warn-m 0.02 \
  --viewer gl
```

`--apple-position-only` (with `--grasp-after-settle`): optional escape hatch —
snap apple **translation** from logged post-grasp data but keep apple
**orientation from the free settle** (ignore logged apple quat). TCP SE(3) is
unchanged; the FIXED offset is rebuilt from settle apple quat + logged TCP.

**Apple orientation:** free rebuild seeds apple orientation from pre-grasp
`apple_pose_4x4` / `apple_quat_xyzw` so the stem–apple joint is in the tracker
frame; default post-grasp weld uses full logged apple SE(3) and matches GT.
See `docs/superpowers/specs/2026-08-07-pre-grasp-apple-orientation-design.md`.

`--dump-params` writes `fruiting_base_pos`, the sim-native
`fruiting_system_params` blob, and catalog-vs-chord diagnostics.
`--strict` fails only if pre-grasp woody bend is not ~0.
`--settle-quiet-every N` zeros cable twists every N settle substeps (default 300).
Residual diagnostics (`|TCP−apple|−r`, apple shift vs measured, TCP +Z vs chord)
**warn and continue**; poses are not moved to force `|TCP−apple|=r` or look-at
orientation.

Libraries: `real_pre_grasp_params.py`, `real_post_grasp_plan.py`.

Headless smoke:

```bash
uv run python robot_replay/example_view_pre_grasp_settle.py \
  --parquet robot_replay/s00-d00.parquet \
  --grasp-after-settle \
  --settle-substeps 80 \
  --post-grasp-settle-substeps 40 \
  --viewer null --num-frames 8
```

## Convert CLI

The converter is a **thin adapter** over the settle-viewer native builders
(`fruiting_params_from_pre_grasp_parquet`, `post_grasp_plan_from_metadata`).

- **Bit 1:** `--out` → batched-style episode metadata JSON (rebuild + grasp init).
- **Bit 2:** `--dataset-out` → 1×1 `batched_sysid_v1` dataset (`manifest.json` +
  `episodes/s00_d00.parquet`) for trajectory viz / FR3 replay.

Parity gate (native vs convert):

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_real_to_batched_sysid.py::test_s00_d00_convert_matches_native_pre_post -q
```

From repo root:

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input robot_replay/s00-d00.parquet \
  --fixture apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json \
  --out /tmp/s00_d00_episode_meta.json
```

Optional: `--weld-direction-sign {+1,-1}` (see CLI `--help`).

Eyeball the converted JSON (same settle / optional weld as the native viewer):

```bash
uv run python robot_replay/example_view_batched_episode_meta.py \
  --episode-meta /tmp/s00_d00_episode_meta.json \
  --grasp-after-settle \
  --settle-substeps 80 \
  --post-grasp-settle-substeps 40 \
  --viewer null --num-frames 8
```

Implementation: `apple_pick_sim/system_id/real_to_batched_sysid.py`.
Design: `docs/superpowers/specs/2026-08-07-real-to-batched-metadata-parity-design.md`.
Bit-2 plan: `docs/superpowers/plans/2026-08-07-real-batched-trajectory-replay-bit2.md`.

### Bit 2 — full dataset + trajectory viz + placement smoke

Preferred source: `s02-d00.parquet` (has non-zero logged wrench + `target_pose_4x4`
+ `dump.controller_gains`).

**Action packing:** compiled real logs store a pose-control **wrench** `[Fx…Tz]` in
`action` (`dump.action_semantics`), not an EE twist. Convert **packs** a 19D
`vic_pose` action instead:

```text
[pos(3), quat_wxyz(4), Kp(6), Kd(6)]
```

from per-frame `target_pose_4x4` and episode `dump.controller_gains`
(`task_prop_gains` / `task_deriv_gains`). Episode metadata stamps
`action_dim=19`, `action_layout=vic_pose_v1`, and
`action_compatible_with_vic_twist=false`.

**Replay modes:** `example_replay_real_batched.py` defaults to `--controller-mode vic`
(6D EE twist). For converted datasets with 19D `vic_pose` actions, pass
`--controller-mode vic_pose`. Default `vic` refuses wrench-marked datasets
unless `--allow-wrench-as-twist` (incorrect physics; format/GL smoke only) —
that hatch covers **legacy 6D** exports only and is rejected outright for
`action_layout=vic_pose_v1` datasets.

`pack_vic_pose_actions.py` is a **legacy** helper for datasets converted **before**
the exporter packed 19D actions (it rebuilds `action` from `target_pose_4x4`,
`tcp_pose_4x4`, or `tcp_pos`/`tcp_quat`). Because convert now writes
`action_layout=vic_pose_v1` itself, **skip packing** for freshly converted
datasets: the packer refuses an already-19D source unless you pass `--force`
(only useful to substitute different constant `Kp`/`Kd`).

`fill_actions_from_tcp_velocity.py` is only for older **zero-action** files; it
does **not** convert wrench logs into correct drive and is **not** equivalent to
real pose-PD wrench / `vic_pose` packing.

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input robot_replay/s02-d00.parquet \
  --dataset-out /tmp/real_batched_s02_d00 \
  --overwrite

uv run python apple_pick_gym/batched_examples/example_batched_sysid_trajectory_viz.py \
  --dataset /tmp/real_batched_s02_d00 \
  --output /tmp/real_batched_s02_d00_viz \
  --no-hold-check

# Legacy only (skip for freshly converted 19D datasets): re-pack a 6D-twist dataset,
# or --force a re-pack with explicit constant Kp/Kd.
uv run python robot_replay/pack_vic_pose_actions.py \
  --dataset-in /tmp/real_batched_s02_d00 \
  --dataset-out /tmp/real_batched_s02_d00_vic_pose \
  --kp 800 800 800 40 40 40 \
  --kd 80 80 80 4 4 4 \
  --overwrite --force

# vic_pose replay (19D pose+gains actions straight from convert):
uv run python robot_replay/example_replay_real_batched.py \
  --dataset /tmp/real_batched_s02_d00 \
  --controller-mode vic_pose \
  --max-frames 24 --viewer null

# Default vic (6D twist) — only for legacy twist-compatible datasets:
uv run python robot_replay/example_replay_real_batched.py \
  --dataset /tmp/real_batched_s02_d00 \
  --max-frames 24 --viewer null

# GL: FR3 trajectory after off-screen settle (--max-frames 0 = full episode).
# Settle defaults match example_view_pre_grasp_settle.
# Open-loop joints from initial_robot_joint_q (skip IK; base at origin).
uv run python robot_replay/example_replay_real_batched.py \
  --dataset /tmp/real_batched_s02_d00 \
  --controller-mode vic_pose \
  --viewer gl --max-frames 0 \
  --settle-substeps 5000 --settle-quiet-every 300 \
  --post-grasp-settle-substeps 500
```

TCP-motion pytest (short settle for CI; skips if preferred parquet missing):

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_real_batched_replay.py::test_real_exported_s02_replay_moves_tcp \
  -q -p no:launch_testing
```

**Note:** Older `s00-d0*` episodes may still have empty `action`
(`real-replay-action-zero`). Prefer `s02-d00.parquet` (or newer fixed
logs). Optional temporary fill for old files:

```bash
uv run python robot_replay/fill_actions_from_tcp_velocity.py \
  --input robot_replay/s00-d03.parquet \
  --out robot_replay/s00-d03_with_actions.parquet
```

Replay rebuilds from converted episode metadata (same native geometry as
`example_view_pre_grasp_settle` / `example_view_batched_episode_meta`): episode
`fruiting_base_pos` and oracle `fruiting_system_params`. FR3 placement is
**open-loop** from `initial_robot_joint_q` (no IK — real grasps near the
workspace edge often fail IK from `robot_base_pos=(0,0,0)`). Physics uses
`gym_defaults` + fixture `sim_build` on the default sim device. CLI settle
defaults match `example_view_pre_grasp_settle.py` (`--settle-substeps 5000`,
`--settle-quiet-every 300`, `--post-grasp-settle-substeps 500`); pytest keeps
a short free settle and skips post-grasp settle for speed.

## Related docs

- `docs/real-sysid-pre-post-grasp-fixes.md` — collection/compile fix list
- `docs/digital-twin.md` — observation-only replay and geometry reconstruction
- `docs/real-world-proxy.md` — bench proxy placement and frames
- `docs/batched-sysid-dataset.md` — target batched episode layout
- `docs/superpowers/specs/2026-08-10-vic-pose-action-controller-design.md` — `vic_pose` controller + 19D action layout
