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
| `s02-d00.parquet` | Preferred bit-2 source: real episode with wrench + `target_pose_4x4` + gains |
| `s02-d00_action.parquet` | Legacy/local alias (optional) |
| `funified.parquet` | Additional real / fused episode artifact |
| `manifest.json` | Collection/run manifest (sim-side batch metadata companion) |
| `example_view_pre_grasp_settle.py` | Plant-only VBD: pre-grasp settle → optional post-grasp weld → viewer |
| `example_view_batched_episode_meta.py` | Same settle/weld view from **converted** episode metadata JSON |
| `example_replay_real_batched.py` | FR3+`vic_pose` real replay (settle→weld→post-grasp settle→GL/actions) |
| `pack_vic_pose_actions.py` | Legacy 6D→19D packer (skip for freshly converted datasets) |
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

Bit 2 Sinkhorn woody/apple come from table columns `branch_pose_4x4`,
`spur_pose_4x4`, and `apple_pose_4x4` (translations). Convert **raises** if any
of those is missing. Packed `woody_part_start_pos` / `woody_part_end_pos` on the
source file are ignored. Converted bags still write `woody_start__primary_spur`,
`woody_start__spur_stem`, and `apple_pos`.

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

**Replay modes:** `example_replay_real_batched.py` defaults to
`--controller-mode vic_pose` (19D pose+gains). Converted wrench-origin datasets
match that default. Use `--controller-mode vic` only for legacy 6D-twist
datasets. `--allow-wrench-as-twist` is a legacy hatch for old 6D wrench copies
(incorrect physics) and is rejected for `action_layout=vic_pose_v1` datasets.

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

# Default path: vic_pose replay (19D pose+gains from convert):
uv run python robot_replay/example_replay_real_batched.py \
  --dataset /tmp/real_batched_s02_d00 \
  --max-frames 24 --viewer null

### Bit 3 — grid smoke (slice 1 plumbing; F/T ranking deferred to slice 2)

After convert, run a tiny support-\(k_p\) × spur/stem \(E\) grid on the same
real-replay builder (`real_batched_replay_build.make_real_replay_build_env_fn`).
Grid auto-detects `vic_pose_v1` / `action_dim=19`; `--include-gt-candidate` is
forced off on real datasets (no `collection.sim_config` oracle).

**Slice 1 success criteria:** envs build, 19D actions replay, no wrench-as-twist,
no `sim_config` crash. **Ranking F/T (Sinkhorn on `ft_wrist`) is not trusted
until slice 2** (F/T frame alignment + LPF on sim bags; pose-only action features).

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input robot_replay/s02-d00.parquet \
  --dataset-out /tmp/real_batched_s02_d00 \
  --overwrite

uv run python apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py \
  --dataset /tmp/real_batched_s02_d00 \
  --output /tmp/real_kp_e_grid \
  --viewer null \
  --support-kp-values 1e3,1e4 \
  --log10-e-spur 9.0 \
  --log10-e-stem 9.0 \
  --no-include-gt-candidate \
  --overwrite
```

CLI help smoke (no GPU): `example_youngs_modulus_sys_id.py --help` must list
`--controller-mode {vic,vic_pose}`. Full GPU grid smoke needs
`robot_replay/s02-d00.parquet` locally (parquet episodes are often not in clone).

Design: `docs/superpowers/specs/2026-08-12-real-replay-cmaes-plumbing-design.md`.

# Legacy only: re-pack a pre-vic_pose 6D dataset, or --force new constant Kp/Kd.
uv run python robot_replay/pack_vic_pose_actions.py \
  --dataset-in /tmp/old_6d_batched \
  --dataset-out /tmp/old_6d_batched_vic_pose \
  --kp 800 800 800 40 40 40 \
  --kd 80 80 80 4 4 4 \
  --overwrite

# Legacy twist mode (6D) — only for twist-compatible datasets:
uv run python robot_replay/example_replay_real_batched.py \
  --dataset /tmp/twist_compatible_batched \
  --controller-mode vic \
  --max-frames 24 --viewer null

# GL: FR3 trajectory after off-screen settle (--max-frames 0 = full episode).
# Settle defaults match example_view_pre_grasp_settle.
# Open-loop joints from initial_robot_joint_q (skip IK; base at origin).
uv run python robot_replay/example_replay_real_batched.py \
  --dataset /tmp/real_batched_s02_d00 \
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
