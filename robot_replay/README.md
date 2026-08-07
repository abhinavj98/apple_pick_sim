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
| `funified.parquet` | Additional real / fused episode artifact |
| `manifest.json` | Collection/run manifest (sim-side batch metadata companion) |
| `example_view_pre_grasp_settle.py` | Plant-only VBD: pre-grasp settle → optional post-grasp weld → viewer |
| `example_view_batched_episode_meta.py` | Same settle/weld view from **converted** episode metadata JSON |
| `convert_real_to_batched_sysid_metadata.py` | CLI → batched-style episode metadata JSON (`--weld-direction-sign`) |

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
(`fruiting_params_from_pre_grasp_parquet`, `post_grasp_plan_from_metadata`). It
emits batched-style **metadata JSON** (rebuild + grasp init). It does **not**
export trajectory rows yet (bit 2).

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

**Note:** Current episodes may still have empty `action` columns
(`real-replay-action-zero`); that blocks full FR3 trajectory replay until bit 2
and/or recompiled logs with a non-zero command channel. Metadata convert for
pre/post rebuild+grasp is supported on `s00-d00`-class files.

### Local drive fixture (bit 2 prep)

While real collection is fixed, build a **temporary** parquet with `action`
filled from `tcp_velocity` (gitignored `*.parquet` — regenerate locally):

```bash
uv run python robot_replay/fill_actions_from_tcp_velocity.py \
  --input robot_replay/s00-d03.parquet \
  --out robot_replay/s00-d03_with_actions.parquet
```

Stamps `dataset_metadata.drive_fill`. Not the long-term contract. Bit-2 plan:
`docs/superpowers/plans/2026-08-07-real-batched-trajectory-replay-bit2.md`.

## Related docs

- `docs/real-sysid-pre-post-grasp-fixes.md` — collection/compile fix list
- `docs/digital-twin.md` — observation-only replay and geometry reconstruction
- `docs/real-world-proxy.md` — bench proxy placement and frames
- `docs/batched-sysid-dataset.md` — target batched episode layout
