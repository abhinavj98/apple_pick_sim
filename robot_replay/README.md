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
| `convert_real_to_batched_sysid_metadata.py` | CLI → batched-style episode metadata JSON (`--weld-direction-sign`) |

## Pre-grasp settle viewer

Rebuild `FruitingSystemParams` from `dataset_metadata.pre_grasp_geometry`
(Branch = T-junction / `fruiting_base_pos`), build a plant-only coupled cable
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

From repo root:

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input robot_replay/s00-d00.parquet \
  --fixture apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json \
  --out /tmp/s00_d00_episode_meta.json
```

Optional: `--weld-direction-sign {+1,-1}` (see CLI `--help`).

Implementation: `apple_pick_sim/system_id/real_to_batched_sysid.py`.

**Note:** Current converter expectations (`step_idx == -1` pre-grasp row,
`robot_joint_q`, quats, `rod_geometry`, …) do **not** yet fully match
`s00-d00.parquet` as compiled. See the fix doc (**C1**) before treating convert
success as the ingest gate. The settle viewer path (`real_pre_grasp_params`) is
the supported consumer for current episodes; convert remains a separate,
still-mismatched contract.

## Related docs

- `docs/real-sysid-pre-post-grasp-fixes.md` — collection/compile fix list
- `docs/digital-twin.md` — observation-only replay and geometry reconstruction
- `docs/real-world-proxy.md` — bench proxy placement and frames
- `docs/batched-sysid-dataset.md` — target batched episode layout
