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
post_grasp →  weld robot TCP to settled apple (pose + approach direction)
```

Full fix list, quirks (including string-encoded `apple_pos`), and compile
asserts: **`docs/real-sysid-pre-post-grasp-fixes.md`**.

## Files

| Path | Role |
| ---- | ---- |
| `s00-d00.parquet` | Example compiled real episode (`real_static_sysid_episode` v1.0.0) |
| `s00-d00_spur_toward_robot.parquet` | Same episode; woody hang rotated so spur aims at robot base (origin) |
| `funified.parquet` | Additional real / fused episode artifact |
| `manifest.json` | Collection/run manifest (sim-side batch metadata companion) |
| `make_spur_toward_robot_parquet.py` | Rotate hang so spur points at robot base; regenerates the variant parquet |

## Pre-grasp settle viewer
| `convert_real_to_batched_sysid_metadata.py` | CLI → batched-style episode metadata JSON |

## Pre-grasp settle viewer

Rebuild `FruitingSystemParams` from `dataset_metadata.pre_grasp_geometry`
(Branch = T-junction / `fruiting_base_pos`), build a plant-only coupled cable
scene (free gripper proxy, no FR3 weld), settle under gravity in the viewer,
then keep simulating.

From repo root:

```bash
uv run python robot_replay/example_view_pre_grasp_settle.py \
  --parquet robot_replay/s00-d00.parquet \
  --fixture apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json \
  --settle-substeps 5000 \
  --viewer gl
```

`--dump-params` writes `fruiting_base_pos`, the sim-native
`fruiting_system_params` blob, and catalog-vs-chord diagnostics (for later
dataset embedding). `--strict` fails only if pre-grasp woody bend is not ~0.
`--settle-quiet-every N` zeros cable body twists every N settle substeps
(default 300; `<=0` disables), matching batched settle behavior.

Library: `apple_pick_sim/system_id/real_pre_grasp_params.py`.

Headless smoke:

```bash
uv run python robot_replay/example_view_pre_grasp_settle.py \
  --parquet robot_replay/s00-d00.parquet \
  --settle-substeps 100 \
  --viewer null --num-frames 5 \
  --dump-params /tmp/s00_d00_params.json
```

## Convert CLI

From repo root:

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input robot_replay/s00-d00.parquet \
  --fixture apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json \
  --out /tmp/s00_d00_episode_meta.json
```

Implementation: `apple_pick_sim/system_id/real_to_batched_sysid.py`.

**Note:** Current converter expectations (`step_idx == -1` pre-grasp row,
`robot_joint_q`, quats, `rod_geometry`, …) do **not** yet fully match
`s00-d00.parquet` as compiled. See the fix doc (**C1**) before treating convert
success as the ingest gate.

## Related docs

- `docs/real-sysid-pre-post-grasp-fixes.md` — collection/compile fix list
- `docs/digital-twin.md` — observation-only replay and geometry reconstruction
- `docs/real-world-proxy.md` — bench proxy placement and frames
- `docs/batched-sysid-dataset.md` — target batched episode layout
