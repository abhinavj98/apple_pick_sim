# Real batched GL replay (FR3 + VIC)

| Field | Value |
| ----- | ----- |
| **Status** | Phase B GL + post-grasp settle **plumbing Done**; **correct action drive pending** (wrench ≠ twist VIC) |
| **Date** | 2026-08-10 |
| **Depends on** | Bit-2 format export (`batched_sysid_v1` from real parquet) |
| **Related** | `docs/real-sysid-pre-post-grasp-fixes.md` (C6), `robot_replay/example_replay_real_batched.py`, `docs/superpowers/specs/2026-08-10-vic-pose-action-controller-design.md` |

## Purpose

Watch a **complete open-loop FR3 trajectory** in Newton GL after rebuild and
settle, using a real episode already converted to `batched_sysid_v1`. Settle
remains off-screen; the window shows trajectory frames only.

**Drive-signal caveat:** real `action` is often a pose-control **wrench**. Twist
`mode=vic` replay is **not** physically correct until pose/wrench mode ships.
Export/replay refuse wrench-as-twist unless `--allow-wrench-as-twist`.

## Sequencing

| Phase | Scope | Status |
| ----- | ----- | ------ |
| **B** | GL `on_step` on real replay CLI; TCP motion gate; rebuild from episode metadata | **Plumbing Done**; TCP gate ≠ correct wrench drive |
| **Post-grasp** | `SceneSettleCollisionConfig.post_grasp_settle_substeps` (CLI default 500); free settle 5000 + quiet 300; rebrand smoke → `example_replay_real_batched.py` | **Done** |
| **Drive** | Pose PD (`vic_pose`) or logged-wrench apply matching real `action_semantics` | **Pending** |
| **A** | Later: teach `example_batched_sysid_mmd_grid.py` the same real clamps / flags so grids, scoring, and GL work on exported real datasets | Follow-on |

## Architecture

```text
real parquet → convert --dataset-out → batched_sysid_v1
        │
        ▼
example_replay_real_batched.py
  • Rebuild from episode metadata (native fruiting_base_pos + oracle params)
  • Open-loop FR3 from initial_robot_joint_q (skip IK)
  • gym_defaults + fixture sim_build support joints (not test_minimal CPU)
  • Free VBD settle (5000) + quiet every 300
  • Weld + seed_fix_to_apple + open-loop joints
  • Post-grasp VBD settle (500) + quiet + re-apply open-loop joints
  • FR3 + VIC open-loop replay_batched_sysid_structure(on_step=…)
        │
        ├─ viewer null: headless / pytest (short settles in CI)
        └─ viewer gl: set_model once; begin_frame / log_state / end_frame per control step
```

**Shared physics:** same library call as MMD-grid / CMA paths
(`replay_batched_sysid_structure`). Post-grasp settle is implemented in
`build_batched_heterogeneous_scene` when `post_grasp_settle_substeps > 0`
(default **0** for existing sim-to-sim callers).

## Geometry + settle + open-loop arm

Convert stores the settle-viewer twin (`fruiting_base_pos`,
`fruiting_system_params`, `initial_robot_joint_q`). Replay must use those
fields. Arm placement writes recorded joints (**no IK**). Batched free settle
also needs fixture ``sim_build`` support-joint kp/ζ on a CUDA-capable device.

## CLI

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input robot_replay/s02-d00_action.parquet \
  --dataset-out /tmp/real_batched_s02_d00 --overwrite

uv run python robot_replay/example_replay_real_batched.py \
  --dataset /tmp/real_batched_s02_d00 \
  --viewer gl \
  --max-frames 0 \
  --settle-substeps 5000 \
  --settle-quiet-every 300 \
  --post-grasp-settle-substeps 500
```

- `--viewer {gl,null}` via `newton.examples.init`.
- No `DISPLAY` / `WAYLAND_DISPLAY` → auto-fallback to `null`.
- Exit: print `tcp_motion_m`; non-zero if TCP stationary.

## Out of scope

- Rendering settle or post-grasp weld phases in GL
- Changing `replay_batched_sysid_structure` internals beyond build config
- MMD-grid real clamps (phase A)
- CMA-ES / scoring / candidate grids
- Single-env `example_gym_replay.py` (different dataset contract)

## Testing

- Keep `test_real_exported_s02_replay_moves_tcp` as the headless TCP gate with
  short free settle and `post_grasp_settle_substeps=0` (no full 5000+500 in CI).
- CLI unit tests: parser defaults 5000 / 300 / 500; mock-viewer `on_step`.
- Build unit tests: `post_grasp_settle_substeps` second VBD settle + rebootstrap.

## Phase A (follow-on)

Optional flag or auto-detect on `example_batched_sysid_mmd_grid.py` so grids,
scoring, and GL work on exported real datasets using the same episode-metadata
rebuild (no fixture pose/length override) without going through the real-replay
CLI.
