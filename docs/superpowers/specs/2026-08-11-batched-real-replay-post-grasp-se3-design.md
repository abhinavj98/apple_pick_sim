# Batched real replay: pre-grasp init + post-grasp SE(3) weld

| Field | Value |
| ----- | ----- |
| **Status** | Implemented |
| **Canonical living doc:** | `docs/handbook-real-replay.md` |
| **Date** | 2026-08-11 |
| **Extends** | `docs/superpowers/specs/2026-08-07-pre-grasp-apple-orientation-design.md`, `docs/superpowers/specs/2026-08-04-true-tcp-pose-weld-design.md`, `docs/superpowers/specs/2026-08-07-real-to-batched-metadata-parity-design.md` |
| **Code** | `gripper_proxy_for_real_batched_replay`, `apply_logged_post_grasp_se3_to_cable` in `batched_digital_twin_init.py`; wired in `robot_replay/example_replay_real_batched.py` |

## Purpose

Make `example_replay_real_batched.py` follow the same apple lifecycle as
`example_view_pre_grasp_settle.py --grasp-after-settle` (default full logged
apple SE(3), no position-only escape hatch on this example):

1. **Before weld** (free settle): apple orientation matches **pre-grasp / init**.
2. **At grasp / weld**: apple + TCP match **logged post-grasp SE(3)**.

## Problem

| Stage | Settle viewer | Batched real replay (today) |
| ----- | ------------- | --------------------------- |
| Free settle | `params.apple_quat_xyzw` = pre-grasp tracker | Same via `fruiting_system_params` |
| At weld | `apply_post_grasp_after_settle` writes logged apple SE(3) + true TCP offset | `seed_fix_to_apple_from_settled*` keeps **settle** apple `body_q`; proxy often from `weld_direction` surface placement, not \(X_{\mathrm{apple}}^{-1} X_{\mathrm{tcp}}\) |

So free-settle init already matches. The gap is **post-grasp**: batched keeps settle
apple orientation instead of applying logged post-grasp SE(3).

## Frame / pose contract

| Stage | Apple | Gripper / TCP |
| ----- | ----- | ------------- |
| Build + free settle | Pre-grasp quat on `FruitingSystemParams` (stem–apple joint baked in that frame) | Free proxy (`fix_to_apple=False`) |
| At weld (after settle) | Logged `initial_apple_pos` / `initial_apple_quat` | Logged `initial_tcp_*` via FIXED offset \(X_{\mathrm{offset}} = X_{\mathrm{apple}}^{-1} X_{\mathrm{tcp}}\) |
| Woody | Settled poses preserved | — |

Joint bake stays on **pre-grasp** params (unchanged from 2026-08-07). Runtime
apple `body_q` at weld switches to **post-grasp** logged SE(3), same as settle
viewer.

## Scope

| In | Out |
| -- | --- |
| `example_replay_real_batched.py` (slice A) | `--apple-position-only` (not on this example) |
| Shared `make_real_replay_build_env_fn` + batched post-grasp SE(3) helper (slice B) | Moving apply into `seed_fix_to_apple_from_settled*` itself (helper runs after seed instead) |
| Full logged apple + true TCP SE(3) at weld | Changing settle-viewer defaults |

## Mechanism (slice A)

1. **Gripper before env build** (from episode metadata already written by the
   real→batched converter):
   - `weld_reference_pos` / `weld_reference_quat` = logged apple
   - `weld_proxy_offset_in_apple_frame` =
     `proxy_offset_from_apple_and_tcp(...)` from `initial_apple_*` +
     `initial_tcp_*` (same helper as settle viewer)
2. **Free settle** unchanged: still uses `true_params` / pre-grasp
   `apple_quat_xyzw`.
3. **After settle→weld seed**, example applies a small helper that:
   - Writes apple `body_q` to logged post-grasp SE(3)
   - Re-aligns proxy from the explicit apple-frame offset
   - Calls `sync_model_body_q_rest_from_state` (and VBD prev-pose align as needed)

Helper lives under `apple_pick_sim/system_id/` (e.g. next to
`real_post_grasp_plan` / digital-twin init) so slice B can call it from
`seed_fix_to_apple_*` without re-homing logic in the example.

## API sketch

```text
gripper_proxy_for_real_batched_replay(meta) -> GripperProxyConfig
  # extends gripper_proxy_from_episode_metadata with explicit TCP offset

apply_logged_post_grasp_se3_to_cable(cable_or_scene, meta) -> None
  # apple body_q ← initial_apple_*; proxy ← apple ⊗ offset; sync rest
```

Exact names may match existing modules; behavior above is normative.

## Tests

- Unit: given synthetic episode meta + minimal cable/state stub (or existing
  coupled scene fixture), after helper:
  - apple pos/quat match `initial_apple_*` (abs quat-dot ≈ 1)
  - proxy pos/quat match `initial_tcp_*` within existing post-grasp tols
- Example / wiring: real replay build path passes gripper with non-`None`
  `weld_proxy_offset_in_apple_frame` when meta has TCP+apple initials
- Do not require GL or full FR3 for the unit test

## Non-goals

- Flipping settle-viewer default to position-only
- Applying this to all batched sys-ID replay consumers in this slice
- Changing converter metadata fields (`initial_*` / `weld_reference_*` already
  carry post-grasp poses)

## Slice B (implemented)

Post-grasp SE(3) runs on the shared real-replay `build_env_fn` after
`ApplePickBatchedSysIdEnv` construct (free settle → `seed_fix_to_apple_from_settled`):

- `make_real_replay_build_env_fn` in `real_batched_replay_build.py` synthesizes
  `gripper_proxy_for_real_batched_replay` (or honors fused `gripper` /
  `per_env_grippers`) and calls `apply_logged_post_grasp_se3_to_cable(..., layout=...)`
  for all envs when `num_envs > 1`.
- Grid/CMA consumers opt in via `dataset_declares_vic_pose` or
  `--controller-mode vic_pose` (`example_youngs_modulus_sys_id.py`).

The example remains a thin CLI over the same builder. Apply is **not** inside
`seed_fix_to_apple_from_settled*`; behavior is normative via the shared env factory.
