# Post-grasp pretension on the real-replay path

| Field | Value |
| ----- | ----- |
| **Date** | 2026-09-05 |
| **Status** | Implemented |
| **Related** | Settle → weld seed; post-grasp SE(3); `EpisodeStateSnapshot` |
| **Supersedes** | `2026-09-05-build-time-vbd-rest-after-settle-design.md` |

## Problem

Plant structural FIXED joints are hard / augmented-Lagrangian. Sustained
pretension lives in SolverVBD `joint_lambda_*`, decaying `joint_penalty_k`, and
`joint_C0_*` — not in geometry. Three places wiped that state:

1. Weld rebuild copies only `body_q` / `body_qd` into a fresh solver (λ = 0).
2. Post-grasp rebootstrap full-synced `model.body_q` from state, moving angular
   rest onto the loaded pose.
3. Episode snapshots omitted AVBD multipliers, so `reset()` paired t=0 poses
   with end-of-episode λ.

Apple rest is shared by `joint_<stem>_apple` and `joint_apple_gripper_proxy`.
Rewriting apple rest to quiet the weld also zeros stem→apple bend preload.

## Decision

1. Freeze as-built `model.body_q` for woody bodies **and the apple**.
2. Quiet the weld with `sync_weld_proxy_rest_from_apple_rest`:
   `model.body_q[proxy] ← model.body_q[apple] * offset`.
3. Re-derive pretension with post-grasp VBD settle on the welded solver.
4. Robot-only rebootstrap after settle (no cable rest overwrite).
5. Capture/restore AVBD λ / `penalty_k` / C0 / `body_q_prev` (and Dahl if
   present) in `EpisodeStateSnapshot`.

## Code map

- `proxy_coupling.sync_weld_proxy_rest_from_apple_rest`
- `settle_then_weld.seed_fix_to_apple_from_settled`
- `batched_digital_twin_init.apply_logged_post_grasp_se3_to_cable`
- `batched_heterogeneous_build._rebootstrap_fr3_after_post_grasp_settle`
- `post_grasp_pretension` (load-split + settle preload report)
- `episode_state_snapshot.EpisodeStateSnapshot`

## Tests

- `test_settle_then_weld.py::test_seed_keeps_plant_rest_at_build_sets_proxy_rest_from_apple_rest`
- `test_batched_heterogeneous_build.py::test_post_grasp_settle_preserves_woody_and_apple_model_rest`
- `test_episode_state_snapshot.py::test_snapshot_restores_avbd_lambda_and_penalty_k`
- `test_post_grasp_pretension.py` (slow)

## Out of scope

Gym RL / MMD npz migration; soft plant joints; `rigid_avbd_beta` / k_start
misconfiguration.
