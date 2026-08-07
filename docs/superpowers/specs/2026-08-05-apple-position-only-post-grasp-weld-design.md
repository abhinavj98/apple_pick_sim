# Apple position-only post-grasp weld (viewer flag)

| Field | Value |
| ----- | ----- |
| **Status** | Implemented |
| **Date** | 2026-08-05 |
| **Extends** | `docs/superpowers/specs/2026-08-04-true-tcp-pose-weld-design.md`, `robot_replay/example_view_pre_grasp_settle.py` |

## Purpose

Optional post-grasp mode that places the apple at the **measured translation**
but keeps the apple **orientation from the free settle**, ignoring logged
`apple_pose_4x4` rotation. TCP remains full logged SE(3).

## Behavior

| Item | Default (today) | `--apple-position-only` |
| ---- | ----------------- | ----------------------- |
| Apple position | `plan.apple_pos_welded` | same |
| Apple quaternion | `plan.apple_quat_xyzw` (logged) | free-scene apple `body_q[3:7]` after settle |
| TCP SE(3) | logged | logged (unchanged) |
| FIXED offset | \(X_{\mathrm{apple}}^{\mathrm{logged}}{}^{-1} X_{\mathrm{tcp}}\) | \(X_{\mathrm{apple}}^{\mathrm{settle}}{}^{-1} X_{\mathrm{tcp}}\) |
| `weld_reference_quat` | logged apple quat | settle apple quat |

Flag is only meaningful with `--grasp-after-settle`.

## API

`apply_post_grasp_after_settle(..., *, keep_apple_settle_orientation: bool = False)`.

When `True`, do not overwrite apple quat from the plan; use settle quat for
offset and weld reference.

## Open issue (debug next)

Default full-SE(3) apple weld **does not match GT**: using the logged
`apple_pose_4x4` rotation yields a scene that disagrees with ground truth, while
position-only placement looks correct. Tracked in `docs/ROADMAP.md` →
**Current focus → Known issues — debug next**. Until that is resolved,
`--apple-position-only` is the trustworthy path for real-episode replay.

## Non-goals

- Ignoring TCP rotation
- Identity apple quaternion
- Changing pre-grasp build / stem direction
