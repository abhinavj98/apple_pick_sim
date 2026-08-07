# Pre-grasp apple orientation (stem–apple frame continuity)

| Field | Value |
| ----- | ----- |
| **Status** | Implemented |
| **Date** | 2026-08-07 |
| **Amends** | `docs/superpowers/specs/2026-08-05-apple-position-only-post-grasp-weld-design.md`; resolves ROADMAP post-grasp apple orientation known issue |
| **Code** | `FruitingSystemParams.apple_quat_xyzw`, `real_pre_grasp_params.map_pre_grasp_geometry`, `fruiting_system/build.py` chain builders |

## Purpose

Seed the free-scene apple body (and stem–apple FIXED child anchor) from the
**logged pre-grasp tracker orientation** so the sim apple body frame matches the
real marker frame. Post-grasp logged rotation then applies in the same frame
family.

## Problem

Free-scene build used `quat_identity()` for the apple. `_connect_rod_tip_to_apple`
bakes the child local anchor in that identity frame. Post-grasp weld then writes
the tracked `apple_pose_4x4` quat onto `body_q`, so stem attachment and
\(X_{\mathrm{offset}}=X_{\mathrm{apple}}^{-1} X_{\mathrm{tcp}}\) are evaluated in a
different frame than the joint was built in. Position-only weld (keep settle
quat) looked correct because it stayed in the identity-built frame.

## Frame contract

| Stage | Apple orientation |
| ----- | ----------------- |
| **Build-time** (params + joint bake) | Pre-grasp tracker quat from preferred woody snapshot |
| **Free settle** | Evolves under VBD from that init |
| **Post-grasp weld** (`body_q`) | Measured post-grasp SE(3); joint still baked with pre-grasp frame on params |

Do **not** overload `GripperProxyConfig.weld_reference_quat` for free-scene init
(weld-time only).

## API

- `FruitingSystemParams.apple_quat_xyzw: tuple[float,float,float,float] | None = None`
  - `None` → identity (unchanged default for all existing callers).
  - Threaded through `to_dict` / `from_dict` / `copy_fruiting_params` / fingerprint.
- Chain builders: `apple_quat = wp.quat(*params.apple_quat_xyzw) if … else identity`.
- `map_pre_grasp_geometry`: from selected snapshot, prefer `apple_quat_xyzw`, else
  `pose_4x4_to_pos_quat(apple_pose_4x4)`; else `None`.
- `fruiting_params_from_pre_grasp_meta` assigns mapped quat onto params.

## Outcome

Resolved: default post-grasp full logged apple SE(3) weld is frame-consistent
with pre-grasp build. `--apple-position-only` remains an optional escape hatch
only.

## Tests

- `test_real_pre_grasp_params.py` — map extracts quat from rest snapshot; missing
  pose → `None`.
- `test_fruiting_system.py` — non-identity params quat → apple `body_q` and
  stem–apple child local \(R^{-1}(-r\,\hat{s})\).
- `test_real_post_grasp_plan.py` — welded rebuild keeps stem–apple child local
  in the pre-grasp apple frame.

## Non-goals

- Changing TCP / `pose_4x4` major-ness (not required once frame continuity held).
- Catalog vs chord length fixes.
- Removing `--apple-position-only`.
