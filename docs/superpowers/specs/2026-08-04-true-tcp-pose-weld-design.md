# True TCP pose post-grasp weld

| Field | Value |
| ----- | ----- |
| **Status** | Implemented |
| **Date** | 2026-08-04 |
| **Scope** | Post-grasp replay weld uses full logged TCP SE(3); drop look-at orientation |
| **Amends** | `docs/superpowers/specs/2026-07-24-real-post-grasp-viewer-design.md` (orientation contract) |
| **Out of scope** | Generic `weld_direction` look-at for other sims; FR3; trajectory replay; woody teleport |

## Warnings

> **Warning — look-at vs logged TCP:** Gym, digital-twin, and generic
> `weld_direction` look-at welds do **not** yet consume a logged TCP SE(3).
> They use tip-out look-at (surface pole + constructed orientation). Only
> post-grasp replay (`real_post_grasp_plan` / `--grasp-after-settle`) uses
> full logged TCP pose. Do not assume look-at orientation matches recorded
> TCP quat.

Related: `robot_replay/example_view_pre_grasp_settle.py`,
`apple_pick_sim/system_id/real_post_grasp_plan.py`,
`apple_pick_sim/fruiting_system/params.py` (`GripperProxyConfig`),
`apple_pick_sim/fruiting_system/build.py` (`_add_gripper_proxy`).

## Problem

The 2026-07-24 post-grasp viewer intentionally **ignored** logged TCP rotation and
built a look-at FIXED joint so proxy **+Z ∥ ŵ**. Real episodes (e.g. `s00-d00`)
have TCP +Z far from ŵ; the code warns and replaces the quat. The desired replay
behavior is **true TCP pose**: proxy world position **and** quaternion from
`post_grasp_geometry.tcp_pose_4x4`.

## Locked behavior

| Quantity | Role |
| -------- | ---- |
| Proxy **position** | Logged TCP translation (`tcp_pos` / `tcp_pose_4x4`) |
| Proxy **orientation** | Logged TCP quat from `tcp_pose_4x4` (**not** look-at) |
| Apple position | Measured `apple_pos` / `apple_pose_4x4` (**no** catalog \(r\) surface snap) |
| Apple orientation | Measured `apple_pose_4x4` quat |
| Woody | Unchanged after long free settle |

Grasp chord (diagnostic only):

```text
ŵ = normalize(p_tcp − p_apple_meas)
```

No forcing of \(|\mathbf{p}_{\mathrm{tcp}}-\mathbf{p}_{\mathrm{apple}}|=r\). Residual
\(|d-r|\) may warn but does not move the apple or TCP.

Orientation: EE **+Z** is tip-out toward the fruit. Diagnostic compares
\(\hat{z}_{\mathrm{tcp}}\cdot\mathrm{unit}(\mathbf{p}_{\mathrm{apple}}-\mathbf{p}_{\mathrm{tcp}})\)
(expect near +1). Misalignment warns only; logged TCP quat is still used.

## Approach (chosen)

**Relative FIXED offset from true poses:**

1. After free settle, plan apple `(p_apple_meas, q_apple)` and TCP `(p_tcp, q_tcp)`.
2. Compute apple-frame offset \(X_{\mathrm{offset}} = X_{\mathrm{apple}}^{-1}\, X_{\mathrm{tcp}}\).
3. Rebuild welded scene with an **explicit** FIXED `parent_xform` = that offset
   (no look-at / no roll randomization).
4. Seed woody from free settle; set apple + proxy body_q from the plan; quiet;
   sync VBD rest as today.

### Rejected

- Build look-at then overwrite proxy quat only — FIXED joint would still encode
  look-at; rest/constraint can fight the seed.
- Catalog surface snap (`apple_welded = tcp − r·ŵ`) — user wants follow-data poses.

## Architecture

### `GripperProxyConfig`

Add optional:

```text
weld_proxy_offset_in_apple_frame: tuple[7 floats] | None = None
# (px, py, pz, qx, qy, qz, qw) parent_xform for joint_apple_gripper_proxy
```

When `fix_to_apple=True` and this field is set:

- `_add_gripper_proxy` uses it as FIXED `parent_xform` / exported
  `gripper_proxy_offset_in_apple_frame`.
- Do **not** construct look-at from `weld_direction`.
- Still require / use `weld_reference_pos` (+ quat) so apple center for placement
  checks and initial proxy world pose match the plan.
- Mutual exclusion: if explicit offset is set, ignore look-at path; `weld_direction`
  may still be present for diagnostics but must not override orientation.

Generic robot-facing / random / `weld_direction` look-at paths remain for other
callers that do not set the explicit offset.

### `real_post_grasp_plan`

- Keep `build_post_grasp_plan` snap math and radius / apple-shift / translation
  mismatch warnings.
- **Remove** (or stop emitting) the “poorly aligned → ignore logged TCP quat”
  warning as a behavior-changing message; optional soft diagnostic is fine only
  if it does **not** claim the quat is ignored.
- `apply_post_grasp_after_settle`:
  - Compute explicit offset from plan poses.
  - Pass `GripperProxyConfig(fix_to_apple=True, weld_proxy_offset_in_apple_frame=…,
    weld_reference_pos=apple_welded, weld_reference_quat=apple_quat)`.
  - Seed apple + proxy to plan SE(3); assert/warn proxy pos≈tcp and
    proxy quat≈tcp_quat (quat sign ambiguity: compare via rotation distance).

### Viewer

`example_view_pre_grasp_settle.py` continues to call the same apply helper; update
docstrings/help that currently say “look-at weld (+Z ∥ ŵ)” to “true TCP pose weld”.

## Warnings / errors

| Condition | Action |
| --------- | ------ |
| Missing `post_grasp_geometry` / poses | Hard fail (unchanged) |
| Zero apple→TCP chord | Hard fail (unchanged) |
| `\|d − r\|` or apple shift > warn tol | Warn, continue (unchanged) |
| TCP +Z poorly aligned with ŵ | **No longer** “ignore quat”; do not change orientation |
| Welded proxy pos vs `tcp_pos` | Warn if beyond tol (unchanged intent) |
| Welded proxy quat vs `tcp_quat` | Warn if rotation distance large |

## Testing

- Unit: offset from known apple/TCP SE(3) round-trips to world TCP pose.
- `apply_post_grasp_after_settle`: proxy pos **and** quat match plan (not look-at).
- Plan tests: misaligned TCP +Z no longer expects “ignore quat” / “poorly aligned”
  as the contract for orientation discard; update `s00-d00` smoke accordingly.
- Existing look-at / `weld_direction` tests for non–post-grasp paths stay green.

## Success criteria

- After `--grasp-after-settle`, proxy world pose equals logged TCP SE(3).
- Apple world pose equals measured post-grasp apple SE(3) (no \(r\)-sphere snap).
- Woody not teleported from post–long-settle.
- Settle-only path unchanged without the flag.
- No warning claiming logged TCP quat is ignored or that catalog-surface correction moved the apple.

## Spec self-review

- No TBDs; orientation and snap roles are explicit.
- Consistent with approach A chosen by user (full TCP SE(3) + keep surface snap).
- Scope limited to post-grasp replay weld + config hook; no FR3 / woody teleport.
