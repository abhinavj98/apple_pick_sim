# True TCP pose post-grasp weld

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement task-by-task. Spec: [docs/superpowers/specs/2026-08-04-true-tcp-pose-weld-design.md](../specs/2026-08-04-true-tcp-pose-weld-design.md).

**Goal:** After `--grasp-after-settle`, welded proxy world pose equals logged TCP SE(3); look-at (+Z ∥ ŵ) is not used for this path.

**Architecture:** Add optional explicit apple-frame FIXED offset on `GripperProxyConfig`. Post-grasp apply computes \(X_{\mathrm{offset}}=X_{\mathrm{apple}}^{-1} X_{\mathrm{tcp}}\) from the plan and builds/seeds with that offset so constraint and state agree. Keep surface-snap apple position; warn on residuals without “fixing” orientation.

**Tech stack:** Newton/Warp FIXED joint, existing `real_post_grasp_plan` + coupled scene rebuild, pytest + `uv run`.

**Status:** Implemented 2026-08-04.

## Task 1 — Explicit FIXED offset in build

- [x] RED/GREEN: `weld_proxy_offset_in_apple_frame` on `GripperProxyConfig`; `_add_gripper_proxy` honors it
- [x] Tests in `test_coupled_cable_scene.py`

## Task 2 — Post-grasp plan uses true TCP SE(3)

- [x] `apply_post_grasp_after_settle` uses explicit offset; soft +Z diagnostic
- [x] Tests assert proxy quat matches logged TCP; viewer copy updated

## Task 3 — Verify + docs

- [x] `pytest apple_pick_sim/tests/test_real_post_grasp_plan.py`
- [x] Headless smoke `--grasp-after-settle`
- [x] Design marked Implemented
