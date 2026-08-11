# Batched real replay post-grasp SE(3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Match `example_replay_real_batched` apple lifecycle to settle viewer (pre-grasp init → logged post-grasp SE(3) at weld); ROADMAP M4.0/CMA lists shared-path (B) follow-up.

**Architecture:** Extend episode-metadata gripper with true TCP apple-frame offset; after env settle→weld, apply logged apple SE(3) + realign proxy. Helpers in `batched_digital_twin_init` for reuse by CMA seed later.

**Tech Stack:** Python, Warp transforms, existing `proxy_offset_from_apple_and_tcp` / settle seed sync.

## Global Constraints

- Slice A: `example_replay_real_batched.py` only; no CMA/MMD behavior change yet
- No `--apple-position-only` on this example
- Pre-grasp `params.apple_quat_xyzw` unchanged for free settle
- ROADMAP M4.0 checklist must include example (A) + shared CMA seed (B)

---

### Task 1: Gripper + apply helpers (TDD)

**Files:**
- Modify: `apple_pick_sim/system_id/batched_digital_twin_init.py`
- Modify: `apple_pick_sim/tests/test_batched_digital_twin_init.py`

- [x] **Step 1:** Write failing tests for `gripper_proxy_for_real_batched_replay` (sets `weld_proxy_offset_in_apple_frame`) and `apply_logged_post_grasp_se3_to_cable` (apple/TCP match meta)
- [x] **Step 2:** Run tests — expect fail
- [x] **Step 3:** Implement helpers
- [x] **Step 4:** Run tests — expect pass

### Task 2: Wire example

**Files:**
- Modify: `robot_replay/example_replay_real_batched.py`

- [x] **Step 1:** Closure over episode meta; override gripper; call apply after env construct
- [x] **Step 2:** Smoke import / unit path if any

### Task 3: ROADMAP + spec status

**Files:**
- Modify: `docs/ROADMAP.md`
- Modify: `docs/superpowers/specs/2026-08-11-batched-real-replay-post-grasp-se3-design.md`

- [x] Checklist: example post-grasp SE(3); CMA/shared `seed_fix` apply (B)
- [x] Spec status → Implemented (A) / B deferred
