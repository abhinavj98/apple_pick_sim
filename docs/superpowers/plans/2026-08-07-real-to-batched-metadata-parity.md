# Real → batched metadata parity (bit 1) Implementation Plan

> **For agentic workers:** Execute inline in this session (user requested implement). TDD required.

**Goal:** Make `build_episode_metadata_from_real` emit batched-style metadata that numerically matches the settle-viewer native pre/post builders.

**Architecture:** Thin adapter over `fruiting_params_from_pre_grasp_parquet` + `post_grasp_plan_from_metadata`; pytest parity on `s00-d00`; small JSON viewer script.

**Tech Stack:** Python, pyarrow parquet, existing `apple_pick_sim.system_id` APIs, pytest, uv.

## Global Constraints

- Source of truth: `example_view_pre_grasp_settle.py` native stack
- Metadata JSON only (no trajectory)
- Radii from `pre_grasp_geometry.parts` via native params
- Do not require `rod_geometry`, `step_idx==-1`, `tcp_quat`, `robot_joint_q`

---

### Task 1: Parity test (RED → GREEN converter)

**Files:**
- Modify: `apple_pick_sim/tests/test_real_to_batched_sysid.py`
- Modify: `apple_pick_sim/system_id/real_to_batched_sysid.py`

- [ ] Write `test_s00_d00_convert_matches_native_pre_post`
- [ ] Confirm RED (missing columns / rod_geometry)
- [ ] Rewrite `build_episode_metadata_from_real` to shared builders
- [ ] Confirm GREEN + existing helper tests still pass

### Task 2: Batched-metadata viewer + docs

**Files:**
- Create: `robot_replay/example_view_batched_episode_meta.py`
- Modify: `robot_replay/README.md`
- Modify: `docs/handbook-real-replay.md` (C1 note)

- [ ] Viewer loads converted JSON, settle + optional grasp
- [ ] Document convert + parity commands
- [ ] Smoke convert CLI on `s00-d00`
