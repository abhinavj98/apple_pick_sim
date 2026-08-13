# GL Video Record Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record Newton GL framebuffer frames during real batched replay into an MP4 via `--record-video`.

**Architecture:** Reusable `GlVideoRecorder` in `robot_replay/gl_video_recorder.py` streams RGB from `viewer.get_frame()` through `imageio_ffmpeg`. Wire into `example_replay_real_batched.make_replay_on_step` after `end_frame`; FPS = sim `control_hz`.

**Tech Stack:** Python, NumPy, Warp (`get_frame`), `imageio-ffmpeg`, pytest.

## Global Constraints

- Requires `ViewerGL` (`--viewer gl`, including `--headless`); not `--viewer null`
- FPS = `control_hz` (resolved on first capture from env runtime / fallback)
- Optional dep: `imageio-ffmpeg` under `gym` extra
- Trajectory frames only (same hook as current GL render); no settle-phase recording
- Spec: `docs/superpowers/specs/2026-08-12-gl-video-record-design.md`

---

### Task 1: `GlVideoRecorder` (TDD)

**Files:**
- Create: `robot_replay/gl_video_recorder.py`
- Create: `robot_replay/tests/test_gl_video_recorder.py`
- Modify: `pyproject.toml` (add `imageio-ffmpeg` to `gym` extra)

**Interfaces:**
- Produces: `GlVideoRecorder(path: Path, *, fps: float | None = None)` with `set_fps(fps)`, `capture(viewer)`, `close()`, `frame_count: int`, context manager

- [x] **Step 1: Write failing tests**
- [x] **Step 2: Run tests — expect fail** (import / missing module)
- [x] **Step 3: Implement `GlVideoRecorder` + add dep**
- [x] **Step 4: Run tests — expect pass**; `uv sync --extra gym`
- [x] **Step 5: Commit** `feat: add GlVideoRecorder for GL MP4 capture` — deferred (user did not request commit)

---

### Task 2: Wire `--record-video` into example (TDD)

**Files:**
- Modify: `robot_replay/example_replay_real_batched.py`
- Modify: `apple_pick_gym/tests/test_real_batched_replay_cli.py`

**Interfaces:**
- Consumes: `GlVideoRecorder`
- Produces: `make_replay_on_step(..., recorder=None)`; CLI `--record-video`

- [x] **Step 1: Failing tests** in `test_real_batched_replay_cli.py`
- [x] **Step 2: Run — expect fail**
- [x] **Step 3: Wire example**
- [x] **Step 4: Run CLI unit tests — pass**
- [x] **Step 5: Commit** — deferred (user did not request commit)

---

### Task 3: Docs + spec status

**Files:**
- Modify: `docs/superpowers/specs/2026-08-12-gl-video-record-design.md` (Status → Implemented)
- Modify: example module docstring with `--record-video` usage line
- Optionally one README line near robot_replay if that section exists

- [x] Update docstring example; mark spec Implemented; commit deferred
