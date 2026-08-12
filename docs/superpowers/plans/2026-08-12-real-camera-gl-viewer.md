# Real camera GL viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist `camera_to_base_4x4` in converted episode metadata and apply it once to the Newton GL viewer in real batched replay.

**Architecture:** Extract the extrinsic in `build_episode_metadata_from_real`; convert SE(3)→`(pos, pitch, yaw)` with a pure helper in the replay example; call `set_camera` on first viewer init.

**Tech Stack:** Python, NumPy, PyArrow parquet schema metadata, Newton `viewer.set_camera`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-12-real-camera-gl-viewer-design.md`

## Global Constraints

- Patch current branch (`main`) — user override of worktree rule for this task.
- No new CLI flags.
- OpenCV optical axis = camera **+Z** in base.
- Omit key if extrinsic missing; never fail convert/replay for that alone.
- Do not commit unless the user asks.

---

### Task 1: Copy `camera_to_base_4x4` into episode metadata

**Files:**
- Modify: `apple_pick_sim/system_id/real_to_batched_sysid.py`
- Test: `apple_pick_sim/tests/test_real_to_batched_sysid.py`
- Modify helper: `_write_synthetic_real` to accept optional `camera_to_base_4x4`

**Interfaces:**
- Produces: `camera_to_base_4x4_from_dataset_metadata(dm: dict) -> list[list[float]] | None`
- Produces: `build_episode_metadata_from_real(...)` may include `"camera_to_base_4x4"`

- [ ] **Step 1: Write failing tests**

```python
def test_camera_to_base_4x4_from_dataset_metadata_prefers_used():
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        camera_to_base_4x4_from_dataset_metadata,
    )
    used = [[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, 0.2], [0.0, 0.0, 1.0, 0.3], [0.0, 0.0, 0.0, 1.0]]
    snap = [[1.0, 0.0, 0.0, 9.0], [0.0, 1.0, 0.0, 9.0], [0.0, 0.0, 1.0, 9.0], [0.0, 0.0, 0.0, 1.0]]
    got = camera_to_base_4x4_from_dataset_metadata(
        {"camera_to_base_4x4_used": used, "pre_grasp_geometry": {"settled_snapshot": {"camera_to_base_4x4": snap}}}
    )
    assert got == used

def test_build_episode_metadata_copies_camera_to_base_4x4(tmp_path: Path):
    path = tmp_path / "real.parquet"
    T = [[1.0, 0.0, 0.0, -0.3], [0.0, 1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.4], [0.0, 0.0, 0.0, 1.0]]
    _write_synthetic_real(path, camera_to_base_4x4=T)
    meta = build_episode_metadata_from_real(path, fixture_path=VARIANCE)
    assert meta["camera_to_base_4x4"] == T
```

- [ ] **Step 2: Run tests — expect FAIL** (`camera_to_base_4x4_from_dataset_metadata` missing / key absent)

```bash
uv run --env-file pytest.env pytest apple_pick_sim/tests/test_real_to_batched_sysid.py::test_camera_to_base_4x4_from_dataset_metadata_prefers_used apple_pick_sim/tests/test_real_to_batched_sysid.py::test_build_episode_metadata_copies_camera_to_base_4x4 -q
```

- [ ] **Step 3: Implement extractor + wire into `build_episode_metadata_from_real`**

```python
def camera_to_base_4x4_from_dataset_metadata(dm: dict[str, Any]) -> list[list[float]] | None:
    used = dm.get("camera_to_base_4x4_used")
    parsed = _as_4x4(used)
    if parsed is not None:
        return parsed
    pre = dm.get("pre_grasp_geometry")
    if isinstance(pre, dict):
        for snap in pre.values():
            if isinstance(snap, dict):
                parsed = _as_4x4(snap.get("camera_to_base_4x4"))
                if parsed is not None:
                    return parsed
    return None
```

Extend `_write_synthetic_real(..., camera_to_base_4x4=None)` to set `dataset_metadata["camera_to_base_4x4_used"]`.

In `build_episode_metadata_from_real`, after building `meta`:

```python
cam = camera_to_base_4x4_from_dataset_metadata(dm)
if cam is not None:
    meta["camera_to_base_4x4"] = cam
```

- [ ] **Step 4: Run tests — expect PASS**

---

### Task 2: SE(3) → GL `set_camera` args + wire replay

**Files:**
- Modify: `robot_replay/example_replay_real_batched.py`
- Test: `apple_pick_gym/tests/test_real_batched_replay_cli.py`

**Interfaces:**
- Consumes: `camera_to_base_4x4` nested 4×4
- Produces: `gl_camera_from_camera_to_base(T) -> tuple[tuple[float,float,float], float, float] | None`
- Produces: `make_replay_on_step(..., camera_to_base_4x4=None)` calls `set_camera` once on init

- [ ] **Step 1: Write failing tests**

```python
def test_gl_camera_from_camera_to_base_looks_along_plus_z():
    mod = _load_replay()
    # Camera at origin, +Z along +X → yaw=0, pitch=0
    T = [[0, 0, 1, 0.1], [1, 0, 0, 0.2], [0, 1, 0, 0.3], [0, 0, 0, 1]]
    # Wait: columns are camera axes in base. +Z column = third column of R = (1,0,0) if row0=[0,0,1]?
    # Row-major nested: T[i][j] = row i. Optical +Z = (T[0][2], T[1][2], T[2][2]).
    T = [
        [0.0, 0.0, 1.0, 0.1],
        [0.0, 1.0, 0.0, 0.2],
        [-1.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 1.0],
    ]
    # +Z = (1, 0, 0) → look +X, pitch=0, yaw=0
    pos, pitch, yaw = mod.gl_camera_from_camera_to_base(T)
    assert pos == pytest.approx((0.1, 0.2, 0.3))
    assert pitch == pytest.approx(0.0, abs=1e-6)
    assert yaw == pytest.approx(0.0, abs=1e-6)

def test_make_replay_on_step_sets_camera_from_extrinsic_once():
    mod = _load_replay()
    calls = []
    class Viewer:
        def set_model(self, model): calls.append(("set_model", model))
        def hide_loading_splash(self): pass
        def set_camera(self, pos, pitch, yaw): calls.append(("set_camera", tuple(pos), pitch, yaw))
        def begin_frame(self, t): pass
        def log_state(self, state): pass
        def end_frame(self): pass
        def is_running(self): return True
    # ... env with cable as existing test ...
    T = [[1,0,0,-0.3],[0,1,0,0.5],[0,0,1,0.4],[0,0,0,1]]  # +Z = +Z world → pitch=90
    on_step = mod.make_replay_on_step(Viewer(), max_frames=0, camera_to_base_4x4=T)
    on_step(frame_idx=0, env=env)
    on_step(frame_idx=1, env=env)
    cam_calls = [c for c in calls if c[0] == "set_camera"]
    assert len(cam_calls) == 1
    assert cam_calls[0][1] == pytest.approx((-0.3, 0.5, 0.4))
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
uv run --env-file pytest.env pytest apple_pick_gym/tests/test_real_batched_replay_cli.py::test_gl_camera_from_camera_to_base_looks_along_plus_z apple_pick_gym/tests/test_real_batched_replay_cli.py::test_make_replay_on_step_sets_camera_from_extrinsic_once -q
```

- [ ] **Step 3: Implement helpers + wire `make_replay_on_step` / `_run`**

```python
def gl_camera_from_camera_to_base(T) -> tuple[tuple[float, float, float], float, float] | None:
    arr = np.asarray(T, dtype=np.float64)
    if arr.shape != (4, 4):
        return None
    pos = (float(arr[0, 3]), float(arr[1, 3]), float(arr[2, 3]))
    front = arr[:3, 2]
    n = float(np.linalg.norm(front))
    if n < 1e-12:
        return None
    d = front / n
    pitch = float(np.rad2deg(np.arcsin(np.clip(d[2], -1.0, 1.0))))
    yaw = float(np.rad2deg(np.arctan2(d[1], d[0])))
    return pos, pitch, yaw
```

On first init, if `camera_to_base_4x4` and `hasattr(viewer, "set_camera")`:

```python
pose = gl_camera_from_camera_to_base(camera_to_base_4x4)
if pose is not None:
    pos, pitch, yaw = pose
    viewer.set_camera(wp.vec3(*pos), pitch, yaw)
```

Pass `episode_meta.get("camera_to_base_4x4")` from `_run` into `make_replay_on_step`.

- [ ] **Step 4: Run both test files for changed cases — expect PASS**

```bash
uv run --env-file pytest.env pytest apple_pick_sim/tests/test_real_to_batched_sysid.py::test_camera_to_base_4x4_from_dataset_metadata_prefers_used apple_pick_sim/tests/test_real_to_batched_sysid.py::test_build_episode_metadata_copies_camera_to_base_4x4 apple_pick_gym/tests/test_real_batched_replay_cli.py::test_gl_camera_from_camera_to_base_looks_along_plus_z apple_pick_gym/tests/test_real_batched_replay_cli.py::test_make_replay_on_step_sets_camera_from_extrinsic_once apple_pick_gym/tests/test_real_batched_replay_cli.py::test_make_replay_on_step_renders_and_stops_at_max_frames -q
```

---

## Spec coverage

| Spec requirement | Task |
| ---------------- | ---- |
| Copy `camera_to_base_4x4_used` / fallback snapshots | Task 1 |
| Omit if missing | Task 1 |
| GL set from episode meta, +Z look, once | Task 2 |
| No new CLI | Task 2 |
