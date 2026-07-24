# Real Post-Grasp Settle → Grasp Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the pre-grasp settle viewer so that after a long free settle it applies a TCP-anchored post-grasp snap (apple on catalog surface, proxy at measured TCP pose), runs a short settle, and keeps simulating — proxy-only for this slice; FR3 deferred.

**Architecture:** Pure CPU `PostGraspPlan` from parquet `post_grasp_geometry` + catalog `r`. After free settle, rebuild a welded `CoupledCableScene` with an **explicit FIXED-joint offset** (apple→TCP relative pose, not look-at-only), seed woody/`body_q` from the free settle (apple overridden), quiet, then short settle in the same viewer loop.

**Tech Stack:** Python, NumPy, PyArrow, Newton/Warp, `GripperProxyConfig`, `uv`, pytest.

**Spec:** `docs/superpowers/specs/2026-07-24-real-post-grasp-viewer-design.md`

## Global Constraints

- Slice A only in this plan: **no FR3**, no trajectory/`action` replay, no CMA-ES dataset writer.
- Weld direction: \(\hat{w}=\mathrm{normalize}(\mathbf{p}_{\mathrm{tcp}}-\mathbf{p}_{\mathrm{apple}}^{\mathrm{meas}})\) (apple → TCP).
- Apple snap: quat from `apple_pose_4x4`; pos \(\mathbf{p}_{\mathrm{apple}}^{\mathrm{welded}}=\mathbf{p}_{\mathrm{tcp}}-r\hat{w}\).
- Proxy: measured `tcp_pose_4x4` (pos + quat). Invariant \(\mathbf{p}_{\mathrm{apple}}^{\mathrm{welded}}+r\hat{w}=\mathbf{p}_{\mathrm{tcp}}\).
- Woody bodies: keep post–long-settle poses (do not teleport to post-grasp woody).
- Warn (continue) if \(\big||\mathbf{p}_{\mathrm{tcp}}-\mathbf{p}_{\mathrm{apple}}^{\mathrm{meas}}|-r\big|>0.02\) or \(|\mathbf{p}_{\mathrm{apple}}^{\mathrm{welded}}-\mathbf{p}_{\mathrm{apple}}^{\mathrm{meas}}|>0.02\); same tol configurable via `--tcp-radius-warn-m`.
- Hard-fail on missing post-grasp fields / zero-length TCP−apple when `--grasp-after-settle`.
- TDD; run tests with `uv run --env-file pytest.env python -m pytest …` from repo root.
- Work on current feature branch / worktree as directed by the user.

---

## File map

| File | Responsibility |
|------|----------------|
| `apple_pick_sim/system_id/real_post_grasp_plan.py` | Parse post-grasp; plan math; warnings; relative apple→TCP offset; apply/rebuild helper |
| `apple_pick_sim/tests/test_real_post_grasp_plan.py` | Unit tests for plan + offset + warnings |
| `apple_pick_sim/fruiting_system/params.py` | Optional `weld_fixed_offset_in_apple_frame` on `GripperProxyConfig` |
| `apple_pick_sim/fruiting_system/build.py` | Use explicit FIXED offset when provided (skip look-at) |
| `apple_pick_sim/tests/test_fruiting_system.py` (or focused new test) | Build smoke: explicit offset places proxy at TCP |
| `robot_replay/example_view_pre_grasp_settle.py` | CLI flags + phase machine (long settle → grasp → short settle) |
| `robot_replay/README.md` | Document `--grasp-after-settle` |
| `docs/real-sysid-pre-post-grasp-fixes.md` | Note TCP↔radius contract + warn |
| `docs/superpowers/specs/2026-07-24-real-post-grasp-viewer-design.md` | Status → Implemented (slice A) when done |

**Deferred (not in this plan’s coding tasks):** Slice B FR3 (`--robot fr3`). Spec already describes it; add only a short “Next” note in README.

---

### Task 1: `PostGraspPlan` math + warnings (TDD)

**Files:**
- Create: `apple_pick_sim/system_id/real_post_grasp_plan.py`
- Create: `apple_pick_sim/tests/test_real_post_grasp_plan.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class PostGraspPlan` with fields:
    - `tcp_pos: tuple[float,float,float]`
    - `tcp_quat_xyzw: tuple[float,float,float,float]`
    - `apple_pos_measured: tuple[float,float,float]`
    - `apple_quat_xyzw: tuple[float,float,float,float]`
    - `apple_pos_welded: tuple[float,float,float]`
    - `weld_direction: tuple[float,float,float]`  # unit, apple→TCP
    - `apple_radius_m: float`
    - `tcp_apple_distance_m: float`
    - `tcp_radius_residual_m: float`  # `|d - r|`
    - `apple_shift_m: float`  # `|apple_welded - apple_meas|`
    - `proxy_offset_in_apple_frame: tuple[float,...]`  # length-7 pos+quat xyzw of TCP in apple frame
  - `pose_4x4_to_pos_quat(flat16) -> tuple[pos3, quat_xyzw]`
  - `build_post_grasp_plan(*, tcp_pose_4x4, apple_pose_4x4, apple_radius_m, warn_tol_m=0.02, emit_warnings=True) -> PostGraspPlan`
  - `post_grasp_plan_from_metadata(meta, *, apple_radius_m, warn_tol_m=0.02) -> PostGraspPlan`
  - `relative_pose_child_in_parent(parent_pos, parent_quat_xyzw, child_pos, child_quat_xyzw) -> tuple[7]`

- [ ] **Step 1: Write failing tests**

```python
# apple_pick_sim/tests/test_real_post_grasp_plan.py
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from apple_pick_sim.system_id.real_post_grasp_plan import (
    build_post_grasp_plan,
    pose_4x4_to_pos_quat,
    relative_pose_child_in_parent,
)


def _pose4(pos, R=None):
    if R is None:
        R = np.eye(3)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = pos
    return M.reshape(16).tolist()


def test_pose_4x4_to_pos_quat_identity():
    pos, quat = pose_4x4_to_pos_quat(_pose4([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(pos, (1.0, 2.0, 3.0))
    np.testing.assert_allclose(quat, (0.0, 0.0, 0.0, 1.0), atol=1e-6)


def test_build_plan_forces_surface_and_invariant():
    tcp = np.array([0.0, 0.1, 0.0])
    apple_m = np.array([0.0, 0.0, 0.0])  # d=0.1, r=0.04 → residual 0.06
    r = 0.04
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        plan = build_post_grasp_plan(
            tcp_pose_4x4=_pose4(tcp.tolist()),
            apple_pose_4x4=_pose4(apple_m.tolist()),
            apple_radius_m=r,
            warn_tol_m=0.02,
        )
    assert plan.tcp_apple_distance_m == pytest.approx(0.1)
    assert plan.tcp_radius_residual_m == pytest.approx(0.06)
    assert plan.apple_shift_m == pytest.approx(0.06)
    np.testing.assert_allclose(plan.weld_direction, (0.0, 1.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(
        plan.apple_pos_welded, (0.0, 0.06, 0.0), atol=1e-6
    )  # tcp - r * +Y
    # invariant
    aw = np.asarray(plan.apple_pos_welded)
    wdir = np.asarray(plan.weld_direction)
    np.testing.assert_allclose(aw + r * wdir, tcp, atol=1e-6)
    assert any("tcp" in str(x.message).lower() or "radius" in str(x.message).lower() for x in w)


def test_no_warn_when_within_tol():
    tcp = [0.0, 0.041, 0.0]
    apple = [0.0, 0.0, 0.0]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        build_post_grasp_plan(
            tcp_pose_4x4=_pose4(tcp),
            apple_pose_4x4=_pose4(apple),
            apple_radius_m=0.04,
            warn_tol_m=0.02,
        )
    assert w == []


def test_relative_pose_translation_only():
    off = relative_pose_child_in_parent(
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.04, 0.0), (0.0, 0.0, 0.0, 1.0),
    )
    np.testing.assert_allclose(off[:3], (0.0, 0.04, 0.0), atol=1e-6)
```

- [ ] **Step 2: Run tests — expect FAIL (import / missing)**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_real_post_grasp_plan.py -q
```

- [ ] **Step 3: Implement `real_post_grasp_plan.py` (plan only; no scene apply yet)**

Include:
- Row-major `pose_4x4` → pos + quat `(x,y,z,w)` via orthonormalized rotation matrix (same layout as episode `field_layout`: translation at indices 3,7,11).
- `build_post_grasp_plan`: compute \(\hat{w}\), welded apple, residuals; `warnings.warn` when over tol; set `proxy_offset_in_apple_frame = relative_pose_child_in_parent(apple_welded, apple_quat, tcp_pos, tcp_quat)`.
- `post_grasp_plan_from_metadata`: read `meta["post_grasp_geometry"]`; require `tcp_pose_4x4` + `apple_pose_4x4`; optionally assert `tcp_pos` / `apple_pos` match pose translations within 1e-5 or warn.
- Raise `ValueError` on missing keys or zero-length chord.

- [ ] **Step 4: Tests PASS**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_real_post_grasp_plan.py -q
```

- [ ] **Step 5: Commit**

```bash
git add apple_pick_sim/system_id/real_post_grasp_plan.py \
  apple_pick_sim/tests/test_real_post_grasp_plan.py
git commit -m "Add PostGraspPlan math and TCP–radius warn helpers."
```

---

### Task 2: Explicit FIXED-joint offset on `GripperProxyConfig`

**Why:** Default weld build uses look-at rotation along \(\hat{w}\). Spec requires **TCP quat**. Provide an optional 7D apple-frame offset so the FIXED joint matches measured relative TCP pose.

**Files:**
- Modify: `apple_pick_sim/fruiting_system/params.py` (`GripperProxyConfig`)
- Modify: `apple_pick_sim/fruiting_system/build.py` (`_add_gripper_proxy`)
- Test: extend `apple_pick_sim/tests/test_fruiting_system.py` or add `apple_pick_sim/tests/test_gripper_proxy_fixed_offset.py`

**Interfaces:**
- Consumes: `PostGraspPlan.proxy_offset_in_apple_frame`
- Produces: `GripperProxyConfig.weld_fixed_offset_in_apple_frame: tuple[float, ...] | None = None`
  - When set with `fix_to_apple=True`, build uses this as `parent_xform` (pos+quat) and skips look-at / random weld orientation.
  - Still set `weld_reference_pos` / `weld_reference_quat` / `weld_direction` for diagnostics and hemisphere checks as needed; if explicit offset is set, **do not** re-derive orientation from `weld_direction`.

- [ ] **Step 1: Failing test — proxy world pose ≈ TCP when apple at welded pose**

Build a minimal coupled cable scene with known params, `fix_to_apple=True`, explicit offset from a synthetic plan, assign apple `body_q` to welded pose, run one FK/sync if required, assert proxy translation ≈ TCP (atol 1e-3). Keep test under existing GPU fixtures used by fruiting_system tests.

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement config field + `_add_gripper_proxy` branch**

When `config.weld_fixed_offset_in_apple_frame is not None`:
- Validate length 7 and `fix_to_apple`
- `parent_xform = wp.transform(pos, quat)` from the 7-tuple
- `child_xform = identity`
- `proxy_offset_in_apple_frame =` that tuple
- Place initial `proxy_pos` consistently with `weld_reference_pos` + offset (world = apple_ref ∘ offset)

- [ ] **Step 4: Test PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "Allow explicit apple-frame FIXED offset for TCP-quat welds."
```

---

### Task 3: Apply grasp after free settle (`CoupledCableScene`)

**Files:**
- Modify: `apple_pick_sim/system_id/real_post_grasp_plan.py`
- Modify: `apple_pick_sim/tests/test_real_post_grasp_plan.py` (GPU/scene test marked like other coupled tests)

**Interfaces:**
- Produces:
  ```python
  def apply_post_grasp_after_settle(
      free_scene: CoupledCableScene,
      plan: PostGraspPlan,
      *,
      ranges: dict,
      params: FruitingSystemParams,
      base_pos: tuple[float, float, float],
      device: str | None,
      robot_base_pos: tuple[float, float, float] | None,
  ) -> CoupledCableScene:
      """Rebuild welded scene; seed woody from free settle; apple+proxy from plan."""
  ```

**Algorithm (normative):**
1. `bq = free_scene.state_0.body_q.numpy().reshape(-1, 7).copy()`
2. `apple_id = free_scene.apple_body` — set `bq[apple_id] = [*plan.apple_pos_welded, *plan.apple_quat_xyzw]`
3. Build `welded = generate_coupled_cable_scene(ranges, seed=0, params=params, base_pos=base_pos, device=device, robot_base_pos=robot_base_pos, gripper_proxy=GripperProxyConfig(fix_to_apple=True, weld_direction=plan.weld_direction, weld_reference_pos=plan.apple_pos_welded, weld_reference_quat=plan.apple_quat_xyzw, weld_fixed_offset_in_apple_frame=plan.proxy_offset_in_apple_frame))`
4. Copy `bq` into welded `state_0`/`state_1` for all bodies that exist in both models with matching indices (assert `apple_body` / woody counts match; **do not** copy free proxy pose blindly — set proxy row to `[*plan.tcp_pos, *plan.tcp_quat_xyzw]`).
5. `quiet_all_cable_bodies(welded)` + sync `body_q_prev` (existing helpers).
6. Return `welded`.

- [ ] **Step 1: Write failing integration test** (reuse fixture ranges + small settle or direct body_q) asserting after apply: apple pos ≈ welded, proxy pos ≈ tcp, woody tip unchanged vs pre-apply snapshot within atol.

- [ ] **Step 2: Implement `apply_post_grasp_after_settle`**

- [ ] **Step 3: Test PASS + commit**

```bash
git commit -m "Apply post-grasp weld rebuild after free cable settle."
```

---

### Task 4: Wire CLI phases in `example_view_pre_grasp_settle.py`

**Files:**
- Modify: `robot_replay/example_view_pre_grasp_settle.py`
- Modify: `robot_replay/README.md`
- Modify: `docs/real-sysid-pre-post-grasp-fixes.md` (short TCP↔r note)

**Interfaces:**
- New flags (defaults from spec):
  - `--grasp-after-settle` (store_true)
  - `--post-grasp-settle-substeps` default `500`
  - `--tcp-radius-warn-m` default `0.02`

**Phase machine in the example class:**
- Fields: `_phase: Literal["long_settle","grasp_pending","short_settle","run"]`, counters for remaining long/short substeps.
- On long settle complete, if grasp enabled: compute plan from metadata + `params.apple_radius` (must match parts radius used at build); print plan summary; `self._scene = apply_post_grasp_after_settle(...)`; rebind `model/state/solver/viewer.set_model`; set short settle remaining; else go to `run`.
- Short settle uses same quiet_every logic as long settle.
- Without `--grasp-after-settle`, behavior identical to today.

- [ ] **Step 1: Implement flags + phase transitions**

- [ ] **Step 2: Headless smoke**

```bash
uv run python robot_replay/example_view_pre_grasp_settle.py \
  --parquet robot_replay/s00-d00.parquet \
  --grasp-after-settle \
  --settle-substeps 80 \
  --post-grasp-settle-substeps 40 \
  --settle-quiet-every 20 \
  --viewer null --num-frames 8
```

Expected: exits 0; stderr/stdout includes TCP–radius warning (~18 mm); “Settle complete” / grasp / short settle messages.

- [ ] **Step 3: Update README + fix-doc one-liner; mark spec status Implemented (slice A)**

- [ ] **Step 4: Commit**

```bash
git commit -m "Add --grasp-after-settle phase to pre-grasp viewer."
```

---

## Spec coverage check

| Spec requirement | Task |
| ---------------- | ---- |
| Grasp plan math apple→TCP, welded apple | Task 1 |
| Warn \|d−r\| and apple shift @ 0.02 m | Task 1 |
| Proxy at tcp pose (pos+quat); surface invariant | Tasks 1–3 |
| Woody unchanged; apple-only snap | Task 3 |
| Long settle → grasp → short settle; one CLI | Task 4 |
| Default settle-only unchanged | Task 4 |
| Slice B FR3 | Deferred (README “Next”) |
| Unit tests without full viewer for plan | Task 1 |
| Headless smoke | Task 4 |

---

## Plan self-review (completed while writing)

1. **Spec coverage:** All slice-A behaviors mapped; FR3 explicitly deferred.
2. **Placeholder scan:** No TBD steps; open build mechanism resolved as **rebuild + explicit FIXED offset**.
3. **Type consistency:** `PostGraspPlan.proxy_offset_in_apple_frame` ↔ `GripperProxyConfig.weld_fixed_offset_in_apple_frame` ↔ apply helper.
4. **Risk called out:** Look-at-only weld would violate TCP quat — Task 2 is required, not optional.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-24-real-post-grasp-viewer.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks
2. **Inline Execution** — execute tasks in this session with checkpoints

Which approach?
