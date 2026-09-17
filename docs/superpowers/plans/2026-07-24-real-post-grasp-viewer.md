# Real Post-Grasp Settle → Grasp Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the pre-grasp settle viewer so that after a long free settle it applies a TCP-anchored post-grasp snap (apple on catalog surface; proxy at TCP position with **+Z ∥ weld direction**), runs a short settle, and keeps simulating — proxy-only for this slice; FR3 deferred.

**Architecture:** Pure CPU `PostGraspPlan` from parquet `post_grasp_geometry` + catalog `r`. After free settle, rebuild a welded `CoupledCableScene` with stock `GripperProxyConfig(fix_to_apple=True, weld_direction=ŵ, weld_reference_*)` (**look-at**, tool +Z along \(\hat{w}\)), seed woody/`body_q` from the free settle (apple overridden), quiet, then short settle.

**Tech Stack:** Python, NumPy, PyArrow, Newton/Warp, `GripperProxyConfig`, `uv`, pytest.

**Spec:** `docs/superpowers/specs/2026-07-24-real-post-grasp-viewer-design.md`

## Global Constraints

- Slice A only: **no FR3**, no trajectory replay, no CMA-ES writer.
- \(\hat{w}=\mathrm{normalize}(\mathbf{p}_{\mathrm{tcp}}-\mathbf{p}_{\mathrm{apple}}^{\mathrm{meas}})\).
- \(\mathbf{p}_{\mathrm{apple}}^{\mathrm{welded}}=\mathbf{p}_{\mathrm{tcp}}-r\hat{w}\); apple quat from `apple_pose_4x4`.
- Proxy **position** = TCP position (surface invariant). Proxy **orientation** = look-at, **+Z ∥ \(\hat{w}\)** — **do not** bake logged TCP quat into the FIXED joint (parquet rotation may be wrong).
- Woody: keep post–long-settle poses.
- Warn if \(\big||d|-r\big|>0.02\) or apple shift \(>0.02\); optional warn if \(|\hat{w}\cdot(+Z_{\mathrm{tcp}})|\`< 0.9.
- Hard-fail on missing post-grasp / zero chord when grasping.
- TDD; `uv run --env-file pytest.env python -m pytest …`.

---

## File map

| File | Responsibility |
|------|----------------|
| `apple_pick_sim/system_id/real_post_grasp_plan.py` | Plan math, warnings, apply/rebuild helper |
| `apple_pick_sim/tests/test_real_post_grasp_plan.py` | Unit + apply tests |
| `robot_replay/example_view_pre_grasp_settle.py` | CLI phases |
| `robot_replay/README.md` | Document flags |
| `docs/handbook-real-replay.md` | TCP↔r and +Z∥ŵ contract notes |
| Spec | Mark Implemented (slice A) when done |

**Not in this plan:** `weld_fixed_offset_in_apple_frame` / build.py look-at bypass (rejected). Slice B FR3 deferred (seed `joint_pos` from grasp table row when implemented).

---

### Task 1: `PostGraspPlan` math + warnings (TDD)

**Files:**
- Create: `apple_pick_sim/system_id/real_post_grasp_plan.py`
- Create: `apple_pick_sim/tests/test_real_post_grasp_plan.py`

**Interfaces:**
- `PostGraspPlan` frozen dataclass:
  - `tcp_pos`, `tcp_quat_xyzw` (parsed for diagnostics only)
  - `apple_pos_measured`, `apple_quat_xyzw`
  - `apple_pos_welded`, `weld_direction`
  - `apple_radius_m`, `tcp_apple_distance_m`, `tcp_radius_residual_m`, `apple_shift_m`
  - `tcp_approach_dot_weld: float`  # \(\hat{w}\cdot(+Z_{\mathrm{tcp}})\)
- `pose_4x4_to_pos_quat`, `build_post_grasp_plan`, `post_grasp_plan_from_metadata`

- [ ] **Step 1: Write failing tests** (surface invariant, warn at 18 mm residual, no warn within 2 cm, optional +Z misalignment warn)

```python
def test_build_plan_forces_surface_and_invariant():
    # tcp at (0,0.1,0), apple at origin, r=0.04 → residual 0.06, welded apple (0,0.06,0)
    ...
    np.testing.assert_allclose(aw + r * wdir, tcp, atol=1e-6)
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_real_post_grasp_plan.py -q
```

- [ ] **Step 3: Implement plan module (no scene apply yet)**

- [ ] **Step 4: Tests PASS + commit**

```bash
git commit -m "Add PostGraspPlan math and TCP–radius / +Z∥ŵ warn helpers."
```

---

### Task 2: Apply grasp after free settle (`CoupledCableScene`)

**Files:**
- Modify: `apple_pick_sim/system_id/real_post_grasp_plan.py`
- Modify: `apple_pick_sim/tests/test_real_post_grasp_plan.py`

**Interfaces:**
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
) -> CoupledCableScene: ...
```

**Algorithm:**
1. Snapshot `body_q` from free scene; set apple row to welded pos + apple quat.
2. Build welded scene:
   `GripperProxyConfig(fix_to_apple=True, weld_direction=plan.weld_direction,
    weld_reference_pos=plan.apple_pos_welded,
    weld_reference_quat=plan.apple_quat_xyzw)`
   — stock look-at (+Z along \(\hat{w}\)).
3. Copy woody (+ overridden apple) into welded state; do not trust free proxy pose.
4. `quiet_all_cable_bodies` + sync `body_q_prev`.
5. Return welded scene.

- [ ] **Step 1: Failing integration test** — after apply, apple ≈ welded, proxy ≈ tcp pos, proxy +Z aligned with \(\hat{w}\) (dot ≥ 0.99), a woody body unchanged vs pre-apply.

- [ ] **Step 2: Implement + PASS + commit**

```bash
git commit -m "Apply post-grasp look-at weld after free cable settle."
```

---

### Task 3: Wire CLI phases

**Files:**
- Modify: `robot_replay/example_view_pre_grasp_settle.py`
- Modify: `robot_replay/README.md`
- Modify: `docs/handbook-real-replay.md`
- Modify: spec status → Implemented (slice A)

**Flags:** `--grasp-after-settle`, `--post-grasp-settle-substeps` (500), `--tcp-radius-warn-m` (0.02).

**Phases:** `long_settle` → (optional) grasp apply + rebind viewer model → `short_settle` → `run`.

- [ ] **Step 1: Implement flags + phase machine**

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

Expected: exit 0; radius warn (~18 mm); optional +Z∥ŵ warn; grasp then short settle.

- [ ] **Step 3: Docs + commit**

```bash
git commit -m "Add --grasp-after-settle look-at weld phase to pre-grasp viewer."
```

---

## Spec coverage check

| Spec requirement | Task |
| ---------------- | ---- |
| Plan math + 2 cm warns | Task 1 |
| +Z ∥ ŵ look-at (not logged TCP quat) | Task 2 |
| Woody unchanged; apple snap | Task 2 |
| CLI long→grasp→short | Task 3 |
| Slice B / joint_pos | Deferred |

---

## Plan self-review

- Removed former Task 2 (`weld_fixed_offset_in_apple_frame`) — contradicts +Z∥ŵ contract.
- Logged TCP quat kept only for diagnostics / misalignment warn.
- Slice B note: `joint_pos` at grasp row for FR3 seed.

---

## Execution Handoff

Plan updated at `docs/superpowers/plans/2026-07-24-real-post-grasp-viewer.md`.

**1. Subagent-Driven (recommended)** · **2. Inline Execution**

Which approach?
