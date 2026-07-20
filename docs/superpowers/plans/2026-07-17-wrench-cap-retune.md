# Wrench-Cap Retune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise shared simulation and replay-instability wrench caps to 100 N and 40 N·m.

**Architecture:** Keep one source of truth in `apple_pick_sim.coupled_fruiting.scene`. The batched stability monitor continues importing those constants, preserving identical simulation-clamp and replay-detection limits.

**Tech Stack:** Python, pytest, PyTorch, uv

## Global Constraints

- Update existing tests only; do not add tests.
- Keep the unstable-frame disqualification threshold at strictly greater than 25%.
- Do not add separate replay limits or CLI options.

---

### Task 1: Retune shared wrench caps

**Files:**
- Modify: `apple_pick_gym/tests/test_batched_stability_monitor.py:71-84`
- Modify: `apple_pick_sim/coupled_fruiting/scene.py:43-45`
- Modify: `docs/batched-stability-monitor-design.md:87-105`
- Modify: `docs/ROADMAP.md:76,116`

**Interfaces:**
- Consumes: `DEFAULT_STEM_FORCE_CAP_N` and `DEFAULT_STEM_TORQUE_CAP_NM`
- Produces: shared defaults of `100.0` N and `40.0` N·m used by simulation builders and `StabilityThresholds`

- [ ] **Step 1: Update the existing numeric assertions**

```python
assert DEFAULT_STEM_FORCE_CAP_N == pytest.approx(100.0)
assert DEFAULT_STEM_TORQUE_CAP_NM == pytest.approx(40.0)
```

- [ ] **Step 2: Run the existing assertion and verify red**

Run:

```bash
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_batched_stability_monitor.py::test_default_stability_thresholds_match_scene_wrench_caps_and_speed_bounds -q
```

Expected: failure showing the current values are 50 N and 20 N·m.

- [ ] **Step 3: Update the shared production constants**

```python
DEFAULT_STEM_FORCE_CAP_N: float = 100.0
DEFAULT_STEM_TORQUE_CAP_NM: float = 40.0
```

- [ ] **Step 4: Update existing documentation**

Replace statements of the shared 50 N / 20 N·m policy with 100 N / 40 N·m in the stability design and roadmap. Do not change the documented 25% fraction policy.

- [ ] **Step 5: Run focused verification**

Run:

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_batched_stability_monitor.py \
  apple_pick_sim/tests/test_batched_heterogeneous_config.py -q
```

Expected: all selected tests pass.
