# Batched Sys-ID MMD Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the V.4.2 batched sys-ID collect stack (`ApplePickBatchedSysIdEnv` + `batched_sysid_collect.py`) with in-process parallel replay and MMD/Wasserstein stiffness-grid scoring on native `batched_sysid_v1` datasets.

**Architecture:** Add sibling orchestration module `batched_sysid_mmd_grid.py` that lockstep-replays recorded actions per `(candidate × direction)` env, collects features via `sysid_numpy_obs`, and scores against offline GT transition features. Geometry comes from frame-0 digital-twin inference; only `bend_stiffness` varies per candidate. No new replay env class.

**Tech Stack:** Python 3.12, Newton/Warp GPU batched sim, `ApplePickBatchedSysIdEnv`, `BatchedSysIdDataset`, NumPy, SciPy (`wasserstein_distance`), PyTorch action tensors, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-07-06-batched-sysid-mmd-grid-design.md`  
**Worktree:** `/home/abhinav/codes/apple_pick_sim-batched-sysid-mmd` on `feature/batched-sysid-mmd`

---

## File map

| File | Responsibility |
|------|----------------|
| `apple_pick_sim/system_id/wasserstein.py` | Mean per-dimension Wasserstein-1 on normalized feature matrices |
| `apple_pick_sim/system_id/mmd_results.py` | Extend CSV/plots for both metrics |
| `apple_pick_sim/system_id/batched_digital_twin_init.py` | Batched digital-twin obs + frame-0 init |
| `apple_pick_sim/system_id/mmd_features.py` | `replay_obs_dict_from_sysid_numpy` adapter |
| `apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py` | Grid helpers, replay collectors, `evaluate_batched_mmd_grid` |
| `apple_pick_gym/batched_examples/example_batched_sysid_mmd_grid.py` | CLI entry point |
| `apple_pick_gym/batched_envs/__init__.py` | Export orchestration API |
| `apple_pick_sim/system_id/__init__.py` | Export new helpers |
| `apple_pick_sim/tests/test_wasserstein.py` | Wasserstein unit tests |
| `apple_pick_sim/tests/test_distance_results.py` | Dual-metric result tests |
| `apple_pick_sim/tests/test_batched_digital_twin_init.py` | Batched obs + init tests |
| `apple_pick_gym/tests/test_batched_sysid_mmd_grid_helpers.py` | Grid/replay helper tests |
| `apple_pick_gym/tests/test_batched_sysid_mmd_grid.py` | Capstone collect→score |
| `apple_pick_gym/tests/test_example_batched_sysid_mmd_grid_cli.py` | CLI smoke |
| `docs/batched-sysid-mmd-grid.md` | User how-to |
| `docs/ROADMAP.md` | V.4.2.1 + V.4.3 validation |

---

### Task 1: Wasserstein distance helper

**Files:**
- Create: `apple_pick_sim/system_id/wasserstein.py`
- Create: `apple_pick_sim/tests/test_wasserstein.py`
- Modify: `apple_pick_sim/system_id/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import numpy as np
import pytest

from apple_pick_sim.system_id.wasserstein import mean_wasserstein1


def test_mean_wasserstein1_zero_for_identical_samples():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    assert mean_wasserstein1(x, x) == pytest.approx(0.0, abs=1e-12)


def test_mean_wasserstein1_positive_for_shifted_samples():
    x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    y = np.array([[10.0, 0.0], [11.0, 1.0]], dtype=np.float64)
    assert mean_wasserstein1(x, y) > 0.0


def test_mean_wasserstein1_rejects_mismatched_feature_dim():
    x = np.ones((3, 2), dtype=np.float64)
    y = np.ones((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="feature dimension mismatch"):
        mean_wasserstein1(x, y)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/abhinav/codes/apple_pick_sim-batched-sysid-mmd
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_wasserstein.py -v
```

Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def _as_feature_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    return arr


def mean_wasserstein1(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = _as_feature_matrix(x, name="x")
    y_arr = _as_feature_matrix(y, name="y")
    if x_arr.shape[1] != y_arr.shape[1]:
        raise ValueError(
            f"feature dimension mismatch: x={x_arr.shape[1]} y={y_arr.shape[1]}"
        )
    per_dim = [
        float(wasserstein_distance(x_arr[:, j], y_arr[:, j]))
        for j in range(x_arr.shape[1])
    ]
    return float(np.mean(per_dim))
```

Export `mean_wasserstein1` from `apple_pick_sim/system_id/__init__.py`.

- [ ] **Step 4: Run test to verify it passes** — Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add apple_pick_sim/system_id/wasserstein.py apple_pick_sim/system_id/__init__.py apple_pick_sim/tests/test_wasserstein.py
git commit -m "Add mean per-dimension Wasserstein helper for sys-ID scoring."
```

---

### Task 2: Dual-metric result types

**Files:**
- Modify: `apple_pick_sim/system_id/mmd_results.py`
- Create: `apple_pick_sim/tests/test_distance_results.py`

- [ ] **Step 1: Write failing tests for `DistanceCandidateResult` and `rank_distance_results`**

- [ ] **Step 2: Run tests** — Expected FAIL on import

- [ ] **Step 3: Add `DistanceCandidateResult`, `rank_distance_results`, `write_distance_results_csv`, `write_distance_diagnostic_plots(metric=...)` while keeping legacy `MmdCandidateResult` API**

- [ ] **Step 4: Run tests** — Expected PASS

- [ ] **Step 5: Commit** — `Extend MMD results module with dual-metric candidate records.`

---

### Task 3: Batched digital-twin obs + base params

**Files:**
- Create: `apple_pick_sim/system_id/batched_digital_twin_init.py`
- Create: `apple_pick_sim/tests/test_batched_digital_twin_init.py`

- [ ] **Step 1: Add `tiny_batched_dataset` fixture** using `collect_batched_quasi_static_dataset` (`S=1`, `D=1`, `max_steps=20`, `_test_sim_config` from `test_batched_sysid_collect.py`)

- [ ] **Step 2: Failing tests for `digital_twin_obs_from_batched_episode` and `infer_base_params_for_structure`**

- [ ] **Step 3: Implement** (mirror `digital_twin_obs_from_episode` using `BatchedSysIdDataset.load_episode_metadata` + `load_episode_obs_arrays`)

- [ ] **Step 4–5: Pass tests and commit**

---

### Task 4: Batched frame-0 init

**Files:**
- Modify: `batched_digital_twin_init.py`, tests

- [ ] **Step 1: Failing test** — after `initialize_batched_env_from_dataset`, `sysid_numpy_obs(0)["robot_joint_q"]` matches recorded frame 0

- [ ] **Step 3: Implement** — port `initialize_env_from_parquet` per world using `BatchedEnvLayout`; set `ExcitationContext` from recorded direction

- [ ] **Step 5: Commit** — `Initialize batched sys-ID env from v1 dataset frame-0 observations.`

---

### Task 5: Replay obs adapter

**Files:**
- Modify: `mmd_features.py`, `test_mmd_features.py`

- [ ] Implement `replay_obs_dict_from_sysid_numpy` mapping `sysid_numpy_obs` → `ReplayObservationCollector` keys; parity test at `num_envs=1`

---

### Task 6: Grid + action tensor helpers

**Files:**
- Create: `apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py`
- Create: `apple_pick_gym/tests/test_batched_sysid_mmd_grid_helpers.py`

- [ ] `BendStiffnessCandidate`, `iter_bend_stiffness_candidates`, `ensure_gt_candidate_in_grid`
- [ ] `build_recorded_actions_tensor`, `actions_tensor_from_recorded_frame`
- [ ] `gt_bend_stiffness_candidate_from_structure` from episode JSON params

---

### Task 7: BatchedSysIdReplayCollectors

- [ ] Mirror `BatchedSysIdCollectors`; one `ReplayObservationCollector` per env; `record_step(env, env_idx, frame_idx)`

---

### Task 8: GT context + scoring

- [ ] Port `_prepare_gt_mmd_context` / `_compute_candidate_mmd_result` from `run_system_identification.py`; add Wasserstein; return `DistanceCandidateResult`

---

### Task 9: evaluate_batched_mmd_grid

- [ ] Per-structure loop; chunk candidates; `_replay_candidate_chunk` uses `broadcast_structure_params` from `batched_sysid_collect`; lockstep `env.step`; write `s{ii}/` outputs

---

### Task 10: Example CLI

- [ ] `example_batched_sysid_mmd_grid.py` mirrors `example_batched_collect_sysid_data.py` + stiffness grid flags from `run_system_identification.py`

---

### Task 11: Capstone

- [ ] `test_batched_sysid_mmd_grid.py`: collect (`S=1,D=2`) → grid with GT + wrong stem K → GT ranks #1 on MMD² and Wasserstein

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_wasserstein.py \
  apple_pick_sim/tests/test_distance_results.py \
  apple_pick_sim/tests/test_batched_digital_twin_init.py \
  apple_pick_gym/tests/test_batched_sysid_mmd_grid_helpers.py \
  apple_pick_gym/tests/test_batched_sysid_mmd_grid.py \
  apple_pick_gym/tests/test_example_batched_sysid_mmd_grid_cli.py -q
```

---

### Task 12: Docs

- [ ] `docs/batched-sysid-mmd-grid.md`, update `ROADMAP.md` and `batched-sysid-dataset.md`

---

## Plan self-review

All spec goals mapped to tasks 1–11. No `ApplePickBatchedReplayEnv`. Cross-structure batching deferred. Types consistent across tasks.

---

## Execution handoff

Plan: `docs/superpowers/plans/2026-07-06-batched-sysid-mmd-grid.md`

1. **Subagent-Driven (recommended)** — fresh subagent per task
2. **Inline Execution** — executing-plans with checkpoints

Which approach?
