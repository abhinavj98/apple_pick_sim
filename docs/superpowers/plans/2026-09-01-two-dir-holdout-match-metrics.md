# Two-direction holdout match metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** During CMA holdout eval, write `match_metrics.json` and val `npz` bags with time-series match stats on held-out dirs 0 and 3 (full + hold windows).

**Architecture:** Pure stats live in `apple_pick_sim/system_id/match_metrics.py`. `holdout_evaluation.py` extracts aligned F/T/woody series, calls the stats helpers, persists `npz` via `build_direction_state_npz`, and writes JSON atomically. CLI adds `--write-match-metrics` (default on in holdout mode).

**Tech Stack:** Python 3, NumPy, existing holdout/CMA plumbing.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-01-two-dir-holdout-match-metrics-design.md`
- Do **not** change Sinkhorn fitness or H3 gate semantics
- Real F/T: `scored_ft_wrist`; sim F/T: raw `ft_wrist`
- Lag search ±30 frames (1 s at 30 Hz); positive lag = sim delayed
- `nan` stats serialize as JSON `null`
- **Do not** run the 9 GPU CMA experiment jobs in this implementation slice

---

### Task 1: `match_metrics.py` pure stats

**Files:**
- Create: `apple_pick_sim/system_id/match_metrics.py`
- Test: `apple_pick_sim/tests/test_match_metrics.py`

**Interfaces:**
- Produces: `scalar_match_stats(real, sim, *, max_lag=30) -> dict`
- Produces: `vector3_match_stats(real, sim, *, max_lag=30, peak_mode="norm") -> dict`
- Produces: `tree_label_from_dataset_path(path) -> str`

- [ ] Write failing tests (identical, offset, lag-5, empty, zero-variance R², vector3)
- [ ] Implement helpers; run `uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_match_metrics.py -q`

### Task 2: Holdout integration

**Files:**
- Modify: `apple_pick_gym/batched_envs/holdout_evaluation.py`
- Test: `apple_pick_gym/tests/test_holdout_evaluation.py`

**Interfaces:**
- Consumes: Task 1 stats + `build_direction_state_npz`
- Produces: `build_match_metrics_report(...)`, `write_match_metrics(...)`, `write_val_holdout_npz(...)`

- [ ] Extend `run_holdout_evaluation` with `write_match_metrics`, `dataset_path`, `cma_seed`, `artifact_errors`
- [ ] Holdout tests for JSON schema and npz paths

### Task 3: CLI flag + cleanup

**Files:**
- Modify: `apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py`
- Test: `apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py`

- [ ] Add `--write-match-metrics` / `--no-write-match-metrics`
- [ ] Clear `match_metrics.json` and holdout npz on `--overwrite`
- [ ] CLI tests for write/skip behavior

### Task 4: Seed summary script

**Files:**
- Create: `apple_pick_gym/viz/summarize_match_metrics.py`

- [ ] Read one or more `match_metrics.json` paths; print per-tree mean±std table (fitted, full window, combined F)
