# Batched Sys-ID MMD Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the V.4.2 batched sys-ID collect stack (`ApplePickBatchedSysIdEnv` + `batched_sysid_collect.py`) with in-process parallel replay and MMD/Wasserstein stiffness-grid scoring on native `batched_sysid_v1` datasets.

**Architecture:** Digital-twin geometry from frame-0 obs → parallel lockstep replay of recorded actions on `ApplePickBatchedSysIdEnv` → offline distance scoring vs recorded GT features. Orchestration lives in `batched_sysid_mmd_grid.py` (sibling to `batched_sysid_collect.py`). No new replay env class.

**Implementation order (user-directed):**
1. **Digital twin** — batched obs inference + frame-0 init
2. **Parallelized replay** — collectors, orchestrator, example CLI (replay works end-to-end)
3. **Distance measures** — Wasserstein, dual-metric results, scoring wired into orchestrator, capstone

**Tech Stack:** Python 3.12, Newton/Warp, `ApplePickBatchedSysIdEnv`, `BatchedSysIdDataset`, NumPy, SciPy, PyTorch, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-07-06-batched-sysid-mmd-grid-design.md`  
**Worktree:** `/home/abhinav/codes/apple_pick_sim-batched-sysid-mmd` on `feature/batched-sysid-mmd`

---

## File map

| File | Phase | Responsibility |
|------|-------|----------------|
| `apple_pick_sim/system_id/batched_digital_twin_init.py` | 1 | Batched digital-twin obs + frame-0 init |
| `apple_pick_sim/tests/test_batched_digital_twin_init.py` | 1 | Digital-twin unit + init tests |
| `apple_pick_sim/system_id/mmd_features.py` | 2 | `replay_obs_dict_from_sysid_numpy` adapter |
| `apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py` | 2–3 | Replay orchestration; scoring added in phase 3 |
| `apple_pick_gym/batched_examples/example_batched_sysid_mmd_grid.py` | 2 | CLI (replay first; `--score` or always score in phase 3) |
| `apple_pick_gym/tests/test_batched_sysid_mmd_grid_helpers.py` | 2 | Grid/tensor/collector tests |
| `apple_pick_gym/tests/test_batched_sysid_replay.py` | 2 | Replay-only integration (no distance assert yet) |
| `apple_pick_sim/system_id/wasserstein.py` | 3 | Wasserstein helper |
| `apple_pick_sim/system_id/mmd_results.py` | 3 | Dual-metric CSV/plots |
| `apple_pick_gym/tests/test_batched_sysid_mmd_grid.py` | 3 | Capstone: GT ranks #1 |

---

## Phase 1 — Digital twin

### Task 1: Batched digital-twin observation + base params

**Files:**
- Create: `apple_pick_sim/system_id/batched_digital_twin_init.py`
- Create: `apple_pick_sim/tests/test_batched_digital_twin_init.py`
- Modify: `apple_pick_sim/system_id/__init__.py`

- [ ] **Step 1: Write failing tests**

Add `tiny_batched_dataset` fixture (collect `S=1`, `D=1`, `max_steps=20` via `collect_batched_quasi_static_dataset` + `_test_sim_config` from `test_batched_sysid_collect.py`).

```python
def test_digital_twin_obs_from_batched_episode_has_junction_names(tiny_batched_dataset):
    from apple_pick_sim.system_id.batched_digital_twin_init import (
        digital_twin_obs_from_batched_episode,
    )
    obs = digital_twin_obs_from_batched_episode(tiny_batched_dataset, 0, 0)
    assert obs.junction_names
    assert obs.woody_part_start_pos.size == 3 * len(obs.junction_names)


def test_infer_base_params_for_structure_returns_fruiting_params(tiny_batched_dataset):
    from apple_pick_sim.system_id.batched_digital_twin_init import infer_base_params_for_structure
    params = infer_base_params_for_structure(tiny_batched_dataset, structure_idx=0)
    assert params.primary is not None
    assert params.primary.bend_stiffness > 0.0
```

- [ ] **Step 2: Run tests — expect FAIL** (`ModuleNotFoundError`)

```bash
cd /home/abhinav/codes/apple_pick_sim-batched-sysid-mmd
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_batched_digital_twin_init.py -v
```

- [ ] **Step 3: Implement `digital_twin_obs_from_batched_episode` and `infer_base_params_for_structure`**

Mirror `digital_twin_obs_from_episode` in `parquet_init.py` but take `(structure_idx, direction_idx)` on `BatchedSysIdDataset`. Use `stack_woody_pos_frame` for frame 0. `infer_base_params_for_structure` calls `infer_params_from_obs(obs, fixture_path)`.

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "Add batched digital-twin obs and base-param inference from v1 datasets."
```

---

### Task 2: Batched frame-0 init after reset

**Files:**
- Modify: `apple_pick_sim/system_id/batched_digital_twin_init.py`
- Modify: `apple_pick_sim/tests/test_batched_digital_twin_init.py`

- [ ] **Step 1: Write failing test**

```python
@gymnasium_available
@requires_fr3
def test_initialize_batched_env_from_dataset_sets_joint_q_and_tcp(tiny_batched_dataset):
    from apple_pick_gym.batched_envs import ApplePickBatchedSysIdEnv
    from apple_pick_sim.system_id.batched_digital_twin_init import (
        infer_base_params_for_structure,
        initialize_batched_env_from_dataset,
    )
    dataset = tiny_batched_dataset
    meta = dataset.load_episode_metadata(0, 0)
    arrays = dataset.load_episode_obs_arrays(0, 0)
    n_frames = int(arrays["action"].shape[0])
    base = infer_base_params_for_structure(dataset, 0)

    env = ApplePickBatchedSysIdEnv(
        num_envs=1,
        max_episode_steps=n_frames,
        ranges_path=meta["fixture_path"],
        topology_seed=int(meta["seed"]),
        use_settle_cache=False,
        sim_config=_test_sim_config(num_envs=1),
        per_env_params=[base],
    )
    try:
        env.reset(seed=int(meta["seed"]))
        initialize_batched_env_from_dataset(
            env, dataset, structure_idx=0, num_directions=1,
        )
        live = env.sysid_numpy_obs(0)
        np.testing.assert_allclose(
            live["robot_joint_q"], arrays["robot_joint_q"][0], atol=1e-4,
        )
        np.testing.assert_allclose(live["tcp_pos"], arrays["tcp_pos"][0], atol=1e-3)
    finally:
        env.close()
```

- [ ] **Step 2: Run test — expect FAIL** (`initialize_batched_env_from_dataset` missing)

- [ ] **Step 3: Implement `initialize_batched_env_from_dataset`**

For each `env_idx`, load episode `(structure_idx, env_idx % num_directions)`. Port robot `joint_q`, MuJoCo buffers, VIC `target_tf` from `initialize_env_from_parquet` (`parquet_init.py`) into the correct world slice. Read `batched_layout.py` + `batched_sysid_world_info.py` for world indexing. Call `env.set_excitation_context(env_idx, ExcitationContext(...))` from recorded `excitation_direction[0]`.

**Do not** pass full `fruiting_system_params` from metadata in capstone path — geometry comes from `infer_base_params_for_structure` at build time; init only teleports robot/TCP state from observations.

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "Initialize batched sys-ID env from v1 frame-0 observations."
```

---

## Phase 2 — Parallelized replay

### Task 3: Replay observation adapter

**Files:**
- Modify: `apple_pick_sim/system_id/mmd_features.py`
- Modify: `apple_pick_sim/tests/test_mmd_features.py`

- [ ] **Step 1: Write failing test**

`sysid_numpy_obs` returns `woody_part_start_pos` / `woody_part_end_pos` dicts (see `obs_torch.sysid_numpy_obs_from_batched`). Adapter must flatten to `woody_start` / `woody_end` for `ReplayObservationCollector`.

```python
def test_replay_obs_dict_from_sysid_numpy_flattens_woody():
    from apple_pick_sim.system_id.mmd_features import replay_obs_dict_from_sysid_numpy
    sysid_obs = {
        "ft_wrist": np.arange(6, dtype=np.float32),
        "tcp_velocity": np.arange(6, 10, dtype=np.float32),
        "tcp_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "apple_pos": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        "woody_part_start_pos": {
            "joint_b": np.array([7.0, 8.0, 9.0], dtype=np.float32),
            "joint_a": np.array([10.0, 11.0, 12.0], dtype=np.float32),
        },
        "woody_part_end_pos": {
            "joint_b": np.array([13.0, 14.0, 15.0], dtype=np.float32),
            "joint_a": np.array([16.0, 17.0, 18.0], dtype=np.float32),
        },
    }
    out = replay_obs_dict_from_sysid_numpy(sysid_obs, junction_names=["joint_b", "joint_a"])
    np.testing.assert_allclose(out["woody_start"], [7, 8, 9, 10, 11, 12])
    np.testing.assert_allclose(out["woody_end"], [13, 14, 15, 16, 17, 18])
```

- [ ] **Step 2–5: Implement using `flatten_woody_positions`, test parity with `ReplayObservationCollector` at one frame, commit.**

---

### Task 4: Stiffness grid + recorded-action tensors

**Files:**
- Create: `apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py` (part 1)
- Create: `apple_pick_gym/tests/test_batched_sysid_mmd_grid_helpers.py`

- [ ] Implement:
  - `BendStiffnessCandidate` + `apply_to(base)` via `set_rod_bend_stiffness`
  - `iter_bend_stiffness_candidates` (copy from `run_system_identification.py`)
  - `ensure_gt_candidate_in_grid`
  - `gt_bend_stiffness_candidate_from_structure`
  - `build_recorded_actions_tensor(dataset, structure_idx, num_directions, num_candidates)`
  - `actions_tensor_from_recorded_frame(recorded_actions, frame_idx, device)`

- [ ] Tests for Cartesian product, tensor shapes, GT injection.

- [ ] Commit: `Add bend-stiffness grid and recorded-action tensor helpers.`

---

### Task 5: BatchedSysIdReplayCollectors

**Files:**
- Modify: `batched_sysid_mmd_grid.py`, helper tests

- [ ] Mirror `BatchedSysIdCollectors` from `batched_sysid_collect.py`:
  - One `ReplayObservationCollector` per env, keyed by recorded episode arrays for that env's direction
  - `record_step(env, env_idx, frame_idx)` → `env.sysid_numpy_obs` → adapter → collector
  - `to_arrays(env_idx)`, `merge()` for chunking

- [ ] Commit: `Add batched replay collectors for per-env feature accumulation.`

---

### Task 6: Replay orchestrator (no distance scoring yet)

**Files:**
- Modify: `batched_sysid_mmd_grid.py`
- Create: `apple_pick_gym/tests/test_batched_sysid_replay.py`

- [ ] **Step 1: Implement `_replay_candidate_chunk` and `replay_batched_sysid_structure`**

```python
def replay_batched_sysid_structure(
    *,
    dataset: BatchedSysIdDataset,
    structure_idx: int,
    candidates: Sequence[BendStiffnessCandidate],
    num_directions: int,
    seed: int,
    build_env_fn: Callable[..., ApplePickBatchedSysIdEnv],
) -> BatchedSysIdReplayCollectors:
    """Run lockstep recorded-action replay for all candidates × directions."""
    num_candidates = len(candidates)
    num_envs = num_candidates * num_directions
    base_params = infer_base_params_for_structure(dataset, structure_idx)
    per_env_params = broadcast_structure_params(
        [c.apply_to(base_params) for c in candidates],
        num_directions,
    )
    recorded_actions = build_recorded_actions_tensor(
        dataset, structure_idx=structure_idx,
        num_directions=num_directions, num_candidates=num_candidates,
    )
    n_frames = int(recorded_actions.shape[1])
    env = build_env_fn(
        num_envs=num_envs,
        per_env_params=per_env_params,
        max_episode_steps=n_frames,
    )
    try:
        env.reset(seed=seed)
        initialize_batched_env_from_dataset(
            env, dataset, structure_idx=structure_idx, num_directions=num_directions,
        )
        recorded_by_env = _recorded_metadata_by_env(dataset, structure_idx, num_directions, num_candidates)
        collectors = BatchedSysIdReplayCollectors(num_envs, recorded_by_env)
        for frame_idx in range(n_frames):
            actions = actions_tensor_from_recorded_frame(
                recorded_actions, frame_idx=frame_idx, device=env.device,
            )
            env.step(actions)
            for env_idx in range(num_envs):
                collectors.record_step(env, env_idx=env_idx, frame_idx=frame_idx)
        return collectors
    finally:
        env.close()
```

Import `broadcast_structure_params` from `batched_sysid_collect` — do not duplicate.

- [ ] **Step 2: Integration test `test_replay_batched_structure_produces_frames`**

Collect tiny dataset (`S=1`, `D=2`). Replay with **one** candidate (GT params). Assert collectors have `n_rows == n_frames` per env and `to_arrays(0)["ft_wrist"].shape[0] > 0`.

- [ ] **Step 3: Commit**

```bash
git commit -m "Add batched sys-ID structure replay without distance scoring."
```

---

### Task 7: Parallelized replay example CLI

**Files:**
- Create: `apple_pick_gym/batched_examples/example_batched_sysid_mmd_grid.py`
- Create: `apple_pick_gym/tests/test_example_batched_sysid_mmd_grid_cli.py`

- [ ] Copy `build_sim_config` / module constants from `example_batched_collect_sysid_data.py`.

- [ ] Phase-2 CLI flags: `--dataset`, `--structure-indices`, `--max-envs-per-batch`, stiffness grids, `--replay-only` (runs replay, prints frame counts / optional obs error vs recorded — **no MMD output yet**).

- [ ] `main()` calls chunking wrapper around `replay_batched_sysid_structure` per structure.

- [ ] `--help` smoke test.

- [ ] Commit: `Add batched sys-ID replay example CLI (pre-scoring).`

**Phase 2 gate:** headless replay of a collected dataset completes; digital-twin init used; no legacy materialize bridge.

---

## Phase 3 — Distance measures

### Task 8: GT MMD context + candidate scoring (MMD only first)

**Files:**
- Modify: `batched_sysid_mmd_grid.py`
- Modify: `test_batched_sysid_mmd_grid_helpers.py`

- [ ] Port from `run_system_identification.py`:
  - `_combine_transition_features` / `prepare_gt_distance_context` (GT from **recorded** episodes only)
  - `score_candidate_mmd` → returns existing `MmdCandidateResult` initially

- [ ] Wire `evaluate_batched_mmd_grid` = replay + MMD score per candidate; write `mmd_results.csv` via existing `write_results_csv`.

- [ ] Test: synthetic arrays → GT context + known candidate features → MMD² ordering correct.

- [ ] Commit: `Wire MMD scoring into batched replay orchestrator.`

---

### Task 9: Wasserstein distance

**Files:**
- Create: `apple_pick_sim/system_id/wasserstein.py`
- Create: `apple_pick_sim/tests/test_wasserstein.py`

- [ ] Full TDD for `mean_wasserstein1` (identical → 0, shifted → positive, dim mismatch error).

- [ ] Extend `score_candidate_*` to compute Wasserstein alongside MMD using same GT normalization.

- [ ] Commit: `Add Wasserstein distance for batched sys-ID scoring.`

---

### Task 10: Dual-metric results + plots

**Files:**
- Modify: `mmd_results.py`
- Create: `test_distance_results.py`

- [ ] `DistanceCandidateResult`, `write_distance_results_csv`, `write_distance_diagnostic_plots(metric=...)`.

- [ ] `evaluate_batched_mmd_grid` writes both metrics.

- [ ] Commit: `Add dual-metric CSV and diagnostic plots for batched grid.`

---

### Task 11: Capstone — GT ranks #1

**Files:**
- Create: `apple_pick_gym/tests/test_batched_sysid_mmd_grid.py`

- [ ] Collect `S=1`, `D=2` → grid with GT stem K + wrong stem K → assert best MMD² and best Wasserstein both at GT.

- [ ] Remove `--replay-only` default path or keep flag for debugging.

- [ ] Full validation:

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_batched_digital_twin_init.py \
  apple_pick_sim/tests/test_wasserstein.py \
  apple_pick_gym/tests/test_batched_sysid_replay.py \
  apple_pick_gym/tests/test_batched_sysid_mmd_grid.py \
  apple_pick_gym/tests/test_example_batched_sysid_mmd_grid_cli.py -q
```

- [ ] Commit: `Capstone: batched collect, replay, and grid rank GT stiffness first.`

---

### Task 12: Documentation

- [ ] `docs/batched-sysid-mmd-grid.md` — document three phases; collect → replay → score commands.
- [ ] Update `docs/ROADMAP.md`, `docs/batched-sysid-dataset.md`.
- [ ] Commit docs.

---

## Plan self-review

| Spec requirement | Phase |
|------------------|-------|
| Digital-twin frame-0 init | 1 |
| Native v1 dataset | 1–2 |
| `ApplePickBatchedSysIdEnv` + `batched_sysid_collect` patterns | 2 |
| Parallel replay before scoring | 2 then 3 |
| MMD² + Wasserstein | 3 |
| GT ranks #1 | Task 11 |
| No replay env class | throughout |

---

## Execution handoff

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks.
2. **Inline Execution** — implementing phases 1→2→3 in this session with checkpoints.

Which approach?
