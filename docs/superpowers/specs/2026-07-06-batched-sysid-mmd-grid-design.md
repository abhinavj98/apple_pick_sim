# Batched sys-ID MMD grid — design spec

**Date:** 2026-07-06  
**Branch:** `feature/batched-sysid-mmd`  
**Worktree:** `../apple_pick_sim-batched-sysid-mmd`  
**Roadmap slices:** V.4.2.1 (digital-twin replay init) + V.4.3 (in-process batched MMD)  
**Builds on:** V.4.2 batched collection (`ApplePickBatchedSysIdEnv`, `batched_sysid_collect.py`)  
**Supersedes:** subprocess `apple_pick_gym/examples/run_system_identification.py` for batched datasets

## Problem

Parallel GT collection is done (`example_batched_collect_sysid_data.py` → `batched_sysid_v1`). The next step replays those trajectories under many `bend_stiffness` candidates, scores distribution distance vs recorded GT, and verifies true params rank best. Legacy `run_system_identification.py` does this sequentially via `ApplePickReplayEnv`. We need the **batched collect stack** extended — not a port of the single-env replay env.

## Goals

1. Read `batched_sysid_v1` natively (`BatchedSysIdDataset`).
2. Per GT **structure**: frame-0 obs → digital-twin geometry; Cartesian `bend_stiffness` grid; batch `(candidate × direction)` in one `ApplePickBatchedSysIdEnv` build (chunked).
3. Lockstep replay of **recorded actions** from dataset (mirror collect lockstep stepping).
4. MMD² + mean per-dim Wasserstein-1 on hold-phase transitions; rank; CSV + plots; tests assert GT ranks #1.
5. End here — no CMA-ES (V.5.2).

## Non-goals

- New top-level replay env class patterned on `ApplePickReplayEnv`.
- Legacy `materialize_legacy_episode_dir` bridge.
- Cross-structure batching in one build (deferred).
- Grid axes beyond `bend_stiffness` on four rod segments.
- CMA-ES / CEM.

## Architectural alignment with V.4.2 collection

Collection established the canonical batched sys-ID stack:

| Layer | Collection (V.4.2) | MMD grid (this slice) |
|-------|-------------------|----------------------|
| Env | `ApplePickBatchedSysIdEnv` | **Same env** — extend only if replay-init hooks need thin methods |
| Orchestration | `batched_sysid_collect.py` | **`batched_sysid_mmd_grid.py`** (sibling module) |
| Example | `example_batched_collect_sysid_data.py` | **`example_batched_sysid_mmd_grid.py`** (same build_sim_config / module-constant pattern) |
| Indexing | `structure_and_direction_indices`, `broadcast_structure_params` | **Reuse unchanged** |
| Per-env obs | `env.sysid_numpy_obs(env_idx)` | **Reuse** for feature collection |
| Metadata | `build_episode_metadata`, `per_env_reset_info` | **Read back** from dataset / reload for init |
| Stepping | Lockstep `QuasiStaticTrajectory` → `actions_tensor_for_velocity` | Lockstep **recorded** `(num_envs, n_frames, 6)` → `torch` action tensor per frame |
| Writers | `BatchedSysIdCollectors` | **`BatchedSysIdReplayCollectors`** (parallel naming) — accumulates replay obs for MMD features, no Parquet write |

**Do not** introduce `ApplePickBatchedReplayEnv` as a separate env hierarchy. Replay is orchestration on top of `ApplePickBatchedSysIdEnv` + `BatchedHeterogeneousCoupledSim`, exactly as collection is orchestration on top of the same env.

## Env index mapping (unchanged from collect)

```
num_envs = n_candidates × num_directions   # within one structure chunk

env_idx = candidate_idx * num_directions + direction_idx
```

- `per_env_params = broadcast_structure_params(candidate_params_list, num_directions)`.
- Candidate `i` shares geometry (digital-twin inferred once per structure); only `bend_stiffness` differs.
- Recorded actions for env `i`: load from `BatchedSysIdDataset.load_episode_obs_arrays(structure_idx, direction_idx)["action"]` where `direction_idx = env_idx % num_directions`.

## Components

### 1. `ApplePickBatchedSysIdEnv` (existing — minimal extension only)

**Path:** `apple_pick_gym/batched_envs/apple_pick_batched_sysid_env.py`

Reuse as-is for batched sim, `sysid_numpy_obs`, `set_excitation_contexts`, `per_env_reset_info`.

Optional small additions only if needed:

- `apply_digital_twin_init_from_dataset(...)` delegating to batched init helper, or
- documented call sequence in orchestrator after `reset()`.

No separate replay env class.

### 2. `batched_sysid_mmd_grid.py` (new orchestration — mirror `batched_sysid_collect.py`)

**Path:** `apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py`

Parallel structure to `collect_batched_quasi_static_dataset`:

| Collect helper | MMD grid analogue |
|--------------|-------------------|
| `collect_batched_quasi_static_dataset` | `evaluate_batched_mmd_grid` (or `replay_batched_sysid_mmd_grid`) |
| `BatchedSysIdCollectors` | `BatchedSysIdReplayCollectors` — per-env feature accumulation via `sysid_numpy_obs` + recorded phase/dir from dataset |
| `assign_pull_directions` | Load `pull_direction` / `excitation_direction` from episode metadata (no resampling) |
| `build_episode_metadata` | `load_episode_metadata` from `BatchedSysIdDataset` (read path) |
| `actions_tensor_for_velocity` | `actions_tensor_from_recorded_frame(recorded_actions, frame_idx, device)` |

**`evaluate_batched_mmd_grid` flow (per structure):**

1. Load all direction episodes; build GT transition features from **recorded** arrays (offline, no sim).
2. `infer_params_from_obs` once per structure from frame-0 digital-twin obs.
3. Build candidate grid (`bend_stiffness` Cartesian product + inject GT combo).
4. For each chunk of candidates:
   - Build `per_env_params` via `broadcast_structure_params`.
   - Construct `ApplePickBatchedSysIdEnv` with same `build_sim_config` pattern as collect example.
   - `reset()` + batched digital-twin frame-0 init.
   - Lockstep loop over `frame_idx in range(n_frames)`: stack recorded actions → `env.step(actions)`; `BatchedSysIdReplayCollectors.record_step(...)`.
5. Merge chunk collectors; compute MMD² + Wasserstein per candidate; rank; write outputs.

Export shared helpers from `batched_sysid_collect.py` where already public (`structure_and_direction_indices`, `broadcast_structure_params`) — import, do not duplicate.

### 3. Batched digital-twin frame-0 init

**Path:** `apple_pick_sim/system_id/batched_digital_twin_init.py`

Called from orchestrator after `env.reset()`. Uses `BatchedEnvLayout` world indices + episode parquet frame-0 / schema metadata. Generalizes `initialize_env_from_parquet` for per-env teleports on the batched scene.

Geometry params come from `digital_twin_obs_from_episode` / `infer_params_from_obs` (v1 dataset helpers wrapping `BatchedSysIdDataset`).

### 4. Feature + distance modules

- **`BatchedReplayFeatureCollector`** in `mmd_features.py` — fed by `BatchedSysIdReplayCollectors`; output per `env_idx` arrays for `build_transition_features_by_direction`.
- **`wasserstein.py`** — `mean_wasserstein1` on GT-normalized features (scipy 1D per dim).
- **Extend `mmd_results.py`** — both metrics in CSV/plots.

### 5. Example script

**Path:** `apple_pick_gym/batched_examples/example_batched_sysid_mmd_grid.py`

Same shape as `example_batched_collect_sysid_data.py`:

- `build_sim_config(num_envs=...)` module constants for VIC/settle (not CLI).
- CLI: `--dataset`, stiffness grids, `--max-envs-per-batch`, `--output`, `--structure-indices`, `--max-candidates`.
- Calls `evaluate_batched_mmd_grid(env, ...)`.

### 6. `__init__.py` exports

Add orchestration exports alongside existing batched envs when API stabilizes.

## Data flow

GT features: recorded parquet only. Candidate features: live replay via `ApplePickBatchedSysIdEnv` + `sysid_numpy_obs`. Hold-phase `[s_t, Δs_t]` contract unchanged (`docs/system_identification.md` §3).

## Chunking

`--max-envs-per-batch`: split candidate groups; merge `BatchedSysIdReplayCollectors` per structure before scoring. GT candidate always in grid.

## Testing (TDD)

| Test | Asserts |
|------|---------|
| `test_batched_sysid_mmd_grid.py` | Collect tiny v1 dataset via existing collect path → run grid orchestrator → GT ranks #1 (MMD + Wasserstein) |
| `test_wasserstein.py` | Unit tests |
| Extend `test_mmd_features.py` | Collector parity |
| Extend `test_batched_sysid_collect.py` patterns | Shared indexing/broadcast imports stable |

Capstone must use **`collect_batched_quasi_static_dataset` + `ApplePickBatchedSysIdEnv`** for GT data, then **`evaluate_batched_mmd_grid`** for scoring — proving the collect→replay stack is continuous.

## Deferred: cross-structure batching

~20–30% extra; flatten `(structure, candidate, direction)` indexing + chunk-spanning accumulation. Not in v1.

## Success criteria

1. No `ApplePickBatchedReplayEnv`; orchestration lives in `batched_sysid_mmd_grid.py` beside `batched_sysid_collect.py`.
2. GT stiffness ranks #1 under both metrics on synthetic batched dataset.
3. Observation-only digital-twin init in capstone.
4. Headless end-to-end without per-candidate subprocess env builds.
5. Fast-gate tests green.
