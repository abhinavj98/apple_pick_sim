# Batched Sys-ID Data Collection (V.4.2 slice)

| Field | Value |
| ----- | ----- |
| **Date** | 2026-07-04 |
| **Status** | Approved (brainstorming) |
| **Roadmap** | [V].4 — batched sys-ID (collection slice) |
| **Depends on** | V.3.3 `ApplePickBatchedBaseEnv` (done) |

## Summary

Parallelize quasi-static sys-ID data collection (§2.1) across a full grid of
`(structure, direction)` pairs. Each parallel env runs a **single-direction**
push–hold quasi-static sweep with fix-to-apple welding, records frames via the
existing `TrajectoryWriter` Parquet schema, and writes episodes compatible with
`example_gym_replay_overrides.py`.

**Out of scope for this slice:** batched replay, in-process MMD, multi-topology
batched builds, multi-world viewer.

## Decisions (locked)

| Topic | Choice |
| ----- | ------ |
| Env grid | Full grid: `num_envs = num_structures × num_directions` |
| Structure sampling | Heterogeneous DR: topology fixed via `topology_seed`, material params vary per structure |
| Params per structure | Sample `num_structures` param sets; broadcast each to its `num_directions` envs |
| Deliverable | Collection only; Parquet compatible with existing single-env replay |
| Approach | `ApplePickBatchedSysIdEnv` + thin collection script + small helper module |

## Architecture

### Env index mapping

```
num_structures = S, num_directions = D  →  num_envs = S × D

env_idx = s * D + d
  structure_idx s = env_idx // D
  direction_idx d = env_idx % D
```

### Param broadcast

1. Sample `S` param sets: `sample_heterogeneous_params_list(ranges, topology_seed, num_envs=S)`.
2. Build `per_env_params` of length `S × D` by repeating structure `s`'s params `D` times.
3. Pass to `BatchedHeterogeneousCoupledSim` at build time (override base env's default sampling).

### Pull directions

After `reset()`:

1. For each structure `s`, read geometry from representative env `s * D` (physical stem direction, robot vector).
2. Call `sample_robot_facing_pull_directions(D, physical_stem, robot_vec)` once per structure.
3. Assign direction `d` to env `s * D + d`.
4. Each env runs `QuasiStaticTrajectory([direction], config)` — a single-direction sweep.

### Lockstep stepping

All envs share the same `QuasiStaticStepConfig` and phase schedule (`move_out` → `hold` × N).
Frame counts are identical across envs. Each global frame:

- Build per-env velocity from the shared phase and env-specific direction.
- Stack into `(num_envs, 6)` action tensor.
- Single batched `env.step(actions)`.

No `restore_grasp_pose()` within an episode (single direction per env; no inter-direction teleport).

### Topology constraint

The batched heterogeneous backend requires **uniform segment topology** across all
worlds (`_assert_uniform_topology`). “Different structures” means different material
and geometry scalars sampled from the ranges fixture, not different branch layouts.
Multi-topology parallel collection is deferred.

## Components

### 1. `ApplePickBatchedSysIdEnv`

**Location:** `apple_pick_gym/batched_envs/apple_pick_batched_sysid_env.py`

Extends `ApplePickBatchedBaseEnv`.

**Constructor:** Mirror single-env sys-ID defaults from `ApplePickSysIdEnv`:

- `fix_to_apple=True` (via `BatchedHeterogeneousCoupledSimConfig.gym_defaults`)
- `control_hz=60.0`, VIC gains, stem force caps
- Optional `per_env_params: Sequence[FruitingSystemParams] | None` to inject broadcast params from the collection script

**Observation extensions** (batched tensors, leading dim `num_envs`):

| Field | Source |
| ----- | ------ |
| `tcp_pos`, `tcp_quat` | `obs_bufs.tcp_pose[:, :3]`, `[:, 3:7]` |
| `apple_quat` | Per-world apple body quat gather (extend `batched_obs` or env-side gather) |
| `robot_joint_q` | `obs_bufs.joint_q` |
| `raw_ft_wrist` | Uncapped stem-harvest wrench (batched gather); if unavailable at first, duplicate `ft_wrist` only if parity test shows negligible difference |
| `excitation_type`, `excitation_f_inst`, `excitation_direction` | Per-env buffer updated via `set_excitation_context` |

**Helper:** `sysid_numpy_obs(env_idx: int) -> dict` — maps batched obs to legacy v3 numpy dict (`woody_part_start_pos`, `woody_part_end_pos`, etc.) required by `TrajectoryWriter.record_step`.

**Info extensions** (per-env, as lists indexed by env or explicit dict):

- `weld_direction`, `robot_base_pos`, `fruiting_base_pos`, `rod_radii`, `params_fingerprint`

**Not in v1:** `restore_grasp_pose`, batched replay hooks.

### 2. `batched_sysid_collect.py`

**Location:** `apple_pick_gym/batched_envs/batched_sysid_collect.py`

Pure orchestration (no argparse):

- `structure_and_direction_indices(env_idx, num_directions) -> (s, d)`
- `broadcast_structure_params(structure_params, num_directions) -> list`
- `assign_pull_directions(env, num_structures, num_directions) -> list[np.ndarray]` — one unit direction per env
- `BatchedSysIdCollectors` — holds `S×D` `TrajectoryWriter` instances, `record_step` fan-out, `save_all(output_dir, meta_builder)`

### 3. `example_batched_collect_sysid_data.py`

**Location:** `apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py`

**CLI:** All trajectory/recording flags from `example_gym_sysid.py`, plus:

| Flag | Meaning |
| ---- | ------- |
| `--num-structures` | S |
| `--num-directions` | D |
| (derived) | `num_envs = S × D`; error if explicit `--num-envs` conflicts |

**Flow:**

1. Parse args; build `QuasiStaticStepConfig`.
2. Sample and broadcast params; construct `ApplePickBatchedSysIdEnv`.
3. `reset(seed)`; optional `--save-snapshot` per episode.
4. Assign pull directions; create writers.
5. Lockstep trajectory loop → record → save Parquet.
6. Headless default on Linux without display (`--viewer null`).

**Output layout** (same as single-env sys-ID):

```
<output>/
  metadata.parquet          # one row per episode (S×D rows)
  frames/<episode_id>.parquet
  initial_states/<episode_id>.npz   # only with --save-snapshot
```

Each `EpisodeMeta` row: `n_directions=1`, unique `episode_id`, structure-specific
`params_fingerprint`, `fruiting_system_params`, weld metadata, trajectory config fields.

**Downstream compatibility:** Any episode replays via existing
`example_gym_replay_overrides.py` without modification.

## Testing

| Module | Asserts |
| ------ | ------- |
| `apple_pick_gym/tests/test_batched_sysid_env.py` | Obs shapes; excitation round-trip; `sysid_numpy_obs(0)` keys match single-env sys-ID contract at `num_envs=1` |
| `apple_pick_gym/tests/test_batched_sysid_collect.py` | `S=1, D=1`, capped steps, `--output` tmpdir → valid Parquet schema (`REQUIRED_FRAME_COLUMNS`, metadata columns) |
| `test_batched_sysid_replay_fidelity.py` | **Capstone:** collect `S×D` episodes → observation-only digital-twin reset from frame 0 → open-loop Parquet action replay → `_compare_to_dataset` error vs GT (see [Final integration test](#final-integration-test-collect--digital-twin-replay--ground-truth-error)) |

**Validation (fast):**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_batched_sysid_env.py \
  apple_pick_gym/tests/test_batched_sysid_collect.py -q
```

Full slice validation including capstone replay test: see **Final integration test** section at end of this doc.

## Out of scope

- `example_batched_replay_overrides.py` (V.4.1)
- `gather_transitions()` / in-process batched MMD (V.4.2–V.4.3)
- Different segment topologies per structure (backend change)
- Multi-world Newton GL viewer during collection

## File checklist

| Path | Action |
| ---- | ------ |
| `apple_pick_gym/batched_envs/apple_pick_batched_sysid_env.py` | Add |
| `apple_pick_gym/batched_envs/batched_sysid_collect.py` | Add |
| `apple_pick_gym/batched_envs/__init__.py` | Export new env |
| `apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py` | Add |
| `apple_pick_gym/tests/test_batched_sysid_env.py` | Add |
| `apple_pick_gym/tests/test_batched_sysid_collect.py` | Add |
| `apple_pick_gym/tests/test_batched_sysid_replay_fidelity.py` | Add (capstone GT replay) |
| `apple_pick_sim/batched_obs.py` (optional) | Apple quat gather if not env-local |

## References

- `apple_pick_gym/examples/example_gym_sysid.py` — single-env collection reference
- `apple_pick_gym/examples/example_gym_replay_overrides.py` — replay consumer
- `apple_pick_gym/batched_envs/apple_pick_batched_base_env.py` — batched base
- `apple_pick_sim/system_id/trajectory_store.py` — Parquet writer contract
- `docs/system_identification.md` — §2.1 protocol
- `docs/ROADMAP.md` — V.4 milestones

## Final integration test: collect → digital-twin replay → ground-truth error

The **capstone validation** for this slice is a sim-to-sim round-trip that proves
batched-collected Parquet is not merely schema-valid, but **replayable with
observation-only initialization** and **low error against the recorded ground truth**.

This mirrors the M3 observation-only replay contract (`docs/digital-twin.md`,
`docs/system_identification.md`): treat the collection run as ground truth (GT),
rebuild the post-grasp scene from **frame-0 observations only**, open-loop replay
stored EE actions, and measure per-step error on the same observable features used
for sys-ID / MMD.

### Procedure

1. **Collect (GT)** — Run batched collection with non-trivial grid, e.g.
   `S=2`, `D=3` (6 episodes), `--output <dataset_dir>`, capped steps sufficient
   for at least one hold phase per direction. No `--save-snapshot` (observation-only path).

2. **For each episode** in the dataset:
   - Load via `TrajectoryDataset`.
   - Build observation-only reset options from frame 0:
     `observation_reset_options_from_parquet(dataset, episode_id)` →
     `digital_twin_obs_from_episode` + `infer_params_from_obs` (same path as
     `ApplePickReplayEnv` / `example_gym_replay_overrides.py`).
   - Construct `ApplePickReplayEnv` with **identical** `fruiting_system_params` from
     episode metadata (no stiffness overrides — GT replay, not candidate eval).
   - `reset(seed=meta.seed, options=observation_options)`.
   - Open-loop step through every recorded `action` row from
     `frames/<episode_id>.parquet`.
   - At each frame, compare live obs to recorded GT via `_compare_to_dataset`
     (`apple_pick_gym/examples/example_gym_replay.py`) → `ReplayErrors`.

3. **Aggregate** — `ReplayErrorSummary` per episode; report mean/max over all
   episodes for GT feature errors:
   - `ft_wrist` (RMSE and |ΔF|, |Δτ|)
   - `tcp_pos`, `tcp_velocity`, `tcp_quat`
   - `apple_pos`, `apple_quat`
   - `woody_part_start_pos`, `woody_part_end_pos` (via woody_start / woody_end mm)
   - `robot_joint_q`

4. **Pass criteria** (initial thresholds — tighten once baseline is measured):
   - Replay completes without exception for **every** collected episode.
   - Mean `|Δtcp_pos|` < **5 mm** and mean `ft_wrist` RMSE < **2 N** over hold phases
     (or document measured baseline if physics/compliance gap is larger at first ship).
   - Hold-phase-only filter: compare frames where recorded `phase == hold` to
     suppress move-burst transients (same rationale as §2.1 steady-state logging in
     `example_gym_sysid.py`).

### Test module

| Module | Asserts |
| ------ | ------- |
| `apple_pick_gym/tests/test_batched_sysid_replay_fidelity.py` | End-to-end: batched collect (`S≥2`, `D≥2`) → per-episode observation-only replay → `_compare_to_dataset` on hold frames → thresholds above |

Reuse existing helpers; do **not** duplicate comparison logic:

- `TrajectoryDataset`, `observation_reset_options_from_parquet`, `digital_twin_obs_from_episode`
- `ApplePickReplayEnv`
- `_compare_to_dataset`, `ReplayErrorSummary` from `example_gym_replay.py`

### Validation (full slice)

```bash
# Unit + schema tests
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_batched_sysid_env.py \
  apple_pick_gym/tests/test_batched_sysid_collect.py -q

# Capstone sim-to-sim replay fidelity (slow; mark @pytest.mark.slow if needed)
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_batched_sysid_replay_fidelity.py -q
```

### Notes

- GT is the **collection env's recorded Parquet**, not a privileged snapshot —
  this validates the observation-first digital-twin reconstruction path end-to-end.
- Batched replay (`num_envs > 1` in one replay env) remains out of scope; this
  test replays episodes **sequentially** through the existing single-env replay env.
- Thresholds are starting points; the first implementation run should print
  `ReplayErrorSummary.print_summary()` and adjust constants to sit just above the
  measured GT-replay floor (same workflow as M3 sim-to-sim validation).
