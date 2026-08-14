# Batched sys-ID dataset (v1)

**Last reviewed:** 2026-07-15

Parallel quasi-static collection from
[`example_batched_collect_sysid_data.py`](../apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py)
writes a **batched_sysid_v1** layout optimized for training and analysis over a
`num_structures × num_directions` grid.

**Canonical consumer (V.4.3 Done):** in-process stiffness grid
[`example_batched_sysid_mmd_grid.py`](../apple_pick_gym/batched_examples/example_batched_sysid_mmd_grid.py)
(MSE / Wasserstein + viz). Alignment contract:
[`sysid-mmd-grid-replay-alignment.md`](sysid-mmd-grid-replay-alignment.md).
Status / next (**V.5.3** held-out validation; V.5.2 CMA-ES Done): [`ROADMAP.md`](ROADMAP.md).

Single-env collection (`example_gym_sysid.py`) still uses the legacy layout in
[`sysid-trajectory-storage.md`](sysid-trajectory-storage.md).

## Layout

```
<output_dir>/
  manifest.json
  episodes/
    s00_d00.parquet
    s01_d02.parquet
    ...
```

- **`manifest.json`** — provenance, grid definition, light structure summaries, episode catalog.
- **`episodes/s{ss}_d{dd}.parquet`** — one self-contained episode (metadata once per file + frame table).

Episode filenames are deterministic from grid indices (`structure_idx`, `direction_idx`).

## Indexing

| Field | Rule |
|-------|------|
| `env_idx` | `structure_idx * num_directions + direction_idx` |
| `structure_idx` | `0 … num_structures-1` (shared material/topology per structure) |
| `direction_idx` | `0 … num_directions-1` (pull axis within structure) |

## manifest.json

Top-level keys:

| Key | Description |
|-----|-------------|
| `schema_version` | `"batched_sysid_v1"` |
| `created_at` | ISO-8601 UTC timestamp |
| `command_argv` | argv used to invoke the collector |
| `collection` | `seed`, `topology_seed`, `ranges_path`, `control_hz`, `num_structures`, `num_directions`, `max_steps`, `trajectory`, `sim_config` |
| `structures[]` | Light summary per `structure_idx`: `params_fingerprint`, `junction_names`, `n_woody_parts` |
| `episodes[]` | Catalog entry per episode: indices, `filename`, `episode_id`, `pull_direction`, `n_frames`, optional `excluded` / `excluded_reason` |

### Episode exclusion

Manifest `episodes[]` may include:

| Field | Type | Meaning |
|-------|------|---------|
| `excluded` | bool | When true, default grid load/replay skips this `(structure, direction)` |
| `excluded_reason` | string or null | e.g. `"stability_blowup"` |

Missing `excluded` is treated as false (legacy datasets).

**How `excluded` gets set (two stages):**

1. **Collect** (`collect_batched_quasi_static_dataset`): marks `excluded=True` only for
   sticky soft-disabled envs — NaN/Inf or IK-bootstrap failures via
   `EnvDisableController` + `hard_blowup_mask`. Force/torque/speed cap frames are written
   with parquet `stable=False` but **do not** exclude the episode at collect time.
2. **Offline tool** (`exclude_unstable_episodes`): exclude if already `excluded` **or** the
   fraction of `stable=False` frames is **`>` 0.25**
   (`DEFAULT_UNSTABLE_FRACTION_EXCLUDE`). A single force-cap hit must not wipe an episode.
   Already-excluded rows stay excluded (reason preserved when present).

Grid / load paths skip `excluded` episodes by default (`include_excluded=False`; CLI
`--include-excluded` for debug). Online monitor thresholds and soft-disable ownership:
[`batched-stability-monitor-design.md`](batched-stability-monitor-design.md).

```bash
# Write manifest.filtered.json (default); use --inplace to rewrite manifest.json
uv run python -m apple_pick_gym.batched_envs.exclude_unstable_episodes \
  --dataset /path/to/batched_sysid_dataset
```

Full `fruiting_system_params` live in each episode parquet (not duplicated in `structures[]`).

### `collection.sim_config`

Replay-relevant physics and controller settings from the **effective** sim build at
collection time (`env._sim.config`, not example-script constants). Omitted on legacy
datasets collected before this field existed.

| Key | Description |
|-----|-------------|
| `sub_dt` | VBD substep dt [s] |
| `settle_substeps` | Gravity settle substeps before weld |
| `settle_gravity_ramp`, `settle_max_speed_m_s` | Settle policy |
| `enable_*_collisions` | AVBD collision flags |
| `stem_coupling_gain`, `stem_force_cap_N`, `stem_torque_cap_Nm` | Stem harvest caps |
| `joint_angular_kd_overrides` | Requested per-joint angular kd overrides |
| `joint_angular_kd_applied` | Joints that matched template labels at build |
| `joint_linear_kd_overrides` | Requested per-joint linear kd overrides |
| `joint_linear_kd_applied` | Linear kd joints that matched template labels at build |
| `controller` | `mode`, speeds, `ik_iterations`, `vic_gains` |
| `robot` | `fix_to_apple`, `gripper_mass_kg` |

`control_hz` remains at `collection.control_hz` (not duplicated). Batch-size fields
(`num_envs`, `env_spacing`, …) are excluded.

Replay scripts should match these settings. When
[`replay_batched_sysid_structure`](../apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py)
is called with `replay_sim_config`, mismatches emit `warnings.warn` messages (non-fatal;
legacy datasets without `sim_config` are skipped).

## Episode parquet

### Schema metadata (once per file)

Stored as JSON-encoded strings in Parquet schema metadata. Keys include:

`schema_version`, `episode_id`, `structure_idx`, `direction_idx`, `env_idx`,
`pull_direction`, `params_fingerprint`, `fruiting_system_params`, `excitation_type`,
`control_hz`, `seed`, `n_woody_parts`, `junction_names`, reset poses
(`initial_tcp_pos/quat`, `initial_apple_pos/quat`, `initial_robot_joint_q`),
`fixture_path`, weld/placement fields, and quasi-static trajectory knobs.
Action metadata records `action_dim` (6 or 19), `action_layout` (for example
`vic_pose_v1`), and `action_compatible_with_vic_twist`.

### Frame table (one row per sim step)

Required columns:

| Column | Shape | Description |
|--------|-------|-------------|
| `step_idx` | int | Frame index |
| `phase` | int8 | 0=move_out, 1=hold, 2=return |
| `hold_number` | int32 | Scalar hold ID; converted real data derives it from `hold_index` |
| `excitation_type` | int8 | 0=quasi_static |
| `excitation_direction` | f32×3 | Pull direction unit vector |
| `action` | f32×6 or f32×19 | Required replay drive: 6D `vic` twist or 19D `vic_pose_v1`, as declared by action metadata |
| `tcp_velocity` | f32×6 | Measured TCP twist |
| `ft_wrist` | f32×6 | Wrist F/T |

Bonus columns: `sim_time`, `amplitude_m`, `raw_ft_wrist`, `tcp_pos`, `tcp_quat`,
`apple_pos`, `apple_quat`, `robot_joint_q`, `woody_part_force`, `stable` (per-frame online
monitor: `False` = blow-up/unsafe, **not** QS quality; one `False` does not imply manifest
`excluded`), plus dynamic `woody_start__<junction>` columns. Batched trajectory frames do
**not** write `woody_end__<junction>`; `woody_end` may still exist in runtime observations or
pre-weld reconstruction metadata, but it is not part of the bag/score contract.

Unlike the legacy layout, batched episode frames omit `episode_id` and `dir_idx`
(always one direction per file).

## Python API

```python
from apple_pick_sim.system_id import BatchedSysIdDataset

ds = BatchedSysIdDataset("/tmp/batched_sysid_dataset")
for ep in ds.episode_entries():
    s, d = ep["structure_idx"], ep["direction_idx"]
    meta = ds.load_episode_metadata(s, d)
    arrays = ds.load_episode_obs_arrays(s, d)
    print(ep["filename"], meta["pull_direction"], arrays["action"].shape)
```

Writers: `BatchedEpisodeWriter`, `write_manifest` in
`apple_pick_sim.system_id.batched_trajectory_store`.

## Legacy replay bridge

`materialize_legacy_episode_dir()` exports one v1 episode into legacy
`metadata.parquet` + `frames/<episode_id>.parquet` for `ApplePickReplayEnv` and
other single-episode tooling. `test_batched_sysid_replay_fidelity.py` uses this
bridge today and resets replay with full serialized `fruiting_system_params`.

**Deferred (V.4.2.1):** verify replay through the digital-twin initializer —
`digital_twin_obs_from_episode` / `digital_twin_obs_from_batched_episode` (frame-0
woody anchors) plus `params_fingerprint` / fixture metadata, without relying on
the full params blob alone. Helpers live in
`apple_pick_sim/system_id/parquet_init.py` and
`batched_digital_twin_init.py`; grid CLI `--infer-params` already wires infer
build params. V.4.4 tooling shipped; Current focus is **V.5.3** (`docs/ROADMAP.md`).

## Collect command

```bash
uv run python apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py \
  --viewer null --num-structures 2 --num-directions 3 \
  --max-steps 200 --output /tmp/batched_sysid_dataset
```

Re-running into an existing directory:

- **Default:** emits a `UserWarning` and writes to a sibling directory
  `{output}_{YYYYMMDDTHHMMSSZ}` (see `resolve_batched_dataset_output_dir`).
- **`--overwrite`:** replaces the dataset at `--output`.
- **API:** pass `append_timestamp=False` to `collect_batched_quasi_static_dataset` to
  raise `FileExistsError` instead.

## MMD grid replay export

[`example_batched_sysid_mmd_grid.py`](../apple_pick_gym/batched_examples/example_batched_sysid_mmd_grid.py)
can persist replay trajectories per stiffness candidate with
`--export-replay-dir` (optional `--export-skip-existing` to resume long runs).

Layout under the export root:

```
<export_replay_dir>/
  structure_000/
    candidates/
      c000/
        manifest.json
        episodes/s00_d00.parquet … s00_d19.parquet
      c001/
        …
  structure_001/
    …
```

Each candidate mini-dataset uses `batched_sysid_v1` episode files with
`num_structures=1` (structure index 0 inside the mini-dataset). Episode metadata
copies weld/reset fields from the source GT dataset and updates
`fruiting_system_params` / `params_fingerprint` to the candidate stiffness.

The candidate `manifest.json` adds a `replay` block:

| Key | Description |
|-----|-------------|
| `replay_schema_version` | `"batched_sysid_replay_v1"` |
| `source_dataset` | Absolute path to the GT dataset used for actions |
| `source_structure_idx` | Structure index in the GT dataset |
| `candidate_index` | Grid candidate index |
| `candidate_stiffnesses` | `{primary, secondary, spur, stem}` bend stiffnesses |

Writers: `apple_pick_sim.system_id.batched_replay_export`.

Example:

```bash
uv run python apple_pick_gym/batched_examples/example_batched_sysid_mmd_grid.py \
  --viewer null --dataset tmp/batched_sysid_dataset_20dir \
  --replay-only --export-replay-dir tmp/mmd_grid_replay \
  --export-skip-existing \
  --primary-bend-stiffness-values 0.005,1,10 \
  --secondary-bend-stiffness-values 1 \
  --spur-bend-stiffness-values 0.005,1,10 \
  --stem-bend-stiffness-values 0.005,1,100
```

## Tests

- `apple_pick_sim/tests/test_batched_trajectory_store.py` — writer/loader roundtrip
- `apple_pick_gym/tests/test_batched_sysid_collect.py` — end-to-end collect; `excluded` only for sticky soft-disable
- `apple_pick_gym/tests/test_env_disable_controller.py` — sticky soft-disable masks
- `apple_pick_gym/tests/test_exclude_unstable_episodes.py` — offline fraction `> 0.25` policy
- `apple_pick_gym/tests/test_batched_stability_monitor.py` — online thresholds / `hard_blowup_mask`
- `apple_pick_gym/tests/test_batched_sysid_replay_fidelity.py` — collect → legacy materialize → replay (full `fruiting_system_params`; infer-only fidelity floor is deferred V.4.2.1)
- `apple_pick_gym/tests/test_batched_replay_export.py` — MMD grid replay export roundtrip

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_batched_sysid_collect.py \
  apple_pick_gym/tests/test_exclude_unstable_episodes.py \
  apple_pick_gym/tests/test_env_disable_controller.py \
  apple_pick_gym/tests/test_batched_stability_monitor.py -q
```
