# Batched sys-ID dataset (v1)

Parallel quasi-static collection from
[`example_batched_collect_sysid_data.py`](../apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py)
writes a **batched_sysid_v1** layout optimized for training and analysis over a
`num_structures × num_directions` grid.

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
| `collection` | `seed`, `topology_seed`, `ranges_path`, `control_hz`, `num_structures`, `num_directions`, `max_steps`, `trajectory` |
| `structures[]` | Light summary per `structure_idx`: `params_fingerprint`, `junction_names`, `n_woody_parts` |
| `episodes[]` | Catalog entry per episode: indices, `filename`, `episode_id`, `pull_direction`, `n_frames` |

Full `fruiting_system_params` live in each episode parquet (not duplicated in `structures[]`).

## Episode parquet

### Schema metadata (once per file)

Stored as JSON-encoded strings in Parquet schema metadata. Keys include:

`schema_version`, `episode_id`, `structure_idx`, `direction_idx`, `env_idx`,
`pull_direction`, `params_fingerprint`, `fruiting_system_params`, `excitation_type`,
`control_hz`, `seed`, `n_woody_parts`, `junction_names`, reset poses
(`initial_tcp_pos/quat`, `initial_apple_pos/quat`, `initial_robot_joint_q`),
`fixture_path`, weld/placement fields, and quasi-static trajectory knobs.

### Frame table (one row per sim step)

Required columns:

| Column | Shape | Description |
|--------|-------|-------------|
| `step_idx` | int | Frame index |
| `phase` | int8 | 0=move_out, 1=hold, 2=return |
| `excitation_type` | int8 | 0=quasi_static |
| `excitation_direction` | f32×3 | Pull direction unit vector |
| `action` | f32×6 | EE velocity command |
| `tcp_velocity` | f32×6 | Measured TCP twist |
| `ft_wrist` | f32×6 | Wrist F/T |

Bonus columns: `sim_time`, `amplitude_m`, `raw_ft_wrist`, `tcp_pos`, `tcp_quat`,
`apple_pos`, `apple_quat`, `robot_joint_q`, `woody_part_force`, plus dynamic
`woody_start__<junction>` / `woody_end__<junction>` pairs.

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

**Next (V.4.2.1):** verify replay through the digital-twin initializer —
`digital_twin_obs_from_episode` (frame-0 woody anchors) plus `params_fingerprint`
/ fixture metadata from episode parquet, without relying on the full params blob alone.
Helpers live in `apple_pick_sim/system_id/parquet_init.py`. Native v1 replay
(without legacy materialize) remains a follow-up (V.4.4).

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

## Tests

- `apple_pick_sim/tests/test_batched_trajectory_store.py` — writer/loader roundtrip
- `apple_pick_gym/tests/test_batched_sysid_collect.py` — end-to-end collect
- `apple_pick_gym/tests/test_batched_sysid_replay_fidelity.py` — collect → legacy materialize → replay (full `fruiting_system_params`; digital-twin frame-0 path is V.4.2.1)
