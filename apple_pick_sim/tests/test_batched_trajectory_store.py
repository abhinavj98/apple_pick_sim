"""Tests for batched sys-ID trajectory Parquet persistence (v1 layout)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from apple_pick_sim.system_id.batched_trajectory_store import (
    BATCHED_REQUIRED_FRAME_COLUMNS,
    BatchedEpisodeWriter,
    BatchedSysIdDataset,
    PRE_WELD_STEP_IDX,
    SCHEMA_VERSION,
    batched_dataset_exists,
    episode_filename,
    materialize_legacy_episode_dir,
    resolve_batched_dataset_output_dir,
    schema_metadata_to_episode_metadata,
    write_manifest,
    woody_end_column,
    woody_start_column,
)
from apple_pick_sim.system_id.trajectory_store import TrajectoryDataset


def _synthetic_obs(*, n_woody: int = 2) -> dict:
    names = [f"joint_{i}" for i in range(n_woody)]
    return {
        "excitation_type": 0,
        "excitation_direction": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "tcp_velocity": np.arange(6, dtype=np.float32),
        "woody_part_start_pos": {
            name: np.arange(3, dtype=np.float32) + float(i * 3)
            for i, name in enumerate(names)
        },
        "woody_part_end_pos": {
            name: np.arange(3, dtype=np.float32) + float(i * 3) + 0.5
            for i, name in enumerate(names)
        },
        "ft_wrist": np.arange(6, dtype=np.float32) + 1.0,
        "raw_ft_wrist": np.arange(6, dtype=np.float32) + 101.0,
        "tcp_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "apple_pos": np.array([0.4, 0.5, 0.6], dtype=np.float32),
        "tcp_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "apple_quat": np.array([0.0, 0.1, 0.0, 0.995], dtype=np.float32),
        "robot_joint_q": np.linspace(0.0, 0.6, 7, dtype=np.float32),
        "woody_part_force": np.arange(n_woody * 6, dtype=np.float32),
    }


def _synthetic_episode_metadata(*, episode_id: str = "ep-batched-001") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "structure_idx": 1,
        "direction_idx": 2,
        "env_idx": 5,
        "pull_direction": [0.0, 1.0, 0.0],
        "params_fingerprint": json.dumps({"topology": "t_junction"}),
        "fruiting_system_params": json.dumps({"topology": "t_junction", "apple_radius": 0.04}),
        "excitation_type": "quasi_static",
        "control_hz": 60.0,
        "seed": 7,
        "n_woody_parts": 2,
        "junction_names": ["joint_0", "joint_1"],
        "initial_tcp_pos": [0.0, 0.0, 0.5],
        "initial_tcp_quat": [0.0, 0.0, 0.0, 1.0],
        "initial_apple_pos": [0.4, 0.5, 0.6],
        "initial_apple_quat": [0.0, 0.1, 0.0, 0.995],
        "initial_robot_joint_q": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "fixture_path": "fixtures/example.json",
        "fruiting_base_pos": None,
        "apple_radius": 0.04,
        "rod_radii": json.dumps({"primary": 0.01}),
        "weld_direction": [0.0, 0.0, 1.0],
        "weld_reference_pos": [0.4, 0.5, 0.6],
        "weld_reference_quat": [0.0, 0.1, 0.0, 0.995],
        "movement_per_step_m": 0.02,
        "total_movement_m": 0.10,
        "hold_duration_s": 1.5,
        "move_speed_mps": 0.2,
        "skip_return": True,
    }


def test_episode_filename_formatting():
    assert episode_filename(0, 0) == "episodes/s00_d00.parquet"
    assert episode_filename(12, 3) == "episodes/s12_d03.parquet"


def test_resolve_batched_dataset_output_dir_fresh_path(tmp_path: Path):
    out = tmp_path / "dataset"
    assert resolve_batched_dataset_output_dir(out) == out


def test_resolve_batched_dataset_output_dir_append_timestamp(tmp_path: Path):
    existing = tmp_path / "dataset"
    write_manifest(
        existing,
        command_argv=["x"],
        collection={},
        structures=[],
        episodes=[],
    )
    fixed = datetime(2026, 7, 6, 19, 30, 45, tzinfo=timezone.utc)
    with pytest.warns(UserWarning, match="already exists"):
        redirected = resolve_batched_dataset_output_dir(
            existing,
            append_timestamp=True,
            now=fixed,
        )
    assert redirected == tmp_path / "dataset_20260706T193045Z"
    assert batched_dataset_exists(existing)
    assert not batched_dataset_exists(redirected)


def test_resolve_batched_dataset_output_dir_refuses_without_options(tmp_path: Path):
    existing = tmp_path / "dataset"
    write_manifest(
        existing,
        command_argv=["x"],
        collection={},
        structures=[],
        episodes=[],
    )
    with pytest.raises(FileExistsError, match="already exists"):
        resolve_batched_dataset_output_dir(existing, append_timestamp=False)


def test_resolve_batched_dataset_output_dir_overwrite(tmp_path: Path):
    existing = tmp_path / "dataset"
    write_manifest(
        existing,
        command_argv=["x"],
        collection={},
        structures=[],
        episodes=[],
    )
    assert resolve_batched_dataset_output_dir(existing, overwrite=True) == existing


def test_batched_writer_metadata_roundtrip(tmp_path: Path):
    writer = BatchedEpisodeWriter(episode_id="ep-batched-001")
    for i in range(3):
        writer.record_step(
            step_idx=i,
            sim_time=i / 60.0,
            phase="hold",
            amplitude_m=0.01 * i,
            action=np.full(6, float(i), dtype=np.float32),
            obs=_synthetic_obs(),
        )
    ep_meta = _synthetic_episode_metadata()
    path = writer.save(tmp_path / episode_filename(1, 2), ep_meta)

    table = pq.read_table(path)
    assert table.num_rows == 3
    for col in BATCHED_REQUIRED_FRAME_COLUMNS:
        assert col in table.column_names
    assert "episode_id" not in table.column_names
    assert "dir_idx" not in table.column_names
    assert woody_start_column("joint_0") in table.column_names
    assert woody_end_column("joint_1") in table.column_names

    loaded_meta = schema_metadata_to_episode_metadata(
        pq.read_metadata(path).schema.to_arrow_schema().metadata
    )
    assert loaded_meta["episode_id"] == "ep-batched-001"
    assert loaded_meta["structure_idx"] == 1
    assert loaded_meta["direction_idx"] == 2
    assert loaded_meta["pull_direction"] == [0.0, 1.0, 0.0]


def test_write_manifest_and_dataset_loader(tmp_path: Path):
    writer = BatchedEpisodeWriter(episode_id="ep-a")
    writer.record_step(
        step_idx=0,
        sim_time=0.0,
        phase="move_out",
        amplitude_m=0.0,
        action=np.zeros(6, dtype=np.float32),
        obs=_synthetic_obs(),
    )
    writer.save(tmp_path / episode_filename(0, 0), _synthetic_episode_metadata(episode_id="ep-a"))

    write_manifest(
        tmp_path,
        command_argv=["collect.py", "--num-structures", "1"],
        collection={"seed": 0, "num_structures": 1, "num_directions": 1},
        structures=[
            {
                "structure_idx": 0,
                "params_fingerprint": "{}",
                "junction_names": ["joint_0", "joint_1"],
                "n_woody_parts": 2,
            }
        ],
        episodes=[
            {
                "structure_idx": 0,
                "direction_idx": 0,
                "env_idx": 0,
                "filename": episode_filename(0, 0),
                "episode_id": "ep-a",
                "pull_direction": [0.0, 1.0, 0.0],
                "n_frames": 1,
            }
        ],
    )

    dataset = BatchedSysIdDataset(tmp_path)
    assert dataset.manifest["schema_version"] == SCHEMA_VERSION
    assert len(dataset.episode_entries()) == 1
    arrays = dataset.load_episode_obs_arrays(0, 0)
    assert arrays["action"].shape == (1, 6)
    assert arrays["ft_wrist"].shape == (1, 6)


def test_write_manifest_refuses_overwrite(tmp_path: Path):
    write_manifest(
        tmp_path,
        command_argv=["x"],
        collection={},
        structures=[],
        episodes=[],
    )
    with pytest.raises(FileExistsError):
        write_manifest(
            tmp_path,
            command_argv=["x"],
            collection={},
            structures=[],
            episodes=[],
        )


def test_load_episode_obs_arrays_skips_missing_woody_end_columns(tmp_path: Path):
    writer = BatchedEpisodeWriter(episode_id="starts-only")
    obs = _synthetic_obs(n_woody=2)
    del obs["woody_part_end_pos"]
    writer.record_step(
        step_idx=0,
        sim_time=0.0,
        phase="hold",
        amplitude_m=0.0,
        action=np.zeros(6, dtype=np.float32),
        obs=obs,
    )
    writer.save(
        tmp_path / episode_filename(0, 0),
        _synthetic_episode_metadata(episode_id="starts-only"),
    )
    write_manifest(
        tmp_path,
        command_argv=["test"],
        collection={"seed": 0, "num_structures": 1, "num_directions": 1},
        structures=[
            {
                "structure_idx": 0,
                "params_fingerprint": "{}",
                "junction_names": ["joint_0", "joint_1"],
                "n_woody_parts": 2,
            }
        ],
        episodes=[
            {
                "structure_idx": 0,
                "direction_idx": 0,
                "env_idx": 0,
                "filename": episode_filename(0, 0),
                "episode_id": "starts-only",
                "n_frames": 1,
            }
        ],
    )
    arrays = BatchedSysIdDataset(tmp_path).load_episode_obs_arrays(0, 0)
    assert set(arrays["woody_part_start_pos"]) == {"joint_0", "joint_1"}
    assert arrays["woody_part_end_pos"] == {}


def test_load_episode_obs_arrays_fills_nan_when_end_only_on_pre_weld_row(tmp_path: Path):
    """Pre-weld row may keep full-set ends for geometry rebuild; trajectory rows don't.

    The reader must not crash on the resulting partial (per-row-null) column;
    non-pre-weld rows should read back as NaN instead.
    """
    writer = BatchedEpisodeWriter(episode_id="pre-weld-ends-only")
    pre_weld_obs = _synthetic_obs(n_woody=2)
    writer.record_step(
        step_idx=PRE_WELD_STEP_IDX,
        sim_time=0.0,
        phase="pre_weld",
        amplitude_m=0.0,
        action=np.zeros(6, dtype=np.float32),
        obs=pre_weld_obs,
    )
    trajectory_obs = _synthetic_obs(n_woody=2)
    del trajectory_obs["woody_part_end_pos"]
    writer.record_step(
        step_idx=0,
        sim_time=1 / 60.0,
        phase="hold",
        amplitude_m=0.0,
        action=np.zeros(6, dtype=np.float32),
        obs=trajectory_obs,
    )
    writer.save(
        tmp_path / episode_filename(0, 0),
        _synthetic_episode_metadata(episode_id="pre-weld-ends-only"),
    )
    write_manifest(
        tmp_path,
        command_argv=["test"],
        collection={"seed": 0, "num_structures": 1, "num_directions": 1},
        structures=[
            {
                "structure_idx": 0,
                "params_fingerprint": "{}",
                "junction_names": ["joint_0", "joint_1"],
                "n_woody_parts": 2,
            }
        ],
        episodes=[
            {
                "structure_idx": 0,
                "direction_idx": 0,
                "env_idx": 0,
                "filename": episode_filename(0, 0),
                "episode_id": "pre-weld-ends-only",
                "n_frames": 2,
            }
        ],
    )
    arrays = BatchedSysIdDataset(tmp_path).load_episode_obs_arrays(0, 0)
    end = arrays["woody_part_end_pos"]
    assert set(end) == {"joint_0", "joint_1"}
    np.testing.assert_allclose(end["joint_0"][0], pre_weld_obs["woody_part_end_pos"]["joint_0"])
    assert np.all(np.isnan(end["joint_0"][1]))


def test_materialize_legacy_episode_dir(tmp_path: Path):
    writer = BatchedEpisodeWriter(episode_id="ep-legacy")
    writer.record_step(
        step_idx=0,
        sim_time=0.0,
        phase="hold",
        amplitude_m=0.0,
        action=np.ones(6, dtype=np.float32),
        obs=_synthetic_obs(),
    )
    writer.save(tmp_path / episode_filename(0, 0), _synthetic_episode_metadata(episode_id="ep-legacy"))
    write_manifest(
        tmp_path,
        command_argv=["collect.py"],
        collection={"seed": 0},
        structures=[],
        episodes=[
            {
                "structure_idx": 0,
                "direction_idx": 0,
                "env_idx": 0,
                "filename": episode_filename(0, 0),
                "episode_id": "ep-legacy",
                "pull_direction": [0.0, 1.0, 0.0],
                "n_frames": 1,
            }
        ],
    )

    legacy_dir = tmp_path / "legacy"
    dataset = BatchedSysIdDataset(tmp_path)
    materialize_legacy_episode_dir(
        dataset,
        structure_idx=0,
        direction_idx=0,
        output_dir=legacy_dir,
    )
    legacy = TrajectoryDataset(legacy_dir)
    assert legacy.episode_ids() == ["ep-legacy"]
    frames = legacy.load_episode_frames("ep-legacy")
    assert frames.num_rows == 1
    assert "episode_id" in frames.column_names
    assert frames.column("episode_id")[0].as_py() == "ep-legacy"


def test_batched_writer_stable_column_roundtrip(tmp_path: Path):
    writer = BatchedEpisodeWriter(episode_id="ep-stable")
    writer.record_step(
        step_idx=0,
        sim_time=0.0,
        phase="hold",
        amplitude_m=0.0,
        action=np.zeros(6, dtype=np.float32),
        obs=_synthetic_obs(),
        stable=True,
    )
    writer.record_step(
        step_idx=1,
        sim_time=1 / 60.0,
        phase="hold",
        amplitude_m=0.01,
        action=np.ones(6, dtype=np.float32),
        obs=_synthetic_obs(),
        stable=False,
    )
    writer.save(
        tmp_path / episode_filename(0, 0),
        _synthetic_episode_metadata(episode_id="ep-stable"),
    )
    write_manifest(
        tmp_path,
        command_argv=["collect.py"],
        collection={"seed": 0},
        structures=[],
        episodes=[
            {
                "structure_idx": 0,
                "direction_idx": 0,
                "env_idx": 0,
                "filename": episode_filename(0, 0),
                "episode_id": "ep-stable",
                "pull_direction": [0.0, 1.0, 0.0],
                "n_frames": 2,
            }
        ],
    )
    dataset = BatchedSysIdDataset(tmp_path)
    arrays = dataset.load_episode_obs_arrays(0, 0)
    assert arrays["stable"].tolist() == [True, False]


def test_load_episode_obs_arrays_stable_defaults_when_column_missing(tmp_path: Path):
    writer = BatchedEpisodeWriter(episode_id="ep-no-stable-col")
    writer.record_step(
        step_idx=0,
        sim_time=0.0,
        phase="hold",
        amplitude_m=0.0,
        action=np.zeros(6, dtype=np.float32),
        obs=_synthetic_obs(),
    )
    path = tmp_path / episode_filename(0, 0)
    writer.save(path, _synthetic_episode_metadata(episode_id="ep-no-stable-col"))
    table = pq.read_table(path)
    names = [n for n in table.column_names if n != "stable"]
    pq.write_table(table.select(names), path)
    write_manifest(
        tmp_path,
        command_argv=["collect.py"],
        collection={"seed": 0},
        structures=[],
        episodes=[
            {
                "structure_idx": 0,
                "direction_idx": 0,
                "env_idx": 0,
                "filename": episode_filename(0, 0),
                "episode_id": "ep-no-stable-col",
                "pull_direction": [0.0, 1.0, 0.0],
                "n_frames": 1,
            }
        ],
    )
    dataset = BatchedSysIdDataset(tmp_path)
    arrays = dataset.load_episode_obs_arrays(0, 0)
    assert arrays["stable"].tolist() == [True]
