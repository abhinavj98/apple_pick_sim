"""Tests for sysID trajectory Parquet persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from apple_pick_sim.system_id.episode_meta import EpisodeMeta
from apple_pick_sim.system_id.trajectory_store import (
    REQUIRED_FRAME_COLUMNS,
    TrajectoryDataset,
    TrajectoryWriter,
    junction_names_from_frame_columns,
    phase_to_int,
    stack_woody_pos_frame,
    woody_end_column,
    woody_start_column,
)


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
        "tcp_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "apple_pos": np.array([0.4, 0.5, 0.6], dtype=np.float32),
        "woody_part_force": np.arange(n_woody * 6, dtype=np.float32),
    }


def _synthetic_meta(*, episode_id: str = "ep-001") -> EpisodeMeta:
    return EpisodeMeta(
        episode_id=episode_id,
        weld_direction=(0.0, 0.0, 1.0),
        excitation_type="quasi_static",
        n_woody_parts=2,
        junction_names=["joint_0", "joint_1"],
        params_fingerprint=json.dumps({"stem_bend_stiffness": 30.0}),
        control_hz=60.0,
        timestamp="2026-06-16T12:00:00Z",
        seed=42,
        n_directions=1,
        initial_tcp_pos=(0.0, 0.0, 0.5),
        fixture_path="fixtures/example.json",
        movement_per_step_m=0.05,
        total_movement_m=0.10,
        hold_duration_s=1.5,
        move_speed_mps=0.2,
        skip_return=True,
    )


def _record_synthetic_frames(writer: TrajectoryWriter, *, n: int = 5) -> None:
    phases = ["move_out", "hold", "return", "hold", "move_out"]
    for i in range(n):
        writer.record_step(
            step_idx=i,
            sim_time=i / 60.0,
            phase=phases[i],
            dir_idx=0,
            amplitude_m=0.05 * (i + 1),
            action=np.full(6, float(i), dtype=np.float32),
            obs=_synthetic_obs(),
        )


def test_phase_to_int_mapping():
    assert phase_to_int("move_out") == 0
    assert phase_to_int("hold") == 1
    assert phase_to_int("return") == 2
    with pytest.raises(ValueError, match="unknown phase"):
        phase_to_int("init")


def test_writer_creates_valid_parquet(tmp_path: Path):
    writer = TrajectoryWriter(episode_id="ep-001")
    _record_synthetic_frames(writer, n=5)
    meta = _synthetic_meta()
    frames_path = writer.save(tmp_path, meta)

    assert frames_path.exists()
    table = pq.read_table(frames_path)
    assert table.num_rows == 5
    for col in REQUIRED_FRAME_COLUMNS:
        assert col in table.column_names
    assert woody_start_column("joint_0") in table.column_names
    assert woody_end_column("joint_1") in table.column_names
    assert junction_names_from_frame_columns(list(table.column_names)) == [
        "joint_0",
        "joint_1",
    ]


def test_metadata_appended_across_runs(tmp_path: Path):
    for ep_id in ("ep-a", "ep-b"):
        writer = TrajectoryWriter(episode_id=ep_id)
        _record_synthetic_frames(writer, n=2)
        writer.save(tmp_path, _synthetic_meta(episode_id=ep_id))

    meta_path = tmp_path / "metadata.parquet"
    table = pq.read_table(meta_path)
    assert table.num_rows == 2
    ids = set(table.column("episode_id").to_pylist())
    assert ids == {"ep-a", "ep-b"}
    assert table.column("junction_names")[0].as_py() == ["joint_0", "joint_1"]


def test_dataset_roundtrip(tmp_path: Path):
    writer = TrajectoryWriter(episode_id="ep-roundtrip")
    _record_synthetic_frames(writer, n=3)
    meta = _synthetic_meta(episode_id="ep-roundtrip")
    writer.save(tmp_path, meta)

    dataset = TrajectoryDataset(tmp_path)
    assert dataset.episode_ids() == ["ep-roundtrip"]

    loaded_meta = dataset.load_episode_meta("ep-roundtrip")
    assert loaded_meta["excitation_type"] == "quasi_static"
    assert loaded_meta["n_woody_parts"] == 2
    assert loaded_meta["junction_names"] == ["joint_0", "joint_1"]

    frames = dataset.load_episode_frames("ep-roundtrip")
    assert frames.num_rows == 3
    action0 = frames.column("action")[0].as_py()
    assert len(action0) == 6
    assert action0[0] == 0.0


def test_dataset_load_episode_obs_arrays(tmp_path: Path):
    writer = TrajectoryWriter(episode_id="ep-obs")
    _record_synthetic_frames(writer, n=4)
    writer.save(tmp_path, _synthetic_meta(episode_id="ep-obs"))

    dataset = TrajectoryDataset(tmp_path)
    arrays = dataset.load_episode_obs_arrays("ep-obs")
    assert arrays["action"].shape == (4, 6)
    assert arrays["ft_wrist"].shape == (4, 6)
    assert arrays["tcp_pos"].shape == (4, 3)
    assert set(arrays["woody_part_start_pos"].keys()) == {"joint_0", "joint_1"}
    assert arrays["woody_part_start_pos"]["joint_0"].shape == (4, 3)
    flat = stack_woody_pos_frame(
        arrays["woody_part_start_pos"], 0, arrays["junction_names"]
    )
    assert flat.shape == (6,)


def test_missing_dataset_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="metadata.parquet"):
        TrajectoryDataset(tmp_path / "missing")
