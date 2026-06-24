"""Tests for sysID trajectory Parquet persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
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
        "raw_ft_wrist": np.arange(6, dtype=np.float32) + 101.0,
        "tcp_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "apple_pos": np.array([0.4, 0.5, 0.6], dtype=np.float32),
        "tcp_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "apple_quat": np.array([0.0, 0.1, 0.0, 0.995], dtype=np.float32),
        "robot_joint_q": np.linspace(0.0, 0.6, 7, dtype=np.float32),
        "woody_part_force": np.arange(n_woody * 6, dtype=np.float32),
    }


def _synthetic_fruiting_params_json() -> str:
    import apple_pick_sim.fruiting_system as fs

    ranges = fs.load_ranges(
        Path(__file__).resolve().parents[1] / "fixtures" / "fruiting_system_ranges_straight_rod_test.json"
    )
    return fs.fruiting_params_to_json(fs.sample_params(ranges, seed=42))


def _synthetic_meta(
    *,
    episode_id: str = "ep-001",
    fruiting_system_params: str | None = None,
) -> EpisodeMeta:
    return EpisodeMeta(
        episode_id=episode_id,
        weld_direction=(0.0, 0.0, 1.0),
        excitation_type="quasi_static",
        n_woody_parts=2,
        junction_names=["joint_0", "joint_1"],
        params_fingerprint=json.dumps({"stem_bend_stiffness": 30.0}),
        fruiting_system_params=fruiting_system_params,
        control_hz=60.0,
        timestamp="2026-06-16T12:00:00Z",
        seed=42,
        n_directions=1,
        initial_tcp_pos=(0.0, 0.0, 0.5),
        initial_tcp_quat=(0.0, 0.0, 0.0, 1.0),
        initial_apple_pos=(0.4, 0.5, 0.6),
        initial_apple_quat=(0.0, 0.1, 0.0, 0.995),
        initial_robot_joint_q=tuple(float(x) for x in np.linspace(0.0, 0.6, 7)),
        fixture_path="fixtures/example.json",
        fruiting_base_pos=(0.0, 0.2, 1.3),
        apple_radius=0.04,
        rod_radii={"primary": 0.012, "secondary": 0.01, "spur": 0.008, "stem": 0.004},
        weld_reference_pos=(0.4, 0.5, 0.6),
        weld_reference_quat=(0.0, 0.1, 0.0, 0.995),
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
    assert table.column("tcp_quat")[0].as_py() == [0.0, 0.0, 0.0, 1.0]
    assert table.column("apple_quat")[0].as_py() == pytest.approx([0.0, 0.1, 0.0, 0.995])
    assert table.column("robot_joint_q")[0].as_py() == pytest.approx(
        np.linspace(0.0, 0.6, 7, dtype=np.float32).tolist()
    )
    assert table.column("raw_ft_wrist")[0].as_py() == pytest.approx(
        (np.arange(6, dtype=np.float32) + 101.0).tolist()
    )


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
    assert table.column("fruiting_base_pos")[0].as_py() == [0.0, 0.2, 1.3]
    assert table.column("apple_radius")[0].as_py() == 0.04
    assert table.column("initial_tcp_quat")[0].as_py() == [0.0, 0.0, 0.0, 1.0]
    assert table.column("initial_apple_pos")[0].as_py() == [0.4, 0.5, 0.6]
    assert table.column("initial_apple_quat")[0].as_py() == [0.0, 0.1, 0.0, 0.995]
    assert table.column("initial_robot_joint_q")[0].as_py() == pytest.approx(
        np.linspace(0.0, 0.6, 7).tolist()
    )
    assert json.loads(table.column("rod_radii")[0].as_py()) == {
        "primary": 0.012,
        "secondary": 0.01,
        "spur": 0.008,
        "stem": 0.004,
    }
    assert table.column("weld_reference_pos")[0].as_py() == [0.4, 0.5, 0.6]
    assert table.column("weld_reference_quat")[0].as_py() == [0.0, 0.1, 0.0, 0.995]


def test_metadata_writes_fruiting_system_params_column(tmp_path: Path):
    params_json = _synthetic_fruiting_params_json()
    writer = TrajectoryWriter(episode_id="ep-params")
    _record_synthetic_frames(writer, n=2)
    writer.save(
        tmp_path,
        _synthetic_meta(
            episode_id="ep-params",
            fruiting_system_params=params_json,
        ),
    )

    table = pq.read_table(tmp_path / "metadata.parquet")
    assert table.column("fruiting_system_params")[0].as_py() == params_json

    dataset = TrajectoryDataset(tmp_path)
    loaded_meta = dataset.load_episode_meta("ep-params")
    assert loaded_meta["fruiting_system_params"] == params_json


def test_metadata_append_promotes_legacy_schema(tmp_path: Path):
    legacy_row = _synthetic_meta(episode_id="legacy").to_row()
    for key in (
        "fruiting_base_pos",
        "fruiting_system_params",
        "initial_tcp_quat",
        "initial_apple_pos",
        "initial_apple_quat",
        "initial_robot_joint_q",
        "apple_radius",
        "rod_radii",
        "weld_reference_pos",
        "weld_reference_quat",
        "movement_per_step_m",
        "total_movement_m",
        "hold_duration_s",
        "move_speed_mps",
        "skip_return",
    ):
        legacy_row.pop(key)
    pq.write_table(pa.Table.from_pylist([legacy_row]), tmp_path / "metadata.parquet")

    writer = TrajectoryWriter(episode_id="new")
    _record_synthetic_frames(writer, n=2)
    writer.save(tmp_path, _synthetic_meta(episode_id="new"))

    table = pq.read_table(tmp_path / "metadata.parquet")
    ids = table.column("episode_id").to_pylist()
    assert ids == ["legacy", "new"]
    assert table.column("fruiting_base_pos")[ids.index("legacy")].as_py() is None
    assert table.column("fruiting_system_params")[ids.index("legacy")].as_py() is None
    assert table.column("fruiting_base_pos")[ids.index("new")].as_py() == [0.0, 0.2, 1.3]
    assert table.column("rod_radii")[ids.index("legacy")].as_py() is None
    assert json.loads(table.column("rod_radii")[ids.index("new")].as_py()) == {
        "primary": 0.012,
        "secondary": 0.01,
        "spur": 0.008,
        "stem": 0.004,
    }


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
    np.testing.assert_allclose(arrays["sim_time"], np.arange(4, dtype=np.float32) / 60.0)
    np.testing.assert_allclose(
        arrays["amplitude_m"],
        np.array([0.05, 0.10, 0.15, 0.20], dtype=np.float32),
    )
    assert arrays["action"].shape == (4, 6)
    assert arrays["ft_wrist"].shape == (4, 6)
    assert arrays["raw_ft_wrist"].shape == (4, 6)
    assert arrays["tcp_pos"].shape == (4, 3)
    assert arrays["tcp_quat"].shape == (4, 4)
    assert arrays["apple_quat"].shape == (4, 4)
    assert arrays["robot_joint_q"].shape == (4, 7)
    assert set(arrays["woody_part_start_pos"].keys()) == {"joint_0", "joint_1"}
    assert arrays["woody_part_start_pos"]["joint_0"].shape == (4, 3)
    flat = stack_woody_pos_frame(
        arrays["woody_part_start_pos"], 0, arrays["junction_names"]
    )
    assert flat.shape == (6,)
    np.testing.assert_array_equal(arrays["phase"], np.array([0, 1, 2, 1], dtype=np.int8))
    np.testing.assert_array_equal(arrays["dir_idx"], np.zeros(4, dtype=np.int32))
    np.testing.assert_array_equal(
        arrays["excitation_type"], np.zeros(4, dtype=np.int8)
    )
    np.testing.assert_allclose(
        arrays["excitation_direction"],
        np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (4, 1)),
    )


def test_digital_twin_obs_from_episode_uses_metadata_and_frame_zero(tmp_path: Path):
    from apple_pick_sim.system_id.parquet_init import digital_twin_obs_from_episode

    writer = TrajectoryWriter(episode_id="ep-init")
    _record_synthetic_frames(writer, n=2)
    writer.save(tmp_path, _synthetic_meta(episode_id="ep-init"))

    dataset = TrajectoryDataset(tmp_path)
    obs = digital_twin_obs_from_episode(dataset, "ep-init")

    assert obs.fruiting_base_pos == (0.0, 0.2, 1.3)
    assert obs.weld_direction == (0.0, 0.0, 1.0)
    assert obs.junction_names == ["joint_0", "joint_1"]
    assert obs.apple_radius == 0.04
    assert obs.rod_radii == {
        "primary": 0.012,
        "secondary": 0.01,
        "spur": 0.008,
        "stem": 0.004,
    }
    np.testing.assert_allclose(obs.woody_part_start_pos[:3], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(obs.woody_part_end_pos[:3], [0.5, 1.5, 2.5])


def test_missing_dataset_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="metadata.parquet"):
        TrajectoryDataset(tmp_path / "missing")
