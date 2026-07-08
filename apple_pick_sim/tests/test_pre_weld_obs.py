"""Tests for pre-weld sys-ID observation helpers."""

from __future__ import annotations

import numpy as np

from apple_pick_sim.system_id.batched_trajectory_store import (
    PRE_WELD_STEP_IDX,
    FIRST_TRAJECTORY_STEP_IDX,
    BatchedEpisodeWriter,
    frame_index_for_step,
)
from apple_pick_sim.system_id.pre_weld_obs import complete_pre_weld_sysid_obs
from apple_pick_sim.system_id.trajectory_store import phase_to_int


def test_phase_to_int_includes_pre_weld():
    assert phase_to_int("pre_weld") == -1


def test_frame_index_for_step_selects_logical_row():
    arrays = {
        "step_idx": np.asarray([-1, 0, 1], dtype=np.int32),
    }
    assert frame_index_for_step(arrays, PRE_WELD_STEP_IDX) == 0
    assert frame_index_for_step(arrays, FIRST_TRAJECTORY_STEP_IDX) == 1
    assert frame_index_for_step(arrays, 99, fallback=2) == 2


def test_batched_writer_records_pre_weld_row(tmp_path):
    from apple_pick_sim.tests.test_batched_trajectory_store import (
        _synthetic_episode_metadata,
        _synthetic_obs,
    )

    writer = BatchedEpisodeWriter(episode_id="ep-pre-weld")
    tree_obs = {
        "woody_part_start_pos": {"joint_0": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
        "woody_part_end_pos": {"joint_0": np.array([1.1, 2.1, 3.1], dtype=np.float32)},
        "apple_pos": np.array([0.4, 0.5, 0.6], dtype=np.float32),
        "apple_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }
    writer.record_step(
        step_idx=PRE_WELD_STEP_IDX,
        sim_time=0.0,
        phase="pre_weld",
        amplitude_m=0.0,
        action=np.zeros(6, dtype=np.float32),
        obs=complete_pre_weld_sysid_obs(tree_obs, pull_direction=[0.0, 1.0, 0.0]),
    )
    writer.record_step(
        step_idx=0,
        sim_time=1 / 30.0,
        phase="move_out",
        amplitude_m=0.01,
        action=np.ones(6, dtype=np.float32),
        obs=_synthetic_obs(n_woody=1),
    )
    path = writer.save(tmp_path / "ep.parquet", _synthetic_episode_metadata())
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    assert table.column("step_idx").to_pylist() == [-1, 0]
    assert table.column("phase").to_pylist() == [phase_to_int("pre_weld"), phase_to_int("move_out")]
