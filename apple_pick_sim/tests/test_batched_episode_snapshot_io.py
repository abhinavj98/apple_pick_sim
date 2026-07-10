"""Numpy round-trip tests for batched episode snapshot I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.system_id.batched_episode_snapshot_io import (
    SNAPSHOT_ARRAY_KEYS,
    _write_world_slice,
    initial_state_path,
    load_npz_for_direction,
    WORLD_FRAME,
)


def test_initial_state_path_layout():
    path = initial_state_path("/tmp/ds", structure_idx=3, direction_idx=7)
    assert path == Path("/tmp/ds/initial_states/s03_d07.npz")


def test_npz_round_trip_write_read(tmp_path: Path):
    arrays = {key: np.arange(i + 1, dtype=np.float32) for i, key in enumerate(SNAPSHOT_ARRAY_KEYS)}
    structure_idx = 0
    direction_idx = 2
    path = initial_state_path(tmp_path, structure_idx, direction_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, origin_frame=np.asarray(WORLD_FRAME), **arrays)

    loaded = load_npz_for_direction(tmp_path, structure_idx=structure_idx, direction_idx=direction_idx)
    assert set(loaded) == set(arrays)
    for key in arrays:
        np.testing.assert_array_equal(loaded[key], arrays[key])


def test_missing_npz_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="missing post-weld snapshot"):
        load_npz_for_direction(tmp_path, structure_idx=0, direction_idx=0)


def test_missing_origin_frame_marker_raises(tmp_path: Path):
    path = initial_state_path(tmp_path, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, robot_body_q=np.zeros((1, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="missing origin_frame marker"):
        load_npz_for_direction(tmp_path, structure_idx=0, direction_idx=0)


def test_load_rejects_template_local_marker(tmp_path: Path):
    arrays = {key: np.arange(i + 1, dtype=np.float32) for i, key in enumerate(SNAPSHOT_ARRAY_KEYS)}
    path = initial_state_path(tmp_path, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, origin_frame=np.asarray("template_local"), **arrays)
    with pytest.raises(ValueError, match="unsupported origin_frame"):
        load_npz_for_direction(tmp_path, structure_idx=0, direction_idx=0)


def test_write_world_slice_copies_world_arrays_unchanged():
    layout_collect = BatchedEnvLayout(
        num_envs=5,
        bodies_per_world=2,
        robot_bodies_per_world=2,
        joints_per_world=1,
        joint_coord_count_per_world=7,
        joint_dof_count_per_world=7,
        template_tcp_body=0,
        template_proxy_body=0,
        template_apple_body=None,
        tcp_body_indices=(0, 2, 4, 6, 8),
        proxy_body_indices=(0, 2, 4, 6, 8),
        apple_body_indices=(),
        env_spacing=(2.0, 2.0, 2.0),
    )
    layout_replay = BatchedEnvLayout(
        num_envs=15,
        bodies_per_world=2,
        robot_bodies_per_world=2,
        joints_per_world=1,
        joint_coord_count_per_world=7,
        joint_dof_count_per_world=7,
        template_tcp_body=0,
        template_proxy_body=0,
        template_apple_body=None,
        tcp_body_indices=tuple(range(0, 30, 2)),
        proxy_body_indices=tuple(range(0, 30, 2)),
        apple_body_indices=(),
        env_spacing=(2.0, 2.0, 2.0),
    )
    direction_idx = 2
    world_src = direction_idx
    world_dst = direction_idx + 10

    world_body_q = np.array(
        [
            [3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 1.0],
            [3.5, 4.5, 5.5, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    local = {
        "robot_body_q": world_body_q.copy(),
        "robot_body_qd": np.zeros((1, 6), dtype=np.float32),
        "robot_joint_q": np.zeros(7, dtype=np.float32),
        "robot_joint_qd": np.zeros(7, dtype=np.float32),
        "model_joint_q": np.zeros(7, dtype=np.float32),
        "model_joint_qd": np.zeros(7, dtype=np.float32),
        "cable_body_q_0": world_body_q.copy(),
        "cable_body_qd_0": np.zeros((1, 6), dtype=np.float32),
        "cable_body_q_1": world_body_q.copy(),
        "cable_body_qd_1": np.zeros((1, 6), dtype=np.float32),
    }
    # layout_collect has non-zero env_spacing here, but snapshot slices are stored in world frame.
    # _write_world_slice must copy body_q unchanged.

    merge = {
        "robot_body_q": np.zeros((30, 7), dtype=np.float32),
        "robot_body_qd": np.zeros((30, 6), dtype=np.float32),
        "robot_joint_q": np.zeros(105, dtype=np.float32),
        "robot_joint_qd": np.zeros(105, dtype=np.float32),
        "model_joint_q": np.zeros(105, dtype=np.float32),
        "model_joint_qd": np.zeros(105, dtype=np.float32),
        "cable_body_q_0": np.zeros((30, 7), dtype=np.float32),
        "cable_body_qd_0": np.zeros((30, 6), dtype=np.float32),
        "cable_body_q_1": np.zeros((30, 7), dtype=np.float32),
        "cable_body_qd_1": np.zeros((30, 6), dtype=np.float32),
    }
    _write_world_slice(merge, local, layout_replay, world_dst)

    rb = slice(world_dst * 2, world_dst * 2 + 2)
    np.testing.assert_allclose(merge["robot_body_q"][rb], world_body_q)
    np.testing.assert_allclose(merge["cable_body_q_0"][rb], world_body_q)
