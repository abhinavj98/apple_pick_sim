"""Tests for hold-only MMD feature construction."""

from __future__ import annotations

import numpy as np
import pytest

from apple_pick_sim.system_id.mmd_features import (
    ReplayObservationCollector,
    build_state_matrix,
    build_transition_features_by_direction,
    flatten_woody_positions,
)


def _arrays_for_steps(*, steps: int, junction_names: list[str] | None = None) -> dict:
    junction_names = junction_names or ["joint_b", "joint_a"]
    base = np.arange(steps, dtype=np.float32).reshape(steps, 1)
    woody_start = {
        "joint_a": np.hstack([base + 100.0, base + 101.0, base + 102.0]).astype(
            np.float32
        ),
        "joint_b": np.hstack([base + 200.0, base + 201.0, base + 202.0]).astype(
            np.float32
        ),
    }
    woody_end = {
        "joint_a": np.hstack([base + 300.0, base + 301.0, base + 302.0]).astype(
            np.float32
        ),
        "joint_b": np.hstack([base + 400.0, base + 401.0, base + 402.0]).astype(
            np.float32
        ),
    }
    return {
        "ft_wrist": np.hstack([base + i for i in range(6)]).astype(np.float32),
        "tcp_velocity": np.hstack([base + 10.0 + i for i in range(6)]).astype(
            np.float32
        ),
        "action": np.hstack([base + 20.0 + i for i in range(6)]).astype(np.float32),
        "tcp_pos": np.hstack([base + 30.0 + i for i in range(3)]).astype(np.float32),
        "apple_pos": np.hstack([base + 40.0 + i for i in range(3)]).astype(np.float32),
        "woody_part_start_pos": woody_start,
        "woody_part_end_pos": woody_end,
        "excitation_direction": np.tile(
            np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (steps, 1)
        ),
        "phase": np.ones(steps, dtype=np.int8),
        "excitation_type": np.zeros(steps, dtype=np.int8),
        "dir_idx": np.zeros(steps, dtype=np.int32),
        "junction_names": junction_names,
    }


def test_flatten_woody_positions_uses_junction_names_order():
    arrays = _arrays_for_steps(steps=1, junction_names=["joint_b", "joint_a"])

    flat = flatten_woody_positions(
        arrays["woody_part_start_pos"],
        frame_idx=0,
        junction_names=arrays["junction_names"],
    )

    np.testing.assert_allclose(flat, [200.0, 201.0, 202.0, 100.0, 101.0, 102.0])


def test_build_state_matrix_uses_exact_feature_order():
    arrays = _arrays_for_steps(steps=1, junction_names=["joint_b", "joint_a"])

    state = build_state_matrix(arrays)

    expected = np.array(
        [
            # ft_wrist
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            # tcp_velocity
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            # action
            20.0,
            21.0,
            22.0,
            23.0,
            24.0,
            25.0,
            # tcp_pos, apple_pos
            30.0,
            31.0,
            32.0,
            40.0,
            41.0,
            42.0,
            # woody starts in junction_names order: joint_b then joint_a
            200.0,
            201.0,
            202.0,
            100.0,
            101.0,
            102.0,
            # woody ends in junction_names order: joint_b then joint_a
            400.0,
            401.0,
            402.0,
            300.0,
            301.0,
            302.0,
            # excitation_direction, phase, excitation_type
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
        ],
        dtype=np.float32,
    )
    assert state.shape == (1, expected.size)
    np.testing.assert_allclose(state[0], expected)


def test_transition_features_are_hold_only_per_direction_and_segment():
    arrays = _arrays_for_steps(steps=14, junction_names=["joint_a"])
    arrays["dir_idx"] = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int32
    )
    arrays["phase"] = np.array(
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 1], dtype=np.int8
    )

    by_direction = build_transition_features_by_direction(arrays)

    assert set(by_direction) == {0, 1}
    state = build_state_matrix(arrays)
    expected_dir0 = np.concatenate([state[3], state[4] - state[3]])
    expected_dir1 = np.concatenate([state[10], state[11] - state[10]])
    assert by_direction[0].shape == (1, state.shape[1] * 2)
    assert by_direction[1].shape == (1, state.shape[1] * 2)
    np.testing.assert_allclose(by_direction[0][0], expected_dir0)
    np.testing.assert_allclose(by_direction[1][0], expected_dir1)


def test_transition_features_fail_when_required_field_is_missing():
    arrays = _arrays_for_steps(steps=2)
    arrays.pop("tcp_pos")

    with pytest.raises(KeyError, match="tcp_pos"):
        build_state_matrix(arrays)


def test_replay_observation_collector_builds_dataset_shaped_arrays():
    recorded = _arrays_for_steps(steps=2, junction_names=["joint_a", "joint_b"])
    recorded["phase"] = np.array([0, 1], dtype=np.int8)
    recorded["dir_idx"] = np.array([0, 1], dtype=np.int32)
    collector = ReplayObservationCollector(recorded)
    obs = {
        "ft_wrist": np.arange(6, dtype=np.float32) + 1000.0,
        "tcp_velocity": np.arange(6, dtype=np.float32) + 2000.0,
        "tcp_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "apple_pos": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        "woody_start": np.array(
            [10.0, 11.0, 12.0, 20.0, 21.0, 22.0], dtype=np.float32
        ),
        "woody_end": np.array(
            [30.0, 31.0, 32.0, 40.0, 41.0, 42.0], dtype=np.float32
        ),
    }

    collector.record(obs, frame_idx=1)
    arrays = collector.to_arrays()

    np.testing.assert_allclose(arrays["action"], recorded["action"][1:2])
    np.testing.assert_array_equal(arrays["phase"], np.array([1], dtype=np.int8))
    np.testing.assert_array_equal(arrays["dir_idx"], np.array([1], dtype=np.int32))
    np.testing.assert_allclose(arrays["ft_wrist"][0], obs["ft_wrist"])
    np.testing.assert_allclose(
        arrays["woody_part_start_pos"]["joint_a"][0], [10.0, 11.0, 12.0]
    )
    np.testing.assert_allclose(
        arrays["woody_part_start_pos"]["joint_b"][0], [20.0, 21.0, 22.0]
    )
