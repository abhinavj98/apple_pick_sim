"""Tests for hold-only MMD feature construction."""

from __future__ import annotations

import numpy as np
import pytest

from apple_pick_sim.system_id.mmd_features import (
    ReplayObservationCollector,
    build_state_matrix,
    build_transition_features_by_direction,
    combine_transition_features,
    flatten_woody_positions,
    iter_kept_hold_segments,
    replay_obs_dict_from_sysid_numpy,
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
            # woody_bending_angles in junction_names order: joint_b then joint_a
            0.0,
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
    arrays["excitation_direction"] = np.vstack([
        np.tile([1.0, 0.0, 0.0], (8, 1)),
        np.tile([0.0, 1.0, 0.0], (6, 1))
    ]).astype(np.float32)
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


def test_replay_obs_dict_from_sysid_numpy_flattens_woody():
    sysid_obs = {
        "ft_wrist": np.arange(6, dtype=np.float32),
        "tcp_velocity": np.arange(6, 12, dtype=np.float32),
        "tcp_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "apple_pos": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        "woody_part_start_pos": {
            "joint_b": np.array([7.0, 8.0, 9.0], dtype=np.float32),
            "joint_a": np.array([10.0, 11.0, 12.0], dtype=np.float32),
        },
        "woody_part_end_pos": {
            "joint_b": np.array([13.0, 14.0, 15.0], dtype=np.float32),
            "joint_a": np.array([16.0, 17.0, 18.0], dtype=np.float32),
        },
    }
    out = replay_obs_dict_from_sysid_numpy(sysid_obs, junction_names=["joint_b", "joint_a"])
    np.testing.assert_allclose(out["woody_start"], [7, 8, 9, 10, 11, 12])
    np.testing.assert_allclose(out["woody_end"], [13, 14, 15, 16, 17, 18])


def test_replay_obs_dict_from_sysid_numpy_matches_collector_contract():
    recorded = _arrays_for_steps(steps=2, junction_names=["joint_a", "joint_b"])
    recorded["phase"] = np.array([0, 1], dtype=np.int8)
    recorded["dir_idx"] = np.array([0, 1], dtype=np.int32)
    junction_names = recorded["junction_names"]
    frame_idx = 1

    sysid_obs = {
        "ft_wrist": recorded["ft_wrist"][frame_idx],
        "tcp_velocity": recorded["tcp_velocity"][frame_idx],
        "tcp_pos": recorded["tcp_pos"][frame_idx],
        "apple_pos": recorded["apple_pos"][frame_idx],
        "woody_part_start_pos": {
            name: recorded["woody_part_start_pos"][name][frame_idx]
            for name in junction_names
        },
        "woody_part_end_pos": {
            name: recorded["woody_part_end_pos"][name][frame_idx]
            for name in junction_names
        },
    }
    adapted = replay_obs_dict_from_sysid_numpy(sysid_obs, junction_names=junction_names)

    direct_collector = ReplayObservationCollector(recorded)
    direct_obs = {
        "ft_wrist": sysid_obs["ft_wrist"],
        "tcp_velocity": sysid_obs["tcp_velocity"],
        "tcp_pos": sysid_obs["tcp_pos"],
        "apple_pos": sysid_obs["apple_pos"],
        "woody_start": flatten_woody_positions(
            recorded["woody_part_start_pos"],
            frame_idx=frame_idx,
            junction_names=junction_names,
        ),
        "woody_end": flatten_woody_positions(
            recorded["woody_part_end_pos"],
            frame_idx=frame_idx,
            junction_names=junction_names,
        ),
    }
    direct_collector.record(direct_obs, frame_idx=frame_idx)

    adapted_collector = ReplayObservationCollector(recorded)
    adapted_collector.record(adapted, frame_idx=frame_idx)

    direct_arrays = direct_collector.to_arrays()
    adapted_arrays = adapted_collector.to_arrays()
    for name in junction_names:
        np.testing.assert_allclose(
            adapted_arrays["woody_part_start_pos"][name],
            direct_arrays["woody_part_start_pos"][name],
        )
        np.testing.assert_allclose(
            adapted_arrays["woody_part_end_pos"][name],
            direct_arrays["woody_part_end_pos"][name],
        )


def test_replay_observation_collector_stable_column():
    recorded = _arrays_for_steps(steps=2, junction_names=["joint_a", "joint_b"])
    recorded["phase"] = np.array([0, 1], dtype=np.int8)
    recorded["dir_idx"] = np.array([0, 1], dtype=np.int32)
    collector = ReplayObservationCollector(recorded)
    obs = {
        "ft_wrist": np.arange(6, dtype=np.float32) + 1000.0,
        "tcp_velocity": np.arange(6, dtype=np.float32) + 2000.0,
        "tcp_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "apple_pos": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        "woody_start": np.array([10.0, 11.0, 12.0, 20.0, 21.0, 22.0], dtype=np.float32),
        "woody_end": np.array([30.0, 31.0, 32.0, 40.0, 41.0, 42.0], dtype=np.float32),
    }
    collector.record(obs, frame_idx=1, stable=False)
    arrays = collector.to_arrays()
    assert arrays["stable"].tolist() == [False]


def test_replay_observation_collector_oob_frame_raises():
    recorded = _arrays_for_steps(steps=2, junction_names=["joint_a"])
    recorded["phase"] = np.array([0, 1], dtype=np.int8)
    recorded["dir_idx"] = np.array([0, 0], dtype=np.int32)
    collector = ReplayObservationCollector(recorded)
    obs = {
        "ft_wrist": np.arange(6, dtype=np.float32),
        "tcp_velocity": np.arange(6, dtype=np.float32),
        "tcp_pos": np.zeros(3, dtype=np.float32),
        "apple_pos": np.zeros(3, dtype=np.float32),
        "woody_start": np.zeros(3, dtype=np.float32),
        "woody_end": np.zeros(3, dtype=np.float32),
    }
    with pytest.raises(IndexError, match="frame_idx"):
        collector.record(obs, frame_idx=2)
    with pytest.raises(IndexError, match="frame_idx"):
        collector.record(obs, frame_idx=-1)


def test_transition_features_exclude_unstable_hold_frame():
    arrays = _arrays_for_steps(steps=8, junction_names=["joint_a"])
    arrays["dir_idx"] = np.zeros(8, dtype=np.int32)
    arrays["excitation_direction"] = np.tile([1.0, 0.0, 0.0], (8, 1)).astype(np.float32)
    arrays["phase"] = np.array([0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8)
    arrays["stable"] = np.array([True, False, False, False, False, False, False, False], dtype=bool)
    by_direction = build_transition_features_by_direction(arrays)
    assert by_direction == {}


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


def test_iter_kept_hold_segments_discards_first_half_and_requires_two_frames():
    phase = np.array([0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8)
    dir_idx = np.zeros(8, dtype=np.int32)

    segments = iter_kept_hold_segments(phase=phase, dir_idx=dir_idx, direction=0)

    assert len(segments) == 1
    assert segments[0].tolist() == [5, 6, 7]


def test_combine_transition_features_concatenates_episodes_per_direction():
    ep0 = _arrays_for_steps(steps=8, junction_names=["joint_a"])
    ep1 = _arrays_for_steps(steps=8, junction_names=["joint_a"])
    ep1["ft_wrist"] = ep1["ft_wrist"] + 1000.0

    single = combine_transition_features([ep0])
    combined = combine_transition_features([ep0, ep1])
    direction = 0

    assert direction in single
    assert combined[direction].shape[0] == single[direction].shape[0] * 2


def test_transition_features_exclude_pre_weld_row_after_strip():
    from apple_pick_gym.batched_envs import batched_sysid_mmd_grid as grid

    arrays = _arrays_for_steps(steps=8, junction_names=["joint_a"])
    arrays = {
        **arrays,
        "step_idx": np.array([-1, *range(8)], dtype=np.int32),
        "phase": np.array([-1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8),
        "ft_wrist": np.vstack(
            [np.full(6, -999.0, dtype=np.float32), arrays["ft_wrist"]]
        ),
    }
    for key in ("tcp_velocity", "action", "tcp_pos", "apple_pos"):
        arrays[key] = np.vstack(
            [np.zeros((1, arrays[key].shape[1]), dtype=np.float32), arrays[key]]
        )
    for woody_key in ("woody_part_start_pos", "woody_part_end_pos"):
        for name in arrays[woody_key]:
            arrays[woody_key][name] = np.vstack(
                [
                    np.zeros((1, 3), dtype=np.float32),
                    arrays[woody_key][name],
                ]
            )
    arrays["excitation_direction"] = np.vstack(
        [arrays["excitation_direction"][0:1], arrays["excitation_direction"]]
    )
    arrays["excitation_type"] = np.hstack(
        [arrays["excitation_type"][0:1], arrays["excitation_type"]]
    )
    arrays["dir_idx"] = np.hstack([arrays["dir_idx"][0:1], arrays["dir_idx"]])

    stripped = grid.strip_pre_weld_rows(arrays)
    by_direction = build_transition_features_by_direction(stripped)

    assert by_direction
    for features in by_direction.values():
        assert not np.any(features[:, :6] == -999.0)
