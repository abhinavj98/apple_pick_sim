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

    by_direction = build_transition_features_by_direction(arrays, use_median=False)

    assert set(by_direction) == {0, 1}
    state = build_state_matrix(arrays)
    # Full hold segments (no latter-half): dir0 holds [1..4] and [6,7]
    expected_dir0_first = np.concatenate([state[1], state[2] - state[1]])
    assert by_direction[0].shape[0] == 3 + 1  # 3 in first hold, 1 in second
    assert by_direction[0].shape[1] == state.shape[1] * 2
    np.testing.assert_allclose(by_direction[0][0], expected_dir0_first)


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
    """All hold frames unstable → no aggregatable samples → empty bag."""
    arrays = _arrays_for_steps(steps=8, junction_names=["joint_a"])
    arrays["dir_idx"] = np.zeros(8, dtype=np.int32)
    arrays["excitation_direction"] = np.tile([1.0, 0.0, 0.0], (8, 1)).astype(np.float32)
    arrays["phase"] = np.array([0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8)
    arrays["stable"] = np.array([True, False, False, False, False, False, False, False], dtype=bool)
    by_direction = build_transition_features_by_direction(arrays)
    assert by_direction == {}


def test_iter_kept_hold_segments_does_not_split_on_unstable():
    """stable=False mid-hold must not flush; segmentation is phase+dir only."""
    phase = np.array([0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8)
    dir_idx = np.zeros(8, dtype=np.int32)
    stable = np.array([True, True, True, False, True, True, True, True], dtype=bool)

    segments = iter_kept_hold_segments(
        phase=phase, dir_idx=dir_idx, direction=0, stable=stable
    )

    assert len(segments) == 1
    assert segments[0].tolist() == [1, 2, 3, 4, 5, 6, 7]


def test_median_features_mid_hold_unstable_keeps_one_hold_pair():
    """One unstable frame mid-hold must not invent a fake hold→hold transition."""
    arrays = _arrays_for_steps(steps=10, junction_names=["joint_a"])
    arrays["dir_idx"] = np.zeros(10, dtype=np.int32)
    arrays["phase"] = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0], dtype=np.int8)
    arrays["stable"] = np.array(
        [True, True, False, True, True, True, True, True, True, True], dtype=bool
    )
    by_direction = build_transition_features_by_direction(arrays, use_median=True)
    assert set(by_direction) == {0}
    # Two real holds → exactly one transition (not hold0a→hold0b from the glitch).
    assert by_direction[0].shape[0] == 1

    state = build_state_matrix(arrays)
    med0 = np.median(state[[1, 3]], axis=0)  # stable frames only in hold 0
    med1 = np.median(state[[5, 6, 7]], axis=0)
    expected = np.concatenate([med0, med1 - med0]).astype(np.float32)
    np.testing.assert_allclose(by_direction[0][0], expected, rtol=1e-5)


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


def test_iter_kept_hold_segments_keeps_full_hold():
    phase = np.array([0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8)
    dir_idx = np.zeros(8, dtype=np.int32)

    segments = iter_kept_hold_segments(phase=phase, dir_idx=dir_idx, direction=0)

    assert len(segments) == 1
    assert segments[0].tolist() == [1, 2, 3, 4, 5, 6, 7]


def test_median_hold_to_hold_features_use_full_hold_medians():
    """Two holds → one [median_s0, median_s1 - median_s0] row; outlier ignored."""
    arrays = _arrays_for_steps(steps=10, junction_names=["joint_a"])
    arrays["dir_idx"] = np.zeros(10, dtype=np.int32)
    arrays["phase"] = np.array(
        [0, 1, 1, 1, 0, 1, 1, 1, 0, 0], dtype=np.int8
    )
    # Outlier on first frame of hold0 ftp force dim0
    arrays["ft_wrist"][1, 0] = 1.0e6

    by_direction = build_transition_features_by_direction(arrays, use_median=True)
    assert set(by_direction) == {0}
    assert by_direction[0].shape[0] == 1

    state = build_state_matrix(arrays)
    med0 = np.median(state[[1, 2, 3]], axis=0)
    med1 = np.median(state[[5, 6, 7]], axis=0)
    expected = np.concatenate([med0, med1 - med0]).astype(np.float32)
    np.testing.assert_allclose(by_direction[0][0], expected, rtol=1e-5)
    # Median should not equal the outlier frame
    assert by_direction[0][0, 0] != pytest.approx(1.0e6)


def test_hold_id_onehot_appended_for_median_transitions():
    arrays = _arrays_for_steps(steps=10, junction_names=["joint_a"])
    arrays["dir_idx"] = np.zeros(10, dtype=np.int32)
    arrays["phase"] = np.array(
        [0, 1, 1, 1, 0, 1, 1, 1, 0, 0], dtype=np.int8
    )
    by_direction = build_transition_features_by_direction(
        arrays, use_median=True, hold_id_onehot=True, n_holds=3
    )
    row = by_direction[0][0]
    state_dim = build_state_matrix(arrays).shape[1]
    assert row.shape[0] == state_dim * 2 + 3
    np.testing.assert_allclose(row[-3:], [1.0, 0.0, 0.0])  # source hold 0


def test_one_hot_hold_id_raises_on_oob():
    from apple_pick_sim.system_id.mmd_features import _one_hot_hold_id

    with pytest.raises(ValueError, match="hold_idx"):
        _one_hot_hold_id(3, n_holds=3)


def test_one_hot_dir_id_raises_on_oob():
    from apple_pick_sim.system_id.mmd_features import _one_hot_dir_id

    with pytest.raises(ValueError, match="dir_idx"):
        _one_hot_dir_id(5, n_directions=5)


def test_dir_id_onehot_appended_for_median_transitions():
    """dir_idx one-hot uses fixed n_directions width; trailing units mark source dir."""
    arrays = _arrays_for_steps(steps=20, junction_names=["joint_a"])
    # Two holds in dir 0, then two holds in dir 2 (indexes reserved for missing dirs).
    arrays["dir_idx"] = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        dtype=np.int32,
    )
    arrays["phase"] = np.array(
        [
            0, 1, 1, 1, 0, 1, 1, 1, 0, 0,
            0, 1, 1, 1, 0, 1, 1, 1, 0, 0,
        ],
        dtype=np.int8,
    )
    by_direction = build_transition_features_by_direction(
        arrays, use_median=True, dir_id_onehot=True, n_directions=5
    )
    state_dim = build_state_matrix(arrays).shape[1]
    assert set(by_direction) == {0, 2}
    assert by_direction[0].shape == (1, state_dim * 2 + 5)
    assert by_direction[2].shape == (1, state_dim * 2 + 5)
    np.testing.assert_allclose(by_direction[0][0, -5:], [1.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(by_direction[2][0, -5:], [0.0, 0.0, 1.0, 0.0, 0.0])


def test_dir_id_onehot_after_hold_id_onehot():
    arrays = _arrays_for_steps(steps=10, junction_names=["joint_a"])
    arrays["dir_idx"] = np.full(10, 1, dtype=np.int32)
    arrays["phase"] = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0], dtype=np.int8)
    by_direction = build_transition_features_by_direction(
        arrays,
        use_median=True,
        hold_id_onehot=True,
        n_holds=3,
        dir_id_onehot=True,
        n_directions=4,
    )
    row = by_direction[1][0]
    state_dim = build_state_matrix(arrays).shape[1]
    assert row.shape[0] == state_dim * 2 + 3 + 4
    np.testing.assert_allclose(row[-7:-4], [1.0, 0.0, 0.0])  # hold 0
    np.testing.assert_allclose(row[-4:], [0.0, 1.0, 0.0, 0.0])  # dir 1


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
