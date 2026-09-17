"""Tests for sys-ID dataset dashboard data preparation."""

from __future__ import annotations

import numpy as np

from apple_pick_sim.system_id.dashboard_data import (
    build_frame_mask,
    hold_summaries,
    phase_names_for_values,
    woody_endpoint_series,
)


def _dashboard_arrays() -> dict:
    base = np.arange(6, dtype=np.float32).reshape(6, 1)
    return {
        "step_idx": np.arange(6, dtype=np.int32),
        "phase": np.array([0, 1, 1, 2, 1, 1], dtype=np.int8),
        "dir_idx": np.array([0, 0, 0, 0, 1, 1], dtype=np.int32),
        "amplitude_m": np.array([0.0, 0.02, 0.02, 0.02, 0.04, 0.04], dtype=np.float32),
        "action": np.zeros((6, 6), dtype=np.float32),
        "tcp_velocity": np.hstack([base + i for i in range(6)]).astype(np.float32),
        "ft_wrist": np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [3.0, 4.0, 0.0, 0.1, 0.2, 0.3],
                [3.0, 4.0, 0.0, 0.2, 0.3, 0.4],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 6.0, 8.0, 0.0, 0.0, 0.0],
                [0.0, 6.0, 8.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "raw_ft_wrist": np.zeros((6, 6), dtype=np.float32),
        "tcp_pos": np.array(
            [
                [0.00, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.04, 0.0, 0.0],
                [0.04, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "apple_pos": np.tile(np.array([[0.1, 0.0, 0.0]], dtype=np.float32), (6, 1)),
        "woody_part_start_pos": {
            "joint_a": np.hstack([base + 10.0 + i for i in range(3)]).astype(np.float32),
            "joint_b": np.hstack([base + 20.0 + i for i in range(3)]).astype(np.float32),
        },
        "woody_part_end_pos": {
            "joint_a": np.hstack([base + 30.0 + i for i in range(3)]).astype(np.float32),
            "joint_b": np.hstack([base + 40.0 + i for i in range(3)]).astype(np.float32),
        },
        "junction_names": ["joint_b", "joint_a"],
    }


def test_phase_names_for_values_maps_known_codes():
    assert phase_names_for_values(np.array([0, 1, 2, 9], dtype=np.int8)) == [
        "move_out",
        "hold",
        "return",
        "unknown_9",
    ]


def test_build_frame_mask_filters_direction_and_named_phases():
    arrays = _dashboard_arrays()

    mask = build_frame_mask(arrays, direction=0, phases=("hold",))

    np.testing.assert_array_equal(np.flatnonzero(mask), np.array([1, 2]))


def test_woody_endpoint_series_respects_junction_order_and_mask():
    arrays = _dashboard_arrays()
    mask = build_frame_mask(arrays, direction=0, phases=("hold",))

    series = woody_endpoint_series(arrays, mask)

    assert [item.name for item in series] == [
        "joint_b start",
        "joint_b end",
        "joint_a start",
        "joint_a end",
    ]
    np.testing.assert_allclose(series[0].xyz, [[21.0, 22.0, 23.0], [22.0, 23.0, 24.0]])
    np.testing.assert_allclose(series[1].xyz, [[41.0, 42.0, 43.0], [42.0, 43.0, 44.0]])


def test_woody_endpoint_series_omits_end_when_missing():
    """Bags no longer persist woody_part_end_pos; series should degrade to starts only."""
    arrays = _dashboard_arrays()
    del arrays["woody_part_end_pos"]
    mask = build_frame_mask(arrays, direction=0, phases=("hold",))

    series = woody_endpoint_series(arrays, mask)

    assert [item.name for item in series] == ["joint_b start", "joint_a start"]
    np.testing.assert_allclose(series[0].xyz, [[21.0, 22.0, 23.0], [22.0, 23.0, 24.0]])


def test_hold_summaries_group_by_direction_and_amplitude():
    arrays = _dashboard_arrays()

    summaries = hold_summaries(arrays, initial_tcp_pos=np.array([0.0, 0.0, 0.0]))

    assert [(s.direction, s.amplitude_m, s.n_frames) for s in summaries] == [
        (0, 0.02, 2),
        (1, 0.04, 2),
    ]
    np.testing.assert_allclose(summaries[0].mean_force_n, [3.0, 4.0, 0.0])
    assert summaries[0].force_norm_n == 5.0
    assert summaries[0].tcp_displacement_m == 0.02
    assert summaries[0].stiffness_n_per_m == 250.0
    np.testing.assert_allclose(summaries[1].mean_force_n, [0.0, 6.0, 8.0])
    assert summaries[1].force_norm_n == 10.0
    assert summaries[1].tcp_displacement_m == 0.04
    assert summaries[1].stiffness_n_per_m == 250.0
