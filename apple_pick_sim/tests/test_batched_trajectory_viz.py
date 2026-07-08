"""Tests for batched sys-ID trajectory visualization."""

from __future__ import annotations

import numpy as np

from apple_pick_sim.system_id.batched_trajectory_viz import (
    make_episode_spatial_figure,
    make_episode_time_series_figure,
)
from apple_pick_sim.system_id.trajectory_store import PHASE_TO_INT


def _synthetic_arrays(*, n: int = 8) -> tuple[dict, dict]:
    time = np.linspace(0.0, 1.0, n, dtype=np.float64)
    phase = np.full(n, int(PHASE_TO_INT["move_out"]), dtype=np.int8)
    phase[-2:] = int(PHASE_TO_INT["hold"])
    direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    tcp = np.stack([time * 0.1, np.zeros(n), np.zeros(n)], axis=1)
    apple = tcp + np.array([0.0, 0.05, 0.02], dtype=np.float64)
    arrays = {
        "sim_time": time.astype(np.float32),
        "phase": phase,
        "tcp_pos": tcp.astype(np.float32),
        "apple_pos": apple.astype(np.float32),
        "excitation_direction": np.tile(direction, (n, 1)).astype(np.float32),
        "junction_names": ["stem_apple"],
        "woody_part_start_pos": {
            "stem_apple": (apple - np.array([0.0, 0.0, 0.03])).astype(np.float32),
        },
    }
    metadata = {
        "structure_idx": 0,
        "direction_idx": 0,
        "pull_direction": direction.tolist(),
        "total_movement_m": 0.1,
    }
    return arrays, metadata


def test_make_episode_time_series_figure_builds():
    arrays, metadata = _synthetic_arrays()
    fig = make_episode_time_series_figure(arrays, metadata)
    assert len(fig.data) >= 2


def test_make_episode_spatial_figure_builds_movement_arrows():
    arrays, metadata = _synthetic_arrays()
    fig = make_episode_spatial_figure(arrays, metadata)
    names = {trace.name for trace in fig.data}
    assert "TCP" in names
    assert "pull direction" in names
    assert "TCP move step" in names


def test_make_hold_quasi_static_figure_builds():
    from apple_pick_sim.system_id.batched_hold_quasi_static import analyze_episode_hold_quasi_static
    from apple_pick_sim.system_id.batched_trajectory_viz import make_hold_quasi_static_figure
    from apple_pick_sim.tests.test_batched_hold_quasi_static import _synthetic_hold_arrays

    arrays, metadata = _synthetic_hold_arrays()
    reports = analyze_episode_hold_quasi_static(arrays, metadata, use_latter_half=False)
    fig = make_hold_quasi_static_figure(arrays, metadata, reports)
    assert len(fig.data) >= 2
