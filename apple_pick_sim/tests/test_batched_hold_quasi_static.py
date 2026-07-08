"""Tests for batched hold quasi-static diagnostics."""

from __future__ import annotations

import numpy as np

from apple_pick_sim.system_id.batched_hold_quasi_static import (
    StiffnessIdHoldThresholds,
    analyze_episode_hold_quasi_static,
)
from apple_pick_sim.system_id.trajectory_store import PHASE_TO_INT


def _synthetic_hold_arrays(*, n_hold: int = 40) -> tuple[dict, dict]:
    time = np.linspace(0.0, float(n_hold - 1) / 30.0, n_hold, dtype=np.float64)
    phase = np.full(n_hold, int(PHASE_TO_INT["hold"]), dtype=np.int8)
    direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    tcp = np.stack([np.full(n_hold, 0.01), np.zeros(n_hold), np.zeros(n_hold)], axis=1)
    apple = tcp + np.array([0.0, 0.02, 0.01], dtype=np.float64)
    arrays = {
        "sim_time": time.astype(np.float32),
        "phase": phase,
        "amplitude_m": np.full(n_hold, 0.02, dtype=np.float32),
        "action": np.zeros((n_hold, 6), dtype=np.float32),
        "tcp_velocity": np.zeros((n_hold, 6), dtype=np.float32),
        "tcp_pos": tcp.astype(np.float32),
        "apple_pos": apple.astype(np.float32),
        "ft_wrist": np.zeros((n_hold, 6), dtype=np.float32),
        "excitation_direction": np.tile(direction, (n_hold, 1)).astype(np.float32),
    }
    metadata = {
        "structure_idx": 0,
        "direction_idx": 0,
        "control_hz": 30.0,
        "hold_duration_s": 1.0,
    }
    return arrays, metadata


def test_stable_hold_segment_passes_default_thresholds():
    arrays, metadata = _synthetic_hold_arrays()
    reports = analyze_episode_hold_quasi_static(arrays, metadata, use_latter_half=False)
    assert len(reports) == 1
    assert reports[0].commanded_zero_action
    assert reports[0].is_quasi_static


def test_high_force_oscillation_fails_cv_threshold():
    """Force oscillating ±50 % of mean → force_cv >> threshold → fail."""
    arrays, metadata = _synthetic_hold_arrays(n_hold=40)
    ft = arrays["ft_wrist"].copy()
    # Alternate between 2 N and 10 N → CV = std/mean ≈ 4/6 ≈ 0.67 >> 0.10
    ft[:, 0] = np.where(np.arange(40) % 2 == 0, 10.0, 2.0)
    arrays = dict(arrays)
    arrays["ft_wrist"] = ft
    reports = analyze_episode_hold_quasi_static(
        arrays,
        metadata,
        thresholds=StiffnessIdHoldThresholds(max_force_cv=0.10),
        use_latter_half=False,
    )
    assert len(reports) == 1
    assert not reports[0].is_quasi_static
    assert "force_oscillation" in reports[0].issues


def test_tcp_excursion_fails_threshold():
    """TCP drifting 200 mm from hold start → fails tcp_excursion gate."""
    arrays, metadata = _synthetic_hold_arrays(n_hold=40)
    tcp = arrays["tcp_pos"].copy()
    # linear drift of 0.3 m over hold
    tcp[:, 0] += np.linspace(0, 0.3, 40, dtype=np.float32)
    arrays = dict(arrays)
    arrays["tcp_pos"] = tcp
    reports = analyze_episode_hold_quasi_static(
        arrays,
        metadata,
        thresholds=StiffnessIdHoldThresholds(max_tcp_excursion_m=0.05),
        use_latter_half=False,
    )
    assert len(reports) == 1
    assert not reports[0].is_quasi_static
    assert "tcp_excursion" in reports[0].issues
