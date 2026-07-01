"""Tests for zero-VIC pose logging helpers."""

from __future__ import annotations

import numpy as np
import pytest

from apple_pick_sim.diagnostics.log_zero_vic_poses import apple_vel_from_position_delta


def test_apple_vel_from_position_delta_first_sample_zero():
    pos = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    vel = apple_vel_from_position_delta(pos, apple_pos_prev=None, t_s=0.0, t_s_prev=None)
    assert vel == pytest.approx(np.zeros(3))


def test_apple_vel_from_position_delta_finite_diff():
    pos_prev = np.array([0.0, 0.0, 0.5], dtype=np.float64)
    pos = np.array([0.05, 0.0, 0.5], dtype=np.float64)
    vel = apple_vel_from_position_delta(
        pos, apple_pos_prev=pos_prev, t_s=1.0, t_s_prev=0.0
    )
    assert vel == pytest.approx(np.array([0.05, 0.0, 0.0]))


def test_apple_vel_from_position_delta_non_positive_dt_zero():
    pos = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    vel = apple_vel_from_position_delta(
        pos, apple_pos_prev=np.zeros(3), t_s=1.0, t_s_prev=1.0
    )
    assert vel == pytest.approx(np.zeros(3))
