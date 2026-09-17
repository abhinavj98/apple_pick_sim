"""Tests for holdout time-series match metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from apple_pick_sim.system_id.match_metrics import (
    scalar_match_stats,
    tree_label_from_dataset_path,
    vector3_match_stats,
)


def test_scalar_identical_series():
    x = np.linspace(0.0, 1.0, 40)
    stats = scalar_match_stats(x, x)
    assert stats["mse"] == pytest.approx(0.0)
    assert stats["bias"] == pytest.approx(0.0)
    assert stats["error_variance"] == pytest.approx(0.0)
    assert stats["r_squared"] == pytest.approx(1.0)
    assert stats["cross_corr_lag"] == 0
    assert stats["phase_corrected_rmse"] == pytest.approx(0.0)
    assert stats["peak_magnitude_error"] == pytest.approx(0.0)


def test_scalar_constant_offset():
    real = np.ones(20)
    sim = np.full(20, 3.0)
    stats = scalar_match_stats(real, sim)
    assert stats["bias"] == pytest.approx(2.0)
    assert stats["mse"] == pytest.approx(4.0)
    assert stats["error_variance"] == pytest.approx(0.0)


def test_scalar_sim_delayed_by_five_frames():
    t = np.arange(120, dtype=np.float64)
    real = np.sin(0.3 * t)
    sim = np.zeros_like(real)
    sim[5:] = real[:-5]
    stats = scalar_match_stats(real, sim, max_lag=30)
    assert stats["cross_corr_lag"] == 5
    assert stats["phase_corrected_rmse"] == pytest.approx(0.0, abs=1e-12)


def test_scalar_zero_real_variance_r_squared_is_nan():
    real = np.full(10, 2.0)
    sim = np.full(10, 3.0)
    stats = scalar_match_stats(real, sim)
    assert math.isnan(stats["r_squared"])


def test_scalar_empty_series():
    stats = scalar_match_stats(np.array([]), np.array([]))
    for key in (
        "mse",
        "bias",
        "error_variance",
        "r_squared",
        "cross_corr_lag",
        "phase_corrected_rmse",
        "peak_magnitude_error",
    ):
        assert math.isnan(stats[key])


def test_vector3_bias_length_and_mse():
    real = np.tile(np.array([1.0, 0.0, 0.0]), (10, 1))
    sim = real + np.array([1.0, 2.0, 3.0])
    stats = vector3_match_stats(real, sim)
    assert len(stats["bias"]) == 3
    assert stats["bias"] == pytest.approx([1.0, 2.0, 3.0])
    assert stats["mse"] == pytest.approx(14.0)


def test_tree_label_from_dataset_path():
    assert tree_label_from_dataset_path("tmp/real_batched_s04") == "s04"
    assert tree_label_from_dataset_path("/data/custom_name") == "custom_name"
