"""Tests for MMD math helpers."""

from __future__ import annotations

import numpy as np

from apple_pick_sim.system_id.mmd import (
    apply_normalization,
    biased_mmd2,
    fit_gt_normalization,
    rbf_bandwidth_median,
)


def test_gt_normalization_is_per_feature_and_uses_gt_only_statistics():
    gt = np.array(
        [
            [1.0, 10.0],
            [3.0, 10.0],
            [5.0, 10.0],
        ],
        dtype=np.float32,
    )
    candidate = np.array(
        [
            [100.0, -1000.0],
            [200.0, -2000.0],
        ],
        dtype=np.float32,
    )

    stats = fit_gt_normalization(gt)
    normalized_gt = apply_normalization(gt, stats)
    normalized_candidate = apply_normalization(candidate, stats)

    np.testing.assert_allclose(stats.mean, [3.0, 10.0])
    np.testing.assert_allclose(stats.std, [np.std([1.0, 3.0, 5.0]), 1.0e-6])
    np.testing.assert_allclose(normalized_gt[:, 1], 0.0)
    assert np.all(np.isfinite(normalized_candidate))
    assert normalized_candidate[0, 0] > 50.0


def test_rbf_bandwidth_uses_median_pairwise_distance():
    gt_norm = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [6.0, 8.0],
        ],
        dtype=np.float32,
    )

    assert rbf_bandwidth_median(gt_norm) == 5.0


def test_rbf_bandwidth_falls_back_for_degenerate_inputs():
    assert rbf_bandwidth_median(np.zeros((3, 2), dtype=np.float32)) == 1.0
    assert rbf_bandwidth_median(np.zeros((1, 2), dtype=np.float32)) == 1.0


def test_biased_mmd2_is_zero_for_identical_samples_and_larger_for_shifted_samples():
    x = np.array(
        [
            [-1.0, 0.0],
            [0.0, 0.5],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    shifted = x + np.array([3.0, 0.0], dtype=np.float32)

    same = biased_mmd2(x, x, bandwidth=1.0)
    different = biased_mmd2(x, shifted, bandwidth=1.0)

    assert abs(same) < 1.0e-8
    assert different > same
    assert different > 0.1
