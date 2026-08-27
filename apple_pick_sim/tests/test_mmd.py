"""Tests for MMD math helpers."""

from __future__ import annotations

import numpy as np
import pytest

from apple_pick_sim.system_id.mmd import (
    apply_normalization,
    biased_mmd2,
    fit_gt_normalization,
    rbf_bandwidth_median,
)


def test_fit_gt_normalization_uses_fixed_physical_scale_not_gt_std():
    from apple_pick_sim.system_id.mmd_features import (
        STATE_VECTOR_PHYS_SCALE,
        transition_feature_scale,
    )

    state_dim = len(STATE_VECTOR_PHYS_SCALE)
    # Minimal transition row width: [s, Δs] only (no one-hot)
    n = 2 * state_dim
    gt = np.zeros((3, n), dtype=np.float64)
    # Column 0 is Fx: give GT real variance that must NOT become the divisor
    gt[:, 0] = [0.0, 3.0, 6.0]
    # Matching Δ column (index state_dim) left at 0

    stats = fit_gt_normalization(gt)
    scale = transition_feature_scale(n)
    np.testing.assert_allclose(stats.std, scale)
    np.testing.assert_allclose(stats.mean[0], 3.0)
    # Candidate residual 3 N on Fx → 3/0.5 = 6 after apply, not 3/std(GT)=3/sqrt(6)
    cand = np.zeros((1, n), dtype=np.float64)
    cand[0, 0] = 6.0  # 3 N above GT mean
    out = apply_normalization(cand, stats)
    assert out[0, 0] == pytest.approx(3.0 / 0.5)
    assert STATE_VECTOR_PHYS_SCALE[0] == pytest.approx(0.5)
    np.testing.assert_allclose(STATE_VECTOR_PHYS_SCALE[3:6], 1.0)
    np.testing.assert_allclose(STATE_VECTOR_PHYS_SCALE[12:15], 0.005)
    np.testing.assert_allclose(STATE_VECTOR_PHYS_SCALE[15:21], 0.005)
    np.testing.assert_allclose(STATE_VECTOR_PHYS_SCALE[21:23], 0.05)


def test_near_zero_gt_velocity_does_not_explode_candidate_residual():
    from apple_pick_sim.system_id.mmd_features import STATE_VECTOR_PHYS_SCALE

    state_dim = len(STATE_VECTOR_PHYS_SCALE)
    n = 2 * state_dim
    vx = 6  # index of tcp_velocity vx in STATE_VECTOR after dropping action
    assert STATE_VECTOR_PHYS_SCALE[vx] == pytest.approx(0.02)

    gt = np.zeros((4, n), dtype=np.float64)
    gt[:, vx] = [0.0, 1e-4, -1e-4, 0.0]  # tiny hold variance
    stats = fit_gt_normalization(gt)
    cand = np.zeros((1, n), dtype=np.float64)
    cand[0, vx] = 0.01  # 1 cm/s residual
    out = apply_normalization(cand, stats)
    # Fixed scale 0.02 → 0.5; old GT-std path would be O(100)
    assert out[0, vx] == pytest.approx(0.01 / 0.02)
    assert abs(out[0, vx]) < 2.0


def test_trailing_onehot_is_not_mean_centered():
    from apple_pick_sim.system_id.mmd_features import STATE_VECTOR_PHYS_SCALE

    state_dim = len(STATE_VECTOR_PHYS_SCALE)
    n_holds = 4
    n = 2 * state_dim + n_holds
    gt = np.zeros((2, n), dtype=np.float64)
    gt[0, -4:] = [1, 0, 0, 0]
    gt[1, -4:] = [0, 1, 0, 0]
    stats = fit_gt_normalization(gt)
    np.testing.assert_allclose(stats.mean[-4:], 0.0)
    np.testing.assert_allclose(stats.std[-4:], 1.0)
    out = apply_normalization(gt, stats)
    np.testing.assert_allclose(out[0, -4:], [1, 0, 0, 0])


def _state_dim_for_junctions(n_junctions: int) -> int:
    # ft(6)+vel(6)+tcp(3)+woody(3J)+bend(J)
    return 15 + 4 * int(n_junctions)


def test_fit_gt_normalization_scales_one_junction_woody_and_bend():
    n_junctions = 1
    state_dim = _state_dim_for_junctions(n_junctions)
    n = 2 * state_dim
    gt = np.zeros((2, n), dtype=np.float64)
    gt[:, 0] = [0.0, 6.0]  # Fx mean 3 N

    stats = fit_gt_normalization(gt, n_junctions=n_junctions)

    assert stats.std.shape == (n,)
    np.testing.assert_allclose(stats.std[0], 0.5)
    np.testing.assert_allclose(stats.std[12:15], 0.005)  # tcp_pos
    np.testing.assert_allclose(stats.std[15:18], 0.005)  # woody_start
    assert stats.std[18] == pytest.approx(0.05)
    np.testing.assert_allclose(stats.std[state_dim : state_dim + 6], stats.std[:6])
    cand = np.zeros((1, n), dtype=np.float64)
    cand[0, 0] = 6.0
    out = apply_normalization(cand, stats)
    assert out[0, 0] == pytest.approx(3.0 / 0.5)


def test_fit_gt_normalization_does_not_treat_extra_junctions_as_onehots():
    n_junctions = 3
    state_dim = _state_dim_for_junctions(n_junctions)
    n = 2 * state_dim
    gt = np.full((2, n), 0.4, dtype=np.float64)

    stats = fit_gt_normalization(gt, n_junctions=n_junctions)

    woody0 = 15
    last_bend = state_dim - 1
    np.testing.assert_allclose(stats.std[woody0 : woody0 + 9], 0.005)
    np.testing.assert_allclose(stats.std[woody0 + 9 : state_dim], 0.05)
    np.testing.assert_allclose(stats.mean[last_bend], 0.4)
    np.testing.assert_allclose(stats.std[last_bend], 0.05)
    np.testing.assert_allclose(stats.mean[state_dim + last_bend], 0.4)
    np.testing.assert_allclose(stats.std[state_dim + last_bend], 0.05)


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
