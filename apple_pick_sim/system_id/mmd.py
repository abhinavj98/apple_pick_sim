"""Maximum Mean Discrepancy helpers for offline sys-ID diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizationStats:
    """GT mean plus fixed physical-scale divisors (``std`` field stores scale)."""

    mean: np.ndarray
    std: np.ndarray


def _as_feature_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    return arr


def fit_gt_normalization(
    gt: np.ndarray,
    eps: float = 1.0e-6,
    *,
    n_junctions: int = 2,
) -> NormalizationStats:
    """Fit GT mean; use fixed physical scales as divisors.

    ``eps`` is retained for call-site compatibility but ignored: scale no longer
    comes from GT std. Trailing columns beyond ``2 * state_dim(n_junctions)``
    (hold/dir one-hots) use mean=0 and scale=1. Default ``n_junctions=2`` is the
    CMA woody pair; pass the bag's junction count for 1-junction MMD examples.
    """
    del eps  # unused; kept for signature compatibility
    from apple_pick_sim.system_id.mmd_features import (
        state_vector_phys_scale,
        transition_feature_scale,
    )

    gt_arr = _as_feature_matrix(gt, name="gt")
    mean = np.mean(gt_arr, axis=0)
    scale = transition_feature_scale(gt_arr.shape[1], n_junctions=n_junctions)
    state_dim = int(state_vector_phys_scale(n_junctions).size)
    mean = mean.copy()
    mean[2 * state_dim :] = 0.0
    return NormalizationStats(mean=mean, std=scale)


def apply_normalization(values: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Apply previously fit GT normalization to a feature matrix."""

    arr = _as_feature_matrix(values, name="values")
    if arr.shape[1] != stats.mean.shape[0] or arr.shape[1] != stats.std.shape[0]:
        raise ValueError(
            "normalization feature dimension mismatch: "
            f"values={arr.shape[1]} mean={stats.mean.shape[0]} std={stats.std.shape[0]}"
        )
    return (arr - stats.mean.reshape(1, -1)) / stats.std.reshape(1, -1)


def _pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - y[None, :, :]
    return np.sum(diff * diff, axis=2)


def rbf_bandwidth_median(gt_norm: np.ndarray, eps: float = 1.0e-6) -> float:
    """Return median pairwise GT distance, falling back to 1.0 when degenerate."""

    gt_arr = _as_feature_matrix(gt_norm, name="gt_norm")
    if gt_arr.shape[0] < 2:
        return 1.0
    dists = np.sqrt(_pairwise_sq_dists(gt_arr, gt_arr))
    upper = dists[np.triu_indices(gt_arr.shape[0], k=1)]
    finite = upper[np.isfinite(upper)]
    finite = finite[finite > float(eps)]
    if finite.size == 0:
        return 1.0
    bandwidth = float(np.median(finite))
    if not np.isfinite(bandwidth) or bandwidth <= float(eps):
        return 1.0
    return bandwidth


def _rbf_kernel_mean(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    sq_dists = _pairwise_sq_dists(x, y)
    kernel = np.exp(-sq_dists / (2.0 * float(bandwidth) * float(bandwidth)))
    return float(np.mean(kernel))


def biased_mmd2(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    """Compute biased RBF MMD^2, including same-sample self-pairs."""

    x_arr = _as_feature_matrix(x, name="x")
    y_arr = _as_feature_matrix(y, name="y")
    if x_arr.shape[1] != y_arr.shape[1]:
        raise ValueError(
            f"MMD feature dimension mismatch: x={x_arr.shape[1]} y={y_arr.shape[1]}"
        )
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError(f"bandwidth must be finite and positive, got {bandwidth}")
    value = (
        _rbf_kernel_mean(x_arr, x_arr, bandwidth)
        + _rbf_kernel_mean(y_arr, y_arr, bandwidth)
        - 2.0 * _rbf_kernel_mean(x_arr, y_arr, bandwidth)
    )
    return max(0.0, float(value))
