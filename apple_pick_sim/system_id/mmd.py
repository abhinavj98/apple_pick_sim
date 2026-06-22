"""Maximum Mean Discrepancy helpers for offline sys-ID diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizationStats:
    """Per-feature normalization statistics fit from GT transitions."""

    mean: np.ndarray
    std: np.ndarray


def _as_feature_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    return arr


def fit_gt_normalization(gt: np.ndarray, eps: float = 1.0e-6) -> NormalizationStats:
    """Fit per-feature mean/std from GT transitions only."""

    gt_arr = _as_feature_matrix(gt, name="gt")
    mean = np.mean(gt_arr, axis=0)
    std = np.std(gt_arr, axis=0)
    std = np.where(std < float(eps), float(eps), std)
    return NormalizationStats(mean=mean, std=std)


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
