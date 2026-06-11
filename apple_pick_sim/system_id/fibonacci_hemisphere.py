"""Fibonacci-lattice sampling on the forward hemisphere."""

from __future__ import annotations

import numpy as np


def _fibonacci_sphere(n_samples: int) -> np.ndarray:
    """Return ``(n_samples, 3)`` unit vectors on the full sphere."""
    if n_samples <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    if n_samples == 1:
        return np.array([[0.0, 0.0, 1.0]], dtype=np.float64)

    indices = np.arange(n_samples, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n_samples)
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    theta = 2.0 * np.pi * indices / golden_ratio
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    pts = np.stack([x, y, z], axis=1)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def sample_fibonacci_hemisphere(n: int, stem_dir: np.ndarray) -> np.ndarray:
    """Sample ``n`` approximately uniform unit directions in the forward hemisphere.

    Uses a golden-ratio Fibonacci lattice on the unit sphere, keeps points with
    ``dot(direction, stem_dir) >= 0``, and reflects duplicates if fewer than ``n``
    forward-facing samples are available.
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    stem = np.asarray(stem_dir, dtype=np.float64).reshape(3)
    stem_norm = float(np.linalg.norm(stem))
    if stem_norm < 1e-12:
        raise ValueError("stem_dir must be non-zero")
    stem = stem / stem_norm

    pool_size = max(n * 4, 64)
    lattice = _fibonacci_sphere(pool_size)
    forward_mask = lattice @ stem >= 0.0
    forward = lattice[forward_mask]

    if len(forward) == 0:
        forward = np.array([stem], dtype=np.float64)

    selected: list[np.ndarray] = []
    for i in range(n):
        selected.append(forward[i % len(forward)].copy())

    out = np.stack(selected, axis=0)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / norms).astype(np.float64)
