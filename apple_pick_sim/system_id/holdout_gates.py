"""Pure helpers for direction holdout splits and verification gates."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence

import numpy as np

from apple_pick_sim.system_id.mmd_features import iter_kept_hold_segments

DIRECTION_SPLIT_SEED = 17
MAGNITUDE_RATIO_MIN = 1.0 / 3.0
MAGNITUDE_RATIO_MAX = 3.0
TREND_PEARSON_MIN = 0.5
FORCE_FLOOR_N = 0.2
TORQUE_FLOOR_NM = 0.05
FLOOR_SLACK_FACTOR = 3.0
FORCE_SLACK_N = 0.4  # also the torque additive slack, in N·m


def choose_direction_split(
    directions: Iterable[int], *, seed: int, n_train: int = 5
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return (train, val) sorted disjoint direction indices."""
    dirs = list(directions)
    if len(dirs) != len(set(dirs)):
        raise ValueError("duplicate direction indices are not allowed")
    if n_train >= len(dirs):
        raise ValueError(f"n_train ({n_train}) must be less than population size ({len(dirs)})")
    sorted_dirs = sorted(dirs)
    train = tuple(sorted(random.Random(seed).sample(sorted_dirs, n_train)))
    val = tuple(d for d in sorted_dirs if d not in train)
    return train, val


def magnitude_ratio_ok(
    *, real_mean: float, fitted_mean: float, floor: float, slack: float
) -> tuple[bool, float]:
    """Return (passed, ratio). Uses the additive floor rule when real_mean < floor."""
    if real_mean == 0.0:
        ratio = math.inf if fitted_mean > 0.0 else 1.0
    else:
        ratio = fitted_mean / real_mean

    if real_mean < floor:
        passed = fitted_mean < FLOOR_SLACK_FACTOR * real_mean + slack
    else:
        passed = MAGNITUDE_RATIO_MIN <= ratio <= MAGNITUDE_RATIO_MAX
    return passed, ratio


def trend_pearson_ok(
    real: Sequence[float], fitted: Sequence[float], *, magnitude_passed: bool
) -> tuple[bool, float | None]:
    """Pearson r >= TREND_PEARSON_MIN; zero-variance passes iff magnitude passed."""
    if len(real) < 3 or len(fitted) < 3:
        return False, None
    if len(real) != len(fitted):
        return False, None

    real_arr = np.asarray(real, dtype=float)
    fitted_arr = np.asarray(fitted, dtype=float)
    if real_arr.std() == 0.0 or fitted_arr.std() == 0.0:
        return magnitude_passed, None

    r = float(np.corrcoef(real_arr, fitted_arr)[0, 1])
    if not math.isfinite(r):
        return magnitude_passed, None
    return r >= TREND_PEARSON_MIN, r


def signed_parallel_series(
    values: np.ndarray, pull_direction: Sequence[float]
) -> np.ndarray:
    """Project (T, 3) rows onto the unit pull axis -> (T,) signed scalars."""
    axis = np.asarray(pull_direction, dtype=float).reshape(-1)
    if axis.shape != (3,):
        raise ValueError(f"pull_direction must have length 3, got shape {axis.shape}")
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        raise ValueError("pull_direction must be non-zero")
    unit = axis / norm
    vals = np.asarray(values, dtype=float)
    if vals.ndim != 2 or vals.shape[1] != 3:
        raise ValueError(f"values must have shape (T, 3), got {vals.shape}")
    return vals @ unit


def per_hold_means(
    series: np.ndarray, *, phase: np.ndarray, dir_idx: np.ndarray, direction: int
) -> np.ndarray:
    """Mean of `series` over each contiguous hold segment of one direction."""
    series_arr = np.asarray(series, dtype=float).reshape(-1)
    segments = iter_kept_hold_segments(
        phase=phase, dir_idx=dir_idx, direction=direction, min_frames=1
    )
    means: list[float] = []
    for segment in segments:
        if segment.size == 0:
            continue
        means.append(float(series_arr[segment].mean()))
    return np.asarray(means, dtype=float)


def tcp_displacement_along_pull(
    tcp_pos: np.ndarray,
    *,
    phase: np.ndarray,
    dir_idx: np.ndarray,
    direction: int,
    pull_direction: Sequence[float],
) -> np.ndarray:
    """Hold-frame signed TCP displacement: s = (x - x_hold0) · p_hat.

    ``x_hold0`` is TCP at the **first hold frame** of this direction, not
    episode frame 0 (that frame is still on the pull-in).
    """
    segments = iter_kept_hold_segments(
        phase=phase, dir_idx=dir_idx, direction=direction, min_frames=1
    )
    hold_indices = [idx for segment in segments for idx in segment.tolist()]
    if not hold_indices:
        raise ValueError(f"direction {direction} has no hold frames")

    tcp = np.asarray(tcp_pos, dtype=float)
    if tcp.ndim != 2 or tcp.shape[1] != 3:
        raise ValueError(f"tcp_pos must have shape (T, 3), got {tcp.shape}")

    hold_tcp = tcp[hold_indices]
    along_pull = signed_parallel_series(hold_tcp, pull_direction)
    return np.round(along_pull - along_pull[0], decimals=12)
