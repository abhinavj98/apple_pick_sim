from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np


def bend_stiffness_values_match(
    a: tuple[float, float, float, float] | Iterable[float],
    b: tuple[float, float, float, float] | Iterable[float],
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
) -> bool:
    """True if two (primary, secondary, spur, stem) tuples are equal within tolerance."""
    return all(
        math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol)
        for x, y in zip(a, b, strict=True)
    )


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(diff * diff))


def _as_woody_2d(source: Mapping[str, Any], key: str, name: str, n: int) -> np.ndarray:
    woody = source.get(key)
    if woody is None or name not in woody:
        raise KeyError(f"missing woody endpoint {key!r} for junction {name!r}")
    return np.asarray(woody[name], dtype=np.float64).reshape(-1, 3)[:n]


def _aggregate_rows(values: np.ndarray, *, aggregation: Literal["mean", "median"]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] == 0:
        raise ValueError("cannot aggregate an empty hold window")
    if aggregation == "mean":
        return np.mean(arr, axis=0)
    if aggregation == "median":
        return np.median(arr, axis=0)
    raise ValueError(f"unsupported hold aggregation: {aggregation!r}")


def _woody_nan_dict(junction_names: Sequence[str]) -> dict[str, float]:
    return {str(name): float("nan") for name in junction_names}


def _has_woody_data(source: Mapping[str, Any], junction_names: Sequence[str]) -> bool:
    if not junction_names:
        return False
    for key in ("woody_part_start_pos", "woody_part_end_pos"):
        woody = source.get(key)
        if not isinstance(woody, dict):
            return False
        for name in junction_names:
            if name not in woody:
                return False
    return True


def woody_segment_pos_mse_masked(
    *,
    replay: Mapping[str, Any],
    recorded: Mapping[str, Any],
    junction_names: Sequence[str],
    n: int,
    mask: np.ndarray,
) -> dict[str, float]:
    """Per-segment MSE combining start+end endpoints over frame-masked rows."""
    names = [str(name) for name in junction_names]
    if not names or not _has_woody_data(replay, names) or not _has_woody_data(recorded, names):
        return {}
    mask = np.asarray(mask, dtype=bool).reshape(-1)[: int(n)]
    if int(np.count_nonzero(mask)) == 0:
        return _woody_nan_dict(names)

    out: dict[str, float] = {}
    for name in names:
        start_rep = _as_woody_2d(replay, "woody_part_start_pos", name, n)[mask]
        end_rep = _as_woody_2d(replay, "woody_part_end_pos", name, n)[mask]
        start_rec = _as_woody_2d(recorded, "woody_part_start_pos", name, n)[mask]
        end_rec = _as_woody_2d(recorded, "woody_part_end_pos", name, n)[mask]
        rep = np.concatenate([start_rep, end_rep], axis=1)
        rec = np.concatenate([start_rec, end_rec], axis=1)
        out[name] = _mse(rep, rec)
    return out


def woody_segment_pos_mse_hold_aggregated(
    *,
    replay: Mapping[str, Any],
    recorded: Mapping[str, Any],
    junction_names: Sequence[str],
    n: int,
    hold_idx: np.ndarray,
    aggregation: Literal["mean", "median"],
) -> dict[str, float]:
    """Per-segment MSE combining one hold-aggregate (mean/median) per endpoint."""
    names = [str(name) for name in junction_names]
    if not names or not _has_woody_data(replay, names) or not _has_woody_data(recorded, names):
        return {}
    hold_idx = np.asarray(hold_idx, dtype=np.int64).reshape(-1)
    if hold_idx.size == 0:
        return _woody_nan_dict(names)

    out: dict[str, float] = {}
    for name in names:
        start_rep = _as_woody_2d(replay, "woody_part_start_pos", name, n)[hold_idx]
        end_rep = _as_woody_2d(replay, "woody_part_end_pos", name, n)[hold_idx]
        start_rec = _as_woody_2d(recorded, "woody_part_start_pos", name, n)[hold_idx]
        end_rec = _as_woody_2d(recorded, "woody_part_end_pos", name, n)[hold_idx]
        rep = np.concatenate(
            [
                _aggregate_rows(start_rep, aggregation=aggregation),
                _aggregate_rows(end_rep, aggregation=aggregation),
            ]
        )
        rec = np.concatenate(
            [
                _aggregate_rows(start_rec, aggregation=aggregation),
                _aggregate_rows(end_rec, aggregation=aggregation),
            ]
        )
        out[name] = _mse(rep.reshape(1, -1), rec.reshape(1, -1))
    return out


def log_l2_distance_to_gt(
    candidate: dict[str, float],
    gt: dict[str, float],
    *,
    keys: Sequence[str],
    eps: float = 1e-12,
) -> float:
    """Log-space L2 distance between candidate and GT over selected stiffness keys."""
    if not keys:
        return 0.0
    c = np.array([float(candidate[k]) for k in keys], dtype=np.float64)
    g = np.array([float(gt[k]) for k in keys], dtype=np.float64)
    d = np.log(c + float(eps)) - np.log(g + float(eps))
    return float(np.linalg.norm(d))


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return average ranks for values (handles ties). Ranks are 1..n."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(v.size)
    if n == 0:
        return v
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman_r(x: Iterable[float], y: Iterable[float]) -> float:
    """Spearman rank correlation (no SciPy dependency)."""
    x_arr = np.asarray(list(x), dtype=np.float64).reshape(-1)
    y_arr = np.asarray(list(y), dtype=np.float64).reshape(-1)
    if x_arr.size != y_arr.size:
        raise ValueError("spearman_r requires x and y with the same length")
    n = int(x_arr.size)
    if n < 2:
        return 0.0

    rx = average_ranks(x_arr)
    ry = average_ranks(y_arr)
    rx = rx - float(np.mean(rx))
    ry = ry - float(np.mean(ry))
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(rx, ry) / denom)

