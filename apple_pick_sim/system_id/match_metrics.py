"""Time-series match metrics for holdout reporting (report-only; not CMA fitness)."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from apple_pick_sim.system_id.mmd_features import (
    CMA_WOODY_JUNCTIONS,
    flatten_woody_positions,
    iter_kept_hold_segments,
    scored_ft_wrist,
)

DEFAULT_MAX_LAG_FRAMES = 30
_STAT_KEYS = (
    "mse",
    "bias",
    "error_variance",
    "r_squared",
    "cross_corr_lag",
    "phase_corrected_rmse",
    "peak_magnitude_error",
)


def tree_label_from_dataset_path(dataset_path: str | Path) -> str:
    """Extract ``sNN`` from ``real_batched_s04``; else return the directory stem."""
    name = Path(dataset_path).name
    match = re.search(r"(s\d+)", name, flags=re.IGNORECASE)
    return match.group(1) if match else name


def _nan_stats(*, vector_bias: bool = False) -> dict[str, Any]:
    if vector_bias:
        return {
            "mse": float("nan"),
            "bias": [float("nan"), float("nan"), float("nan")],
            "error_variance": float("nan"),
            "r_squared": float("nan"),
            "cross_corr_lag": float("nan"),
            "phase_corrected_rmse": float("nan"),
            "peak_magnitude_error": float("nan"),
        }
    return {key: float("nan") for key in _STAT_KEYS}


def _aligned_pair(
    real: np.ndarray, sim: np.ndarray, lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Align series for lag search; positive lag means sim is delayed."""
    real = np.asarray(real, dtype=np.float64).reshape(-1)
    sim = np.asarray(sim, dtype=np.float64).reshape(-1)
    n = min(int(real.size), int(sim.size))
    if n <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    lag = int(lag)
    if lag >= n:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    if lag >= 0:
        return real[: n - lag], sim[lag:n]
    lead = -lag
    if lead >= n:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    return real[lead:n], sim[: n - lead]


def _normalized_cross_corr(real: np.ndarray, sim: np.ndarray) -> float:
    if real.size < 2 or sim.size < 2 or real.size != sim.size:
        return float("nan")
    r0 = real - float(np.mean(real))
    s0 = sim - float(np.mean(sim))
    denom = float(np.linalg.norm(r0) * np.linalg.norm(s0))
    if denom == 0.0:
        return 1.0 if np.allclose(r0, s0) else 0.0
    return float(np.dot(r0, s0) / denom)


def best_cross_corr_lag(
    real: np.ndarray, sim: np.ndarray, *, max_lag: int = DEFAULT_MAX_LAG_FRAMES
) -> int:
    """Return integer lag maximizing normalized cross-correlation."""
    if real.size == 0 or sim.size == 0:
        return 0
    best_lag = 0
    best_corr = float("-inf")
    for lag in range(-int(max_lag), int(max_lag) + 1):
        r, s = _aligned_pair(real, sim, lag)
        corr = _normalized_cross_corr(r, s)
        if not math.isfinite(corr):
            continue
        if corr > best_corr + 1e-12:
            best_corr = corr
            best_lag = lag
        elif abs(corr - best_corr) <= 1e-12 and (
            abs(lag) < abs(best_lag) or (abs(lag) == abs(best_lag) and lag == 0)
        ):
            best_lag = lag
    return int(best_lag)


def scalar_match_stats(
    real: np.ndarray,
    sim: np.ndarray,
    *,
    max_lag: int = DEFAULT_MAX_LAG_FRAMES,
) -> dict[str, float]:
    """Seven scalar stats for one aligned 1D series pair."""
    real = np.asarray(real, dtype=np.float64).reshape(-1)
    sim = np.asarray(sim, dtype=np.float64).reshape(-1)
    n = min(int(real.size), int(sim.size))
    if n <= 0:
        return _nan_stats()
    real = real[:n]
    sim = sim[:n]
    err = sim - real
    bias = float(np.mean(err))
    mse = float(np.mean(err**2))
    error_variance = float(np.mean((err - bias) ** 2))
    ss_res = float(np.sum(err**2))
    centered = real - float(np.mean(real))
    ss_tot = float(np.sum(centered**2))
    r_squared = float("nan") if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    lag = best_cross_corr_lag(real, sim, max_lag=max_lag)
    aligned_r, aligned_s = _aligned_pair(real, sim, lag)
    if aligned_r.size == 0:
        phase_rmse = float("nan")
    else:
        phase_rmse = float(np.sqrt(np.mean((aligned_s - aligned_r) ** 2)))
    peak_error = float(abs(float(np.max(sim)) - float(np.max(real))))
    return {
        "mse": mse,
        "bias": bias,
        "error_variance": error_variance,
        "r_squared": r_squared,
        "cross_corr_lag": float(lag),
        "phase_corrected_rmse": phase_rmse,
        "peak_magnitude_error": peak_error,
    }


def vector3_match_stats(
    real: np.ndarray,
    sim: np.ndarray,
    *,
    max_lag: int = DEFAULT_MAX_LAG_FRAMES,
    peak_mode: str = "norm",
) -> dict[str, Any]:
    """Seven stats for aligned (T, 3) series; peak_mode ``norm`` or ``displacement``."""
    real = np.asarray(real, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    if real.ndim == 1:
        real = real.reshape(-1, 3)
    if sim.ndim == 1:
        sim = sim.reshape(-1, 3)
    n = min(int(real.shape[0]), int(sim.shape[0]))
    if n <= 0 or real.shape[1] != 3 or sim.shape[1] != 3:
        return _nan_stats(vector_bias=True)
    real = real[:n]
    sim = sim[:n]
    err = sim - real
    bias_vec = np.mean(err, axis=0)
    mse = float(np.mean(np.sum(err**2, axis=1)))
    centered_err = err - bias_vec.reshape(1, 3)
    error_variance = float(np.mean(np.sum(centered_err**2, axis=1)))
    ss_res = float(np.sum(np.sum(err**2, axis=1)))
    real_centered = real - np.mean(real, axis=0, keepdims=True)
    ss_tot = float(np.sum(np.sum(real_centered**2, axis=1)))
    r_squared = float("nan") if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    if peak_mode == "displacement":
        real_mag = np.linalg.norm(real - real[0:1], axis=1)
        sim_mag = np.linalg.norm(sim - sim[0:1], axis=1)
    else:
        real_mag = np.linalg.norm(real, axis=1)
        sim_mag = np.linalg.norm(sim, axis=1)
    lag = best_cross_corr_lag(real_mag, sim_mag, max_lag=max_lag)
    aligned_r, aligned_s = _aligned_pair(real_mag, sim_mag, lag)
    if aligned_r.size == 0:
        phase_rmse = float("nan")
    else:
        phase_rmse = float(np.sqrt(np.mean((aligned_s - aligned_r) ** 2)))
    peak_error = float(abs(float(np.max(sim_mag)) - float(np.max(real_mag))))
    return {
        "mse": mse,
        "bias": [float(x) for x in bias_vec],
        "error_variance": error_variance,
        "r_squared": r_squared,
        "cross_corr_lag": float(lag),
        "phase_corrected_rmse": phase_rmse,
        "peak_magnitude_error": peak_error,
    }


def match_stats_to_jsonable(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Convert stats dict; non-finite floats become JSON null."""
    out: dict[str, Any] = {}
    for key, value in stats.items():
        if isinstance(value, list):
            out[key] = [
                None if not math.isfinite(float(v)) else float(v) for v in value
            ]
        else:
            number = float(value)
            out[key] = None if not math.isfinite(number) else number
    return out


def _runtime_mask(episode: Mapping[str, Any]) -> np.ndarray:
    n = int(np.asarray(episode["phase"]).reshape(-1).size)
    step_idx = np.asarray(
        episode.get("step_idx", np.arange(n, dtype=np.int32)), dtype=np.int32
    ).reshape(-1)
    stable = np.asarray(
        episode.get("stable", np.ones(n, dtype=bool)), dtype=bool
    ).reshape(-1)
    return (step_idx >= 0) & stable


def _hold_mask(episode: Mapping[str, Any], direction: int) -> np.ndarray:
    ep = dict(episode)
    if "dir_idx" not in ep:
        n = int(np.asarray(ep["phase"]).reshape(-1).size)
        ep["dir_idx"] = np.full(n, int(direction), dtype=np.int32)
    phase = np.asarray(ep["phase"]).reshape(-1)
    dir_idx = np.asarray(ep["dir_idx"]).reshape(-1)
    segments = iter_kept_hold_segments(
        phase=phase,
        dir_idx=dir_idx,
        direction=int(direction),
        min_frames=1,
    )
    mask = np.zeros(phase.shape[0], dtype=bool)
    if segments:
        hold_idx = np.concatenate(segments)
        mask[hold_idx] = True
    return mask & _runtime_mask(ep)


def _window_mask(episode: Mapping[str, Any], direction: int, window: str) -> np.ndarray:
    if window == "hold":
        return _hold_mask(episode, direction)
    if window != "full":
        raise ValueError(f"unknown window {window!r}")
    return _runtime_mask(episode)


def _apply_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[mask]
    return arr[mask]


def _sim_ft_wrist(episode: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(episode["ft_wrist"], dtype=np.float64)


def _woody_stack(
    episode: Mapping[str, Any],
    *,
    junction_names: Sequence[str],
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    woody = episode.get("woody_part_start_pos")
    if woody is None:
        n = int(np.sum(mask))
        zeros = np.zeros((n, 3), dtype=np.float64)
        return {name: zeros.copy() for name in junction_names}
    out: dict[str, np.ndarray] = {}
    for name in junction_names:
        if name not in woody:
            raise KeyError(f"missing woody_part_start_pos[{name!r}]")
        values = np.asarray(woody[name], dtype=np.float64)
        if values.ndim == 1:
            tiled = np.tile(values.reshape(1, 3), (mask.shape[0], 1))
            out[name] = tiled[mask]
        else:
            out[name] = values[mask]
    return out


def extract_window_series(
    *,
    real: Mapping[str, Any],
    sim: Mapping[str, Any],
    direction: int,
    window: str,
    junction_names: Sequence[str],
) -> dict[str, Any]:
    """Extract aligned truncated F/T/woody arrays for one window."""
    real_mask = _window_mask(real, direction, window)
    sim_mask = _window_mask(sim, direction, window)
    real_ft = _apply_mask(np.asarray(scored_ft_wrist(real), dtype=np.float64), real_mask)
    sim_ft = _apply_mask(_sim_ft_wrist(sim), sim_mask)
    n = min(int(real_ft.shape[0]), int(sim_ft.shape[0]))
    if n <= 0:
        empty3 = np.zeros((0, 3), dtype=np.float64)
        return {
            "force": empty3,
            "torque": empty3,
            "woody": {name: empty3.copy() for name in junction_names},
        }
    real_ft = real_ft[:n]
    sim_ft = sim_ft[:n]
    real_woody = _woody_stack(real, junction_names=junction_names, mask=real_mask)
    sim_woody = _woody_stack(sim, junction_names=junction_names, mask=sim_mask)
    return {
        "force": (real_ft[:, :3], sim_ft[:, :3]),
        "torque": (real_ft[:, 3:6], sim_ft[:, 3:6]),
        "woody": {
            name: (real_woody[name][:n], sim_woody[name][:n]) for name in junction_names
        },
    }


def direction_window_match_metrics(
    *,
    real: Mapping[str, Any],
    sim: Mapping[str, Any],
    direction: int,
    window: str,
    junction_names: Sequence[str],
    max_lag: int = DEFAULT_MAX_LAG_FRAMES,
) -> dict[str, Any]:
    """Per-axis + combined F/T and per-junction woody stats for one window."""
    series = extract_window_series(
        real=real,
        sim=sim,
        direction=direction,
        window=window,
        junction_names=junction_names,
    )
    force_r, force_s = series["force"]
    torque_r, torque_s = series["torque"]
    force_out: dict[str, Any] = {
        "fx": match_stats_to_jsonable(
            scalar_match_stats(force_r[:, 0], force_s[:, 0], max_lag=max_lag)
        ),
        "fy": match_stats_to_jsonable(
            scalar_match_stats(force_r[:, 1], force_s[:, 1], max_lag=max_lag)
        ),
        "fz": match_stats_to_jsonable(
            scalar_match_stats(force_r[:, 2], force_s[:, 2], max_lag=max_lag)
        ),
        "combined": match_stats_to_jsonable(
            vector3_match_stats(force_r, force_s, max_lag=max_lag, peak_mode="norm")
        ),
    }
    torque_out: dict[str, Any] = {
        "tx": match_stats_to_jsonable(
            scalar_match_stats(torque_r[:, 0], torque_s[:, 0], max_lag=max_lag)
        ),
        "ty": match_stats_to_jsonable(
            scalar_match_stats(torque_r[:, 1], torque_s[:, 1], max_lag=max_lag)
        ),
        "tz": match_stats_to_jsonable(
            scalar_match_stats(torque_r[:, 2], torque_s[:, 2], max_lag=max_lag)
        ),
        "combined": match_stats_to_jsonable(
            vector3_match_stats(torque_r, torque_s, max_lag=max_lag, peak_mode="norm")
        ),
    }
    woody_out: dict[str, Any] = {}
    for name in junction_names:
        w_r, w_s = series["woody"][name]
        woody_out[name] = match_stats_to_jsonable(
            vector3_match_stats(
                w_r, w_s, max_lag=max_lag, peak_mode="displacement"
            )
        )
    return {"force": force_out, "torque": torque_out, "woody": woody_out}


def resolve_junction_names(episode: Mapping[str, Any]) -> list[str]:
    names = episode.get("junction_names")
    if names is None:
        return list(CMA_WOODY_JUNCTIONS)
    return [str(name) for name in names]


def build_match_metrics_report(
    *,
    tree: str,
    cma_seed: int | None,
    train_direction_indices: Sequence[int],
    val_direction_indices: Sequence[int],
    baseline_log10: Sequence[float],
    fitted_log10: Sequence[float],
    recorded_val: Mapping[int, Mapping[str, Any]],
    baseline_replay_by_direction: Sequence[Mapping[str, Any]],
    fitted_replay_by_direction: Sequence[Mapping[str, Any]],
    val_direction_order: Sequence[int],
) -> dict[str, Any]:
    """Assemble the top-level ``match_metrics.json`` payload."""
    directions: dict[str, Any] = {}
    for local_i, direction in enumerate(val_direction_order):
        direction = int(direction)
        real = recorded_val[direction]
        junction_names = resolve_junction_names(real)
        baseline_sim = baseline_replay_by_direction[local_i]
        fitted_sim = fitted_replay_by_direction[local_i]
        dir_entry: dict[str, Any] = {}
        for window in ("full", "hold"):
            dir_entry[window] = {
                "baseline": direction_window_match_metrics(
                    real=real,
                    sim=baseline_sim,
                    direction=direction,
                    window=window,
                    junction_names=junction_names,
                ),
                "fitted": direction_window_match_metrics(
                    real=real,
                    sim=fitted_sim,
                    direction=direction,
                    window=window,
                    junction_names=junction_names,
                ),
            }
        directions[str(direction)] = dir_entry
    report: dict[str, Any] = {
        "tree": str(tree),
        "train_direction_indices": sorted(int(d) for d in train_direction_indices),
        "val_direction_indices": sorted(int(d) for d in val_direction_indices),
        "phenotype_log10": {
            "baseline": [float(x) for x in baseline_log10],
            "fitted": [float(x) for x in fitted_log10],
        },
        "directions": directions,
    }
    if cma_seed is not None:
        report["cma_seed"] = int(cma_seed)
    return report
