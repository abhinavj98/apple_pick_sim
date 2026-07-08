from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from apple_pick_gym.grid_viz_metrics import bend_stiffness_values_match
from apple_pick_gym.grid_viz_metrics import log_l2_distance_to_gt
from apple_pick_gym.grid_viz_metrics import (
    woody_segment_pos_mse_hold_aggregated,
    woody_segment_pos_mse_masked,
)
from apple_pick_sim.system_id.batched_hold_quasi_static import hold_metric_frame_indices


def _as_2d(source: dict[str, Any], key: str, cols: int) -> np.ndarray:
    return np.asarray(source[key], dtype=np.float64).reshape(-1, cols)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(d * d))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def _require_no_leading_pre_weld(recorded: dict[str, Any]) -> None:
    if "step_idx" in recorded:
        step_idx = np.asarray(recorded["step_idx"], dtype=np.int32).reshape(-1)
        if step_idx.size >= 1 and int(step_idx[0]) == -1:
            raise ValueError(
                "recorded episode still contains a pre_weld row; call strip_pre_weld_rows first"
            )
    if "phase" in recorded:
        phase = np.asarray(recorded["phase"], dtype=np.int64).reshape(-1)
        if phase.size >= 1 and int(phase[0]) == -1:
            raise ValueError(
                "recorded episode still contains a pre_weld row; call strip_pre_weld_rows first"
            )


def _phase_mask(recorded: dict[str, Any], *, include_phase: int | None) -> np.ndarray:
    n = int(np.asarray(recorded["ft_wrist"]).reshape(-1, 6).shape[0])
    if "phase" not in recorded:
        return np.ones(n, dtype=bool)
    phase = np.asarray(recorded["phase"], dtype=np.int64).reshape(-1)
    if phase.shape[0] != n:
        phase = phase[:n]
    if include_phase is None:
        return phase != -1
    return (phase == int(include_phase)) & (phase != -1)


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


def replay_vs_recorded_errors(
    *,
    replay: dict[str, Any],
    recorded: dict[str, Any],
    include_phase: int | None,
) -> dict[str, float]:
    """Compute replay-vs-recorded errors over selected phases.

    include_phase:
      - None: include all phases except phase == -1 (pre_weld)
      - int: include only frames where recorded.phase == include_phase (and != -1)
    """
    _require_no_leading_pre_weld(recorded)

    ft_rep = _as_2d(replay, "ft_wrist", 6)
    tcp_rep = _as_2d(replay, "tcp_pos", 3)
    apple_rep = _as_2d(replay, "apple_pos", 3)

    ft_rec = _as_2d(recorded, "ft_wrist", 6)
    tcp_rec = _as_2d(recorded, "tcp_pos", 3)
    apple_rec = _as_2d(recorded, "apple_pos", 3)

    n = min(int(ft_rep.shape[0]), int(ft_rec.shape[0]))
    junction_names = list(recorded.get("junction_names", []))
    if n <= 0:
        return {
            "n_frames": 0.0,
            "n_used_frames": 0.0,
            "ft_force_rmse": float("nan"),
            "ft_torque_rmse": float("nan"),
            "tcp_pos_mse": float("nan"),
            "apple_pos_mse": float("nan"),
            "woody_pos_mse_by_segment": {},
        }

    ft_rep = ft_rep[:n]
    tcp_rep = tcp_rep[:n]
    apple_rep = apple_rep[:n]
    ft_rec = ft_rec[:n]
    tcp_rec = tcp_rec[:n]
    apple_rec = apple_rec[:n]

    rec_phase_aligned = np.asarray(recorded.get("phase", np.zeros(n)), dtype=np.int64).reshape(-1)[:n]
    mask = _phase_mask({"phase": rec_phase_aligned, "ft_wrist": ft_rec}, include_phase=include_phase)
    mask = np.asarray(mask, dtype=bool).reshape(-1)[:n]
    used = int(np.count_nonzero(mask))
    if used <= 0:
        return {
            "n_frames": float(n),
            "n_used_frames": 0.0,
            "ft_force_rmse": float("nan"),
            "ft_torque_rmse": float("nan"),
            "tcp_pos_mse": float("nan"),
            "apple_pos_mse": float("nan"),
            "woody_pos_mse_by_segment": woody_segment_pos_mse_masked(
                replay=replay,
                recorded=recorded,
                junction_names=junction_names,
                n=n,
                mask=mask,
            ),
        }

    return {
        "n_frames": float(n),
        "n_used_frames": float(used),
        "ft_force_rmse": _rmse(ft_rep[mask, :3], ft_rec[mask, :3]),
        "ft_torque_rmse": _rmse(ft_rep[mask, 3:], ft_rec[mask, 3:]),
        "tcp_pos_mse": _mse(tcp_rep[mask], tcp_rec[mask]),
        "apple_pos_mse": _mse(apple_rep[mask], apple_rec[mask]),
        "woody_pos_mse_by_segment": woody_segment_pos_mse_masked(
            replay=replay,
            recorded=recorded,
            junction_names=junction_names,
            n=n,
            mask=mask,
        ),
    }


def replay_vs_recorded_hold_aggregated_errors(
    *,
    replay: dict[str, Any],
    recorded: dict[str, Any],
    hold_phase_value: int = 1,
    aggregation: Literal["mean", "median"] = "median",
    use_latter_half: bool = True,
) -> dict[str, float]:
    """Compare replay vs recorded using one hold aggregate per signal.

    Matches ``trajectory_hold_aggregated_mse``: latter-half hold segments by default.
    """
    _require_no_leading_pre_weld(recorded)

    ft_rep = _as_2d(replay, "ft_wrist", 6)
    tcp_rep = _as_2d(replay, "tcp_pos", 3)
    apple_rep = _as_2d(replay, "apple_pos", 3)
    ft_rec = _as_2d(recorded, "ft_wrist", 6)
    tcp_rec = _as_2d(recorded, "tcp_pos", 3)
    apple_rec = _as_2d(recorded, "apple_pos", 3)

    n = min(int(ft_rep.shape[0]), int(ft_rec.shape[0]))
    junction_names = list(recorded.get("junction_names", []))
    if n <= 0:
        return {
            "n_frames": 0.0,
            "n_used_frames": 0.0,
            "ft_force_rmse": float("nan"),
            "ft_torque_rmse": float("nan"),
            "tcp_pos_mse": float("nan"),
            "apple_pos_mse": float("nan"),
            "woody_pos_mse_by_segment": {},
        }

    ft_rep = ft_rep[:n]
    tcp_rep = tcp_rep[:n]
    apple_rep = apple_rep[:n]
    ft_rec = ft_rec[:n]
    tcp_rec = tcp_rec[:n]
    apple_rec = apple_rec[:n]

    recorded_slice: dict[str, Any] = {
        "phase": np.asarray(recorded["phase"], dtype=np.int8).reshape(-1)[:n],
    }
    if "amplitude_m" in recorded:
        recorded_slice["amplitude_m"] = np.asarray(recorded["amplitude_m"], dtype=np.float64).reshape(
            -1
        )[:n]
    if "dir_idx" in recorded:
        recorded_slice["dir_idx"] = np.asarray(recorded["dir_idx"], dtype=np.int32).reshape(-1)[:n]

    hold_idx = hold_metric_frame_indices(recorded_slice, use_latter_half=use_latter_half)
    hold_idx = hold_idx[np.asarray(recorded_slice["phase"][hold_idx], dtype=np.int64) == int(hold_phase_value)]
    used = int(hold_idx.size)
    if used <= 0:
        return {
            "n_frames": float(n),
            "n_used_frames": 0.0,
            "ft_force_rmse": float("nan"),
            "ft_torque_rmse": float("nan"),
            "tcp_pos_mse": float("nan"),
            "apple_pos_mse": float("nan"),
            "woody_pos_mse_by_segment": woody_segment_pos_mse_hold_aggregated(
                replay=replay,
                recorded=recorded,
                junction_names=junction_names,
                n=n,
                hold_idx=hold_idx,
                aggregation=aggregation,
            ),
        }

    ft_rep_agg = _aggregate_rows(ft_rep[hold_idx], aggregation=aggregation)
    tcp_rep_agg = _aggregate_rows(tcp_rep[hold_idx], aggregation=aggregation)
    apple_rep_agg = _aggregate_rows(apple_rep[hold_idx], aggregation=aggregation)
    ft_rec_agg = _aggregate_rows(ft_rec[hold_idx], aggregation=aggregation)
    tcp_rec_agg = _aggregate_rows(tcp_rec[hold_idx], aggregation=aggregation)
    apple_rec_agg = _aggregate_rows(apple_rec[hold_idx], aggregation=aggregation)

    return {
        "n_frames": float(n),
        "n_used_frames": float(used),
        "ft_force_rmse": _rmse(ft_rep_agg[:3].reshape(1, -1), ft_rec_agg[:3].reshape(1, -1)),
        "ft_torque_rmse": _rmse(ft_rep_agg[3:].reshape(1, -1), ft_rec_agg[3:].reshape(1, -1)),
        "tcp_pos_mse": _mse(tcp_rep_agg.reshape(1, -1), tcp_rec_agg.reshape(1, -1)),
        "apple_pos_mse": _mse(apple_rep_agg.reshape(1, -1), apple_rec_agg.reshape(1, -1)),
        "woody_pos_mse_by_segment": woody_segment_pos_mse_hold_aggregated(
            replay=replay,
            recorded=recorded,
            junction_names=junction_names,
            n=n,
            hold_idx=hold_idx,
            aggregation=aggregation,
        ),
    }


def _mean_woody_pos_mse(segments: dict[str, float]) -> float:
    if not segments:
        return float("nan")
    vals = [float(v) for v in segments.values()]
    if not all(np.isfinite(v) for v in vals):
        return float("nan")
    return float(np.mean(vals))


def _parse_pos_weights(pos_weights: tuple[float, ...]) -> tuple[float, float, float]:
    if len(pos_weights) < 2:
        raise ValueError("pos_weights must have at least (w_tcp, w_apple)")
    w_tcp, w_apple = float(pos_weights[0]), float(pos_weights[1])
    w_woody = float(pos_weights[2]) if len(pos_weights) >= 3 else 1.0
    return w_tcp, w_apple, w_woody


def _composite_pos_err(
    *,
    tcp: float,
    apple: float,
    woody: float,
    w_tcp: float,
    w_apple: float,
    w_woody: float,
) -> float:
    err = float(w_tcp) * float(tcp) + float(w_apple) * float(apple)
    if np.isfinite(float(woody)):
        err += float(w_woody) * float(woody)
    return float(err)


def _mean_dict_all_directions(
    key: str,
    items: list[dict[str, Any]],
    *,
    expected_n: int,
) -> dict[str, float]:
    if len(items) != expected_n:
        raise ValueError(
            f"expected metrics for {expected_n} directions, got {len(items)}"
        )
    if not items:
        return {}
    first = items[0].get(key, {})
    if not isinstance(first, dict):
        raise ValueError(f"expected {key!r} to be a dict, got {type(first)!r}")
    names = sorted(str(name) for name in first)
    if not names:
        return {}
    out: dict[str, float] = {}
    for name in names:
        vals = []
        for item in items:
            segments = item.get(key, {})
            if not isinstance(segments, dict):
                raise ValueError(f"expected {key!r} to be a dict, got {type(segments)!r}")
            if name not in segments:
                raise ValueError(
                    f"segment {name!r} missing from {key!r} in one direction"
                )
            val = float(segments[name])
            if not np.isfinite(val):
                raise ValueError(
                    f"non-finite {key!r}[{name!r}] across directions: "
                    f"{sum(not np.isfinite(float(segments.get(n, float('nan')))) for n in names)}/{expected_n} invalid"
                )
            vals.append(val)
        out[name] = float(np.mean(vals))
    return out


def _mean_all_directions(key: str, items: list[dict[str, float]], *, expected_n: int) -> float:
    if len(items) != expected_n:
        raise ValueError(
            f"expected metrics for {expected_n} directions, got {len(items)}"
        )
    vals = [float(m[key]) for m in items]
    if not all(np.isfinite(v) for v in vals):
        raise ValueError(
            f"non-finite {key!r} across directions: "
            f"{sum(not np.isfinite(float(v)) for v in vals)}/{expected_n} invalid"
        )
    return float(np.mean(vals))


@dataclass(frozen=True)
class GridVizRow:
    structure_idx: int
    candidate_index: int
    gt_flag: bool
    primary: float
    secondary: float
    spur: float
    stem: float
    dist_log_gt: float
    n_frames_all: float
    err_pos_all: float
    err_force_all: float
    err_torque_all: float
    n_frames_hold: float  # number of hold-phase frames used for hold metrics
    err_pos_hold: float
    err_force_hold: float
    err_torque_hold: float
    woody_pos_mse_all: dict[str, float]
    woody_pos_mse_hold: dict[str, float]
    err_woody_pos_all: float
    err_woody_pos_hold: float
    n_directions_all: float
    n_directions_hold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_grid_viz_rows(
    *,
    structure_idx: int,
    candidates: list[Any],
    gt_candidate: Any,
    recorded_eps: list[dict[str, Any]],
    replay_eps_by_candidate: list[list[dict[str, Any]]],
    hold_phase_value: int = 1,
    pos_weights: tuple[float, ...] = (1.0, 1.0, 1.0),
    dist_keys: tuple[str, ...] = ("primary", "spur", "stem"),
    hold_aggregation: Literal["mean", "median", "none"] = "median",
    hold_use_latter_half: bool = True,
) -> list[GridVizRow]:
    """Build one row per candidate with separate pose vs force/torque errors.

    Inputs are already materialized arrays so this stays pure/fast for tests.
    """
    w_tcp, w_apple, w_woody = _parse_pos_weights(pos_weights)
    num_directions = len(recorded_eps)

    gt = {
        "primary": float(getattr(gt_candidate, "primary")),
        "secondary": float(getattr(gt_candidate, "secondary")),
        "spur": float(getattr(gt_candidate, "spur")),
        "stem": float(getattr(gt_candidate, "stem")),
    }

    out: list[GridVizRow] = []
    for cand_idx, candidate in enumerate(candidates):
        c = {
            "primary": float(getattr(candidate, "primary")),
            "secondary": float(getattr(candidate, "secondary")),
            "spur": float(getattr(candidate, "spur")),
            "stem": float(getattr(candidate, "stem")),
        }
        dist = log_l2_distance_to_gt(c, gt, keys=dist_keys)
        gt_flag = bend_stiffness_values_match(
            (c["primary"], c["secondary"], c["spur"], c["stem"]),
            (gt["primary"], gt["secondary"], gt["spur"], gt["stem"]),
        )

        per_dir_all = []
        per_dir_hold = []
        replay_dirs = replay_eps_by_candidate[cand_idx]
        if len(replay_dirs) != len(recorded_eps):
            raise ValueError("replay directions length must match recorded directions length")
        for d in range(len(recorded_eps)):
            per_dir_all.append(
                replay_vs_recorded_errors(
                    replay=replay_dirs[d],
                    recorded=recorded_eps[d],
                    include_phase=None,
                )
            )
            if hold_aggregation == "none":
                per_dir_hold.append(
                    replay_vs_recorded_errors(
                        replay=replay_dirs[d],
                        recorded=recorded_eps[d],
                        include_phase=int(hold_phase_value),
                    )
                )
            else:
                per_dir_hold.append(
                    replay_vs_recorded_hold_aggregated_errors(
                        replay=replay_dirs[d],
                        recorded=recorded_eps[d],
                        hold_phase_value=int(hold_phase_value),
                        aggregation=hold_aggregation,
                        use_latter_half=bool(hold_use_latter_half),
                    )
                )

        n_dirs_all = sum(
            1
            for m in per_dir_all
            if np.isfinite(float(m["n_used_frames"])) and float(m["n_used_frames"]) > 0.0
        )
        n_dirs_hold = sum(
            1
            for m in per_dir_hold
            if np.isfinite(float(m["n_used_frames"])) and float(m["n_used_frames"]) > 0.0
        )

        tcp_all = _mean_all_directions("tcp_pos_mse", per_dir_all, expected_n=num_directions)
        apple_all = _mean_all_directions("apple_pos_mse", per_dir_all, expected_n=num_directions)
        tcp_hold = _mean_all_directions("tcp_pos_mse", per_dir_hold, expected_n=num_directions)
        apple_hold = _mean_all_directions("apple_pos_mse", per_dir_hold, expected_n=num_directions)
        woody_all = _mean_dict_all_directions(
            "woody_pos_mse_by_segment",
            per_dir_all,
            expected_n=num_directions,
        )
        woody_hold = _mean_dict_all_directions(
            "woody_pos_mse_by_segment",
            per_dir_hold,
            expected_n=num_directions,
        )
        err_woody_all = _mean_woody_pos_mse(woody_all)
        err_woody_hold = _mean_woody_pos_mse(woody_hold)

        out.append(
            GridVizRow(
                structure_idx=int(structure_idx),
                candidate_index=int(cand_idx),
                gt_flag=bool(gt_flag),
                primary=c["primary"],
                secondary=c["secondary"],
                spur=c["spur"],
                stem=c["stem"],
                dist_log_gt=float(dist),
                n_frames_all=_mean_all_directions("n_used_frames", per_dir_all, expected_n=num_directions),
                err_pos_all=_composite_pos_err(
                    tcp=tcp_all,
                    apple=apple_all,
                    woody=err_woody_all,
                    w_tcp=w_tcp,
                    w_apple=w_apple,
                    w_woody=w_woody,
                ),
                err_force_all=_mean_all_directions("ft_force_rmse", per_dir_all, expected_n=num_directions),
                err_torque_all=_mean_all_directions("ft_torque_rmse", per_dir_all, expected_n=num_directions),
                n_frames_hold=_mean_all_directions("n_used_frames", per_dir_hold, expected_n=num_directions),
                err_pos_hold=_composite_pos_err(
                    tcp=tcp_hold,
                    apple=apple_hold,
                    woody=err_woody_hold,
                    w_tcp=w_tcp,
                    w_apple=w_apple,
                    w_woody=w_woody,
                ),
                err_force_hold=_mean_all_directions("ft_force_rmse", per_dir_hold, expected_n=num_directions),
                err_torque_hold=_mean_all_directions("ft_torque_rmse", per_dir_hold, expected_n=num_directions),
                woody_pos_mse_all=woody_all,
                woody_pos_mse_hold=woody_hold,
                err_woody_pos_all=err_woody_all,
                err_woody_pos_hold=err_woody_hold,
                n_directions_all=float(n_dirs_all),
                n_directions_hold=float(n_dirs_hold),
            )
        )

    return out
