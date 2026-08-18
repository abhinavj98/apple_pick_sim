"""Holdout evaluation, verification gates, and holdout_report.json assembly."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np

from apple_pick_gym.batched_envs.batched_sysid_cmaes import (
    StructureCmaState,
    YoungsModulusBatchEvaluation,
    YoungsModulusEvaluation,
    generation_score_summary,
    to_strict_jsonable,
)
from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
    load_episode_metadata_for_directions,
    load_recorded_episodes_for_structure,
)
from apple_pick_gym.youngs_modulus_overlay_viz import (
    OverlayEpisode,
    write_youngs_modulus_overlay_html,
)
from apple_pick_sim.system_id.holdout_gates import (
    FORCE_FLOOR_N,
    FORCE_SLACK_N,
    TORQUE_FLOOR_NM,
    magnitude_ratio_ok,
    per_hold_means,
    signed_parallel_series,
    tcp_displacement_along_pull,
    trend_pearson_ok,
)
from apple_pick_sim.system_id.mmd_features import iter_kept_hold_segments, scored_ft_wrist

_METRIC_FLOAT_KEYS = (
    "eligible_mean_sinkhorn",
    "mae_force_n",
    "mae_torque_nm",
)


def ensure_dir_idx(episode: Mapping[str, Any], direction: int) -> dict[str, Any]:
    """Fill missing ``dir_idx`` with the episode direction index."""
    out = dict(episode)
    if "dir_idx" not in out:
        n_frames = int(np.asarray(out["action"]).shape[0])
        out["dir_idx"] = np.full(n_frames, int(direction), dtype=np.int32)
    return out


def _hold_frame_indices(episode: Mapping[str, Any], direction: int) -> np.ndarray:
    phase = np.asarray(episode["phase"]).reshape(-1)
    dir_idx = np.asarray(episode["dir_idx"]).reshape(-1)
    segments = iter_kept_hold_segments(
        phase=phase,
        dir_idx=dir_idx,
        direction=int(direction),
        min_frames=1,
    )
    if not segments:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(segments)


def cartesian_ft_mae(
    *,
    real: Mapping[str, Any],
    fitted: Mapping[str, Any],
    direction: int,
) -> tuple[float, float]:
    """Hold-frame mean |ΔF| (N) and |Δτ| (N·m); world ft_wrist via scored_ft_wrist."""
    real_ep = ensure_dir_idx(real, direction)
    fit_ep = ensure_dir_idx(fitted, direction)
    hold_idx = _hold_frame_indices(real_ep, direction)
    if hold_idx.size == 0:
        raise ValueError(f"direction {direction} has no hold frames for MAE")

    real_ft = np.asarray(scored_ft_wrist(real_ep), dtype=np.float64)
    fit_ft = np.asarray(scored_ft_wrist(fit_ep), dtype=np.float64)
    force_err = np.linalg.norm(real_ft[hold_idx, :3] - fit_ft[hold_idx, :3], axis=1)
    torque_err = np.linalg.norm(real_ft[hold_idx, 3:6] - fit_ft[hold_idx, 3:6], axis=1)
    return float(force_err.mean()), float(torque_err.mean())


def resolve_pull_direction(
    real: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None,
    direction: int,
) -> np.ndarray:
    """Episode metadata pull axis, else first hold-frame excitation_direction."""
    if metadata is not None:
        pull = metadata.get("pull_direction")
        if pull is not None:
            arr = np.asarray(pull, dtype=np.float64).reshape(-1)
            if arr.size == 3:
                return arr
    real_ep = ensure_dir_idx(real, direction)
    hold_idx = _hold_frame_indices(real_ep, direction)
    if hold_idx.size == 0:
        raise ValueError(f"direction {direction} has no hold frames for pull axis")
    excitation = np.asarray(real_ep["excitation_direction"], dtype=np.float64)
    if excitation.ndim == 1:
        return excitation.reshape(3)
    return excitation[int(hold_idx[0])].reshape(3)


def _tcp_displacement_full_series(
    episode: Mapping[str, Any],
    *,
    direction: int,
    pull_direction: Sequence[float],
) -> np.ndarray:
    """Embed hold-frame TCP displacement into a full-length phase-aligned series."""
    ep = ensure_dir_idx(episode, direction)
    phase = np.asarray(ep["phase"]).reshape(-1)
    hold_idx = _hold_frame_indices(ep, direction)
    hold_disp = tcp_displacement_along_pull(
        np.asarray(ep["tcp_pos"], dtype=np.float64),
        phase=phase,
        dir_idx=np.asarray(ep["dir_idx"]),
        direction=int(direction),
        pull_direction=pull_direction,
    )
    full = np.zeros(int(phase.shape[0]), dtype=np.float64)
    full[hold_idx] = hold_disp
    return full


def _signed_force_parallel(
    episode: Mapping[str, Any],
    *,
    direction: int,
    pull_direction: Sequence[float],
) -> np.ndarray:
    ep = ensure_dir_idx(episode, direction)
    ft = np.asarray(scored_ft_wrist(ep), dtype=np.float64)
    return signed_parallel_series(ft[:, :3], pull_direction)


def _torque_magnitudes(episode: Mapping[str, Any], *, direction: int) -> np.ndarray:
    ep = ensure_dir_idx(episode, direction)
    ft = np.asarray(scored_ft_wrist(ep), dtype=np.float64)
    return np.linalg.norm(ft[:, 3:6], axis=1)


def direction_verification(
    *,
    real: Mapping[str, Any],
    fitted: Mapping[str, Any],
    direction: int,
    pull_direction: Sequence[float],
) -> dict[str, Any]:
    """Per-direction force/TCP gates plus apple-pose diagnostics."""
    real_ep = ensure_dir_idx(real, direction)
    fit_ep = ensure_dir_idx(fitted, direction)
    pull = np.asarray(pull_direction, dtype=np.float64).reshape(3)

    real_f_par = _signed_force_parallel(real_ep, direction=direction, pull_direction=pull)
    fit_f_par = _signed_force_parallel(fit_ep, direction=direction, pull_direction=pull)
    real_tau = _torque_magnitudes(real_ep, direction=direction)
    fit_tau = _torque_magnitudes(fit_ep, direction=direction)
    hold_idx = _hold_frame_indices(real_ep, direction)

    real_f_mean = float(np.mean(np.abs(real_f_par[hold_idx])))
    fit_f_mean = float(np.mean(np.abs(fit_f_par[hold_idx])))
    real_t_mean = float(np.mean(real_tau[hold_idx]))
    fit_t_mean = float(np.mean(fit_tau[hold_idx]))

    force_mag_ok, force_ratio = magnitude_ratio_ok(
        real_mean=real_f_mean,
        fitted_mean=fit_f_mean,
        floor=FORCE_FLOOR_N,
        slack=FORCE_SLACK_N,
    )
    torque_mag_ok, torque_ratio = magnitude_ratio_ok(
        real_mean=real_t_mean,
        fitted_mean=fit_t_mean,
        floor=TORQUE_FLOOR_NM,
        slack=FORCE_SLACK_N,  # N·m; same numeric slack as force (0.4)
    )
    force_magnitude_ok = bool(force_mag_ok and torque_mag_ok)

    real_f_hold = per_hold_means(
        real_f_par,
        phase=np.asarray(real_ep["phase"]),
        dir_idx=np.asarray(real_ep["dir_idx"]),
        direction=int(direction),
    )
    fit_f_hold = per_hold_means(
        fit_f_par,
        phase=np.asarray(fit_ep["phase"]),
        dir_idx=np.asarray(fit_ep["dir_idx"]),
        direction=int(direction),
    )
    force_trend_ok, force_pearson_r = trend_pearson_ok(
        real_f_hold.tolist(),
        fit_f_hold.tolist(),
        magnitude_passed=force_magnitude_ok,
    )

    real_tcp_s = _tcp_displacement_full_series(
        real_ep,
        direction=int(direction),
        pull_direction=pull,
    )
    fit_tcp_s = _tcp_displacement_full_series(
        fit_ep,
        direction=int(direction),
        pull_direction=pull,
    )
    real_tcp_mean = float(np.mean(np.abs(real_tcp_s[hold_idx])))
    fit_tcp_mean = float(np.mean(np.abs(fit_tcp_s[hold_idx])))
    # Pose is meters (~cm); never reuse Newton force floor/slack.
    tcp_pose_magnitude_ok, tcp_ratio = magnitude_ratio_ok(
        real_mean=real_tcp_mean,
        fitted_mean=fit_tcp_mean,
        floor=0.0,
        slack=0.0,
    )
    real_tcp_hold = per_hold_means(
        real_tcp_s,
        phase=np.asarray(real_ep["phase"]),
        dir_idx=np.asarray(real_ep["dir_idx"]),
        direction=int(direction),
    )
    fit_tcp_hold = per_hold_means(
        fit_tcp_s,
        phase=np.asarray(fit_ep["phase"]),
        dir_idx=np.asarray(fit_ep["dir_idx"]),
        direction=int(direction),
    )
    tcp_pose_trend_ok, tcp_pearson_r = trend_pearson_ok(
        real_tcp_hold.tolist(),
        fit_tcp_hold.tolist(),
        magnitude_passed=bool(tcp_pose_magnitude_ok),
    )

    real_apple = signed_parallel_series(
        np.asarray(real_ep["apple_pos"], dtype=np.float64),
        pull,
    )
    fit_apple = signed_parallel_series(
        np.asarray(fit_ep["apple_pos"], dtype=np.float64),
        pull,
    )
    apple_ratio = (
        float(np.mean(np.abs(fit_apple[hold_idx])) / np.mean(np.abs(real_apple[hold_idx])))
        if float(np.mean(np.abs(real_apple[hold_idx]))) != 0.0
        else math.inf
    )
    _, apple_pearson_r = trend_pearson_ok(
        per_hold_means(
            real_apple,
            phase=np.asarray(real_ep["phase"]),
            dir_idx=np.asarray(real_ep["dir_idx"]),
            direction=int(direction),
        ).tolist(),
        per_hold_means(
            fit_apple,
            phase=np.asarray(fit_ep["phase"]),
            dir_idx=np.asarray(fit_ep["dir_idx"]),
            direction=int(direction),
        ).tolist(),
        magnitude_passed=True,
    )

    return {
        "force_magnitude_ok": bool(force_magnitude_ok),
        "force_trend_ok": bool(force_trend_ok),
        "tcp_pose_magnitude_ok": bool(tcp_pose_magnitude_ok),
        "tcp_pose_trend_ok": bool(tcp_pose_trend_ok),
        "force_ratio": float(force_ratio),
        "torque_ratio": float(torque_ratio),
        "force_pearson_r": force_pearson_r,
        "tcp_ratio": float(tcp_ratio),
        "tcp_pearson_r": tcp_pearson_r,
        "apple_ratio": float(apple_ratio),
        "apple_pearson_r": apple_pearson_r,
    }


def _require_finite(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number}")
    return number


def _normalize_metric_block(block: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _METRIC_FLOAT_KEYS:
        out[key] = _require_finite(float(block[key]), name=key)
    per_f = block.get("per_direction_mae_force_n", {})
    per_t = block.get("per_direction_mae_torque_nm", {})
    out["per_direction_mae_force_n"] = {
        str(int(k)): _require_finite(float(v), name=f"mae_force_n[{k}]")
        for k, v in sorted(per_f.items(), key=lambda item: int(item[0]))
    }
    out["per_direction_mae_torque_nm"] = {
        str(int(k)): _require_finite(float(v), name=f"mae_torque_nm[{k}]")
        for k, v in sorted(per_t.items(), key=lambda item: int(item[0]))
    }
    return out


def build_holdout_report(
    *,
    structure_idx: int,
    direction_split_seed: int | None,
    train_direction_indices: Sequence[int],
    val_direction_indices: Sequence[int],
    baseline_log10: Sequence[float],
    fitted_log10: Sequence[float],
    train_fitted: Mapping[str, Any],
    val_baseline: Mapping[str, Any],
    val_fitted: Mapping[str, Any],
    train_eligible_means: Sequence[float],
    val_overlay_paths: Mapping[int, str],
    val_verification_by_direction: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble the Slice-4 holdout report payload."""
    means = [float(x) for x in train_eligible_means if math.isfinite(float(x))]
    train_sinkhorn_decreased = len(means) >= 2 and means[-1] < means[0]

    val_base_sink = _require_finite(
        float(val_baseline["eligible_mean_sinkhorn"]),
        name="val_baseline.eligible_mean_sinkhorn",
    )
    val_fit_sink = _require_finite(
        float(val_fitted["eligible_mean_sinkhorn"]),
        name="val_fitted.eligible_mean_sinkhorn",
    )
    val_sinkhorn_improved = val_fit_sink < val_base_sink

    report: dict[str, Any] = {
        "structure_idx": int(structure_idx),
        "train_direction_indices": sorted(int(d) for d in train_direction_indices),
        "val_direction_indices": sorted(int(d) for d in val_direction_indices),
        "phenotype_log10": {
            "baseline": [float(x) for x in baseline_log10],
            "fitted": [float(x) for x in fitted_log10],
        },
        "train_fitted": _normalize_metric_block(train_fitted),
        "val_baseline": _normalize_metric_block(val_baseline),
        "val_fitted": _normalize_metric_block(val_fitted),
        "verification": {
            "train_sinkhorn_decreased": bool(train_sinkhorn_decreased),
            "val_sinkhorn_improved": bool(val_sinkhorn_improved),
            "by_direction": {
                str(int(direction)): dict(val_verification_by_direction[int(direction)])
                for direction in sorted(val_direction_indices)
            },
        },
        "val_overlay_paths": {
            str(int(direction)): str(val_overlay_paths[int(direction)])
            for direction in sorted(val_direction_indices)
        },
    }
    if direction_split_seed is not None:
        report["direction_split_seed"] = int(direction_split_seed)
    return report


def holdout_gate_failures(report: Mapping[str, Any]) -> list[str]:
    """Return human-readable failure messages for failed holdout gates."""
    verification = report.get("verification", {})
    failures: list[str] = []
    if not bool(verification.get("train_sinkhorn_decreased")):
        failures.append("train_sinkhorn_decreased")
    if not bool(verification.get("val_sinkhorn_improved")):
        failures.append("val_sinkhorn_improved")
    by_dir = verification.get("by_direction", {})
    for direction in sorted(by_dir.keys(), key=int):
        entry = by_dir[direction]
        for gate in (
            "force_magnitude_ok",
            "force_trend_ok",
            "tcp_pose_magnitude_ok",
            "tcp_pose_trend_ok",
        ):
            if not bool(entry.get(gate)):
                failures.append(f"{gate} direction {direction}")
    return failures


def write_holdout_report(output_dir: Path, report: Mapping[str, Any]) -> Path:
    """Write ``holdout_report.json`` atomically next to ``cmaes_report.json``."""
    path = Path(output_dir) / "holdout_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    strict = to_strict_jsonable(report)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(strict, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return path


def train_eligible_means_from_state(state: StructureCmaState) -> list[float]:
    """Eligible-mean Sinkhorn series from CMA generation records."""
    means: list[float] = []
    for record in state.generations:
        summary = generation_score_summary(
            record.penalty_metadata,
            penalized_fitness=record.penalized_fitness,
        )
        eligible_mean = summary.get("eligible_mean")
        if eligible_mean is None:
            continue
        value = float(eligible_mean)
        if math.isfinite(value):
            means.append(value)
    return means


def _metric_block_from_evaluations(
    *,
    evaluation: YoungsModulusEvaluation,
    recorded_by_direction: Mapping[int, Mapping[str, Any]],
    direction_indices: Sequence[int],
) -> dict[str, Any]:
    if not evaluation.scores:
        raise ValueError("evaluation has no scores")
    replay_by_dir = evaluation.replay_episodes[0]
    per_f: dict[int, float] = {}
    per_t: dict[int, float] = {}
    for local_i, direction in enumerate(direction_indices):
        mae_f, mae_t = cartesian_ft_mae(
            real=recorded_by_direction[int(direction)],
            fitted=replay_by_dir[local_i],
            direction=int(direction),
        )
        per_f[int(direction)] = mae_f
        per_t[int(direction)] = mae_t
    return {
        "eligible_mean_sinkhorn": float(evaluation.scores[0].aggregate_sinkhorn),
        "mae_force_n": float(np.mean(list(per_f.values()))),
        "mae_torque_nm": float(np.mean(list(per_t.values()))),
        "per_direction_mae_force_n": per_f,
        "per_direction_mae_torque_nm": per_t,
    }


def _overlay_episode_from_arrays(
    *,
    arrays: Mapping[str, Any],
    direction: int,
    structure_key: int,
    candidate_label: str,
    log10_e: tuple[float, float, float],
    pull_direction: np.ndarray,
) -> OverlayEpisode:
    phase = np.asarray(arrays["phase"], dtype=np.int8)
    n = int(phase.shape[0])
    sim_time = np.asarray(
        arrays.get("sim_time", np.arange(n, dtype=np.float64)),
        dtype=np.float64,
    ).reshape(-1)[:n]
    return OverlayEpisode(
        structure_idx=int(structure_key),
        direction_idx=int(direction),
        candidate_label=str(candidate_label),
        log10_e=log10_e,
        sim_time=sim_time,
        phase=phase,
        ft_wrist=np.asarray(scored_ft_wrist(arrays), dtype=np.float64),
        tcp_pos=np.asarray(arrays["tcp_pos"], dtype=np.float64),
        pull_direction=np.asarray(pull_direction, dtype=np.float64).reshape(3),
        excluded=False,
    )


def write_val_holdout_overlays(
    *,
    output_dir: Path,
    structure_idx: int,
    val_direction_indices: Sequence[int],
    recorded_by_direction: Mapping[int, Mapping[str, Any]],
    fitted_replay_by_direction: Sequence[Mapping[str, Any]],
    metadata_by_direction: Mapping[int, Mapping[str, Any]],
    fitted_log10: Sequence[float],
) -> dict[int, str]:
    """Write one real-vs-fitted overlay HTML per validation direction."""
    holdout_dir = (
        Path(output_dir) / f"structure_{int(structure_idx):03d}" / "holdout"
    )
    holdout_dir.mkdir(parents=True, exist_ok=True)
    fitted_log10_tuple = (
        float(fitted_log10[0]),
        float(fitted_log10[1]),
        float(fitted_log10[2]),
    )
    paths: dict[int, str] = {}
    for local_i, direction in enumerate(val_direction_indices):
        direction = int(direction)
        real = ensure_dir_idx(recorded_by_direction[direction], direction)
        fitted = ensure_dir_idx(fitted_replay_by_direction[local_i], direction)
        pull = resolve_pull_direction(
            real,
            metadata=metadata_by_direction.get(direction),
            direction=direction,
        )
        episodes = [
            _overlay_episode_from_arrays(
                arrays=real,
                direction=direction,
                structure_key=0,
                candidate_label="real",
                log10_e=(0.0, 0.0, 0.0),
                pull_direction=pull,
            ),
            _overlay_episode_from_arrays(
                arrays=fitted,
                direction=direction,
                structure_key=1,
                candidate_label="fitted",
                log10_e=fitted_log10_tuple,
                pull_direction=pull,
            ),
        ]
        overlay_path = holdout_dir / f"direction_{direction:03d}.html"
        write_youngs_modulus_overlay_html(
            episodes,
            overlay_path,
            max_overlay_candidates=2,
            title=f"Holdout dir {direction} — structure {int(structure_idx)}",
        )
        paths[direction] = str(overlay_path.resolve())
    return paths


def run_holdout_evaluation(
    *,
    output_dir: Path,
    dataset: Any,
    structure_idx: int,
    state: StructureCmaState,
    train_direction_indices: Sequence[int],
    val_direction_indices: Sequence[int],
    direction_split_seed: int | None,
    baseline_log10: Sequence[float],
    fitted_log10: Sequence[float],
    num_directions: int,
    include_excluded: bool,
    evaluate_val: Callable[[Sequence[float], Sequence[int]], YoungsModulusBatchEvaluation],
) -> tuple[dict[str, Any], list[str]]:
    """Evaluate val baseline/fitted, build report, write overlays; return failures."""
    if state.final_evaluation is None:
        raise RuntimeError("fitted structure missing final_evaluation for holdout")

    train_recorded = load_recorded_episodes_for_structure(
        dataset,
        structure_idx=int(structure_idx),
        num_directions=int(num_directions),
        direction_indices=tuple(int(d) for d in train_direction_indices),
        include_excluded=bool(include_excluded),
    )
    val_recorded = load_recorded_episodes_for_structure(
        dataset,
        structure_idx=int(structure_idx),
        num_directions=int(num_directions),
        direction_indices=tuple(int(d) for d in val_direction_indices),
        include_excluded=bool(include_excluded),
    )
    recorded_train = {
        int(d): ep
        for d, ep in zip(train_direction_indices, train_recorded, strict=True)
    }
    recorded_val = {
        int(d): ep for d, ep in zip(val_direction_indices, val_recorded, strict=True)
    }
    metadata_val = load_episode_metadata_for_directions(
        dataset,
        structure_idx=int(structure_idx),
        direction_indices=[int(d) for d in val_direction_indices],
    )

    train_fitted = _metric_block_from_evaluations(
        evaluation=state.final_evaluation,
        recorded_by_direction=recorded_train,
        direction_indices=train_direction_indices,
    )

    val_dirs = tuple(int(d) for d in val_direction_indices)
    val_baseline_batch = evaluate_val(list(baseline_log10), val_dirs)
    val_fitted_batch = evaluate_val(list(fitted_log10), val_dirs)
    val_baseline_eval = val_baseline_batch.evaluations[int(structure_idx)]
    val_fitted_eval = val_fitted_batch.evaluations[int(structure_idx)]

    val_baseline = _metric_block_from_evaluations(
        evaluation=val_baseline_eval,
        recorded_by_direction=recorded_val,
        direction_indices=val_direction_indices,
    )
    val_fitted = _metric_block_from_evaluations(
        evaluation=val_fitted_eval,
        recorded_by_direction=recorded_val,
        direction_indices=val_direction_indices,
    )

    val_verification: dict[int, dict[str, Any]] = {}
    for local_i, direction in enumerate(val_direction_indices):
        direction = int(direction)
        pull = resolve_pull_direction(
            recorded_val[direction],
            metadata=metadata_val.get(direction),
            direction=direction,
        )
        val_verification[direction] = direction_verification(
            real=recorded_val[direction],
            fitted=val_fitted_eval.replay_episodes[0][local_i],
            direction=direction,
            pull_direction=pull,
        )

    overlay_paths = write_val_holdout_overlays(
        output_dir=output_dir,
        structure_idx=int(structure_idx),
        val_direction_indices=val_direction_indices,
        recorded_by_direction=recorded_val,
        fitted_replay_by_direction=val_fitted_eval.replay_episodes[0],
        metadata_by_direction=metadata_val,
        fitted_log10=fitted_log10,
    )

    report = build_holdout_report(
        structure_idx=int(structure_idx),
        direction_split_seed=direction_split_seed,
        train_direction_indices=train_direction_indices,
        val_direction_indices=val_direction_indices,
        baseline_log10=baseline_log10,
        fitted_log10=fitted_log10,
        train_fitted=train_fitted,
        val_baseline=val_baseline,
        val_fitted=val_fitted,
        train_eligible_means=train_eligible_means_from_state(state),
        val_overlay_paths=overlay_paths,
        val_verification_by_direction=val_verification,
    )
    write_holdout_report(output_dir, report)
    return report, holdout_gate_failures(report)
