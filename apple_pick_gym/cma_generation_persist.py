"""Sparse per-generation CMA trajectory persist (bags + STATE_VECTOR HTML)."""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from apple_pick_gym.batched_envs.batched_sysid_cmaes import (
    CmaGenerationRecord,
    YoungsModulusCandidateScore,
    YoungsModulusEvaluation,
)
from apple_pick_sim.system_id.mmd_features import (
    build_state_matrix,
    mean_hold_block_errors,
)
from apple_pick_sim.system_id.trajectory_store import PHASE_TO_INT

SPARSE_ROLES: tuple[str, ...] = ("best", "mean", "worst_force")

_MOVE = int(PHASE_TO_INT["move_out"])
_HOLD = int(PHASE_TO_INT["hold"])

_STATE_BLOCK_SPECS: tuple[tuple[str, slice], ...] = (
    ("F_x", slice(0, 1)),
    ("F_y", slice(1, 2)),
    ("F_z", slice(2, 3)),
    ("τ_x", slice(3, 4)),
    ("τ_y", slice(4, 5)),
    ("τ_z", slice(5, 6)),
    ("v_x", slice(6, 7)),
    ("v_y", slice(7, 8)),
    ("v_z", slice(8, 9)),
    ("ω_x", slice(9, 10)),
    ("ω_y", slice(10, 11)),
    ("ω_z", slice(11, 12)),
    ("x", slice(12, 13)),
    ("y", slice(13, 14)),
    ("z", slice(14, 15)),
    ("rx", slice(15, 16)),
    ("ry", slice(16, 17)),
    ("rz", slice(17, 18)),
    ("w0_x", slice(18, 19)),
    ("w0_y", slice(19, 20)),
    ("w0_z", slice(20, 21)),
    ("w1_x", slice(21, 22)),
    ("w1_y", slice(22, 23)),
    ("w1_z", slice(23, 24)),
    ("bend0", slice(24, 25)),
    ("bend1", slice(25, 26)),
)

_COLUMN_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("F (N)", (0, 1, 2)),
    ("τ (N·m)", (3, 4, 5)),
    ("tcp v/ω", (6, 7, 8, 9, 10, 11)),
    ("tcp pos (m)", (12, 13, 14)),
    ("tcp rotvec (rad)", (15, 16, 17)),
    ("woody (m)", (18, 19, 20, 21, 22, 23)),
    ("bend (rad)", (24, 25)),
)


def select_sparse_generation_roles(
    *,
    scores: Sequence[YoungsModulusCandidateScore],
    penalized_fitness: Sequence[float],
    ask_samples_log10: Sequence[tuple[float, ...]],
    ask_mean_log10: Sequence[float],
) -> dict[str, int | None]:
    """Return candidate indices for best / mean / worst_force roles."""
    eligible: list[int] = []
    for idx, score in enumerate(scores):
        if score.disqualified:
            continue
        raw = float(score.aggregate_sinkhorn)
        if not math.isfinite(raw):
            continue
        eligible.append(int(idx))

    def _pick_best(role_scores: list[tuple[int, float]]) -> int | None:
        if not role_scores:
            return None
        role_scores.sort(key=lambda item: (item[1], item[0]))
        return int(role_scores[0][0])

    def _pick_worst(role_scores: list[tuple[int, float]]) -> int | None:
        if not role_scores:
            return None
        role_scores.sort(key=lambda item: (-item[1], item[0]))
        return int(role_scores[0][0])

    best_scores = [
        (idx, float(penalized_fitness[idx]))
        for idx in eligible
        if idx < len(penalized_fitness) and math.isfinite(float(penalized_fitness[idx]))
    ]
    mean_vec = np.asarray(ask_mean_log10, dtype=np.float64).reshape(-1)
    mean_scores: list[tuple[int, float]] = []
    for idx in eligible:
        if idx >= len(ask_samples_log10):
            continue
        sample = np.asarray(ask_samples_log10[idx], dtype=np.float64).reshape(-1)
        if sample.shape != mean_vec.shape:
            continue
        dist = float(np.linalg.norm(sample - mean_vec))
        mean_scores.append((idx, dist))

    force_scores: list[tuple[int, float]] = []
    for idx in eligible:
        err = scores[idx].mean_hold_force_err_n
        if err is None or not math.isfinite(float(err)):
            continue
        force_scores.append((idx, float(err)))

    return {
        "best": _pick_best(best_scores),
        "mean": _pick_best(mean_scores),
        "worst_force": _pick_worst(force_scores),
    }


def _runtime_arrays(arrays: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(arrays)
    step_idx = np.asarray(
        out.get("step_idx", np.arange(len(out["phase"]))), dtype=np.int32
    ).reshape(-1)
    mask = step_idx >= 0
    for key, value in list(out.items()):
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == mask.shape[0]:
            out[key] = arr[mask]
    return out


def _episode_sim_time(arrays: Mapping[str, Any], n_frames: int) -> np.ndarray:
    if "sim_time" in arrays:
        return np.asarray(arrays["sim_time"], dtype=np.float64).reshape(-1)
    return np.arange(int(n_frames), dtype=np.float64)


def _align_sim_time_to_real(
    *,
    real_time: np.ndarray,
    sim_rt: Mapping[str, Any],
    sim_n: int,
) -> np.ndarray:
    """Replay collectors omit ``sim_time``; use recorded seconds when frame-aligned."""
    if "sim_time" in sim_rt:
        return np.asarray(sim_rt["sim_time"], dtype=np.float64).reshape(-1)
    if int(real_time.size) == int(sim_n):
        return np.asarray(real_time, dtype=np.float64).reshape(-1).copy()
    return np.arange(int(sim_n), dtype=np.float64)


def build_direction_state_npz(
    *,
    real: Mapping[str, Any],
    sim: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    real_rt = _runtime_arrays(real)
    sim_rt = _runtime_arrays(sim)
    real_state = build_state_matrix(real_rt).astype(np.float32, copy=False)
    sim_state = build_state_matrix(sim_rt).astype(np.float32, copy=False)
    real_time = _episode_sim_time(real_rt, real_state.shape[0])
    sim_time = _align_sim_time_to_real(
        real_time=real_time,
        sim_rt=sim_rt,
        sim_n=sim_state.shape[0],
    )
    real_phase = np.asarray(real_rt["phase"], dtype=np.int8).reshape(-1)
    sim_phase = np.asarray(sim_rt["phase"], dtype=np.int8).reshape(-1)
    real_stable = np.asarray(
        real_rt.get("stable", np.ones(real_phase.shape[0], dtype=bool)), dtype=bool
    ).reshape(-1)
    sim_stable = np.asarray(
        sim_rt.get("stable", np.ones(sim_phase.shape[0], dtype=bool)), dtype=bool
    ).reshape(-1)
    return {
        "sim_time_real": real_time,
        "sim_time_sim": sim_time,
        "phase_real": real_phase,
        "phase_sim": sim_phase,
        "stable_real": real_stable,
        "stable_sim": sim_stable,
        "real_state": real_state,
        "sim_state": sim_state,
    }


def _phase_intervals(
    time: np.ndarray, phase: np.ndarray, code: int
) -> list[tuple[float, float]]:
    t = np.asarray(time, dtype=np.float64)
    p = np.asarray(phase, dtype=np.int8)
    if t.size == 0:
        return []
    intervals: list[tuple[float, float]] = []
    i = 0
    n = int(t.size)
    while i < n:
        if int(p[i]) != int(code):
            i += 1
            continue
        j = i
        while j < n and int(p[j]) == int(code):
            j += 1
        t0 = float(t[i])
        t1 = float(t[j - 1]) if j > i else t0
        intervals.append((t0, t1))
        i = j
    return intervals


def _block_table_rows(
    per_direction: Mapping[int, Mapping[str, Any]],
    direction_indices: Sequence[int],
) -> tuple[list[str], list[list[str]]]:
    headers = [
        "dir",
        "Sinkhorn",
        "|F| ratio",
        "force err (N)",
        "torque err (N·m)",
        "woody err (m)",
        "bend err (rad)",
    ]
    rows: list[list[str]] = []
    for direction in direction_indices:
        payload = per_direction.get(int(direction))
        if payload is None:
            continue
        err = payload.get("block_errors") or {}
        sink = payload.get("sinkhorn")
        ratio = payload.get("force_ratio")
        rows.append(
            [
                str(int(direction)),
                "" if sink is None else f"{float(sink):.2g}",
                "" if ratio is None else f"{float(ratio):.3f}",
                "" if err.get("force_err_n") is None else f"{float(err['force_err_n']):.3g}",
                "" if err.get("torque_err_nm") is None else f"{float(err['torque_err_nm']):.3g}",
                "" if err.get("woody_start_m") is None else f"{float(err['woody_start_m']):.3g}",
                "" if err.get("woody_bend_rad") is None else f"{float(err['woody_bend_rad']):.3g}",
            ]
        )
    return headers, rows


def make_generation_features_figure(
    *,
    per_direction: Mapping[int, Mapping[str, Any]],
    direction_indices: Sequence[int],
    title: str | None = None,
    log10_e: Sequence[float] | None = None,
) -> Any:
    """Build faceted real-vs-sim STATE_VECTOR time-series figure."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    dirs = [int(d) for d in direction_indices if int(d) in per_direction]
    if not dirs:
        fig = go.Figure()
        fig.update_layout(title=title or "CMA generation features (empty)")
        return fig

    n_dirs = len(dirs)
    n_cols = len(_COLUMN_GROUPS)
    subplot_titles: list[str] = []
    for d in dirs:
        for group_name, _ in _COLUMN_GROUPS:
            subplot_titles.append(f"dir {d} · {group_name}")

    fig = make_subplots(
        rows=n_dirs + 1,
        cols=n_cols,
        shared_xaxes=False,
        vertical_spacing=0.04,
        horizontal_spacing=0.03,
        subplot_titles=tuple(subplot_titles),
        specs=[[{"type": "xy"} for _ in range(n_cols)] for _ in range(n_dirs)]
        + [[{"type": "table", "colspan": n_cols}, *[None] * (n_cols - 1)]],
        row_heights=[1.0] * n_dirs + [0.35],
    )

    palette = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    )

    for row_i, direction in enumerate(dirs, start=1):
        payload = per_direction[int(direction)]
        real_state = np.asarray(payload["real_state"], dtype=np.float64)
        sim_state = np.asarray(payload["sim_state"], dtype=np.float64)
        t_real = np.asarray(payload["sim_time_real"], dtype=np.float64)
        t_sim = np.asarray(payload["sim_time_sim"], dtype=np.float64)
        phase_real = np.asarray(payload["phase_real"], dtype=np.int8)
        phase_sim = np.asarray(payload["phase_sim"], dtype=np.int8)
        raw_sink = payload.get("sinkhorn")
        sink = float("nan") if raw_sink is None else float(raw_sink)
        ratio = payload.get("force_ratio")
        hover_extra = f"dir={direction}<br>Sinkhorn={sink:.3g}"
        if ratio is not None:
            hover_extra += f"<br>|F| ratio={float(ratio):.3f}"

        for col_i, (_group_name, channel_indices) in enumerate(_COLUMN_GROUPS, start=1):
            for ch_offset, ch_idx in enumerate(channel_indices):
                name, sl = _STATE_BLOCK_SPECS[ch_idx]
                color = palette[ch_offset % len(palette)]
                y_real = real_state[:, sl].reshape(-1)
                y_sim = sim_state[:, sl].reshape(-1)
                fig.add_trace(
                    go.Scatter(
                        x=t_real,
                        y=y_real,
                        mode="lines",
                        name=f"real {name}",
                        legendgroup=f"real-{name}",
                        showlegend=row_i == 1 and col_i == 1 and ch_offset == 0,
                        line=dict(color=color, width=1.5),
                        hovertemplate=f"real {name}<br>%{{y:.4g}}<br>{hover_extra}<extra></extra>",
                    ),
                    row=row_i,
                    col=col_i,
                )
                fig.add_trace(
                    go.Scatter(
                        x=t_sim,
                        y=y_sim,
                        mode="lines",
                        name=f"sim {name}",
                        legendgroup=f"sim-{name}",
                        showlegend=False,
                        line=dict(color=color, width=1.5, dash="dash"),
                        hovertemplate=f"sim {name}<br>%{{y:.4g}}<br>{hover_extra}<extra></extra>",
                    ),
                    row=row_i,
                    col=col_i,
                )

            sample_phase = phase_real if phase_real.size else phase_sim
            sample_time = t_real if t_real.size else t_sim
            for code, color in ((_MOVE, "LightSkyBlue"), (_HOLD, "NavajoWhite")):
                for t0, t1 in _phase_intervals(sample_time, sample_phase, code):
                    fig.add_vrect(
                        x0=t0,
                        x1=t1,
                        fillcolor=color,
                        opacity=0.2,
                        line_width=0,
                        row=row_i,
                        col=col_i,
                        layer="below",
                    )

    headers, table_rows = _block_table_rows(per_direction, dirs)
    cell_values = (
        list(zip(*table_rows)) if table_rows else [[""] for _ in headers]
    )
    fig.add_trace(
        go.Table(
            header=dict(values=headers),
            cells=dict(values=cell_values),
        ),
        row=n_dirs + 1,
        col=1,
    )

    subtitle = ""
    if log10_e is not None:
        subtitle = " log10=" + ", ".join(f"{float(v):.4f}" for v in log10_e)
    fig.update_layout(
        title=(title or "CMA generation STATE_VECTOR") + subtitle,
        height=max(480, 260 * n_dirs + 180),
        template="plotly_white",
    )
    return fig


def write_generation_features_html(
    *,
    per_direction: Mapping[int, Mapping[str, Any]],
    direction_indices: Sequence[int],
    path: Path,
    title: str | None = None,
    log10_e: Sequence[float] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = make_generation_features_figure(
        per_direction=per_direction,
        direction_indices=direction_indices,
        title=title,
        log10_e=log10_e,
    )
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path.resolve()


def _score_to_metadata(
    score: YoungsModulusCandidateScore,
    *,
    role: str,
    candidate_index: int,
    log10_e: Sequence[float],
    fitness: float | None,
) -> dict[str, Any]:
    return {
        "role": str(role),
        "candidate_index": int(candidate_index),
        "log10_e": [float(v) for v in log10_e],
        "fitness": None if fitness is None else float(fitness),
        "aggregate_sinkhorn": float(score.aggregate_sinkhorn),
        "per_direction_sinkhorn": {
            str(int(k)): float(v) for k, v in score.per_direction_sinkhorn.items()
        },
        "mean_hold_force_err_n": score.mean_hold_force_err_n,
        "mean_hold_torque_err_nm": score.mean_hold_torque_err_nm,
        "mean_hold_woody_start_m": score.mean_hold_woody_start_m,
        "mean_hold_woody_bend_rad": score.mean_hold_woody_bend_rad,
        "per_direction_mean_hold_force_err_n": score.per_direction_mean_hold_force_err_n,
        "per_direction_mean_hold_torque_err_nm": score.per_direction_mean_hold_torque_err_nm,
        "per_direction_mean_hold_woody_start_m": score.per_direction_mean_hold_woody_start_m,
        "per_direction_mean_hold_woody_bend_rad": score.per_direction_mean_hold_woody_bend_rad,
        "per_direction_mean_hold_force_norm_n": score.per_direction_mean_hold_force_norm_n,
        "per_direction_mean_hold_torque_norm_nm": score.per_direction_mean_hold_torque_norm_nm,
    }


def _write_role_artifacts(
    *,
    role_dir: Path,
    role: str,
    candidate_index: int,
    score: YoungsModulusCandidateScore,
    log10_e: Sequence[float],
    fitness: float | None,
    replay_episodes: Sequence[Mapping[str, Any]],
    recorded_episodes: Sequence[Mapping[str, Any]],
    direction_indices: Sequence[int],
) -> list[str]:
    errors: list[str] = []
    role_dir.mkdir(parents=True, exist_ok=True)
    metadata = _score_to_metadata(
        score,
        role=role,
        candidate_index=candidate_index,
        log10_e=log10_e,
        fitness=fitness,
    )
    (role_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    per_direction: dict[int, dict[str, Any]] = {}
    for replay, recorded, direction in zip(
        replay_episodes, recorded_episodes, direction_indices, strict=True
    ):
        d = int(direction)
        try:
            npz_arrays = build_direction_state_npz(real=recorded, sim=replay)
            np.savez_compressed(role_dir / f"dir_{d:02d}.npz", **npz_arrays)
            block = mean_hold_block_errors(real=recorded, sim=replay, direction=d)
            sink = score.per_direction_sinkhorn.get(d)
            norms = (score.per_direction_mean_hold_force_norm_n or {}).get(d, {})
            real_norm = norms.get("real")
            sim_norm = norms.get("sim")
            ratio = None
            if (
                real_norm is not None
                and sim_norm is not None
                and float(real_norm) > 0.0
            ):
                ratio = float(sim_norm) / float(real_norm)
            per_direction[d] = {
                **npz_arrays,
                "block_errors": block,
                "sinkhorn": sink,
                "force_ratio": ratio,
            }
        except Exception as exc:
            errors.append(f"{role} dir {d}: {exc}")

    if per_direction:
        try:
            write_generation_features_html(
                per_direction=per_direction,
                direction_indices=direction_indices,
                path=role_dir / "features.html",
                title=f"CMA {role} candidate {candidate_index}",
                log10_e=log10_e,
            )
        except Exception as exc:
            errors.append(f"{role} features.html: {exc}")
    else:
        errors.append(f"{role}: no direction npz written")
    return errors


def persist_cma_generation_wave(
    *,
    output_dir: Path,
    structure_idx: int,
    record: CmaGenerationRecord,
    evaluation: YoungsModulusEvaluation,
    recorded_episodes: Sequence[Mapping[str, Any]],
    persist: bool = True,
) -> dict[str, Any]:
    """Write sparse generation artifacts; return summary for cmaes_report.json."""
    if not persist:
        return {
            "generation_index": int(record.generation_index),
            "structure_idx": int(structure_idx),
            "skipped": True,
        }

    gen_index = int(record.generation_index)
    summary: dict[str, Any] = {
        "generation_index": gen_index,
        "structure_idx": int(structure_idx),
        "ask_mean_log10": list(record.ask_distribution.mean_log10),
        "roles": {},
        "errors": [],
    }

    roles = select_sparse_generation_roles(
        scores=record.raw_scores,
        penalized_fitness=record.penalized_fitness,
        ask_samples_log10=record.ask_samples_log10,
        ask_mean_log10=record.ask_distribution.mean_log10,
    )
    gen_dir = (
        Path(output_dir)
        / f"structure_{int(structure_idx):03d}"
        / "generations"
        / f"gen_{gen_index:02d}"
    )
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)

    score_by_index = {int(s.candidate_index): s for s in record.raw_scores}
    role_entries: dict[str, Any] = {}
    all_errors: list[str] = []

    for role in SPARSE_ROLES:
        cand_idx = roles.get(role)
        if cand_idx is None:
            role_entries[role] = None
            continue
        score = score_by_index.get(int(cand_idx))
        if score is None:
            role_entries[role] = None
            all_errors.append(f"{role}: missing score for candidate {cand_idx}")
            continue
        if cand_idx >= len(evaluation.replay_episodes):
            role_entries[role] = None
            all_errors.append(f"{role}: missing replay for candidate {cand_idx}")
            continue
        log10_e = record.ask_samples_log10[int(cand_idx)]
        fitness = (
            float(record.penalized_fitness[int(cand_idx)])
            if int(cand_idx) < len(record.penalized_fitness)
            else None
        )
        role_dir = gen_dir / role
        role_errors = _write_role_artifacts(
            role_dir=role_dir,
            role=role,
            candidate_index=int(cand_idx),
            score=score,
            log10_e=log10_e,
            fitness=fitness,
            replay_episodes=evaluation.replay_episodes[int(cand_idx)],
            recorded_episodes=recorded_episodes,
            direction_indices=evaluation.direction_indices,
        )
        all_errors.extend(role_errors)
        role_entries[role] = {
            "candidate_index": int(cand_idx),
            "log10_e": [float(v) for v in log10_e],
            "fitness": fitness,
            "mean_hold_force_err_n": score.mean_hold_force_err_n,
            "path": str(role_dir.relative_to(output_dir)),
        }

    roles_path = gen_dir / "roles.json"
    roles_path.write_text(
        json.dumps(
            {
                "generation_index": gen_index,
                "structure_idx": int(structure_idx),
                "ask_mean_log10": list(record.ask_distribution.mean_log10),
                "roles": role_entries,
                "errors": all_errors,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary["roles"] = role_entries
    summary["errors"] = all_errors
    summary["path"] = str(gen_dir.relative_to(output_dir))
    return summary


def persist_cma_generation_wave_batch(
    *,
    output_dir: Path,
    wave_records: Mapping[int, CmaGenerationRecord],
    batch_evaluation: Any,
    dataset: Any,
    num_directions: int,
    include_excluded: bool,
    persist: bool = True,
) -> dict[int, list[dict[str, Any]]]:
    """Persist all structures in one generation wave."""
    from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
        load_recorded_episodes_for_structure,
    )

    summaries: dict[int, list[dict[str, Any]]] = {}
    if batch_evaluation is None:
        return summaries
    for structure_idx, record in wave_records.items():
        if structure_idx not in batch_evaluation.evaluations:
            continue
        evaluation = batch_evaluation.evaluations[structure_idx]
        recorded = load_recorded_episodes_for_structure(
            dataset,
            structure_idx=int(structure_idx),
            num_directions=int(num_directions),
            direction_indices=evaluation.direction_indices,
            include_excluded=bool(include_excluded),
        )
        entry = persist_cma_generation_wave(
            output_dir=output_dir,
            structure_idx=int(structure_idx),
            record=record,
            evaluation=evaluation,
            recorded_episodes=recorded,
            persist=persist,
        )
        summaries.setdefault(int(structure_idx), []).append(entry)
    return summaries
