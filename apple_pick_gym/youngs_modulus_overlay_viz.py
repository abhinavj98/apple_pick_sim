"""Faceted multi-E overlay plots for Young's-modulus grid verification.

Hygiene rules (see V.5.2 first-slice plan):
- Facet by pull direction; never mix directions on the same wrench axes.
- Default time panels show norms only (‖F‖, ‖T‖, ‖Δtcp‖).
- Cap distinct candidates; refuse if over the cap.
- Phase as vertical shaded bands, not extra legend entries.
- Excluded episodes are omitted from traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from apple_pick_sim.system_id.trajectory_store import PHASE_TO_INT

_MOVE = int(PHASE_TO_INT["move_out"])
_HOLD = int(PHASE_TO_INT["hold"])

# Qualitative Plotly qualitative palette (capped candidate count keeps this readable).
_CANDIDATE_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
)


@dataclass(frozen=True)
class OverlayEpisode:
    """One (candidate, direction) trajectory for overlay plotting."""

    structure_idx: int
    direction_idx: int
    candidate_label: str
    log10_e: tuple[float, float, float]
    sim_time: np.ndarray
    phase: np.ndarray
    ft_wrist: np.ndarray
    tcp_pos: np.ndarray
    pull_direction: np.ndarray
    excluded: bool = False


def _unique_candidate_keys(episodes: Sequence[OverlayEpisode]) -> list[int]:
    keys: list[int] = []
    for ep in episodes:
        if ep.excluded:
            continue
        if ep.structure_idx not in keys:
            keys.append(int(ep.structure_idx))
    return keys


def _color_for_structure(structure_idx: int, ordered_keys: Sequence[int]) -> str:
    i = list(ordered_keys).index(int(structure_idx))
    return _CANDIDATE_COLORS[i % len(_CANDIDATE_COLORS)]


def _phase_intervals(time: np.ndarray, phase: np.ndarray, code: int) -> list[tuple[float, float]]:
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


def _force_norm(ft: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(ft, dtype=np.float64)[:, :3], axis=1)


def _torque_norm(ft: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(ft, dtype=np.float64)[:, 3:6], axis=1)


def _tcp_disp_norm(tcp: np.ndarray) -> np.ndarray:
    tcp = np.asarray(tcp, dtype=np.float64)
    return np.linalg.norm(tcp - tcp[0], axis=1)


def _move_summary(ep: OverlayEpisode) -> tuple[float, float, float]:
    """Return (pull_azimuth_deg, final_disp_m, hold_median_force_n)."""
    tcp = np.asarray(ep.tcp_pos, dtype=np.float64)
    phase = np.asarray(ep.phase, dtype=np.int8)
    pull = np.asarray(ep.pull_direction, dtype=np.float64).reshape(3)
    pull = pull / max(float(np.linalg.norm(pull)), 1e-12)
    azimuth = float(np.degrees(np.arctan2(pull[1], pull[0])))
    move_mask = phase == _MOVE
    if np.any(move_mask):
        last_i = int(np.where(move_mask)[0][-1])
    else:
        last_i = int(tcp.shape[0] - 1)
    disp = float(np.linalg.norm(tcp[last_i] - tcp[0]))
    hold_mask = phase == _HOLD
    f_norm = _force_norm(ep.ft_wrist)
    if np.any(hold_mask):
        hold_f = float(np.median(f_norm[hold_mask]))
    else:
        hold_f = float(np.median(f_norm))
    return azimuth, disp, hold_f


def make_youngs_modulus_overlay_figure(
    episodes: Sequence[OverlayEpisode],
    *,
    max_overlay_candidates: int = 8,
    title: str | None = None,
) -> Any:
    """Build a faceted Plotly figure comparing candidates (norms by default)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if max_overlay_candidates < 1:
        raise ValueError("max_overlay_candidates must be >= 1")

    active = [ep for ep in episodes if not ep.excluded]
    cand_keys = _unique_candidate_keys(active)
    if len(cand_keys) > int(max_overlay_candidates):
        raise ValueError(
            f"got {len(cand_keys)} candidates but max_overlay_candidates="
            f"{int(max_overlay_candidates)}; shrink the log10-E grid "
            f"(or raise --max-overlay-candidates)"
        )

    dir_ids = sorted({int(ep.direction_idx) for ep in active})
    if not dir_ids:
        # Empty figure placeholder (all excluded / empty input).
        fig = go.Figure()
        fig.update_layout(title=title or "Young's modulus overlay (no episodes)")
        return fig

    n_dirs = len(dir_ids)
    row_titles = [f"dir {d}" for d in dir_ids] + ["move vs pull"]
    fig = make_subplots(
        rows=n_dirs + 1,
        cols=3,
        shared_xaxes=False,
        vertical_spacing=0.06,
        horizontal_spacing=0.06,
        subplot_titles=tuple(
            sum(
                (
                    [f"‖F‖ (dir {d})", f"‖T‖ (dir {d})", f"‖Δtcp‖ (dir {d})"]
                    for d in dir_ids
                ),
                [],
            )
            + ["Δtcp vs pull azimuth", "", ""]
        ),
        specs=(
            [[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}] for _ in dir_ids]
            + [[{"type": "xy", "colspan": 3}, None, None]]
        ),
        row_titles=row_titles,
    )

    shown_legend: set[str] = set()
    for ep in active:
        row_i = dir_ids.index(int(ep.direction_idx)) + 1
        color = _color_for_structure(ep.structure_idx, cand_keys)
        label = ep.candidate_label
        show_legend = label not in shown_legend
        if show_legend:
            shown_legend.add(label)
        t = np.asarray(ep.sim_time, dtype=np.float64)
        hover = (
            f"E_p={10 ** ep.log10_e[0]:.3g} Pa<br>"
            f"E_spur={10 ** ep.log10_e[1]:.3g} Pa<br>"
            f"E_stem={10 ** ep.log10_e[2]:.3g} Pa"
        )
        for col, y, y_name in (
            (1, _force_norm(ep.ft_wrist), "‖F‖"),
            (2, _torque_norm(ep.ft_wrist), "‖T‖"),
            (3, _tcp_disp_norm(ep.tcp_pos), "‖Δtcp‖"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=y,
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=show_legend and col == 1,
                    line=dict(color=color, width=1.5),
                    hovertemplate=f"{hover}<br>{y_name}=%{{y:.3g}}<extra></extra>",
                ),
                row=row_i,
                col=col,
            )

    # Move-vs-pull summary (all directions, color by candidate).
    move_shown: set[str] = set()
    for ep in active:
        az, disp, hold_f = _move_summary(ep)
        color = _color_for_structure(ep.structure_idx, cand_keys)
        label = ep.candidate_label
        show = label not in move_shown and label not in shown_legend
        if show:
            move_shown.add(label)
        fig.add_trace(
            go.Scatter(
                x=[az],
                y=[disp],
                mode="markers",
                name=label,
                legendgroup=label,
                showlegend=show,
                marker=dict(
                    color=color,
                    size=8.0 + 4.0 * min(hold_f / 10.0, 1.5),
                    symbol="circle",
                ),
                hovertemplate=(
                    f"{label}<br>azimuth=%{{x:.1f}}°<br>"
                    f"Δtcp=%{{y:.4g}} m<br>‖F‖_hold_med={hold_f:.3g} N"
                    f"<br>dir={ep.direction_idx}<extra></extra>"
                ),
            ),
            row=n_dirs + 1,
            col=1,
        )

    # Phase bands after traces exist (Plotly add_vrect needs subplot axes first).
    for row_i, d in enumerate(dir_ids, start=1):
        sample = next(ep for ep in active if int(ep.direction_idx) == d)
        for code, color in (
            (_MOVE, "LightSkyBlue"),
            (_HOLD, "NavajoWhite"),
        ):
            for t0, t1 in _phase_intervals(sample.sim_time, sample.phase, code):
                for col in (1, 2, 3):
                    fig.add_vrect(
                        x0=t0,
                        x1=t1,
                        fillcolor=color,
                        opacity=0.25,
                        line_width=0,
                        row=row_i,
                        col=col,
                        layer="below",
                    )

    fig.update_layout(
        title=title or "Young's modulus E-grid overlay",
        height=max(360, 220 * (n_dirs + 1)),
        legend_title_text="candidate",
        template="plotly_white",
    )
    fig.update_xaxes(title_text="sim_time (s)", row=1, col=1)
    fig.update_yaxes(title_text="N", row=1, col=1)
    fig.update_yaxes(title_text="N·m", row=1, col=2)
    fig.update_yaxes(title_text="m", row=1, col=3)
    fig.update_xaxes(title_text="pull azimuth (deg)", row=n_dirs + 1, col=1)
    fig.update_yaxes(title_text="‖Δtcp‖ end of move (m)", row=n_dirs + 1, col=1)
    return fig


def write_youngs_modulus_overlay_html(
    episodes: Sequence[OverlayEpisode],
    path: str | Path,
    *,
    max_overlay_candidates: int = 8,
    title: str | None = None,
) -> Path:
    """Write the overlay figure to ``path`` and return the resolved path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = make_youngs_modulus_overlay_figure(
        episodes,
        max_overlay_candidates=max_overlay_candidates,
        title=title,
    )
    fig.write_html(str(out), include_plotlyjs="cdn")
    return out.resolve()


def overlay_episodes_from_batched_dataset(
    dataset: Any,
    *,
    candidate_labels: Sequence[str],
    candidate_log10_e: Sequence[tuple[float, float, float]],
) -> list[OverlayEpisode]:
    """Load overlay episodes from a ``batched_sysid_v1`` dataset.

    Structure index ``s`` maps to ``candidate_labels[s]`` / ``candidate_log10_e[s]``.
    """
    if len(candidate_labels) != len(candidate_log10_e):
        raise ValueError("candidate_labels and candidate_log10_e length mismatch")
    collection = dataset.manifest.get("collection", {})
    n_structures = int(collection.get("num_structures", len(dataset.structure_summaries())))
    n_directions = int(collection.get("num_directions", 0))
    if n_directions < 1:
        # Infer from episode catalog when collection block is incomplete.
        dirs = {
            int(e["direction_idx"])
            for e in dataset.episode_entries()
        }
        n_directions = max(dirs) + 1 if dirs else 0
    if n_structures != len(candidate_labels):
        raise ValueError(
            f"dataset has {n_structures} structures but got "
            f"{len(candidate_labels)} candidate labels"
        )
    episodes: list[OverlayEpisode] = []
    excluded_lookup: dict[tuple[int, int], bool] = {}
    for entry in dataset.episode_entries():
        excluded_lookup[
            (int(entry["structure_idx"]), int(entry["direction_idx"]))
        ] = bool(entry.get("excluded", False))

    for s in range(n_structures):
        for d in range(n_directions):
            arrays = dataset.load_episode_obs_arrays(s, d)
            meta = dataset.load_episode_metadata(s, d)
            pull = np.asarray(
                meta.get("pull_direction", [1.0, 0.0, 0.0]), dtype=np.float64
            )
            step_idx = np.asarray(
                arrays.get("step_idx", np.arange(len(arrays["phase"]))),
                dtype=np.int32,
            )
            runtime_mask = step_idx >= 0
            sim_time = np.asarray(
                arrays.get("sim_time", np.arange(len(arrays["phase"]))),
                dtype=np.float64,
            )[runtime_mask]
            phase = np.asarray(arrays["phase"], dtype=np.int8)[runtime_mask]
            ft_wrist = np.asarray(arrays["ft_wrist"], dtype=np.float64)[runtime_mask]
            tcp_pos = np.asarray(arrays["tcp_pos"], dtype=np.float64)[runtime_mask]
            episodes.append(
                OverlayEpisode(
                    structure_idx=s,
                    direction_idx=d,
                    candidate_label=str(candidate_labels[s]),
                    log10_e=(
                        float(candidate_log10_e[s][0]),
                        float(candidate_log10_e[s][1]),
                        float(candidate_log10_e[s][2]),
                    ),
                    sim_time=sim_time,
                    phase=phase,
                    ft_wrist=ft_wrist,
                    tcp_pos=tcp_pos,
                    pull_direction=pull.reshape(3),
                    excluded=excluded_lookup.get((s, d), False),
                )
            )
    return episodes


def select_overlay_candidate_indices(
    scores: Sequence[Any],
    *,
    max_candidates: int,
) -> list[int]:
    """Pick top-ranked eligible candidates plus GT when not already included."""
    if int(max_candidates) < 1:
        raise ValueError("max_candidates must be >= 1")

    eligible = sorted(
        (
            score
            for score in scores
            if not bool(getattr(score, "disqualified", True))
            and getattr(score, "rank", None) is not None
        ),
        key=lambda score: (int(score.rank), int(score.candidate_index)),
    )
    if not eligible:
        return []

    gt_idx = next(
        (int(score.candidate_index) for score in scores if bool(getattr(score, "is_gt", False))),
        None,
    )
    top_indices = [int(score.candidate_index) for score in eligible[: int(max_candidates)]]
    if int(max_candidates) == 1:
        return top_indices
    gt_eligible = gt_idx is not None and any(
        int(score.candidate_index) == int(gt_idx) for score in eligible
    )
    if gt_eligible and int(gt_idx) not in top_indices:
        selected: list[int] = []
        for score in eligible:
            if int(score.candidate_index) == int(gt_idx):
                continue
            if len(selected) >= int(max_candidates) - 1:
                break
            selected.append(int(score.candidate_index))
        selected.append(int(gt_idx))
        return selected
    return top_indices


def overlay_episodes_from_replay_evaluation(
    evaluation: Any,
    candidate_indices: Sequence[int],
) -> list[OverlayEpisode]:
    """Build overlay episodes directly from in-memory replay evaluation data."""
    import math

    scores_by_index = {
        int(score.candidate_index): score for score in evaluation.scores
    }
    episodes: list[OverlayEpisode] = []
    direction_ids = tuple(int(d) for d in evaluation.direction_indices)

    for candidate_index in candidate_indices:
        cand_idx = int(candidate_index)
        score = scores_by_index[cand_idx]
        candidate = score.candidate
        log10_e = (
            math.log10(float(candidate.primary)),
            math.log10(float(candidate.spur)),
            math.log10(float(candidate.stem)),
        )
        replay_by_direction = evaluation.replay_episodes[cand_idx]
        for dir_local, replay in enumerate(replay_by_direction):
            direction_idx = int(direction_ids[dir_local]) if dir_local < len(direction_ids) else int(
                np.asarray(replay.get("dir_idx", [0]), dtype=np.int32).reshape(-1)[0]
            )
            phase = np.asarray(replay["phase"], dtype=np.int8)
            n = int(phase.shape[0])
            sim_time = np.asarray(
                replay.get("sim_time", np.arange(n, dtype=np.float64)),
                dtype=np.float64,
            ).reshape(-1)[:n]
            ft_wrist = np.asarray(replay["ft_wrist"], dtype=np.float64)
            tcp_pos = np.asarray(replay["tcp_pos"], dtype=np.float64)
            pull = np.asarray(
                replay.get("excitation_direction", [1.0, 0.0, 0.0]),
                dtype=np.float64,
            ).reshape(-1, 3)[0]
            stable = np.asarray(replay.get("stable", np.ones(n, dtype=bool)), dtype=bool).reshape(-1)[:n]
            excluded = not bool(np.all(stable))
            episodes.append(
                OverlayEpisode(
                    structure_idx=cand_idx,
                    direction_idx=direction_idx,
                    candidate_label=str(candidate.short_label()),
                    log10_e=log10_e,
                    sim_time=sim_time,
                    phase=phase,
                    ft_wrist=ft_wrist,
                    tcp_pos=tcp_pos,
                    pull_direction=pull.reshape(3),
                    excluded=excluded,
                )
            )
    return episodes
