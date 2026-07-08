"""Plotly visualization for batched sys-ID trajectory datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from apple_pick_sim.system_id.batched_hold_quasi_static import (
    HoldSegmentReport,
    StiffnessIdHoldThresholds,
    analyze_episode_hold_quasi_static,
    expected_hold_frame_count,
)
from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset
from apple_pick_sim.system_id.trajectory_store import PHASE_TO_INT

INT_TO_PHASE: dict[int, str] = {value: name for name, value in PHASE_TO_INT.items()}

_SUPPORT_JUNCTION_PREFIX = "primary_support_"

_TRACE_COLORS: dict[str, str] = {
    "tcp": "#1f77b4",
    "apple": "#ff7f0e",
    "primary_spur": "#2ca02c",
    "spur_stem": "#17becf",
    "stem_apple": "#d62728",
    "pull": "#9467bd",
    "move": "#8c564b",
}


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


def _viz_junction_names(junction_names: Sequence[str]) -> list[str]:
    return [
        name
        for name in junction_names
        if not str(name).startswith(_SUPPORT_JUNCTION_PREFIX)
    ]


def _phase_name(phase_value: int) -> str:
    return INT_TO_PHASE.get(int(phase_value), f"phase_{int(phase_value)}")


def _arrow_segments(
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tips = origins + directions * float(length)
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for origin, tip in zip(origins, tips, strict=True):
        xs.extend([float(origin[0]), float(tip[0]), None])
        ys.extend([float(origin[1]), float(tip[1]), None])
        zs.extend([float(origin[2]), float(tip[2]), None])
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), np.asarray(zs, dtype=np.float64)


def _subsample_indices(n: int, *, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, max_points, dtype=np.int64))


def episode_arrays_from_dataset(
    dataset: BatchedSysIdDataset,
    *,
    structure_idx: int,
    direction_idx: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load stacked frame arrays and episode metadata."""
    arrays = dataset.load_episode_obs_arrays(structure_idx, direction_idx)
    metadata = dataset.load_episode_metadata(structure_idx, direction_idx)
    return arrays, metadata


def make_episode_time_series_figure(
    arrays: dict[str, Any],
    metadata: dict[str, Any],
    *,
    title: str | None = None,
) -> Any:
    """Position components vs ``sim_time`` with phase-colored markers."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    time = np.asarray(arrays.get("sim_time", np.arange(arrays["tcp_pos"].shape[0])), dtype=np.float64)
    phase = np.asarray(arrays["phase"], dtype=np.int8)
    tcp = np.asarray(arrays["tcp_pos"], dtype=np.float64)
    apple = np.asarray(arrays["apple_pos"], dtype=np.float64)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("x [m]", "y [m]", "z [m]"),
        vertical_spacing=0.06,
    )

    axis_labels = ("x", "y", "z")
    for row, axis in enumerate(axis_labels, start=1):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=tcp[:, row - 1],
                mode="lines+markers",
                name="TCP",
                legendgroup="tcp",
                showlegend=(row == 1),
                line=dict(color=_TRACE_COLORS["tcp"]),
                marker=dict(size=4, color=phase, colorscale="Viridis", showscale=(row == 1), colorbar=dict(title="phase") if row == 1 else None),
                hovertemplate=(
                    f"time=%{{x:.3f}} s<br>tcp {axis}=%{{y:.4f}}<br>"
                    "phase=%{customdata}<extra></extra>"
                ),
                customdata=[_phase_name(int(p)) for p in phase],
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=time,
                y=apple[:, row - 1],
                mode="lines+markers",
                name="apple",
                legendgroup="apple",
                showlegend=(row == 1),
                line=dict(color=_TRACE_COLORS["apple"], dash="dash"),
                marker=dict(size=4),
                hovertemplate=f"time=%{{x:.3f}} s<br>apple {axis}=%{{y:.4f}}<extra></extra>",
            ),
            row=row,
            col=1,
        )

    pull = np.asarray(metadata.get("pull_direction", arrays["excitation_direction"][0]), dtype=np.float64).reshape(3)
    pull_text = f"pull=({pull[0]:+.3f}, {pull[1]:+.3f}, {pull[2]:+.3f})"
    fig.update_layout(
        title=title or (
            f"s{int(metadata.get('structure_idx', 0)):02d} "
            f"d{int(metadata.get('direction_idx', 0)):02d}: position vs time ({pull_text})"
        ),
        height=720,
        margin=dict(l=50, r=20, t=60, b=40),
    )
    fig.update_xaxes(title_text="sim_time [s]", row=3, col=1)
    return fig


def make_episode_spatial_figure(
    arrays: dict[str, Any],
    metadata: dict[str, Any],
    *,
    title: str | None = None,
    arrow_stride: int = 4,
    max_move_arrows: int = 12,
) -> Any:
    """3D trajectories with pull and incremental TCP movement arrows."""
    import plotly.graph_objects as go

    tcp = np.asarray(arrays["tcp_pos"], dtype=np.float64)
    apple = np.asarray(arrays["apple_pos"], dtype=np.float64)
    time = np.asarray(arrays.get("sim_time", np.arange(tcp.shape[0])), dtype=np.float64)
    phase = np.asarray(arrays["phase"], dtype=np.int8)
    junction_names = _viz_junction_names(arrays.get("junction_names", metadata.get("junction_names", ())))

    pull = np.asarray(metadata.get("pull_direction", arrays["excitation_direction"][0]), dtype=np.float64).reshape(3)
    pull = pull / max(float(np.linalg.norm(pull)), 1e-12)

    fig = go.Figure()

    def _line_trace(points: np.ndarray, *, name: str, color: str, width: int = 3) -> None:
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=width),
                marker=dict(size=3, color=time, colorscale="Plasma", showscale=False),
                hovertemplate=(
                    f"{name}<br>x=%{{x:.4f}} y=%{{y:.4f}} z=%{{z:.4f}}<extra></extra>"
                ),
            )
        )

    _line_trace(tcp, name="TCP", color=_TRACE_COLORS["tcp"])
    _line_trace(apple, name="apple", color=_TRACE_COLORS["apple"], width=2)

    woody_start = arrays.get("woody_part_start_pos", {})
    for name in junction_names:
        points = np.asarray(woody_start[name], dtype=np.float64)
        color = _TRACE_COLORS.get(name, "#7f7f7f")
        _line_trace(points, name=f"woody {name}", color=color, width=2)

    move_mask = phase == int(PHASE_TO_INT["move_out"])
    move_idx = np.where(move_mask)[0]
    if move_idx.size >= 2:
        move_tcp = tcp[move_mask]
        deltas = np.diff(move_tcp, axis=0)
        origins = move_tcp[:-1]
        dirs = _normalize_rows(deltas)
        pick = _subsample_indices(origins.shape[0], max_points=max_move_arrows)
        origins = origins[pick]
        dirs = dirs[pick]
        step_norm = float(np.median(np.linalg.norm(deltas, axis=1))) if deltas.size else 0.02
        arrow_len = max(0.015, step_norm * float(arrow_stride))
        xs, ys, zs = _arrow_segments(origins, dirs, length=arrow_len)
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                name="TCP move step",
                line=dict(color=_TRACE_COLORS["move"], width=4),
                hoverinfo="skip",
            )
        )

    anchor = apple[move_idx[0]] if move_idx.size else apple[0]
    pull_len = max(0.08, float(metadata.get("total_movement_m", 0.1)) * 0.9)
    xs, ys, zs = _arrow_segments(anchor.reshape(1, 3), pull.reshape(1, 3), length=pull_len)
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            name="pull direction",
            line=dict(color=_TRACE_COLORS["pull"], width=6),
            hovertemplate=(
                f"pull=({pull[0]:+.3f}, {pull[1]:+.3f}, {pull[2]:+.3f})<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title or (
            f"s{int(metadata.get('structure_idx', 0)):02d} "
            f"d{int(metadata.get('direction_idx', 0)):02d}: spatial trajectories"
        ),
        scene=dict(
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m]",
            aspectmode="data",
        ),
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def make_hold_quasi_static_figure(
    arrays: dict[str, Any],
    metadata: dict[str, Any],
    reports: Sequence[HoldSegmentReport],
    *,
    thresholds: StiffnessIdHoldThresholds | None = None,
    title: str | None = None,
) -> Any:
    """Force-stability plot for no-damping quasi-static stiffness-ID hold check.

    Three subplots (hold frames only):
    1. Force norm [N] with per-segment mean ± std bands and CV threshold line.
    2. TCP excursion from hold-start [mm] with threshold line.
    3. Per-segment force_cv and force_mean_drift_frac bar chart.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    th = thresholds or StiffnessIdHoldThresholds()
    phase = np.asarray(arrays["phase"], dtype=np.int8)
    hold_mask = phase == int(PHASE_TO_INT["hold"])
    time = np.asarray(arrays.get("sim_time", np.arange(phase.shape[0])), dtype=np.float64)
    amplitude = np.asarray(arrays.get("amplitude_m", np.zeros(phase.shape[0])), dtype=np.float64)
    ft = np.asarray(arrays["ft_wrist"], dtype=np.float64).reshape(-1, 6)
    tcp_pos = np.asarray(arrays["tcp_pos"], dtype=np.float64).reshape(-1, 3)
    hold_idx = np.where(hold_mask)[0]

    n_pass = sum(1 for r in reports if r.is_quasi_static)
    status = f"{n_pass}/{len(reports)} hold segments quasi-static"

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        subplot_titles=(
            "Force norm [N] during hold (no-damping: oscillates around elastic equilibrium)",
            "TCP excursion from hold-start [mm] (controller stiffness proxy)",
            "Per-segment quality: force_cv and mean-drift fraction (← lower is better)",
        ),
        vertical_spacing=0.10,
        row_heights=[0.40, 0.30, 0.30],
    )

    # --- subplot 1: force norm time series ---
    if hold_idx.size:
        fn = np.linalg.norm(ft[hold_idx, :3], axis=1)
        fig.add_trace(
            go.Scatter(
                x=time[hold_idx],
                y=fn,
                mode="lines+markers",
                name="force norm",
                line=dict(color=_TRACE_COLORS["tcp"], width=1),
                marker=dict(size=3, color=amplitude[hold_idx], colorscale="Viridis",
                            showscale=True, colorbar=dict(title="amp [m]", len=0.35, y=0.78)),
                hovertemplate="t=%{x:.3f} s<br>|F|=%{y:.3f} N<extra></extra>",
            ),
            row=1, col=1,
        )
        # Overlay per-segment mean ± std bands
        for report in reports:
            seg_mask = hold_mask & np.isclose(amplitude, report.amplitude_m)
            seg_idx = np.where(seg_mask)[0]
            if seg_idx.size == 0:
                continue
            if report.metrics_window == "latter_half":
                seg_idx = seg_idx[len(seg_idx) // 2:]
            t_seg = time[seg_idx]
            mean_line = np.full(t_seg.size, report.force_norm_mean_n)
            color = "rgba(255,100,0,0.6)" if report.is_quasi_static else "rgba(200,0,0,0.6)"
            fig.add_trace(
                go.Scatter(
                    x=t_seg, y=mean_line,
                    mode="lines", line=dict(color=color, width=2, dash="dash"),
                    name=f"mean amp={report.amplitude_m:.3f}",
                    showlegend=False,
                    hovertemplate=f"mean={report.force_norm_mean_n:.3f} N  cv={report.force_cv:.3f}<extra></extra>",
                ),
                row=1, col=1,
            )

    # --- subplot 2: TCP excursion ---
    for report in reports:
        seg_mask = hold_mask & np.isclose(amplitude, report.amplitude_m)
        seg_idx = np.where(seg_mask)[0]
        if seg_idx.size == 0:
            continue
        if report.metrics_window == "latter_half":
            seg_idx = seg_idx[len(seg_idx) // 2:]
        t_seg = time[seg_idx]
        excursion_mm = np.linalg.norm(tcp_pos[seg_idx] - tcp_pos[seg_idx[0]], axis=1) * 1000.0
        color = _TRACE_COLORS["tcp"] if report.is_quasi_static else "crimson"
        fig.add_trace(
            go.Scatter(
                x=t_seg, y=excursion_mm,
                mode="lines", line=dict(color=color, width=1),
                name=f"excursion amp={report.amplitude_m:.3f}",
                showlegend=False,
                hovertemplate=f"amp={report.amplitude_m:.3f} m<br>excursion=%{{y:.1f}} mm<extra></extra>",
            ),
            row=2, col=1,
        )
    fig.add_hline(y=th.max_tcp_excursion_m * 1000.0,
                  line_dash="dash", line_color="crimson",
                  annotation_text=f"threshold ({th.max_tcp_excursion_m*1000:.0f} mm)",
                  row=2, col=1)

    # --- subplot 3: bar chart of per-segment CV and mean-drift ---
    if reports:
        amps = [f"{r.amplitude_m:.3f}" for r in reports]
        cvs = [r.force_cv for r in reports]
        drifts = [r.force_mean_drift_frac for r in reports]
        colors = ["#2ca02c" if r.is_quasi_static else "#d62728" for r in reports]
        fig.add_trace(
            go.Bar(x=amps, y=cvs, name="force_cv",
                   marker_color=colors, opacity=0.75,
                   hovertemplate="amp=%{x} m<br>force_cv=%{y:.4f}<extra></extra>"),
            row=3, col=1,
        )
        fig.add_trace(
            go.Bar(x=amps, y=drifts, name="mean_drift_frac",
                   marker_color=colors, opacity=0.4,
                   hovertemplate="amp=%{x} m<br>mean_drift=%{y:.4f}<extra></extra>"),
            row=3, col=1,
        )
        fig.add_hline(y=th.max_force_cv, line_dash="dash", line_color="crimson",
                      annotation_text=f"force_cv threshold ({th.max_force_cv})",
                      row=3, col=1)
        fig.add_hline(y=th.max_force_mean_drift_frac, line_dash="dot", line_color="darkorange",
                      annotation_text=f"drift threshold ({th.max_force_mean_drift_frac})",
                      row=3, col=1)

    fig.update_layout(
        title=title or (
            f"s{int(metadata.get('structure_idx', 0)):02d} "
            f"d{int(metadata.get('direction_idx', 0)):02d}: hold quasi-static check — {status}"
        ),
        barmode="overlay",
        height=860,
        margin=dict(l=60, r=20, t=70, b=40),
    )
    fig.update_xaxes(title_text="amplitude [m]", row=3, col=1)
    fig.update_xaxes(title_text="sim_time [s]", row=1, col=1)
    fig.update_xaxes(title_text="sim_time [s]", row=2, col=1)
    fig.update_yaxes(title_text="force norm [N]", row=1, col=1)
    fig.update_yaxes(title_text="excursion [mm]", row=2, col=1)
    fig.update_yaxes(title_text="ratio (lower = better)", row=3, col=1)
    return fig


def write_episode_trajectory_bundle(
    dataset: BatchedSysIdDataset,
    *,
    structure_idx: int,
    direction_idx: int,
    output_dir: Path | str,
    check_hold: bool = True,
    thresholds: StiffnessIdHoldThresholds | None = None,
) -> list[Path]:
    """Write time-series and spatial Plotly HTML for one episode."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays, metadata = episode_arrays_from_dataset(
        dataset,
        structure_idx=structure_idx,
        direction_idx=direction_idx,
    )
    prefix = f"s{int(structure_idx):02d}_d{int(direction_idx):02d}"
    written: list[Path] = []

    time_path = out_dir / f"{prefix}_time_series.html"
    make_episode_time_series_figure(arrays, metadata).write_html(str(time_path), include_plotlyjs="cdn")
    written.append(time_path)

    spatial_path = out_dir / f"{prefix}_spatial_3d.html"
    make_episode_spatial_figure(arrays, metadata).write_html(str(spatial_path), include_plotlyjs="cdn")
    written.append(spatial_path)

    if check_hold:
        reports = analyze_episode_hold_quasi_static(
            arrays,
            metadata,
            thresholds=thresholds,
            manifest=dataset.manifest,
        )
        hold_path = out_dir / f"{prefix}_hold_quasi_static.html"
        make_hold_quasi_static_figure(arrays, metadata, reports, thresholds=thresholds).write_html(
            str(hold_path),
            include_plotlyjs="cdn",
        )
        written.append(hold_path)
    return written


def write_dataset_trajectory_viz(
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    structure_indices: Iterable[int] | None = None,
    direction_indices: Iterable[int] | None = None,
    check_hold: bool = True,
    thresholds: StiffnessIdHoldThresholds | None = None,
) -> list[Path]:
    """Write trajectory plots for selected episodes in a batched dataset."""
    from apple_pick_sim.system_id.batched_hold_quasi_static import write_dataset_hold_quasi_static_report

    dataset = BatchedSysIdDataset(dataset_dir)
    entries = dataset.episode_entries()
    if structure_indices is not None:
        structure_set = {int(v) for v in structure_indices}
        entries = [e for e in entries if int(e["structure_idx"]) in structure_set]
    if direction_indices is not None:
        direction_set = {int(v) for v in direction_indices}
        entries = [e for e in entries if int(e["direction_idx"]) in direction_set]

    written: list[Path] = []
    for entry in entries:
        written.extend(
            write_episode_trajectory_bundle(
                dataset,
                structure_idx=int(entry["structure_idx"]),
                direction_idx=int(entry["direction_idx"]),
                output_dir=output_dir,
                check_hold=check_hold,
                thresholds=thresholds,
            )
        )
    if check_hold:
        written.append(
            write_dataset_hold_quasi_static_report(
                dataset_dir,
                output_dir,
                thresholds=thresholds,
                structure_indices=structure_indices,
                direction_indices=direction_indices,
            )
        )
    return written
