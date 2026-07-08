from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from apple_pick_gym.grid_viz_metrics import average_ranks
from apple_pick_gym.grid_viz_table import GridVizRow


Metric = Literal[
    "err_pos_all",
    "err_force_all",
    "err_torque_all",
    "err_woody_pos_all",
    "err_pos_hold",
    "err_force_hold",
    "err_torque_hold",
    "err_woody_pos_hold",
]

PARETO_DEFAULT_METRICS: tuple[Metric, Metric, Metric] = (
    "err_force_hold",
    "err_torque_hold",
    "err_pos_hold",
)


def _metric_label(metric: str) -> str:
    return {
        "err_pos_all": "pos_err (tcp+apple+woody MSE, all frames)",
        "err_force_all": "force_rmse (N, all frames)",
        "err_torque_all": "torque_rmse (N·m, all frames)",
        "err_woody_pos_all": "woody_pos_mse (mean over segments, all frames)",
        "err_pos_hold": "pos_err (tcp+apple+woody MSE, hold)",
        "err_force_hold": "force_rmse (N, hold)",
        "err_torque_hold": "torque_rmse (N·m, hold)",
        "err_woody_pos_hold": "woody_pos_mse (mean over segments, hold)",
    }.get(metric, metric)


def compute_pareto_front_mask(
    rows: Sequence[GridVizRow],
    *,
    metrics: Sequence[Metric] = PARETO_DEFAULT_METRICS,
) -> np.ndarray:
    """Return a boolean mask for the nondominated (Pareto) set (minimization)."""
    if len(metrics) < 2:
        raise ValueError("Pareto front requires >= 2 metrics")
    if not rows:
        return np.zeros((0,), dtype=bool)

    # (n, d) objective array (lower is better).
    obj = np.stack(
        [np.array([float(getattr(r, m)) for r in rows], dtype=np.float64) for m in metrics],
        axis=1,
    )

    n = int(obj.shape[0])
    dominated = np.zeros((n,), dtype=bool)
    # O(n^2 d) is fine for small grids (e.g. 27 candidates).
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(obj[j] <= obj[i]) and np.any(obj[j] < obj[i]):
                dominated[i] = True
                break
    return ~dominated


def _log_range_sensitivity(
    rows: Sequence[GridVizRow],
    *,
    param: Literal["primary", "spur", "stem"],
    metric: Metric,
    eps: float = 1e-30,
) -> dict[str, float]:
    """Estimate main-effect sensitivity of `metric` to `param` on a grid.

    For each slice where the other two params are held fixed, compute:
      delta = max(log(metric)) - min(log(metric))
    Then summarize deltas across slices with median/mean.
    """
    if not rows:
        return {"median_log_range": float("nan"), "mean_log_range": float("nan")}

    other_params: tuple[str, str]
    if param == "primary":
        other_params = ("spur", "stem")
    elif param == "spur":
        other_params = ("primary", "stem")
    else:
        other_params = ("primary", "spur")

    # Group by the other two params (exact float equality is OK here: values come from a grid).
    groups: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for r in rows:
        key = (float(getattr(r, other_params[0])), float(getattr(r, other_params[1])))
        groups.setdefault(key, []).append((float(getattr(r, param)), float(getattr(r, metric))))

    deltas: list[float] = []
    for pts in groups.values():
        if len(pts) < 2:
            continue
        pts.sort(key=lambda t: t[0])
        ys = [float(np.log(max(y, eps))) for _, y in pts]
        deltas.append(float(max(ys) - min(ys)))

    if not deltas:
        return {"median_log_range": 0.0, "mean_log_range": 0.0}

    arr = np.array(deltas, dtype=np.float64)
    return {"median_log_range": float(np.median(arr)), "mean_log_range": float(np.mean(arr))}


def make_pareto_hold_projections(
    *,
    rows: Sequence[GridVizRow],
    metrics: Sequence[Metric] = PARETO_DEFAULT_METRICS,
    title: str,
) -> Any:
    """2D projections of hold Pareto front (3 subplots)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if len(metrics) != 3:
        raise ValueError("make_pareto_hold_projections expects exactly 3 metrics")

    m0, m1, m2 = metrics
    pareto = compute_pareto_front_mask(rows, metrics=metrics)
    is_gt = np.array([bool(r.gt_flag) for r in rows], dtype=bool)

    def _scatter(xm: Metric, ym: Metric, *, showlegend: bool) -> list[Any]:
        x = np.array([float(getattr(r, xm)) for r in rows], dtype=np.float64)
        y = np.array([float(getattr(r, ym)) for r in rows], dtype=np.float64)
        traces: list[Any] = []

        # Non-Pareto candidates.
        mask_non = (~pareto) & (~is_gt)
        if np.any(mask_non):
            traces.append(
                go.Scatter(
                    x=x[mask_non],
                    y=y[mask_non],
                    mode="markers",
                    name="candidates",
                    marker=dict(size=7, color="rgba(120,120,120,0.35)"),
                    showlegend=showlegend,
                    hovertext=[
                        f"candidate={rows[i].candidate_index}<br>dist_log_gt={rows[i].dist_log_gt:.3g}"
                        for i in np.where(mask_non)[0].tolist()
                    ],
                    hoverinfo="text",
                )
            )

        # Pareto candidates.
        mask_p = pareto & (~is_gt)
        if np.any(mask_p):
            traces.append(
                go.Scatter(
                    x=x[mask_p],
                    y=y[mask_p],
                    mode="markers",
                    name="Pareto",
                    marker=dict(size=8, color="rgba(0,120,255,0.85)", line=dict(width=1, color="black")),
                    showlegend=showlegend,
                    hovertext=[
                        f"candidate={rows[i].candidate_index}<br>dist_log_gt={rows[i].dist_log_gt:.3g}"
                        for i in np.where(mask_p)[0].tolist()
                    ],
                    hoverinfo="text",
                )
            )

        # GT point.
        if np.any(is_gt):
            traces.append(
                go.Scatter(
                    x=x[is_gt],
                    y=y[is_gt],
                    mode="markers",
                    name="GT",
                    marker=dict(size=11, symbol="diamond", color="rgba(255,80,80,0.9)", line=dict(width=2, color="black")),
                    showlegend=showlegend,
                    hovertext=[
                        f"candidate={rows[i].candidate_index}<br>GT=True<br>dist_log_gt={rows[i].dist_log_gt:.3g}"
                        for i in np.where(is_gt)[0].tolist()
                    ],
                    hoverinfo="text",
                )
            )

        return traces

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"{m0} vs {m1}",
            f"{m0} vs {m2}",
            f"{m1} vs {m2}",
        ),
    )
    for t in _scatter(m0, m1, showlegend=True):
        fig.add_trace(t, row=1, col=1)
    for t in _scatter(m0, m2, showlegend=False):
        fig.add_trace(t, row=1, col=2)
    for t in _scatter(m1, m2, showlegend=False):
        fig.add_trace(t, row=1, col=3)

    fig.update_xaxes(title_text=_metric_label(m0), row=1, col=1)
    fig.update_yaxes(title_text=_metric_label(m1), row=1, col=1)
    fig.update_xaxes(title_text=_metric_label(m0), row=1, col=2)
    fig.update_yaxes(title_text=_metric_label(m2), row=1, col=2)
    fig.update_xaxes(title_text=_metric_label(m1), row=1, col=3)
    fig.update_yaxes(title_text=_metric_label(m2), row=1, col=3)
    fig.update_layout(title=title, margin=dict(l=40, r=20, t=70, b=40))
    return fig


def make_param_sensitivity_bars(
    *,
    rows: Sequence[GridVizRow],
    metric: Metric,
    title: str,
) -> Any:
    """Bar chart of per-param main-effect sensitivity (median slice log-range)."""
    import plotly.graph_objects as go

    params: tuple[Literal["primary", "spur", "stem"], ...] = ("primary", "spur", "stem")
    stats = {p: _log_range_sensitivity(rows, param=p, metric=metric) for p in params}
    med = np.array([stats[p]["median_log_range"] for p in params], dtype=np.float64)
    x_mult = np.exp(med)

    fig = go.Figure(
        data=[
            go.Bar(
                x=list(params),
                y=x_mult.tolist(),
                text=[f"{v:.2g}×" for v in x_mult.tolist()],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title=title,
        yaxis_title="median slice effect (multiplicative, exp(log-range))",
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


def load_grid_viz_rows_from_json(path: Path | str) -> list[GridVizRow]:
    rows: list[GridVizRow] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rows.append(GridVizRow(**json.loads(line)))
    return rows


def compute_dual_metric_ranks(
    rows: Sequence[GridVizRow],
    *,
    pos_metric: Metric = "err_pos_hold",
    force_metric: Metric = "err_force_hold",
) -> dict[str, np.ndarray]:
    pos = np.array([float(getattr(r, pos_metric)) for r in rows], dtype=np.float64)
    force = np.array([float(getattr(r, force_metric)) for r in rows], dtype=np.float64)
    rank_pos = average_ranks(pos)
    rank_force = average_ranks(force)
    rank_combined = 0.5 * (rank_pos + rank_force)
    return {
        "rank_pos": rank_pos,
        "rank_force": rank_force,
        "rank_combined": rank_combined,
        pos_metric: pos,
        force_metric: force,
    }


def make_3d_rank_scatter(
    *,
    rows: Sequence[GridVizRow],
    title: str,
    pos_metric: Metric = "err_pos_hold",
    force_metric: Metric = "err_force_hold",
) -> Any:
    import plotly.graph_objects as go

    a = _rows_to_arrays(rows)
    ranks = compute_dual_metric_ranks(rows, pos_metric=pos_metric, force_metric=force_metric)
    is_gt = a["gt_flag"]
    non = ~is_gt

    def _marker_size(rank_force: np.ndarray) -> np.ndarray:
        n = max(1, int(rank_force.size))
        return 11.0 - 6.0 * (rank_force - 1.0) / max(1.0, float(n - 1))

    def _scatter(mask: np.ndarray, *, name: str, symbol: str, base_size: float, line_width: int):
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            return None
        hover = []
        for i in idxs.tolist():
            r = rows[int(i)]
            hover.append(
                "<br>".join(
                    [
                        f"candidate={r.candidate_index}",
                        f"GT={r.gt_flag}",
                        f"primary={r.primary:.3g}",
                        f"spur={r.spur:.3g}",
                        f"stem={r.stem:.3g}",
                        f"dist_log_gt={r.dist_log_gt:.3g}",
                        f"pos_rank={ranks['rank_pos'][i]:.2f}",
                        f"force_rank={ranks['rank_force'][i]:.2f}",
                        f"combined_rank={ranks['rank_combined'][i]:.2f}",
                        f"{pos_metric}={float(getattr(r, pos_metric)):.6g}",
                        f"{force_metric}={float(getattr(r, force_metric)):.6g}",
                    ]
                )
            )
        sizes = _marker_size(ranks["rank_force"][mask]) if name == "candidates" else np.full(int(np.count_nonzero(mask)), base_size)
        return go.Scatter3d(
            x=np.log10(a["primary"][mask]),
            y=np.log10(a["spur"][mask]),
            z=np.log10(a["stem"][mask]),
            mode="markers",
            name=name,
            marker=dict(
                size=sizes,
                symbol=symbol,
                color=ranks["rank_combined"][mask],
                colorscale="Viridis_r",
                cmin=1.0,
                cmax=float(max(ranks["rank_combined"])),
                colorbar=dict(title="combined rank<br>(pos+force)/2"),
                line=dict(width=line_width, color="black"),
            ),
            text=hover,
            hoverinfo="text",
        )

    traces = []
    t_non = _scatter(non, name="candidates", symbol="circle", base_size=5.0, line_width=0)
    if t_non is not None:
        traces.append(t_non)
    t_gt = _scatter(is_gt, name="GT", symbol="diamond", base_size=9.0, line_width=2)
    if t_gt is not None:
        traces.append(t_gt)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="log10(primary)",
            yaxis_title="log10(spur)",
            zaxis_title="log10(stem)",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def write_rank_grid_from_rows_json(
    *,
    rows_json: Path | str,
    output_html: Path | str | None = None,
    pos_metric: Metric = "err_pos_hold",
    force_metric: Metric = "err_force_hold",
) -> Path:
    rows_path = Path(rows_json)
    rows = load_grid_viz_rows_from_json(rows_path)
    if not rows:
        raise ValueError(f"no rows found in {rows_path}")

    structure_idx = int(rows[0].structure_idx)
    out_path = (
        Path(output_html)
        if output_html is not None
        else rows_path.with_name(f"structure_{structure_idx:03d}_rank_pos_force_3d.html")
    )
    fig = make_3d_rank_scatter(
        rows=rows,
        title=(
            f"structure {structure_idx}: 3D stiffness grid "
            f"(ranks: {pos_metric} + {force_metric})"
        ),
        pos_metric=pos_metric,
        force_metric=force_metric,
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path


def _rows_to_arrays(rows: Sequence[GridVizRow]) -> dict[str, np.ndarray]:
    return {
        "primary": np.array([r.primary for r in rows], dtype=np.float64),
        "spur": np.array([r.spur for r in rows], dtype=np.float64),
        "stem": np.array([r.stem for r in rows], dtype=np.float64),
        "dist_log_gt": np.array([r.dist_log_gt for r in rows], dtype=np.float64),
        "gt_flag": np.array([bool(r.gt_flag) for r in rows], dtype=bool),
        "candidate_index": np.array([r.candidate_index for r in rows], dtype=np.int64),
    }


def make_3d_scatter(
    *,
    rows: Sequence[GridVizRow],
    metric: Metric,
    title: str,
) -> Any:
    import plotly.graph_objects as go

    a = _rows_to_arrays(rows)
    values = np.array([float(getattr(r, metric)) for r in rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    cmin = float(np.min(finite)) if finite.size else 0.0
    cmax = float(np.max(finite)) if finite.size else 1.0
    if not np.isfinite(cmin) or not np.isfinite(cmax):
        cmin, cmax = 0.0, 1.0
    if cmin == cmax:
        # Avoid a degenerate colorscale range (single value across rows).
        cmin -= 0.5
        cmax += 0.5

    # Split GT vs non-GT for clearer marker styling.
    is_gt = a["gt_flag"]
    non = ~is_gt

    def _scatter(
        mask: np.ndarray,
        *,
        name: str,
        symbol: str,
        size: int,
        line_width: int,
        show_scale: bool,
    ):
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            return None
        hover = []
        for i in idxs.tolist():
            r = rows[int(i)]
            hover.append(
                "<br>".join(
                    [
                        f"candidate={r.candidate_index}",
                        f"GT={r.gt_flag}",
                        f"primary={r.primary:.3g}",
                        f"spur={r.spur:.3g}",
                        f"stem={r.stem:.3g}",
                        f"dist_log_gt={r.dist_log_gt:.3g}",
                        f"{metric}={float(getattr(r, metric)):.6g}",
                    ]
                )
            )
        return go.Scatter3d(
            x=np.log10(a["primary"][mask]),
            y=np.log10(a["spur"][mask]),
            z=np.log10(a["stem"][mask]),
            mode="markers",
            name=name,
            marker=dict(
                size=size,
                symbol=symbol,
                color=values[mask],
                colorscale="Viridis",
                cmin=cmin,
                cmax=cmax,
                showscale=bool(show_scale),
                colorbar=dict(title=_metric_label(metric)) if show_scale else None,
                line=dict(width=line_width, color="black"),
            ),
            text=hover,
            hoverinfo="text",
        )

    traces = []
    t_non = _scatter(
        non,
        name="candidates",
        symbol="circle",
        size=4,
        line_width=0,
        show_scale=True,
    )
    if t_non is not None:
        traces.append(t_non)
    t_gt = _scatter(
        is_gt,
        name="GT",
        symbol="diamond",
        size=7,
        line_width=2,
        show_scale=False,
    )
    if t_gt is not None:
        traces.append(t_gt)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="log10(primary)",
            yaxis_title="log10(spur)",
            zaxis_title="log10(stem)",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def _binned_median(x: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return np.zeros(0), np.zeros(0)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    n_bins = max(1, int(n_bins))
    edges = np.linspace(0, x.size, n_bins + 1, dtype=int)
    xs: list[float] = []
    ys: list[float] = []
    for i in range(n_bins):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi - lo < 1:
            continue
        xs.append(float(np.median(x[lo:hi])))
        ys.append(float(np.median(y[lo:hi])))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def make_dist_vs_error(
    *,
    rows: Sequence[GridVizRow],
    metric: Metric,
    title: str,
    n_bins: int = 10,
) -> Any:
    import plotly.graph_objects as go

    dist = np.asarray([float(r.dist_log_gt) for r in rows], dtype=np.float64)
    values = np.asarray([float(getattr(r, metric)) for r in rows], dtype=np.float64)
    gt = np.asarray([bool(r.gt_flag) for r in rows], dtype=bool)

    mask = np.isfinite(dist) & np.isfinite(values)
    dist = dist[mask]
    values = values[mask]
    gt = gt[mask]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dist[~gt],
            y=values[~gt],
            mode="markers",
            name="candidates",
            marker=dict(size=6),
        )
    )
    if np.any(gt):
        fig.add_trace(
            go.Scatter(
                x=dist[gt],
                y=values[gt],
                mode="markers",
                name="GT",
                marker=dict(size=10, symbol="diamond", line=dict(width=2, color="black")),
            )
        )

    bx, by = _binned_median(dist, values, n_bins=int(n_bins))
    if bx.size:
        fig.add_trace(
            go.Scatter(
                x=bx,
                y=by,
                mode="lines+markers",
                name=f"binned median (n={int(n_bins)})",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="dist_log_gt = ||log(k)-log(k_gt)||_2",
        yaxis_title=_metric_label(metric),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def write_structure_bundle(
    *,
    output_dir: Path | str,
    structure_idx: int,
    rows: Sequence[GridVizRow],
    metrics: Sequence[Metric],
    n_bins: int = 10,
) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    # Save raw rows for downstream analysis.
    rows_path = out_dir / f"structure_{int(structure_idx):03d}_rows.json"
    rows_path.write_text(
        "\n".join([__import__("json").dumps(asdict(r), sort_keys=True) for r in rows]) + "\n"
    )
    written.append(rows_path)

    for metric in metrics:
        fig3d = make_3d_scatter(
            rows=rows,
            metric=metric,
            title=f"structure {int(structure_idx)}: 3D stiffness scatter ({metric})",
        )
        path3d = out_dir / f"structure_{int(structure_idx):03d}_{metric}_3d.html"
        fig3d.write_html(str(path3d), include_plotlyjs="cdn")
        written.append(path3d)

        fig2d = make_dist_vs_error(
            rows=rows,
            metric=metric,
            title=f"structure {int(structure_idx)}: dist_log_gt vs error ({metric})",
            n_bins=int(n_bins),
        )
        path2d = out_dir / f"structure_{int(structure_idx):03d}_{metric}_dist.html"
        fig2d.write_html(str(path2d), include_plotlyjs="cdn")
        written.append(path2d)

    # Pareto + sensitivity helpers (useful when grids look visually flat).
    try:
        pareto_metrics = PARETO_DEFAULT_METRICS
        fig_pareto = make_pareto_hold_projections(
            rows=rows,
            metrics=pareto_metrics,
            title=f"structure {int(structure_idx)}: Pareto front (hold metrics)",
        )
        path_pareto = out_dir / f"structure_{int(structure_idx):03d}_pareto_hold.html"
        fig_pareto.write_html(str(path_pareto), include_plotlyjs="cdn")
        written.append(path_pareto)
    except Exception:
        # Keep plotting best-effort: do not fail the whole run due to optional viz.
        pass

    for metric in ("err_force_hold", "err_torque_hold", "err_pos_hold"):
        if metric not in metrics:
            continue
        try:
            fig_sens = make_param_sensitivity_bars(
                rows=rows,
                metric=metric,  # type: ignore[arg-type]
                title=f"structure {int(structure_idx)}: sensitivity by param ({metric})",
            )
            path_sens = out_dir / f"structure_{int(structure_idx):03d}_sensitivity_{metric}.html"
            fig_sens.write_html(str(path_sens), include_plotlyjs="cdn")
            written.append(path_sens)
        except Exception:
            pass

    return written

