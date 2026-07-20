"""Visualization helpers for Young's-modulus CMA-ES reports."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from apple_pick_gym.batched_envs.batched_sysid_cmaes import generation_score_summary

# Optimizer coordinates are log10([E_primary, E_spur, E_stem]).
_PRIMARY_IDX = 0
_SPUR_IDX = 1
_STEM_IDX = 2


def generation_score_series_from_structure(
    structure: Mapping[str, Any],
) -> dict[str, list[Any]]:
    """Extract per-generation eligible mean/variance series from one structure report."""
    generation_index: list[int] = []
    eligible_mean: list[float | None] = []
    eligible_variance: list[float | None] = []
    eligible_std: list[float | None] = []
    best_eligible: list[float | None] = []
    penalized_mean: list[float | None] = []
    for generation in structure.get("generations", []):
        summary = generation.get("score_summary")
        if summary is None:
            summary = generation_score_summary(
                generation.get("penalty_metadata", []),
                penalized_fitness=generation.get("penalized_fitness", []),
            )
        generation_index.append(int(generation["generation_index"]))
        eligible_mean.append(summary.get("eligible_mean"))
        eligible_variance.append(summary.get("eligible_variance"))
        eligible_std.append(summary.get("eligible_std"))
        best_eligible.append(summary.get("best_eligible"))
        penalized_mean.append(summary.get("penalized_mean"))
    return {
        "generation_index": generation_index,
        "eligible_mean": eligible_mean,
        "eligible_variance": eligible_variance,
        "eligible_std": eligible_std,
        "best_eligible": best_eligible,
        "penalized_mean": penalized_mean,
    }


def generation_distribution_series_from_structure(
    structure: Mapping[str, Any],
) -> dict[str, list[Any]]:
    """Extract post-tell mean / covariance diagnostics over generations.

    Uses ``post_tell_distribution`` so each generation reflects the search
    distribution after that generation's fitness feedback.
    """
    gt = structure.get("gt")
    gt_log10: list[float] | None = None
    if isinstance(gt, Mapping) and gt.get("log10_e") is not None:
        gt_log10 = [float(v) for v in gt["log10_e"]]

    generation_index: list[int] = []
    mean_primary: list[float | None] = []
    mean_spur: list[float | None] = []
    mean_stem: list[float | None] = []
    sigma: list[float | None] = []
    phenotype_std_primary: list[float | None] = []
    phenotype_std_spur: list[float | None] = []
    phenotype_std_stem: list[float | None] = []
    effective_cov_trace: list[float | None] = []
    effective_cov_log10_det: list[float | None] = []
    distance_to_gt: list[float | None] = []
    distance_to_gt_spur_stem: list[float | None] = []

    for generation in structure.get("generations", []):
        generation_index.append(int(generation["generation_index"]))
        dist = generation.get("post_tell_distribution")
        if not isinstance(dist, Mapping):
            mean_primary.append(None)
            mean_spur.append(None)
            mean_stem.append(None)
            sigma.append(None)
            phenotype_std_primary.append(None)
            phenotype_std_spur.append(None)
            phenotype_std_stem.append(None)
            effective_cov_trace.append(None)
            effective_cov_log10_det.append(None)
            distance_to_gt.append(None)
            distance_to_gt_spur_stem.append(None)
            continue

        mean = dist.get("mean_log10")
        if mean is None or len(mean) < 3:
            mean_vals: list[float | None] = [None, None, None]
        else:
            mean_vals = [float(mean[_PRIMARY_IDX]), float(mean[_SPUR_IDX]), float(mean[_STEM_IDX])]
        mean_primary.append(mean_vals[0])
        mean_spur.append(mean_vals[1])
        mean_stem.append(mean_vals[2])
        sigma.append(None if dist.get("sigma") is None else float(dist["sigma"]))

        cov = dist.get("covariance")
        if not isinstance(cov, Mapping):
            phenotype_std_primary.append(None)
            phenotype_std_spur.append(None)
            phenotype_std_stem.append(None)
            effective_cov_trace.append(None)
            effective_cov_log10_det.append(None)
        else:
            pheno = cov.get("phenotype_std")
            if pheno is None or len(pheno) < 3:
                phenotype_std_primary.append(None)
                phenotype_std_spur.append(None)
                phenotype_std_stem.append(None)
            else:
                phenotype_std_primary.append(float(pheno[_PRIMARY_IDX]))
                phenotype_std_spur.append(float(pheno[_SPUR_IDX]))
                phenotype_std_stem.append(float(pheno[_STEM_IDX]))
            effective = cov.get("effective_unbounded_covariance")
            effective_cov_trace.append(_matrix_trace(effective))
            effective_cov_log10_det.append(_matrix_log10_det(effective))

        if gt_log10 is None or any(v is None for v in mean_vals):
            distance_to_gt.append(None)
            distance_to_gt_spur_stem.append(None)
        else:
            assert mean_vals[0] is not None and mean_vals[1] is not None and mean_vals[2] is not None
            distance_to_gt.append(
                math.sqrt(
                    sum(
                        (float(mean_vals[i]) - float(gt_log10[i])) ** 2
                        for i in range(3)
                    )
                )
            )
            distance_to_gt_spur_stem.append(
                math.sqrt(
                    (float(mean_vals[1]) - float(gt_log10[1])) ** 2
                    + (float(mean_vals[2]) - float(gt_log10[2])) ** 2
                )
            )

    return {
        "generation_index": generation_index,
        "mean_primary": mean_primary,
        "mean_spur": mean_spur,
        "mean_stem": mean_stem,
        "sigma": sigma,
        "phenotype_std_primary": phenotype_std_primary,
        "phenotype_std_spur": phenotype_std_spur,
        "phenotype_std_stem": phenotype_std_stem,
        "effective_cov_trace": effective_cov_trace,
        "effective_cov_log10_det": effective_cov_log10_det,
        "distance_to_gt": distance_to_gt,
        "distance_to_gt_spur_stem": distance_to_gt_spur_stem,
        "gt_log10": gt_log10,
    }


def build_cma_evaluated_points(structure: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build log10-space evaluated points for one structure, including GT/final mean."""
    points: list[dict[str, Any]] = []
    structure_idx = int(structure.get("structure_idx", -1))
    for generation in structure.get("generations", []):
        gen_idx = int(generation["generation_index"])
        samples = list(generation.get("ask_samples_log10", []))
        metadata = list(generation.get("penalty_metadata", []))
        for sample, meta in zip(samples, metadata, strict=False):
            penalized = bool(meta.get("penalized"))
            raw = meta.get("raw_aggregate_sinkhorn")
            score = None if raw is None else float(raw)
            points.append(
                {
                    "structure_idx": structure_idx,
                    "generation_index": gen_idx,
                    "candidate_index": int(meta.get("candidate_index", -1)),
                    "log10_e": [float(v) for v in sample],
                    "score": score,
                    "penalized": penalized,
                    "kind": "disqualified" if penalized else "candidate",
                    "disqualification_reason": meta.get("disqualification_reason"),
                }
            )
    gt = structure.get("gt")
    if isinstance(gt, Mapping) and gt.get("log10_e") is not None:
        points.append(
            {
                "structure_idx": structure_idx,
                "generation_index": None,
                "candidate_index": None,
                "log10_e": [float(v) for v in gt["log10_e"]],
                "score": None,
                "penalized": False,
                "kind": "gt",
                "disqualification_reason": None,
            }
        )
    final_mean = structure.get("final_mean")
    if isinstance(final_mean, Mapping) and final_mean.get("log10_e") is not None:
        points.append(
            {
                "structure_idx": structure_idx,
                "generation_index": None,
                "candidate_index": None,
                "log10_e": [float(v) for v in final_mean["log10_e"]],
                "score": (
                    None
                    if final_mean.get("aggregate_sinkhorn") is None
                    else float(final_mean["aggregate_sinkhorn"])
                ),
                "penalized": False,
                "kind": "final_mean",
                "disqualification_reason": None,
            }
        )
    best = structure.get("best_sample")
    if isinstance(best, Mapping) and best.get("log10_e") is not None:
        points.append(
            {
                "structure_idx": structure_idx,
                "generation_index": None,
                "candidate_index": None,
                "log10_e": [float(v) for v in best["log10_e"]],
                "score": (
                    None if best.get("fitness") is None else float(best["fitness"])
                ),
                "penalized": False,
                "kind": "best_sample",
                "disqualification_reason": None,
            }
        )
    return points


def write_generation_score_figures(
    report: Mapping[str, Any] | Path | str,
    output_html: Path | str,
) -> Path:
    """Write mean/variance-over-generation Plotly HTML for all structures."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    payload = _load_report(report)
    structures = _structure_items(payload)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Eligible Sinkhorn mean (solid) and best (dashed)",
            "Eligible Sinkhorn variance (sample, ddof=1)",
        ),
        vertical_spacing=0.12,
    )
    for structure_idx, structure in structures:
        series = generation_score_series_from_structure(structure)
        xs = series["generation_index"]
        means = series["eligible_mean"]
        variances = series["eligible_variance"]
        bests = series["best_eligible"]
        name = f"structure {structure_idx}"
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=means,
                mode="lines+markers",
                name=f"{name} mean",
                legendgroup=name,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=bests,
                mode="lines",
                name=f"{name} best",
                legendgroup=name,
                line=dict(dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=variances,
                mode="lines+markers",
                name=f"{name} variance",
                legendgroup=name,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    timing = payload.get("timing") if isinstance(payload.get("timing"), Mapping) else {}
    title = "CMA-ES generation score mean / variance"
    if timing.get("fit_seconds") is not None:
        title += f" (fit {float(timing['fit_seconds']):.1f}s"
        if timing.get("command_seconds") is not None:
            title += f", command {float(timing['command_seconds']):.1f}s"
        title += ")"
    fig.update_layout(title=title, height=700, template="plotly_white")
    fig.update_xaxes(title_text="generation", row=2, col=1)
    fig.update_yaxes(title_text="Sinkhorn mean", row=1, col=1)
    fig.update_yaxes(title_text="Sinkhorn variance", row=2, col=1)
    out = Path(output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    return out


def write_structure_generation_score_figure(
    structure: Mapping[str, Any],
    output_html: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """Write generation vs eligible Sinkhorn mean/variance for one structure."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    structure_idx = int(structure.get("structure_idx", -1))
    series = generation_score_series_from_structure(structure)
    xs = series["generation_index"]
    means = series["eligible_mean"]
    variances = series["eligible_variance"]
    stds = series["eligible_std"]
    bests = series["best_eligible"]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Eligible Sinkhorn mean (±1 std) and best",
            "Eligible Sinkhorn variance",
        ),
        vertical_spacing=0.12,
    )
    band_x: list[int] = []
    band_upper: list[float] = []
    band_lower: list[float] = []
    for x, mean, std in zip(xs, means, stds, strict=True):
        if mean is None or std is None:
            continue
        band_x.append(int(x))
        band_upper.append(float(mean) + float(std))
        band_lower.append(float(mean) - float(std))
    if band_x:
        fig.add_trace(
            go.Scatter(
                x=band_x + band_x[::-1],
                y=band_upper + band_lower[::-1],
                fill="toself",
                fillcolor="rgba(31, 119, 180, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="mean ±1 std",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=means,
            mode="lines+markers",
            name="eligible mean",
            line=dict(color="rgb(31, 119, 180)"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=bests,
            mode="lines+markers",
            name="best eligible",
            line=dict(color="rgb(255, 127, 14)", dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=variances,
            mode="lines+markers",
            name="eligible variance",
            line=dict(color="rgb(44, 160, 44)"),
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=title or f"structure {structure_idx}: generation Sinkhorn mean / variance",
        height=650,
        template="plotly_white",
    )
    fig.update_xaxes(title_text="generation", row=2, col=1)
    fig.update_yaxes(title_text="Sinkhorn", row=1, col=1)
    fig.update_yaxes(title_text="variance", row=2, col=1)
    out = Path(output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    return out


def write_spur_stem_sinkhorn_scatter_3d(
    structure: Mapping[str, Any],
    output_html: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """Write spur/stem log10(E) vs Sinkhorn 3D scatter with best and GT markers.

    Primary is omitted from the axes because it typically barely moves under the
    CMA bounds used here. Coordinates are still ``log10([primary, spur, stem])``.
    """
    import plotly.graph_objects as go

    points = build_cma_evaluated_points(structure)
    structure_idx = int(structure.get("structure_idx", -1))
    scored = [
        p
        for p in points
        if p["kind"] in {"candidate", "disqualified", "best_sample", "final_mean"}
        and p["score"] is not None
        and math.isfinite(float(p["score"]))
    ]
    score_values = [float(p["score"]) for p in scored]
    z_floor = float(min(score_values)) if score_values else 0.0
    z_ceil = float(max(score_values)) if score_values else 1.0
    if z_floor == z_ceil:
        z_floor -= 0.5
        z_ceil += 0.5

    def _xy(point: Mapping[str, Any]) -> tuple[float, float]:
        log10_e = point["log10_e"]
        return float(log10_e[_SPUR_IDX]), float(log10_e[_STEM_IDX])

    def _hover(point: Mapping[str, Any]) -> str:
        score = "n/a" if point["score"] is None else f"{float(point['score']):.6g}"
        log10_e = point["log10_e"]
        return "<br>".join(
            [
                f"kind={point['kind']}",
                f"generation={point['generation_index']}",
                f"candidate={point['candidate_index']}",
                (
                    f"log10(primary,spur,stem)=("
                    f"{log10_e[_PRIMARY_IDX]:.4g}, "
                    f"{log10_e[_SPUR_IDX]:.4g}, "
                    f"{log10_e[_STEM_IDX]:.4g})"
                ),
                f"score={score}",
                f"disqualification={point['disqualification_reason']}",
            ]
        )

    traces: list[Any] = []

    candidates = [p for p in points if p["kind"] == "candidate" and p["score"] is not None]
    if candidates:
        xs, ys, zs = [], [], []
        hover = []
        gens = []
        for p in candidates:
            x, y = _xy(p)
            xs.append(x)
            ys.append(y)
            zs.append(float(p["score"]))
            hover.append(_hover(p))
            gens.append(
                -1 if p["generation_index"] is None else int(p["generation_index"])
            )
        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                name="candidates",
                marker=dict(
                    size=4,
                    symbol="circle",
                    color=gens,
                    colorscale="Viridis",
                    colorbar=dict(title="generation"),
                    showscale=True,
                    line=dict(width=0, color="black"),
                ),
                text=hover,
                hoverinfo="text",
            )
        )

    disqualified = [
        p for p in points if p["kind"] == "disqualified" and p["score"] is not None
    ]
    if disqualified:
        xs, ys, zs, hover = [], [], [], []
        for p in disqualified:
            x, y = _xy(p)
            xs.append(x)
            ys.append(y)
            zs.append(float(p["score"]))
            hover.append(_hover(p))
        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                name="disqualified",
                marker=dict(size=4, symbol="x", color="gray"),
                text=hover,
                hoverinfo="text",
            )
        )

    for kind, name, symbol, size, color in (
        ("best_sample", "best sample", "square", 8.0, "orange"),
        ("final_mean", "final mean", "diamond-open", 8.0, "red"),
    ):
        selected = [p for p in points if p["kind"] == kind and p["score"] is not None]
        if not selected:
            continue
        xs, ys, zs, hover = [], [], [], []
        for p in selected:
            x, y = _xy(p)
            xs.append(x)
            ys.append(y)
            zs.append(float(p["score"]))
            hover.append(_hover(p))
        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                name=name,
                marker=dict(
                    size=size,
                    symbol=symbol,
                    color=color,
                    line=dict(width=2, color="black"),
                ),
                text=hover,
                hoverinfo="text",
            )
        )

    gt_points = [p for p in points if p["kind"] == "gt"]
    for p in gt_points:
        x, y = _xy(p)
        # GT Sinkhorn is not evaluated in the CMA loop; mark its (spur, stem)
        # location with a vertical line through the observed score range.
        traces.append(
            go.Scatter3d(
                x=[x, x],
                y=[y, y],
                z=[z_floor, z_ceil],
                mode="lines",
                name="GT (spur/stem)",
                line=dict(color="black", width=6),
                hoverinfo="skip",
                showlegend=True,
            )
        )
        traces.append(
            go.Scatter3d(
                x=[x],
                y=[y],
                z=[z_floor],
                mode="markers",
                name="GT",
                marker=dict(
                    size=10,
                    symbol="diamond",
                    color="black",
                    line=dict(width=2, color="white"),
                ),
                text=[_hover(p) + "<br>score=n/a (parameter location only)"],
                hoverinfo="text",
                showlegend=False,
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title
        or (
            f"structure {structure_idx}: log10(spur), log10(stem), Sinkhorn "
            "(primary omitted)"
        ),
        scene=dict(
            xaxis_title="log10(spur)",
            yaxis_title="log10(stem)",
            zaxis_title="Sinkhorn",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        template="plotly_white",
    )
    out = Path(output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    return out


# Back-compat alias for older call sites / notebooks.
write_log10_score_scatter_3d = write_spur_stem_sinkhorn_scatter_3d


def write_structure_optimizer_diagnostics_figure(
    structure: Mapping[str, Any],
    output_html: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """Plot mean motion and covariance shrinkage diagnostics for one structure.

    Working optimization typically shows: mean moving toward GT (distance down),
    and sigma / phenotype std / effective covariance volume shrinking.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    structure_idx = int(structure.get("structure_idx", -1))
    series = generation_distribution_series_from_structure(structure)
    xs = series["generation_index"]
    gt_log10 = series["gt_log10"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Mean log10(E) vs generation (GT dashed)",
            "Mean path in spur/stem plane",
            "||mean − GT|| in log10 space",
            "Covariance scale (sigma, phenotype std, trace)",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
    )

    for name, key, color in (
        ("primary", "mean_primary", "rgb(127,127,127)"),
        ("spur", "mean_spur", "rgb(31,119,180)"),
        ("stem", "mean_stem", "rgb(255,127,14)"),
    ):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=series[key],
                mode="lines+markers",
                name=f"mean {name}",
                line=dict(color=color),
                legendgroup=name,
            ),
            row=1,
            col=1,
        )
        if gt_log10 is not None:
            idx = {"primary": 0, "spur": 1, "stem": 2}[name]
            fig.add_hline(
                y=float(gt_log10[idx]),
                line=dict(color=color, dash="dash", width=1),
                row=1,
                col=1,
            )

    spur_means = [v for v in series["mean_spur"] if v is not None]
    stem_means = [v for v in series["mean_stem"] if v is not None]
    path_gens = [
        g
        for g, spur, stem in zip(
            xs, series["mean_spur"], series["mean_stem"], strict=True
        )
        if spur is not None and stem is not None
    ]
    if spur_means and stem_means:
        fig.add_trace(
            go.Scatter(
                x=spur_means,
                y=stem_means,
                mode="lines+markers+text",
                name="mean path",
                text=[str(g) for g in path_gens],
                textposition="top center",
                marker=dict(
                    size=8,
                    color=path_gens,
                    colorscale="Viridis",
                    showscale=False,
                ),
                line=dict(color="rgb(31,119,180)"),
            ),
            row=1,
            col=2,
        )
    if gt_log10 is not None:
        fig.add_trace(
            go.Scatter(
                x=[float(gt_log10[_SPUR_IDX])],
                y=[float(gt_log10[_STEM_IDX])],
                mode="markers",
                name="GT",
                marker=dict(size=12, symbol="diamond", color="black"),
            ),
            row=1,
            col=2,
        )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=series["distance_to_gt"],
            mode="lines+markers",
            name="||mean-GT|| (3D)",
            line=dict(color="rgb(148,103,189)"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=series["distance_to_gt_spur_stem"],
            mode="lines+markers",
            name="||mean-GT|| (spur/stem)",
            line=dict(color="rgb(214,39,40)"),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=series["sigma"],
            mode="lines+markers",
            name="sigma",
            line=dict(color="rgb(44,160,44)"),
        ),
        row=2,
        col=2,
    )
    for name, key, color in (
        ("pheno std spur", "phenotype_std_spur", "rgb(31,119,180)"),
        ("pheno std stem", "phenotype_std_stem", "rgb(255,127,14)"),
    ):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=series[key],
                mode="lines+markers",
                name=name,
                line=dict(color=color, dash="dot"),
            ),
            row=2,
            col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=series["effective_cov_trace"],
            mode="lines+markers",
            name="tr(effective cov)",
            line=dict(color="rgb(140,86,75)", dash="dash"),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=title
        or f"structure {structure_idx}: CMA mean / covariance diagnostics",
        height=800,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.18),
    )
    fig.update_xaxes(title_text="generation", row=1, col=1)
    fig.update_xaxes(title_text="log10(spur)", row=1, col=2)
    fig.update_xaxes(title_text="generation", row=2, col=1)
    fig.update_xaxes(title_text="generation", row=2, col=2)
    fig.update_yaxes(title_text="mean log10(E)", row=1, col=1)
    fig.update_yaxes(title_text="log10(stem)", row=1, col=2)
    fig.update_yaxes(title_text="distance", row=2, col=1)
    fig.update_yaxes(title_text="scale", row=2, col=2, type="log")

    out = Path(output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    return out


def write_cmaes_visualization_bundle(
    report: Mapping[str, Any] | Path | str,
    output_dir: Path | str,
) -> list[Path]:
    """Write generation score, optimizer diagnostics, and spur/stem Sinkhorn figures."""
    payload = _load_report(report)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = [
        write_generation_score_figures(
            payload,
            out_dir / "generation_score_mean_variance.html",
        )
    ]
    for structure_idx, structure in _structure_items(payload):
        written.append(
            write_structure_generation_score_figure(
                structure,
                out_dir / f"structure_{int(structure_idx):03d}_generation_scores.html",
            )
        )
        written.append(
            write_structure_optimizer_diagnostics_figure(
                structure,
                out_dir
                / f"structure_{int(structure_idx):03d}_optimizer_diagnostics.html",
            )
        )
        written.append(
            write_spur_stem_sinkhorn_scatter_3d(
                structure,
                out_dir
                / f"structure_{int(structure_idx):03d}_spur_stem_sinkhorn_3d.html",
            )
        )
    return written


def _matrix_trace(matrix: Any) -> float | None:
    if not isinstance(matrix, Sequence) or not matrix:
        return None
    total = 0.0
    for i, row in enumerate(matrix):
        if not isinstance(row, Sequence) or i >= len(row):
            return None
        total += float(row[i])
    return total


def _matrix_log10_det(matrix: Any) -> float | None:
    if not isinstance(matrix, Sequence) or len(matrix) != 3:
        return None
    try:
        a = [[float(matrix[i][j]) for j in range(3)] for i in range(3)]
    except (TypeError, ValueError, IndexError):
        return None
    det = (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )
    if det <= 0.0 or not math.isfinite(det):
        return None
    return math.log10(det)


def _load_report(report: Mapping[str, Any] | Path | str) -> Mapping[str, Any]:
    if isinstance(report, Mapping):
        return report
    path = Path(report)
    return json.loads(path.read_text(encoding="utf-8"))


def _structure_items(
    report: Mapping[str, Any],
) -> list[tuple[int, Mapping[str, Any]]]:
    structures = report.get("structures", {})
    if isinstance(structures, Mapping):
        items = [(int(idx), value) for idx, value in structures.items()]
    elif isinstance(structures, Sequence):
        items = [(int(value.get("structure_idx", i)), value) for i, value in enumerate(structures)]
    else:
        items = []
    return sorted(items, key=lambda item: item[0])
