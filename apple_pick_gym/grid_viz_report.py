from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np

from apple_pick_gym.grid_viz_metrics import spearman_r
from apple_pick_gym.grid_viz_table import GridVizRow


MetricName = Literal[
    "err_pos_all",
    "err_force_all",
    "err_torque_all",
    "err_woody_pos_all",
    "err_pos_hold",
    "err_force_hold",
    "err_torque_hold",
    "err_woody_pos_hold",
]


@dataclass(frozen=True)
class MetricSummary:
    metric: str
    gt_rank: int | None
    gt_value: float | None
    best_value: float | None
    best_candidate_index: int | None
    best_is_gt: bool | None
    spearman_dist_vs_err: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructureReport:
    structure_idx: int
    n_candidates: int
    summaries: list[MetricSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "structure_idx": int(self.structure_idx),
            "n_candidates": int(self.n_candidates),
            "summaries": [s.to_dict() for s in self.summaries],
        }


@dataclass(frozen=True)
class AcrossStructuresReport:
    metrics: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"metrics": dict(self.metrics)}


def _finite(vals: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(vals), dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def summarize_structure(
    *,
    structure_idx: int,
    rows: Sequence[GridVizRow],
    metrics: Sequence[MetricName],
) -> StructureReport:
    if not rows:
        raise ValueError("rows must be non-empty")

    summaries: list[MetricSummary] = []
    dist = np.asarray([float(r.dist_log_gt) for r in rows], dtype=np.float64)
    gt_indices = [i for i, r in enumerate(rows) if bool(r.gt_flag)]

    for metric in metrics:
        values = np.asarray([float(getattr(r, metric)) for r in rows], dtype=np.float64)
        best_idx = int(np.nanargmin(values)) if np.any(np.isfinite(values)) else None
        best_val = float(values[best_idx]) if best_idx is not None else None
        best_cand = int(rows[best_idx].candidate_index) if best_idx is not None else None

        gt_rank = None
        gt_val = None
        best_is_gt = None
        if gt_indices:
            # If multiple GT rows exist (shouldn’t), choose the first.
            gt_i = int(gt_indices[0])
            gt_val = float(values[gt_i]) if np.isfinite(values[gt_i]) else None
            # Rank among finite values (1 = best).
            finite_order = np.argsort(values, kind="mergesort")
            # Compute rank position of gt_i among finite values.
            if np.isfinite(values[gt_i]):
                gt_rank = 1 + int(np.where(finite_order == gt_i)[0][0])
            if best_idx is not None:
                best_is_gt = bool(gt_i == best_idx)

        # Correlation over finite pairs.
        finite_mask = np.isfinite(values) & np.isfinite(dist)
        corr = spearman_r(dist[finite_mask], values[finite_mask]) if int(np.count_nonzero(finite_mask)) >= 2 else 0.0

        summaries.append(
            MetricSummary(
                metric=str(metric),
                gt_rank=gt_rank,
                gt_value=gt_val,
                best_value=best_val,
                best_candidate_index=best_cand,
                best_is_gt=best_is_gt,
                spearman_dist_vs_err=float(corr),
            )
        )

    return StructureReport(
        structure_idx=int(structure_idx),
        n_candidates=len(rows),
        summaries=summaries,
    )


def summarize_across_structures(
    *,
    structure_reports: Sequence[StructureReport],
) -> AcrossStructuresReport:
    if not structure_reports:
        return AcrossStructuresReport(metrics={})

    # Collect per-metric arrays.
    by_metric: dict[str, dict[str, list[float]]] = {}
    for rep in structure_reports:
        for s in rep.summaries:
            m = str(s.metric)
            entry = by_metric.setdefault(
                m,
                {"gt_rank": [], "best_is_gt": [], "spearman": []},
            )
            if s.gt_rank is not None:
                entry["gt_rank"].append(float(s.gt_rank))
            if s.best_is_gt is not None:
                entry["best_is_gt"].append(1.0 if s.best_is_gt else 0.0)
            entry["spearman"].append(float(s.spearman_dist_vs_err))

    out: dict[str, dict[str, Any]] = {}
    for metric, vals in by_metric.items():
        spearman = _finite(vals["spearman"])
        best_is_gt = _finite(vals["best_is_gt"])
        gt_rank = _finite(vals["gt_rank"])
        out[metric] = {
            "n_structures": len(structure_reports),
            "p_best_is_gt": float(np.mean(best_is_gt)) if best_is_gt.size else None,
            "gt_rank_mean": float(np.mean(gt_rank)) if gt_rank.size else None,
            "spearman_median": float(np.median(spearman)) if spearman.size else None,
            "spearman_iqr": (
                float(np.percentile(spearman, 75) - np.percentile(spearman, 25))
                if spearman.size
                else None
            ),
        }

    return AcrossStructuresReport(metrics=out)

