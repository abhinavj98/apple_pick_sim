"""Ranking validation helpers for Sinkhorn vs hold MSE grid diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from apple_pick_sim.system_id.wasserstein import WassersteinCandidateResult


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    n = int(values.size)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman_r(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation without SciPy."""
    x_arr = np.asarray(list(x), dtype=np.float64).reshape(-1)
    y_arr = np.asarray(list(y), dtype=np.float64).reshape(-1)
    if x_arr.size != y_arr.size:
        raise ValueError("spearman_r requires x and y with the same length")
    if int(x_arr.size) < 2:
        return 0.0
    rx = _average_ranks(x_arr)
    ry = _average_ranks(y_arr)
    rx = rx - float(np.mean(rx))
    ry = ry - float(np.mean(ry))
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(rx, ry) / denom)


@dataclass(frozen=True)
class SinkhornGtPreference:
    """Whether GT minimizes aggregate Sinkhorn among eligible candidates."""

    gt_candidate_index: int | None
    gt_rank: int | None
    best_candidate_index: int | None
    best_is_gt: bool | None
    gt_disqualified: bool
    n_disqualified: int
    n_candidates: int


@dataclass(frozen=True)
class SinkhornMseSpearman:
    """Spearman correlation between Sinkhorn and hold MSE scalar errors."""

    metric: str
    spearman: float


def sinkhorn_gt_preference(
    *,
    results: Sequence[WassersteinCandidateResult],
    gt_candidate_index: int,
    disqualified: Sequence[bool],
) -> SinkhornGtPreference:
    """Rank candidates by aggregate Sinkhorn (lower is better), excluding disqualified."""
    if len(results) != len(disqualified):
        raise ValueError("results and disqualified must have the same length")
    if not results:
        raise ValueError("results must be non-empty")

    eligible = [
        (int(result.candidate_index), float(result.aggregate_sinkhorn))
        for result, bad in zip(results, disqualified, strict=True)
        if not bool(bad) and np.isfinite(float(result.aggregate_sinkhorn))
    ]
    n_disqualified = sum(1 for bad in disqualified if bool(bad))

    gt_pos = next(
        (
            i
            for i, result in enumerate(results)
            if int(result.candidate_index) == int(gt_candidate_index)
        ),
        None,
    )
    if gt_pos is None:
        raise ValueError(
            f"gt_candidate_index={gt_candidate_index} not found in results"
        )

    best_idx = None
    best_is_gt = None
    gt_rank = None
    gt_disqualified = bool(disqualified[gt_pos])
    if eligible:
        eligible_sorted = sorted(eligible, key=lambda item: (item[1], item[0]))
        best_idx = int(eligible_sorted[0][0])
        best_is_gt = None if gt_disqualified else bool(best_idx == int(gt_candidate_index))
        rank_by_index = {
            cand_idx: rank
            for rank, (cand_idx, _) in enumerate(eligible_sorted, start=1)
        }
        gt_rank = rank_by_index.get(int(gt_candidate_index))

    return SinkhornGtPreference(
        gt_candidate_index=int(gt_candidate_index),
        gt_rank=gt_rank,
        best_candidate_index=best_idx,
        best_is_gt=best_is_gt,
        gt_disqualified=gt_disqualified,
        n_disqualified=int(n_disqualified),
        n_candidates=len(results),
    )


def sinkhorn_mse_spearman(
    *,
    sinkhorn_values: Sequence[float],
    mse_values: Sequence[float],
    metric: str,
    disqualified: Sequence[bool] | None = None,
) -> SinkhornMseSpearman:
    """Spearman between Sinkhorn loss and a hold MSE scalar (higher MSE, higher rank)."""
    sink = np.asarray(list(sinkhorn_values), dtype=np.float64).reshape(-1)
    mse = np.asarray(list(mse_values), dtype=np.float64).reshape(-1)
    if sink.size != mse.size:
        raise ValueError("sinkhorn_values and mse_values must have the same length")
    mask = np.isfinite(sink) & np.isfinite(mse)
    if disqualified is not None:
        disq = np.asarray(list(disqualified), dtype=bool).reshape(-1)
        if disq.size != sink.size:
            raise ValueError("disqualified must have the same length as sinkhorn_values")
        mask = mask & (~disq)
    if int(np.count_nonzero(mask)) < 2:
        return SinkhornMseSpearman(metric=str(metric), spearman=0.0)
    corr = spearman_r(sink[mask].tolist(), mse[mask].tolist())
    return SinkhornMseSpearman(metric=str(metric), spearman=float(corr))
