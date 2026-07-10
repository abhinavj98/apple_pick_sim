"""Sinkhorn (entropic Wasserstein) helpers for sys-ID transition-bag scoring."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from apple_pick_sim.system_id.mmd import NormalizationStats, apply_normalization, fit_gt_normalization
from apple_pick_sim.system_id.mmd_features import combine_transition_features

SINKHORN_P = 2
SINKHORN_BLUR = 1.0
LOW_SAMPLE_MIN_TRANSITIONS = 8


@dataclass(frozen=True)
class WassersteinDirectionContext:
    """GT-normalized transition bag for one excitation direction."""

    gt_norm: np.ndarray
    stats: NormalizationStats


@dataclass(frozen=True)
class WassersteinCandidateResult:
    """Sinkhorn loss for one stiffness-grid candidate."""

    candidate_index: int
    stiffnesses: dict[str, float]
    aggregate_sinkhorn: float
    per_direction_sinkhorn: dict[int, float]
    per_direction_n_transitions: dict[int, int]
    low_sample_directions: tuple[int, ...]
    missing_directions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.per_direction_sinkhorn:
            raise ValueError("Wasserstein candidate result must include at least one direction")


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return str(device)
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for Sinkhorn scoring; sync with --extra vic"
        ) from exc
    return "cuda" if torch.cuda.is_available() else "cpu"


def _samples_loss(*, device: str):
    try:
        from geomloss import SamplesLoss
    except ImportError as exc:
        raise ImportError(
            "geomloss is required for Sinkhorn scoring; sync with --extra gym"
        ) from exc
    return SamplesLoss("sinkhorn", p=SINKHORN_P, blur=SINKHORN_BLUR).to(device)


def sinkhorn_distance(
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    *,
    device: str | None = None,
) -> float:
    """Compute Sinkhorn divergence between two normalized transition bags."""
    dev = _resolve_device(device)
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for Sinkhorn scoring; sync with --extra vic"
        ) from exc
    loss_fn = _samples_loss(device=dev)
    x_t = torch.as_tensor(np.asarray(x_norm, dtype=np.float64), device=dev)
    y_t = torch.as_tensor(np.asarray(y_norm, dtype=np.float64), device=dev)
    if x_t.ndim != 2 or y_t.ndim != 2:
        raise ValueError("sinkhorn_distance expects 2D feature matrices")
    if x_t.shape[0] == 0 or y_t.shape[0] == 0:
        raise ValueError("sinkhorn_distance requires at least one sample per bag")
    if x_t.shape[1] != y_t.shape[1]:
        raise ValueError(
            f"feature dimension mismatch: x={x_t.shape[1]} y={y_t.shape[1]}"
        )
    with torch.no_grad():
        value = loss_fn(x_t, y_t)
    return float(value.detach().cpu().item())


def prepare_gt_wasserstein_context(
    recorded_episodes: list[dict],
) -> dict[int, WassersteinDirectionContext]:
    """Fit per-direction GT normalization from recorded transition bags."""
    gt_by_direction = combine_transition_features(recorded_episodes)
    if not gt_by_direction:
        raise ValueError("No valid hold-only GT transition features were found.")

    context: dict[int, WassersteinDirectionContext] = {}
    for direction, gt_features in gt_by_direction.items():
        stats = fit_gt_normalization(gt_features)
        gt_norm = apply_normalization(gt_features, stats)
        context[int(direction)] = WassersteinDirectionContext(gt_norm=gt_norm, stats=stats)
    return context


def score_candidate_wasserstein(
    *,
    candidate_index: int,
    stiffnesses: dict[str, float],
    gt_context: dict[int, WassersteinDirectionContext],
    replay_observations: list[dict],
    device: str | None = None,
) -> WassersteinCandidateResult:
    """Score one replayed candidate against precomputed GT Wasserstein context."""
    candidate_by_direction = combine_transition_features(replay_observations)
    per_direction: dict[int, float] = {}
    per_direction_n: dict[int, int] = {}
    low_sample: list[int] = []
    missing_directions: list[int] = []

    missing = sorted(set(gt_context) - set(candidate_by_direction))
    if missing:
        missing_directions = [int(d) for d in missing]
        warnings.warn(
            "candidate missing GT directions: "
            + ", ".join(str(direction) for direction in missing),
            stacklevel=2,
        )

    for direction, context in gt_context.items():
        candidate_features = candidate_by_direction.get(int(direction))
        if candidate_features is None:
            continue
        per_direction_n[int(direction)] = int(candidate_features.shape[0])
        if candidate_features.shape[1] != context.gt_norm.shape[1]:
            raise ValueError(
                "Wasserstein feature dimension mismatch for direction "
                f"{direction}: gt={context.gt_norm.shape[1]} "
                f"candidate={candidate_features.shape[1]}"
            )
        if int(candidate_features.shape[0]) < LOW_SAMPLE_MIN_TRANSITIONS:
            low_sample.append(direction)
        candidate_norm = apply_normalization(candidate_features, context.stats)
        per_direction[int(direction)] = sinkhorn_distance(
            context.gt_norm,
            candidate_norm,
            device=device,
        )

    if not per_direction:
        raise ValueError(
            "No candidate directions had valid hold-only Wasserstein transitions."
        )

    # Transition-count weighted mean over directions (still per-direction GT z-score).
    weights = np.asarray(
        [float(per_direction_n[d]) for d in per_direction.keys()], dtype=np.float64
    )
    values = np.asarray(
        [float(per_direction[d]) for d in per_direction.keys()], dtype=np.float64
    )
    if weights.size == 0 or not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
        aggregate = float(np.mean(list(per_direction.values())))
    else:
        aggregate = float(np.sum(weights * values) / np.sum(weights))
    return WassersteinCandidateResult(
        candidate_index=int(candidate_index),
        stiffnesses=dict(stiffnesses),
        aggregate_sinkhorn=aggregate,
        per_direction_sinkhorn=per_direction,
        per_direction_n_transitions=per_direction_n,
        low_sample_directions=tuple(low_sample),
        missing_directions=tuple(missing_directions),
    )
