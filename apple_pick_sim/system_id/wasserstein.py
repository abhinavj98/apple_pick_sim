"""Sinkhorn (entropic Wasserstein) helpers for sys-ID transition-bag scoring."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from apple_pick_sim.system_id.mmd import NormalizationStats, apply_normalization, fit_gt_normalization
from apple_pick_sim.system_id.mmd_features import (
    combine_transition_features,
    n_junctions_from_episodes,
)

SINKHORN_P = 2
SINKHORN_BLUR = 1.0
LOW_SAMPLE_MIN_TRANSITIONS = 8
POOLED_DIRECTION_KEY = -1


@dataclass(frozen=True)
class WassersteinDirectionContext:
    """GT-normalized transition bag for one excitation direction (or pooled)."""

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
        if self.per_direction_sinkhorn:
            return
        # Incomplete complete-scorer results may omit diagnostics only when marked
        # missing and the aggregate fitness is non-finite.
        if self.missing_directions and not np.isfinite(self.aggregate_sinkhorn):
            return
        raise ValueError(
            "Wasserstein candidate result must include at least one direction"
        )


@dataclass(frozen=True)
class WassersteinScoringContext:
    """GT bags for pooled optimizer fitness and physical-direction diagnostics."""

    pooled: WassersteinDirectionContext
    per_direction: dict[int, WassersteinDirectionContext]

    @property
    def expected_directions(self) -> tuple[int, ...]:
        return tuple(sorted(self.per_direction))


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
    """Compute Sinkhorn divergence between two normalized transition bags.

    For singleton (Dirac) bags with ``SINKHORN_P == 2``, GeomLoss's debiased
    Sinkhorn equals the ground cost ``C(x,y) = 0.5 * ||x-y||_2^2`` (see GeomLoss
    ``SamplesLoss`` docs for ``p=2``). Identical singletons have zero diameter, so
    GeomLoss's ε-scaling heuristic fails; use that exact cost instead of catching
    GeomLoss errors.
    """
    x_arr = np.asarray(x_norm, dtype=np.float64)
    y_arr = np.asarray(y_norm, dtype=np.float64)
    if x_arr.ndim != 2 or y_arr.ndim != 2:
        raise ValueError("sinkhorn_distance expects 2D feature matrices")
    if x_arr.shape[0] == 0 or y_arr.shape[0] == 0:
        raise ValueError("sinkhorn_distance requires at least one sample per bag")
    if x_arr.shape[1] != y_arr.shape[1]:
        raise ValueError(
            f"feature dimension mismatch: x={x_arr.shape[1]} y={y_arr.shape[1]}"
        )
    # Dirac↔Dirac: OT_ε is unique and equals C; debiased Sinkhorn is exactly C.
    if x_arr.shape[0] == 1 and y_arr.shape[0] == 1:
        if SINKHORN_P != 2:
            raise ValueError(
                f"singleton Sinkhorn shortcut requires SINKHORN_P=2; got {SINKHORN_P}"
            )
        diff = x_arr[0] - y_arr[0]
        return float(0.5 * np.dot(diff, diff))

    dev = _resolve_device(device)
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for Sinkhorn scoring; sync with --extra vic"
        ) from exc
    loss_fn = _samples_loss(device=dev)
    x_t = torch.as_tensor(x_arr, device=dev)
    y_t = torch.as_tensor(y_arr, device=dev)
    with torch.no_grad():
        value = loss_fn(x_t, y_t)
    return float(value.detach().cpu().item())


def _feature_kwargs(
    *,
    use_median: bool,
    hold_id_onehot: bool,
    n_holds: int | None,
    dir_id_onehot: bool,
    n_directions: int | None,
    hold_reduce: str | None = None,
) -> dict[str, Any]:
    return {
        "use_median": bool(use_median),
        "hold_id_onehot": bool(hold_id_onehot),
        "n_holds": n_holds,
        "dir_id_onehot": bool(dir_id_onehot),
        "n_directions": n_directions,
        "hold_reduce": hold_reduce,
    }


def _pool_by_direction(by_direction: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    if not by_direction:
        return {}
    pooled = np.concatenate(
        [by_direction[k] for k in sorted(by_direction.keys())],
        axis=0,
    )
    return {POOLED_DIRECTION_KEY: pooled}


def prepare_gt_wasserstein_context(
    recorded_episodes: list[dict],
    *,
    use_median: bool = False,
    hold_id_onehot: bool = False,
    n_holds: int | None = None,
    pool_directions: bool = False,
    n_directions: int | None = None,
    hold_reduce: str | None = None,
) -> dict[int, WassersteinDirectionContext]:
    """Fit GT normalization from recorded transition bags (per-dir or pooled)."""
    dir_id_onehot = bool(pool_directions)
    gt_by_direction = combine_transition_features(
        recorded_episodes,
        **_feature_kwargs(
            use_median=use_median,
            hold_id_onehot=hold_id_onehot,
            n_holds=n_holds,
            dir_id_onehot=dir_id_onehot,
            n_directions=n_directions,
            hold_reduce=hold_reduce,
        ),
    )
    if pool_directions:
        gt_by_direction = _pool_by_direction(gt_by_direction)
    if not gt_by_direction:
        raise ValueError("No valid hold-only GT transition features were found.")

    n_junctions = n_junctions_from_episodes(recorded_episodes)
    context: dict[int, WassersteinDirectionContext] = {}
    for direction, gt_features in gt_by_direction.items():
        stats = fit_gt_normalization(gt_features, n_junctions=n_junctions)
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
    use_median: bool = False,
    hold_id_onehot: bool = False,
    n_holds: int | None = None,
    pool_directions: bool = False,
    n_directions: int | None = None,
    hold_reduce: str | None = None,
) -> WassersteinCandidateResult:
    """Score one replayed candidate against precomputed GT Wasserstein context."""
    dir_id_onehot = bool(pool_directions)
    candidate_by_direction = combine_transition_features(
        replay_observations,
        **_feature_kwargs(
            use_median=use_median,
            hold_id_onehot=hold_id_onehot,
            n_holds=n_holds,
            dir_id_onehot=dir_id_onehot,
            n_directions=n_directions,
            hold_reduce=hold_reduce,
        ),
    )
    if pool_directions:
        candidate_by_direction = _pool_by_direction(candidate_by_direction)

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


def prepare_gt_wasserstein_scoring_context(
    recorded_episodes: list[dict],
    *,
    use_median: bool = False,
    hold_id_onehot: bool = False,
    n_holds: int | None = None,
    pool_directions: bool = True,
    n_directions: int | None = None,
    hold_reduce: str | None = None,
) -> WassersteinScoringContext:
    """Build pooled fitness GT and independently normalized per-direction diagnostics.

    ``pool_directions`` is accepted for API symmetry with
    ``score_candidate_wasserstein_complete``; prepare always builds both the
    pooled fitness bag and physical-direction diagnostics.
    """
    del pool_directions  # always build pooled + per-direction contexts
    per_direction_features = combine_transition_features(
        recorded_episodes,
        **_feature_kwargs(
            use_median=use_median,
            hold_id_onehot=hold_id_onehot,
            n_holds=n_holds,
            dir_id_onehot=False,
            n_directions=n_directions,
            hold_reduce=hold_reduce,
        ),
    )
    if not per_direction_features:
        raise ValueError("No valid hold-only GT transition features were found.")

    n_junctions = n_junctions_from_episodes(recorded_episodes)
    per_direction: dict[int, WassersteinDirectionContext] = {}
    for direction, gt_features in per_direction_features.items():
        stats = fit_gt_normalization(gt_features, n_junctions=n_junctions)
        gt_norm = apply_normalization(gt_features, stats)
        per_direction[int(direction)] = WassersteinDirectionContext(
            gt_norm=gt_norm, stats=stats
        )

    # Pooled fitness bag always uses fixed-width physical-direction one-hot
    # columns, independent of whether the scorer later uses pooled fitness.
    pooled_source = combine_transition_features(
        recorded_episodes,
        **_feature_kwargs(
            use_median=use_median,
            hold_id_onehot=hold_id_onehot,
            n_holds=n_holds,
            dir_id_onehot=True,
            n_directions=n_directions,
            hold_reduce=hold_reduce,
        ),
    )
    pooled_features = _pool_by_direction(pooled_source)
    if not pooled_features:
        raise ValueError("No valid hold-only GT transition features were found.")
    pooled_raw = pooled_features[POOLED_DIRECTION_KEY]
    pooled_stats = fit_gt_normalization(pooled_raw, n_junctions=n_junctions)
    pooled = WassersteinDirectionContext(
        gt_norm=apply_normalization(pooled_raw, pooled_stats),
        stats=pooled_stats,
    )
    return WassersteinScoringContext(pooled=pooled, per_direction=per_direction)


def score_candidate_wasserstein_complete(
    *,
    candidate_index: int,
    stiffnesses: dict[str, float],
    gt_context: WassersteinScoringContext,
    replay_observations: list[dict],
    device: str | None = None,
    use_median: bool = False,
    hold_id_onehot: bool = False,
    n_holds: int | None = None,
    pool_directions: bool = True,
    n_directions: int | None = None,
    hold_reduce: str | None = None,
) -> WassersteinCandidateResult:
    """Score a candidate with pooled fitness and physical-direction diagnostics."""
    candidate_per_direction = combine_transition_features(
        replay_observations,
        **_feature_kwargs(
            use_median=use_median,
            hold_id_onehot=hold_id_onehot,
            n_holds=n_holds,
            dir_id_onehot=False,
            n_directions=n_directions,
            hold_reduce=hold_reduce,
        ),
    )

    expected = set(gt_context.expected_directions)
    unexpected = sorted(set(candidate_per_direction) - expected)
    if unexpected:
        raise ValueError(
            "unexpected replay directions not present in GT scoring context: "
            + ", ".join(str(d) for d in unexpected)
        )

    missing = tuple(sorted(expected - set(candidate_per_direction)))

    per_direction: dict[int, float] = {}
    per_direction_n: dict[int, int] = {}
    low_sample: list[int] = []

    for direction in gt_context.expected_directions:
        candidate_features = candidate_per_direction.get(int(direction))
        if candidate_features is None:
            continue
        context = gt_context.per_direction[int(direction)]
        per_direction_n[int(direction)] = int(candidate_features.shape[0])
        if candidate_features.shape[1] != context.gt_norm.shape[1]:
            raise ValueError(
                "Wasserstein feature dimension mismatch for direction "
                f"{direction}: gt={context.gt_norm.shape[1]} "
                f"candidate={candidate_features.shape[1]}"
            )
        if int(candidate_features.shape[0]) < LOW_SAMPLE_MIN_TRANSITIONS:
            low_sample.append(int(direction))
        candidate_norm = apply_normalization(candidate_features, context.stats)
        per_direction[int(direction)] = sinkhorn_distance(
            context.gt_norm,
            candidate_norm,
            device=device,
        )

    if missing:
        return WassersteinCandidateResult(
            candidate_index=int(candidate_index),
            stiffnesses=dict(stiffnesses),
            aggregate_sinkhorn=float("nan"),
            per_direction_sinkhorn=per_direction,
            per_direction_n_transitions=per_direction_n,
            low_sample_directions=tuple(low_sample),
            missing_directions=missing,
        )

    if pool_directions:
        candidate_with_dir = combine_transition_features(
            replay_observations,
            **_feature_kwargs(
                use_median=use_median,
                hold_id_onehot=hold_id_onehot,
                n_holds=n_holds,
                dir_id_onehot=True,
                n_directions=n_directions,
                hold_reduce=hold_reduce,
            ),
        )
        # Restrict to expected physical directions before pooling.
        candidate_with_dir = {
            int(d): candidate_with_dir[int(d)]
            for d in gt_context.expected_directions
            if int(d) in candidate_with_dir
        }
        pooled_candidate = _pool_by_direction(candidate_with_dir)
        if not pooled_candidate:
            raise ValueError(
                "No candidate directions had valid hold-only Wasserstein transitions."
            )
        pooled_features = pooled_candidate[POOLED_DIRECTION_KEY]
        if pooled_features.shape[1] != gt_context.pooled.gt_norm.shape[1]:
            raise ValueError(
                "Wasserstein pooled feature dimension mismatch: "
                f"gt={gt_context.pooled.gt_norm.shape[1]} "
                f"candidate={pooled_features.shape[1]}"
            )
        candidate_norm = apply_normalization(pooled_features, gt_context.pooled.stats)
        aggregate = sinkhorn_distance(
            gt_context.pooled.gt_norm,
            candidate_norm,
            device=device,
        )
    else:
        weights = np.asarray(
            [float(per_direction_n[d]) for d in per_direction.keys()],
            dtype=np.float64,
        )
        values = np.asarray(
            [float(per_direction[d]) for d in per_direction.keys()],
            dtype=np.float64,
        )
        if (
            weights.size == 0
            or not np.all(np.isfinite(weights))
            or float(np.sum(weights)) <= 0.0
        ):
            aggregate = float(np.mean(list(per_direction.values())))
        else:
            aggregate = float(np.sum(weights * values) / np.sum(weights))

    return WassersteinCandidateResult(
        candidate_index=int(candidate_index),
        stiffnesses=dict(stiffnesses),
        aggregate_sinkhorn=float(aggregate),
        per_direction_sinkhorn=per_direction,
        per_direction_n_transitions=per_direction_n,
        low_sample_directions=tuple(low_sample),
        missing_directions=(),
    )
