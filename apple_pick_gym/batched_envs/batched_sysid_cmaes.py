"""CMA-ES / Young's-modulus candidate helpers for batched sys-ID (V.5.2).

This module currently owns the material candidate type and log10 maps used by
the interactive E-grid demos. The full pycma loop lands in a later slice.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, NamedTuple

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
)
from apple_pick_sim.fruiting_system import params as fs
from apple_pick_sim.fruiting_system.params import FruitingSystemParams
from apple_pick_sim.system_id.wasserstein import (
    prepare_gt_wasserstein_context,
    score_candidate_wasserstein,
)
from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
    UNSTABLE_DISQUALIFY_THRESHOLD,
    direction_episodes_from_collectors,
    load_recorded_episodes_for_structure,
    replay_candidates_for_structure,
    replay_instability_fraction_all_frames,
    resolve_direction_indices,
)

if TYPE_CHECKING:
    from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset


class YoungsModulusCandidate(NamedTuple):
    """One material candidate: Young's modulus (Pa) for primary, spur, stem."""

    primary: float
    spur: float
    stem: float

    def apply_to(self, base: FruitingSystemParams) -> FruitingSystemParams:
        """Return a copy with candidate ``E`` re-derived into VBD knobs.

        Only ``primary``, ``spur``, and ``stem`` are updated when present on
        ``base``. ``secondary`` (and any other fields) are left unchanged.
        Geometry and ``damping_ratio`` are frozen; axial stretch overrides on
        the base rod are preserved when they differ from beam theory.
        """
        out = base
        for segment, value in (
            ("primary", self.primary),
            ("spur", self.spur),
            ("stem", self.stem),
        ):
            if getattr(base, segment) is not None:
                out = fs.set_rod_youngs_modulus(out, segment, float(value))
        return out

    def short_label(self) -> str:
        """Compact legend label."""
        return (
            f"log10=({math.log10(self.primary):.2f},"
            f"{math.log10(self.spur):.2f},"
            f"{math.log10(self.stem):.2f})"
        )


def iter_youngs_modulus_candidates(
    *,
    primary_values: Sequence[float],
    spur_values: Sequence[float],
    stem_values: Sequence[float],
) -> Iterable[YoungsModulusCandidate]:
    """Yield Young's-modulus grid candidates in Cartesian product order."""
    for primary, spur, stem in product(primary_values, spur_values, stem_values):
        yield YoungsModulusCandidate(
            primary=float(primary),
            spur=float(spur),
            stem=float(stem),
        )


def candidates_from_log10_e(
    log10_e: Sequence[float],
) -> YoungsModulusCandidate:
    """Map ``log10([E_primary, E_spur, E_stem])`` to a physical candidate."""
    if len(log10_e) != 3:
        raise ValueError(f"log10_e must have length 3, got {len(log10_e)}")
    return YoungsModulusCandidate(
        primary=10.0 ** float(log10_e[0]),
        spur=10.0 ** float(log10_e[1]),
        stem=10.0 ** float(log10_e[2]),
    )


def log10_e_from_params(params: FruitingSystemParams) -> tuple[float, float, float]:
    """Extract ``log10(E)`` for primary, spur, stem (hard error if missing)."""
    if params.primary is None or params.spur is None or params.stem is None:
        raise ValueError(
            "params must include primary, spur, and stem rods for log10_e_from_params"
        )
    return (
        math.log10(float(params.primary.youngs_modulus_pa)),
        math.log10(float(params.spur.youngs_modulus_pa)),
        math.log10(float(params.stem.youngs_modulus_pa)),
    )


def youngs_modulus_candidate_from_params(
    params: FruitingSystemParams,
) -> YoungsModulusCandidate:
    """Build a candidate from absolute ``E`` on primary/spur/stem."""
    if params.primary is None or params.spur is None or params.stem is None:
        raise ValueError(
            "params must include primary, spur, and stem rods"
        )
    return YoungsModulusCandidate(
        primary=float(params.primary.youngs_modulus_pa),
        spur=float(params.spur.youngs_modulus_pa),
        stem=float(params.stem.youngs_modulus_pa),
    )


def gt_youngs_modulus_candidate_from_structure(
    dataset: BatchedSysIdDataset,
    structure_idx: int,
) -> YoungsModulusCandidate:
    import apple_pick_gym.batched_envs.batched_sysid_cmaes as _mod

    return youngs_modulus_candidate_from_params(
        _mod.true_params_for_structure(dataset, int(structure_idx))
    )


def youngs_modulus_values_match(
    left: YoungsModulusCandidate,
    right: YoungsModulusCandidate,
    *,
    log10_atol: float = 1e-9,
) -> bool:
    return all(
        math.isclose(math.log10(a), math.log10(b), rel_tol=0.0, abs_tol=log10_atol)
        for a, b in zip(left, right, strict=True)
    )


def maybe_include_gt_candidate(
    candidates: Sequence[YoungsModulusCandidate],
    gt: YoungsModulusCandidate,
    *,
    include_gt: bool,
) -> list[YoungsModulusCandidate]:
    items = list(candidates)
    if not include_gt or any(youngs_modulus_values_match(item, gt) for item in items):
        return items
    return [*items, gt]


from apple_pick_sim.system_id.batched_digital_twin_init import (  # noqa: E402
    true_params_for_structure,
)


@dataclass(frozen=True)
class YoungsModulusScoringConfig:
    use_median: bool = True
    hold_id_onehot: bool = True
    pool_directions: bool = True
    n_holds: int | None = None
    n_directions: int | None = None
    device: str | None = None


@dataclass(frozen=True)
class YoungsModulusCandidateScore:
    candidate_index: int
    candidate: YoungsModulusCandidate
    aggregate_sinkhorn: float
    per_direction_sinkhorn: dict[int, float]
    instability_fraction: float
    disqualified: bool
    disqualification_reason: str | None
    rank: int | None
    is_gt: bool


@dataclass
class YoungsModulusEvaluation:
    structure_idx: int
    gt_candidate: YoungsModulusCandidate
    fixed_secondary_e_pa: float | None
    direction_indices: tuple[int, ...]
    scores: list[YoungsModulusCandidateScore]
    replay_episodes: list[list[dict[str, Any]]]
    applied_params: list[FruitingSystemParams]


def evaluate_youngs_modulus_candidates(
    *,
    dataset: BatchedSysIdDataset,
    structure_idx: int,
    candidates: Sequence[YoungsModulusCandidate],
    num_directions: int,
    build_env_fn: Callable[..., Any],
    scoring: YoungsModulusScoringConfig,
    max_envs_per_batch: int = 0,
    seed: int | None = None,
    include_excluded: bool = False,
    on_step: Callable[..., bool] | None = None,
    replay_sim_config: BatchedHeterogeneousCoupledSimConfig | None = None,
) -> YoungsModulusEvaluation:
    """Replay, score, and rank Young's-modulus candidates for one structure."""
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("candidates must be non-empty")

    direction_indices_list = resolve_direction_indices(
        dataset,
        structure_idx=int(structure_idx),
        num_directions=int(num_directions),
        include_excluded=bool(include_excluded),
    )
    direction_indices = tuple(int(d) for d in direction_indices_list)
    num_usable_directions = len(direction_indices_list)

    recorded = load_recorded_episodes_for_structure(
        dataset,
        structure_idx=int(structure_idx),
        num_directions=num_usable_directions,
        direction_indices=direction_indices_list,
        include_excluded=bool(include_excluded),
    )
    gt_context = prepare_gt_wasserstein_context(
        recorded,
        use_median=bool(scoring.use_median),
        hold_id_onehot=bool(scoring.hold_id_onehot),
        n_holds=scoring.n_holds,
        pool_directions=bool(scoring.pool_directions),
        n_directions=scoring.n_directions,
    )
    collectors = replay_candidates_for_structure(
        dataset=dataset,
        structure_idx=int(structure_idx),
        candidates=candidate_list,
        num_directions=num_usable_directions,
        direction_indices=direction_indices_list,
        seed=seed,
        build_env_fn=build_env_fn,
        max_envs_per_batch=int(max_envs_per_batch),
        on_step=on_step,
        replay_sim_config=replay_sim_config,
        use_oracle_params=True,
        include_excluded=bool(include_excluded),
    )

    base_params = true_params_for_structure(dataset, int(structure_idx))
    gt_candidate = gt_youngs_modulus_candidate_from_structure(dataset, int(structure_idx))
    fixed_secondary_e_pa: float | None = None
    if base_params.secondary is not None:
        fixed_secondary_e_pa = float(base_params.secondary.youngs_modulus_pa)
    applied_params = [candidate.apply_to(base_params) for candidate in candidate_list]

    provisional: list[YoungsModulusCandidateScore] = []
    replay_episodes: list[list[dict[str, Any]]] = []

    for candidate_index, candidate in enumerate(candidate_list):
        replay_eps = direction_episodes_from_collectors(
            collectors,
            candidate_index=int(candidate_index),
            num_directions=num_usable_directions,
        )
        replay_episodes.append(replay_eps)

        direction_instability = [
            replay_instability_fraction_all_frames(
                replay=replay_eps[dir_local],
                recorded=recorded[dir_local],
            )
            for dir_local in range(num_usable_directions)
        ]
        finite_instability = [
            float(fraction)
            for fraction in direction_instability
            if math.isfinite(float(fraction))
        ]
        instability_fraction = (
            max(finite_instability) if finite_instability else float("nan")
        )

        disqualified = False
        disqualification_reason: str | None = None
        if any(
            math.isfinite(float(fraction))
            and float(fraction) > float(UNSTABLE_DISQUALIFY_THRESHOLD)
            for fraction in direction_instability
        ):
            disqualified = True
            disqualification_reason = "replay_instability"

        w_result = score_candidate_wasserstein(
            candidate_index=int(candidate_index),
            stiffnesses={
                "primary_e_pa": float(candidate.primary),
                "spur_e_pa": float(candidate.spur),
                "stem_e_pa": float(candidate.stem),
            },
            gt_context=gt_context,
            replay_observations=replay_eps,
            device=scoring.device,
            use_median=bool(scoring.use_median),
            hold_id_onehot=bool(scoring.hold_id_onehot),
            n_holds=scoring.n_holds,
            pool_directions=bool(scoring.pool_directions),
            n_directions=scoring.n_directions,
        )

        if w_result.missing_directions:
            disqualified = True
            if disqualification_reason is None:
                disqualification_reason = "missing_directions"

        aggregate_sinkhorn = float(w_result.aggregate_sinkhorn)
        if not math.isfinite(aggregate_sinkhorn):
            disqualified = True
            if disqualification_reason is None:
                disqualification_reason = "non_finite_sinkhorn"

        provisional.append(
            YoungsModulusCandidateScore(
                candidate_index=int(candidate_index),
                candidate=candidate,
                aggregate_sinkhorn=aggregate_sinkhorn,
                per_direction_sinkhorn=dict(w_result.per_direction_sinkhorn),
                instability_fraction=float(instability_fraction),
                disqualified=bool(disqualified),
                disqualification_reason=disqualification_reason,
                rank=None,
                is_gt=youngs_modulus_values_match(candidate, gt_candidate),
            )
        )

    eligible = [
        score
        for score in provisional
        if not score.disqualified and math.isfinite(score.aggregate_sinkhorn)
    ]
    ordered = sorted(
        eligible,
        key=lambda score: (score.aggregate_sinkhorn, score.candidate_index),
    )
    rank_by_index = {
        score.candidate_index: rank
        for rank, score in enumerate(ordered, start=1)
    }

    scores = [
        YoungsModulusCandidateScore(
            candidate_index=score.candidate_index,
            candidate=score.candidate,
            aggregate_sinkhorn=score.aggregate_sinkhorn,
            per_direction_sinkhorn=dict(score.per_direction_sinkhorn),
            instability_fraction=score.instability_fraction,
            disqualified=score.disqualified,
            disqualification_reason=score.disqualification_reason,
            rank=rank_by_index.get(score.candidate_index),
            is_gt=score.is_gt,
        )
        for score in provisional
    ]

    return YoungsModulusEvaluation(
        structure_idx=int(structure_idx),
        gt_candidate=gt_candidate,
        fixed_secondary_e_pa=fixed_secondary_e_pa,
        direction_indices=direction_indices,
        scores=scores,
        replay_episodes=replay_episodes,
        applied_params=applied_params,
    )
