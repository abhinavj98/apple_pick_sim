"""CMA-ES / Young's-modulus candidate helpers for batched sys-ID (V.5.2).

This module currently owns the material candidate type and log10 maps used by
the interactive E-grid demos. The full pycma loop lands in a later slice.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, NamedTuple

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
)
from apple_pick_sim.fruiting_system import params as fs
from apple_pick_sim.fruiting_system.params import FruitingSystemParams
from apple_pick_sim.system_id.batched_digital_twin_init import (
    gripper_proxy_from_episode_metadata,
    true_params_for_structure,
)
from apple_pick_sim.system_id.wasserstein import (
    WassersteinScoringContext,
    prepare_gt_wasserstein_scoring_context,
    score_candidate_wasserstein_complete,
)
from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
    UNSTABLE_DISQUALIFY_THRESHOLD,
    direction_episodes_from_collectors,
    load_recorded_episodes_for_structure,
    replay_candidates_for_structure,
    replay_instability_fraction_all_frames,
    resolve_direction_indices,
)
from apple_pick_gym.batched_envs.batched_sysid_multi_replay import (
    MultiStructureReplayDiagnostics,
    ReplayFusionIncompatible,
    ReplaySlotKey,
    ReplayStructureRequest,
    build_replay_candidate_blocks,
    replay_multi_structure_candidate_blocks,
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
    return youngs_modulus_candidate_from_params(
        true_params_for_structure(dataset, int(structure_idx))
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


@dataclass(frozen=True)
class PreparedYoungsModulusStructure:
    """Structure-local immutable inputs shared by scalar and fused replay."""

    replay_request: ReplayStructureRequest
    candidates: tuple[YoungsModulusCandidate, ...]
    gt_candidate: YoungsModulusCandidate
    fixed_secondary_e_pa: float | None
    direction_indices: tuple[int, ...]
    recorded_episodes: tuple[dict[str, Any], ...]
    gt_context: WassersteinScoringContext
    scoring_n_directions: int


@dataclass
class YoungsModulusBatchEvaluation:
    evaluations: dict[int, YoungsModulusEvaluation]
    errors: dict[int, str]
    replay_diagnostics: MultiStructureReplayDiagnostics | None
    retried_structures: tuple[int, ...]
    prepared_structures: int = 0
    scoring_seconds: float = 0.0
    total_seconds: float = 0.0


def prepare_youngs_modulus_structure(
    *,
    dataset: BatchedSysIdDataset,
    structure_idx: int,
    candidates: Sequence[YoungsModulusCandidate],
    num_directions: int,
    scoring: YoungsModulusScoringConfig,
    include_excluded: bool = False,
) -> PreparedYoungsModulusStructure:
    """Load and prepare one structure without running physical replay."""
    candidate_list = tuple(candidates)
    if not candidate_list:
        raise ValueError("candidates must be non-empty")
    resolved = resolve_direction_indices(
        dataset,
        structure_idx=int(structure_idx),
        num_directions=int(num_directions),
        include_excluded=bool(include_excluded),
    )
    direction_indices = tuple(int(direction_idx) for direction_idx in resolved)
    recorded = tuple(
        load_recorded_episodes_for_structure(
            dataset,
            structure_idx=int(structure_idx),
            num_directions=len(direction_indices),
            direction_indices=direction_indices,
            include_excluded=bool(include_excluded),
        )
    )
    if len(recorded) != len(direction_indices):
        raise RuntimeError(
            "recorded episode count does not match resolved direction count: "
            f"{len(recorded)} != {len(direction_indices)}"
        )
    scoring_n_directions = (
        int(num_directions)
        if scoring.n_directions is None
        else int(scoring.n_directions)
    )
    gt_context = prepare_gt_wasserstein_scoring_context(
        recorded,
        use_median=bool(scoring.use_median),
        hold_id_onehot=bool(scoring.hold_id_onehot),
        n_holds=scoring.n_holds,
        n_directions=scoring_n_directions,
    )
    base_params = true_params_for_structure(dataset, int(structure_idx))
    gt_candidate = youngs_modulus_candidate_from_params(base_params)
    fixed_secondary_e_pa = (
        None
        if base_params.secondary is None
        else float(base_params.secondary.youngs_modulus_pa)
    )
    first_direction_idx = direction_indices[0]
    gripper = gripper_proxy_from_episode_metadata(
        dataset.load_episode_metadata(int(structure_idx), first_direction_idx)
    )
    recorded_by_direction = dict(zip(direction_indices, recorded, strict=True))
    return PreparedYoungsModulusStructure(
        replay_request=ReplayStructureRequest(
            structure_idx=int(structure_idx),
            candidates=candidate_list,
            direction_indices=direction_indices,
            base_params=base_params,
            recorded_by_direction=recorded_by_direction,
            gripper=gripper,
        ),
        candidates=candidate_list,
        gt_candidate=gt_candidate,
        fixed_secondary_e_pa=fixed_secondary_e_pa,
        direction_indices=direction_indices,
        recorded_episodes=recorded,
        gt_context=gt_context,
        scoring_n_directions=scoring_n_directions,
    )


def score_prepared_youngs_modulus_structure(
    prepared: PreparedYoungsModulusStructure,
    *,
    replay_by_key: dict[ReplaySlotKey, dict[str, Any]],
    scoring: YoungsModulusScoringConfig,
) -> YoungsModulusEvaluation:
    """Score routed replay using original structure/candidate/direction identity."""
    scoring_n_directions = int(prepared.scoring_n_directions)
    provisional: list[YoungsModulusCandidateScore] = []
    replay_episodes: list[list[dict[str, Any]]] = []

    for local_candidate_idx, candidate in enumerate(prepared.candidates):
        keys = tuple(
            ReplaySlotKey(
                structure_idx=prepared.replay_request.structure_idx,
                local_candidate_idx=local_candidate_idx,
                direction_idx=direction_idx,
            )
            for direction_idx in prepared.direction_indices
        )
        for key, direction_idx in zip(keys, prepared.direction_indices, strict=True):
            if (
                key.structure_idx != prepared.replay_request.structure_idx
                or key.local_candidate_idx != local_candidate_idx
                or key.direction_idx != direction_idx
            ):
                raise RuntimeError(f"invalid routed replay key: {key}")
        replay_eps = [replay_by_key[key] for key in keys]
        replay_episodes.append(replay_eps)
        direction_instability = [
            replay_instability_fraction_all_frames(replay=replay, recorded=recorded)
            for replay, recorded in zip(
                replay_eps, prepared.recorded_episodes, strict=True
            )
        ]
        finite_instability = [
            float(fraction)
            for fraction in direction_instability
            if math.isfinite(float(fraction))
        ]
        instability_fraction = (
            max(finite_instability) if finite_instability else float("nan")
        )
        disqualified = any(
            math.isfinite(float(fraction))
            and float(fraction) > float(UNSTABLE_DISQUALIFY_THRESHOLD)
            for fraction in direction_instability
        )
        disqualification_reason = "replay_instability" if disqualified else None

        w_result = score_candidate_wasserstein_complete(
            candidate_index=local_candidate_idx,
            stiffnesses={
                "primary_e_pa": float(candidate.primary),
                "spur_e_pa": float(candidate.spur),
                "stem_e_pa": float(candidate.stem),
            },
            gt_context=prepared.gt_context,
            replay_observations=replay_eps,
            device=scoring.device,
            use_median=bool(scoring.use_median),
            hold_id_onehot=bool(scoring.hold_id_onehot),
            n_holds=scoring.n_holds,
            pool_directions=bool(scoring.pool_directions),
            n_directions=scoring_n_directions,
        )
        if int(w_result.candidate_index) != local_candidate_idx:
            raise RuntimeError(
                "Wasserstein scorer candidate index mismatch: "
                f"expected {local_candidate_idx}, got {w_result.candidate_index}"
            )
        if w_result.missing_directions:
            disqualified = True
            if disqualification_reason is None:
                expected = set(prepared.gt_context.expected_directions)
                missing = {int(direction) for direction in w_result.missing_directions}
                disqualification_reason = (
                    "empty_transition_bag"
                    if expected and missing == expected
                    else "missing_directions"
                )
        aggregate_sinkhorn = float(w_result.aggregate_sinkhorn)
        if not math.isfinite(aggregate_sinkhorn):
            disqualified = True
            if disqualification_reason is None:
                disqualification_reason = "non_finite_sinkhorn"
        provisional.append(
            YoungsModulusCandidateScore(
                candidate_index=local_candidate_idx,
                candidate=candidate,
                aggregate_sinkhorn=aggregate_sinkhorn,
                per_direction_sinkhorn=dict(w_result.per_direction_sinkhorn),
                instability_fraction=float(instability_fraction),
                disqualified=bool(disqualified),
                disqualification_reason=disqualification_reason,
                rank=None,
                is_gt=youngs_modulus_values_match(candidate, prepared.gt_candidate),
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
        score.candidate_index: rank for rank, score in enumerate(ordered, start=1)
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
    base_params = prepared.replay_request.base_params
    return YoungsModulusEvaluation(
        structure_idx=prepared.replay_request.structure_idx,
        gt_candidate=prepared.gt_candidate,
        fixed_secondary_e_pa=prepared.fixed_secondary_e_pa,
        direction_indices=prepared.direction_indices,
        scores=scores,
        replay_episodes=replay_episodes,
        applied_params=[
            candidate.apply_to(base_params) for candidate in prepared.candidates
        ],
    )


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
    """Compatibility wrapper using scalar per-structure physical replay."""
    prepared = prepare_youngs_modulus_structure(
        dataset=dataset,
        structure_idx=int(structure_idx),
        candidates=candidates,
        num_directions=int(num_directions),
        scoring=scoring,
        include_excluded=bool(include_excluded),
    )
    collectors = replay_candidates_for_structure(
        dataset=dataset,
        structure_idx=int(structure_idx),
        candidates=prepared.candidates,
        num_directions=len(prepared.direction_indices),
        direction_indices=prepared.direction_indices,
        seed=seed,
        build_env_fn=build_env_fn,
        max_envs_per_batch=int(max_envs_per_batch),
        on_step=on_step,
        replay_sim_config=replay_sim_config,
        use_oracle_params=True,
        include_excluded=bool(include_excluded),
    )
    replay_by_key: dict[ReplaySlotKey, dict[str, Any]] = {}
    for candidate_index in range(len(prepared.candidates)):
        replay_eps = direction_episodes_from_collectors(
            collectors,
            candidate_index=candidate_index,
            num_directions=len(prepared.direction_indices),
        )
        for direction_idx, replay in zip(
            prepared.direction_indices, replay_eps, strict=True
        ):
            replay_by_key[
                ReplaySlotKey(
                    structure_idx=int(structure_idx),
                    local_candidate_idx=candidate_index,
                    direction_idx=direction_idx,
                )
            ] = replay
    return score_prepared_youngs_modulus_structure(
        prepared,
        replay_by_key=replay_by_key,
        scoring=scoring,
    )


def evaluate_youngs_modulus_structures(
    *,
    dataset: BatchedSysIdDataset,
    structures: Sequence[tuple[int, Sequence[YoungsModulusCandidate]]],
    num_directions: int,
    build_env_fn: Callable[..., Any],
    scoring: YoungsModulusScoringConfig,
    max_envs_per_batch: int = 0,
    seed: int | None = None,
    include_excluded: bool = False,
    fail_fast: bool = False,
    on_step: Callable[..., bool] | None = None,
    replay_sim_config: BatchedHeterogeneousCoupledSimConfig | None = None,
) -> YoungsModulusBatchEvaluation:
    """Prepare independently, replay compatibly in fused chunks, and score."""
    total_started = time.perf_counter()
    prepared_by_idx: dict[int, PreparedYoungsModulusStructure] = {}
    errors: dict[int, str] = {}
    structures_by_idx: dict[int, tuple[YoungsModulusCandidate, ...]] = {}
    for structure_idx, candidates in structures:
        idx = int(structure_idx)
        if idx in structures_by_idx:
            raise ValueError(f"duplicate structure_idx request: {idx}")
        structures_by_idx[idx] = tuple(candidates)
        try:
            prepared_by_idx[idx] = prepare_youngs_modulus_structure(
                dataset=dataset,
                structure_idx=idx,
                candidates=structures_by_idx[idx],
                num_directions=int(num_directions),
                scoring=scoring,
                include_excluded=bool(include_excluded),
            )
        except Exception as exc:
            if fail_fast:
                raise
            errors[idx] = str(exc)

    evaluations: dict[int, YoungsModulusEvaluation] = {}
    retried: list[int] = []
    replay_diagnostics: MultiStructureReplayDiagnostics | None = None
    scoring_seconds = 0.0

    def scalar_retry(structure_idx: int) -> None:
        if structure_idx in retried:
            return
        retried.append(structure_idx)
        try:
            evaluations[structure_idx] = evaluate_youngs_modulus_candidates(
                dataset=dataset,
                structure_idx=structure_idx,
                candidates=structures_by_idx[structure_idx],
                num_directions=int(num_directions),
                build_env_fn=build_env_fn,
                scoring=scoring,
                max_envs_per_batch=int(max_envs_per_batch),
                seed=seed,
                include_excluded=bool(include_excluded),
                on_step=on_step,
                replay_sim_config=replay_sim_config,
            )
        except Exception as exc:
            if fail_fast:
                raise
            errors[structure_idx] = str(exc)

    prepared_items = tuple(prepared_by_idx.values())
    if prepared_items:
        try:
            blocks = build_replay_candidate_blocks(
                tuple(item.replay_request for item in prepared_items)
            )
        except ReplayFusionIncompatible:
            for structure_idx in prepared_by_idx:
                scalar_retry(structure_idx)
        else:
            outcome = replay_multi_structure_candidate_blocks(
                dataset=dataset,
                blocks=blocks,
                build_env_fn=build_env_fn,
                max_envs_per_batch=int(max_envs_per_batch),
                seed=seed,
                fail_fast=bool(fail_fast),
                on_step=on_step,
            )
            replay_diagnostics = outcome.diagnostics
            for structure_idx, prepared in prepared_by_idx.items():
                if structure_idx in outcome.failed_structures:
                    scalar_retry(structure_idx)
                    continue
                scoring_started = time.perf_counter()
                try:
                    evaluations[structure_idx] = (
                        score_prepared_youngs_modulus_structure(
                            prepared,
                            replay_by_key=outcome.replay_by_key,
                            scoring=scoring,
                        )
                    )
                except Exception as exc:
                    if fail_fast:
                        raise
                    errors[structure_idx] = str(exc)
                finally:
                    scoring_seconds += time.perf_counter() - scoring_started

    ordered_evaluations = {
        structure_idx: evaluations[structure_idx]
        for structure_idx in structures_by_idx
        if structure_idx in evaluations
    }
    ordered_errors = {
        structure_idx: errors[structure_idx]
        for structure_idx in structures_by_idx
        if structure_idx in errors and structure_idx not in evaluations
    }
    return YoungsModulusBatchEvaluation(
        evaluations=ordered_evaluations,
        errors=ordered_errors,
        replay_diagnostics=replay_diagnostics,
        retried_structures=tuple(retried),
        prepared_structures=len(prepared_by_idx),
        scoring_seconds=float(scoring_seconds),
        total_seconds=float(time.perf_counter() - total_started),
    )
