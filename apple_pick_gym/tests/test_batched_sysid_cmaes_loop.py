"""Tests for synchronized CMA-ES generation waves and coordinator."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes


def _bounds() -> cmaes.YoungsModulusCmaBounds:
    return cmaes.extract_youngs_modulus_cma_bounds(
        {
            "primary": {"youngs_modulus_pa": {"min": 1.0e7, "max": 1.0e9}},
            "spur": {"youngs_modulus_pa": {"min": 1.0e6, "max": 1.0e8}},
            "stem": {"youngs_modulus_pa": {"min": 1.0e5, "max": 1.0e7}},
        }
    )


@dataclass
class FakeOptimizer:
    """Minimal ask/tell stand-in preserving original sample objects."""

    samples: list[list[float]]
    told: list[tuple[list[list[float]], list[float]]] = field(default_factory=list)
    ask_count: int = 0
    mean: list[float] = field(default_factory=lambda: [7.0, 6.0, 5.0])
    sigma: float = 1.0
    stop_map: dict[str, Any] = field(default_factory=dict)
    popsize: int = 0

    def __post_init__(self) -> None:
        if self.popsize <= 0:
            self.popsize = len(self.samples)

    def ask(self) -> list[list[float]]:
        self.ask_count += 1
        # Return the same object identities expected by tell().
        return self.samples

    def tell(self, solutions: list[list[float]], fitness: list[float]) -> None:
        self.told.append((solutions, list(fitness)))
        self.mean = [float(np.mean([row[i] for row in solutions])) for i in range(3)]

    def stop(self) -> dict[str, Any]:
        return dict(self.stop_map)

    @property
    def C(self) -> np.ndarray:
        return np.eye(3, dtype=float)

    @property
    def sigma_vec(self) -> Any:
        return type("SV", (), {"scaling": np.ones(3, dtype=float)})()

    @property
    def result(self) -> Any:
        return type("R", (), {"xfavorite": list(self.mean)})()


def _score(
    candidate_index: int,
    candidate: cmaes.YoungsModulusCandidate,
    sinkhorn: float,
    *,
    disqualified: bool = False,
    reason: str | None = None,
) -> cmaes.YoungsModulusCandidateScore:
    return cmaes.YoungsModulusCandidateScore(
        candidate_index=candidate_index,
        candidate=candidate,
        aggregate_sinkhorn=float(sinkhorn),
        per_direction_sinkhorn={0: float(sinkhorn)},
        instability_fraction=0.0,
        disqualified=disqualified,
        disqualification_reason=reason,
        rank=None,
        is_gt=False,
    )


def _evaluation(
    structure_idx: int,
    candidates: list[cmaes.YoungsModulusCandidate],
    sinkhorns: list[float],
    *,
    disqualified: set[int] | None = None,
    direction_indices: tuple[int, ...] = (0, 2),
) -> cmaes.YoungsModulusEvaluation:
    bad = disqualified or set()
    scores = [
        _score(
            i,
            cand,
            sinkhorns[i],
            disqualified=i in bad or not math.isfinite(sinkhorns[i]),
            reason="replay_instability" if i in bad else (
                "non_finite_sinkhorn" if not math.isfinite(sinkhorns[i]) else None
            ),
        )
        for i, cand in enumerate(candidates)
    ]
    return cmaes.YoungsModulusEvaluation(
        structure_idx=structure_idx,
        gt_candidate=cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6),
        fixed_secondary_e_pa=None,
        direction_indices=direction_indices,
        scores=scores,
        replay_episodes=[],
        applied_params=[],
    )


def test_penalize_disqualified_uses_worst_finite_plus_margin():
    candidates = [
        cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6),
        cmaes.YoungsModulusCandidate(2e8, 1e7, 1e6),
        cmaes.YoungsModulusCandidate(3e8, 1e7, 1e6),
    ]
    scores = [
        _score(0, candidates[0], 1.5),
        _score(1, candidates[1], 4.0, disqualified=True, reason="replay_instability"),
        _score(2, candidates[2], 2.0),
    ]
    fitness, meta = cmaes.penalize_youngs_modulus_scores(scores)
    # Eligible finite scores are 1.5 and 2.0; worst finite is 2.0.
    assert fitness == pytest.approx([1.5, 2.0 + max(1.0, abs(2.0)), 2.0])
    assert meta[1]["penalized"] is True
    assert meta[1]["raw_aggregate_sinkhorn"] == 4.0
    assert meta[0]["penalized"] is False
    assert meta[2]["penalized"] is False


def test_penalize_all_invalid_raises():
    candidates = [
        cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6),
        cmaes.YoungsModulusCandidate(2e8, 1e7, 1e6),
    ]
    scores = [
        _score(0, candidates[0], float("nan"), disqualified=True, reason="non_finite"),
        _score(1, candidates[1], 1.0, disqualified=True, reason="unstable"),
    ]
    with pytest.raises(cmaes.CmaGenerationFailure, match="all_invalid|no eligible"):
        cmaes.penalize_youngs_modulus_scores(scores)


def test_penalize_overflow_raises():
    candidates = [
        cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6),
        cmaes.YoungsModulusCandidate(2e8, 1e7, 1e6),
    ]
    scores = [
        _score(0, candidates[0], 1.0e308),
        _score(1, candidates[1], 1.0, disqualified=True, reason="unstable"),
    ]
    with pytest.raises(cmaes.CmaGenerationFailure, match="penalty|overflow"):
        cmaes.penalize_youngs_modulus_scores(scores)


def test_generation_wave_routes_by_structure_and_candidate_order():
    bounds = _bounds()
    s0 = [
        [7.0, 6.0, 5.0],
        [7.1, 6.1, 5.1],
    ]
    s1 = [
        [8.0, 7.0, 5.5],
        [8.2, 7.2, 5.7],
    ]
    state0 = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=FakeOptimizer(samples=s0),
        bounds=bounds,
        effective_seed=1,
        population_size=2,
    )
    state1 = cmaes.StructureCmaState(
        structure_idx=5,
        optimizer=FakeOptimizer(samples=s1),
        bounds=bounds,
        effective_seed=2,
        population_size=2,
    )
    seen: dict[str, Any] = {}

    def evaluate_fn(*, structures, **_kwargs):
        seen["structures"] = [
            (int(idx), [tuple(c) for c in cands]) for idx, cands in structures
        ]
        evals = {}
        for idx, cands in structures:
            cand_list = list(cands)
            if idx == 0:
                sinkhorns = [3.0, 1.0]
            else:
                sinkhorns = [2.5, 4.0]
            evals[int(idx)] = _evaluation(int(idx), cand_list, sinkhorns)
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations=evals,
            errors={},
            replay_diagnostics=None,
            retried_structures=(),
        )

    wave = cmaes.run_cma_generation_wave(
        {0: state0, 5: state1},
        evaluate_fn=evaluate_fn,
        generation_index=0,
    )
    assert state0.optimizer.ask_count == 1
    assert state1.optimizer.ask_count == 1
    assert seen["structures"][0][0] == 0
    assert seen["structures"][1][0] == 5
    assert seen["structures"][0][1] == [
        (10 ** 7.0, 10 ** 6.0, 10 ** 5.0),
        (10 ** 7.1, 10 ** 6.1, 10 ** 5.1),
    ]
    # tell receives original sample objects in original order
    told0_samples, told0_fit = state0.optimizer.told[0]
    assert told0_samples is s0
    assert told0_fit == pytest.approx([3.0, 1.0])
    told1_samples, told1_fit = state1.optimizer.told[0]
    assert told1_samples is s1
    assert told1_fit == pytest.approx([2.5, 4.0])
    assert 0 in wave.records and 5 in wave.records
    assert wave.failures == {}


def test_generation_wave_fails_structure_on_missing_or_duplicate_candidate_index():
    bounds = _bounds()
    samples = [[7.0, 6.0, 5.0], [7.1, 6.1, 5.1]]
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=FakeOptimizer(samples=samples),
        bounds=bounds,
        effective_seed=1,
        population_size=2,
    )

    def evaluate_missing(*, structures, **_kwargs):
        idx, cands = structures[0]
        cand_list = list(cands)
        evaluation = _evaluation(int(idx), cand_list, [1.0, 2.0])
        evaluation.scores[1] = _score(0, cand_list[1], 2.0)  # duplicate index 0
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations={int(idx): evaluation},
            errors={},
            replay_diagnostics=None,
            retried_structures=(),
        )

    wave = cmaes.run_cma_generation_wave(
        {0: state},
        evaluate_fn=evaluate_missing,
        generation_index=0,
    )
    assert 0 in wave.failures
    assert state.optimizer.told == []


def test_generation_wave_isolates_structure_local_errors_and_tells_peers():
    bounds = _bounds()
    ok_samples = [[7.0, 6.0, 5.0], [7.1, 6.1, 5.1]]
    bad_samples = [[8.0, 7.0, 5.5], [8.1, 7.1, 5.6]]
    ok = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=FakeOptimizer(samples=ok_samples),
        bounds=bounds,
        effective_seed=1,
        population_size=2,
    )
    bad = cmaes.StructureCmaState(
        structure_idx=1,
        optimizer=FakeOptimizer(samples=bad_samples),
        bounds=bounds,
        effective_seed=2,
        population_size=2,
    )

    def evaluate_fn(*, structures, **_kwargs):
        evals = {}
        errors = {}
        for idx, cands in structures:
            if int(idx) == 1:
                errors[1] = "structure-local boom"
                continue
            evals[int(idx)] = _evaluation(int(idx), list(cands), [1.0, 2.0])
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations=evals,
            errors=errors,
            replay_diagnostics=None,
            retried_structures=(1,),
        )

    wave = cmaes.run_cma_generation_wave(
        {0: ok, 1: bad},
        evaluate_fn=evaluate_fn,
        generation_index=1,
    )
    assert ok.optimizer.told
    assert bad.optimizer.told == []
    assert 0 in wave.records
    assert 1 in wave.failures
    assert wave.failures[1].stage == "generation_evaluation"


def test_generation_wave_top_level_evaluator_exception_fails_whole_wave():
    bounds = _bounds()
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=FakeOptimizer(samples=[[7.0, 6.0, 5.0]]),
        bounds=bounds,
        effective_seed=1,
        population_size=1,
    )

    def evaluate_fn(**_kwargs):
        raise RuntimeError("fused evaluator exploded")

    with pytest.raises(RuntimeError, match="fused evaluator exploded"):
        cmaes.run_cma_generation_wave(
            {0: state},
            evaluate_fn=evaluate_fn,
            generation_index=0,
        )
    assert state.optimizer.told == []


def test_generation_wave_records_ask_and_post_tell_distributions_separately():
    bounds = _bounds()
    samples = [[7.0, 6.0, 5.0], [8.0, 7.0, 6.0]]
    opt = FakeOptimizer(samples=samples, mean=[7.5, 6.5, 5.5], sigma=0.8)
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=opt,
        bounds=bounds,
        effective_seed=1,
        population_size=2,
    )

    def evaluate_fn(*, structures, **_kwargs):
        idx, cands = structures[0]
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations={int(idx): _evaluation(int(idx), list(cands), [2.0, 1.0])},
            errors={},
            replay_diagnostics=None,
            retried_structures=(),
        )

    wave = cmaes.run_cma_generation_wave(
        {0: state},
        evaluate_fn=evaluate_fn,
        generation_index=0,
    )
    record = wave.records[0]
    assert record.ask_distribution.mean_log10 == pytest.approx((7.5, 6.5, 5.5))
    assert record.ask_distribution.sigma == pytest.approx(0.8)
    # Fake tell updates mean to sample-wise mean.
    assert record.post_tell_distribution.mean_log10 == pytest.approx((7.5, 6.5, 5.5))
    assert record.ask_distribution is not record.post_tell_distribution
    assert record.ask_distribution.covariance is not None
    assert "C" in record.ask_distribution.covariance
    assert "effective_unbounded_covariance" in record.ask_distribution.covariance
    assert record.post_tell_distribution.covariance is not None
    assert "C" in record.post_tell_distribution.covariance


def test_generation_wave_cancel_propagates_and_skips_tell():
    from apple_pick_gym.batched_envs.batched_sysid_multi_replay import (
        SysIdReplayCancelled,
    )

    bounds = _bounds()
    opt = FakeOptimizer(samples=[[7.0, 6.0, 5.0]])
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=opt,
        bounds=bounds,
        effective_seed=1,
        population_size=1,
    )

    def evaluate_fn(*, structures, **_kwargs):
        raise SysIdReplayCancelled("viewer closed")

    with pytest.raises(SysIdReplayCancelled, match="viewer closed"):
        cmaes.run_cma_generation_wave(
            {0: state},
            evaluate_fn=evaluate_fn,
            generation_index=0,
        )
    assert opt.told == []
    assert state.status == "active"


def test_generation_wave_all_invalid_reasks_zero_still_fails():
    """Opt-out: zero re-asks keeps the old fail-without-tell behavior."""
    bounds = _bounds()
    samples = [[7.0, 6.0, 5.0], [7.1, 6.1, 5.1]]
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=FakeOptimizer(samples=samples),
        bounds=bounds,
        effective_seed=1,
        population_size=2,
    )

    def evaluate_fn(*, structures, **_kwargs):
        idx, cands = structures[0]
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations={
                int(idx): _evaluation(
                    int(idx),
                    list(cands),
                    [1.0, 2.0],
                    disqualified={0, 1},
                )
            },
            errors={},
            replay_diagnostics=None,
            retried_structures=(),
        )

    wave = cmaes.run_cma_generation_wave(
        {0: state},
        evaluate_fn=evaluate_fn,
        generation_index=0,
        all_invalid_reasks=0,
    )
    assert state.optimizer.told == []
    assert wave.failures[0].stage == "all_invalid"
    assert state.status == "failed"


def test_generation_wave_all_invalid_reasks_then_recovers():
    """Re-ask at same mean/σ until a population has eligible scores."""
    bounds = _bounds()
    opt = FakeOptimizer(samples=[[7.0, 6.0, 5.0], [7.1, 6.1, 5.1]])
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=opt,
        bounds=bounds,
        effective_seed=1,
        population_size=2,
    )
    eval_calls = {"n": 0}

    def evaluate_fn(*, structures, **_kwargs):
        eval_calls["n"] += 1
        idx, cands = structures[0]
        # First two populations fully unstable; third has one eligible.
        bad = {0, 1} if eval_calls["n"] < 3 else {1}
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations={
                int(idx): _evaluation(
                    int(idx),
                    list(cands),
                    [1.0, 2.0],
                    disqualified=bad,
                )
            },
            errors={},
            replay_diagnostics=None,
            retried_structures=(),
        )

    wave = cmaes.run_cma_generation_wave(
        {0: state},
        evaluate_fn=evaluate_fn,
        generation_index=0,
        all_invalid_reasks=3,
    )
    assert state.status == "active"
    assert state.failure is None
    assert 0 not in wave.failures
    assert eval_calls["n"] == 3
    assert opt.ask_count == 3
    assert len(opt.told) == 1
    assert state.completed_generations == 1
    # Normal penalty path: eligible 1.0, DQ gets 1.0 + max(1,1) = 2.0
    assert opt.told[0][1] == pytest.approx([1.0, 2.0])
    record = wave.records[0]
    assert record.penalty_metadata[0].get("all_invalid_reasks") == 2


def test_generation_wave_all_invalid_exhausted_uses_flat_penalty_tell():
    """After re-asks still all-DQ → tell with flat penalty (structure stays active)."""
    bounds = _bounds()
    opt = FakeOptimizer(samples=[[7.0, 6.0, 5.0], [7.1, 6.1, 5.1]])
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=opt,
        bounds=bounds,
        effective_seed=1,
        population_size=2,
    )
    eval_calls = {"n": 0}

    def evaluate_fn(*, structures, **_kwargs):
        eval_calls["n"] += 1
        idx, cands = structures[0]
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations={
                int(idx): _evaluation(
                    int(idx),
                    list(cands),
                    [1.0, 2.0],
                    disqualified={0, 1},
                )
            },
            errors={},
            replay_diagnostics=None,
            retried_structures=(),
        )

    wave = cmaes.run_cma_generation_wave(
        {0: state},
        evaluate_fn=evaluate_fn,
        generation_index=0,
        all_invalid_reasks=3,
    )
    assert state.status == "active"
    assert state.failure is None
    assert 0 not in wave.failures
    # 1 initial + 3 re-asks
    assert eval_calls["n"] == 4
    assert opt.ask_count == 4
    assert len(opt.told) == 1
    assert state.completed_generations == 1
    flat = float(cmaes.ALL_INVALID_FLAT_PENALTY)
    assert opt.told[0][1] == pytest.approx([flat, flat])
    record = wave.records[0]
    assert all(m.get("flat_penalty_tell") for m in record.penalty_metadata)
    assert record.penalty_metadata[0].get("all_invalid_reasks") == 3


def test_generation_wave_real_pycma_all_invalid_reasks_then_tell():
    """Real pycma: all-invalid → re-ask → tell advances countiter / xfavorite."""
    import cma

    bounds = _bounds()
    search = ((6.0, 5.0, 4.0), (10.0, 9.0, 8.0))
    es, seed, _ = cmaes.create_structure_cma_optimizer(
        bounds,
        initial_mean_log10=(8.0, 7.0, 6.0),
        initial_sigma_log10=0.5,
        base_seed=42,
        structure_idx=0,
        population_size=4,
        search_bounds_log10=search,
    )
    assert isinstance(es, cma.CMAEvolutionStrategy)
    countiter_before = int(es.countiter)
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=es,
        bounds=bounds,
        effective_seed=seed,
        population_size=4,
        search_bounds_log10=search,
    )
    eval_calls = {"n": 0}

    def evaluate_fn(*, structures, **_kwargs):
        eval_calls["n"] += 1
        idx, cands = structures[0]
        # First two populations fully DQ; third leaves candidate 0 eligible.
        bad = set(range(len(cands))) if eval_calls["n"] < 3 else set(range(1, len(cands)))
        sinkhorns = [1.0 + 0.1 * i for i in range(len(cands))]
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations={
                int(idx): _evaluation(
                    int(idx),
                    list(cands),
                    sinkhorns,
                    disqualified=bad,
                )
            },
            errors={},
            replay_diagnostics=None,
            retried_structures=(),
        )

    wave = cmaes.run_cma_generation_wave(
        {0: state},
        evaluate_fn=evaluate_fn,
        generation_index=0,
        all_invalid_reasks=3,
    )
    assert state.status == "active"
    assert state.failure is None
    assert 0 not in wave.failures
    assert eval_calls["n"] == 3
    assert state.completed_generations == 1
    assert int(es.countiter) == countiter_before + 1
    favorite = list(es.result.xfavorite)
    assert len(favorite) == 3
    assert all(math.isfinite(float(v)) for v in favorite)
    record = wave.records[0]
    assert record.penalty_metadata[0].get("all_invalid_reasks") == 2
    assert not any(m.get("flat_penalty_tell") for m in record.penalty_metadata)


def test_generation_wave_real_pycma_all_invalid_exhausted_flat_tell():
    """Real pycma: exhausted re-asks still tell with flat penalty (no fail)."""
    import cma

    bounds = _bounds()
    search = ((6.0, 5.0, 4.0), (10.0, 9.0, 8.0))
    es, seed, _ = cmaes.create_structure_cma_optimizer(
        bounds,
        initial_mean_log10=(8.0, 7.0, 6.0),
        initial_sigma_log10=0.5,
        base_seed=7,
        structure_idx=0,
        population_size=4,
        search_bounds_log10=search,
    )
    assert isinstance(es, cma.CMAEvolutionStrategy)
    countiter_before = int(es.countiter)
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=es,
        bounds=bounds,
        effective_seed=seed,
        population_size=4,
        search_bounds_log10=search,
    )
    eval_calls = {"n": 0}

    def evaluate_fn(*, structures, **_kwargs):
        eval_calls["n"] += 1
        idx, cands = structures[0]
        bad = set(range(len(cands)))
        sinkhorns = [1.0 + 0.1 * i for i in range(len(cands))]
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations={
                int(idx): _evaluation(
                    int(idx),
                    list(cands),
                    sinkhorns,
                    disqualified=bad,
                )
            },
            errors={},
            replay_diagnostics=None,
            retried_structures=(),
        )

    wave = cmaes.run_cma_generation_wave(
        {0: state},
        evaluate_fn=evaluate_fn,
        generation_index=0,
        all_invalid_reasks=2,
    )
    assert state.status == "active"
    assert state.failure is None
    assert 0 not in wave.failures
    # 1 initial + 2 re-asks
    assert eval_calls["n"] == 3
    assert state.completed_generations == 1
    assert int(es.countiter) == countiter_before + 1
    favorite = list(es.result.xfavorite)
    assert len(favorite) == 3
    assert all(math.isfinite(float(v)) for v in favorite)
    record = wave.records[0]
    assert all(m.get("flat_penalty_tell") for m in record.penalty_metadata)
    assert record.penalty_metadata[0].get("all_invalid_reasks") == 2
    flat = float(cmaes.ALL_INVALID_FLAT_PENALTY)
    assert all(float(m["fitness"]) == pytest.approx(flat) for m in record.penalty_metadata)


# --- Task 3: coordinator + final means ---


@dataclass
class CapStopOptimizer(FakeOptimizer):
    """Stops after ``stop_after`` successful tell() calls via native stop()."""

    stop_after: int = 10**9
    tells: int = 0

    def tell(self, solutions: list[list[float]], fitness: list[float]) -> None:
        super().tell(solutions, fitness)
        self.tells += 1
        if self.tells >= self.stop_after:
            self.stop_map = {"manualcap": True}


def test_fit_passes_explicit_wave_kind_generation_then_final_mean():
    bounds = _bounds()
    opt = CapStopOptimizer(
        samples=[[7.0, 6.0, 5.0]],
        mean=[7.0, 6.0, 5.0],
        stop_after=1,
    )
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=opt,
        bounds=bounds,
        effective_seed=1,
        population_size=1,
    )
    wave_kinds: list[str] = []

    def evaluate_fn(*, structures, wave_kind="generation", **_kwargs):
        wave_kinds.append(str(wave_kind))
        evals = {
            int(idx): _evaluation(int(idx), list(cands), [0.25] * len(list(cands)))
            for idx, cands in structures
        }
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations=evals,
            errors={},
            replay_diagnostics=None,
            retried_structures=(),
        )

    cmaes.fit_youngs_modulus_structures(
        {0: state},
        max_generations=3,
        evaluate_fn=evaluate_fn,
    )
    assert wave_kinds[0] == "generation"
    assert wave_kinds[-1] == "final_mean"
    assert wave_kinds.count("final_mean") == 1


def test_fit_youngs_modulus_structures_stops_independently_and_scores_final_means():
    bounds = _bounds()
    opt0 = CapStopOptimizer(
        samples=[[7.0, 6.0, 5.0], [7.2, 6.2, 5.2]],
        mean=[7.1, 6.1, 5.1],
        stop_after=1,
    )
    opt1 = CapStopOptimizer(
        samples=[[8.0, 7.0, 5.5], [8.1, 7.1, 5.6]],
        mean=[8.05, 7.05, 5.55],
        stop_after=2,
    )
    opt2 = CapStopOptimizer(
        samples=[[7.5, 6.5, 5.3], [7.6, 6.6, 5.4]],
        mean=[7.55, 6.55, 5.35],
        stop_after=10,
    )
    states = {
        0: cmaes.StructureCmaState(
            structure_idx=0, optimizer=opt0, bounds=bounds, effective_seed=1, population_size=2
        ),
        1: cmaes.StructureCmaState(
            structure_idx=1, optimizer=opt1, bounds=bounds, effective_seed=2, population_size=2
        ),
        2: cmaes.StructureCmaState(
            structure_idx=2, optimizer=opt2, bounds=bounds, effective_seed=3, population_size=2
        ),
    }
    wave_calls: list[list[int]] = []
    final_calls: list[list[tuple[int, tuple[float, float, float]]]] = []

    def evaluate_fn(*, structures, **_kwargs):
        idxs = [int(i) for i, _ in structures]
        # Final-mean wave: one candidate per structure.
        if all(len(list(cands)) == 1 for _, cands in structures):
            final_calls.append(
                [
                    (int(i), tuple(float(x) for x in cands[0]))
                    for i, cands in structures
                ]
            )
            evals = {}
            for idx, cands in structures:
                cand_list = list(cands)
                if int(idx) == 2:
                    # Force final-mean failure for structure 2.
                    evals[int(idx)] = _evaluation(
                        int(idx), cand_list, [1.0], disqualified={0}
                    )
                else:
                    evals[int(idx)] = _evaluation(int(idx), cand_list, [0.5])
            return cmaes.YoungsModulusBatchEvaluation(
                evaluations=evals,
                errors={},
                replay_diagnostics=None,
                retried_structures=(),
            )
        wave_calls.append(idxs)
        evals = {
            int(idx): _evaluation(int(idx), list(cands), [1.0, 2.0])
            for idx, cands in structures
        }
        # Fail structure 2 on its first generation evaluation.
        if 2 in evals and len(wave_calls) == 1:
            return cmaes.YoungsModulusBatchEvaluation(
                evaluations={0: evals[0], 1: evals[1]},
                errors={2: "peer failed"},
                replay_diagnostics=None,
                retried_structures=(2,),
            )
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations=evals,
            errors={},
            replay_diagnostics=None,
            retried_structures=(),
        )

    result = cmaes.fit_youngs_modulus_structures(
        states,
        max_generations=2,
        evaluate_fn=evaluate_fn,
    )
    assert wave_calls[0] == [0, 1, 2]
    assert wave_calls[1] == [1]  # 0 stopped by cap after wave 1; 2 failed
    assert opt0.ask_count == 1
    assert opt1.ask_count == 2
    assert opt2.ask_count == 1
    assert states[0].status == "fitted"
    assert states[1].status == "fitted"
    assert states[2].status == "failed"
    assert states[0].stop_kind in {"generation_cap", "pycma", "both"}
    assert states[0].final_mean_log10 == pytest.approx((7.1, 6.1, 5.1))
    assert states[1].final_mean_log10 == pytest.approx((8.05, 7.05, 5.55))
    assert len(final_calls) == 1
    final_idxs = [idx for idx, _ in final_calls[0]]
    assert final_idxs == [0, 1]
    assert result.fitted_structure_indices == (0, 1)
    assert 2 in result.failed_structure_indices
    assert states[1].stop_kind in {"generation_cap", "pycma", "both"}


def test_snapshot_optimizer_distribution_mean_uses_phenotype_xfavorite():
    """Report mean_log10 must be phenotype ``xfavorite``, not genotype ``mean``.

    With BoundTransform, ``es.mean`` can sit slightly outside the physical box
    while ``result.xfavorite`` stays in phenotype (bounded log10-E) coordinates.
    """

    class PhenotypeMeanOptimizer(FakeOptimizer):
        def __init__(self):
            super().__init__(
                samples=[[8.95, 7.0, 6.0]],
                mean=[8.901, 7.0, 6.0],
            )
            self._favorite = [8.903, 7.0, 6.0]

        @property
        def result(self):
            return type("R", (), {"xfavorite": list(self._favorite)})()

    opt = PhenotypeMeanOptimizer()
    snap = cmaes.snapshot_optimizer_distribution(opt)
    assert snap.mean_log10 == pytest.approx((8.903, 7.0, 6.0))
    assert tuple(opt.mean) == pytest.approx((8.901, 7.0, 6.0))
    assert snap.mean_log10 != pytest.approx(tuple(opt.mean))


def test_fit_uses_xfavorite_not_mean_or_xbest():
    bounds = _bounds()

    class FavoriteOptimizer(FakeOptimizer):
        def __init__(self):
            super().__init__(samples=[[7.0, 6.0, 5.0]], mean=[9.0, 9.0, 9.0])
            self._favorite = [7.25, 6.25, 5.25]
            self.stop_map = {"tolx": 1e-12}

        @property
        def result(self):
            return type(
                "R",
                (),
                {
                    "xfavorite": list(self._favorite),
                    "xbest": [1.0, 1.0, 1.0],
                },
            )()

    opt = FavoriteOptimizer()
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=opt,
        bounds=bounds,
        effective_seed=1,
        population_size=1,
    )

    def evaluate_fn(*, structures, **_kwargs):
        evals = {
            int(idx): _evaluation(int(idx), list(cands), [0.1] * len(list(cands)))
            for idx, cands in structures
        }
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations=evals,
            errors={},
            replay_diagnostics=None,
            retried_structures=(),
        )

    cmaes.fit_youngs_modulus_structures(
        {0: state},
        max_generations=10,
        evaluate_fn=evaluate_fn,
    )
    assert state.final_mean_log10 == pytest.approx((7.25, 6.25, 5.25))
    assert state.stop_kind in {"pycma", "both"}


def test_real_pycma_multi_bowl_moves_toward_distinct_optima():
    """Generic ask/tell mechanics: ``_ask_structure`` now yields SupportKpYoungsCandidate."""
    bounds = cmaes.extract_youngs_modulus_cma_bounds(
        {
            "primary": {"youngs_modulus_pa": {"min": 1.0e6, "max": 1.0e10}},
            "spur": {"youngs_modulus_pa": {"min": 1.0e5, "max": 1.0e9}},
            "stem": {"youngs_modulus_pa": {"min": 1.0e4, "max": 1.0e8}},
        }
    )
    targets = {
        0: (7.0, 6.0, 5.0),
        1: (8.0, 7.0, 6.0),
    }
    states = {}
    for idx, target in targets.items():
        es, seed, _rng = cmaes.create_structure_cma_optimizer(
            bounds,
            initial_sigma_log10=0.5,
            base_seed=0,
            structure_idx=idx,
            population_size=8,
        )
        states[idx] = cmaes.StructureCmaState(
            structure_idx=idx,
            optimizer=es,
            bounds=bounds,
            effective_seed=seed,
            population_size=int(es.popsize),
        )

    def evaluate_fn(*, structures, **_kwargs):
        evals = {}
        for idx, cands in structures:
            target = targets[int(idx)]
            scores = []
            for local_i, cand in enumerate(cands):
                assert isinstance(cand, cmaes.SupportKpYoungsCandidate)
                log10 = (
                    math.log10(cand.support_kp),
                    math.log10(cand.spur),
                    math.log10(cand.stem),
                )
                sinkhorn = sum((a - b) ** 2 for a, b in zip(log10, target))
                scores.append(_score(local_i, cand, sinkhorn))
            evals[int(idx)] = cmaes.YoungsModulusEvaluation(
                structure_idx=int(idx),
                gt_candidate=cmaes.SupportKpYoungsCandidate(1e4, 1e7, 1e6),
                fixed_secondary_e_pa=None,
                direction_indices=(0,),
                scores=scores,
                replay_episodes=[],
                applied_params=[],
            )
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations=evals,
            errors={},
            replay_diagnostics=None,
            retried_structures=(),
        )

    result = cmaes.fit_youngs_modulus_structures(
        states,
        max_generations=12,
        evaluate_fn=evaluate_fn,
    )
    assert set(result.fitted_structure_indices) == {0, 1}
    for idx, target in targets.items():
        mean = states[idx].final_mean_log10
        assert mean is not None
        assert mean == pytest.approx(target, abs=0.35)


def test_optimizer_covariance_diagnostics_are_reported():
    bounds = _bounds()
    es, seed, _ = cmaes.create_structure_cma_optimizer(
        bounds, base_seed=0, structure_idx=0, population_size=6
    )
    diag = cmaes.optimizer_covariance_diagnostics(es)
    assert "C" in diag
    assert "sigma" in diag
    assert "sigma_vec_scaling" in diag
    assert "phenotype_std" in diag
    assert "effective_unbounded_covariance" in diag
    assert len(diag["C"]) == 3
    assert len(diag["effective_unbounded_covariance"]) == 3


# --- Task 4: progress snapshots, strict reports, aggregates ---


def test_fit_emits_on_progress_after_init_wave_and_final():
    bounds = _bounds()
    opt = CapStopOptimizer(
        samples=[[7.0, 6.0, 5.0]],
        mean=[7.0, 6.0, 5.0],
        stop_after=1,
    )
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=opt,
        bounds=bounds,
        effective_seed=1,
        population_size=1,
    )
    events: list[tuple[str, ...]] = []

    def on_progress(states):
        events.append(tuple(states[idx].status for idx in sorted(states)))

    def evaluate_fn(*, structures, **_kwargs):
        evals = {
            int(idx): _evaluation(int(idx), list(cands), [0.25] * len(list(cands)))
            for idx, cands in structures
        }
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations=evals,
            errors={},
            replay_diagnostics=None,
            retried_structures=(),
        )

    cmaes.fit_youngs_modulus_structures(
        {0: state},
        max_generations=3,
        evaluate_fn=evaluate_fn,
        on_progress=on_progress,
    )
    # init, after wave (stopped_pending), after final (fitted)
    assert events[0] == ("active",)
    assert "stopped_pending_final_evaluation" in events[1]
    assert events[-1] == ("fitted",)


def test_to_strict_jsonable_converts_numpy_and_rejects_non_finite():
    payload = {
        "vec": np.array([1.0, 2.0], dtype=np.float64),
        "scalar": np.float64(3.5),
        "nested": {"i": np.int64(4)},
    }
    converted = cmaes.to_strict_jsonable(payload)
    assert converted == {"vec": [1.0, 2.0], "scalar": 3.5, "nested": {"i": 4}}
    json.dumps(converted, allow_nan=False)

    with pytest.raises(ValueError, match="non-finite|NaN|Infinity"):
        cmaes.to_strict_jsonable({"bad": float("nan")})
    with pytest.raises(ValueError, match="non-finite|NaN|Infinity"):
        cmaes.to_strict_jsonable({"bad": np.float64(float("inf"))})


def test_structure_report_snapshot_includes_per_generation_covariance():
    bounds = _bounds()
    es, _seed, _ = cmaes.create_structure_cma_optimizer(
        bounds, base_seed=0, structure_idx=0, population_size=4
    )
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=es,
        bounds=bounds,
        effective_seed=1,
        population_size=4,
    )
    samples = es.ask()
    fitness = [float(i) for i in range(len(samples))]
    ask_dist = cmaes.snapshot_optimizer_distribution(es)
    es.tell(samples, fitness)
    post_dist = cmaes.snapshot_optimizer_distribution(es)
    candidate = cmaes.candidates_from_log10_e(tuple(float(v) for v in samples[0]))
    score = _score(0, candidate, 1.0)
    state.generations.append(
        cmaes.CmaGenerationRecord(
            generation_index=0,
            structure_idx=0,
            ask_samples_log10=tuple(
                tuple(float(v) for v in row) for row in samples  # type: ignore[misc]
            ),
            candidates=tuple(
                cmaes.candidates_from_log10_e(tuple(float(v) for v in row))
                for row in samples
            ),
            raw_scores=tuple(
                _score(
                    i,
                    cmaes.candidates_from_log10_e(tuple(float(v) for v in row)),
                    float(i),
                )
                for i, row in enumerate(samples)
            ),
            penalized_fitness=tuple(fitness),
            penalty_metadata=tuple({} for _ in samples),
            ask_distribution=ask_dist,
            post_tell_distribution=post_dist,
        )
    )
    snapshot = cmaes.structure_cma_report_snapshot(
        state,
        base_seed=0,
        initial_sigma_log10=1.0,
    )
    gen0 = snapshot["generations"][0]
    assert "covariance" in gen0["ask_distribution"]
    assert gen0["ask_distribution"]["covariance"] is not None
    assert "C" in gen0["ask_distribution"]["covariance"]
    assert "effective_unbounded_covariance" in gen0["ask_distribution"]["covariance"]
    assert "covariance" in gen0["post_tell_distribution"]
    assert gen0["post_tell_distribution"]["covariance"] is not None
    assert snapshot["covariance"] is not None


def test_structure_report_covariance_null_only_on_narrow_adapter_errors():
    bounds = _bounds()

    class BrokenCOptimizer(FakeOptimizer):
        @property
        def C(self):
            raise RuntimeError("unexpected adapter bug")

    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=BrokenCOptimizer(samples=[[7.0, 6.0, 5.0]]),
        bounds=bounds,
        effective_seed=1,
        population_size=1,
    )
    with pytest.raises(RuntimeError, match="unexpected adapter bug"):
        cmaes.structure_cma_report_snapshot(
            state,
            base_seed=0,
            initial_sigma_log10=1.0,
        )


# --- Task 2 wave integration: sparse dirs / chunks / compat / fused retry ---


def _wave_recorded(direction_idx: int, *, frames: int = 3) -> dict[str, Any]:
    return {
        "action": np.zeros((frames, 6), dtype=np.float32),
        "junction_names": ["support", "primary_spur", "spur_stem"],
    }


def _wave_prepared(
    structure_idx: int,
    candidates: tuple[cmaes.YoungsModulusCandidate, ...],
    directions: tuple[int, ...],
    *,
    base_params: Any,
):
    from apple_pick_gym.batched_envs import batched_sysid_multi_replay as multi
    from apple_pick_sim.fruiting_system.params import GripperProxyConfig

    request = multi.ReplayStructureRequest(
        structure_idx=structure_idx,
        candidates=candidates,
        direction_indices=directions,
        base_params=base_params,
        recorded_by_direction={
            d: _wave_recorded(d) for d in directions
        },
        gripper=GripperProxyConfig(
            fix_to_apple=True,
            weld_direction=(1.0, 0.0, 0.0),
        ),
    )
    return cmaes.PreparedYoungsModulusStructure(
        replay_request=request,
        candidates=candidates,
        gt_candidate=cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6),
        fixed_secondary_e_pa=None,
        direction_indices=directions,
        recorded_episodes=tuple(_wave_recorded(d) for d in directions),
        gt_context=MagicMock(),
        scoring_n_directions=max(directions) + 1 if directions else 1,
    )


def test_wave_through_evaluator_routes_sparse_direction_ids(monkeypatch):
    """Generation wave + real evaluator preserve sparse source direction IDs."""
    from apple_pick_gym.batched_envs import batched_sysid_multi_replay as multi
    from apple_pick_sim.fruiting_system import params as fs
    from apple_pick_sim.tests.conftest import RANGES_FIXTURE

    bounds = _bounds()
    samples = [[7.0, 6.0, 5.0], [7.1, 6.1, 5.1]]
    state = cmaes.StructureCmaState(
        structure_idx=4,
        optimizer=FakeOptimizer(samples=samples),
        bounds=bounds,
        effective_seed=1,
        population_size=2,
    )
    sparse_dirs = (0, 3)
    base_params = fs.sample_params(fs.load_ranges(RANGES_FIXTURE), seed=4)
    prepared_dirs: list[tuple[int, ...]] = []
    scored_keys: list[multi.ReplaySlotKey] = []

    def fake_prepare(**kwargs):
        candidates = tuple(kwargs["candidates"])
        structure_idx = int(kwargs["structure_idx"])
        prepared_dirs.append(sparse_dirs)
        return _wave_prepared(
            structure_idx, candidates, sparse_dirs, base_params=base_params
        )

    def fake_replay(**kwargs):
        blocks = kwargs["blocks"]
        replay_by_key = {}
        for block in blocks:
            for slot in block.slots:
                replay_by_key[slot.key] = {"obs": {}}
                scored_keys.append(slot.key)
        return multi.MultiStructureReplayOutcome(
            replay_by_key=replay_by_key,
            failed_structures={},
            diagnostics=multi.MultiStructureReplayDiagnostics(
                candidate_blocks=len(blocks),
                flattened_envs=sum(len(b.slots) for b in blocks),
                chunk_env_counts=(sum(len(b.slots) for b in blocks),),
                failed_chunk_indices=(),
                build_seconds=0.0,
                replay_seconds=0.0,
            ),
        )

    def fake_score(prepared, *, replay_by_key, scoring):
        scores = [
            _score(i, cand, 1.0 + i) for i, cand in enumerate(prepared.candidates)
        ]
        return cmaes.YoungsModulusEvaluation(
            structure_idx=prepared.replay_request.structure_idx,
            gt_candidate=prepared.gt_candidate,
            fixed_secondary_e_pa=None,
            direction_indices=prepared.direction_indices,
            scores=scores,
            replay_episodes=[],
            applied_params=[],
        )

    monkeypatch.setattr(cmaes, "prepare_youngs_modulus_structure", fake_prepare)
    monkeypatch.setattr(cmaes, "replay_multi_structure_candidate_blocks", fake_replay)
    monkeypatch.setattr(cmaes, "score_prepared_youngs_modulus_structure", fake_score)

    def evaluate_fn(*, structures, **_kwargs):
        return cmaes.evaluate_youngs_modulus_structures(
            dataset=MagicMock(),
            structures=structures,
            num_directions=5,
            build_env_fn=MagicMock(),
            scoring=cmaes.YoungsModulusScoringConfig(n_directions=5),
        )

    wave = cmaes.run_cma_generation_wave(
        {4: state},
        evaluate_fn=evaluate_fn,
        generation_index=0,
    )
    assert 4 in wave.records
    assert prepared_dirs == [sparse_dirs]
    assert {key.direction_idx for key in scored_keys} == {0, 3}
    assert len(scored_keys) == 4  # 2 candidates × 2 sparse dirs
    assert state.optimizer.told


def test_wave_through_evaluator_compat_fallback_and_fused_retry(monkeypatch):
    """Compat mismatch scalar-falls-back; fused failure retries only affected."""
    from apple_pick_gym.batched_envs import batched_sysid_multi_replay as multi
    from apple_pick_sim.fruiting_system import params as fs
    from apple_pick_sim.tests.conftest import RANGES_FIXTURE

    bounds = _bounds()
    states = {
        4: cmaes.StructureCmaState(
            structure_idx=4,
            optimizer=FakeOptimizer(samples=[[7.0, 6.0, 5.0]]),
            bounds=bounds,
            effective_seed=1,
            population_size=1,
        ),
        1: cmaes.StructureCmaState(
            structure_idx=1,
            optimizer=FakeOptimizer(samples=[[8.0, 7.0, 5.5]]),
            bounds=bounds,
            effective_seed=2,
            population_size=1,
        ),
    }
    base_params = fs.sample_params(fs.load_ranges(RANGES_FIXTURE), seed=4)

    # First wave: fusion incompatible -> full scalar fallback.
    monkeypatch.setattr(
        cmaes,
        "prepare_youngs_modulus_structure",
        lambda **kwargs: _wave_prepared(
            int(kwargs["structure_idx"]),
            tuple(kwargs["candidates"]),
            (0, 2) if int(kwargs["structure_idx"]) == 4 else (1, 3),
            base_params=base_params,
        ),
    )
    monkeypatch.setattr(
        cmaes,
        "build_replay_candidate_blocks",
        MagicMock(side_effect=multi.ReplayFusionIncompatible("topology mismatch")),
    )
    scalar_calls: list[int] = []

    def fake_scalar(**kwargs):
        idx = int(kwargs["structure_idx"])
        scalar_calls.append(idx)
        cands = list(kwargs["candidates"])
        return _evaluation(idx, cands, [0.5] * len(cands))

    monkeypatch.setattr(cmaes, "evaluate_youngs_modulus_candidates", fake_scalar)

    def evaluate_fn(*, structures, **_kwargs):
        return cmaes.evaluate_youngs_modulus_structures(
            dataset=MagicMock(),
            structures=structures,
            num_directions=2,
            build_env_fn=MagicMock(),
            scoring=cmaes.YoungsModulusScoringConfig(n_directions=2),
        )

    wave = cmaes.run_cma_generation_wave(
        states,
        evaluate_fn=evaluate_fn,
        generation_index=0,
    )
    assert set(wave.records) == {4, 1}
    assert sorted(scalar_calls) == [1, 4]
    assert states[4].optimizer.told and states[1].optimizer.told

    # Second scenario: fused runtime failure retries only structure 4.
    scalar_calls.clear()
    for state in states.values():
        state.optimizer.told.clear()
        state.generations.clear()
        state.completed_generations = 0

    monkeypatch.setattr(
        cmaes,
        "prepare_youngs_modulus_structure",
        lambda **kwargs: _wave_prepared(
            int(kwargs["structure_idx"]),
            tuple(kwargs["candidates"]),
            (0, 2),
            base_params=base_params,
        ),
    )
    monkeypatch.setattr(
        cmaes,
        "build_replay_candidate_blocks",
        multi.build_replay_candidate_blocks,
    )

    def fake_fused(**kwargs):
        blocks = kwargs["blocks"]
        replay_by_key = {}
        for block in blocks:
            if int(block.structure_idx) == 4:
                continue
            for slot in block.slots:
                replay_by_key[slot.key] = {"obs": {}}
        return multi.MultiStructureReplayOutcome(
            replay_by_key=replay_by_key,
            failed_structures={4: "chunk 0: synthetic failure"},
            diagnostics=multi.MultiStructureReplayDiagnostics(
                candidate_blocks=len(blocks),
                flattened_envs=sum(len(b.slots) for b in blocks),
                chunk_env_counts=(sum(len(b.slots) for b in blocks),),
                failed_chunk_indices=(0,),
                build_seconds=0.0,
                replay_seconds=0.0,
            ),
        )

    monkeypatch.setattr(cmaes, "replay_multi_structure_candidate_blocks", fake_fused)

    def fake_score(prepared, *, replay_by_key, scoring):
        scores = [_score(0, prepared.candidates[0], 0.4)]
        return cmaes.YoungsModulusEvaluation(
            structure_idx=prepared.replay_request.structure_idx,
            gt_candidate=prepared.gt_candidate,
            fixed_secondary_e_pa=None,
            direction_indices=prepared.direction_indices,
            scores=scores,
            replay_episodes=[],
            applied_params=[],
        )

    monkeypatch.setattr(cmaes, "score_prepared_youngs_modulus_structure", fake_score)

    wave2 = cmaes.run_cma_generation_wave(
        states,
        evaluate_fn=evaluate_fn,
        generation_index=1,
    )
    assert set(wave2.records) == {4, 1}
    assert scalar_calls == [4]
    assert states[4].optimizer.told and states[1].optimizer.told


def test_wave_through_evaluator_respects_chunk_boundaries(monkeypatch):
    """Forced chunking still routes scores by stable ReplaySlotKey identity."""
    from apple_pick_gym.batched_envs import batched_sysid_multi_replay as multi
    from apple_pick_sim.fruiting_system import params as fs
    from apple_pick_sim.tests.conftest import RANGES_FIXTURE

    bounds = _bounds()
    samples = [[7.0, 6.0, 5.0], [7.2, 6.2, 5.2], [7.4, 6.4, 5.4]]
    state = cmaes.StructureCmaState(
        structure_idx=4,
        optimizer=FakeOptimizer(samples=samples),
        bounds=bounds,
        effective_seed=1,
        population_size=3,
    )
    base_params = fs.sample_params(fs.load_ranges(RANGES_FIXTURE), seed=4)
    chunk_sizes: list[int] = []

    def fake_prepare(**kwargs):
        return _wave_prepared(
            int(kwargs["structure_idx"]),
            tuple(kwargs["candidates"]),
            (0, 2),
            base_params=base_params,
        )

    def fake_replay(**kwargs):
        blocks = kwargs["blocks"]
        # Simulate the planner's max_envs_per_batch=2 with 2 dirs -> 1 cand/chunk.
        max_envs = int(kwargs.get("max_envs_per_batch") or 0)
        assert max_envs == 2
        chunks = multi.chunk_replay_candidate_blocks(blocks, max_envs_per_batch=2)
        replay_by_key = {}
        for chunk in chunks:
            chunk_sizes.append(sum(len(block.slots) for block in chunk))
            for block in chunk:
                for slot in block.slots:
                    replay_by_key[slot.key] = {"obs": {}}
        return multi.MultiStructureReplayOutcome(
            replay_by_key=replay_by_key,
            failed_structures={},
            diagnostics=multi.MultiStructureReplayDiagnostics(
                candidate_blocks=len(blocks),
                flattened_envs=sum(len(b.slots) for b in blocks),
                chunk_env_counts=tuple(chunk_sizes),
                failed_chunk_indices=(),
                build_seconds=0.0,
                replay_seconds=0.0,
            ),
        )

    def fake_score(prepared, *, replay_by_key, scoring):
        scores = [
            _score(i, cand, float(i) + 0.1)
            for i, cand in enumerate(prepared.candidates)
        ]
        return cmaes.YoungsModulusEvaluation(
            structure_idx=prepared.replay_request.structure_idx,
            gt_candidate=prepared.gt_candidate,
            fixed_secondary_e_pa=None,
            direction_indices=prepared.direction_indices,
            scores=scores,
            replay_episodes=[],
            applied_params=[],
        )

    monkeypatch.setattr(cmaes, "prepare_youngs_modulus_structure", fake_prepare)
    monkeypatch.setattr(cmaes, "replay_multi_structure_candidate_blocks", fake_replay)
    monkeypatch.setattr(cmaes, "score_prepared_youngs_modulus_structure", fake_score)

    def evaluate_fn(*, structures, **_kwargs):
        return cmaes.evaluate_youngs_modulus_structures(
            dataset=MagicMock(),
            structures=structures,
            num_directions=2,
            build_env_fn=MagicMock(),
            scoring=cmaes.YoungsModulusScoringConfig(n_directions=2),
            max_envs_per_batch=2,
        )

    wave = cmaes.run_cma_generation_wave(
        {4: state},
        evaluate_fn=evaluate_fn,
        generation_index=0,
    )
    assert 4 in wave.records
    assert chunk_sizes == [2, 2, 2]
    assert len(state.optimizer.told[0][1]) == 3


def test_evaluate_structures_propagates_cancel_without_scalar_retry(monkeypatch):
    from apple_pick_gym.batched_envs import batched_sysid_multi_replay as multi
    from apple_pick_sim.fruiting_system import params as fs
    from apple_pick_sim.tests.conftest import RANGES_FIXTURE

    base_params = fs.sample_params(fs.load_ranges(RANGES_FIXTURE), seed=4)
    monkeypatch.setattr(
        cmaes,
        "prepare_youngs_modulus_structure",
        lambda **kwargs: _wave_prepared(
            int(kwargs["structure_idx"]),
            tuple(kwargs["candidates"]),
            (0, 2),
            base_params=base_params,
        ),
    )
    scalar = MagicMock(side_effect=AssertionError("scalar retry must not run"))
    monkeypatch.setattr(cmaes, "evaluate_youngs_modulus_candidates", scalar)

    def fake_fused(**_kwargs):
        raise multi.SysIdReplayCancelled("viewer closed")

    monkeypatch.setattr(cmaes, "replay_multi_structure_candidate_blocks", fake_fused)

    with pytest.raises(multi.SysIdReplayCancelled, match="viewer closed"):
        cmaes.evaluate_youngs_modulus_structures(
            dataset=MagicMock(),
            structures=(
                (4, (cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6),)),
                (1, (cmaes.YoungsModulusCandidate(2e8, 2e7, 2e6),)),
            ),
            num_directions=2,
            build_env_fn=MagicMock(),
            scoring=cmaes.YoungsModulusScoringConfig(n_directions=2),
            fail_fast=False,
        )
    scalar.assert_not_called()


def test_structure_report_snapshot_includes_required_fields_and_counters():
    bounds = _bounds()
    es, seed, _ = cmaes.create_structure_cma_optimizer(
        bounds, base_seed=0, structure_idx=0, population_size=4
    )
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=es,
        bounds=bounds,
        effective_seed=seed,
        population_size=4,
        status="fitted",
        completed_generations=2,
        optimizer_samples_told=8,
        final_mean_log10=(7.0, 6.0, 5.0),
        best_sample_log10=(7.1, 6.1, 5.1),
        best_sample_fitness=1.25,
        stop_kind="generation_cap",
        stop_conditions={"tolfun": 1e-12},
        gt_candidate=cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6),
    )
    state.final_evaluation = _evaluation(
        0,
        [cmaes.YoungsModulusCandidate(1e7, 1e6, 1e5)],
        [0.4],
        direction_indices=(0, 3),
    )
    snapshot = cmaes.structure_cma_report_snapshot(
        state,
        base_seed=0,
        initial_sigma_log10=1.0,
        replay_candidate_evaluations=9,
        final_mean_evaluations=1,
        physical_env_slots=12,
        scalar_retries=2,
    )
    assert snapshot["status"] == "fitted"
    assert snapshot["structure_idx"] == 0
    assert snapshot["completed_generations"] == 2
    assert snapshot["optimizer_samples_told"] == 8
    assert snapshot["replay_candidate_evaluations"] == 9
    assert snapshot["final_mean_evaluations"] == 1
    assert snapshot["physical_env_slots"] == 12
    assert snapshot["scalar_retries"] == 2
    assert snapshot["bounds"] is None
    assert snapshot["final_mean"]["log10_e"] == [7.0, 6.0, 5.0]
    assert snapshot["best_sample"]["fitness"] == 1.25
    assert snapshot["gt"]["e_pa"] == [1e8, 1e7, 1e6]
    assert "covariance" in snapshot
    json.dumps(cmaes.to_strict_jsonable(snapshot), allow_nan=False)


def test_structure_report_uses_evaluated_generation_samples_for_extrema():
    bounds = _bounds()
    optimizer = FakeOptimizer(samples=[[7.0, 6.0, 5.0]])
    state = cmaes.StructureCmaState(
        structure_idx=4,
        optimizer=optimizer,
        bounds=bounds,
        effective_seed=17,
        population_size=2,
        status="fitted",
        final_mean_log10=(8.9, 7.9, 6.9),
    )
    candidate = cmaes.YoungsModulusCandidate(1e7, 1e6, 1e5)
    score = _score(0, candidate, 1.0)
    distribution = cmaes.CmaDistributionSnapshot(
        mean_log10=(8.0, 7.0, 6.0),
        sigma=1.0,
    )
    for generation_index, samples in enumerate(
        (
            ((7.25, 6.5, 5.75), (8.5, 7.25, 6.0)),
            ((7.0, 6.75, 5.5), (8.25, 7.5, 6.25)),
        )
    ):
        state.generations.append(
            cmaes.CmaGenerationRecord(
                generation_index=generation_index,
                structure_idx=4,
                ask_samples_log10=samples,
                candidates=(candidate, candidate),
                raw_scores=(score, score),
                penalized_fitness=(1.0, 1.0),
                penalty_metadata=({}, {}),
                ask_distribution=distribution,
                post_tell_distribution=distribution,
            )
        )

    snapshot = cmaes.structure_cma_report_snapshot(
        state,
        base_seed=0,
        initial_sigma_log10=1.0,
    )

    extrema = snapshot["evaluated_history_extrema"]
    assert extrema["min_log10_e"] == pytest.approx([7.0, 6.5, 5.5])
    assert extrema["max_log10_e"] == pytest.approx([8.5, 7.5, 6.25])
    assert extrema["min_e_pa"] == pytest.approx([1e7, 10**6.5, 10**5.5])
    assert extrema["max_e_pa"] == pytest.approx([10**8.5, 10**7.5, 10**6.25])
    # The extrema come only from evaluated generation populations, not fixture
    # bounds or the separately evaluated final distribution mean.
    assert extrema["max_log10_e"] != list(bounds.log10_upper)
    assert extrema["max_log10_e"] != list(state.final_mean_log10)
    assert "covariance" in snapshot


def test_structure_report_has_null_evaluated_extrema_before_first_population():
    bounds = _bounds()
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=FakeOptimizer(samples=[[7.0, 6.0, 5.0]]),
        bounds=bounds,
        effective_seed=1,
        population_size=1,
    )

    snapshot = cmaes.structure_cma_report_snapshot(
        state,
        base_seed=0,
        initial_sigma_log10=1.0,
    )

    assert snapshot["evaluated_history_extrema"] == {
        "min_log10_e": None,
        "max_log10_e": None,
        "min_e_pa": None,
        "max_e_pa": None,
    }
    assert "covariance" in snapshot


def test_aggregate_fitted_youngs_modulus_stats_component_wise():
    fitted = [
        cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6),
        cmaes.YoungsModulusCandidate(1e10, 1e9, 1e8),
    ]
    gt = [
        cmaes.YoungsModulusCandidate(2e8, 2e7, 2e6),
        cmaes.YoungsModulusCandidate(3e8, 3e7, 3e6),
    ]
    stats = cmaes.aggregate_fitted_youngs_modulus_stats(
        fitted_candidates=fitted,
        gt_candidates=gt,
        requested_count=3,
        failed_count=1,
    )
    assert stats["requested_structures"] == 3
    assert stats["fitted_structures"] == 2
    assert stats["failed_structures"] == 1
    assert stats["mean_log10_e"] == pytest.approx([9.0, 8.0, 7.0])
    assert stats["geometric_mean_e_pa"] == pytest.approx([1e9, 1e8, 1e7])
    assert stats["mean_e_pa"] == pytest.approx([5.05e9, 5.05e8, 5.05e7])
    assert stats["min_e_pa"] == pytest.approx([1e8, 1e7, 1e6])
    assert stats["max_e_pa"] == pytest.approx([1e10, 1e9, 1e8])
    assert stats["mean_gt_e_pa"] == pytest.approx([2.5e8, 2.5e7, 2.5e6])
    # ddof=1 sample covariance of two log10 vectors
    logs = np.log10([[1e8, 1e7, 1e6], [1e10, 1e9, 1e8]])
    expected_cov = np.cov(logs, rowvar=False, ddof=1)
    assert np.allclose(stats["sample_cov_log10_e"], expected_cov)
    assert stats["sample_std_log10_e"] == pytest.approx(
        np.std(logs, axis=0, ddof=1).tolist()
    )


def test_aggregate_stats_null_covariance_below_two_fits():
    fitted = [cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6)]
    stats = cmaes.aggregate_fitted_youngs_modulus_stats(
        fitted_candidates=fitted,
        gt_candidates=fitted,
        requested_count=1,
        failed_count=0,
    )
    assert stats["fitted_structures"] == 1
    assert stats["sample_cov_log10_e"] is None
    assert stats["sample_std_log10_e"] is None


def test_generation_score_summary_mean_and_sample_variance():
    summary = cmaes.generation_score_summary(
        [
            {"penalized": False, "raw_aggregate_sinkhorn": 1.0, "fitness": 1.0},
            {"penalized": False, "raw_aggregate_sinkhorn": 3.0, "fitness": 3.0},
            {"penalized": True, "raw_aggregate_sinkhorn": 9.0, "fitness": 10.0},
        ],
        penalized_fitness=[1.0, 3.0, 10.0],
    )
    assert summary["n_eligible"] == 2
    assert summary["eligible_mean"] == pytest.approx(2.0)
    assert summary["eligible_variance"] == pytest.approx(2.0)  # ddof=1
    assert summary["eligible_std"] == pytest.approx(math.sqrt(2.0))
    assert summary["best_eligible"] == pytest.approx(1.0)
    assert summary["penalized_mean"] == pytest.approx((1.0 + 3.0 + 10.0) / 3.0)
    assert summary["n_penalized"] == 1


def test_generation_score_summary_null_variance_below_two_eligible():
    summary = cmaes.generation_score_summary(
        [{"penalized": False, "raw_aggregate_sinkhorn": 4.0, "fitness": 4.0}],
        penalized_fitness=[4.0],
    )
    assert summary["eligible_mean"] == pytest.approx(4.0)
    assert summary["eligible_variance"] is None
    assert summary["eligible_std"] is None


def test_fit_records_wave_and_total_timing():
    bounds = _bounds()
    opt = CapStopOptimizer(
        samples=[[7.0, 6.0, 5.0]],
        mean=[7.0, 6.0, 5.0],
        stop_after=1,
    )
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=opt,
        bounds=bounds,
        effective_seed=1,
        population_size=1,
    )

    def evaluate_fn(*, structures, wave_kind="generation", **_kwargs):
        import time

        time.sleep(0.01)
        evals = {
            int(idx): _evaluation(int(idx), list(cands), [0.25] * len(list(cands)))
            for idx, cands in structures
        }
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations=evals,
            errors={},
            replay_diagnostics=None,
            retried_structures=(),
        )

    result = cmaes.fit_youngs_modulus_structures(
        {0: state},
        max_generations=3,
        evaluate_fn=evaluate_fn,
    )
    timing = result.timing
    assert timing["fit_seconds"] >= 0.02
    assert len(timing["waves"]) == 2
    assert timing["waves"][0]["wave_kind"] == "generation"
    assert timing["waves"][1]["wave_kind"] == "final_mean"
    assert all(float(w["seconds"]) >= 0.01 for w in timing["waves"])
    assert state.generations[0].wave_seconds == pytest.approx(
        timing["waves"][0]["seconds"], rel=0, abs=1e-3
    )


def test_structure_report_snapshot_includes_generation_score_summary_and_wave_seconds():
    bounds = _bounds()
    opt = FakeOptimizer(samples=[[7.0, 6.0, 5.0], [7.1, 6.1, 5.1]])
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=opt,
        bounds=bounds,
        effective_seed=1,
        population_size=2,
    )
    cand0 = cmaes.YoungsModulusCandidate(1e7, 1e6, 1e5)
    cand1 = cmaes.YoungsModulusCandidate(1.1e7, 1.1e6, 1.1e5)
    state.generations.append(
        cmaes.CmaGenerationRecord(
            generation_index=0,
            structure_idx=0,
            ask_samples_log10=((7.0, 6.0, 5.0), (7.1, 6.1, 5.1)),
            candidates=(cand0, cand1),
            raw_scores=(
                _score(0, cand0, 2.0),
                _score(1, cand1, 4.0),
            ),
            penalized_fitness=(2.0, 4.0),
            penalty_metadata=(
                {
                    "candidate_index": 0,
                    "penalized": False,
                    "raw_aggregate_sinkhorn": 2.0,
                    "fitness": 2.0,
                    "disqualification_reason": None,
                },
                {
                    "candidate_index": 1,
                    "penalized": False,
                    "raw_aggregate_sinkhorn": 4.0,
                    "fitness": 4.0,
                    "disqualification_reason": None,
                },
            ),
            ask_distribution=cmaes.CmaDistributionSnapshot((7.0, 6.0, 5.0), 1.0, None),
            post_tell_distribution=cmaes.CmaDistributionSnapshot(
                (7.05, 6.05, 5.05), 0.9, None
            ),
            wave_seconds=1.25,
        )
    )
    snapshot = cmaes.structure_cma_report_snapshot(
        state,
        base_seed=0,
        initial_sigma_log10=1.0,
    )
    gen0 = snapshot["generations"][0]
    assert gen0["wave_seconds"] == pytest.approx(1.25)
    assert gen0["score_summary"]["eligible_mean"] == pytest.approx(3.0)
    assert gen0["score_summary"]["eligible_variance"] == pytest.approx(2.0)


# --- Task 6: support-k_p CMA vector semantics ---


def test_extract_support_kp_youngs_modulus_cma_bounds_uses_absolute_box_not_fixture():
    """Slot 0 is an absolute support_kp box, never fixture primary-E epsilon-bands."""
    ranges = {
        # Deliberately different from the [2, 6] support_kp default so a bug
        # that reads fixture "primary" bounds would be caught.
        "primary": {"youngs_modulus_pa": {"min": 1.0e7, "max": 1.0e9}},
        "spur": {"youngs_modulus_pa": {"min": 1.0e6, "max": 1.0e8}},
        "stem": {"youngs_modulus_pa": {"min": 1.0e5, "max": 1.0e7}},
    }
    bounds = cmaes.extract_support_kp_youngs_modulus_cma_bounds(ranges)
    assert bounds.log10_lower == pytest.approx((2.0, 6.0, 5.0))
    assert bounds.log10_upper == pytest.approx((6.0, 8.0, 7.0))
    assert bounds.log10_midpoint == pytest.approx((4.0, 7.0, 6.0))
    assert bounds.support_kp.physical_min_pa == pytest.approx(1.0e2)
    assert bounds.support_kp.physical_max_pa == pytest.approx(1.0e6)
    # Alias: support_kp is literally the primary slot.
    assert bounds.support_kp is bounds.primary


def test_extract_support_kp_youngs_modulus_cma_bounds_custom_box():
    ranges = {
        "spur": {"youngs_modulus_pa": {"min": 1.0e6, "max": 1.0e8}},
        "stem": {"youngs_modulus_pa": {"min": 1.0e5, "max": 1.0e7}},
    }
    bounds = cmaes.extract_support_kp_youngs_modulus_cma_bounds(
        ranges, support_kp_log10_lower=1.0, support_kp_log10_upper=3.0
    )
    assert bounds.log10_lower[0] == pytest.approx(1.0)
    assert bounds.log10_upper[0] == pytest.approx(3.0)


def test_extract_support_kp_youngs_modulus_cma_bounds_rejects_inverted_box():
    with pytest.raises(ValueError, match="log10_min < log10_max"):
        cmaes.extract_support_kp_youngs_modulus_cma_bounds(
            {
                "spur": {"youngs_modulus_pa": {"min": 1.0e6, "max": 1.0e8}},
                "stem": {"youngs_modulus_pa": {"min": 1.0e5, "max": 1.0e7}},
            },
            support_kp_log10_lower=6.0,
            support_kp_log10_upper=2.0,
        )


def test_ask_structure_conversion_yields_support_kp_candidates_via_wave():
    """Production ask/tell boundary uses candidates_from_log10_vector, not _e."""
    bounds = cmaes.extract_support_kp_youngs_modulus_cma_bounds(
        {
            "spur": {"youngs_modulus_pa": {"min": 1.0e6, "max": 1.0e8}},
            "stem": {"youngs_modulus_pa": {"min": 1.0e5, "max": 1.0e7}},
        }
    )
    samples = [[4.0, 9.0, 8.5]]
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=FakeOptimizer(samples=samples),
        bounds=bounds,
        effective_seed=1,
        population_size=1,
    )
    seen: dict[str, Any] = {}

    def evaluate_fn(*, structures, **_kwargs):
        idx, cands = structures[0]
        seen["candidate"] = cands[0]
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations={int(idx): _evaluation(int(idx), list(cands), [1.0])},
            errors={},
            replay_diagnostics=None,
            retried_structures=(),
        )

    cmaes.run_cma_generation_wave(
        {0: state}, evaluate_fn=evaluate_fn, generation_index=0
    )
    candidate = seen["candidate"]
    assert isinstance(candidate, cmaes.SupportKpYoungsCandidate)
    assert candidate.support_kp == pytest.approx(10.0**4.0)
    assert candidate.spur == pytest.approx(10.0**9.0)
    assert candidate.stem == pytest.approx(10.0**8.5)


def test_evaluate_final_means_uses_candidates_from_log10_vector():
    """Final-mean scoring also maps through the support_kp vector, not log10_e."""
    bounds = _bounds()
    opt = CapStopOptimizer(
        samples=[[4.0, 9.0, 8.5]],
        mean=[4.0, 9.0, 8.5],
        stop_after=1,
    )
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=opt,
        bounds=bounds,
        effective_seed=1,
        population_size=1,
    )
    seen: dict[str, Any] = {}

    def evaluate_fn(*, structures, wave_kind="generation", **_kwargs):
        idx, cands = structures[0]
        if wave_kind == "final_mean":
            seen["candidate"] = cands[0]
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations={int(idx): _evaluation(int(idx), list(cands), [0.1] * len(cands))},
            errors={},
            replay_diagnostics=None,
            retried_structures=(),
        )

    cmaes.fit_youngs_modulus_structures(
        {0: state}, max_generations=3, evaluate_fn=evaluate_fn
    )
    candidate = seen["candidate"]
    assert isinstance(candidate, cmaes.SupportKpYoungsCandidate)
    assert candidate.support_kp == pytest.approx(10.0**4.0)


def test_to_strict_jsonable_supports_support_kp_candidate():
    candidate = cmaes.SupportKpYoungsCandidate(support_kp=1.0e4, spur=1.0e9, stem=1.0e8)
    converted = cmaes.to_strict_jsonable(candidate)
    assert converted == {"support_kp": 1.0e4, "spur": 1.0e9, "stem": 1.0e8}
    json.dumps(converted, allow_nan=False)


def test_structure_report_snapshot_final_mean_and_gt_use_support_kp_fields():
    bounds = _bounds()
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=FakeOptimizer(samples=[[4.0, 9.0, 8.5]]),
        bounds=bounds,
        effective_seed=1,
        population_size=1,
        status="fitted",
        final_mean_log10=(4.0, 9.0, 8.5),
        gt_candidate=cmaes.SupportKpYoungsCandidate(1.0e4, 1.0e9, 1.0e8),
    )
    snapshot = cmaes.structure_cma_report_snapshot(
        state, base_seed=0, initial_sigma_log10=1.0
    )
    assert snapshot["final_mean"]["log10_e"] == pytest.approx([4.0, 9.0, 8.5])
    assert snapshot["final_mean"]["e_pa"] == pytest.approx(
        [10.0**4.0, 10.0**9.0, 10.0**8.5]
    )
    assert snapshot["gt"]["e_pa"] == pytest.approx([1.0e4, 1.0e9, 1.0e8])
    json.dumps(cmaes.to_strict_jsonable(snapshot), allow_nan=False)
