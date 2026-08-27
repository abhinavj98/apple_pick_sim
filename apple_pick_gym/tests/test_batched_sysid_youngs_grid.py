"""Tests for Young's-modulus candidate evaluation and ranking."""

from __future__ import annotations

import dataclasses
import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from apple_pick_sim.fruiting_system import params as fs
from apple_pick_sim.system_id.wasserstein import WassersteinCandidateResult
from apple_pick_sim.tests.conftest import RANGES_FIXTURE
from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes
from apple_pick_gym.batched_envs import batched_sysid_mmd_grid as grid
from apple_pick_gym.batched_envs import batched_sysid_multi_replay as multi


def _base_primary_spur_stem_with_secondary(seed: int = 0) -> fs.FruitingSystemParams:
    ranges = fs.load_ranges(RANGES_FIXTURE)
    params = fs.sample_params(ranges, seed=seed)
    assert params.primary is not None and params.spur is not None and params.stem is not None
    assert params.secondary is not None
    return params


def _dummy_recorded_episode(*, direction_idx: int = 0, n_frames: int = 8) -> dict:
    junction_names = ["joint_a"]
    base = np.arange(n_frames, dtype=np.float32).reshape(n_frames, 1)
    return {
        "action": np.zeros((n_frames, 6), dtype=np.float32),
        "phase": np.ones(n_frames, dtype=np.int8),
        "dir_idx": np.full(n_frames, direction_idx, dtype=np.int32),
        "excitation_type": np.zeros(n_frames, dtype=np.int8),
        "excitation_direction": np.tile(
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            (n_frames, 1),
        ),
        "junction_names": junction_names,
        "ft_wrist": np.zeros((n_frames, 6), dtype=np.float32),
        "tcp_velocity": np.zeros((n_frames, 6), dtype=np.float32),
        "tcp_pos": np.zeros((n_frames, 3), dtype=np.float32),
        "apple_pos": np.zeros((n_frames, 3), dtype=np.float32),
        "woody_part_start_pos": {
            "joint_a": np.zeros((n_frames, 3), dtype=np.float32),
        },
        "woody_part_end_pos": {
            "joint_a": np.zeros((n_frames, 3), dtype=np.float32),
        },
        "stable": np.ones(n_frames, dtype=bool),
    }


def _mock_collectors(*, num_candidates: int, num_directions: int) -> grid.BatchedSysIdReplayCollectors:
    recorded_by_env = [
        _dummy_recorded_episode(direction_idx=d)
        for _c in range(num_candidates)
        for d in range(num_directions)
    ]
    return grid.BatchedSysIdReplayCollectors(
        num_envs=len(recorded_by_env),
        recorded_by_env=recorded_by_env,
    )


def _wasserstein_result(
    *,
    candidate_index: int,
    aggregate: float,
    per_direction: dict[int, float] | None = None,
    missing_directions: tuple[int, ...] = (),
) -> WassersteinCandidateResult:
    per_dir = per_direction if per_direction is not None else {0: float(aggregate)}
    return WassersteinCandidateResult(
        candidate_index=int(candidate_index),
        stiffnesses={"primary_e_pa": 1.0, "spur_e_pa": 1.0, "stem_e_pa": 1.0},
        aggregate_sinkhorn=float(aggregate),
        per_direction_sinkhorn=per_dir,
        per_direction_n_transitions={k: 16 for k in per_dir},
        low_sample_directions=(),
        missing_directions=missing_directions,
    )


@pytest.fixture
def gt_params() -> fs.FruitingSystemParams:
    return _base_primary_spur_stem_with_secondary()


@pytest.fixture
def gt_candidate(gt_params: fs.FruitingSystemParams) -> cmaes.YoungsModulusCandidate:
    return cmaes.youngs_modulus_candidate_from_params(gt_params)


@pytest.fixture
def far_candidate(gt_candidate: cmaes.YoungsModulusCandidate) -> cmaes.YoungsModulusCandidate:
    return cmaes.YoungsModulusCandidate(
        primary=gt_candidate.primary * 10.0,
        spur=gt_candidate.spur * 10.0,
        stem=gt_candidate.stem * 10.0,
    )


@pytest.fixture
def unstable_candidate(gt_candidate: cmaes.YoungsModulusCandidate) -> cmaes.YoungsModulusCandidate:
    return cmaes.YoungsModulusCandidate(
        primary=gt_candidate.primary * 0.5,
        spur=gt_candidate.spur * 0.5,
        stem=gt_candidate.stem * 0.5,
    )


def _mock_gt_scoring_context(*, expected_directions: tuple[int, ...]) -> MagicMock:
    ctx = MagicMock()
    ctx.expected_directions = tuple(int(d) for d in expected_directions)
    return ctx


def _patch_evaluator_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gt_params: fs.FruitingSystemParams,
    num_candidates: int,
    num_directions: int,
    instability_by_candidate: dict[int, float] | None = None,
    sinkhorn_by_candidate: dict[int, float] | None = None,
    missing_directions_by_candidate: dict[int, tuple[int, ...]] | None = None,
    replay_call: dict | None = None,
    dataset: MagicMock | None = None,
    expected_directions: tuple[int, ...] | None = None,
) -> MagicMock:
    if dataset is None:
        dataset = MagicMock()
        dataset.episode_entries.return_value = [
            {"structure_idx": 0, "direction_idx": d} for d in range(num_directions)
        ]

    monkeypatch.setattr(cmaes, "true_params_for_structure", lambda _ds, _idx: gt_params)
    monkeypatch.setattr(
        cmaes,
        "gripper_proxy_from_episode_metadata",
        lambda _meta: fs.GripperProxyConfig(
            fix_to_apple=True,
            weld_direction=(1.0, 0.0, 0.0),
        ),
    )

    monkeypatch.setattr(
        cmaes,
        "load_recorded_episodes_for_structure",
        lambda *_args, **_kwargs: [
            _dummy_recorded_episode(direction_idx=i) for i in range(num_directions)
        ],
    )
    expected = (
        tuple(int(d) for d in expected_directions)
        if expected_directions is not None
        else tuple(range(int(num_directions)))
    )
    monkeypatch.setattr(
        cmaes,
        "prepare_gt_wasserstein_scoring_context",
        lambda *_args, **_kwargs: _mock_gt_scoring_context(expected_directions=expected),
    )

    def fake_replay(*, direction_indices=None, candidates=None, num_directions=None, **_kwargs):
        if replay_call is not None:
            replay_call["direction_indices"] = list(direction_indices or [])
            replay_call["num_directions"] = num_directions
        return _mock_collectors(
            num_candidates=len(list(candidates or [])),
            num_directions=num_directions,
        )

    monkeypatch.setattr(cmaes, "replay_candidates_for_structure", fake_replay)

    instability = instability_by_candidate or {}
    sinkhorn = sinkhorn_by_candidate or {}
    missing = missing_directions_by_candidate or {}

    def fake_direction_episodes(_collectors, *, candidate_index, num_directions):
        tagged = [
            _dummy_recorded_episode(direction_idx=dir_local)
            for dir_local in range(int(num_directions))
        ]
        for ep in tagged:
            ep["_candidate_index"] = int(candidate_index)
        return tagged

    monkeypatch.setattr(cmaes, "direction_episodes_from_collectors", fake_direction_episodes)

    def fake_instability(*, replay, recorded):
        del recorded
        env_idx = int(replay.get("_candidate_index", 0))
        return float(instability.get(env_idx, 0.0))

    def fake_score(*, candidate_index, replay_observations, **_kwargs):
        del replay_observations
        idx = int(candidate_index)
        missing_dirs = missing.get(idx, ())
        if missing_dirs:
            aggregate = float("nan")
            per_direction = {}
        else:
            aggregate = float(sinkhorn.get(idx, 1.0))
            per_direction = None
        return _wasserstein_result(
            candidate_index=idx,
            aggregate=aggregate,
            per_direction=per_direction,
            missing_directions=missing_dirs,
        )

    monkeypatch.setattr(cmaes, "replay_instability_fraction_all_frames", fake_instability)
    monkeypatch.setattr(cmaes, "score_candidate_wasserstein_complete", fake_score)
    return dataset


def test_evaluator_ranks_finite_eligible_scores_and_marks_gt(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
    gt_candidate: cmaes.YoungsModulusCandidate,
    far_candidate: cmaes.YoungsModulusCandidate,
    unstable_candidate: cmaes.YoungsModulusCandidate,
):
  # Candidate losses: far=3.0, GT=0.2, unstable=0.1 (disqualified for instability).
    _patch_evaluator_dependencies(
        monkeypatch,
        gt_params=gt_params,
        num_candidates=3,
        num_directions=2,
        instability_by_candidate={2: grid.UNSTABLE_DISQUALIFY_THRESHOLD + 0.05},
        sinkhorn_by_candidate={0: 3.0, 1: 0.2, 2: 0.1},
    )

    evaluation = cmaes.evaluate_youngs_modulus_candidates(
        dataset=MagicMock(),
        structure_idx=0,
        candidates=[far_candidate, gt_candidate, unstable_candidate],
        num_directions=2,
        build_env_fn=MagicMock(),
        scoring=cmaes.YoungsModulusScoringConfig(n_holds=5, n_directions=2),
    )

    assert [score.rank for score in evaluation.scores] == [2, 1, None]
    assert evaluation.scores[1].is_gt is True
    assert evaluation.scores[2].disqualification_reason == "replay_instability"


def test_evaluator_uses_strict_25_percent_instability_boundary(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
    gt_candidate: cmaes.YoungsModulusCandidate,
    unstable_candidate: cmaes.YoungsModulusCandidate,
):
    _patch_evaluator_dependencies(
        monkeypatch,
        gt_params=gt_params,
        num_candidates=2,
        num_directions=1,
        instability_by_candidate={0: 0.25, 1: 0.26},
        sinkhorn_by_candidate={0: 0.2, 1: 0.1},
    )

    evaluation = cmaes.evaluate_youngs_modulus_candidates(
        dataset=MagicMock(),
        structure_idx=0,
        candidates=[gt_candidate, unstable_candidate],
        num_directions=1,
        build_env_fn=MagicMock(),
        scoring=cmaes.YoungsModulusScoringConfig(n_holds=5, n_directions=1),
    )

    at_boundary, above_boundary = evaluation.scores
    assert at_boundary.instability_fraction == pytest.approx(0.25)
    assert at_boundary.disqualified is False
    assert at_boundary.rank == 1
    assert above_boundary.instability_fraction == pytest.approx(0.26)
    assert above_boundary.disqualified is True
    assert above_boundary.disqualification_reason == "replay_instability"
    assert above_boundary.rank is None


@pytest.mark.parametrize(
    (
        "configured_n_directions",
        "expected_scoring_n_directions",
        "pool_directions",
    ),
    [(None, 5, True), (3, 3, False)],
)
def test_evaluator_uses_source_direction_width_for_sparse_ids(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
    gt_candidate: cmaes.YoungsModulusCandidate,
    configured_n_directions: int | None,
    expected_scoring_n_directions: int,
    pool_directions: bool,
):
    replay_call: dict = {}
    dataset = MagicMock()
    dataset.episode_entries.return_value = [
        {"structure_idx": 0, "direction_idx": 0},
        {"structure_idx": 0, "direction_idx": 2},
    ]
    gt_context_calls: list[dict] = []
    score_calls: list[dict] = []

    monkeypatch.setattr(cmaes, "true_params_for_structure", lambda _ds, _idx: gt_params)
    monkeypatch.setattr(
        cmaes,
        "gripper_proxy_from_episode_metadata",
        lambda _meta: fs.GripperProxyConfig(
            fix_to_apple=True,
            weld_direction=(1.0, 0.0, 0.0),
        ),
    )
    monkeypatch.setattr(
        cmaes,
        "load_recorded_episodes_for_structure",
        lambda *_args, **_kwargs: [
            _dummy_recorded_episode(direction_idx=0),
            _dummy_recorded_episode(direction_idx=2),
        ],
    )

    def capture_gt_context(*_args, **kwargs):
        gt_context_calls.append(dict(kwargs))
        return _mock_gt_scoring_context(expected_directions=(0, 2))

    monkeypatch.setattr(cmaes, "prepare_gt_wasserstein_scoring_context", capture_gt_context)

    def fake_replay(*, direction_indices=None, candidates=None, num_directions=None, **_kwargs):
        replay_call["direction_indices"] = list(direction_indices or [])
        replay_call["num_directions"] = num_directions
        return _mock_collectors(num_candidates=len(list(candidates or [])), num_directions=2)

    monkeypatch.setattr(cmaes, "replay_candidates_for_structure", fake_replay)
    monkeypatch.setattr(
        cmaes,
        "direction_episodes_from_collectors",
        lambda _collectors, *, candidate_index, num_directions: [
            _dummy_recorded_episode(direction_idx=dir_local)
            for dir_local in range(int(num_directions))
        ],
    )
    monkeypatch.setattr(
        cmaes,
        "replay_instability_fraction_all_frames",
        lambda **_kwargs: 0.0,
    )
    monkeypatch.setattr(
        cmaes,
        "score_candidate_wasserstein_complete",
        lambda **kwargs: (
            score_calls.append(dict(kwargs))
            or _wasserstein_result(
                candidate_index=int(kwargs["candidate_index"]),
                aggregate=0.1,
            )
        ),
    )

    evaluation = cmaes.evaluate_youngs_modulus_candidates(
        dataset=dataset,
        structure_idx=0,
        candidates=[gt_candidate],
        num_directions=5,
        build_env_fn=MagicMock(),
        scoring=cmaes.YoungsModulusScoringConfig(
            n_holds=5,
            n_directions=configured_n_directions,
            pool_directions=pool_directions,
        ),
    )

    assert evaluation.direction_indices == (0, 2)
    assert replay_call["direction_indices"] == [0, 2]
    assert replay_call["num_directions"] == 2
    assert gt_context_calls[0]["n_directions"] == expected_scoring_n_directions
    assert "pool_directions" not in gt_context_calls[0]
    assert score_calls[0]["n_directions"] == expected_scoring_n_directions
    assert score_calls[0]["pool_directions"] is pool_directions


def test_evaluator_disqualifies_non_finite_sinkhorn(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
    gt_candidate: cmaes.YoungsModulusCandidate,
    far_candidate: cmaes.YoungsModulusCandidate,
):
    _patch_evaluator_dependencies(
        monkeypatch,
        gt_params=gt_params,
        num_candidates=2,
        num_directions=1,
        sinkhorn_by_candidate={0: 0.5, 1: float("nan")},
    )

    evaluation = cmaes.evaluate_youngs_modulus_candidates(
        dataset=MagicMock(),
        structure_idx=0,
        candidates=[gt_candidate, far_candidate],
        num_directions=1,
        build_env_fn=MagicMock(),
        scoring=cmaes.YoungsModulusScoringConfig(n_holds=5, n_directions=1),
    )

    assert evaluation.scores[0].rank == 1
    assert evaluation.scores[1].disqualified is True
    assert evaluation.scores[1].disqualification_reason == "non_finite_sinkhorn"
    assert evaluation.scores[1].rank is None


def test_evaluator_disqualifies_missing_direction_bags(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
    gt_candidate: cmaes.YoungsModulusCandidate,
    far_candidate: cmaes.YoungsModulusCandidate,
):
    _patch_evaluator_dependencies(
        monkeypatch,
        gt_params=gt_params,
        num_candidates=2,
        num_directions=2,
        sinkhorn_by_candidate={0: 0.3, 1: 0.1},
        missing_directions_by_candidate={1: (1,)},
    )

    evaluation = cmaes.evaluate_youngs_modulus_candidates(
        dataset=MagicMock(),
        structure_idx=0,
        candidates=[gt_candidate, far_candidate],
        num_directions=2,
        build_env_fn=MagicMock(),
        scoring=cmaes.YoungsModulusScoringConfig(n_holds=5, n_directions=2),
    )

    assert evaluation.scores[0].rank == 1
    assert evaluation.scores[1].disqualified is True
    assert evaluation.scores[1].disqualification_reason == "missing_directions"
    assert evaluation.scores[1].rank is None
    assert not math.isfinite(evaluation.scores[1].aggregate_sinkhorn)


def test_evaluator_empty_candidate_does_not_abort_structure(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
    gt_candidate: cmaes.YoungsModulusCandidate,
    far_candidate: cmaes.YoungsModulusCandidate,
    unstable_candidate: cmaes.YoungsModulusCandidate,
):
    """One empty transition bag disqualifies only that candidate."""
    _patch_evaluator_dependencies(
        monkeypatch,
        gt_params=gt_params,
        num_candidates=3,
        num_directions=2,
        sinkhorn_by_candidate={0: 0.4, 1: 0.1, 2: 0.2},
        missing_directions_by_candidate={1: (0, 1)},
    )

    evaluation = cmaes.evaluate_youngs_modulus_candidates(
        dataset=MagicMock(),
        structure_idx=0,
        candidates=[far_candidate, unstable_candidate, gt_candidate],
        num_directions=2,
        build_env_fn=MagicMock(),
        scoring=cmaes.YoungsModulusScoringConfig(n_holds=5, n_directions=2),
    )

    assert evaluation.scores[0].rank is not None
    assert evaluation.scores[1].rank is None
    assert evaluation.scores[1].disqualified is True
    assert evaluation.scores[1].disqualification_reason == "empty_transition_bag"
    assert not math.isfinite(evaluation.scores[1].aggregate_sinkhorn)
    assert evaluation.scores[2].rank is not None


def test_evaluator_replay_instability_precedes_empty_transition_bag(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
    gt_candidate: cmaes.YoungsModulusCandidate,
):
    _patch_evaluator_dependencies(
        monkeypatch,
        gt_params=gt_params,
        num_candidates=1,
        num_directions=2,
        instability_by_candidate={0: grid.UNSTABLE_DISQUALIFY_THRESHOLD + 0.01},
        missing_directions_by_candidate={0: (0, 1)},
    )

    evaluation = cmaes.evaluate_youngs_modulus_candidates(
        dataset=MagicMock(),
        structure_idx=0,
        candidates=[gt_candidate],
        num_directions=2,
        build_env_fn=MagicMock(),
        scoring=cmaes.YoungsModulusScoringConfig(n_holds=5, n_directions=2),
    )

    assert evaluation.scores[0].disqualification_reason == "replay_instability"
    assert not math.isfinite(evaluation.scores[0].aggregate_sinkhorn)


def test_evaluator_missing_directions_precedes_non_finite_sinkhorn(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
    gt_candidate: cmaes.YoungsModulusCandidate,
):
    _patch_evaluator_dependencies(
        monkeypatch,
        gt_params=gt_params,
        num_candidates=1,
        num_directions=2,
        missing_directions_by_candidate={0: (1,)},
    )

    evaluation = cmaes.evaluate_youngs_modulus_candidates(
        dataset=MagicMock(),
        structure_idx=0,
        candidates=[gt_candidate],
        num_directions=2,
        build_env_fn=MagicMock(),
        scoring=cmaes.YoungsModulusScoringConfig(n_holds=5, n_directions=2),
    )

    assert evaluation.scores[0].disqualification_reason == "missing_directions"
    assert not math.isfinite(evaluation.scores[0].aggregate_sinkhorn)


def test_evaluator_rejects_mismatched_scorer_candidate_index(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
    gt_candidate: cmaes.YoungsModulusCandidate,
):
    _patch_evaluator_dependencies(
        monkeypatch,
        gt_params=gt_params,
        num_candidates=1,
        num_directions=1,
    )
    monkeypatch.setattr(
        cmaes,
        "score_candidate_wasserstein_complete",
        lambda **_kwargs: _wasserstein_result(candidate_index=7, aggregate=0.1),
    )

    with pytest.raises(
        RuntimeError,
        match=r"candidate index mismatch.*expected 0, got 7",
    ):
        cmaes.evaluate_youngs_modulus_candidates(
            dataset=MagicMock(),
            structure_idx=0,
            candidates=[gt_candidate],
            num_directions=1,
            build_env_fn=MagicMock(),
            scoring=cmaes.YoungsModulusScoringConfig(n_holds=5, n_directions=1),
        )


def test_evaluator_breaks_ties_by_candidate_index(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
):
    cand_a = cmaes.YoungsModulusCandidate(1.0e8, 1.0e7, 1.0e7)
    cand_b = cmaes.YoungsModulusCandidate(2.0e8, 2.0e7, 2.0e7)
    _patch_evaluator_dependencies(
        monkeypatch,
        gt_params=gt_params,
        num_candidates=2,
        num_directions=1,
        sinkhorn_by_candidate={0: 0.5, 1: 0.5},
    )

    evaluation = cmaes.evaluate_youngs_modulus_candidates(
        dataset=MagicMock(),
        structure_idx=0,
        candidates=[cand_a, cand_b],
        num_directions=1,
        build_env_fn=MagicMock(),
        scoring=cmaes.YoungsModulusScoringConfig(n_holds=5, n_directions=1),
    )

    assert [score.rank for score in evaluation.scores] == [1, 2]
    assert math.isclose(
        evaluation.scores[0].aggregate_sinkhorn,
        evaluation.scores[1].aggregate_sinkhorn,
    )


def test_evaluator_records_fixed_secondary_and_applied_params(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
    gt_candidate: cmaes.YoungsModulusCandidate,
):
    _patch_evaluator_dependencies(
        monkeypatch,
        gt_params=gt_params,
        num_candidates=1,
        num_directions=1,
        sinkhorn_by_candidate={0: 0.1},
    )

    evaluation = cmaes.evaluate_youngs_modulus_candidates(
        dataset=MagicMock(),
        structure_idx=0,
        candidates=[gt_candidate],
        num_directions=1,
        build_env_fn=MagicMock(),
        scoring=cmaes.YoungsModulusScoringConfig(n_holds=5, n_directions=1),
    )

    assert gt_params.secondary is not None
    assert evaluation.fixed_secondary_e_pa == pytest.approx(
        gt_params.secondary.youngs_modulus_pa
    )
    assert len(evaluation.applied_params) == 1
    assert evaluation.applied_params[0].secondary is not None
    assert evaluation.applied_params[0].secondary.youngs_modulus_pa == pytest.approx(
        gt_params.secondary.youngs_modulus_pa
    )
    assert evaluation.applied_params[0].primary is not None
    assert evaluation.applied_params[0].primary.youngs_modulus_pa == pytest.approx(
        gt_candidate.primary
    )


def _frozen_replay_collectors(episodes: list[dict]) -> grid.BatchedSysIdReplayCollectors:
    """Collectors backed by pre-built arrays (skip live record_step)."""
    out = grid.BatchedSysIdReplayCollectors.__new__(grid.BatchedSysIdReplayCollectors)
    out._recorded_by_env = list(episodes)
    out._collectors = [grid._FrozenReplayCollector(dict(ep)) for ep in episodes]
    return out


def _sentinel_episode(*, physical_dir: int, sentinel: float, n_frames: int = 4) -> dict:
    ep = _dummy_recorded_episode(direction_idx=int(physical_dir), n_frames=n_frames)
    ep["ft_wrist"] = np.full((n_frames, 6), float(sentinel), dtype=np.float32)
    return ep


def test_evaluator_forced_chunk_sentinel_routing_preserves_candidate_major_layout(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
):
    """Chunked replay concat + candidate-major flatten keeps score provenance."""
    candidates = [
        cmaes.YoungsModulusCandidate(1.0e8, 1.0e7, 1.0e7),
        cmaes.YoungsModulusCandidate(2.0e8, 2.0e7, 2.0e7),
        cmaes.YoungsModulusCandidate(3.0e8, 3.0e7, 3.0e7),
    ]
    dataset = MagicMock()
    dataset.episode_entries.return_value = [
        {"structure_idx": 0, "direction_idx": 0},
        {"structure_idx": 0, "direction_idx": 2},
    ]
    score_sentinels: dict[int, list[float]] = {}
    chunk_ordinal = {"n": 0}

    monkeypatch.setattr(cmaes, "true_params_for_structure", lambda _ds, _idx: gt_params)
    monkeypatch.setattr(
        cmaes,
        "gripper_proxy_from_episode_metadata",
        lambda _meta: fs.GripperProxyConfig(
            fix_to_apple=True,
            weld_direction=(1.0, 0.0, 0.0),
        ),
    )
    monkeypatch.setattr(
        cmaes,
        "load_recorded_episodes_for_structure",
        lambda *_args, **_kwargs: [
            _dummy_recorded_episode(direction_idx=0),
            _dummy_recorded_episode(direction_idx=2),
        ],
    )
    monkeypatch.setattr(
        cmaes,
        "prepare_gt_wasserstein_scoring_context",
        lambda *_args, **_kwargs: _mock_gt_scoring_context(expected_directions=(0, 2)),
    )
    monkeypatch.setattr(
        cmaes,
        "replay_instability_fraction_all_frames",
        lambda **_kwargs: 0.0,
    )

    def fake_replay_batched_sysid_structure(*, candidates, direction_indices=None, **_kwargs):
        chunk = list(candidates)
        assert len(chunk) == 1, "max_envs_per_batch=2 with 2 dirs forces one candidate/chunk"
        global_candidate_index = int(chunk_ordinal["n"])
        chunk_ordinal["n"] += 1
        dirs = [int(d) for d in (direction_indices or [])]
        assert dirs == [0, 2]
        episodes = [
            _sentinel_episode(
                physical_dir=source_direction_id,
                sentinel=100.0 * global_candidate_index + float(source_direction_id),
            )
            for source_direction_id in dirs
        ]
        return _frozen_replay_collectors(episodes)

    monkeypatch.setattr(grid, "replay_batched_sysid_structure", fake_replay_batched_sysid_structure)

    def fake_score(*, candidate_index, replay_observations, **_kwargs):
        idx = int(candidate_index)
        score_sentinels[idx] = [
            float(np.asarray(ep["ft_wrist"])[0, 0]) for ep in replay_observations
        ]
        return _wasserstein_result(
            candidate_index=idx,
            aggregate=0.1 * (idx + 1),
        )

    monkeypatch.setattr(cmaes, "score_candidate_wasserstein_complete", fake_score)

    evaluation = cmaes.evaluate_youngs_modulus_candidates(
        dataset=dataset,
        structure_idx=0,
        candidates=candidates,
        num_directions=5,
        build_env_fn=MagicMock(),
        scoring=cmaes.YoungsModulusScoringConfig(n_holds=5, n_directions=5),
        max_envs_per_batch=2,
    )

    assert evaluation.direction_indices == (0, 2)
    assert chunk_ordinal["n"] == 3

    expected_by_candidate = {
        0: [0.0, 2.0],
        1: [100.0, 102.0],
        2: [200.0, 202.0],
    }
    for candidate_index, expected_sentinels in expected_by_candidate.items():
        episodes = evaluation.replay_episodes[candidate_index]
        dirs = [int(np.asarray(ep["dir_idx"])[0]) for ep in episodes]
        sentinels = [float(np.asarray(ep["ft_wrist"])[0, 0]) for ep in episodes]
        assert dirs == [0, 2]
        assert sentinels == expected_sentinels
        assert score_sentinels[candidate_index] == expected_sentinels
        assert evaluation.scores[candidate_index].candidate_index == candidate_index


def _prepared_structure(
    *,
    structure_idx: int,
    candidates: tuple[cmaes.YoungsModulusCandidate, ...],
    directions: tuple[int, ...],
    gt_params: fs.FruitingSystemParams,
) -> object:
    recorded = tuple(
        _dummy_recorded_episode(direction_idx=direction_idx)
        for direction_idx in directions
    )
    return cmaes.PreparedYoungsModulusStructure(
        replay_request=multi.ReplayStructureRequest(
            structure_idx=structure_idx,
            candidates=candidates,
            direction_indices=directions,
            base_params=gt_params,
            recorded_by_direction=dict(zip(directions, recorded, strict=True)),
            gripper=fs.GripperProxyConfig(
                fix_to_apple=True,
                weld_direction=(1.0, 0.0, 0.0),
            ),
        ),
        candidates=candidates,
        gt_candidate=cmaes.youngs_modulus_candidate_from_params(gt_params),
        fixed_secondary_e_pa=(
            None
            if gt_params.secondary is None
            else float(gt_params.secondary.youngs_modulus_pa)
        ),
        direction_indices=directions,
        recorded_episodes=recorded,
        gt_context=_mock_gt_scoring_context(expected_directions=directions),
        scoring_n_directions=max(directions) + 1,
    )


def test_prepare_vic_pose_dataset_uses_real_gripper_and_skips_gt(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
):
    candidate = cmaes.SupportKpYoungsCandidate(1.0e4, 1.0e9, 1.0e9)
    recorded = _dummy_recorded_episode(direction_idx=0)
    meta = {
        "initial_apple_pos": [0.0, 0.0, 0.0],
        "initial_apple_quat": [0.0, 0.0, 0.0, 1.0],
        "initial_tcp_pos": [0.01, 0.02, 0.03],
        "initial_tcp_quat": [0.0, 0.0, 0.0, 1.0],
    }
    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "action_layout": "vic_pose_v1",
            "action_dim": 19,
            "num_directions": 1,
            "seed": 0,
        }
    }
    dataset.load_episode_metadata.return_value = meta
    monkeypatch.setattr(cmaes, "resolve_direction_indices", lambda *_a, **_k: [0])
    monkeypatch.setattr(
        cmaes,
        "load_recorded_episodes_for_structure",
        lambda *_a, **_k: [recorded],
    )
    monkeypatch.setattr(
        cmaes,
        "prepare_gt_wasserstein_scoring_context",
        lambda *_a, **_k: MagicMock(),
    )
    monkeypatch.setattr(cmaes, "true_params_for_structure", lambda *_a, **_k: gt_params)

    prepared = cmaes.prepare_youngs_modulus_structure(
        dataset=dataset,
        structure_idx=0,
        candidates=(candidate,),
        num_directions=1,
        scoring=cmaes.YoungsModulusScoringConfig(
            use_median=True,
            hold_id_onehot=False,
            pool_directions=True,
            n_holds=1,
            n_directions=1,
            device="cpu",
        ),
    )

    assert prepared.gt_candidate is None
    assert (
        prepared.replay_request.gripper.weld_proxy_offset_in_apple_frame
        is not None
    )


def test_score_is_gt_false_when_gt_candidate_is_none(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
):
    candidate = cmaes.YoungsModulusCandidate(1.0e8, 1.0e7, 1.0e7)
    prepared = dataclasses.replace(
        _prepared_structure(
            structure_idx=0,
            candidates=(candidate,),
            directions=(0,),
            gt_params=gt_params,
        ),
        gt_candidate=None,
    )
    replay_by_key = {
        multi.ReplaySlotKey(0, 0, 0): _dummy_recorded_episode(direction_idx=0)
    }
    monkeypatch.setattr(
        cmaes,
        "replay_instability_fraction_all_frames",
        lambda **_kwargs: 0.0,
    )
    monkeypatch.setattr(
        cmaes,
        "score_candidate_wasserstein_complete",
        lambda **_kwargs: _wasserstein_result(candidate_index=0, aggregate=0.1),
    )

    evaluation = cmaes.score_prepared_youngs_modulus_structure(
        prepared,
        replay_by_key=replay_by_key,
        scoring=cmaes.YoungsModulusScoringConfig(n_directions=1),
    )

    assert evaluation.gt_candidate is None
    assert evaluation.scores[0].is_gt is False


def test_score_prepared_disqualifies_candidate_with_invalid_replay_features(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
):
    candidate = cmaes.YoungsModulusCandidate(1.0e8, 1.0e7, 1.0e7)
    prepared = _prepared_structure(
        structure_idx=0,
        candidates=(candidate,),
        directions=(0,),
        gt_params=gt_params,
    )
    broken = _dummy_recorded_episode(direction_idx=0)

    def _not_a_mapping() -> None:
        pass

    broken["woody_part_start_pos"] = _not_a_mapping
    replay_by_key = {multi.ReplaySlotKey(0, 0, 0): broken}
    monkeypatch.setattr(
        cmaes,
        "replay_instability_fraction_all_frames",
        lambda **_kwargs: 0.0,
    )

    evaluation = cmaes.score_prepared_youngs_modulus_structure(
        prepared,
        replay_by_key=replay_by_key,
        scoring=cmaes.YoungsModulusScoringConfig(
            n_directions=1,
            hold_aggregation="mean",
        ),
    )

    assert evaluation.scores[0].disqualified is True
    assert evaluation.scores[0].disqualification_reason.startswith(
        "invalid_replay_features:"
    )
    assert evaluation.scores[0].mean_hold_force_err_n is None


def test_scalar_evaluation_uses_resolved_action_dim(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
):
    candidate = cmaes.YoungsModulusCandidate(1.0e8, 1.0e7, 1.0e7)
    prepared = _prepared_structure(
        structure_idx=0,
        candidates=(candidate,),
        directions=(0,),
        gt_params=gt_params,
    )
    dataset = MagicMock()
    dataset.manifest = {"collection": {}}
    replay_call: dict[str, object] = {}
    sentinel = MagicMock()
    monkeypatch.setattr(
        cmaes,
        "prepare_youngs_modulus_structure",
        lambda **_kwargs: prepared,
    )

    def fake_replay(**kwargs):
        replay_call.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(cmaes, "replay_candidates_for_structure", fake_replay)
    monkeypatch.setattr(
        cmaes,
        "direction_episodes_from_collectors",
        lambda *_a, **_k: [_dummy_recorded_episode(direction_idx=0)],
    )
    monkeypatch.setattr(
        cmaes,
        "score_prepared_youngs_modulus_structure",
        lambda *_a, **_k: sentinel,
    )

    result = cmaes.evaluate_youngs_modulus_candidates(
        dataset=dataset,
        structure_idx=0,
        candidates=(candidate,),
        num_directions=1,
        build_env_fn=MagicMock(),
        scoring=cmaes.YoungsModulusScoringConfig(n_directions=1),
        action_dim=19,
    )

    assert result is sentinel
    assert replay_call["action_dim"] == 19


def test_score_prepared_routes_original_structure_local_candidate_and_physical_direction(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
):
    candidate_4a = cmaes.YoungsModulusCandidate(1.0e8, 1.0e7, 1.0e7)
    candidate_4b = cmaes.YoungsModulusCandidate(2.0e8, 2.0e7, 2.0e7)
    candidate_1 = cmaes.YoungsModulusCandidate(3.0e8, 3.0e7, 3.0e7)
    prepared_4 = _prepared_structure(
        structure_idx=4,
        candidates=(candidate_4a, candidate_4b),
        directions=(2, 0),
        gt_params=gt_params,
    )
    prepared_1 = _prepared_structure(
        structure_idx=1,
        candidates=(candidate_1,),
        directions=(2, 0),
        gt_params=gt_params,
    )
    replay_by_key = {
        multi.ReplaySlotKey(4, 1, 0): _sentinel_episode(physical_dir=0, sentinel=410),
        multi.ReplaySlotKey(1, 0, 0): _sentinel_episode(physical_dir=0, sentinel=10),
        multi.ReplaySlotKey(4, 0, 2): _sentinel_episode(physical_dir=2, sentinel=402),
        multi.ReplaySlotKey(4, 1, 2): _sentinel_episode(physical_dir=2, sentinel=412),
        multi.ReplaySlotKey(1, 0, 2): _sentinel_episode(physical_dir=2, sentinel=12),
        multi.ReplaySlotKey(4, 0, 0): _sentinel_episode(physical_dir=0, sentinel=400),
    }
    observed: dict[tuple[int, int], list[float]] = {}
    monkeypatch.setattr(
        cmaes,
        "replay_instability_fraction_all_frames",
        lambda **_kwargs: 0.0,
    )

    def fake_score(*, candidate_index, replay_observations, **_kwargs):
        structure_idx = 1 if replay_observations[0]["ft_wrist"][0, 0] < 100 else 4
        observed[(structure_idx, int(candidate_index))] = [
            float(ep["ft_wrist"][0, 0]) for ep in replay_observations
        ]
        return _wasserstein_result(candidate_index=int(candidate_index), aggregate=0.1)

    monkeypatch.setattr(cmaes, "score_candidate_wasserstein_complete", fake_score)

    evaluation_1 = cmaes.score_prepared_youngs_modulus_structure(
        prepared_1,
        replay_by_key=replay_by_key,
        scoring=cmaes.YoungsModulusScoringConfig(n_directions=5),
    )
    evaluation_4 = cmaes.score_prepared_youngs_modulus_structure(
        prepared_4,
        replay_by_key=replay_by_key,
        scoring=cmaes.YoungsModulusScoringConfig(n_directions=5),
    )

    assert observed == {
        (1, 0): [12.0, 10.0],
        (4, 0): [402.0, 400.0],
        (4, 1): [412.0, 410.0],
    }
    assert [score.candidate_index for score in evaluation_1.scores] == [0]
    assert [score.candidate_index for score in evaluation_4.scores] == [0, 1]
    assert evaluation_1.direction_indices == (2, 0)
    assert evaluation_4.direction_indices == (2, 0)


def test_score_prepared_rejects_wasserstein_candidate_index_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
):
    candidate = cmaes.youngs_modulus_candidate_from_params(gt_params)
    prepared = _prepared_structure(
        structure_idx=4,
        candidates=(candidate,),
        directions=(2, 0),
        gt_params=gt_params,
    )
    replay_by_key = {
        multi.ReplaySlotKey(4, 0, direction_idx): _dummy_recorded_episode(
            direction_idx=direction_idx
        )
        for direction_idx in (2, 0)
    }
    monkeypatch.setattr(
        cmaes,
        "replay_instability_fraction_all_frames",
        lambda **_kwargs: 0.0,
    )
    monkeypatch.setattr(
        cmaes,
        "score_candidate_wasserstein_complete",
        lambda **_kwargs: _wasserstein_result(candidate_index=7, aggregate=0.1),
    )

    with pytest.raises(RuntimeError, match=r"candidate index mismatch.*expected 0, got 7"):
        cmaes.score_prepared_youngs_modulus_structure(
            prepared,
            replay_by_key=replay_by_key,
            scoring=cmaes.YoungsModulusScoringConfig(n_directions=5),
        )


def test_evaluate_multi_structure_fuses_in_requested_order_and_scores_local_candidates(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
):
    structures = (
        (
            4,
            (
                cmaes.YoungsModulusCandidate(1.0e8, 1.0e7, 1.0e7),
                cmaes.YoungsModulusCandidate(2.0e8, 2.0e7, 2.0e7),
            ),
        ),
        (1, (cmaes.YoungsModulusCandidate(3.0e8, 3.0e7, 3.0e7),)),
    )
    prepared = {
        structure_idx: _prepared_structure(
            structure_idx=structure_idx,
            candidates=tuple(candidates),
            directions=(2, 0),
            gt_params=gt_params,
        )
        for structure_idx, candidates in structures
    }
    monkeypatch.setattr(
        cmaes,
        "prepare_youngs_modulus_structure",
        lambda **kwargs: prepared[int(kwargs["structure_idx"])],
    )
    blocks = multi.build_replay_candidate_blocks(
        tuple(item.replay_request for item in prepared.values())
    )
    replay_by_key = {
        slot.key: _dummy_recorded_episode(direction_idx=slot.key.direction_idx)
        for block in blocks
        for slot in block.slots
    }
    diagnostics = multi.MultiStructureReplayDiagnostics(
        candidate_blocks=3,
        flattened_envs=6,
        chunk_env_counts=(6,),
        failed_chunk_indices=(),
        build_seconds=0.1,
        replay_seconds=0.2,
    )
    replay_calls: list[tuple[multi.ReplayCandidateBlock, ...]] = []

    def fake_replay(**kwargs):
        replay_calls.append(tuple(kwargs["blocks"]))
        return multi.MultiStructureReplayOutcome(
            replay_by_key=replay_by_key,
            failed_structures={},
            diagnostics=diagnostics,
        )

    monkeypatch.setattr(cmaes, "replay_multi_structure_candidate_blocks", fake_replay)
    monkeypatch.setattr(
        cmaes,
        "replay_instability_fraction_all_frames",
        lambda **_kwargs: 0.0,
    )
    monkeypatch.setattr(
        cmaes,
        "score_candidate_wasserstein_complete",
        lambda **kwargs: _wasserstein_result(
            candidate_index=int(kwargs["candidate_index"]),
            aggregate=float(kwargs["candidate_index"] + 1),
        ),
    )

    batch = cmaes.evaluate_youngs_modulus_structures(
        dataset=MagicMock(),
        structures=structures,
        num_directions=5,
        build_env_fn=MagicMock(),
        scoring=cmaes.YoungsModulusScoringConfig(n_directions=5),
        max_envs_per_batch=100,
        seed=7,
    )

    assert tuple(batch.evaluations) == (4, 1)
    assert [len(batch.evaluations[idx].scores) for idx in (4, 1)] == [2, 1]
    assert batch.errors == {}
    assert batch.retried_structures == ()
    assert len(replay_calls) == 1
    assert [(b.structure_idx, b.local_candidate_idx) for b in replay_calls[0]] == [
        (4, 0),
        (4, 1),
        (1, 0),
    ]


def test_evaluate_multi_structure_fusion_incompatibility_falls_back_without_errors(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
):
    structures = (
        (4, (cmaes.YoungsModulusCandidate(1.0e8, 1.0e7, 1.0e7),)),
        (1, (cmaes.YoungsModulusCandidate(2.0e8, 2.0e7, 2.0e7),)),
    )
    prepared = {
        structure_idx: _prepared_structure(
            structure_idx=structure_idx,
            candidates=tuple(candidates),
            directions=(2, 0) if structure_idx == 4 else (0, 2),
            gt_params=gt_params,
        )
        for structure_idx, candidates in structures
    }
    monkeypatch.setattr(
        cmaes,
        "prepare_youngs_modulus_structure",
        lambda **kwargs: prepared[int(kwargs["structure_idx"])],
    )
    replay_fused = MagicMock(side_effect=AssertionError("fused replay must not run"))
    monkeypatch.setattr(cmaes, "replay_multi_structure_candidate_blocks", replay_fused)
    scalar_calls: list[int] = []

    def fake_scalar(**kwargs):
        structure_idx = int(kwargs["structure_idx"])
        assert kwargs["action_dim"] == 19
        scalar_calls.append(structure_idx)
        item = prepared[structure_idx]
        return cmaes.YoungsModulusEvaluation(
            structure_idx=structure_idx,
            gt_candidate=item.gt_candidate,
            fixed_secondary_e_pa=item.fixed_secondary_e_pa,
            direction_indices=item.direction_indices,
            scores=[],
            replay_episodes=[],
            applied_params=[],
        )

    monkeypatch.setattr(cmaes, "evaluate_youngs_modulus_candidates", fake_scalar)

    batch = cmaes.evaluate_youngs_modulus_structures(
        dataset=MagicMock(),
        structures=structures,
        num_directions=5,
        build_env_fn=MagicMock(),
        scoring=cmaes.YoungsModulusScoringConfig(n_directions=5),
        action_dim=19,
    )

    assert tuple(batch.evaluations) == (4, 1)
    assert batch.errors == {}
    assert batch.retried_structures == (4, 1)
    assert scalar_calls == [4, 1]
    replay_fused.assert_not_called()


def test_evaluate_multi_structure_runtime_failure_retries_only_failed_structure(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
):
    structures = (
        (4, (cmaes.YoungsModulusCandidate(1.0e8, 1.0e7, 1.0e7),)),
        (1, (cmaes.YoungsModulusCandidate(2.0e8, 2.0e7, 2.0e7),)),
    )
    prepared = {
        structure_idx: _prepared_structure(
            structure_idx=structure_idx,
            candidates=tuple(candidates),
            directions=(2, 0),
            gt_params=gt_params,
        )
        for structure_idx, candidates in structures
    }
    monkeypatch.setattr(
        cmaes,
        "prepare_youngs_modulus_structure",
        lambda **kwargs: prepared[int(kwargs["structure_idx"])],
    )
    replay_for_1 = {
        multi.ReplaySlotKey(1, 0, direction_idx): _dummy_recorded_episode(
            direction_idx=direction_idx
        )
        for direction_idx in (2, 0)
    }
    monkeypatch.setattr(
        cmaes,
        "replay_multi_structure_candidate_blocks",
        lambda **_kwargs: multi.MultiStructureReplayOutcome(
            replay_by_key=replay_for_1,
            failed_structures={4: "chunk 0: synthetic failure"},
            diagnostics=multi.MultiStructureReplayDiagnostics(
                candidate_blocks=2,
                flattened_envs=4,
                chunk_env_counts=(2, 2),
                failed_chunk_indices=(0,),
                build_seconds=0.1,
                replay_seconds=0.2,
            ),
        ),
    )
    monkeypatch.setattr(
        cmaes,
        "replay_instability_fraction_all_frames",
        lambda **_kwargs: 0.0,
    )
    monkeypatch.setattr(
        cmaes,
        "score_candidate_wasserstein_complete",
        lambda **kwargs: _wasserstein_result(
            candidate_index=int(kwargs["candidate_index"]),
            aggregate=0.1,
        ),
    )
    scalar_calls: list[int] = []

    def fake_scalar(**kwargs):
        structure_idx = int(kwargs["structure_idx"])
        scalar_calls.append(structure_idx)
        item = prepared[structure_idx]
        return cmaes.YoungsModulusEvaluation(
            structure_idx=structure_idx,
            gt_candidate=item.gt_candidate,
            fixed_secondary_e_pa=item.fixed_secondary_e_pa,
            direction_indices=item.direction_indices,
            scores=[],
            replay_episodes=[],
            applied_params=[],
        )

    monkeypatch.setattr(cmaes, "evaluate_youngs_modulus_candidates", fake_scalar)

    batch = cmaes.evaluate_youngs_modulus_structures(
        dataset=MagicMock(),
        structures=structures,
        num_directions=5,
        build_env_fn=MagicMock(),
        scoring=cmaes.YoungsModulusScoringConfig(n_directions=5),
    )

    assert tuple(batch.evaluations) == (4, 1)
    assert batch.errors == {}
    assert batch.retried_structures == (4,)
    assert scalar_calls == [4]


def test_evaluate_multi_structure_records_preparation_error_and_fail_fast_raises(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
):
    candidate = cmaes.YoungsModulusCandidate(1.0e8, 1.0e7, 1.0e7)
    prepared_1 = _prepared_structure(
        structure_idx=1,
        candidates=(candidate,),
        directions=(2, 0),
        gt_params=gt_params,
    )

    def fake_prepare(**kwargs):
        if int(kwargs["structure_idx"]) == 4:
            raise ValueError("malformed structure 4")
        return prepared_1

    monkeypatch.setattr(cmaes, "prepare_youngs_modulus_structure", fake_prepare)
    monkeypatch.setattr(
        cmaes,
        "replay_multi_structure_candidate_blocks",
        lambda **kwargs: multi.MultiStructureReplayOutcome(
            replay_by_key={
                slot.key: _dummy_recorded_episode(direction_idx=slot.key.direction_idx)
                for block in kwargs["blocks"]
                for slot in block.slots
            },
            failed_structures={},
            diagnostics=multi.MultiStructureReplayDiagnostics(
                candidate_blocks=1,
                flattened_envs=2,
                chunk_env_counts=(2,),
                failed_chunk_indices=(),
                build_seconds=0.1,
                replay_seconds=0.2,
            ),
        ),
    )
    monkeypatch.setattr(
        cmaes,
        "replay_instability_fraction_all_frames",
        lambda **_kwargs: 0.0,
    )
    monkeypatch.setattr(
        cmaes,
        "score_candidate_wasserstein_complete",
        lambda **kwargs: _wasserstein_result(
            candidate_index=int(kwargs["candidate_index"]),
            aggregate=0.1,
        ),
    )

    batch = cmaes.evaluate_youngs_modulus_structures(
        dataset=MagicMock(),
        structures=((4, (candidate,)), (1, (candidate,))),
        num_directions=5,
        build_env_fn=MagicMock(),
        scoring=cmaes.YoungsModulusScoringConfig(n_directions=5),
    )

    assert tuple(batch.evaluations) == (1,)
    assert tuple(batch.errors) == (4,)
    # Prepare-stage errors must carry a traceback, like replay/scoring errors do:
    # a bare message leaves an instant wave failure undiagnosable.
    assert batch.errors[4].startswith("malformed structure 4")
    assert "Traceback (most recent call last)" in batch.errors[4]
    assert "fake_prepare" in batch.errors[4]

    with pytest.raises(ValueError, match="malformed structure 4"):
        cmaes.evaluate_youngs_modulus_structures(
            dataset=MagicMock(),
            structures=((4, (candidate,)), (1, (candidate,))),
            num_directions=5,
            build_env_fn=MagicMock(),
            scoring=cmaes.YoungsModulusScoringConfig(n_directions=5),
            fail_fast=True,
        )
