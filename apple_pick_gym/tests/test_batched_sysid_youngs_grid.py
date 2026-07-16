"""Tests for Young's-modulus candidate evaluation and ranking."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from apple_pick_sim.fruiting_system import params as fs
from apple_pick_sim.system_id.wasserstein import WassersteinCandidateResult
from apple_pick_sim.tests.conftest import RANGES_FIXTURE
from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes
from apple_pick_gym.batched_envs import batched_sysid_mmd_grid as grid


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
) -> MagicMock:
    if dataset is None:
        dataset = MagicMock()
        dataset.episode_entries.return_value = [
            {"structure_idx": 0, "direction_idx": d} for d in range(num_directions)
        ]

    monkeypatch.setattr(cmaes, "true_params_for_structure", lambda _ds, _idx: gt_params)

    monkeypatch.setattr(
        cmaes,
        "load_recorded_episodes_for_structure",
        lambda *_args, **_kwargs: [
            _dummy_recorded_episode(direction_idx=i) for i in range(num_directions)
        ],
    )
    monkeypatch.setattr(
        cmaes,
        "prepare_gt_wasserstein_context",
        lambda *_args, **_kwargs: {0: MagicMock()},
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
        aggregate = float(sinkhorn.get(idx, 1.0))
        return _wasserstein_result(
            candidate_index=idx,
            aggregate=aggregate,
            missing_directions=missing.get(idx, ()),
        )

    monkeypatch.setattr(cmaes, "replay_instability_fraction_all_frames", fake_instability)
    monkeypatch.setattr(cmaes, "score_candidate_wasserstein", fake_score)
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


@pytest.mark.parametrize(
    ("configured_n_directions", "expected_scoring_n_directions"),
    [(None, 5), (3, 3)],
)
def test_evaluator_uses_source_direction_width_for_sparse_ids(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
    gt_candidate: cmaes.YoungsModulusCandidate,
    configured_n_directions: int | None,
    expected_scoring_n_directions: int,
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
        "load_recorded_episodes_for_structure",
        lambda *_args, **_kwargs: [
            _dummy_recorded_episode(direction_idx=0),
            _dummy_recorded_episode(direction_idx=2),
        ],
    )

    def capture_gt_context(*_args, **kwargs):
        gt_context_calls.append(dict(kwargs))
        return {0: MagicMock()}

    monkeypatch.setattr(cmaes, "prepare_gt_wasserstein_context", capture_gt_context)

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
        "score_candidate_wasserstein",
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
        ),
    )

    assert evaluation.direction_indices == (0, 2)
    assert replay_call["direction_indices"] == [0, 2]
    assert replay_call["num_directions"] == 2
    assert gt_context_calls[0]["n_directions"] == expected_scoring_n_directions
    assert score_calls[0]["n_directions"] == expected_scoring_n_directions


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
        missing_directions_by_candidate={1: (2,)},
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
