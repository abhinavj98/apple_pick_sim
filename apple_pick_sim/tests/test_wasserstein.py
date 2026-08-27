"""Tests for Sinkhorn Wasserstein sys-ID scoring."""

from __future__ import annotations

import numpy as np
import pytest

from apple_pick_sim.system_id import wasserstein
from apple_pick_sim.system_id.wasserstein import (
    LOW_SAMPLE_MIN_TRANSITIONS,
    POOLED_DIRECTION_KEY,
    prepare_gt_wasserstein_context,
    prepare_gt_wasserstein_scoring_context,
    score_candidate_wasserstein,
    score_candidate_wasserstein_complete,
    sinkhorn_distance,
)
from apple_pick_sim.system_id.wasserstein_ranking import (
    sinkhorn_gt_preference,
    sinkhorn_mse_spearman,
)

torch = pytest.importorskip("torch")
geomloss = pytest.importorskip("geomloss")


def _arrays_for_steps(*, steps: int, shift: float = 0.0) -> dict:
    junction_names = ["primary_spur", "spur_stem"]
    base = np.arange(steps, dtype=np.float32).reshape(steps, 1) + float(shift)
    woody_start = {
        "primary_spur": np.hstack([base + 100.0, base + 101.0, base + 102.0]).astype(
            np.float32
        ),
        "spur_stem": np.hstack([base + 200.0, base + 201.0, base + 202.0]).astype(
            np.float32
        ),
    }
    return {
        "ft_wrist": np.hstack([base + i for i in range(6)]).astype(np.float32),
        "tcp_velocity": np.hstack([base + 10.0 + i for i in range(6)]).astype(np.float32),
        "action": np.hstack([base + 20.0 + i for i in range(6)]).astype(np.float32),
        "tcp_pos": np.hstack([base + 30.0 + i for i in range(3)]).astype(np.float32),
        "apple_pos": np.hstack([base + 40.0 + i for i in range(3)]).astype(np.float32),
        "woody_part_start_pos": woody_start,
        "excitation_direction": np.tile(
            np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (steps, 1)
        ),
        "phase": np.ones(steps, dtype=np.int8),
        "excitation_type": np.zeros(steps, dtype=np.int8),
        "dir_idx": np.zeros(steps, dtype=np.int32),
        "junction_names": junction_names,
    }


def _episode_with_median_holds(
    *, dir_idx: int, n_holds: int = 3, shift: float = 0.0
) -> dict:
    """Episode with ``n_holds`` median holds (``n_holds - 1`` transitions).

    ``n_holds=2`` yields a singleton transition bag. Sinkhorn scoring must keep
    that case finite via the GeomLoss p=2 Dirac cost (not by dropping the bag).
    """
    if n_holds < 2:
        raise ValueError("n_holds must be >= 2 for median transitions")
    chunks: list[np.ndarray] = []
    for _ in range(n_holds):
        chunks.append(np.zeros(1, dtype=np.int8))
        chunks.append(np.ones(3, dtype=np.int8))
    chunks.append(np.zeros(1, dtype=np.int8))
    phase = np.concatenate(chunks)
    steps = int(phase.size)
    ep = _arrays_for_steps(steps=steps, shift=shift)
    ep["dir_idx"] = np.full(steps, int(dir_idx), dtype=np.int32)
    ep["phase"] = phase
    if dir_idx != 0:
        ep["excitation_direction"] = np.tile(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (steps, 1)
        )
    return ep


def _two_hold_episode(*, dir_idx: int, shift: float = 0.0, steps: int = 10) -> dict:
    """Two median holds → one transition (singleton Sinkhorn bag)."""
    del steps  # size is derived from the hold pattern
    return _episode_with_median_holds(dir_idx=dir_idx, n_holds=2, shift=shift)


def _episode_without_valid_holds(*, dir_idx: int, steps: int = 10) -> dict:
    ep = _arrays_for_steps(steps=steps)
    ep["dir_idx"] = np.full(steps, int(dir_idx), dtype=np.int32)
    ep["phase"] = np.zeros(steps, dtype=np.int8)
    if dir_idx != 0:
        ep["excitation_direction"] = np.tile(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (steps, 1)
        )
    return ep


def _multi_median_hold_episode(
    *, dir_idx: int, n_holds: int, shift: float = 0.0
) -> dict:
    return _episode_with_median_holds(dir_idx=dir_idx, n_holds=n_holds, shift=shift)


def test_sinkhorn_distance_identical_clouds_near_zero():
    cloud = np.random.default_rng(0).normal(size=(32, 4)).astype(np.float64)
    dist = sinkhorn_distance(cloud, cloud.copy(), device="cpu")
    assert dist == pytest.approx(0.0, abs=1e-3)


def test_sinkhorn_distance_far_shift_larger_than_near_shift():
    rng = np.random.default_rng(1)
    gt = rng.normal(size=(32, 4)).astype(np.float64)
    near = gt + 0.05
    far = gt + 5.0
    near_dist = sinkhorn_distance(gt, near, device="cpu")
    far_dist = sinkhorn_distance(gt, far, device="cpu")
    assert far_dist > near_dist


def test_sinkhorn_distance_singleton_identical_is_zero():
    """Singleton-vs-singleton must stay finite; GeomLoss fails on zero diameter."""
    point = np.array([[0.25, -1.5, 2.0, 0.5]], dtype=np.float64)
    dist = sinkhorn_distance(point, point.copy(), device="cpu")
    assert dist == pytest.approx(0.0, abs=1e-12)


def test_sinkhorn_distance_singleton_shift_matches_geomloss_half_sqdist():
    """GeomLoss p=2 cost is (1/2)||x-y||^2; Dirac Sinkhorn equals that cost."""
    x = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)
    y = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    dist = sinkhorn_distance(x, y, device="cpu")
    expected = 0.5 * float(np.sum((x - y) ** 2))
    assert dist == pytest.approx(expected, abs=1e-12)
    assert dist > 0.0


def test_feature_kwargs_forward_hold_reduce_mean():
    kwargs = wasserstein._feature_kwargs(
        use_median=False,
        hold_id_onehot=False,
        n_holds=5,
        dir_id_onehot=False,
        n_directions=8,
        hold_reduce="mean",
    )
    assert kwargs["hold_reduce"] == "mean"
    assert kwargs["use_median"] is False


def test_complete_score_single_transition_bags_are_low_sample_and_finite():
    """Exactly two median holds → one transition per direction; keep finite scores."""
    gt = [_two_hold_episode(dir_idx=0), _two_hold_episode(dir_idx=2)]
    shifted = [
        _two_hold_episode(dir_idx=0, shift=0.0),
        _two_hold_episode(dir_idx=2, shift=25.0),
    ]
    context = prepare_gt_wasserstein_scoring_context(
        gt,
        use_median=True,
        hold_id_onehot=True,
        n_holds=5,
        pool_directions=True,
        n_directions=5,
    )
    identical = score_candidate_wasserstein_complete(
        candidate_index=0,
        stiffnesses={"primary_e_pa": 1.0},
        gt_context=context,
        replay_observations=gt,
        device="cpu",
        use_median=True,
        hold_id_onehot=True,
        n_holds=5,
        n_directions=5,
    )
    shifted_result = score_candidate_wasserstein_complete(
        candidate_index=1,
        stiffnesses={"primary_e_pa": 2.0},
        gt_context=context,
        replay_observations=shifted,
        device="cpu",
        use_median=True,
        hold_id_onehot=True,
        n_holds=5,
        n_directions=5,
    )

    assert identical.missing_directions == ()
    assert set(identical.per_direction_sinkhorn) == {0, 2}
    assert identical.per_direction_n_transitions == {0: 1, 2: 1}
    assert identical.low_sample_directions == (0, 2)
    assert np.isfinite(identical.aggregate_sinkhorn)
    assert identical.aggregate_sinkhorn == pytest.approx(0.0, abs=1e-2)
    assert all(np.isfinite(v) for v in identical.per_direction_sinkhorn.values())
    assert np.isfinite(shifted_result.aggregate_sinkhorn)
    assert shifted_result.aggregate_sinkhorn > identical.aggregate_sinkhorn
    assert shifted_result.per_direction_sinkhorn[2] > identical.per_direction_sinkhorn[2]


def test_prepare_gt_wasserstein_context_per_direction_stats():
    episodes = [_arrays_for_steps(steps=8)]
    context = prepare_gt_wasserstein_context(episodes)
    direction = 0
    assert direction in context
    assert context[direction].gt_norm.ndim == 2
    assert context[direction].gt_norm.shape[0] >= 1


def test_score_candidate_wasserstein_identical_near_zero():
    episodes = [_arrays_for_steps(steps=20)]
    gt_context = prepare_gt_wasserstein_context(episodes)
    result = score_candidate_wasserstein(
        candidate_index=0,
        stiffnesses={"primary": 1.0, "secondary": 2.0, "spur": 3.0, "stem": 4.0},
        gt_context=gt_context,
        replay_observations=episodes,
        device="cpu",
    )
    assert result.aggregate_sinkhorn == pytest.approx(0.0, abs=1e-2)
    assert result.low_sample_directions == ()


def test_score_candidate_wasserstein_shifted_higher_than_identical():
    episodes = [_arrays_for_steps(steps=20)]
    gt_context = prepare_gt_wasserstein_context(episodes)
    shifted = [_arrays_for_steps(steps=20, shift=50.0)]
    identical = score_candidate_wasserstein(
        candidate_index=0,
        stiffnesses={"primary": 1.0},
        gt_context=gt_context,
        replay_observations=episodes,
        device="cpu",
    )
    shifted_result = score_candidate_wasserstein(
        candidate_index=1,
        stiffnesses={"primary": 2.0},
        gt_context=gt_context,
        replay_observations=shifted,
        device="cpu",
    )
    assert shifted_result.aggregate_sinkhorn > identical.aggregate_sinkhorn


def test_score_candidate_wasserstein_aggregate_is_mean_of_directions():
    ep0 = _arrays_for_steps(steps=8)
    ep1 = _arrays_for_steps(steps=8)
    ep1["excitation_direction"] = np.tile(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (8, 1)
    )
    gt_context = prepare_gt_wasserstein_context([ep0, ep1])
    result = score_candidate_wasserstein(
        candidate_index=0,
        stiffnesses={"primary": 1.0},
        gt_context=gt_context,
        replay_observations=[ep0, ep1],
        device="cpu",
    )
    per_dir = list(result.per_direction_sinkhorn.values())
    weights = list(result.per_direction_n_transitions.values())
    assert result.aggregate_sinkhorn == pytest.approx(
        float(np.average(per_dir, weights=weights))
    )


def test_score_candidate_wasserstein_flags_low_sample_direction():
    # Full-hold frame→frame: n_frames-1 transitions; keep below LOW_SAMPLE_MIN.
    episodes = [_arrays_for_steps(steps=6)]
    gt_context = prepare_gt_wasserstein_context(episodes)
    shifted = [_arrays_for_steps(steps=6, shift=1.0)]
    result = score_candidate_wasserstein(
        candidate_index=0,
        stiffnesses={"primary": 1.0},
        gt_context=gt_context,
        replay_observations=shifted,
        device="cpu",
    )
    assert result.low_sample_directions == (0,)
    assert result.per_direction_n_transitions[0] < LOW_SAMPLE_MIN_TRANSITIONS
    assert LOW_SAMPLE_MIN_TRANSITIONS == 8


def test_pool_directions_appends_dir_id_onehot():
    """Pooling auto-enables dir one-hot; feature dim grows by n_directions."""

    def _two_hold_episode(*, dir_idx: int, shift: float = 0.0) -> dict:
        ep = _arrays_for_steps(steps=10, shift=shift)
        ep["dir_idx"] = np.full(10, int(dir_idx), dtype=np.int32)
        ep["phase"] = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0], dtype=np.int8)
        if dir_idx != 0:
            ep["excitation_direction"] = np.tile(
                np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (10, 1)
            )
        return ep

    ep0 = _two_hold_episode(dir_idx=0)
    ep2 = _two_hold_episode(dir_idx=2)
    unpooled = prepare_gt_wasserstein_context([ep0, ep2], use_median=True)
    pooled = prepare_gt_wasserstein_context(
        [ep0, ep2],
        use_median=True,
        pool_directions=True,
        n_directions=5,
    )
    assert POOLED_DIRECTION_KEY in pooled
    assert set(unpooled) == {0, 2}
    base_dim = unpooled[0].gt_norm.shape[1]
    assert pooled[POOLED_DIRECTION_KEY].gt_norm.shape[1] == base_dim + 5
    result = score_candidate_wasserstein(
        candidate_index=0,
        stiffnesses={"primary": 1.0},
        gt_context=pooled,
        replay_observations=[ep0, ep2],
        device="cpu",
        use_median=True,
        pool_directions=True,
        n_directions=5,
    )
    assert POOLED_DIRECTION_KEY in result.per_direction_sinkhorn
    assert np.isfinite(result.aggregate_sinkhorn)


def test_score_candidate_wasserstein_raises_when_no_directions():
    episodes = [_arrays_for_steps(steps=8)]
    gt_context = prepare_gt_wasserstein_context(episodes)
    empty = [_arrays_for_steps(steps=2)]
    empty[0]["phase"] = np.zeros(2, dtype=np.int8)
    with pytest.raises(ValueError, match="No candidate directions"):
        score_candidate_wasserstein(
            candidate_index=0,
            stiffnesses={"primary": 1.0},
            gt_context=gt_context,
            replay_observations=empty,
            device="cpu",
        )


def test_complete_pooled_score_reports_missing_physical_direction():
    gt = [_two_hold_episode(dir_idx=0), _two_hold_episode(dir_idx=2)]
    replay = [
        _two_hold_episode(dir_idx=0),
        _episode_without_valid_holds(dir_idx=2),
    ]
    context = prepare_gt_wasserstein_scoring_context(
        gt,
        use_median=True,
        hold_id_onehot=True,
        n_holds=5,
        pool_directions=True,
        n_directions=5,
    )

    result = score_candidate_wasserstein_complete(
        candidate_index=0,
        stiffnesses={"primary_e_pa": 1.0},
        gt_context=context,
        replay_observations=replay,
        device="cpu",
        use_median=True,
        hold_id_onehot=True,
        n_holds=5,
        n_directions=5,
    )

    assert result.missing_directions == (2,)
    assert set(result.per_direction_sinkhorn) == {0}
    assert not np.isfinite(result.aggregate_sinkhorn)


def test_complete_pooled_score_succeeds_for_all_expected_directions():
    gt = [_two_hold_episode(dir_idx=0), _two_hold_episode(dir_idx=2)]
    replay = [_two_hold_episode(dir_idx=0), _two_hold_episode(dir_idx=2)]
    context = prepare_gt_wasserstein_scoring_context(
        gt,
        use_median=True,
        hold_id_onehot=True,
        n_holds=5,
        pool_directions=True,
        n_directions=5,
    )

    result = score_candidate_wasserstein_complete(
        candidate_index=0,
        stiffnesses={"primary_e_pa": 1.0},
        gt_context=context,
        replay_observations=replay,
        device="cpu",
        use_median=True,
        hold_id_onehot=True,
        n_holds=5,
        n_directions=5,
    )

    assert result.missing_directions == ()
    assert set(result.per_direction_sinkhorn) == {0, 2}
    assert POOLED_DIRECTION_KEY not in result.per_direction_sinkhorn
    assert np.isfinite(result.aggregate_sinkhorn)


def test_complete_score_all_replay_directions_empty():
    gt = [_two_hold_episode(dir_idx=0), _two_hold_episode(dir_idx=2)]
    replay = [
        _episode_without_valid_holds(dir_idx=0),
        _episode_without_valid_holds(dir_idx=2),
    ]
    context = prepare_gt_wasserstein_scoring_context(
        gt,
        use_median=True,
        hold_id_onehot=True,
        n_holds=5,
        pool_directions=True,
        n_directions=5,
    )

    result = score_candidate_wasserstein_complete(
        candidate_index=0,
        stiffnesses={"primary_e_pa": 1.0},
        gt_context=context,
        replay_observations=replay,
        device="cpu",
        use_median=True,
        hold_id_onehot=True,
        n_holds=5,
        n_directions=5,
    )

    assert result.missing_directions == (0, 2)
    assert result.per_direction_sinkhorn == {}
    assert not np.isfinite(result.aggregate_sinkhorn)


def test_complete_score_sparse_direction_ids_fixed_onehot_width():
    gt = [_two_hold_episode(dir_idx=0), _two_hold_episode(dir_idx=2)]
    context = prepare_gt_wasserstein_scoring_context(
        gt,
        use_median=True,
        hold_id_onehot=True,
        n_holds=5,
        pool_directions=True,
        n_directions=5,
    )
    assert context.expected_directions == (0, 2)
    # Per-direction diagnostics have no dir one-hot; pooled bag adds width 5.
    base_dim = context.per_direction[0].gt_norm.shape[1]
    assert context.pooled.gt_norm.shape[1] == base_dim + 5

    result = score_candidate_wasserstein_complete(
        candidate_index=0,
        stiffnesses={"primary_e_pa": 1.0},
        gt_context=context,
        replay_observations=gt,
        device="cpu",
        use_median=True,
        hold_id_onehot=True,
        n_holds=5,
        n_directions=5,
    )
    assert set(result.per_direction_sinkhorn) == {0, 2}
    assert np.isfinite(result.aggregate_sinkhorn)


def test_complete_score_rejects_unexpected_replay_direction():
    gt = [_two_hold_episode(dir_idx=0), _two_hold_episode(dir_idx=2)]
    replay = [
        _two_hold_episode(dir_idx=0),
        _two_hold_episode(dir_idx=2),
        _two_hold_episode(dir_idx=4),
    ]
    context = prepare_gt_wasserstein_scoring_context(
        gt,
        use_median=True,
        hold_id_onehot=True,
        n_holds=5,
        pool_directions=True,
        n_directions=5,
    )
    with pytest.raises(ValueError, match="unexpected"):
        score_candidate_wasserstein_complete(
            candidate_index=0,
            stiffnesses={"primary_e_pa": 1.0},
            gt_context=context,
            replay_observations=replay,
            device="cpu",
            use_median=True,
            hold_id_onehot=True,
            n_holds=5,
            n_directions=5,
        )


def test_complete_score_low_sample_keyed_by_physical_direction():
    # One median transition (< LOW_SAMPLE_MIN_TRANSITIONS) per expected direction.
    gt = [_two_hold_episode(dir_idx=0), _two_hold_episode(dir_idx=2)]
    context = prepare_gt_wasserstein_scoring_context(
        gt,
        use_median=True,
        hold_id_onehot=True,
        n_holds=5,
        pool_directions=True,
        n_directions=5,
    )
    result = score_candidate_wasserstein_complete(
        candidate_index=0,
        stiffnesses={"primary_e_pa": 1.0},
        gt_context=context,
        replay_observations=gt,
        device="cpu",
        use_median=True,
        hold_id_onehot=True,
        n_holds=5,
        n_directions=5,
    )
    assert result.low_sample_directions == (0, 2)
    assert result.per_direction_n_transitions[0] < LOW_SAMPLE_MIN_TRANSITIONS
    assert result.per_direction_n_transitions[2] < LOW_SAMPLE_MIN_TRANSITIONS
    assert POOLED_DIRECTION_KEY not in result.low_sample_directions


def test_complete_pooled_aggregate_differs_from_per_direction_mean():
    """Pooled Sinkhorn is the fitness; it is not a mean of per-dir diagnostics."""
    gt = [
        _multi_median_hold_episode(dir_idx=0, n_holds=4),
        _multi_median_hold_episode(dir_idx=2, n_holds=4),
    ]
    # Shift only direction 2 so per-dir losses differ and pooling mixes bags.
    replay = [
        _multi_median_hold_episode(dir_idx=0, n_holds=4, shift=0.0),
        _multi_median_hold_episode(dir_idx=2, n_holds=4, shift=40.0),
    ]
    context = prepare_gt_wasserstein_scoring_context(
        gt,
        use_median=True,
        hold_id_onehot=True,
        n_holds=5,
        pool_directions=True,
        n_directions=5,
    )
    result = score_candidate_wasserstein_complete(
        candidate_index=0,
        stiffnesses={"primary_e_pa": 1.0},
        gt_context=context,
        replay_observations=replay,
        device="cpu",
        use_median=True,
        hold_id_onehot=True,
        n_holds=5,
        n_directions=5,
    )
    weights = [
        float(result.per_direction_n_transitions[d])
        for d in result.per_direction_sinkhorn
    ]
    values = [float(result.per_direction_sinkhorn[d]) for d in result.per_direction_sinkhorn]
    weighted_mean = float(np.average(values, weights=weights))
    assert np.isfinite(result.aggregate_sinkhorn)
    assert result.aggregate_sinkhorn != pytest.approx(weighted_mean, rel=1e-3, abs=1e-3)


def test_complete_unpooled_aggregate_is_transition_count_weighted_mean():
    gt = [
        _multi_median_hold_episode(dir_idx=0, n_holds=3),
        _multi_median_hold_episode(dir_idx=2, n_holds=5),
    ]
    replay = [
        _multi_median_hold_episode(dir_idx=0, n_holds=3, shift=5.0),
        _multi_median_hold_episode(dir_idx=2, n_holds=5, shift=20.0),
    ]
    context = prepare_gt_wasserstein_scoring_context(
        gt,
        use_median=True,
        hold_id_onehot=True,
        n_holds=5,
        pool_directions=False,
        n_directions=5,
    )
    result = score_candidate_wasserstein_complete(
        candidate_index=0,
        stiffnesses={"primary_e_pa": 1.0},
        gt_context=context,
        replay_observations=replay,
        device="cpu",
        use_median=True,
        hold_id_onehot=True,
        n_holds=5,
        pool_directions=False,
        n_directions=5,
    )
    weights = [
        float(result.per_direction_n_transitions[d])
        for d in result.per_direction_sinkhorn
    ]
    values = [float(result.per_direction_sinkhorn[d]) for d in result.per_direction_sinkhorn]
    assert result.aggregate_sinkhorn == pytest.approx(
        float(np.average(values, weights=weights))
    )
    assert POOLED_DIRECTION_KEY not in result.per_direction_sinkhorn


def test_sinkhorn_gt_preference_ranks_gt_first():
    from apple_pick_sim.system_id.wasserstein import WassersteinCandidateResult

    results = [
        WassersteinCandidateResult(
            candidate_index=0,
            stiffnesses={"primary": 1.0},
            aggregate_sinkhorn=0.5,
            per_direction_sinkhorn={0: 0.5},
            per_direction_n_transitions={0: 10},
            low_sample_directions=(),
            missing_directions=(),
        ),
        WassersteinCandidateResult(
            candidate_index=1,
            stiffnesses={"primary": 2.0},
            aggregate_sinkhorn=0.1,
            per_direction_sinkhorn={0: 0.1},
            per_direction_n_transitions={0: 10},
            low_sample_directions=(),
            missing_directions=(),
        ),
        WassersteinCandidateResult(
            candidate_index=2,
            stiffnesses={"primary": 3.0},
            aggregate_sinkhorn=0.3,
            per_direction_sinkhorn={0: 0.3},
            per_direction_n_transitions={0: 10},
            low_sample_directions=(),
            missing_directions=(),
        ),
    ]
    pref = sinkhorn_gt_preference(
        results=results,
        gt_candidate_index=1,
        disqualified=[False, False, False],
    )
    assert pref.best_is_gt is True
    assert pref.gt_rank == 1
    assert pref.best_candidate_index == 1


def test_sinkhorn_gt_preference_excludes_disqualified():
    from apple_pick_sim.system_id.wasserstein import WassersteinCandidateResult

    results = [
        WassersteinCandidateResult(
            candidate_index=0,
            stiffnesses={"primary": 1.0},
            aggregate_sinkhorn=0.01,
            per_direction_sinkhorn={0: 0.01},
            per_direction_n_transitions={0: 10},
            low_sample_directions=(),
            missing_directions=(),
        ),
        WassersteinCandidateResult(
            candidate_index=1,
            stiffnesses={"primary": 2.0},
            aggregate_sinkhorn=0.5,
            per_direction_sinkhorn={0: 0.5},
            per_direction_n_transitions={0: 10},
            low_sample_directions=(),
            missing_directions=(),
        ),
    ]
    pref = sinkhorn_gt_preference(
        results=results,
        gt_candidate_index=1,
        disqualified=[True, False],
    )
    assert pref.gt_disqualified is False
    assert pref.best_is_gt is True
    assert pref.best_candidate_index == 1
    assert pref.n_disqualified == 1


def test_sinkhorn_gt_preference_resolves_noncontiguous_candidate_index():
    from apple_pick_sim.system_id.wasserstein import WassersteinCandidateResult

    results = [
        WassersteinCandidateResult(
            candidate_index=2,
            stiffnesses={"primary": 1.0},
            aggregate_sinkhorn=0.9,
            per_direction_sinkhorn={0: 0.9},
            per_direction_n_transitions={0: 10},
            low_sample_directions=(),
            missing_directions=(),
        ),
        WassersteinCandidateResult(
            candidate_index=5,
            stiffnesses={"primary": 2.0},
            aggregate_sinkhorn=0.1,
            per_direction_sinkhorn={0: 0.1},
            per_direction_n_transitions={0: 10},
            low_sample_directions=(),
            missing_directions=(),
        ),
        WassersteinCandidateResult(
            candidate_index=9,
            stiffnesses={"primary": 3.0},
            aggregate_sinkhorn=0.2,
            per_direction_sinkhorn={0: 0.2},
            per_direction_n_transitions={0: 10},
            low_sample_directions=(),
            missing_directions=(),
        ),
    ]
    pref = sinkhorn_gt_preference(
        results=results,
        gt_candidate_index=5,
        disqualified=[True, False, False],
    )
    assert pref.gt_disqualified is False
    assert pref.best_is_gt is True
    assert pref.gt_rank == 1
    assert pref.best_candidate_index == 5
    assert pref.n_disqualified == 1


def test_sinkhorn_mse_spearman_positive_for_monotone_errors():
    sinkhorn = [0.1, 0.3, 0.5]
    mse = [0.0, 0.2, 0.4]
    out = sinkhorn_mse_spearman(
        sinkhorn_values=sinkhorn,
        mse_values=mse,
        metric="err_pos_hold",
    )
    assert out.spearman == pytest.approx(1.0)


def test_sinkhorn_mse_spearman_excludes_disqualified():
    sinkhorn = [0.1, 0.3, 1000.0]
    mse = [0.0, 0.2, -999.0]
    out = sinkhorn_mse_spearman(
        sinkhorn_values=sinkhorn,
        mse_values=mse,
        metric="err_pos_hold",
        disqualified=[False, False, True],
    )
    assert out.spearman == pytest.approx(1.0)


def test_near_constant_gt_column_does_not_explode_sinkhorn():
    """Hold-constant features must not dominate cost via eps std floor."""
    steps = 12
    base = _arrays_for_steps(steps=steps)
    gt_ep = _episode_with_median_holds(dir_idx=0, n_holds=4)
    ctx = prepare_gt_wasserstein_scoring_context(
        [gt_ep], use_median=True, hold_id_onehot=True, pool_directions=True, n_directions=1
    )
    sim_ep = _episode_with_median_holds(dir_idx=0, n_holds=4, shift=0.05)
    self_score = score_candidate_wasserstein_complete(
        candidate_index=0,
        stiffnesses={},
        gt_context=ctx,
        replay_observations=[gt_ep],
        use_median=True,
        hold_id_onehot=True,
        pool_directions=True,
        n_directions=1,
    )
    sim_score = score_candidate_wasserstein_complete(
        candidate_index=0,
        stiffnesses={},
        gt_context=ctx,
        replay_observations=[sim_ep],
        use_median=True,
        hold_id_onehot=True,
        pool_directions=True,
        n_directions=1,
    )
    assert self_score.aggregate_sinkhorn == pytest.approx(0.0, abs=1e-6)
    assert sim_score.aggregate_sinkhorn < 1.0e6
