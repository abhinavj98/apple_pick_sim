"""Tests for Sinkhorn Wasserstein sys-ID scoring."""

from __future__ import annotations

import numpy as np
import pytest

from apple_pick_sim.system_id.wasserstein import (
    LOW_SAMPLE_MIN_TRANSITIONS,
    POOLED_DIRECTION_KEY,
    prepare_gt_wasserstein_context,
    score_candidate_wasserstein,
    sinkhorn_distance,
)
from apple_pick_sim.system_id.wasserstein_ranking import (
    sinkhorn_gt_preference,
    sinkhorn_mse_spearman,
)

torch = pytest.importorskip("torch")
geomloss = pytest.importorskip("geomloss")


def _arrays_for_steps(*, steps: int, shift: float = 0.0) -> dict:
    junction_names = ["joint_a"]
    base = np.arange(steps, dtype=np.float32).reshape(steps, 1) + float(shift)
    woody_start = {
        "joint_a": np.hstack([base + 100.0, base + 101.0, base + 102.0]).astype(np.float32),
    }
    woody_end = {
        "joint_a": np.hstack([base + 300.0, base + 301.0, base + 302.0]).astype(np.float32),
    }
    return {
        "ft_wrist": np.hstack([base + i for i in range(6)]).astype(np.float32),
        "tcp_velocity": np.hstack([base + 10.0 + i for i in range(6)]).astype(np.float32),
        "action": np.hstack([base + 20.0 + i for i in range(6)]).astype(np.float32),
        "tcp_pos": np.hstack([base + 30.0 + i for i in range(3)]).astype(np.float32),
        "apple_pos": np.hstack([base + 40.0 + i for i in range(3)]).astype(np.float32),
        "woody_part_start_pos": woody_start,
        "woody_part_end_pos": woody_end,
        "excitation_direction": np.tile(
            np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (steps, 1)
        ),
        "phase": np.ones(steps, dtype=np.int8),
        "excitation_type": np.zeros(steps, dtype=np.int8),
        "dir_idx": np.zeros(steps, dtype=np.int32),
        "junction_names": junction_names,
    }


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
