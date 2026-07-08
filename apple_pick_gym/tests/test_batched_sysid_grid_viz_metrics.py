from __future__ import annotations

import numpy as np
import pytest


def test_log_distance_and_spearman_support_gt_argmin_and_positive_correlation():
    import apple_pick_gym.grid_viz_metrics as m

    gt = {"primary": 1e-3, "spur": 1e-2, "stem": 1e-1}
    candidates = [
        {"primary": 1e-4, "spur": 1e-2, "stem": 1e-1},  # far in primary
        {"primary": 1e-3, "spur": 1e-2, "stem": 1e-1},  # GT
        {"primary": 1e-2, "spur": 1e-2, "stem": 1e-1},  # far in primary
    ]

    d = np.array([m.log_l2_distance_to_gt(c, gt, keys=("primary", "spur", "stem")) for c in candidates])
    # Synthetic error monotone in distance.
    err = d**2

    assert int(np.argmin(err)) == 1
    assert d[1] == pytest.approx(0.0)

    corr = m.spearman_r(d, err)
    assert corr == pytest.approx(1.0)


def test_spearman_r_handles_ties_without_nan():
    import apple_pick_gym.grid_viz_metrics as m

    # Two identical x values should not produce NaN.
    x = np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float64)
    y = np.array([0.0, 0.0, 1.0, 4.0], dtype=np.float64)

    corr = m.spearman_r(x, y)
    assert np.isfinite(corr)


def _woody_episode(*, n: int, shift: float = 0.0) -> dict:
    junction_names = ["primary_spur", "spur_stem"]
    start = {
        name: np.tile(np.array([float(shift), 0.0, 0.0], dtype=np.float32), (n, 1))
        for name in junction_names
    }
    end = {
        name: np.tile(np.array([float(shift) + 1.0, 0.0, 0.0], dtype=np.float32), (n, 1))
        for name in junction_names
    }
    return {
        "woody_part_start_pos": start,
        "woody_part_end_pos": end,
        "junction_names": junction_names,
    }


def test_woody_segment_pos_mse_masked_matching_replay_is_zero():
    import apple_pick_gym.grid_viz_metrics as m

    n = 3
    recorded = _woody_episode(n=n)
    replay = _woody_episode(n=n)
    mask = np.ones(n, dtype=bool)

    out = m.woody_segment_pos_mse_masked(
        replay=replay,
        recorded=recorded,
        junction_names=recorded["junction_names"],
        n=n,
        mask=mask,
    )

    assert set(out) == {"primary_spur", "spur_stem"}
    assert out["primary_spur"] == pytest.approx(0.0)
    assert out["spur_stem"] == pytest.approx(0.0)


def test_woody_segment_pos_mse_masked_shifted_replay_is_positive():
    import apple_pick_gym.grid_viz_metrics as m

    n = 2
    recorded = _woody_episode(n=n, shift=0.0)
    replay = _woody_episode(n=n, shift=1.0)
    mask = np.ones(n, dtype=bool)

    out = m.woody_segment_pos_mse_masked(
        replay=replay,
        recorded=recorded,
        junction_names=recorded["junction_names"],
        n=n,
        mask=mask,
    )

    assert out["primary_spur"] > 0.0
    assert out["spur_stem"] > 0.0


def test_woody_segment_pos_mse_masked_empty_junction_names_returns_empty_dict():
    import apple_pick_gym.grid_viz_metrics as m

    out = m.woody_segment_pos_mse_masked(
        replay={},
        recorded={},
        junction_names=[],
        n=0,
        mask=np.zeros(0, dtype=bool),
    )
    assert out == {}


def test_woody_segment_pos_mse_masked_empty_mask_returns_nan_per_segment():
    import apple_pick_gym.grid_viz_metrics as m

    n = 2
    recorded = _woody_episode(n=n)
    replay = _woody_episode(n=n)
    mask = np.zeros(n, dtype=bool)

    out = m.woody_segment_pos_mse_masked(
        replay=replay,
        recorded=recorded,
        junction_names=recorded["junction_names"],
        n=n,
        mask=mask,
    )

    assert not np.isfinite(out["primary_spur"])
    assert not np.isfinite(out["spur_stem"])


def test_woody_segment_pos_mse_hold_aggregated_matching_replay_is_zero():
    import apple_pick_gym.grid_viz_metrics as m

    n = 4
    recorded = _woody_episode(n=n)
    replay = _woody_episode(n=n)
    hold_idx = np.array([1, 2, 3], dtype=np.int64)

    out = m.woody_segment_pos_mse_hold_aggregated(
        replay=replay,
        recorded=recorded,
        junction_names=recorded["junction_names"],
        n=n,
        hold_idx=hold_idx,
        aggregation="median",
    )

    assert out["primary_spur"] == pytest.approx(0.0)
    assert out["spur_stem"] == pytest.approx(0.0)


def test_woody_segment_pos_mse_hold_aggregated_empty_hold_idx_returns_nan():
    import apple_pick_gym.grid_viz_metrics as m

    n = 2
    recorded = _woody_episode(n=n)
    replay = _woody_episode(n=n)

    out = m.woody_segment_pos_mse_hold_aggregated(
        replay=replay,
        recorded=recorded,
        junction_names=recorded["junction_names"],
        n=n,
        hold_idx=np.zeros(0, dtype=np.int64),
        aggregation="mean",
    )

    assert not np.isfinite(out["primary_spur"])
    assert not np.isfinite(out["spur_stem"])

