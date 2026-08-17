import numpy as np
import pytest

from apple_pick_sim.system_id.holdout_gates import (
    DIRECTION_SPLIT_SEED,
    FORCE_FLOOR_N,
    choose_direction_split,
    magnitude_ratio_ok,
    per_hold_means,
    signed_parallel_series,
    tcp_displacement_along_pull,
    trend_pearson_ok,
)


def test_choose_direction_split_seed_17_is_pinned():
    train, val = choose_direction_split(range(8), seed=DIRECTION_SPLIT_SEED)
    assert train == (2, 4, 5, 6, 7)
    assert val == (0, 1, 3)
    assert not set(train) & set(val)


def test_choose_direction_split_is_seed_sensitive_and_covers_population():
    train, val = choose_direction_split(range(8), seed=7)
    assert sorted(train + val) == list(range(8))
    assert train != (2, 4, 5, 6, 7)


def test_choose_direction_split_rejects_bad_population():
    with pytest.raises(ValueError, match="n_train"):
        choose_direction_split(range(4), seed=17, n_train=5)
    with pytest.raises(ValueError, match="duplicate"):
        choose_direction_split([0, 0, 1], seed=17, n_train=1)


def test_magnitude_ratio_passes_within_factor_three_and_fails_outside():
    ok, ratio = magnitude_ratio_ok(real_mean=2.0, fitted_mean=5.0, floor=FORCE_FLOOR_N, slack=0.4)
    assert ok and ratio == pytest.approx(2.5)
    ok, ratio = magnitude_ratio_ok(real_mean=2.0, fitted_mean=7.0, floor=FORCE_FLOOR_N, slack=0.4)
    assert not ok and ratio == pytest.approx(3.5)
    ok, _ = magnitude_ratio_ok(real_mean=2.0, fitted_mean=0.5, floor=FORCE_FLOOR_N, slack=0.4)
    assert not ok


def test_magnitude_ratio_uses_additive_rule_below_floor():
    # real below floor: pass iff fitted < 3*real + slack, ratio is still reported
    ok, _ = magnitude_ratio_ok(real_mean=0.1, fitted_mean=0.6, floor=FORCE_FLOOR_N, slack=0.4)
    assert ok
    ok, _ = magnitude_ratio_ok(real_mean=0.1, fitted_mean=0.9, floor=FORCE_FLOOR_N, slack=0.4)
    assert not ok


def test_trend_requires_pearson_half():
    real = [1.0, 2.0, 3.0, 4.0]
    ok, r = trend_pearson_ok(real, [1.1, 2.2, 2.9, 4.4], magnitude_passed=True)
    assert ok and r > 0.9
    ok, r = trend_pearson_ok(real, [4.0, 3.0, 2.0, 1.0], magnitude_passed=True)
    assert not ok and r < 0.0


def test_trend_zero_variance_defers_to_magnitude():
    flat = [1.0, 1.0, 1.0]
    ok, r = trend_pearson_ok(flat, flat, magnitude_passed=True)
    assert ok and r is None
    ok, r = trend_pearson_ok(flat, flat, magnitude_passed=False)
    assert not ok and r is None


def test_trend_requires_three_points():
    ok, _ = trend_pearson_ok([1.0, 2.0], [1.0, 2.0], magnitude_passed=True)
    assert not ok


def test_signed_parallel_series_is_signed_not_norm():
    vals = np.array([[0.0, 0.0, -3.0], [0.0, 0.0, 4.0]])
    out = signed_parallel_series(vals, (0.0, 0.0, 2.0))  # non-unit axis is normalized
    assert out.tolist() == [-3.0, 4.0]


def test_per_hold_means_averages_each_contiguous_hold():
    phase = np.array([0, 1, 1, 0, 1, 1], dtype=np.int8)
    dir_idx = np.zeros(6, dtype=np.int32)
    series = np.array([9.0, 1.0, 3.0, 9.0, 10.0, 20.0])
    out = per_hold_means(series, phase=phase, dir_idx=dir_idx, direction=0)
    assert out.tolist() == [2.0, 15.0]


def test_tcp_displacement_references_first_hold_frame_not_episode_start():
    # Episode frame 0 is a pull-in at the origin; first hold is at +1 m along z.
    tcp = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.2],
        ]
    )
    phase = np.array([0, 1, 1], dtype=np.int8)
    dir_idx = np.zeros(3, dtype=np.int32)
    s = tcp_displacement_along_pull(
        tcp, phase=phase, dir_idx=dir_idx, direction=0, pull_direction=(0.0, 0.0, 1.0)
    )
    assert s == pytest.approx([0.0, 0.2])
