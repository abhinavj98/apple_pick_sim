"""Eval JSON: success-conditioned per-episode stats are pooled weighted by each episode's valid wins."""

from __future__ import annotations

import math

import pytest

from apple_pick_gym.rl.eval_vic_harvest import success_weighted


def _ep(success, invalid, value):
    return {"Episode / success rate": success, "Episode / invalid fraction": invalid, "x": value}


def test_success_weighted_mean_skips_episodes_without_wins_and_nans():
    eps = [_ep(0.5, 0.0, 10.0), _ep(1.0, 0.0, 40.0), _ep(0.0, 0.0, 99.0), _ep(0.5, 0.0, float("nan"))]
    assert success_weighted(eps, "x") == pytest.approx((0.5 * 10 + 1.0 * 40) / 1.5)
    assert math.isnan(success_weighted([_ep(0.0, 0.0, 5.0)], "x"))
