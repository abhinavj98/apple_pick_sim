"""Policy action space [-1, 1]^13 <-> harvest env units (pure torch)."""

from __future__ import annotations

import math

import pytest
import torch

from apple_pick_gym.batched_envs.harvest_action import HarvestActionBounds
from apple_pick_gym.rl.action_scaling import HarvestActionScaler


@pytest.fixture
def scaler():
    return HarvestActionScaler(HarvestActionBounds())


def test_plus_minus_one_map_exactly_to_the_bounds(scaler):
    b = HarvestActionBounds()
    lo = scaler.to_env(-torch.ones(1, 13))[0]
    hi = scaler.to_env(torch.ones(1, 13))[0]
    torch.testing.assert_close(lo[:3], torch.full((3,), -b.linear_delta_m))
    torch.testing.assert_close(hi[3:6], torch.full((3,), b.angular_delta_rad))
    torch.testing.assert_close(lo[6:9], torch.full((3,), b.k_lin_min))
    torch.testing.assert_close(hi[6:9], torch.full((3,), b.k_lin_max))
    torch.testing.assert_close(lo[9:12], torch.full((3,), b.k_ang_min))
    torch.testing.assert_close(hi[9:12], torch.full((3,), b.k_ang_max))
    assert float(lo[12]) == pytest.approx(b.zeta_min)
    assert float(hi[12]) == pytest.approx(b.zeta_max)


def test_zero_is_zero_delta_geometric_mean_stiffness_mid_zeta(scaler):
    b = HarvestActionBounds()
    mid = scaler.to_env(torch.zeros(1, 13))[0]
    torch.testing.assert_close(mid[:6], torch.zeros(6))
    assert float(mid[6]) == pytest.approx(math.sqrt(b.k_lin_min * b.k_lin_max), rel=1e-5)
    assert float(mid[9]) == pytest.approx(math.sqrt(b.k_ang_min * b.k_ang_max), rel=1e-5)
    assert float(mid[12]) == pytest.approx(0.5 * (b.zeta_min + b.zeta_max), rel=1e-6)


def test_round_trip(scaler):
    u = torch.rand(64, 13, generator=torch.Generator().manual_seed(0)) * 2 - 1
    torch.testing.assert_close(scaler.to_policy(scaler.to_env(u)), u, atol=1e-5, rtol=1e-5)


def test_out_of_range_policy_actions_are_clipped(scaler):
    torch.testing.assert_close(scaler.to_env(torch.full((2, 13), 5.0)), scaler.to_env(torch.ones(2, 13)))


def test_stiffness_is_log_affine(scaler):
    """Equal steps in u multiply K by a constant factor (stiffness spans a decade)."""
    u = torch.zeros(3, 13)
    u[:, 6] = torch.tensor([-0.5, 0.0, 0.5])
    k = scaler.to_env(u)[:, 6]
    assert float(k[1] / k[0]) == pytest.approx(float(k[2] / k[1]), rel=1e-5)


def test_shape_is_checked(scaler):
    with pytest.raises(ValueError):
        scaler.to_env(torch.zeros(2, 12))
