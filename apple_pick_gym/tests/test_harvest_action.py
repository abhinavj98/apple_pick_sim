"""13-D delta-pose harvest action: bounded delta integration + Kp/zeta -> 19-D vic_pose."""

from __future__ import annotations

import math

import torch

from apple_pick_gym.batched_envs.harvest_action import (
    HarvestActionBounds,
    derive_critical_damping,
    integrate_delta_pose,
    pack_vic_pose_action,
    split_harvest_action,
)


def _identity_target(n: int) -> torch.Tensor:
    """``(N, 7)`` ``[pos(3)=0, quat_wxyz=(1,0,0,0)]``."""
    t = torch.zeros((n, 7), dtype=torch.float32)
    t[:, 3] = 1.0
    return t


def test_derive_critical_damping_matches_formula():
    k = torch.tensor([[4.0, 9.0, 16.0]])
    d = derive_critical_damping(k, zeta=1.0)
    torch.testing.assert_close(d, torch.tensor([[4.0, 6.0, 8.0]]))


def test_derive_critical_damping_zero_zeta_gives_zero_damping():
    k = torch.tensor([[4.0, 9.0]])
    d = derive_critical_damping(k, zeta=0.0)
    torch.testing.assert_close(d, torch.zeros_like(k))


def test_derive_critical_damping_clamps_negative_stiffness():
    k = torch.tensor([[-4.0, 9.0]])
    d = derive_critical_damping(k, zeta=1.0)
    assert d[0, 0].item() == 0.0
    assert d[0, 1].item() == 6.0


def test_derive_critical_damping_per_env_zeta_tensor():
    k = torch.tensor([[4.0], [4.0]])
    zeta = torch.tensor([[1.0], [2.0]])
    d = derive_critical_damping(k, zeta)
    torch.testing.assert_close(d, torch.tensor([[4.0], [8.0]]))


def test_integrate_delta_pose_accumulates_position():
    target = _identity_target(1)
    delta = torch.zeros((1, 6), dtype=torch.float32)
    delta[:, 0] = 0.01  # +x
    for _ in range(5):
        target = integrate_delta_pose(target, delta)
    torch.testing.assert_close(target[:, :3], torch.tensor([[0.05, 0.0, 0.0]]), atol=1e-6, rtol=0)


def test_integrate_delta_pose_zero_delta_is_noop():
    target = _identity_target(2)
    target[:, :3] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    delta = torch.zeros((2, 6), dtype=torch.float32)
    out = integrate_delta_pose(target, delta)
    torch.testing.assert_close(out, target, atol=1e-6, rtol=0)


def test_integrate_delta_pose_rotation_stays_normalized_over_many_steps():
    target = _identity_target(3)
    delta = torch.zeros((3, 6), dtype=torch.float32)
    delta[:, 3] = 0.05  # small +x-axis rotation per step
    delta[:, 4] = 0.03
    for _ in range(200):
        target = integrate_delta_pose(target, delta)
    quat_norm = torch.linalg.norm(target[:, 3:7], dim=-1)
    torch.testing.assert_close(quat_norm, torch.ones(3), atol=1e-5, rtol=0)


def test_integrate_delta_pose_quarter_turn_matches_known_quaternion():
    target = _identity_target(1)
    delta = torch.zeros((1, 6), dtype=torch.float32)
    delta[:, 5] = math.pi / 2  # +z-axis quarter turn, single step
    out = integrate_delta_pose(target, delta)
    expected = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]])  # [w, x, y, z]
    torch.testing.assert_close(out[:, 3:7], expected, atol=1e-6, rtol=0)


def test_split_harvest_action_clamps_delta_norm():
    bounds = HarvestActionBounds(linear_delta_m=0.02, angular_delta_rad=0.1)
    actions = torch.zeros((1, 13), dtype=torch.float32)
    actions[:, 0] = 10.0  # way over the linear delta bound
    actions[:, 3] = 10.0  # way over the angular delta bound
    split = split_harvest_action(actions, bounds)
    dp_norm = torch.linalg.norm(split.delta[:, :3], dim=-1)
    drot_norm = torch.linalg.norm(split.delta[:, 3:6], dim=-1)
    torch.testing.assert_close(dp_norm, torch.tensor([bounds.linear_delta_m]), atol=1e-6, rtol=0)
    torch.testing.assert_close(drot_norm, torch.tensor([bounds.angular_delta_rad]), atol=1e-6, rtol=0)


def test_split_harvest_action_clamps_stiffness_and_zeta():
    bounds = HarvestActionBounds(
        k_lin_min=20.0, k_lin_max=800.0, k_ang_min=2.0, k_ang_max=80.0, zeta_min=0.3, zeta_max=2.0
    )
    actions = torch.zeros((1, 13), dtype=torch.float32)
    actions[:, 6:9] = 5000.0  # over k_lin_max
    actions[:, 9:12] = -50.0  # under k_ang_min
    actions[:, 12] = 10.0  # over zeta_max
    split = split_harvest_action(actions, bounds)
    assert torch.all(split.linear_k == bounds.k_lin_max)
    assert torch.all(split.angular_k == bounds.k_ang_min)
    assert split.zeta.item() == bounds.zeta_max


def test_split_harvest_action_rejects_wrong_width():
    import pytest

    with pytest.raises(ValueError):
        split_harvest_action(torch.zeros((1, 12)), HarvestActionBounds())


def test_pack_vic_pose_action_shape_and_quat_order():
    target = _identity_target(2)
    target[:, 3:7] = torch.tensor([[0.7071, 0.0, 0.7071, 0.0], [0.0, 1.0, 0.0, 0.0]])
    linear_k = torch.full((2, 3), 100.0)
    angular_k = torch.full((2, 3), 10.0)
    zeta = torch.full((2, 1), 1.0)
    packed = pack_vic_pose_action(target, linear_k, angular_k, zeta)
    assert packed.shape == (2, 19)
    # action[3:7] must be the SAME wxyz quaternion, unreordered (H2 sec 3 external contract).
    torch.testing.assert_close(packed[:, 3:7], target[:, 3:7], atol=1e-6, rtol=0)
    # Kp = [Fx,Fy,Fz,Tx,Ty,Tz]
    torch.testing.assert_close(packed[:, 7:10], linear_k)
    torch.testing.assert_close(packed[:, 10:13], angular_k)
    # Kd = 2*zeta*sqrt(Kp)
    expected_kd = 2.0 * 1.0 * torch.sqrt(torch.cat([linear_k, angular_k], dim=-1))
    torch.testing.assert_close(packed[:, 13:19], expected_kd, atol=1e-4, rtol=1e-4)


def test_full_pipeline_split_integrate_pack():
    """End-to-end: raw 13-D action -> bounded delta + gains -> integrated target -> 19-D vic_pose."""
    bounds = HarvestActionBounds()
    n = 4
    target = _identity_target(n)
    actions = torch.zeros((n, 13), dtype=torch.float32)
    actions[:, 2] = 1.0  # request +z delta (will be clamped to bounds.linear_delta_m)
    actions[:, 6:9] = 200.0
    actions[:, 9:12] = 10.0
    actions[:, 12] = 1.0

    split = split_harvest_action(actions, bounds)
    new_target = integrate_delta_pose(target, split.delta)
    packed = pack_vic_pose_action(new_target, split.linear_k, split.angular_k, split.zeta)

    assert packed.shape == (n, 19)
    torch.testing.assert_close(
        new_target[:, 2], torch.full((n,), bounds.linear_delta_m), atol=1e-6, rtol=0
    )
