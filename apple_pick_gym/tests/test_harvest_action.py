"""13-D delta-pose harvest action: bounded delta integration + Kp/zeta -> 19-D vic_pose."""

from __future__ import annotations

import math

import pytest
import torch

from apple_pick_gym.batched_envs.harvest_action import (
    HarvestActionBounds,
    derive_critical_damping,
    integrate_delta_pose,
    leash_target_pose,
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


def _quat_about_z_wxyz(angle: float) -> torch.Tensor:
    return torch.tensor([[math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]])


def test_leash_leaves_targets_inside_the_leash_untouched():
    target = _identity_target(1)
    target[:, :3] = torch.tensor([0.01, 0.0, 0.0])
    tcp = _identity_target(1)
    out = leash_target_pose(target, tcp, max_pos_offset_m=0.05, max_rot_offset_rad=0.5)
    torch.testing.assert_close(out, target)


def test_leash_projects_position_onto_the_sphere_around_the_tcp():
    """VIC force is K*(target - tcp): bounding the offset bounds the commanded force."""
    tcp = _identity_target(2)
    tcp[1, :3] = torch.tensor([1.0, 1.0, 1.0])
    target = tcp.clone()
    target[0, :3] += torch.tensor([0.3, 0.4, 0.0])  # 0.5 m away
    target[1, :3] += torch.tensor([0.0, 0.0, -0.2])
    out = leash_target_pose(target, tcp, max_pos_offset_m=0.1, max_rot_offset_rad=None)
    torch.testing.assert_close(out[0, :3], torch.tensor([0.06, 0.08, 0.0]))
    torch.testing.assert_close(out[1, :3], torch.tensor([1.0, 1.0, 0.9]))
    torch.testing.assert_close(out[:, 3:], target[:, 3:])


def test_leash_limits_rotation_offset_about_the_same_axis():
    tcp = _identity_target(1)
    target = _identity_target(1)
    target[:, 3:] = _quat_about_z_wxyz(1.0)
    out = leash_target_pose(target, tcp, max_pos_offset_m=None, max_rot_offset_rad=0.25)
    torch.testing.assert_close(out[:, 3:], _quat_about_z_wxyz(0.25), atol=1e-6, rtol=0)


def test_leash_rotation_is_relative_to_a_non_identity_tcp():
    tcp = _identity_target(1)
    tcp[:, 3:] = _quat_about_z_wxyz(0.5)
    target = _identity_target(1)
    target[:, 3:] = _quat_about_z_wxyz(-0.5)  # 1 rad away, the other way
    out = leash_target_pose(target, tcp, max_pos_offset_m=None, max_rot_offset_rad=0.2)
    torch.testing.assert_close(out[:, 3:], _quat_about_z_wxyz(0.3), atol=1e-6, rtol=0)


def test_leash_handles_double_cover_quaternions():
    tcp = _identity_target(1)
    target = _identity_target(1)
    target[:, 3:] = -_quat_about_z_wxyz(0.1)  # same rotation, negated quaternion
    out = leash_target_pose(target, tcp, max_pos_offset_m=None, max_rot_offset_rad=0.2)
    angle = 2 * torch.acos(torch.abs(out[:, 3]).clamp(max=1.0))
    assert float(angle) == pytest.approx(0.1, abs=1e-5)


# [D17] tool-frame action: the policy's pose delta is expressed in the current TCP frame (a twist
# about the grip axis is one action dim) and rotated to world before integration.


def _quat_wxyz_about_z(angle: float) -> torch.Tensor:
    return torch.tensor([[math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]])


def test_tool_delta_to_world_is_identity_for_an_identity_tcp():
    from apple_pick_gym.batched_envs.harvest_action import tool_delta_to_world

    delta = torch.tensor([[0.001, -0.002, 0.003, 0.01, 0.0, -0.02]])
    torch.testing.assert_close(tool_delta_to_world(delta, torch.tensor([[1.0, 0.0, 0.0, 0.0]])), delta)


def test_tool_delta_to_world_rotates_translation_and_rotation_by_the_tcp():
    from apple_pick_gym.batched_envs.harvest_action import tool_delta_to_world

    tcp_q = _quat_wxyz_about_z(math.pi / 2)  # tool x = world y, tool y = world -x
    delta = torch.tensor([[0.002, 0.0, 0.0, 0.0, 0.01, 0.0]])  # +x tool move, +y tool rotation
    out = tool_delta_to_world(delta, tcp_q)
    torch.testing.assert_close(out, torch.tensor([[0.0, 0.002, 0.0, -0.01, 0.0, 0.0]]), atol=1e-7, rtol=0)


def test_tool_delta_preserves_the_norm_so_the_speed_cap_is_frame_free():
    from apple_pick_gym.batched_envs.harvest_action import tool_delta_to_world

    g = torch.Generator().manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(16, 4, generator=g), dim=-1)
    d = torch.randn(16, 6, generator=g)
    out = tool_delta_to_world(d, q)
    torch.testing.assert_close(torch.linalg.norm(out[:, :3], dim=-1), torch.linalg.norm(d[:, :3], dim=-1))
    torch.testing.assert_close(torch.linalg.norm(out[:, 3:], dim=-1), torch.linalg.norm(d[:, 3:], dim=-1))


def test_pack_with_a_realized_stiffness_scale_scales_kp_but_derives_kd_from_the_command():
    # [D17] realized-K DR: the arm renders s * K_cmd; damping stays the commanded 2*zeta*sqrt(K_cmd)
    target = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    lin, ang, zeta = torch.tensor([[100.0, 100.0, 100.0]]), torch.tensor([[10.0, 10.0, 10.0]]), torch.tensor([[1.0]])
    scale = torch.tensor([[0.8, 1.0, 1.2, 0.9, 1.0, 1.1]])
    packed = pack_vic_pose_action(target, lin, ang, zeta, kp_scale=scale)
    torch.testing.assert_close(packed[:, 7:13], torch.tensor([[80.0, 100.0, 120.0, 9.0, 10.0, 11.0]]))
    torch.testing.assert_close(packed[:, 13:19], pack_vic_pose_action(target, lin, ang, zeta)[:, 13:19])


def test_d17_default_grasp_sampler_draws_directions_in_the_cone_and_bounded_rolls():
    import numpy as np

    from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import ApplePickVicHarvestEnv

    pole = (0.0, -1.0, 0.0)
    grips = ApplePickVicHarvestEnv._sample_default_grasps(
        num_envs=200, pole=pole, max_polar_angle_rad=math.pi / 3, seed=4, roll_max_rad=math.pi / 2
    )
    d = np.array([g.weld_direction for g in grips])
    polar = np.degrees(np.arccos(np.clip(d @ np.array(pole), -1, 1)))
    assert polar.max() <= 60.0 + 1e-6 and polar.max() > 55.0
    roll = np.array([g.weld_roll_rad for g in grips])
    assert np.abs(roll).max() <= math.pi / 2 and roll.std() > 0.5
    no_roll = ApplePickVicHarvestEnv._sample_default_grasps(num_envs=4, pole=pole, max_polar_angle_rad=0.5, seed=4)
    assert all(g.weld_roll_rad is None for g in no_roll)


def test_d17_post_grasp_hold_is_stiffer_with_the_same_damping_ratios():
    from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import HoldSettleConfig

    h = HoldSettleConfig()
    assert (h.linear_k, h.angular_k) == (500.0, 30.0)
    # zeta kept from the old 100 N/m / 15 and 10 N*m/rad / 3 hold
    assert h.linear_d / (2 * math.sqrt(h.linear_k)) == pytest.approx(15.0 / (2 * math.sqrt(100.0)), rel=1e-3)
    assert h.angular_d / (2 * math.sqrt(h.angular_k)) == pytest.approx(3.0 / (2 * math.sqrt(10.0)), rel=1e-3)


def test_d17_default_grasp_sampler_sets_the_weld_frame():
    from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import ApplePickVicHarvestEnv

    grips = ApplePickVicHarvestEnv._sample_default_grasps(
        num_envs=3, pole=(0.0, -1.0, 0.0), max_polar_angle_rad=0.5, seed=1, weld_frame="world"
    )
    assert all(g.weld_direction_frame == "world" for g in grips)
    legacy = ApplePickVicHarvestEnv._sample_default_grasps(num_envs=3, pole=(0.0, 0.0, -1.0), max_polar_angle_rad=0.5, seed=1)
    assert all(g.weld_direction_frame == "apple" for g in legacy)


def test_d17_grasp_roll_does_not_reuse_the_seeds_dr_stream():
    # [D17] the support DR draws from default_rng(dr_seed); a roll drawn from default_rng(seed) with
    # the same seed was a monotone function of support kp (rank correlation 1.0 in the v3 pilot)
    import numpy as np

    from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import ApplePickVicHarvestEnv

    grips = ApplePickVicHarvestEnv._sample_default_grasps(
        num_envs=256, pole=(0.0, -1.0, 0.0), max_polar_angle_rad=math.pi / 3, seed=7, roll_max_rad=math.pi / 2
    )
    roll = np.array([g.weld_roll_rad for g in grips])
    same_stream = np.random.default_rng(7).uniform(0.0, 1.0, size=256)
    rank = lambda x: np.argsort(np.argsort(x))
    assert abs(np.corrcoef(rank(roll), rank(same_stream))[0, 1]) < 0.2


def test_d17_random_grasp_sampling_varies_with_the_seed_and_is_area_uniform_in_the_cone():
    import numpy as np

    from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import ApplePickVicHarvestEnv

    pole, cap = np.array([0.0, -1.0, 0.0]), math.pi / 3
    draw = lambda seed, mode: np.array(
        [
            g.weld_direction
            for g in ApplePickVicHarvestEnv._sample_default_grasps(
                num_envs=4000, pole=tuple(pole), max_polar_angle_rad=cap, seed=seed, sampling=mode
            )
        ]
    )
    a, b = draw(1, "random"), draw(2, "random")
    assert not np.allclose(a[:10], b[:10])
    cos = a @ pole
    assert cos.min() >= math.cos(cap) - 1e-6
    assert abs(cos.mean() - (1 + math.cos(cap)) / 2) < 0.01  # area-uniform: cos(polar) ~ U(cos cap, 1)
    np.testing.assert_allclose(draw(1, "fibonacci"), draw(2, "fibonacci"))  # legacy: seed-independent


def test_d17m_no_advance_wall_projects_the_target_back_onto_the_grasp_plane():
    # [D17m] the robot may pull, never push into the tree: no target position past the plane through
    # the grasp TCP normal to the grip axis (+ margin). Lateral motion and orientation are untouched.
    from apple_pick_gym.batched_envs.harvest_action import clamp_target_advance

    start = torch.tensor([[0.0, 0.0, 0.0]] * 4)
    axis = torch.tensor([[0.0, 1.0, 0.0]] * 4)  # grip axis at reset: into the tree
    target = _identity_target(4)
    target[:, :3] = torch.tensor(
        [
            [0.01, -0.03, 0.02],  # pulled back: behind the wall
            [0.01, 0.00, 0.02],  # on the wall
            [0.01, 0.04, 0.02],  # pushed 40 mm in
            [0.05, 0.002, -0.01],  # 2 mm in
        ]
    )
    target[2, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    out = clamp_target_advance(target, start, axis, max_advance_m=0.0)
    expected = target.clone()
    expected[2, 1] = 0.0
    expected[3, 1] = 0.0
    torch.testing.assert_close(out, expected)
    # with a margin, up to the margin is allowed
    out5 = clamp_target_advance(target, start, axis, max_advance_m=0.005)
    torch.testing.assert_close(out5[3], target[3])
    torch.testing.assert_close(out5[2, :3], torch.tensor([0.01, 0.005, 0.02]))


def test_d17m_no_advance_wall_works_along_any_grip_axis_and_from_any_start():
    from apple_pick_gym.batched_envs.harvest_action import clamp_target_advance

    a = torch.nn.functional.normalize(torch.tensor([[1.0, 2.0, -0.5]]), dim=-1)
    start = torch.tensor([[0.3, -0.2, 1.1]])
    target = _identity_target(1)
    lateral = torch.cross(a, torch.tensor([[0.0, 0.0, 1.0]]), dim=-1)
    target[:, :3] = start + 0.07 * a + 0.02 * lateral
    out = clamp_target_advance(target, start, a, max_advance_m=0.0)
    torch.testing.assert_close(((out[:, :3] - start) * a).sum(-1), torch.zeros(1), atol=1e-6, rtol=0)
    torch.testing.assert_close(out[:, :3], start + 0.02 * lateral)


def test_d17m_action_bounds_wall_is_off_by_default():
    assert HarvestActionBounds().max_advance_m is None
    with pytest.raises(ValueError):
        HarvestActionBounds(max_advance_m=-0.01)


def test_d17m_grip_axis_is_tcp_z_from_an_xyzw_quat():
    from apple_pick_gym.batched_envs.harvest_action import grip_axis

    s = math.sqrt(0.5)
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0], [s, 0.0, 0.0, s]])  # identity; +90 deg about x
    torch.testing.assert_close(grip_axis(q), torch.tensor([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]))


# [D17n] commanded-force clamp: bounds ||K (target - tcp)|| so the policy cannot store a 75 N spring
def _tcp(n, quat_wxyz=(1.0, 0.0, 0.0, 0.0)):
    q = torch.tensor([quat_wxyz] * n, dtype=torch.float32)
    return torch.cat([torch.zeros(n, 3), q], dim=-1)


def test_d17n_force_clamp_is_a_no_op_under_the_cap():
    from apple_pick_gym.batched_envs.harvest_action import clamp_commanded_force

    tcp = _tcp(1)
    target = tcp.clone()
    target[:, :3] = torch.tensor([0.05, 0.0, 0.0])  # 500 N/m * 5 cm = 25 N
    out = clamp_commanded_force(target, tcp, torch.full((1, 3), 500.0), max_force_n=40.0, tool_frame=False)
    torch.testing.assert_close(out, target)


def test_d17n_force_clamp_scales_an_over_cap_offset_to_exactly_the_cap():
    from apple_pick_gym.batched_envs.harvest_action import clamp_commanded_force

    tcp = _tcp(1)
    target = tcp.clone()
    target[:, :3] = torch.tensor([0.15, 0.0, 0.0])  # 75 N
    target[:, 3:] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # orientation untouched
    out = clamp_commanded_force(target, tcp, torch.full((1, 3), 500.0), max_force_n=40.0, tool_frame=False)
    torch.testing.assert_close(out[:, :3], torch.tensor([[0.08, 0.0, 0.0]]))
    torch.testing.assert_close(out[:, 3:], target[:, 3:])


def test_d17n_force_clamp_uses_anisotropic_gains_in_the_tool_frame():
    import math

    from apple_pick_gym.batched_envs.harvest_action import clamp_commanded_force

    h = math.sqrt(0.5)
    tcp = _tcp(1, (h, 0.0, 0.0, h))  # TCP turned 90 deg about z: tool x = world y
    target = tcp.clone()
    target[:, :3] = torch.tensor([0.0, 0.1, 0.0])  # along tool x
    k = torch.tensor([[200.0, 600.0, 600.0]])  # tool x is soft: 20 N, under the cap
    out = clamp_commanded_force(target, tcp, k, max_force_n=40.0, tool_frame=True)
    torch.testing.assert_close(out, target)
    world = clamp_commanded_force(target, tcp, k, max_force_n=40.0, tool_frame=False)  # world y: 60 N
    torch.testing.assert_close(world[:, :3], torch.tensor([[0.0, 40.0 / 600.0, 0.0]]))


def test_d17n_force_clamp_is_off_by_default_and_validated():
    assert HarvestActionBounds().max_commanded_force_n is None
    with pytest.raises(ValueError):
        HarvestActionBounds(max_commanded_force_n=0.0)
