"""Observation flattening: proprioception+F/T actor vector + privileged critic vector.

The actor is scoped to proprioception and F/T only -- no vision-tracked
geometry (apple_pos/apple_quat/woody_part_start_pos/woody_part_end_pos are
available via info, not obs; see apple_pick_vic_harvest_env.py). The critic's
privileged junction-wrench block (dict[str, (N,6)]) still flattens
deterministically by sorted junction name, so its layout is stable across
runs and topologies with the same junction set.
"""

from __future__ import annotations

import torch

from apple_pick_gym.batched_envs.harvest_obs import (
    actor_obs_layout,
    critic_obs_layout,
    flatten_actor_obs,
    flatten_critic_obs,
)

_JUNCTIONS = ["stem_apple", "primary_spur", "spur_stem", "support"]


def _make_obs(n: int) -> dict:
    return {
        "tcp_pos": torch.randn(n, 3),
        "tcp_quat": torch.randn(n, 4),
        "tcp_velocity": torch.randn(n, 6),
        "ft_wrist": torch.randn(n, 6),
        "robot_joint_q": torch.randn(n, 7),
        "last_action": torch.randn(n, 13),
        "step_frac": torch.rand(n, 1),
    }


def _make_privileged(n: int) -> dict:
    return {
        "spur_flexural_modulus_pa": torch.rand(n, 1) * 1e8,
        "stem_flexural_modulus_pa": torch.rand(n, 1) * 1e8,
        "spur_youngs_modulus_pa": torch.rand(n, 1) * 1e8,
        "stem_youngs_modulus_pa": torch.rand(n, 1) * 1e8,
        "spur_damping_ratio": torch.rand(n, 1),
        "stem_damping_ratio": torch.rand(n, 1),
        "primary_density": torch.rand(n, 1) * 900,
        "support_kp": torch.rand(n, 1) * 1e6,
        "support_roll_kp": torch.rand(n, 1) * 7.5,
        "support_zeta": torch.rand(n, 1),
        "arm_armature": torch.rand(n, 7),
        "arm_friction": torch.rand(n, 7),
        "arm_joint_damping": torch.rand(n, 7),
        "arm_link_mass_scale": torch.rand(n, 1),
        "arm_link_inertia_scale": torch.rand(n, 1),
        "arm_ee_mass_scale": torch.rand(n, 1),
        "arm_ee_inertia_scale": torch.rand(n, 1),
        "detach_scale": torch.rand(n, 1),
        "realized_k_scale": torch.rand(n, 6),
    }


def _make_woody_force(n: int, junctions: list[str]) -> dict:
    return {j: torch.randn(n, 6) for j in junctions}


def test_actor_layout_has_documented_fixed_width():
    layout = actor_obs_layout()
    # 3+4+6+6+7 = 26 fixed fields (tcp_pos/quat/velocity, ft_wrist, robot_joint_q)
    # + 13 (last_action) + 1 (step_frac); no junction-keyed content in the actor.
    expected = 26 + 13 + 1
    assert layout.total_width == expected


def test_actor_layout_has_no_junction_or_geometry_fields():
    layout = actor_obs_layout()
    names = {e.name for e in layout.entries}
    assert names == {
        "tcp_pos",
        "tcp_quat",
        "tcp_velocity",
        "ft_wrist",
        "robot_joint_q",
        "last_action",
        "step_frac",
    }


def test_critic_layout_order_is_junction_name_sorted_not_dict_insertion_order():
    unsorted_junctions = ["support", "stem_apple", "primary_spur", "spur_stem"]
    layout_a = critic_obs_layout(unsorted_junctions)
    layout_b = critic_obs_layout(list(reversed(unsorted_junctions)))
    assert layout_a.entries == layout_b.entries


def test_flatten_actor_obs_matches_layout_width():
    obs = _make_obs(5)
    out = flatten_actor_obs(obs)
    layout = actor_obs_layout()
    assert out.shape == (5, layout.total_width)


def test_flatten_actor_obs_is_deterministic_for_same_input():
    obs = _make_obs(3)
    a = flatten_actor_obs(obs)
    b = flatten_actor_obs(obs)
    torch.testing.assert_close(a, b)


def test_flatten_actor_obs_places_known_fields_at_documented_slices():
    obs = _make_obs(2)
    out = flatten_actor_obs(obs)
    layout = actor_obs_layout()
    torch.testing.assert_close(out[:, layout.slice_for("tcp_pos")], obs["tcp_pos"])
    torch.testing.assert_close(out[:, layout.slice_for("ft_wrist")], obs["ft_wrist"])
    torch.testing.assert_close(out[:, layout.slice_for("robot_joint_q")], obs["robot_joint_q"])
    torch.testing.assert_close(out[:, layout.slice_for("last_action")], obs["last_action"])
    torch.testing.assert_close(out[:, layout.slice_for("step_frac")], obs["step_frac"])


def test_critic_layout_is_actor_layout_plus_privileged_fields():
    actor_layout = actor_obs_layout()
    critic_layout = critic_obs_layout(_JUNCTIONS)
    assert critic_layout.total_width > actor_layout.total_width
    # Strict prefix: every actor entry appears at the same (start, width) in critic.
    actor_by_name = {e.name: (e.start, e.width) for e in actor_layout.entries}
    critic_by_name = {e.name: (e.start, e.width) for e in critic_layout.entries}
    for name, slice_ in actor_by_name.items():
        assert critic_by_name[name] == slice_


def test_flatten_critic_obs_has_actor_obs_as_exact_prefix():
    obs = _make_obs(4)
    privileged = _make_privileged(4)
    woody_force = _make_woody_force(4, _JUNCTIONS)
    actor_out = flatten_actor_obs(obs)
    critic_out = flatten_critic_obs(obs, privileged, woody_force)
    assert critic_out.shape[0] == 4
    assert critic_out.shape[1] > actor_out.shape[1]
    torch.testing.assert_close(critic_out[:, : actor_out.shape[1]], actor_out)


def test_no_privileged_field_name_appears_in_actor_layout():
    """Explicit by-name check: none of the privileged/force field names leak
    into the actor's layout entries."""
    actor_layout = actor_obs_layout()
    actor_names = {e.name for e in actor_layout.entries}
    privileged = _make_privileged(1)
    for name in privileged:
        assert name not in actor_names, f"privileged field {name!r} leaked into actor layout"
    for j in _JUNCTIONS:
        assert f"woody_part_force/{j}" not in actor_names


def test_mutating_privileged_inputs_does_not_change_actor_output():
    """flatten_actor_obs never receives privileged data, so it cannot leak by
    construction; this test pins that contract."""
    obs = _make_obs(3)
    before = flatten_actor_obs(obs).clone()
    privileged = _make_privileged(3)
    woody_force = _make_woody_force(3, _JUNCTIONS)
    privileged["support_kp"] *= 1000.0
    woody_force["support"] *= 1000.0
    _ = flatten_critic_obs(obs, privileged, woody_force)
    after = flatten_actor_obs(obs)
    torch.testing.assert_close(before, after)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------------------------
# [D17] tool-frame actor observations: everything relative to the episode start and expressed in
# the current TCP frame, so the policy cannot memorise where the apple sits in the world.
# ---------------------------------------------------------------------------------------------

import math  # noqa: E402

import pytest  # noqa: E402

from apple_pick_gym.batched_envs.harvest_obs import quat_xyzw_to_rotvec, tool_frame_obs  # noqa: E402


def _q_xyzw(axis, angle):
    a = torch.tensor(axis, dtype=torch.float32)
    a = a / torch.linalg.norm(a)
    return torch.cat([a * math.sin(angle / 2), torch.tensor([math.cos(angle / 2)])]).unsqueeze(0)


def test_quat_to_rotvec_and_double_cover():
    q = _q_xyzw([0.0, 0.0, 1.0], 0.3)
    torch.testing.assert_close(quat_xyzw_to_rotvec(q), torch.tensor([[0.0, 0.0, 0.3]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(quat_xyzw_to_rotvec(-q), torch.tensor([[0.0, 0.0, 0.3]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(quat_xyzw_to_rotvec(torch.tensor([[0.0, 0.0, 0.0, 1.0]])), torch.zeros(1, 3))


def _tool_obs(q_now, p_now=(0.1, 0.2, 0.3), p_start=(0.1, 0.2, 0.3), q_start=None, v=(0.0,) * 6, ft=(0.0,) * 6,
              p_tgt=None, q_tgt=None):
    q_start = q_now if q_start is None else q_start
    p_tgt = p_now if p_tgt is None else p_tgt
    q_tgt = q_now if q_tgt is None else q_tgt
    t = lambda x: torch.tensor([x], dtype=torch.float32)
    return tool_frame_obs(
        tcp_pos=t(p_now), tcp_quat=q_now, tcp_velocity=t(v), ft=t(ft),
        start_pos=t(p_start), start_quat=q_start, target_pos=t(p_tgt), target_quat=q_tgt,
    )


def test_tool_frame_obs_identity_tcp_is_the_world_frame():
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    o = _tool_obs(q, p_now=(0.1, 0.2, 0.33), v=(1, 2, 3, 4, 5, 6), ft=(1, 0, 0, 0, 0, 2), p_tgt=(0.1, 0.25, 0.33))
    torch.testing.assert_close(o["d_pos_tool"], torch.tensor([[0.0, 0.0, 0.03]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(o["gravity_tool"], torch.tensor([[0.0, 0.0, -1.0]]))
    torch.testing.assert_close(o["twist_tool"], torch.tensor([[1.0, 2, 3, 4, 5, 6]]))
    torch.testing.assert_close(o["ft_tool"], torch.tensor([[1.0, 0, 0, 0, 0, 2]]))
    torch.testing.assert_close(o["target_offset_tool"], torch.tensor([[0.0, 0.05, 0.0, 0.0, 0.0, 0.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(o["d_rot_tool"], torch.zeros(1, 3))


def test_tool_frame_obs_rotates_world_vectors_into_the_tcp_frame():
    q = _q_xyzw([0.0, 0.0, 1.0], math.pi / 2)  # tool x = world y, tool y = world -x
    o = _tool_obs(q, v=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0), ft=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    torch.testing.assert_close(o["twist_tool"][:, :3], torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(o["ft_tool"][:, :3], torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6, rtol=0)
    # gripper pointing horizontally (tool z = world -y): gravity reads along tool -y... check via a tilt
    q_side = _q_xyzw([1.0, 0.0, 0.0], math.pi / 2)  # tool z = world -y, tool y = world z
    torch.testing.assert_close(_tool_obs(q_side)["gravity_tool"], torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6, rtol=0)


def test_tool_frame_obs_rotation_since_start_is_a_body_frame_rotvec():
    q0 = _q_xyzw([1.0, 0.0, 0.0], math.pi / 2)
    # rotate 0.2 rad about the tool's own z axis since the start: q_now = q0 * Rz(0.2)
    rz = _q_xyzw([0.0, 0.0, 1.0], 0.2)
    x0, y0, z0, w0 = q0[0]
    x1, y1, z1, w1 = rz[0]
    q_now = torch.tensor([[
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
    ]])
    o = _tool_obs(q_now, q_start=q0)
    torch.testing.assert_close(o["d_rot_tool"], torch.tensor([[0.0, 0.0, 0.2]]), atol=1e-5, rtol=0)


def test_tool_actor_layout_drops_absolute_pose_joints_and_clock():
    lay = actor_obs_layout("tool")
    names = [e.name for e in lay.entries]
    assert names == ["d_pos_tool", "d_rot_tool", "gravity_tool", "twist_tool", "ft_tool", "target_offset_tool", "last_action"]
    assert lay.total_width == 3 + 3 + 3 + 6 + 6 + 6 + 13
    assert actor_obs_layout() == actor_obs_layout("world")  # the D1-D16 layout is the default
    with pytest.raises(ValueError):
        actor_obs_layout("body")


def test_tool_critic_layout_has_the_tool_actor_prefix():
    jn = ["spur_stem", "stem_apple"]
    crit = critic_obs_layout(jn, frame="tool")
    act = actor_obs_layout("tool")
    assert crit.entries[: len(act.entries)] == act.entries
