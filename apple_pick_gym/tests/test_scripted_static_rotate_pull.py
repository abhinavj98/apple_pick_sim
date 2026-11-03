"""ScriptedStaticRotatePullPolicy: a fixed open-loop schedule -- turn the gripper in place by an angle, then pull straight
back along the (rotated) tool axis. No oracle input: it reads only gravity in the tool frame from the actor obs."""

from __future__ import annotations

import math
import types

import pytest
import torch

from apple_pick_gym.batched_envs.harvest_action import HarvestActionBounds
from apple_pick_gym.rl.action_scaling import HarvestActionScaler
from apple_pick_gym.rl.baselines import BASELINES, ScriptedStaticRotatePullPolicy, rotation_matrix


class _Layout:
    def slice_for(self, name):
        assert name == "gravity_tool"
        return slice(0, 3)


def _wrapper(n=2, bounds=None, frame="tool"):
    env = types.SimpleNamespace(obs_frame=frame)  # deliberately nothing else: no stem / junction / pose access
    return types.SimpleNamespace(
        _env=env, num_envs=n, device="cpu", actor_layout=_Layout(),
        action_scaler=HarvestActionScaler(bounds or HarvestActionBounds(linear_delta_m=0.004, angular_delta_rad=0.01)),
    )


# gravity (down) in the tool frame: +y; the outward grip axis is -z of the TCP
_G = torch.tensor([[0.0, 1.0, 0.0]])


def _act(policy, w, gravity=_G):
    obs = gravity.repeat(w.num_envs, 1)
    a = w.action_scaler.to_env(policy.act(w, obs, None))
    return a[:, :3], a[:, 3:6]


def test_registered():
    assert isinstance(BASELINES["scripted_static_rotate_pull"](), ScriptedStaticRotatePullPolicy)


def test_sideways_rotation_turns_the_pull_axis_toward_down_by_the_full_angle_within_the_env_cap():
    w = _wrapper()
    p = ScriptedStaticRotatePullPolicy(rotate_deg=30.0, axis="sideways", rate_m_per_step=0.001)
    p.reset(w)
    total = torch.zeros(3)
    first_pull = None
    for step in range(120):
        dp, drot = _act(p, w)
        assert float(drot[0].norm()) <= 0.01 + 1e-9
        if float(drot[0].norm()) == 0.0 and first_pull is None:
            first_pull = step
        total += drot[0]
        if first_pull is None:  # the gripper turns in place: no translation while rotating
            assert float(dp[0].norm()) == pytest.approx(0.0, abs=1e-9)
    assert math.degrees(float(total.norm())) == pytest.approx(30.0, abs=0.3)
    assert first_pull is not None and first_pull >= 52
    # the rotation axis is the horizontal sideways axis (tool x here) and +angle turns the outward axis (-z) toward down (+y)
    assert abs(float(total[0])) == pytest.approx(float(total.norm()), rel=1e-4)
    r = rotation_matrix(total[None] / total.norm(), torch.tensor([float(total.norm())]))[0]
    turned_out = r @ torch.tensor([0.0, 0.0, -1.0])
    assert float(turned_out[1]) > 0.0


def test_pull_is_a_constant_tool_frame_command_back_along_the_tool_axis():
    w = _wrapper()
    p = ScriptedStaticRotatePullPolicy(rotate_deg=30.0, rate_m_per_step=0.00105)
    p.reset(w)
    for _ in range(70):
        p.act(w, _G.repeat(2, 1), None)
    cmds = [_act(p, w) for _ in range(5)]
    for dp, drot in cmds:
        assert float(drot.norm()) == pytest.approx(0.0, abs=1e-9)
        torch.testing.assert_close(dp[0], torch.tensor([0.0, 0.0, -0.00105]), atol=1e-7, rtol=0)  # -z tool = outward


def test_rotation_axis_is_latched_at_the_first_step_so_the_schedule_is_static():
    w = _wrapper(n=1)
    p = ScriptedStaticRotatePullPolicy(rotate_deg=30.0)
    p.reset(w)
    _, d0 = _act(p, w)
    _, d1 = _act(p, w, gravity=torch.tensor([[0.3, 0.9, 0.1]]))  # gravity reading changes; the axis must not
    torch.testing.assert_close(d0, d1)


def test_outward_axis_mode_twists_about_the_tool_z_axis_and_sign_flips_the_direction():
    w = _wrapper(n=1)
    p = ScriptedStaticRotatePullPolicy(axis="outward", rotate_sign=1.0)
    q = ScriptedStaticRotatePullPolicy(axis="outward", rotate_sign=-1.0)
    p.reset(w)
    q.reset(w)
    _, dp_ = _act(p, w)
    _, dq_ = _act(q, w)
    assert abs(float(dp_[0, 2])) == pytest.approx(float(dp_.norm()), rel=1e-5)
    torch.testing.assert_close(dp_, -dq_)


def test_zero_rotation_is_a_plain_static_pull():
    w = _wrapper(n=1)
    p = ScriptedStaticRotatePullPolicy(rotate_deg=0.0, rate_m_per_step=0.001)
    p.reset(w)
    dp, drot = _act(p, w)
    assert float(drot.norm()) == 0.0
    torch.testing.assert_close(dp[0], torch.tensor([0.0, 0.0, -0.001]), atol=1e-7, rtol=0)


def test_needs_a_tool_frame_action():
    w = _wrapper(frame="world")
    p = ScriptedStaticRotatePullPolicy()
    p.reset(w)
    with pytest.raises(ValueError, match="tool"):
        p.act(w, _G.repeat(2, 1), None)


def test_pull_down_follows_the_live_gravity_direction_in_the_tool_frame():
    w = _wrapper(n=1)
    p = ScriptedStaticRotatePullPolicy(rotate_deg=0.0, pull_dir="down", rate_m_per_step=0.001)
    p.reset(w)
    dp, drot = _act(p, w)  # gravity along +y
    assert float(drot.norm()) == 0.0
    torch.testing.assert_close(dp[0], torch.tensor([0.0, 0.001, 0.0]), atol=1e-7, rtol=0)
    g = torch.tensor([[0.0, 0.6, 0.8]])  # the gripper has since turned: gravity reads differently, the pull follows it
    dp, _ = _act(p, w, gravity=g)
    torch.testing.assert_close(dp[0], torch.tensor([0.0, 0.0006, 0.0008]), atol=1e-7, rtol=0)


def test_tip_up_then_pull_down_turns_first_then_pulls_along_gravity():
    w = _wrapper(n=1)
    p = ScriptedStaticRotatePullPolicy(rotate_deg=30.0, axis="sideways", rotate_sign=-1.0, pull_dir="down", rate_m_per_step=0.001)
    p.reset(w)
    total = torch.zeros(3)
    for _ in range(70):
        dp, drot = _act(p, w)
        total += drot[0]
        if float(drot[0].norm()) > 0:
            assert float(dp[0].norm()) == pytest.approx(0.0, abs=1e-9)
    assert math.degrees(float(total.norm())) == pytest.approx(30.0, abs=0.3)
    # tip up: the outward axis (-z) turns toward up (-y) under the total rotation
    r = rotation_matrix(total[None] / total.norm(), torch.tensor([float(total.norm())]))[0]
    assert float((r @ torch.tensor([0.0, 0.0, -1.0]))[1]) < 0.0
    dp, drot = _act(p, w)
    assert float(drot.norm()) == 0.0
    torch.testing.assert_close(dp[0], torch.tensor([0.0, 0.001, 0.0]), atol=1e-7, rtol=0)


def test_bad_pull_dir_is_refused():
    with pytest.raises(ValueError, match="pull_dir"):
        ScriptedStaticRotatePullPolicy(pull_dir="sideways")


def test_world_back_pull_is_the_horizontal_outward_direction_whatever_the_tilt():
    """pull_dir='world_back': along (gravity x sideways axis) -- horizontal, away from the plant, not tilting with the gripper."""
    w = _wrapper(n=1)
    p = ScriptedStaticRotatePullPolicy(rotate_deg=0.0, pull_dir="world_back", rate_m_per_step=0.001)
    p.reset(w)
    dp, _ = _act(p, w)  # gravity +y, outward -z: horizontal outward is the tool -z axis
    torch.testing.assert_close(dp[0], torch.tensor([0.0, 0.0, -0.001]), atol=1e-7, rtol=0)
    g = torch.tensor([[0.0, 0.8, -0.6]])  # the gripper has tipped about its x axis: gravity reads differently in the tool frame
    dp, _ = _act(p, w, gravity=g)
    expect = torch.tensor([0.0, -0.6, -0.8]) * 0.001  # unit, perpendicular to gravity and to the tilt axis
    torch.testing.assert_close(dp[0], expect, atol=1e-7, rtol=0)
    assert abs(float(dp[0] @ g[0])) < 1e-9  # horizontal in the world


def test_world_back_needs_the_sideways_axis():
    with pytest.raises(ValueError, match="world_back"):
        ScriptedStaticRotatePullPolicy(axis="outward", pull_dir="world_back")
