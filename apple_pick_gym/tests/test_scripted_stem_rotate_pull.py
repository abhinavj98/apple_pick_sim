"""ScriptedStemRotatePullPolicy: swing the TCP 30 deg about the outward axis through the spur-stem junction, then pull along the stem."""

from __future__ import annotations

import math
import types

import numpy as np
import pytest
import torch

from apple_pick_gym.batched_envs.harvest_action import HarvestActionBounds, _rotate_by_quat_wxyz
from apple_pick_gym.rl.action_scaling import HarvestActionScaler
from apple_pick_gym.rl.baselines import BASELINES, ScriptedStemRotatePullPolicy, rotation_matrix


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


def test_rotation_matrix_matches_rodrigues():
    axis = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ang = torch.tensor([0.3, -0.7])
    r = rotation_matrix(axis, ang).numpy()
    for i in range(2):
        np.testing.assert_allclose(r[i], _rodrigues(axis[i].numpy(), float(ang[i])), atol=1e-6)


# TCP orientation whose -z (the outward grip axis) is world +x: rotate z onto -x (about y by -90 deg)
_Q_OUT_X = torch.tensor([[math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0]])


class _FakeEnv:
    """Hanging stem below the junction; the outward grip axis (-z of the TCP) is +x, world frame."""

    def __init__(self, n: int, tool_quat=None, quat=None):
        self.n = n
        self.obs_frame = "world" if tool_quat is None else "tool"
        q = _Q_OUT_X if quat is None else quat
        self.pivot = torch.tensor([[0.5, 0.0, 1.0]]).repeat(n, 1)
        self.stem_axis = torch.tensor([[0.0, 0.0, -1.0]]).repeat(n, 1)  # junction -> apple, straight down
        self._target_pose = torch.cat([self.pivot + self.stem_axis * 0.05, q.repeat(n, 1)], -1)
        self._last_tcp_pose_wxyz = self._target_pose.clone()
        self._last_full_obs = {"woody_part_info": {"spur_stem": {"anchors_pos": torch.cat([self.pivot * 0, self.pivot], -1)}}}

    def _target_stem_axis(self):
        return self.stem_axis


def _wrapper(env, bounds=None):
    return types.SimpleNamespace(
        _env=env, num_envs=env.n, device="cpu", action_scaler=HarvestActionScaler(bounds or HarvestActionBounds())
    )


def _world_delta(policy, wrapper):
    env_a = wrapper.action_scaler.to_env(policy.act(wrapper, None, None))
    return env_a[:, :3], env_a[:, 3:6]


def test_registered_baseline():
    assert isinstance(BASELINES["scripted_stem_rotate_pull"](), ScriptedStemRotatePullPolicy)


def test_swing_is_about_the_outward_axis_so_the_apple_goes_sideways():
    env = _FakeEnv(2)
    w = _wrapper(env)
    p = ScriptedStemRotatePullPolicy(rotate_deg=30.0, rotate_step_deg=2.0, rate_m_per_step=0.002)
    p.reset(w)
    dp, drot = _world_delta(p, w)
    axis = torch.nn.functional.normalize(drot, dim=-1)
    torch.testing.assert_close(axis, torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1), atol=1e-5, rtol=0)  # +outward
    assert float(dp[0, 0]) == pytest.approx(0.0, abs=1e-6)  # the arc stays in the plane normal to outward
    assert abs(float(dp[0, 1])) > 0.0  # the apple moves sideways


def test_swing_is_rigid_about_the_pivot_and_totals_thirty_degrees():
    env = _FakeEnv(1)
    w = _wrapper(env)
    p = ScriptedStemRotatePullPolicy(rotate_deg=30.0, rotate_step_deg=2.0, rate_m_per_step=0.002)
    p.reset(w)
    pos = env._target_pose[0, :3].clone()
    total = torch.zeros(3)
    for _ in range(15):
        dp, drot = _world_delta(p, w)
        pos = pos + dp[0]
        total = total + drot[0]
        env._target_pose[:, :3] = pos
    arm0 = torch.tensor([0.5, 0.0, 0.95]) - env.pivot[0]
    arm = pos - env.pivot[0]
    assert float(arm.norm()) == pytest.approx(float(arm0.norm()), rel=2e-3)  # radius about the pivot is kept
    cosang = float(arm @ arm0 / (arm.norm() * arm0.norm()))
    assert math.degrees(math.acos(cosang)) == pytest.approx(30.0, abs=0.2)
    assert math.degrees(float(total.norm())) == pytest.approx(30.0, abs=0.2)


def test_pull_follows_the_current_stem_axis_every_step():
    env = _FakeEnv(1)
    w = _wrapper(env)
    p = ScriptedStemRotatePullPolicy(rotate_deg=30.0, rotate_step_deg=2.0, rate_m_per_step=0.002)
    p.reset(w)
    for _ in range(15):
        p.act(w, None, None)
    for ang in (0.0, 20.0, -35.0):  # the stem moves; the pull direction follows it
        env.stem_axis = torch.tensor([[math.sin(math.radians(ang)), 0.0, -math.cos(math.radians(ang))]])
        dp, drot = _world_delta(p, w)
        assert float(drot.norm()) == pytest.approx(0.0, abs=1e-6)
        assert float(dp.norm()) == pytest.approx(0.002, rel=1e-3)
        torch.testing.assert_close(dp[0] / dp[0].norm(), env.stem_axis[0], atol=1e-4, rtol=0)


def test_the_swing_pivot_and_outward_axis_are_re_read_each_step():
    env = _FakeEnv(1)
    w = _wrapper(env)
    p = ScriptedStemRotatePullPolicy()
    p.reset(w)
    _, d0 = _world_delta(p, w)
    # the TCP turns so that outward is now +y; the next swing axis follows
    q = torch.tensor([[math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0]])  # z -> +y ... -z -> -y
    env._last_tcp_pose_wxyz[:, 3:7] = q
    out = -_rotate_by_quat_wxyz(q, torch.tensor([[0.0, 0.0, 1.0]]))
    _, d1 = _world_delta(p, w)
    torch.testing.assert_close(torch.nn.functional.normalize(d1, dim=-1), out, atol=1e-5, rtol=0)
    assert not torch.allclose(torch.nn.functional.normalize(d0, dim=-1), out)


def test_tool_frame_action_is_the_world_motion_in_the_tcp_frame():
    q = torch.tensor([[math.cos(0.4), 0.0, math.sin(0.4), 0.0]])  # a TCP pitched about y
    env_t = _FakeEnv(1, tool_quat=q, quat=q)
    env_w = _FakeEnv(1, quat=q)
    pt, pw = ScriptedStemRotatePullPolicy(), ScriptedStemRotatePullPolicy()
    wt, ww = _wrapper(env_t), _wrapper(env_w)
    pt.reset(wt)
    pw.reset(ww)
    dp_t, drot_t = _world_delta(pt, wt)
    dp_w, drot_w = _world_delta(pw, ww)
    torch.testing.assert_close(_rotate_by_quat_wxyz(q, dp_t), dp_w, atol=1e-5, rtol=0)
    torch.testing.assert_close(_rotate_by_quat_wxyz(q, drot_t), drot_w, atol=1e-5, rtol=0)


def test_gains_are_held_fixed():
    env = _FakeEnv(1)
    w = _wrapper(env)
    p = ScriptedStemRotatePullPolicy(k_lin=150.0, k_ang=20.0, zeta=0.9)
    p.reset(w)
    a1 = p.act(w, None, None)
    for _ in range(30):
        a2 = p.act(w, None, None)
    torch.testing.assert_close(a1[:, 6:], a2[:, 6:])


def test_swing_respects_the_env_rotation_bound_and_still_totals_the_full_angle():
    """The learned-policy env caps rotation at 0.01 rad / step: the swing must take more, smaller steps, not be clipped."""
    bounds = HarvestActionBounds(linear_delta_m=0.004, angular_delta_rad=0.01)
    env = _FakeEnv(1)
    w = _wrapper(env, bounds)
    p = ScriptedStemRotatePullPolicy(rotate_deg=30.0, rate_m_per_step=0.002)  # default 2 deg / step is over the bound
    p.reset(w)
    pos = env._target_pose[0, :3].clone()
    rot_total, first_pull = 0.0, None
    for step in range(120):
        dp, drot = _world_delta(p, w)
        assert float(drot.norm()) <= 0.01 + 1e-9
        pos = pos + dp[0]
        env._target_pose[:, :3] = pos
        rot_total += float(drot.norm())
        if first_pull is None and float(drot.norm()) == 0.0:
            first_pull = step
    assert math.degrees(rot_total) == pytest.approx(30.0, abs=0.3)  # orientation really turned by the swing angle
    assert first_pull is not None and first_pull >= 53  # 30 deg / 0.57 deg per step
