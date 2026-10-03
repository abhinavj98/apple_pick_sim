"""[D17] SurrogateBendPullEnv: the v2 task (equator grasp, bend-and-pull detach, tool frame) on
an analytic pendulum-stem plant. Like the v1 surrogate, its numbers mean nothing about the plant;
its *structure* must match the task: bending the stem lowers the detach force, so bend-then-pull
detaches at lower tension and collateral than a straight pull.
"""

from __future__ import annotations

import math

import pytest
import torch

from apple_pick_gym.batched_envs.harvest_detach import DetachEnvelopeConfig
from apple_pick_gym.batched_envs.harvest_episode import EpisodeConfig
from apple_pick_gym.batched_envs.harvest_obs import actor_obs_layout, flatten_actor_obs
from apple_pick_gym.batched_envs.harvest_reward import HarvestRewardConfig, quat_rotate_vector
from apple_pick_gym.batched_envs.sensor_realism import FtSensorConfig
from apple_pick_gym.rl.surrogate_env import SurrogateBendPullEnv

N = 8
_BOUNDS = dict(linear_delta_m=0.4 / 60, angular_delta_rad=0.3 / 60, k_lin_max=500.0)


def _env(**kw):
    from apple_pick_gym.batched_envs.harvest_action import HarvestActionBounds

    kw.setdefault("ft_sensor_config", FtSensorConfig())
    kw.setdefault("action_bounds", HarvestActionBounds(**_BOUNDS))
    kw.setdefault(
        "reward_config",
        HarvestRewardConfig(detach=DetachEnvelopeConfig(criterion="bend_pull", detach_scale_range=None)),
    )
    kw.setdefault("episode_config", EpisodeConfig(success_streak_steps=3, safety_force_cap_n=1e12, safety_torque_cap_nm=1e12))
    return SurrogateBendPullEnv(num_envs=N, max_episode_steps=kw.pop("steps", 200), seed=kw.pop("seed", 0), **kw)


def _act(dp=(0.0, 0.0, 0.0), drot=(0.0, 0.0, 0.0), k_lin=300.0, k_ang=20.0, zeta=1.0):
    a = torch.zeros(N, 13)
    a[:, :3] = torch.as_tensor(dp, dtype=torch.float32)
    a[:, 3:6] = torch.as_tensor(drot, dtype=torch.float32)
    a[:, 6:9], a[:, 9:12], a[:, 12] = k_lin, k_ang, zeta
    return a


def test_rest_state_is_far_from_detaching_and_publishes_the_bend_pull_state():
    env = _env()
    obs, info = env.reset()
    torch.testing.assert_close(info["theta_b_deg"], torch.zeros(N), atol=1e-4, rtol=0)
    assert bool((info["stem_tension"] > 2.0).all())
    assert float(info["detach_utilization"].max()) < 0.3
    for k in ("detach_force", "detach_scale", "detach_index"):
        assert info[k].shape == (N,)
    assert env.obs_frame == "tool"
    # tool +z points into the apple (as in the real env): -z_tool is the weld (apple -> gripper)
    z = quat_rotate_vector(obs["tcp_quat"], torch.tensor([[0.0, 0.0, 1.0]] * N))
    torch.testing.assert_close(-z, env.plant_geometry()["weld_direction"], atol=1e-5, rtol=0)
    # the equator cone: grasps within 60 deg of world -y
    w = env.plant_geometry()["weld_direction"]
    assert float(torch.rad2deg(torch.arccos(torch.clamp(-w[:, 1], -1, 1))).max()) <= 60.0 + 1e-4


def test_pulling_down_loads_the_stem_and_a_lateral_move_bends_it():
    env = _env(action_frame="world")
    env.reset()
    for _ in range(15):
        _, _, _, _, info = env.step(_act(dp=(0.0, 0.0, -0.002)))
    assert float(info["stem_tension"].min()) > 6.0
    assert float(info["theta_b_deg"].max()) < 5.0
    env.reset()
    for _ in range(15):
        _, _, _, _, info = env.step(_act(dp=(0.002, 0.0, 0.0)))  # along the branch, sideways to the stem
    assert float(info["theta_b_deg"].min()) > 15.0
    assert bool((info["detach_force"] < 15.0).all())  # bending lowers the pull needed


def _pick(env, script, steps=200):
    env.reset()
    peak_t = torch.zeros(N)
    peak_c = torch.zeros(N)
    done = torch.zeros(N, dtype=torch.bool)
    for t in range(steps):
        _, _, _, _, info = env.step(script(t))
        live = ~done
        peak_t = torch.where(live, torch.maximum(peak_t, info["stem_tension"]), peak_t)
        peak_c = torch.where(live, torch.maximum(peak_c, info["reward_terms"]["raw"]["collateral"]), peak_c)
        done |= info["episode"]["success_achieved"]
    return done, peak_t, peak_c


def test_bend_then_pull_detaches_with_less_tension_and_collateral_than_a_straight_pull():
    env = _env(action_frame="world")
    straight = lambda t: _act(dp=(0.0, 0.0, -0.002))
    bend_pull = lambda t: _act(dp=(0.003, 0.0, 0.0)) if t < 12 else _act(dp=(0.0, 0.0, -0.002))
    ok_s, t_s, c_s = _pick(env, straight)
    ok_b, t_b, c_b = _pick(env, bend_pull)
    assert bool(ok_s.all()) and bool(ok_b.all())
    assert float(t_b.mean()) < 0.5 * float(t_s.mean())
    assert float(c_b.mean()) < float(c_s.mean())


def test_tool_frame_obs_and_a_tool_z_pull_moves_out_along_the_grasp():
    env = _env()
    obs, info = env.reset()
    lay = actor_obs_layout("tool")
    assert flatten_actor_obs(obs, "tool").shape == (N, lay.total_width)
    torch.testing.assert_close(obs["d_pos_tool"], torch.zeros(N, 3), atol=1e-6, rtol=0)
    weld = env.plant_geometry()["weld_direction"]
    x0 = obs["tcp_pos"].clone()
    for _ in range(10):
        obs, *_ = env.step(_act(dp=(0.0, 0.0, -0.002), k_lin=500.0))
    d = obs["tcp_pos"] - x0
    cos = (torch.nn.functional.normalize(d, dim=-1) * weld).sum(-1)
    assert float(cos.min()) > 0.8


def test_realized_stiffness_dr_resamples_per_reset_and_is_privileged():
    env = _env(realized_k_scale_range=(0.8, 1.2))
    env.reset()
    k0 = env.privileged_fields()["realized_k_scale"].clone()
    assert bool(((k0 >= 0.8) & (k0 <= 1.2)).all())
    env.reset()
    assert not torch.equal(env.privileged_fields()["realized_k_scale"], k0)


def test_detach_scale_is_drawn_per_reset():
    env = _env(reward_config=HarvestRewardConfig(detach=DetachEnvelopeConfig(criterion="bend_pull")))
    _, info = env.reset()
    s0 = info["detach_scale"].clone()
    assert bool(((s0 >= 0.85) & (s0 <= 1.15)).all())
    _, info = env.reset()
    assert not torch.equal(info["detach_scale"], s0)
    torch.testing.assert_close(env.privileged_fields()["detach_scale"], info["detach_scale"].reshape(N, 1))


def test_stays_finite_under_random_actions_and_extreme_gains():
    env = _env()
    env.reset()
    g = torch.Generator().manual_seed(0)
    for t in range(150):
        a = (torch.rand(N, 13, generator=g) * 2 - 1)
        a[:, :6] *= 0.01
        a[:, 6:9] = 20.0 if t % 2 else 500.0
        a[:, 9:12] = 2.0 if t % 3 else 40.0
        a[:, 12] = 0.3 if t % 5 else 2.0
        obs, r, *_ , info = env.step(a)
        assert bool(torch.isfinite(r).all()) and bool(torch.isfinite(info["stem_tension"]).all())
        assert bool(torch.isfinite(flatten_actor_obs(obs, "tool")).all())


def test_envelope_criterion_is_refused():
    with pytest.raises(ValueError, match="bend_pull"):
        _env(reward_config=HarvestRewardConfig())
