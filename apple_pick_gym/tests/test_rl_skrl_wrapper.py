"""HarvestSkrlWrapper over the surrogate env (same contract as ApplePickVicHarvestEnv)."""

from __future__ import annotations

import pytest
import torch

from apple_pick_gym.batched_envs.harvest_action import HarvestActionBounds
from apple_pick_gym.batched_envs.harvest_obs import actor_obs_layout
from apple_pick_gym.batched_envs.sensor_realism import FtSensorConfig
from apple_pick_gym.rl.action_scaling import HarvestActionScaler
from apple_pick_gym.rl.critic_state import critic_state_layout
from apple_pick_gym.rl.skrl_wrapper import HarvestSkrlWrapper
from apple_pick_gym.rl.surrogate_env import SurrogateHarvestEnv

N, T = 6, 10


class _Spy(SurrogateHarvestEnv):
    """Records the env-unit actions it receives and counts resets."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.received: list[torch.Tensor] = []
        self.resets = 0

    def reset(self, **kw):
        self.resets += 1
        return super().reset(**kw)

    def step(self, action):
        self.received.append(action.clone())
        return super().step(action)


def _wrapped(**kw):
    kw.setdefault("ft_sensor_config", FtSensorConfig())
    env = _Spy(num_envs=N, max_episode_steps=T, seed=0, **kw)
    return env, HarvestSkrlWrapper(env)


def test_spaces_and_skrl_api():
    env, w = _wrapped()
    assert w.num_envs == N and w.num_agents == 1
    assert w.observation_space.shape == (actor_obs_layout().total_width,)
    layout = critic_state_layout(env.junction_names)
    assert w.state_space.shape == (layout.total_width,)
    assert w.action_space.shape == (13,)
    assert float(w.action_space.low.min()) == -1.0 and float(w.action_space.high.max()) == 1.0
    obs, info = w.reset()
    assert obs.shape == (N, actor_obs_layout().total_width)
    assert w.state().shape == (N, layout.total_width)


def test_critic_state_has_actor_obs_prefix_and_named_extras():
    env, w = _wrapped()
    obs, _ = w.reset()
    st = w.state()
    torch.testing.assert_close(st[:, : obs.shape[1]], obs)
    layout = critic_state_layout(env.junction_names)
    names = [e.name for e in layout.entries]
    for extra in ("weld_direction", "target_junction_wrench", "detach_index", "raw_ft_wrist", "frozen", "invalid"):
        assert extra in names, extra
    torch.testing.assert_close(st[:, layout.slice_for("weld_direction")], env.weld)


def test_policy_actions_reach_the_env_in_env_units():
    env, w = _wrapped()
    w.reset()
    u = torch.rand(N, 13, generator=torch.Generator().manual_seed(0)) * 2 - 1
    w.step(u)
    torch.testing.assert_close(env.received[-1], HarvestActionScaler(env.action_bounds).to_env(u))


def test_state_pairs_with_the_obs_just_returned_every_step_and_after_autoreset():
    env, w = _wrapped()
    w.reset()
    for _ in range(T + 3):
        obs, *_ = w.step(torch.zeros(N, 13))
        torch.testing.assert_close(w.state()[:, : obs.shape[1]], obs)


def test_time_limit_step_is_terminated_and_truncated_and_next_obs_is_the_reset_obs():
    env, w = _wrapped()
    w.reset()
    for t in range(T):
        obs, r, term, trunc, info = w.step(torch.zeros(N, 13))
        if t < T - 1:
            assert not bool(trunc.any())
    assert bool(trunc.all()) and bool(term.all())  # time is observed -> no bootstrap across it
    assert env.resets == 2  # the wrapper auto-reset the whole batch
    step_frac = obs[:, actor_obs_layout().slice_for("step_frac")]
    assert float(step_frac.abs().max()) == 0.0  # first obs of the new episode


def test_freeze_edge_terminates_once_mid_episode():
    env, w = _wrapped(action_bounds=HarvestActionBounds(max_target_pos_offset_m=0.12))
    env._max_episode_steps = 200
    w.reset()
    terms = []
    for _ in range(150):
        scaler = HarvestActionScaler(env.action_bounds)
        a = scaler.to_policy(torch.cat([env.weld * 0.004, torch.zeros(N, 3), torch.full((N, 3), 300.0), torch.full((N, 3), 20.0), torch.ones(N, 1)], -1))
        _, _, term, trunc, _ = w.step(a)
        terms.append(term.flatten())
        assert not bool(trunc.any())
    assert torch.stack(terms).sum(0).tolist() == [1] * N


def test_episode_stats_are_emitted_once_per_episode_as_scalars():
    env, w = _wrapped()
    w.reset()
    for t in range(T):
        _, _, _, _, info = w.step(torch.zeros(N, 13))
        if t < T - 1:
            assert "log" not in info or not any(k.startswith("Episode /") for k in info["log"])
    log = info["log"]
    for key in (
        "Episode / success rate",
        "Episode / safety rate",
        "Episode / invalid fraction",
        "Episode / return (mean)",
        "Episode / peak detach index (mean)",
        "Episode / peak collateral N (mean)",
        "Episode / steps to success (mean)",
    ):
        assert key in log, key
        assert isinstance(log[key], torch.Tensor) and log[key].numel() == 1
    assert float(log["Episode / success rate"]) == 0.0


def test_invalid_envs_are_excluded_from_rates():
    env, w = _wrapped(invalid_fraction=0.5)
    w.reset()
    for _ in range(T):
        _, _, _, _, info = w.step(torch.zeros(N, 13))
    assert float(info["log"]["Episode / invalid fraction"]) == pytest.approx(0.5)
    st = w.state()
    layout = critic_state_layout(env.junction_names)
    torch.testing.assert_close(st[:, layout.slice_for("invalid")].flatten(), env.invalid_env_mask.float())
