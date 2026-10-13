"""HarvestSkrlWrapper over the surrogate env (same contract as ApplePickVicHarvestEnv)."""

from __future__ import annotations

import math

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


class _NanEnv(SurrogateHarvestEnv):
    """Env 1 blows up on every step (NaN obs, reward and forces)."""

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        obs["tcp_pos"][1] = float("nan")
        r[1] = float("nan")
        info["woody_part_force"]["stem_apple"][1] = float("inf")
        return obs, r, term, trunc, info


def test_nonfinite_env_rows_are_sanitized_zero_rewarded_and_counted():
    env = _NanEnv(num_envs=N, max_episode_steps=T, seed=0, ft_sensor_config=FtSensorConfig())
    w = HarvestSkrlWrapper(env)
    w.reset()
    obs, r, _, _, info = w.step(torch.zeros(N, 13))
    assert bool(torch.isfinite(obs).all()) and bool(torch.isfinite(w.state()).all())
    assert bool(torch.isfinite(r).all()) and float(r[1]) == 0.0
    assert float(info["log"]["Step / nonfinite envs"]) == 1.0


def test_d6_success_conditioned_peak_collateral():
    # [D6] peak collateral averaged over successful (valid) envs only; NaN when none succeeded
    from apple_pick_gym.rl.skrl_wrapper import _EpisodeStats

    st = _EpisodeStats(4, torch.device("cpu"))
    st.reset(torch.tensor([False, False, False, True]))
    st.peak_coll = torch.tensor([10.0, 30.0, 50.0, 70.0])
    st.success = torch.tensor([True, False, True, True])  # env 3 is invalid
    s = st.summary()
    assert float(s["Episode / peak collateral N (mean)"]) == pytest.approx(30.0)
    assert float(s["Episode / peak collateral N, successful (mean)"]) == pytest.approx(30.0)  # (10 + 50) / 2
    st.success = torch.zeros(4, dtype=torch.bool)
    assert torch.isnan(st.summary()["Episode / peak collateral N, successful (mean)"])


def test_episode_stats_report_tcp_speed():
    # how fast the policy moves the arm: peak and mean (over live steps) TCP linear speed
    env, w = _wrapped()
    w.reset()
    for _ in range(T):
        _, _, _, _, info = w.step(torch.zeros(N, 13))
    still = info["log"]
    assert float(still["Episode / peak TCP speed m/s (mean)"]) < 1e-3
    env2, w2 = _wrapped()
    w2.reset()
    a = torch.zeros(N, 13)
    a[:, 0] = 1.0  # full-scale +x target step every step
    for _ in range(T):
        _, _, _, _, info = w2.step(a)
    moving = info["log"]
    peak, mean = float(moving["Episode / peak TCP speed m/s (mean)"]), float(moving["Episode / mean TCP speed m/s (mean)"])
    assert peak > 0.01 and 0.0 < mean <= peak + 1e-6


def test_episode_stats_report_peak_force_per_junction():
    # which junctions the collateral load sits on (series statics: stem_apple / primary_spur carry the pull)
    env, w = _wrapped()
    w.reset()
    for _ in range(T):
        _, _, _, _, info = w.step(torch.zeros(N, 13))
    log = info["log"]
    for name in env.junction_names:
        v = log[f"Episode / peak force {name} N (mean)"]
        assert isinstance(v, torch.Tensor) and v.numel() == 1 and float(v) >= 0.0


def test_episode_stats_report_target_wrench_at_detach_and_peak_torque():
    # [maintainer] how the target junction fails: force vs torque (torsion / bending) at the detach step
    from apple_pick_gym.batched_envs.harvest_detach import DetachEnvelopeConfig
    from apple_pick_gym.rl.skrl_wrapper import _EpisodeStats

    st = _EpisodeStats(2, torch.device("cpu"), detach=DetachEnvelopeConfig())
    w = torch.tensor([[16.0, 0.0, 0.0, 0.03, 0.0, 0.04], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    info = {
        "episode": {
            "frozen": torch.tensor([False, False]),
            "success_achieved": torch.tensor([True, False]),
            "safety_junction": torch.tensor([False, False]),
            "safety_wrist": torch.tensor([False, False]),
        },
        "reward_terms": {
            "raw": {"collateral": torch.zeros(2)},
            "weighted": {k: torch.zeros(2) for k in ("progress", "pullout", "wrist", "collateral", "slack")},
            "terminal": torch.zeros(2),
        },
        "detach_index": torch.ones(2),
        "target_junction_wrench": w,
        "target_junction_axis": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        "ft_wrist": torch.zeros(2, 6),
        "woody_part_force": {},
    }
    st.update(torch.zeros(2, 1), info, torch.tensor([True, False]), torch.zeros(2, 13))
    s = st.summary()
    assert float(s["Episode / detach force N (mean)"]) == pytest.approx(16.0)
    assert float(s["Episode / detach torque N*m (mean)"]) == pytest.approx(0.05)
    assert float(s["Episode / detach torsion N*m (mean)"]) == pytest.approx(0.04)
    assert float(s["Episode / detach bending N*m (mean)"]) == pytest.approx(0.03)
    assert float(s["Episode / detach force share (mean)"]) == pytest.approx(0.64)  # (16/20)^2
    assert float(s["Episode / detach torque share (mean)"]) == pytest.approx(1.0)  # (0.05/0.05)^2
    assert float(s["Episode / peak target torque N*m (mean)"]) == pytest.approx(0.025)  # mean of 0.05, 0


def test_episode_stats_report_which_safety_cap_tripped_and_peak_wrist_torque():
    from apple_pick_gym.batched_envs.harvest_episode import EpisodeConfig
    from apple_pick_gym.rl.skrl_wrapper import _EpisodeStats

    st = _EpisodeStats(4, torch.device("cpu"), safety=EpisodeConfig())  # caps 40 N / 10 N*m
    tw = torch.zeros(4, 6)
    tw[0, 0] = 50.0  # target force cap
    ft = torch.zeros(4, 6)
    ft[1, 3] = 12.0  # wrist torque cap
    ft[2, 3] = 3.0  # below every cap
    info = {
        "episode": {
            "frozen": torch.zeros(4, dtype=torch.bool),
            "success_achieved": torch.zeros(4, dtype=torch.bool),
            "safety_junction": torch.tensor([True, False, False, False]),
            "safety_wrist": torch.tensor([False, True, False, False]),
        },
        "reward_terms": {
            "raw": {"collateral": torch.zeros(4)},
            "weighted": {k: torch.zeros(4) for k in ("progress", "pullout", "wrist", "collateral", "slack")},
            "terminal": torch.zeros(4),
        },
        "detach_index": torch.zeros(4),
        "target_junction_wrench": tw,
        "ft_wrist": ft,
        "woody_part_force": {},
    }
    st.update(torch.zeros(4, 1), info, torch.tensor([True, True, False, False]), torch.zeros(4, 13))
    s = st.summary()
    assert float(s["Episode / safety target force (frac)"]) == pytest.approx(0.25)
    assert float(s["Episode / safety target torque (frac)"]) == 0.0
    assert float(s["Episode / safety wrist force (frac)"]) == 0.0
    assert float(s["Episode / safety wrist torque (frac)"]) == pytest.approx(0.25)
    assert float(s["Episode / peak wrist torque N*m (mean)"]) == pytest.approx((12.0 + 3.0) / 4)


def test_episode_stats_record_target_force_at_and_before_a_safety_trip():
    # tells a one-step solver blow-up (10 N -> 900 N) from a policy that ramps into the cap
    from apple_pick_gym.batched_envs.harvest_episode import EpisodeConfig
    from apple_pick_gym.rl.skrl_wrapper import _EpisodeStats

    st = _EpisodeStats(2, torch.device("cpu"), safety=EpisodeConfig())

    def info(f0, f1, safe):
        tw = torch.zeros(2, 6)
        tw[0, 0], tw[1, 0] = f0, f1
        return {
            "episode": {
                "frozen": torch.zeros(2, dtype=torch.bool),
                "success_achieved": torch.zeros(2, dtype=torch.bool),
                "safety_junction": torch.tensor(safe),
                "safety_wrist": torch.zeros(2, dtype=torch.bool),
            },
            "reward_terms": {
                "raw": {"collateral": torch.zeros(2)},
                "weighted": {k: torch.zeros(2) for k in ("progress", "pullout", "wrist", "collateral", "slack")},
                "terminal": torch.zeros(2),
            },
            "detach_index": torch.zeros(2),
            "target_junction_wrench": tw,
            "ft_wrist": torch.zeros(2, 6),
            "woody_part_force": {},
        }

    st.update(torch.zeros(2, 1), info(10.0, 38.0, [False, False]), torch.zeros(2, dtype=torch.bool), torch.zeros(2, 13))
    st.update(torch.zeros(2, 1), info(900.0, 41.0, [True, True]), torch.ones(2, dtype=torch.bool), torch.zeros(2, 13))
    s = st.summary()
    assert float(s["Episode / safety trip target force N (median)"]) == pytest.approx(470.5)  # median of 900, 41
    assert float(s["Episode / safety trip prev target force N (median)"]) == pytest.approx(24.0)  # median of 10, 38
    assert float(s["Episode / safety trip target force jump > 5x (frac of trips)"]) == pytest.approx(0.5)


def test_d12_blowup_worlds_are_excluded_from_rates_and_counted():
    from apple_pick_gym.rl.skrl_wrapper import _EpisodeStats

    st = _EpisodeStats(4, torch.device("cpu"))
    info = {
        "episode": {
            "frozen": torch.zeros(4, dtype=torch.bool),
            "success_achieved": torch.tensor([True, False, False, False]),
            "safety_junction": torch.zeros(4, dtype=torch.bool),
            "safety_wrist": torch.zeros(4, dtype=torch.bool),
            "blowup": torch.tensor([False, True, False, False]),
        },
        "reward_terms": {
            "raw": {"collateral": torch.zeros(4)},
            "weighted": {k: torch.zeros(4) for k in ("progress", "pullout", "wrist", "collateral", "slack")},
            "terminal": torch.zeros(4),
        },
        "detach_index": torch.zeros(4),
        "target_junction_wrench": torch.zeros(4, 6),
        "ft_wrist": torch.zeros(4, 6),
        "woody_part_force": {},
    }
    st.update(torch.zeros(4, 1), info, torch.tensor([True, True, False, False]), torch.zeros(4, 13))
    s = st.summary()
    assert float(s["Episode / blowup fraction"]) == pytest.approx(0.25)
    assert float(s["Episode / success rate"]) == pytest.approx(1.0 / 3.0)  # 1 of the 3 non-blown-up envs


class _BlowupEnv(SurrogateHarvestEnv):
    """Env 1 blows up at its 3rd step: flagged by the outcome, garbage wrist reading from then on."""

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        if self._step_count >= 3:
            obs["ft_wrist"][1] = 1900.0
            info["episode"]["blowup"][1] = self._step_count == 3
        return obs, r, term, trunc, info


def test_d14_blown_up_world_emits_its_last_good_obs_and_state_until_reset():
    # [D14] keeps post-blow-up garbage out of the RunningStandardScaler stats and the PPO batch
    env = _BlowupEnv(num_envs=N, max_episode_steps=T, seed=0, ft_sensor_config=FtSensorConfig())
    w = HarvestSkrlWrapper(env)
    w.reset()
    a = torch.zeros(N, 13)
    a[:, 0] = 0.5
    for _ in range(2):
        obs, *_ = w.step(a)
    good_obs, good_state = obs[1].clone(), w.state()[1].clone()
    for _ in range(3):
        obs, *_ = w.step(a)
        torch.testing.assert_close(obs[1], good_obs)
        torch.testing.assert_close(w.state()[1], good_state)
        assert float(obs.abs().max()) < 1000.0
    assert not torch.equal(obs[0], good_obs)  # healthy worlds still update
    w.reset()
    assert not torch.equal(w._held, torch.ones_like(w._held))  # released at reset


def test_success_collateral_distribution():
    # median / p10 / p90 and low-force fractions over successful valid envs (NaN when none)
    from apple_pick_gym.rl.skrl_wrapper import _EpisodeStats

    st = _EpisodeStats(6, torch.device("cpu"))
    st.reset(torch.tensor([False] * 5 + [True]))
    st.peak_coll = torch.tensor([10.0, 20.0, 30.0, 40.0, 99.0, 5.0])
    st.success = torch.tensor([True, True, True, True, False, True])  # env 5 invalid, env 4 failed
    s = st.summary()
    k = "Episode / peak collateral N, successful"
    assert float(s[f"{k} (median)"]) == pytest.approx(25.0)
    assert float(s[f"{k} (p10)"]) == pytest.approx(13.0)
    assert float(s[f"{k} (p90)"]) == pytest.approx(37.0)
    assert float(s[f"{k} < 15 N (frac)"]) == pytest.approx(0.25)
    assert float(s[f"{k} < 22 N (frac)"]) == pytest.approx(0.5)
    st.success = torch.zeros(6, dtype=torch.bool)
    s = st.summary()
    assert all(torch.isnan(s[f"{k} {q}"]) for q in ("(median)", "(p10)", "(p90)", "< 15 N (frac)", "< 22 N (frac)"))


def test_d17_episode_stats_report_bend_pull_state_at_detach_and_over_40n_peaks():
    # [D17] with the safety trip off, forces over the old 40 N cap are reported, not penalised;
    # the bend-pull criterion's bending angle and stem tension are logged at detach and at peak.
    from apple_pick_gym.rl.skrl_wrapper import _EpisodeStats

    st = _EpisodeStats(3, torch.device("cpu"))
    tw = torch.zeros(3, 6)
    tw[:, 2] = torch.tensor([5.0, 45.0, 1.0])
    ft = torch.zeros(3, 6)
    ft[:, 0] = torch.tensor([50.0, 10.0, 1.0])
    info = {
        "episode": {
            "frozen": torch.zeros(3, dtype=torch.bool),
            "success_achieved": torch.tensor([True, False, False]),
            "safety_junction": torch.zeros(3, dtype=torch.bool),
            "safety_wrist": torch.zeros(3, dtype=torch.bool),
        },
        "reward_terms": {
            "raw": {"collateral": torch.zeros(3)},
            "weighted": {k: torch.zeros(3) for k in ("progress", "pullout", "wrist", "collateral", "slack")},
            "terminal": torch.zeros(3),
        },
        "detach_index": torch.tensor([1.0, 0.5, 0.1]),
        "theta_b_deg": torch.tensor([40.0, 12.0, 0.0]),
        "stem_tension": torch.tensor([4.0, 30.0, 2.0]),
        "target_junction_wrench": tw,
        "ft_wrist": ft,
        "woody_part_force": {},
    }
    st.update(torch.zeros(3, 1), info, torch.tensor([True, False, False]), torch.zeros(3, 13))
    s = st.summary()
    assert float(s["Episode / detach theta_b deg (mean)"]) == pytest.approx(40.0)
    assert float(s["Episode / detach stem tension N (mean)"]) == pytest.approx(4.0)
    assert float(s["Episode / peak theta_b deg (mean)"]) == pytest.approx(52.0 / 3)
    assert float(s["Episode / peak wrist force > 40 N (frac)"]) == pytest.approx(1 / 3)
    assert float(s["Episode / peak target force > 40 N (frac)"]) == pytest.approx(1 / 3)


def test_d17_episode_stats_decompose_the_motion_at_detach():
    # [D17] what kind of motion detaches: TCP displacement since the start split into along-grip
    # (out = -tool z), down (gravity) and the remaining sideways part; rotation into twist about the
    # grip and tilt; and the commanded stiffness per tool axis.
    from apple_pick_gym.rl.skrl_wrapper import _EpisodeStats

    st = _EpisodeStats(2, torch.device("cpu"))
    obs = {
        # env0: tool z horizontal, gravity along tool -y; moved 30 mm out, 20 mm down, 40 mm sideways (x)
        "d_pos_tool": torch.tensor([[0.04, -0.02, -0.03], [0.0, 0.0, 0.0]]),
        "gravity_tool": torch.tensor([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]]),
        "d_rot_tool": torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.0]]),  # 0.1 rad twist about the grip
    }
    info = {
        "episode": {
            "frozen": torch.zeros(2, dtype=torch.bool),
            "success_achieved": torch.tensor([True, False]),
            "safety_junction": torch.zeros(2, dtype=torch.bool),
            "safety_wrist": torch.zeros(2, dtype=torch.bool),
        },
        "reward_terms": {
            "raw": {"collateral": torch.zeros(2)},
            "weighted": {k: torch.zeros(2) for k in ("progress", "pullout", "wrist", "collateral", "slack")},
            "terminal": torch.zeros(2),
        },
        "detach_index": torch.tensor([1.0, 0.1]),
        "target_junction_wrench": torch.zeros(2, 6),
        "ft_wrist": torch.zeros(2, 6),
        "woody_part_force": {},
    }
    act = torch.zeros(2, 13)
    act[:, 6:9] = torch.tensor([50.0, 120.0, 150.0])
    st.update(torch.zeros(2, 1), info, torch.tensor([True, False]), act, obs=obs)
    s = st.summary()
    assert float(s["Episode / detach motion out along grip mm (mean)"]) == pytest.approx(30.0)
    assert float(s["Episode / detach motion down mm (mean)"]) == pytest.approx(20.0)
    assert float(s["Episode / detach motion sideways mm (mean)"]) == pytest.approx(40.0)
    assert float(s["Episode / detach motion total mm (mean)"]) == pytest.approx(math.sqrt(30**2 + 20**2 + 40**2), rel=1e-5)
    assert float(s["Episode / detach twist about grip deg (mean)"]) == pytest.approx(math.degrees(0.1), rel=1e-5)
    assert float(s["Episode / detach tilt deg (mean)"]) == pytest.approx(0.0, abs=1e-6)
    assert float(s["Episode / detach K_lin tool x (mean)"]) == pytest.approx(50.0)
    assert float(s["Episode / detach K_lin tool z grip (mean)"]) == pytest.approx(150.0)


def _advance_info(adv, frozen=(False, False)):
    return {
        "episode": {
            "frozen": torch.tensor(frozen),
            "success_achieved": torch.zeros(2, dtype=torch.bool),
            "safety_junction": torch.zeros(2, dtype=torch.bool),
            "safety_wrist": torch.zeros(2, dtype=torch.bool),
        },
        "reward_terms": {
            "raw": {"collateral": torch.zeros(2)},
            "weighted": {k: torch.zeros(2) for k in ("progress", "pullout", "wrist", "collateral", "slack")},
            "terminal": torch.zeros(2),
        },
        "detach_index": torch.zeros(2),
        "target_junction_wrench": torch.zeros(2, 6),
        "ft_wrist": torch.zeros(2, 6),
        "woody_part_force": {},
        **({} if adv is None else {"tcp_advance_m": torch.tensor(adv)}),
    }


def test_d17m_episode_stats_report_the_peak_tcp_advance_into_the_tree():
    # [D17m] the wall is on the VIC target; compliance can still carry the TCP past the grasp plane,
    # so the episode's peak TCP advance along the reset grip axis is reported (pulling back counts 0).
    from apple_pick_gym.rl.skrl_wrapper import _EpisodeStats

    st = _EpisodeStats(2, torch.device("cpu"))
    no = torch.zeros(2, dtype=torch.bool)
    st.update(torch.zeros(2, 1), _advance_info([0.003, -0.02]), no, torch.zeros(2, 13))
    st.update(torch.zeros(2, 1), _advance_info([0.001, -0.03]), no, torch.zeros(2, 13))
    st.update(torch.zeros(2, 1), _advance_info([0.009, 0.05], frozen=(True, True)), no, torch.zeros(2, 13))  # frozen
    s = st.summary()
    assert float(s["Episode / peak TCP advance into tree mm (mean)"]) == pytest.approx(1.5)
    assert float(s["Episode / peak TCP advance into tree mm (max)"]) == pytest.approx(3.0)
    st = _EpisodeStats(2, torch.device("cpu"))
    st.update(torch.zeros(2, 1), _advance_info(None), no, torch.zeros(2, 13))
    assert math.isnan(float(st.summary()["Episode / peak TCP advance into tree mm (mean)"]))
