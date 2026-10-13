"""Per-step reward / success / freeze / terminated bookkeeping shared by the real harvest
env and the RL surrogate env (pure torch)."""

from __future__ import annotations

import pytest
import torch

from apple_pick_gym.batched_envs.harvest_episode import EpisodeConfig, FreezeMask, SuccessStreakTracker
from apple_pick_gym.batched_envs.harvest_outcome import evaluate_harvest_step
from apple_pick_gym.batched_envs.harvest_reward import HarvestRewardConfig

N = 3
TJ = "spur_stem"


def _info(target_f, wrist_f=0.0, other=(0.0, 0.0, 0.0)):
    w = torch.zeros(N, 6)
    w[:, 2] = torch.as_tensor(target_f, dtype=torch.float32)
    wrist = torch.zeros(N, 6)
    wrist[:, 0] = wrist_f
    o = torch.zeros(N, 6)
    o[:, 2] = torch.as_tensor(other, dtype=torch.float32)
    return {
        "target_junction_wrench": w,
        "target_junction_force": w,
        "detach_index": (w[:, 2] / 20.0) ** 2,
        "detach_utilization": w[:, 2] / 20.0,
        "ft_wrist": wrist,
        "woody_part_force": {TJ: w, "stem_apple": o},
        "collateral_baseline_norm": {"stem_apple": torch.zeros(N)},
    }


def _run(seq, **kw):
    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(N, 1)}
    tracker, mask = SuccessStreakTracker(N, "cpu"), FreezeMask(N, "cpu")
    out = []
    for info in seq:
        out.append(
            evaluate_harvest_step(
                obs,
                info,
                reward_cfg=HarvestRewardConfig(progress_mode="absolute", w_slack=0.0),
                episode_cfg=EpisodeConfig(success_streak_steps=2, **kw),
                tracker=tracker,
                freeze_mask=mask,
                target_junction_name=TJ,
            )
        )
    return out


def test_success_needs_a_streak_then_terminates_once_and_freezes():
    seq = [_info([25.0, 5.0, 25.0])] * 2 + [_info([25.0, 5.0, 0.0])] * 2
    out = _run(seq)
    edges = torch.stack([o.terminated for o in out])
    assert edges.tolist() == [[False, False, False], [True, False, True], [False, False, False], [False, False, False]]
    r = torch.stack([o.reward.flatten() for o in out])
    # success bonus lands on the edge step, then the frozen env's reward is exactly 0
    assert float(r[1, 0]) > 10.0 - 1e-3 and float(r[2, 0]) == 0.0 and float(r[3, 2]) == 0.0
    assert out[-1].episode["frozen"].tolist() == [True, False, True]
    assert out[1].episode["success_achieved"].tolist() == [True, False, True]


def test_safety_violation_is_a_penalized_terminal_failure():
    out = _run([_info([5.0, 5.0, 5.0], wrist_f=50.0)], safety_force_cap_n=40.0)
    assert out[0].terminated.tolist() == [True, True, True]
    assert torch.all(out[0].episode["safety_wrist"])
    assert torch.all(out[0].reward.flatten() < -19.0)


def test_reward_terms_and_total_are_reported():
    out = _run([_info([10.0, 10.0, 10.0], other=(3.0, 3.0, 3.0))])[0]
    assert set(out.reward_terms["raw"]) == {"progress", "pullout", "wrist", "collateral", "slack"}
    torch.testing.assert_close(out.reward_terms["raw"]["progress"], torch.full((N,), 0.5))
    torch.testing.assert_close(out.reward_terms["raw"]["collateral"], torch.full((N,), 3.0))
    torch.testing.assert_close(out.reward_terms["total"], out.reward.flatten())


def test_envs_frozen_before_the_step_get_zero_reward_and_no_edge():
    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(N, 1)}
    tracker, mask = SuccessStreakTracker(N, "cpu"), FreezeMask(N, "cpu")
    mask.update(torch.tensor([True, False, False]))  # invalid env frozen at reset
    o = evaluate_harvest_step(
        obs,
        _info([30.0, 30.0, 30.0]),
        reward_cfg=HarvestRewardConfig(progress_mode="absolute"),
        episode_cfg=EpisodeConfig(success_streak_steps=1),
        tracker=tracker,
        freeze_mask=mask,
        target_junction_name=TJ,
    )
    assert o.terminated.tolist() == [False, True, True]
    assert float(o.reward[0]) == 0.0


def test_delta_progress_rewards_the_increase_in_utilization_not_hovering():
    """progress_mode='delta': reward w*(u_t - u_{t-1}), so hovering below the envelope earns ~0
    and detaching (the terminal bonus) is what pays -- with 'absolute', hovering at u~0.9
    for the rest of the episode out-earns the success bonus because success freezes reward."""
    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(N, 1)}
    tracker, mask = SuccessStreakTracker(N, "cpu"), FreezeMask(N, "cpu")
    cfg = HarvestRewardConfig(w_pullout=0.0, w_collateral=0.0, w_slack=0.0, progress_mode="delta")
    prev = torch.full((N,), 0.25)
    rewards = []
    for f in ([10.0] * N, [18.0] * N, [18.0] * N):
        o = evaluate_harvest_step(
            obs, _info(f), reward_cfg=cfg, episode_cfg=EpisodeConfig(), tracker=tracker, freeze_mask=mask,
            target_junction_name=TJ, progress_prev=prev,
        )
        prev = o.progress
        rewards.append(o.reward.flatten())
    torch.testing.assert_close(rewards[0], torch.full((N,), 0.25))  # 0.25 -> 0.5
    torch.testing.assert_close(rewards[1], torch.full((N,), 0.4))  # 0.5 -> 0.9
    torch.testing.assert_close(rewards[2], torch.zeros(N))  # hovering pays nothing


def test_delta_progress_needs_the_previous_utilization():
    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(N, 1)}
    with pytest.raises(ValueError, match="progress_prev"):
        evaluate_harvest_step(
            obs, _info([1.0] * N), reward_cfg=HarvestRewardConfig(progress_mode="delta"), episode_cfg=EpisodeConfig(),
            tracker=SuccessStreakTracker(N, "cpu"), freeze_mask=FreezeMask(N, "cpu"), target_junction_name=TJ,
        )


def test_slack_costs_every_live_step_and_nothing_once_frozen():
    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(N, 1)}
    tracker, mask = SuccessStreakTracker(N, "cpu"), FreezeMask(N, "cpu")
    mask.update(torch.tensor([True, False, False]))
    cfg = HarvestRewardConfig(w_progress=0.0, w_pullout=0.0, w_collateral=0.0, w_slack=0.05)
    o = evaluate_harvest_step(
        obs, _info([1.0] * N), reward_cfg=cfg, episode_cfg=EpisodeConfig(), tracker=tracker, freeze_mask=mask,
        target_junction_name=TJ, progress_prev=torch.zeros(N),
    )
    torch.testing.assert_close(o.reward.flatten(), torch.tensor([0.0, -0.05, -0.05]))
    torch.testing.assert_close(o.reward_terms["weighted"]["slack"], torch.full((N,), -0.05))


def test_d12_physically_impossible_force_is_a_blowup_frozen_without_penalty_or_success():
    # [D12] solver blow-ups (single worlds at ~1-2 kN on GPU) are numerics, not policy: freeze the
    # world with zero reward, no failure penalty, no success; a 41 N overshoot stays a safety failure
    out = _run([_info([5.0, 1500.0, 41.0])], safety_force_cap_n=40.0, blowup_force_n=200.0)
    ep = out[0].episode
    assert ep["blowup"].tolist() == [False, True, False]
    assert out[0].terminated.tolist() == [False, True, True]
    r = out[0].reward.flatten()
    assert float(r[1]) == 0.0  # no penalty, no bonus, no dense term from the blown-up readout
    assert float(r[2]) < -18.0  # 41 N at the junction: a real safety failure (-20 + progress 1)
    assert ep["success_achieved"].tolist() == [False, False, False]
    assert ep["safety_junction"].tolist() == [False, False, True]
    wrist = _run([_info([5.0, 5.0, 5.0], wrist_f=float("nan"))], blowup_force_n=200.0)[0]
    assert wrist.episode["blowup"].all() and float(wrist.reward.abs().max()) == 0.0


def test_d12_blowup_guard_off_keeps_old_behaviour():
    out = _run([_info([5.0, 1500.0, 5.0])], safety_force_cap_n=40.0, blowup_force_n=None)
    assert out[0].episode["safety_junction"].tolist() == [False, True, False]
    assert not out[0].episode["blowup"].any()


def test_d13_success_pays_minus_w_times_the_episode_peak_collateral():
    # [D13] collateral per successful pick is the objective: charge the episode's peak collateral once,
    # at the success edge (the per-step collateral term mostly measured time, not the pick's load)
    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(N, 1)}
    tracker, mask = SuccessStreakTracker(N, "cpu"), FreezeMask(N, "cpu")
    cfg = HarvestRewardConfig(progress_mode="absolute", w_slack=0.0, w_progress=0.0, w_collateral=0.0, w_peak_collateral=0.5)
    ep = EpisodeConfig(success_streak_steps=1)
    peak = None
    seq = [_info([5.0, 5.0, 5.0], other=(30.0, 10.0, 0.0)), _info([25.0, 25.0, 5.0], other=(4.0, 4.0, 0.0))]
    outs = []
    for info in seq:
        o = evaluate_harvest_step(
            obs, info, reward_cfg=cfg, episode_cfg=ep, tracker=tracker, freeze_mask=mask,
            target_junction_name=TJ, peak_collateral_prev=peak,
        )
        peak = o.peak_collateral
        outs.append(o)
    torch.testing.assert_close(outs[0].peak_collateral, torch.tensor([30.0, 10.0, 0.0]))
    torch.testing.assert_close(outs[1].peak_collateral, torch.tensor([30.0, 10.0, 0.0]))  # max, not last
    r = outs[1].reward.flatten()
    # envs 0/1 succeed: bonus 10 - 0.5 * peak ; env 2 does not succeed: no charge
    torch.testing.assert_close(r[:2], torch.tensor([10.0 - 15.0, 10.0 - 5.0]))
    assert float(r[2]) == 0.0


def _v2_episode_returns(u_traj, coll_traj, wrist_f=0.0):
    """[D17] Return per env of a 250-step episode under the v2 reward (4 terms, no safety trip)."""
    reward_cfg = HarvestRewardConfig(
        w_progress=10.0, progress_mode="delta", w_pullout=0.0, w_wrist=0.0, w_peak_collateral=0.0,
        w_collateral=0.02, collateral_clip_n=50.0, w_slack=0.01, success_bonus=60.0, failure_penalty=-40.0,
    )
    episode_cfg = EpisodeConfig(success_streak_steps=3, safety_force_cap_n=1e12, safety_torque_cap_nm=1e12)
    n = u_traj.shape[1]
    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(n, 1)}
    tracker, mask = SuccessStreakTracker(n, "cpu"), FreezeMask(n, "cpu")
    prev, total, worst = u_traj[0].clone(), torch.zeros(n), torch.zeros(n)
    for t in range(1, u_traj.shape[0]):
        other = torch.zeros(n, 6)
        other[:, 2] = coll_traj[t]
        wrist = torch.zeros(n, 6)
        wrist[:, 0] = wrist_f
        info = {
            "target_junction_wrench": torch.zeros(n, 6),
            "detach_utilization": u_traj[t],
            "detach_index": u_traj[t],
            "ft_wrist": wrist,
            "woody_part_force": {TJ: torch.zeros(n, 6), "stem_apple": other},
            "collateral_baseline_norm": {"stem_apple": torch.zeros(n)},
        }
        out = evaluate_harvest_step(
            obs, info, reward_cfg=reward_cfg, episode_cfg=episode_cfg, tracker=tracker, freeze_mask=mask,
            target_junction_name=TJ, progress_prev=prev,
        )
        prev = out.progress
        r = out.reward.flatten()
        total += r
        worst = torch.minimum(worst, out.reward_terms["dense"])
        assert not bool(out.episode["safety_wrist"].any())
    return total, worst, mask.done_mask


def test_d17_v2_reward_ranks_bend_pick_over_pull_pick_over_idle_and_is_bounded():
    t = torch.arange(251, dtype=torch.float32)
    u0 = 0.14
    bend = torch.clamp(u0 + (1.0 - u0) * t / 100.0, max=1.2)  # slow, gentle: 10 N collateral
    pull = torch.clamp(u0 + (1.0 - u0) * t / 40.0, max=1.2)  # fast, hard: 60 N (clipped to 50)
    idle = torch.full_like(t, u0)
    u = torch.stack([bend, pull, idle], dim=1)
    coll = torch.stack([torch.full_like(t, 10.0), torch.full_like(t, 60.0), torch.zeros_like(t)], dim=1)
    ret, worst, done = _v2_episode_returns(u, coll, wrist_f=100.0)  # 100 N wrist: no safety trip any more
    assert done.tolist() == [True, True, False]
    assert float(ret[0]) > float(ret[1]) > float(ret[2])
    assert float(ret[2]) == pytest.approx(-2.5, abs=1e-4)  # idle: slack only
    # dense per-step bound: progress >= -10, collateral >= -1 (clip), slack -0.01
    assert float(worst.min()) >= -(10.0 + 1.0 + 0.01) - 1e-5


def test_d17m_peak_collateral_charge_reads_the_deviation_term():
    # [D17m] the success-edge charge reads the same collateral term as the dense one, so with
    # "deviation" an episode that only *unloaded* the tree (lifted the apple) is charged too.
    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(N, 1)}
    cfg = HarvestRewardConfig(
        progress_mode="absolute", w_slack=0.0, w_progress=0.0, w_collateral=0.0, w_peak_collateral=1.0,
        collateral_mode="deviation",
    )
    ep = EpisodeConfig(success_streak_steps=1)
    rest = {"stem_apple": torch.tensor([[0.0, 0.0, 6.0]] * N)}
    peaks = {}
    for mode_cfg in (cfg, HarvestRewardConfig(**{**cfg.__dict__, "collateral_mode": "relu_norm"})):
        tracker, mask = SuccessStreakTracker(N, "cpu"), FreezeMask(N, "cpu")
        info = _info([25.0, 25.0, 25.0], other=(6.0, 2.0, 9.0))  # at rest / unloaded 4 N / loaded 3 N
        info["collateral_baseline_norm"] = {"stem_apple": torch.full((N,), 6.0)}
        info["collateral_baseline_force"] = rest
        o = evaluate_harvest_step(
            obs, info, reward_cfg=mode_cfg, episode_cfg=ep, tracker=tracker, freeze_mask=mask, target_junction_name=TJ
        )
        peaks[mode_cfg.collateral_mode] = o.peak_collateral
        torch.testing.assert_close(o.reward.flatten(), mode_cfg.success_bonus - o.peak_collateral)
    torch.testing.assert_close(peaks["deviation"], torch.tensor([0.0, 4.0, 3.0]))
    torch.testing.assert_close(peaks["relu_norm"], torch.tensor([0.0, 0.0, 3.0]))
