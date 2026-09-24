"""Per-step reward / success / freeze / terminated bookkeeping shared by the real harvest
env and the RL surrogate env (pure torch)."""

from __future__ import annotations

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
                reward_cfg=HarvestRewardConfig(),
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
    assert set(out.reward_terms["raw"]) == {"progress", "pullout", "collateral"}
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
        reward_cfg=HarvestRewardConfig(),
        episode_cfg=EpisodeConfig(success_streak_steps=1),
        tracker=tracker,
        freeze_mask=mask,
        target_junction_name=TJ,
    )
    assert o.terminated.tolist() == [False, True, True]
    assert float(o.reward[0]) == 0.0
