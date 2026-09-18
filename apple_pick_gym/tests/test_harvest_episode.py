"""Success-streak tracking, frozen-env action/reward masking, and safety-cap
termination for fixed-length, batch-synchronized harvest episodes.

All envs truncate together at max_episode_steps (LSTM hidden-state resets
stay batch-uniform); an env that satisfies the success condition freezes
(action held, reward masked) rather than ending its own episode early.
"""

from __future__ import annotations

import torch

from apple_pick_gym.batched_envs.harvest_episode import (
    EpisodeConfig,
    FreezeMask,
    SuccessStreakTracker,
    check_safety_violation,
    compute_terminal_reward,
)


def test_success_streak_requires_consecutive_steps():
    cfg = EpisodeConfig(success_streak_steps=3)
    tracker = SuccessStreakTracker(num_envs=1, device="cpu")
    # success, success, FAIL (gap), success, success, success -> only the last 3 count
    sequence = [True, True, False, True, True, True]
    achieved = []
    for s in sequence:
        achieved.append(tracker.update(torch.tensor([s]), cfg).item())
    assert achieved == [False, False, False, False, False, True]


def test_success_streak_gap_resets_counter_not_just_pauses():
    cfg = EpisodeConfig(success_streak_steps=2)
    tracker = SuccessStreakTracker(num_envs=1, device="cpu")
    for s in [True, False, True]:
        result = tracker.update(torch.tensor([s]), cfg)
    # After True,False,True the streak is 1 (reset by the gap), not 2.
    assert result.item() is False


def test_success_streak_per_env_independent():
    cfg = EpisodeConfig(success_streak_steps=2)
    tracker = SuccessStreakTracker(num_envs=2, device="cpu")
    r1 = tracker.update(torch.tensor([True, False]), cfg)
    r2 = tracker.update(torch.tensor([True, False]), cfg)
    assert r2.tolist() == [True, False]


def test_freeze_mask_holds_last_action_for_frozen_envs():
    mask = FreezeMask(num_envs=2, device="cpu")
    mask.update(torch.tensor([True, False]))
    new_action = torch.tensor([[9.0, 9.0], [9.0, 9.0]])
    last_action = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    out = mask.apply_to_action(new_action, last_action)
    torch.testing.assert_close(out, torch.tensor([[1.0, 1.0], [9.0, 9.0]]))


def test_freeze_mask_masks_reward_for_frozen_envs_only():
    mask = FreezeMask(num_envs=2, device="cpu")
    mask.update(torch.tensor([True, False]))
    reward = torch.tensor([[5.0], [5.0]])
    out = mask.apply_to_reward(reward)
    torch.testing.assert_close(out, torch.tensor([[0.0], [5.0]]))


def test_freeze_mask_is_sticky_across_updates():
    mask = FreezeMask(num_envs=2, device="cpu")
    mask.update(torch.tensor([True, False]))
    mask.update(torch.tensor([False, False]))  # no new terminations
    reward = torch.tensor([[5.0], [5.0]])
    out = mask.apply_to_reward(reward)
    torch.testing.assert_close(out, torch.tensor([[0.0], [5.0]]))


def test_freeze_mask_reset_clears_all_envs():
    mask = FreezeMask(num_envs=2, device="cpu")
    mask.update(torch.tensor([True, True]))
    mask.reset()
    reward = torch.tensor([[5.0], [5.0]])
    out = mask.apply_to_reward(reward)
    torch.testing.assert_close(out, reward)


def test_check_safety_violation_force_and_torque():
    cfg = EpisodeConfig(safety_force_cap_n=40.0, safety_torque_cap_nm=10.0)
    wrench = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ok
            [50.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # force violation
            [0.0, 0.0, 0.0, 15.0, 0.0, 0.0],  # torque violation
        ]
    )
    violated = check_safety_violation(wrench, cfg)
    assert violated.tolist() == [False, True, True]


def test_terminal_reward_success_bonus_no_penalty():
    cfg = EpisodeConfig(success_streak_steps=1)
    from apple_pick_gym.batched_envs.harvest_reward import HarvestRewardConfig

    reward_cfg = HarvestRewardConfig(success_bonus=10.0, failure_penalty=-20.0)
    success = torch.tensor([True, False])
    violation = torch.tensor([False, False])
    terminal = compute_terminal_reward(success, violation, reward_cfg)
    torch.testing.assert_close(terminal, torch.tensor([[10.0], [0.0]]))


def test_terminal_reward_failure_penalty_no_bonus_even_if_also_successful():
    """Safety violation is a failure regardless of a simultaneous success streak
    (no bonus + violation in the same step -- termination is a failure)."""
    from apple_pick_gym.batched_envs.harvest_reward import HarvestRewardConfig

    reward_cfg = HarvestRewardConfig(success_bonus=10.0, failure_penalty=-20.0)
    success = torch.tensor([True])
    violation = torch.tensor([True])
    terminal = compute_terminal_reward(success, violation, reward_cfg)
    torch.testing.assert_close(terminal, torch.tensor([[-20.0]]))


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
