"""Success-streak tracking, frozen-env action/reward masking, and safety-cap
termination for fixed-length, batch-synchronized harvest episodes.

``BatchedHeterogeneousCoupledSim`` supports whole-batch reset only (no
per-env ``reset_idx``). Rather than fight this, all envs truncate together at
``max_episode_steps``; an env that satisfies the success condition
**freezes** (its action is held at the last commanded value and its reward
is masked to zero) instead of ending its own episode early. This keeps LSTM
hidden-state resets batch-uniform -- every env's episode boundary lands on
the same step, so the recurrent actor/critic never need to reset hidden
state for a subset of the batch mid-rollout. The freeze mask is entirely
gym-layer state (not sim state), so swapping in a real per-env
``reset_idx`` later is a contained, single-seam change.
"""

from __future__ import annotations

import dataclasses

import torch

from apple_pick_gym.batched_envs.harvest_reward import HarvestRewardConfig


@dataclasses.dataclass(frozen=True)
class EpisodeConfig:
    """Success-streak length and safety caps for episode-level logic."""

    success_streak_steps: int = 10
    safety_force_cap_n: float = 40.0
    safety_torque_cap_nm: float = 10.0


class SuccessStreakTracker:
    """Per-env consecutive-success counter; any non-success step resets it to 0."""

    def __init__(self, num_envs: int, device: str | torch.device) -> None:
        self.device = torch.device(device)
        self.streak = torch.zeros(int(num_envs), dtype=torch.int64, device=self.device)

    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        if env_mask is None:
            self.streak.zero_()
        else:
            self.streak[env_mask] = 0

    def update(self, success_this_step: torch.Tensor, cfg: EpisodeConfig) -> torch.Tensor:
        """``success_this_step``: (N,) bool. Returns (N,) bool: streak reached the target length."""
        self.streak = torch.where(
            success_this_step, self.streak + 1, torch.zeros_like(self.streak)
        )
        return self.streak >= int(cfg.success_streak_steps)


def check_safety_violation(wrench: torch.Tensor, cfg: EpisodeConfig) -> torch.Tensor:
    """``wrench``: (N,6) ``[Fx,Fy,Fz,Tx,Ty,Tz]``. Returns (N,) bool: force or torque cap exceeded."""
    force_norm = torch.linalg.norm(wrench[:, :3], dim=-1)
    torque_norm = torch.linalg.norm(wrench[:, 3:6], dim=-1)
    return (force_norm > float(cfg.safety_force_cap_n)) | (
        torque_norm > float(cfg.safety_torque_cap_nm)
    )


class FreezeMask:
    """Per-env sticky 'already finished' mask: action held, reward masked once set."""

    def __init__(self, num_envs: int, device: str | torch.device) -> None:
        self.device = torch.device(device)
        self.done_mask = torch.zeros(int(num_envs), dtype=torch.bool, device=self.device)

    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        if env_mask is None:
            self.done_mask.zero_()
        else:
            self.done_mask[env_mask] = False

    def update(self, newly_terminated: torch.Tensor) -> None:
        """Sticky OR: an env, once frozen, stays frozen until ``reset()``."""
        self.done_mask = self.done_mask | newly_terminated

    def apply_to_action(self, action: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
        mask = self.done_mask.reshape(-1, *([1] * (action.dim() - 1)))
        return torch.where(mask, last_action, action)

    def apply_to_delta_action(
        self, action: torch.Tensor, last_action: torch.Tensor, *, delta_dims: int
    ) -> torch.Tensor:
        """Like :meth:`apply_to_action`, but frozen rows get a zero pose delta.

        The leading ``delta_dims`` entries of a delta-pose action are integrated
        into the target every step, so replaying them would keep a frozen env's
        target moving; frozen envs hold their target and keep their last gains.
        """
        held = last_action.clone()
        held[:, :delta_dims] = 0.0
        return self.apply_to_action(action, held)

    def apply_to_reward(self, reward: torch.Tensor) -> torch.Tensor:
        mask = self.done_mask.reshape(-1, *([1] * (reward.dim() - 1)))
        return torch.where(mask, torch.zeros_like(reward), reward)


def compute_terminal_reward(
    success_achieved: torch.Tensor,
    safety_violation: torch.Tensor,
    cfg: HarvestRewardConfig,
) -> torch.Tensor:
    """One-time terminal reward, shape (N,1).

    A safety violation is a failure regardless of a simultaneous success
    streak (no bonus, even if both trigger on the same step) -- termination
    on a cap violation is defined as a failure.
    """
    bonus = torch.where(
        success_achieved & ~safety_violation,
        torch.full_like(success_achieved, cfg.success_bonus, dtype=torch.float32),
        torch.zeros(success_achieved.shape, dtype=torch.float32, device=success_achieved.device),
    )
    penalty = torch.where(
        safety_violation,
        torch.full_like(safety_violation, cfg.failure_penalty, dtype=torch.float32),
        torch.zeros(safety_violation.shape, dtype=torch.float32, device=safety_violation.device),
    )
    return (bonus + penalty).unsqueeze(-1)
