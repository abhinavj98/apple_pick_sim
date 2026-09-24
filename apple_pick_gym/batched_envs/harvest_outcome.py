"""One harvest step's reward, success streak, safety check, freeze and ``terminated``.

Shared by ``ApplePickVicHarvestEnv.compute_reward`` and the RL surrogate env so both
have the same episode semantics:

- dense reward = weighted progress (the per-step increase in detach-envelope utilization,
  or ``progress_mode="absolute"`` utilization) + pull-out + collateral + slack terms
  (``harvest_reward``);
- success = ``info["detach_index"] >= 1`` for ``success_streak_steps`` consecutive steps;
- safety = the target junction's anchor-frame wrench or the raw wrist wrench over the caps;
- terminal = success bonus or failure penalty (``compute_terminal_reward``);
- the step reward is masked to 0 for envs frozen *before* this step;
- ``terminated`` is the freeze edge -- envs that freeze on this step, once.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from apple_pick_gym.batched_envs.harvest_episode import (
    EpisodeConfig,
    FreezeMask,
    SuccessStreakTracker,
    check_safety_violation,
    compute_terminal_reward,
)
from apple_pick_gym.batched_envs.harvest_reward import (
    HarvestRewardConfig,
    compute_dense_reward_terms,
    weight_dense_reward_terms,
)


@dataclasses.dataclass(frozen=True)
class HarvestStepOutcome:
    reward: torch.Tensor  # (N, 1), freeze-masked
    terminated: torch.Tensor  # (N,) bool, the freeze edge
    progress: torch.Tensor  # (N,) this step's envelope utilization (next step's progress_prev)
    reward_terms: dict[str, Any]
    episode: dict[str, torch.Tensor]


def evaluate_harvest_step(
    obs: dict[str, Any],
    info: dict[str, Any],
    *,
    reward_cfg: HarvestRewardConfig,
    episode_cfg: EpisodeConfig,
    tracker: SuccessStreakTracker,
    freeze_mask: FreezeMask,
    target_junction_name: str,
    progress_prev: torch.Tensor | None = None,
) -> HarvestStepOutcome:
    """Update ``tracker`` / ``freeze_mask`` in place and return this step's outcome.

    ``progress_prev`` (``(N,)``, the previous step's -- or the reset state's -- envelope
    utilization) is required for ``reward_cfg.progress_mode == "delta"``.
    """
    raw_terms = compute_dense_reward_terms(obs, info, target_junction_name=target_junction_name, cfg=reward_cfg)
    weighted_terms = weight_dense_reward_terms(raw_terms, reward_cfg)
    if reward_cfg.progress_mode == "delta":
        if progress_prev is None:
            raise ValueError("progress_mode='delta' needs progress_prev (last step's envelope utilization)")
        weighted_terms["progress"] = reward_cfg.w_progress * (raw_terms["progress"] - progress_prev.to(raw_terms["progress"]))
    elif reward_cfg.progress_mode != "absolute":
        raise ValueError(f"unknown progress_mode {reward_cfg.progress_mode!r}")
    dense = sum(weighted_terms.values()).unsqueeze(-1)

    success_this_step = info["detach_index"] >= 1.0
    success_achieved = tracker.update(success_this_step, episode_cfg)

    # The target junction's wrench is uncapped (privileged debug gather), so this is a real
    # check; ft_wrist is hard-capped by the stem-harvest transfer at the same 40 N / 10 N*m
    # default, so that half is a safety net kept in case the caps diverge.
    safety_junction = check_safety_violation(info["target_junction_wrench"], episode_cfg)
    safety_wrist = check_safety_violation(info["ft_wrist"], episode_cfg)
    safety_violation = safety_junction | safety_wrist

    terminal = compute_terminal_reward(success_achieved, safety_violation, reward_cfg)
    reward = freeze_mask.apply_to_reward(dense + terminal)
    # Once, on the freeze edge: frozen envs keep stepping (whole-batch reset only) and would
    # otherwise re-report termination, making recurrent PPO zero their hidden state and cut
    # GAE on every frozen step.
    terminated = freeze_mask.update(success_achieved | safety_violation)

    n = reward.shape[0]
    return HarvestStepOutcome(
        reward=reward,
        terminated=terminated,
        progress=raw_terms["progress"].detach().clone(),
        reward_terms={
            "raw": raw_terms,
            "weighted": weighted_terms,
            "dense": dense.squeeze(-1),
            "terminal": terminal.squeeze(-1),
            "total": reward.reshape(n),
        },
        episode={
            "success_this_step": success_this_step,
            "success_achieved": success_achieved,
            "success_streak": tracker.streak.clone(),
            "safety_junction": safety_junction,
            "safety_wrist": safety_wrist,
            "frozen": freeze_mask.done_mask.clone(),
            "terminated_edge": terminated.clone(),
            "detach_index": info["detach_index"],
        },
    )
