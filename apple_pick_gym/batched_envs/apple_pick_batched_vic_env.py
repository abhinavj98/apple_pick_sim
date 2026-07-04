"""Batched VIC gym env (GPU heterogeneous coupled sim)."""

from __future__ import annotations

from typing import Any

import torch

from apple_pick_gym.batched_envs.apple_pick_batched_base_env import ApplePickBatchedBaseEnv


class ApplePickBatchedVicEnv(ApplePickBatchedBaseEnv):
    """VIC + fix-to-apple batched env; stub reward/termination for V.3.3 foundation."""

    def compute_reward(
        self, obs: dict[str, Any], info: dict[str, Any]
    ) -> torch.Tensor:
        del obs, info
        return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)

    def compute_terminated(
        self, obs: dict[str, Any], info: dict[str, Any]
    ) -> torch.Tensor:
        del obs, info
        return torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
