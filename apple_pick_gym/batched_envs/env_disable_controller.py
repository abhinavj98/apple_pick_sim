"""Sticky soft-disable for blown-up envs in batched collect/replay loops."""

from __future__ import annotations

import torch

_POSE_ACTION_DIM = 19


class EnvDisableController:
    """Zero actions and gate recording for envs that have gone unstable.

    Hot-path methods (``update``, ``apply_actions``, ``should_record_mask``) stay on
    device tensors and must not call ``.item()`` / ``.cpu()`` / ``.numpy()``.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        device: torch.device | str,
        initial_disabled: torch.Tensor | None = None,
    ) -> None:
        self._num_envs = int(num_envs)
        if self._num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {self._num_envs}")
        self._device = torch.device(device)
        self.disabled = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._held_pose_actions: torch.Tensor | None = None
        if initial_disabled is not None:
            self.update(initial_disabled)

    def update(self, unstable: torch.Tensor) -> None:
        mask = unstable.to(device=self._device, dtype=torch.bool).reshape(-1)
        if int(mask.numel()) != self._num_envs:
            raise ValueError(
                f"unstable length {int(mask.numel())} != num_envs {self._num_envs}"
            )
        self.disabled |= mask

    def apply_actions(self, actions: torch.Tensor) -> torch.Tensor:
        out = actions.clone()
        if actions.ndim == 2 and int(actions.shape[-1]) == _POSE_ACTION_DIM:
            # ``vic_pose`` rows are absolute pose + gains, so a zero row commands the
            # world origin with Kp=Kd=0 (limp arm). Freeze the last commanded row of a
            # disabled env instead; it keeps holding that pose with its own gains.
            if self._held_pose_actions is not None:
                out[self.disabled] = self._held_pose_actions[self.disabled]
            self._held_pose_actions = out.detach().clone()
        else:
            out[self.disabled] = 0
        return out

    def should_record_mask(self) -> torch.Tensor:
        return ~self.disabled
