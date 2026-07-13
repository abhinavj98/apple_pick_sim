"""Sticky soft-disable for blown-up envs in batched collect/replay loops."""

from __future__ import annotations

import torch


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
        out[self.disabled] = 0
        return out

    def should_record_mask(self) -> torch.Tensor:
        return ~self.disabled
