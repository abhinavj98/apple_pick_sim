"""Sensor-realistic ft_wrist: causal EMA + per-env bias + per-step noise + slow drift.

Models what a real-time controller would see on the real rig's F/T channel,
for the RL observation path only. Two invariants this must not confuse:

- The real dataset's ``ft_wrist_lpf`` is a **zero-phase** filter (offline
  ``filtfilt``, used for CMA scoring). No online policy can reproduce that;
  this module's EMA is deliberately causal and not trying to match it.
- The "No sim EMA/LPF" rule (H3, ``docs/handbook-sysid-scoring.md``) still
  governs the ``batched_sysid_v1`` feature bags. This EMA lives strictly on
  the gym observation path and must never reach sys-ID scoring.

An EMA is already a first-order causal low-pass with corner frequency
``fc``: ``a = 1 - exp(-2*pi*fc/f_control)``. At ``f_control=60`` Hz and
``fc=10`` Hz, ``a ~= 0.65``.
"""

from __future__ import annotations

import dataclasses
import math

import torch


@dataclasses.dataclass(frozen=True)
class FtSensorConfig:
    """EMA corner frequency plus bias/noise/drift magnitudes for the (N,6) ft_wrist channel."""

    control_hz: float = 60.0
    cutoff_hz: float = 10.0
    bias_std: float = 0.0
    noise_std: float = 0.0
    drift_std: float = 0.0
    drift_clip: float | None = None

    @property
    def alpha(self) -> float:
        return 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz / self.control_hz)


class FtSensorModel:
    """Batched ``(N, 6)`` sensor-realistic F/T model: EMA + per-env bias + noise + drift.

    All state (EMA, bias, drift) is episode-scoped: :meth:`reset` reseeds
    every one of them, matching the design decision that bias is "resampled
    per env per episode" and drift is "a slow per-episode walk" (rather than
    persisting across resets).
    """

    def __init__(
        self,
        *,
        num_envs: int,
        device: str | torch.device,
        config: FtSensorConfig,
        generator: torch.Generator | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.config = config
        self.alpha = float(config.alpha)
        self._generator = generator

        self._ema = torch.zeros((self.num_envs, 6), device=self.device)
        self._bias = torch.zeros((self.num_envs, 6), device=self.device)
        self._drift = torch.zeros((self.num_envs, 6), device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _randn(self, shape) -> torch.Tensor:
        if self._generator is not None:
            return torch.randn(shape, device=self.device, generator=self._generator)
        return torch.randn(shape, device=self.device)

    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        """Reseed EMA/bias/drift for the given envs (default: all envs)."""
        if env_mask is None:
            env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        n = int(env_mask.sum().item())
        if n == 0:
            return
        self._ema[env_mask] = 0.0
        self._initialized[env_mask] = False
        self._drift[env_mask] = 0.0
        self._bias[env_mask] = self._randn((n, 6)) * self.config.bias_std

    def step(self, raw: torch.Tensor) -> torch.Tensor:
        """``raw`` is ``(N, 6)``; returns the sensor-realistic ``(N, 6)`` observation."""
        init = self._initialized.unsqueeze(-1)
        ema_new = torch.where(init, self.alpha * raw + (1.0 - self.alpha) * self._ema, raw)
        self._ema = ema_new
        self._initialized = torch.ones_like(self._initialized)

        if self.config.drift_std > 0.0:
            self._drift = self._drift + self._randn(raw.shape) * self.config.drift_std
            if self.config.drift_clip is not None:
                self._drift = torch.clamp(
                    self._drift, -self.config.drift_clip, self.config.drift_clip
                )

        noise = (
            self._randn(raw.shape) * self.config.noise_std
            if self.config.noise_std > 0.0
            else torch.zeros_like(raw)
        )
        return self._ema + self._bias + noise + self._drift
