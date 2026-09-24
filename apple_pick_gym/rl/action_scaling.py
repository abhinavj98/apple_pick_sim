"""Map the Gaussian policy's ``[-1, 1]^13`` action box to harvest env units and back.

The env's 13-D action mixes centimetre pose deltas with stiffnesses in the tens
to hundreds (``HarvestActionBounds``). A Gaussian policy with one shared
initial std cannot explore that box well, so the policy acts in ``[-1, 1]``
per dim and this scaler maps:

- pose deltas ``[0:6]``: affine, ``u * bound`` (0 = hold the target);
- stiffness ``[6:12]``: **log**-affine between ``k_min`` and ``k_max`` (0 = the
  geometric mean) -- stiffness spans a decade and matters multiplicatively;
- damping ratio ``[12]``: affine between ``zeta_min`` and ``zeta_max``.
"""

from __future__ import annotations

import math

import torch

from apple_pick_gym.batched_envs.harvest_action import HarvestActionBounds

_ACTION_DIM = 13


class HarvestActionScaler:
    """``to_env``: policy ``(N, 13)`` in ``[-1, 1]`` -> env units; ``to_policy`` inverts it."""

    def __init__(self, bounds: HarvestActionBounds) -> None:
        self.bounds = bounds
        b = bounds
        self._delta_scale = torch.tensor([b.linear_delta_m] * 3 + [b.angular_delta_rad] * 3)
        self._log_k_lo = torch.tensor([math.log(b.k_lin_min)] * 3 + [math.log(b.k_ang_min)] * 3)
        self._log_k_hi = torch.tensor([math.log(b.k_lin_max)] * 3 + [math.log(b.k_ang_max)] * 3)
        self._zeta_lo = float(b.zeta_min)
        self._zeta_hi = float(b.zeta_max)

    def _check(self, a: torch.Tensor) -> None:
        if a.shape[-1] != _ACTION_DIM:
            raise ValueError(f"expected action last dim {_ACTION_DIM}, got {tuple(a.shape)}")

    def to_env(self, u: torch.Tensor) -> torch.Tensor:
        self._check(u)
        u = torch.clamp(u, -1.0, 1.0)
        dev, dt = u.device, u.dtype
        delta = u[:, :6] * self._delta_scale.to(dev, dt)
        lo, hi = self._log_k_lo.to(dev, dt), self._log_k_hi.to(dev, dt)
        k = torch.exp(lo + 0.5 * (u[:, 6:12] + 1.0) * (hi - lo))
        zeta = self._zeta_lo + 0.5 * (u[:, 12:13] + 1.0) * (self._zeta_hi - self._zeta_lo)
        return torch.cat([delta, k, zeta], dim=-1)

    def to_policy(self, a: torch.Tensor) -> torch.Tensor:
        self._check(a)
        dev, dt = a.device, a.dtype
        delta = a[:, :6] / self._delta_scale.to(dev, dt)
        lo, hi = self._log_k_lo.to(dev, dt), self._log_k_hi.to(dev, dt)
        k = 2.0 * (torch.log(a[:, 6:12]) - lo) / (hi - lo) - 1.0
        zeta = 2.0 * (a[:, 12:13] - self._zeta_lo) / (self._zeta_hi - self._zeta_lo) - 1.0
        return torch.cat([delta, k, zeta], dim=-1)
