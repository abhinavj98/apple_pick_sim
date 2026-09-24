"""Reference policies for the harvest task, acting in the wrapper's ``[-1, 1]^13`` box.

- :class:`ZeroPolicy` -- hold the target at mid stiffness / mid damping (the "do nothing"
  floor: rest loads only).
- :class:`RandomPolicy` -- uniform in the action box (exploration floor, safety check).
- :class:`ScriptedPullPolicy` -- the screening pull: retreat along each env's grasp
  axis (``+weld_direction``, away from the plant -- grasps approach from below, so this is
  the direction that loads the stem), optionally twisting about it, at fixed stiffness
  and damping. The policy the learned one has to beat (plan Task 11 gate).

Each exposes ``reset(wrapper)`` (called after every env reset) and
``act(wrapper, obs, state) -> (N, 13)``.
"""

from __future__ import annotations

import torch


class ZeroPolicy:
    name = "zero"

    def reset(self, wrapper) -> None:
        del wrapper

    def act(self, wrapper, obs, state) -> torch.Tensor:
        del obs, state
        return torch.zeros(wrapper.num_envs, 13, device=wrapper.device)


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._gen = torch.Generator().manual_seed(int(seed))

    def reset(self, wrapper) -> None:
        del wrapper

    def act(self, wrapper, obs, state) -> torch.Tensor:
        del obs, state
        u = torch.rand(wrapper.num_envs, 13, generator=self._gen) * 2.0 - 1.0
        return u.to(wrapper.device)


class ScriptedPullPolicy:
    """Constant-rate retreat (and optional twist) along each env's weld axis."""

    def __init__(
        self,
        *,
        rate_m_per_step: float = 0.002,
        twist_rad_per_step: float = 0.0,
        k_lin: float = 150.0,
        k_ang: float = 20.0,
        zeta: float = 0.9,
    ) -> None:
        self.rate = float(rate_m_per_step)
        self.twist = float(twist_rad_per_step)
        self.k_lin, self.k_ang, self.zeta = float(k_lin), float(k_ang), float(zeta)
        self.name = "scripted_twist_pull" if self.twist else "scripted_pull"
        self._action: torch.Tensor | None = None

    def reset(self, wrapper) -> None:
        weld = wrapper._env.plant_geometry()["weld_direction"].to(wrapper.device, torch.float32)
        n = weld.shape[0]
        a = torch.zeros(n, 13, device=wrapper.device)
        a[:, :3] = weld * self.rate
        a[:, 3:6] = weld * self.twist
        a[:, 6:9] = self.k_lin
        a[:, 9:12] = self.k_ang
        a[:, 12] = self.zeta
        self._action = wrapper.action_scaler.to_policy(a).clamp(-1.0, 1.0)

    def act(self, wrapper, obs, state) -> torch.Tensor:
        del obs, state
        if self._action is None:
            self.reset(wrapper)
        return self._action


BASELINES = {
    "zero": ZeroPolicy,
    "random": RandomPolicy,
    "scripted_pull": ScriptedPullPolicy,
    "scripted_twist_pull": lambda: ScriptedPullPolicy(rate_m_per_step=0.001, twist_rad_per_step=0.01),
}
