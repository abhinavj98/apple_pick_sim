"""Reference policies for the harvest task, acting in the wrapper's ``[-1, 1]^13`` box.

- :class:`ZeroPolicy` -- hold the target at mid stiffness / mid damping (the "do nothing"
  floor: rest loads only).
- :class:`RandomPolicy` -- uniform in the action box (exploration floor, safety check).
- :class:`ScriptedPullPolicy` -- the screening pull: retreat along each env's grasp
  axis (``+weld_direction``, away from the plant), optionally twisting about it, at fixed
  stiffness and damping. With a tool-frame action ([D17]) that is ``-z`` of the TCP, which
  points into the apple. The policy the learned one has to beat on success.
- :class:`ScriptedBendPullPolicy` -- [D17] the hand-written bend-and-pull: swing the apple
  sideways to bend the stem, then pull down and out along the bent stem. The policy the learned
  one has to match on collateral. ``scripted_down_pull`` is the same script with no bend and a
  straight-down pull: the axial, no-bend reference. With an equator grasp the pull-out along the
  grip axis is horizontal, so ``scripted_pull`` itself swings the stem (measured on the
  surrogate: theta_b ~39 deg at detach).

Each exposes ``reset(wrapper)`` (called after every env reset) and
``act(wrapper, obs, state) -> (N, 13)``.
"""

from __future__ import annotations

import math

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
        if getattr(wrapper._env, "obs_frame", "world") == "tool":  # [D17] TCP frame: -z is out
            weld = torch.tensor([0.0, 0.0, -1.0], device=wrapper.device).expand(n, 3)
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


class ScriptedBendPullPolicy:
    """[D17] Bend the stem sideways, then pull down and out along it, at fixed gains.

    Needs no plant knowledge: with a hanging stem, any horizontal move of the apple swings the stem
    about the spur-stem junction. Phase 1 moves the TCP ``bend_m`` horizontally (the tool x axis
    projected onto the horizontal plane, so the direction varies with the grasp); phase 2 pulls
    along ``sin(pull_angle) * lateral + cos(pull_angle) * down``, roughly along the bent stem. The
    down / lateral directions come from ``gravity_tool`` in the tool-frame actor obs, or from the
    world axes for a world-frame action.
    """

    def __init__(
        self,
        *,
        rate_m_per_step: float = 0.002,
        bend_m: float = 0.025,
        pull_angle_deg: float = 40.0,
        k_lin: float = 300.0,
        k_ang: float = 20.0,
        zeta: float = 0.9,
        name: str = "scripted_bend_pull",
    ) -> None:
        self.name = name
        self.rate, self.bend_m = float(rate_m_per_step), float(bend_m)
        self.pull_angle = math.radians(float(pull_angle_deg))
        self.k_lin, self.k_ang, self.zeta = float(k_lin), float(k_ang), float(zeta)
        self._lateral: torch.Tensor | None = None
        self._moved: torch.Tensor | None = None

    def reset(self, wrapper) -> None:
        self._lateral = None
        self._moved = torch.zeros(wrapper.num_envs, device=wrapper.device)

    def _down_and_lateral(self, wrapper, obs) -> tuple[torch.Tensor, torch.Tensor]:
        n, dev = wrapper.num_envs, wrapper.device
        if getattr(wrapper._env, "obs_frame", "world") == "tool":
            down = torch.nn.functional.normalize(obs[:, wrapper.actor_layout.slice_for("gravity_tool")].float(), dim=-1)
            ref = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
        else:
            down = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(n, 3)
            ref = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
        lateral = ref - (ref * down).sum(-1, keepdim=True) * down
        # a tool x almost along gravity: fall back to tool y
        alt = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3)
        alt = alt - (alt * down).sum(-1, keepdim=True) * down
        small = torch.linalg.norm(lateral, dim=-1, keepdim=True) < 0.2
        return down, torch.nn.functional.normalize(torch.where(small, alt, lateral), dim=-1)

    def act(self, wrapper, obs, state) -> torch.Tensor:
        del state
        if self._moved is None:
            self.reset(wrapper)
        down, lateral = self._down_and_lateral(wrapper, obs)
        if self._lateral is None:
            self._lateral = lateral
        bending = (self._moved < self.bend_m).unsqueeze(-1)
        pull_dir = torch.nn.functional.normalize(
            math.sin(self.pull_angle) * self._lateral + math.cos(self.pull_angle) * down, dim=-1
        )
        a = torch.zeros(wrapper.num_envs, 13, device=wrapper.device)
        a[:, :3] = self.rate * torch.where(bending, self._lateral, pull_dir)
        a[:, 6:9] = self.k_lin
        a[:, 9:12] = self.k_ang
        a[:, 12] = self.zeta
        self._moved = self._moved + self.rate * bending.squeeze(-1).float()
        return wrapper.action_scaler.to_policy(a).clamp(-1.0, 1.0)


BASELINES = {
    "zero": ZeroPolicy,
    "random": RandomPolicy,
    "scripted_pull": ScriptedPullPolicy,
    "scripted_twist_pull": lambda: ScriptedPullPolicy(rate_m_per_step=0.001, twist_rad_per_step=0.01),
    "scripted_bend_pull": ScriptedBendPullPolicy,
    "scripted_down_pull": lambda: ScriptedBendPullPolicy(bend_m=0.0, pull_angle_deg=0.0, name="scripted_down_pull"),
}
