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

- :class:`ScriptedStemRotatePullPolicy` -- the stem-aware, non-learned baseline: with the spur-stem junction
  and the stem axis known, swing the TCP rigidly 30 deg about the junction (about the sideways axis, so the
  apple goes out toward the robot), then pull at a constant rate along the swung stem axis. It never reads
  the F/T sensor; the env decides detachment.

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


def rotation_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation matrices ``(N, 3, 3)`` for unit-or-not ``axis (N, 3)`` and ``angle (N,)`` (rad)."""
    k = torch.nn.functional.normalize(axis, dim=-1)
    z = torch.zeros_like(k[:, 0])
    kx = torch.stack([z, -k[:, 2], k[:, 1], k[:, 2], z, -k[:, 0], -k[:, 1], k[:, 0], z], dim=-1).reshape(-1, 3, 3)
    eye = torch.eye(3, dtype=k.dtype, device=k.device).expand_as(kx)
    s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
    return eye + s * kx + (1.0 - c) * kx @ kx


class ScriptedStemRotatePullPolicy:
    """Swing ``rotate_deg`` about the spur-stem junction, then pull along the swung stem axis.

    Uses privileged plant state (like the critic): the junction position and the junction -> apple stem
    axis ``a`` at the first step of an episode. The swing axis is ``a x o`` with ``o`` the outward (away
    from the plant, toward the robot) grip axis, so it is horizontal for a hanging stem and rotating by
    ``+rotate_deg`` tips the apple toward ``o``; it falls back to ``o x down`` if ``a`` is parallel to ``o``.
    Phase 1 turns the target pose rigidly about the junction in ``rotate_step_deg`` steps (position moves
    on the arc, orientation turns with it); phase 2 moves ``rate_m_per_step`` per step along ``R a``.
    Gains are fixed. Open loop: no force feedback.
    """

    name = "scripted_stem_rotate_pull"

    def __init__(
        self,
        *,
        rotate_deg: float = 30.0,
        rotate_step_deg: float = 2.0,
        rate_m_per_step: float = 0.002,
        k_lin: float = 150.0,
        k_ang: float = 20.0,
        zeta: float = 0.9,
    ) -> None:
        self.rotate = math.radians(float(rotate_deg))
        self.step = math.radians(float(rotate_step_deg))
        self.rate = float(rate_m_per_step)
        self.k_lin, self.k_ang, self.zeta = float(k_lin), float(k_ang), float(zeta)
        self._turned = 0.0
        self._pivot: torch.Tensor | None = None
        self._axis: torch.Tensor | None = None
        self._pull_dir: torch.Tensor | None = None

    def reset(self, wrapper) -> None:
        del wrapper
        self._turned = 0.0
        self._pivot = self._axis = self._pull_dir = None

    def _latch(self, wrapper) -> None:
        env = wrapper._env
        dev = wrapper.device
        info = env._last_full_obs["woody_part_info"][env.target_junction_name if hasattr(env, "target_junction_name") else "spur_stem"]
        pivot = info["anchors_pos"][:, 3:6].to(dev, torch.float32)
        stem = env._target_stem_axis()
        if stem is None:
            raise RuntimeError("scripted_stem_rotate_pull needs a stem-apple junction (stem axis unavailable)")
        stem = stem.to(dev, torch.float32)
        out = torch.nn.functional.normalize(env.plant_geometry()["weld_direction"].to(dev, torch.float32), dim=-1)
        axis = torch.cross(stem, out, dim=-1)
        down = torch.tensor([0.0, 0.0, -1.0], device=dev).expand_as(out)
        side = torch.cross(out, down, dim=-1)
        axis = torch.where(torch.linalg.norm(axis, dim=-1, keepdim=True) < 0.1, side, axis)
        self._pivot = pivot
        self._axis = torch.nn.functional.normalize(axis, dim=-1)
        full = rotation_matrix(self._axis, torch.full((pivot.shape[0],), self.rotate, device=dev))
        self._pull_dir = torch.nn.functional.normalize((full @ stem.unsqueeze(-1)).squeeze(-1), dim=-1)

    def act(self, wrapper, obs, state) -> torch.Tensor:
        del obs, state
        from apple_pick_gym.batched_envs.harvest_action import _quat_conj_wxyz, _rotate_by_quat_wxyz

        if self._pivot is None:
            self._latch(wrapper)
        env, n = wrapper._env, wrapper.num_envs
        a = torch.zeros(n, 13, device=wrapper.device)
        d_ang = min(self.step, max(self.rotate - self._turned, 0.0))
        if d_ang > 0.0:
            arm = env._target_pose[:, :3].to(wrapper.device, torch.float32) - self._pivot
            r = rotation_matrix(self._axis, torch.full((n,), d_ang, device=wrapper.device))
            a[:, :3] = (r @ arm.unsqueeze(-1)).squeeze(-1) - arm
            a[:, 3:6] = self._axis * d_ang
            self._turned += d_ang
        else:
            a[:, :3] = self._pull_dir * self.rate
        if getattr(env, "obs_frame", "world") == "tool":  # express the world delta in the TCP frame
            q = env._last_tcp_pose_wxyz[:, 3:7].to(wrapper.device, torch.float32)
            inv = _quat_conj_wxyz(q)
            a[:, :3] = _rotate_by_quat_wxyz(inv, a[:, :3])
            a[:, 3:6] = _rotate_by_quat_wxyz(inv, a[:, 3:6])
        a[:, 6:9] = self.k_lin
        a[:, 9:12] = self.k_ang
        a[:, 12] = self.zeta
        return wrapper.action_scaler.to_policy(a).clamp(-1.0, 1.0)


BASELINES = {
    "zero": ZeroPolicy,
    "random": RandomPolicy,
    "scripted_pull": ScriptedPullPolicy,
    "scripted_twist_pull": lambda: ScriptedPullPolicy(rate_m_per_step=0.001, twist_rad_per_step=0.01),
    "scripted_bend_pull": ScriptedBendPullPolicy,
    "scripted_stem_rotate_pull": ScriptedStemRotatePullPolicy,
    "scripted_down_pull": lambda: ScriptedBendPullPolicy(bend_m=0.0, pull_angle_deg=0.0, name="scripted_down_pull"),
}
