"""A fast analytic stand-in for ``ApplePickVicHarvestEnv`` with the same RL contract.

**What it is for.** Infrastructure testing on a laptop CPU: the real harvest env needs
CUDA (the batched arm does not integrate on Newton's MuJoCo-CPU backend) and takes
minutes to build. This env runs thousands of steps per second, exposes the *same*
observation dict, ``info`` keys, action bounds, DR accessors and episode semantics, and
reuses the real reward / success / freeze code
(:func:`apple_pick_gym.batched_envs.harvest_outcome.evaluate_harvest_step`), so the whole
skrl pipeline (wrapper, critic state, LSTM models, trainer, checkpoints) can be exercised
and shown to learn before anything touches the simulator.

**What it is not.** Plant physics. The plant is a set of linear springs around the grasp,
chosen so the task has the right *structure*, not the right numbers:

- the TCP is a gravity-compensated rigid body (mass, inertia) driven by the same VIC law
  the real ``vic_pose`` controller uses: ``K (target - x) - D v`` with ``D = 2 zeta sqrt(K)``;
- pulling the apple along the grasp's weld axis (away from the plant) loads the stem in
  tension; pushing back barely loads it (the stem buckles); sideways motion is stiffer;
- twisting about the weld axis loads the spur-stem junction with torque; bending about the
  lateral axes is stiffer;
- the spur-stem junction carries its rest (gravity) load plus the pull / twist load; the
  other junctions carry their rest load plus a share of the *pull* load (serial chain), but
  almost none of the twist -- so, as with a real picker, twist-and-pull reaches the
  elliptical detach envelope with much less collateral force than a straight pull.

**Domain randomization** mirrors the real env's split: per env, fixed for the run --
spring stiffnesses, rest loads, collateral shares, grasp axis (the "plant" and "grasp");
per reset -- arm mass / damping scales (the "arm joint dynamics") and the F/T sensor's
bias / noise / drift (the same :class:`FtSensorModel`). The privileged accessors return
the same ``_PRIVILEGED_FIELDS`` / ``PLANT_GEOMETRY_FIELDS`` layout, with the surrogate's
draws mapped into those slots (documented in :meth:`privileged_fields`).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import torch

from apple_pick_gym.batched_envs.harvest_action import (
    HarvestActionBounds,
    clamp_commanded_force,
    clamp_target_advance,
    grip_axis,
    integrate_delta_pose,
    leash_target_pose,
    split_harvest_action,
    tool_delta_to_world,
)
from apple_pick_gym.batched_envs.harvest_detach import (
    bend_pull_detach_force,
    chord_angle_deg,
    detach_index,
    detach_scales,
    envelope_thresholds,
)
from apple_pick_gym.batched_envs.harvest_episode import EpisodeConfig, FreezeMask, SuccessStreakTracker
from apple_pick_gym.batched_envs.harvest_obs import (
    _PRIVILEGED_FIELDS,
    rotate_tool_wrench_to_world,
    tool_frame_obs,
)
from apple_pick_gym.batched_envs.harvest_outcome import evaluate_harvest_step
from apple_pick_gym.batched_envs.harvest_privileged import PLANT_GEOMETRY_FIELDS
from apple_pick_gym.batched_envs.harvest_reward import HarvestRewardConfig
from apple_pick_gym.batched_envs.sensor_realism import FtSensorConfig, FtSensorModel

JUNCTION_NAMES = ("primary_spur", "spur_stem", "stem_apple", "primary_support_left", "primary_support_right")
_ACTION_DIM = 13


@dataclasses.dataclass(frozen=True)
class SurrogatePlantRanges:
    """Per-env DR ranges (uniform) of the spring plant; fixed for the run."""

    pull_stiffness_n_m: tuple[float, float] = (150.0, 500.0)
    push_stiffness_ratio: float = 0.05
    lateral_stiffness_n_m: tuple[float, float] = (400.0, 900.0)
    twist_stiffness_nm_rad: tuple[float, float] = (0.03, 0.08)
    bend_stiffness_nm_rad: tuple[float, float] = (0.3, 0.8)
    rest_force_n: tuple[float, float] = (3.5, 5.5)  # spur-stem gravity load
    rest_torque_nm: tuple[float, float] = (0.0, 0.015)
    collateral_rest_force_n: tuple[float, float] = (2.5, 6.5)
    collateral_pull_share: tuple[float, float] = (0.7, 1.0)
    collateral_twist_force_per_nm: tuple[float, float] = (0.5, 2.0)  # N of collateral per N*m of twist
    grasp_max_polar_angle_rad: float = math.pi / 6.0


@dataclasses.dataclass(frozen=True)
class SurrogateArmRanges:
    """Per-reset arm DR (the surrogate's stand-in for joint armature/friction/damping)."""

    mass_kg: float = 1.5
    inertia_kgm2: float = 0.02
    mass_scale: tuple[float, float] = (0.8, 1.2)
    damping_scale: tuple[float, float] = (0.7, 1.3)


class SurrogateHarvestEnv:
    """Batched spring-plant harvest env with ``ApplePickVicHarvestEnv``'s RL contract."""

    TARGET_JUNCTION_NAME = "spur_stem"

    def __init__(
        self,
        *,
        num_envs: int,
        device: str | torch.device = "cpu",
        max_episode_steps: int = 120,
        control_hz: float = 60.0,
        substeps: int = 8,
        action_bounds: HarvestActionBounds | None = None,
        reward_config: HarvestRewardConfig | None = None,
        episode_config: EpisodeConfig | None = None,
        ft_sensor_config: FtSensorConfig | None = None,
        plant_ranges: SurrogatePlantRanges | None = None,
        arm_ranges: SurrogateArmRanges | None = None,
        invalid_fraction: float = 0.0,
        seed: int = 0,
        action_frame: str = "world",
        realized_k_scale_range: tuple[float, float] | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        if action_frame not in ("world", "tool"):
            raise ValueError(f"action_frame must be 'world' or 'tool', got {action_frame!r}")
        self._action_frame = action_frame  # [D17] pose deltas, VIC gains and F/T in the TCP frame
        self._realized_k_scale_range = realized_k_scale_range  # [D17] applied / commanded K DR
        self._realized_k_scale = torch.ones(int(num_envs), 6, device=torch.device(device))
        self._detach_scale = torch.ones(int(num_envs), device=torch.device(device))
        self._start: tuple[torch.Tensor, torch.Tensor] | None = None
        self.device = torch.device(device)
        self._max_episode_steps = int(max_episode_steps)
        self._dt = 1.0 / float(control_hz)
        self._substeps = int(substeps)
        self._action_bounds = action_bounds or HarvestActionBounds()
        self._reward_cfg = reward_config or HarvestRewardConfig()
        self._check_criterion(self._reward_cfg.detach.criterion)
        self._episode_cfg = episode_config or EpisodeConfig()
        self._plant = plant_ranges or SurrogatePlantRanges()
        self._arm = arm_ranges or SurrogateArmRanges()
        self._rng = np.random.default_rng(seed)
        self._gen = torch.Generator(device="cpu").manual_seed(int(seed))
        cfg = ft_sensor_config or FtSensorConfig(control_hz=control_hz)
        self._ft_sensor = FtSensorModel(num_envs=self.num_envs, device=self.device, config=cfg)
        self._tracker = SuccessStreakTracker(self.num_envs, self.device)
        self._freeze = FreezeMask(self.num_envs, self.device)

        self._sample_plant()
        n_invalid = int(round(float(invalid_fraction) * self.num_envs))
        invalid = np.zeros(self.num_envs, dtype=bool)
        if n_invalid:
            invalid[self._rng.choice(self.num_envs, size=n_invalid, replace=False)] = True
        self._invalid = torch.as_tensor(invalid, device=self.device)
        self._joint_map = torch.as_tensor(self._rng.normal(0.0, 0.5, size=(6, 7)), dtype=torch.float32, device=self.device)
        self._q_rest = torch.as_tensor(self._rng.uniform(-1.0, 1.0, size=(1, 7)), dtype=torch.float32, device=self.device)
        self._arm_sample: dict[str, torch.Tensor] | None = None
        self._step_count = 0
        det = self._reward_cfg.detach  # [D7] nominal until the first reset draws per-env limits
        self._peak_collateral: torch.Tensor | None = None
        self._thresholds = torch.tensor([[float(det.f_max_n), float(det.tau_max_nm)]] * self.num_envs, device=self.device)

    def reseed_episode_rng(self, seed: int) -> None:
        """Reseed the per-reset DR stream (arm draws); the per-env plant is unaffected."""
        self._rng = np.random.default_rng(int(seed))

    # ------------------------------------------------------------------ properties
    @property
    def action_bounds(self) -> HarvestActionBounds:
        return self._action_bounds

    @property
    def junction_names(self) -> list[str]:
        return list(JUNCTION_NAMES)

    @property
    def target_junction_name(self) -> str:
        return self.TARGET_JUNCTION_NAME

    @property
    def episode_config(self) -> EpisodeConfig:
        return self._episode_cfg

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    @property
    def invalid_env_mask(self) -> torch.Tensor:
        return self._invalid

    @property
    def unwrapped(self) -> SurrogateHarvestEnv:
        return self

    @property
    def obs_frame(self) -> str:
        """[D17] ``"tool"``: the actor reads the start-relative TCP-frame fields."""
        return self._action_frame

    def _check_criterion(self, criterion: str) -> None:
        if criterion != "envelope":
            raise ValueError(
                "SurrogateHarvestEnv implements only the 'envelope' detach criterion "
                f"(got {criterion!r}); use SurrogateBendPullEnv for 'bend_pull'"
            )

    # ------------------------------------------------------------------ DR
    def _u(self, bounds: tuple[float, float], size: Any = None) -> torch.Tensor:
        n = self.num_envs if size is None else size
        return torch.as_tensor(self._rng.uniform(bounds[0], bounds[1], size=n), dtype=torch.float32, device=self.device)

    def _sample_plant(self) -> None:
        p, n = self._plant, self.num_envs
        self.k_pull = self._u(p.pull_stiffness_n_m)
        self.k_lat = self._u(p.lateral_stiffness_n_m)
        self.k_twist = self._u(p.twist_stiffness_nm_rad)
        self.k_bend = self._u(p.bend_stiffness_nm_rad)
        self.rest_force = self._u(p.rest_force_n)
        # rest torque about a random horizontal axis
        ang = self._u((0.0, 2.0 * math.pi))
        self.rest_torque = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], -1) * self._u(p.rest_torque_nm).unsqueeze(-1)
        others = [j for j in JUNCTION_NAMES if j != self.TARGET_JUNCTION_NAME]
        self.coll_rest = {j: self._u(p.collateral_rest_force_n) for j in others}
        self.coll_pull = {j: self._u(p.collateral_pull_share) for j in others}
        self.coll_twist = {j: self._u(p.collateral_twist_force_per_nm) for j in others}
        # grasp (weld) axis: within a cone around straight down, like the real default grasps
        polar = torch.as_tensor(np.arccos(1.0 - self._rng.uniform(0, 1, n) * (1.0 - math.cos(p.grasp_max_polar_angle_rad))), dtype=torch.float32)
        az = self._u((0.0, 2.0 * math.pi)).cpu()
        self.weld = torch.stack([torch.sin(polar) * torch.cos(az), torch.sin(polar) * torch.sin(az), -torch.cos(polar)], -1).to(self.device)
        self.tcp_rest = torch.as_tensor(self._rng.normal([0.0, 0.65, 0.35], 0.02, size=(n, 3)), dtype=torch.float32, device=self.device)
        self.q_rest = torch.zeros(n, 4, device=self.device)
        self.q_rest[:, 0] = 1.0  # wxyz; the v1 rest orientation is the identity for every env

    def _resample_arm(self) -> None:
        a = self._arm
        self._arm_sample = {"mass_scale": self._u(a.mass_scale), "damping_scale": self._u(a.damping_scale)}

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        n, dev = self.num_envs, self.device
        self._step_count = 0
        self._x = self.tcp_rest.clone()
        self._v = torch.zeros(n, 3, device=dev)
        self._q = self.q_rest.clone()
        self._w = torch.zeros(n, 3, device=dev)
        self._target = torch.cat([self._x, self._q], -1)
        self._last_action = torch.zeros(n, _ACTION_DIM, device=dev)
        self._kp = torch.zeros(n, 6, device=dev)
        self._kd = torch.zeros(n, 6, device=dev)
        self._resample_arm()
        self._thresholds = envelope_thresholds(self._reward_cfg.detach, n, self._rng, device=dev)  # [D7]
        if self._realized_k_scale_range is not None:  # [D17]
            lo, hi = (float(v) for v in self._realized_k_scale_range)
            self._realized_k_scale = self._u((lo, hi), size=(n, 6)).reshape(n, 6)
        self._draw_episode_detach()
        self._start = None  # [D17] recorded by the first observation of the episode
        self._ft_sensor.reset()
        self._tracker.reset()
        self._freeze.reset()
        self._freeze.update(self._invalid)
        obs, info = self._observe()
        self._collateral_baseline = {
            j: torch.linalg.norm(w[:, :3], dim=-1).clone() for j, w in info["woody_part_force"].items() if j != self.TARGET_JUNCTION_NAME
        }
        # [D17m] rest force vectors for collateral_mode="deviation"
        self._collateral_baseline_force = {
            j: w[:, :3].clone() for j, w in info["woody_part_force"].items() if j != self.TARGET_JUNCTION_NAME
        }
        self._collateral_baseline_torque_norm = {  # [D17q]
            j: torch.linalg.norm(t, dim=-1).clone() for j, t in info["woody_part_torque_anchor"].items()
        }
        info["collateral_baseline_norm"] = self._collateral_baseline
        info["collateral_baseline_force"] = self._collateral_baseline_force
        info["collateral_baseline_torque_norm"] = self._collateral_baseline_torque_norm
        info["tcp_advance_m"] = self._tcp_advance()
        self._peak_collateral = None  # [D13] running peak collateral, reset per episode
        self._progress_prev = torch.clamp(info["detach_utilization"], min=0.0, max=1.0)
        return obs, info

    def _tcp_advance(self) -> torch.Tensor:
        """[D17m] TCP travel (m) since the start along the reset grip axis (+ = into the tree), ``(N,)``."""
        start_pos, start_q = self._start
        return ((self._x - start_pos) * grip_axis(start_q)).sum(-1)

    def _draw_episode_detach(self) -> None:
        """Per-reset detach-criterion draws (the envelope criterion has none beyond D7)."""

    def step(self, action: torch.Tensor):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.shape != (self.num_envs, _ACTION_DIM):
            raise ValueError(f"actions shape must be ({self.num_envs}, {_ACTION_DIM}), got {tuple(action.shape)}")
        action = self._freeze.apply_to_delta_action(action, self._last_action, delta_dims=6)
        split = split_harvest_action(action, self._action_bounds)
        b = self._action_bounds
        delta = split.delta
        if self._action_frame == "tool":
            delta = tool_delta_to_world(delta, self._q)  # [D17]
        target = integrate_delta_pose(self._target, delta)
        if b.max_target_pos_offset_m is not None or b.max_target_rot_offset_rad is not None:
            target = leash_target_pose(
                target,
                torch.cat([self._x, self._q], -1),
                max_pos_offset_m=b.max_target_pos_offset_m,
                max_rot_offset_rad=b.max_target_rot_offset_rad,
            )
        if b.max_commanded_force_n is not None:  # [D17n] after the leash, before the wall
            target = clamp_commanded_force(
                target,
                torch.cat([self._x, self._q], -1),
                split.linear_k,
                max_force_n=b.max_commanded_force_n,
                tool_frame=self._action_frame == "tool",
            )
        if b.max_advance_m is not None:  # [D17m] after the leash: the wall is the hard guarantee
            start_pos, start_q = self._start
            target = clamp_target_advance(target, start_pos, grip_axis(start_q), max_advance_m=b.max_advance_m)
        self._target = target
        kp_cmd = torch.cat([split.linear_k, split.angular_k], -1)
        self._kd = 2.0 * split.zeta * torch.sqrt(kp_cmd)
        self._kp = kp_cmd * self._realized_k_scale  # [D17] applied stiffness; damping as commanded
        self._last_action = action.clone()
        self._integrate()
        self._step_count += 1

        obs, info = self._observe()
        info["collateral_baseline_norm"] = self._collateral_baseline
        info["collateral_baseline_force"] = self._collateral_baseline_force
        info["collateral_baseline_torque_norm"] = self._collateral_baseline_torque_norm
        info["tcp_advance_m"] = self._tcp_advance()
        outcome = evaluate_harvest_step(
            obs,
            info,
            reward_cfg=self._reward_cfg,
            episode_cfg=self._episode_cfg,
            tracker=self._tracker,
            freeze_mask=self._freeze,
            target_junction_name=self.TARGET_JUNCTION_NAME,
            progress_prev=self._progress_prev,
            peak_collateral_prev=self._peak_collateral,
        )
        self._progress_prev = outcome.progress
        self._peak_collateral = outcome.peak_collateral
        info["reward_terms"] = outcome.reward_terms
        info["episode"] = outcome.episode
        info["target_pose"] = self._target.clone()
        truncated = torch.full((self.num_envs, 1), self._step_count >= self._max_episode_steps, dtype=torch.bool, device=self.device)
        return obs, outcome.reward, outcome.terminated.unsqueeze(-1), truncated, info

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ physics
    def _plant_wrench_on_tcp(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Plant spring wrench on the TCP, and the pull / twist loads it puts on the stem."""
        e = self.weld
        dx = self._x - self.tcp_rest
        along = (dx * e).sum(-1, keepdim=True)
        perp = dx - along * e
        pull = self.k_pull.unsqueeze(-1) * torch.where(along > 0, along, self._plant.push_stiffness_ratio * along)
        force = -(pull * e) - self.k_lat.unsqueeze(-1) * perp
        rot = _quat_to_rotvec_wxyz(self._q)
        twist = (rot * e).sum(-1, keepdim=True)
        bend = rot - twist * e
        torque = -(self.k_twist.unsqueeze(-1) * twist * e) - self.k_bend.unsqueeze(-1) * bend
        return force, torque, pull.squeeze(-1), (self.k_twist * twist.squeeze(-1))

    def _integrate(self) -> None:
        h = self._dt / self._substeps
        m = self._arm.mass_kg * self._arm_sample["mass_scale"].unsqueeze(-1)
        inertia = self._arm.inertia_kgm2 * self._arm_sample["mass_scale"].unsqueeze(-1)
        dscale = self._arm_sample["damping_scale"].unsqueeze(-1)
        d_lin = dscale * self._kd[:, :3]
        d_ang = dscale * self._kd[:, 3:]
        tool = self._action_frame == "tool"
        for _ in range(self._substeps):
            f_plant, t_plant, _, _ = self._plant_wrench_on_tcp()
            rot_err = _quat_to_rotvec_wxyz(_quat_mul_wxyz(self._target[:, 3:], _quat_conj_wxyz(self._q)))
            # [D17] per-axis gains act in the gain frame (TCP axes for "tool", world axes otherwise);
            # the implicit damping update is per axis, so it runs in that frame too
            g = self._q if tool else None
            to_g = (lambda v: _rotate_wxyz(_quat_conj_wxyz(g), v)) if tool else (lambda v: v)
            from_g = (lambda v: _rotate_wxyz(g, v)) if tool else (lambda v: v)
            e_p, v_g = to_g(self._target[:, :3] - self._x), to_g(self._v)
            e_r, w_g = to_g(rot_err), to_g(self._w)
            # Damping is integrated implicitly: D = 2 zeta sqrt(K) is sized for unit inertia, and
            # explicit Euler diverges once h*D/I > 2 (zeta = 2 at I = 0.02 kg m^2 is h*D/I ~ 3).
            v_g = (v_g + h * (self._kp[:, :3] * e_p + to_g(f_plant)) / m) / (1.0 + h * d_lin / m)
            w_g = (w_g + h * (self._kp[:, 3:] * e_r + to_g(t_plant)) / inertia) / (1.0 + h * d_ang / inertia)
            self._v, self._w = from_g(v_g), from_g(w_g)
            self._x = self._x + h * self._v
            dq = _rotvec_to_quat_wxyz(self._w * h)
            self._q = _quat_mul_wxyz(dq, self._q)
            self._q = self._q / torch.linalg.norm(self._q, dim=-1, keepdim=True)

    def _observe(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        n, dev = self.num_envs, self.device
        f_plant, t_plant, pull, twist = self._plant_wrench_on_tcp()
        e = self.weld
        # spur-stem: gravity rest load (down) + the stem's pull (along the weld axis) + twist
        down = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(n, 3)
        f_ss = self.rest_force.unsqueeze(-1) * down + torch.clamp(pull, min=-self.rest_force).unsqueeze(-1) * e
        t_ss = self.rest_torque + twist.unsqueeze(-1) * e
        target_wrench = torch.cat([f_ss, t_ss], -1)
        woody = {self.TARGET_JUNCTION_NAME: target_wrench}
        for j in JUNCTION_NAMES:
            if j == self.TARGET_JUNCTION_NAME:
                continue
            mag = self.coll_rest[j] + self.coll_pull[j] * torch.clamp(pull, min=0.0) + self.coll_twist[j] * twist.abs()
            woody[j] = torch.cat([mag.unsqueeze(-1) * down, torch.zeros(n, 3, device=dev)], -1)
        # wrist: plant reaction on the TCP (the gravity-compensated arm carries the apple weight too)
        apple_weight = torch.tensor([0.0, 0.0, -0.15 * 9.81], device=dev).expand(n, 3)
        ft_raw = torch.cat([f_plant + apple_weight, t_plant], -1)
        obs = self._build_obs(ft_raw)
        info = {
            "obs_layout": "surrogate_vic_harvest",
            "step_count": self._step_count,
            "invalid_env": self._invalid.clone(),
            "woody_part_force": woody,
            # [D17q] the surrogate's junction moments are already about the joint (no COM offset)
            "woody_part_torque_anchor": {j: w[:, 3:6] for j, w in woody.items() if j != self.TARGET_JUNCTION_NAME},
            "target_junction_force": target_wrench,
            "target_junction_wrench": target_wrench,
            "target_junction_axis": self.weld.clone(),  # the surrogate's stem axis is the grasp axis
            "detach_thresholds": self._thresholds,
            "detach_index": detach_index(target_wrench, self._reward_cfg.detach, stem_axis=self.weld, thresholds=self._thresholds),
            "ft_wrist": ft_raw,
        }
        info["detach_utilization"] = torch.sqrt(info["detach_index"])  # [D17] what progress reads
        return obs, info

    def _build_obs(self, ft_raw: torch.Tensor) -> dict[str, torch.Tensor]:
        """The real env's obs dict (world fields + [D17] TCP-frame fields) from the TCP state."""
        n, dev = self.num_envs, self.device
        q_xyzw = self._q[:, [1, 2, 3, 0]]  # xyzw, as in the real env's obs
        vel = torch.cat([self._v, self._w], -1)
        if self._action_frame == "tool":  # the sensor model runs in the sensor's own frame
            raw_tool = tool_frame_obs(
                tcp_pos=self._x, tcp_quat=q_xyzw, tcp_velocity=vel, ft=ft_raw,
                start_pos=self._x, start_quat=q_xyzw, target_pos=self._x, target_quat=q_xyzw,
            )["ft_tool"]
            ft_tool = self._ft_sensor.step(raw_tool)
            ft_world = rotate_tool_wrench_to_world(ft_tool, q_xyzw)
        else:
            ft_world = self._ft_sensor.step(ft_raw)
        if self._start is None:
            self._start = (self._x.clone(), q_xyzw.clone())
        tool = tool_frame_obs(
            tcp_pos=self._x,
            tcp_quat=q_xyzw,
            tcp_velocity=vel,
            ft=ft_world,
            start_pos=self._start[0],
            start_quat=self._start[1],
            target_pos=self._target[:, :3],
            target_quat=self._target[:, [4, 5, 6, 3]],
        )
        if self._action_frame == "tool":
            tool["ft_tool"] = ft_tool
        rot = _quat_to_rotvec_wxyz(self._q)
        pose6 = torch.cat([self._x - self.tcp_rest, rot], -1)
        return {
            "tcp_pos": self._x.clone(),
            "tcp_quat": q_xyzw.clone(),
            "tcp_velocity": vel,
            "ft_wrist": ft_world,
            "robot_joint_q": self._q_rest + pose6 @ self._joint_map,
            "last_action": self._last_action.clone(),
            "step_frac": torch.full((n, 1), self._step_count / self._max_episode_steps, device=dev),
            **tool,
        }

    # ------------------------------------------------------------------ privileged
    def privileged_fields(self) -> dict[str, torch.Tensor]:
        """The real critic layout, filled with the surrogate's DR draws.

        Mapping: spur/stem flexural and axial moduli slots carry ``log10`` of the pull,
        lateral, twist and bend stiffnesses; damping-ratio slots carry the rest force /
        torque; ``primary_density`` the mean collateral pull share; support slots the
        rest collateral loads; arm armature / friction / damping slots the per-reset arm
        mass and damping scales (broadcast to 7 joints); build-time arm scales are 1.
        """
        col = lambda t: t.reshape(-1, 1)
        others = list(self.coll_rest)
        pull_share = torch.stack([self.coll_pull[j] for j in others], -1).mean(-1)
        rest = torch.stack([self.coll_rest[j] for j in others], -1)
        arm = self._arm_sample or {"mass_scale": torch.ones(self.num_envs, device=self.device), "damping_scale": torch.ones(self.num_envs, device=self.device)}
        ones = torch.ones(self.num_envs, 1, device=self.device)
        vals = {
            "spur_flexural_modulus_pa": col(torch.log10(self.k_bend)),
            "stem_flexural_modulus_pa": col(torch.log10(self.k_twist)),
            "spur_youngs_modulus_pa": col(torch.log10(self.k_lat)),
            "stem_youngs_modulus_pa": col(torch.log10(self.k_pull)),
            "spur_damping_ratio": col(self.rest_force),
            "stem_damping_ratio": col(torch.linalg.norm(self.rest_torque, dim=-1)),
            "primary_density": col(pull_share),
            "support_kp": col(rest[:, 0]),
            "support_roll_kp": col(rest[:, 1]),
            "support_zeta": col(rest[:, 2]),
            "arm_armature": col(arm["mass_scale"]).expand(-1, 7).clone(),
            "arm_friction": col(arm["damping_scale"]).expand(-1, 7).clone(),
            "arm_joint_damping": col(arm["damping_scale"]).expand(-1, 7).clone(),
            "arm_link_mass_scale": ones.clone(),
            "arm_link_inertia_scale": ones.clone(),
            "arm_ee_mass_scale": ones.clone(),
            "arm_ee_inertia_scale": ones.clone(),
            "detach_scale": self._detach_scale.reshape(-1, 1).clone(),  # [D17] 1 for the envelope
            "realized_k_scale": self._realized_k_scale.clone(),  # [D17] applied / commanded K
        }
        return {name: vals[name] for name, _ in _PRIVILEGED_FIELDS}

    def plant_geometry(self) -> dict[str, torch.Tensor]:
        n = self.num_envs
        z = torch.zeros(n, 1, device=self.device)
        geo = {
            "spur_length_m": z + 0.1,
            "spur_radius_m": z + 0.005,
            "stem_length_m": z + 0.03,
            "stem_radius_m": z + 0.002,
            "apple_radius_m": z + 0.04,
            "apple_density": z + 800.0,
            "weld_direction": self.weld.clone(),
            "grasp_roll": z.clone(),  # [D17] the surrogate has no gripper roll
        }
        return {name: geo[name] for name, _ in PLANT_GEOMETRY_FIELDS}


# ---------------------------------------------------------------------- quaternion helpers (wxyz)
def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def _quat_conj_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[:, :1], -q[:, 1:]], -1)


def _quat_to_rotvec_wxyz(q: torch.Tensor) -> torch.Tensor:
    q = torch.where(q[:, :1] < 0, -q, q)
    s = torch.linalg.norm(q[:, 1:], dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s, q[:, :1])
    return q[:, 1:] * torch.where(s > 1e-9, angle / s.clamp_min(1e-9), torch.full_like(s, 2.0))


def _rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``(N, 3)`` vectors by ``(N, 4)`` wxyz unit quaternions."""
    t = 2.0 * torch.cross(q[:, 1:], v, dim=-1)
    return v + q[:, :1] * t + torch.cross(q[:, 1:], t, dim=-1)


def _rotvec_to_quat_wxyz(r: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(r, dim=-1, keepdim=True)
    half = 0.5 * angle
    k = torch.where(angle > 1e-9, torch.sin(half) / angle.clamp_min(1e-9), torch.full_like(angle, 0.5))
    return torch.cat([torch.cos(half), r * k], -1)


# ---------------------------------------------------------------------- [D17] bend-and-pull surrogate
@dataclasses.dataclass(frozen=True)
class SurrogateBendPullRanges:
    """Per-env DR ranges (uniform) of the pendulum-stem plant; fixed for the run.

    Geometry is loosely real (apple ~35 mm radius, ~30 mm stem). Stiffnesses are chosen for the
    task's *structure* (a straight pull must stretch the stem against ``pull_stiffness``; a
    sideways move swings it against the much softer bending spring), not for realism.
    """

    grasp_pole: tuple[float, float, float] = (0.0, -1.0, 0.0)  # apple -> gripper cone axis
    grasp_max_polar_angle_rad: float = math.pi / 3.0
    grasp_roll_max_rad: float = math.pi / 2.0
    apple_radius_m: tuple[float, float] = (0.032, 0.038)
    stem_length_m: tuple[float, float] = (0.02, 0.04)
    pull_stiffness_n_m: tuple[float, float] = (400.0, 1000.0)  # axial, at the stem
    push_stiffness_ratio: float = 0.05
    bend_stiffness_n_m: tuple[float, float] = (30.0, 80.0)  # lateral, at the stem's apple end
    rest_tension_n: tuple[float, float] = (3.0, 5.0)  # gravity load the stem carries at rest
    collateral_rest_force_n: tuple[float, float] = (2.5, 6.5)
    collateral_pull_share: tuple[float, float] = (0.7, 1.0)  # N per N of extra stem tension
    collateral_bend_share: tuple[float, float] = (0.2, 0.6)  # N per N of bending force


class SurrogateBendPullEnv(SurrogateHarvestEnv):
    """[D17] The v2 task on an analytic plant: equator grasp, bend-and-pull detach, tool frame.

    The apple is welded to the TCP. Its stem runs from a fixed junction ``J`` to the stem's
    attachment point ``a`` on top of the apple; the TCP pose places ``a`` rigidly. The stem pulls
    ``a`` back with an axial spring on its stretch (plus the rest tension the gravity-compensated
    arm already balances) and a lateral spring on its sideways swing. Detachment uses the env's
    bend-pull criterion on the chord ``J -> a``: swinging the stem lowers ``F_d``, so bending first
    and then pulling detaches at a fraction of a straight pull's tension and collateral.
    """

    def __init__(self, *, plant_ranges: SurrogateBendPullRanges | None = None, action_frame: str = "tool", **kw: Any) -> None:
        self._bp = plant_ranges or SurrogateBendPullRanges()
        super().__init__(action_frame=action_frame, **kw)

    def _check_criterion(self, criterion: str) -> None:
        if criterion != "bend_pull":
            raise ValueError(f"SurrogateBendPullEnv needs the 'bend_pull' detach criterion, got {criterion!r}")

    def _sample_plant(self) -> None:
        r, n, dev = self._bp, self.num_envs, self.device
        u = lambda b: self._u(b)
        self.apple_r = u(r.apple_radius_m)
        self.stem_len = u(r.stem_length_m)
        self.k_pull = u(r.pull_stiffness_n_m)
        self.k_lat = u(r.bend_stiffness_n_m)  # privileged slot reuse (see privileged_fields)
        self.k_bend = self.k_lat
        self.k_twist = torch.full((n,), 0.05, device=dev)
        self.rest_force = u(r.rest_tension_n)
        self.rest_torque = torch.zeros(n, 3, device=dev)
        others = [j for j in JUNCTION_NAMES if j != self.TARGET_JUNCTION_NAME]
        self.coll_rest = {j: u(r.collateral_rest_force_n) for j in others}
        self.coll_pull = {j: u(r.collateral_pull_share) for j in others}
        self.coll_twist = {j: u(r.collateral_bend_share) for j in others}
        # grasp: area-uniform cone around the pole, then a roll about the approach axis
        polar = torch.as_tensor(np.arccos(1.0 - self._rng.uniform(0, 1, n) * (1.0 - math.cos(r.grasp_max_polar_angle_rad))), dtype=torch.float32)
        az = torch.as_tensor(self._rng.uniform(0.0, 2.0 * math.pi, n), dtype=torch.float32)
        local = torch.stack([torch.sin(polar) * torch.cos(az), torch.sin(polar) * torch.sin(az), torch.cos(polar)], -1)
        self.weld = _rotate_z_to(local, torch.tensor(r.grasp_pole, dtype=torch.float32)).to(dev)
        self.roll = self._u((-r.grasp_roll_max_rad, r.grasp_roll_max_rad)) if r.grasp_roll_max_rad > 0 else torch.zeros(n, device=dev)
        # TCP orientation: look-at (+z into the apple, x from world up) then the roll, as in the build
        z_ax = -self.weld
        up = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        up = torch.where((z_ax * up).sum(-1, keepdim=True).abs() > 0.99, torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3), up)
        x_ax = torch.nn.functional.normalize(torch.cross(up, z_ax, dim=-1), dim=-1)
        y_ax = torch.cross(z_ax, x_ax, dim=-1)
        q_look = _matrix_to_quat_wxyz(torch.stack([x_ax, y_ax, z_ax], dim=-1))
        self.q_rest = _quat_mul_wxyz(q_look, _rotvec_to_quat_wxyz(torch.stack([torch.zeros(n, device=dev), torch.zeros(n, device=dev), self.roll], -1)))
        # apple centre near the real rig's, TCP on its surface along the weld
        centre = torch.as_tensor(self._rng.normal([0.45, 0.68, 0.45], 0.02, size=(n, 3)), dtype=torch.float32, device=dev)
        self.tcp_rest = centre + self.apple_r.unsqueeze(-1) * self.weld
        # apple-fixed points in the TCP frame: the centre, and the stem attachment (top at rest)
        q_inv = _quat_conj_wxyz(self.q_rest)
        self._attach_tool = _rotate_wxyz(q_inv, centre + self.apple_r.unsqueeze(-1) * up_world(n, dev) - self.tcp_rest)
        self._attach_rest = self.tcp_rest + _rotate_wxyz(self.q_rest, self._attach_tool)
        self._junction = self._attach_rest + self.stem_len.unsqueeze(-1) * up_world(n, dev)

    def _draw_episode_detach(self) -> None:
        self._detach_scale = detach_scales(self._reward_cfg.detach, self.num_envs, self._rng, device=self.device)

    def _stem_state(self):
        a = self._x + _rotate_wxyz(self._q, self._attach_tool)
        chord = a - self._junction
        length = torch.linalg.norm(chord, dim=-1).clamp_min(1e-9)
        u_hat = chord / length.unsqueeze(-1)
        stretch = length - self.stem_len
        t_dyn = self.k_pull * torch.where(stretch > 0, stretch, self._bp.push_stiffness_ratio * stretch)
        rest_axis = -up_world(self.num_envs, self.device)
        d = a - self._attach_rest
        perp = d - (d * rest_axis).sum(-1, keepdim=True) * rest_axis
        f_bend = -self.k_lat.unsqueeze(-1) * perp
        return a, chord, u_hat, t_dyn, f_bend

    def _plant_wrench_on_tcp(self):
        a, _chord, u_hat, t_dyn, f_bend = self._stem_state()
        force = -t_dyn.unsqueeze(-1) * u_hat + f_bend  # on the apple at a, carried rigidly by the TCP
        torque = torch.cross(a - self._x, force, dim=-1)
        return force, torque, t_dyn, torch.linalg.norm(f_bend, dim=-1)

    def _observe(self):
        n, dev = self.num_envs, self.device
        f_plant, t_plant, t_dyn, bend_mag = self._plant_wrench_on_tcp()
        _a, chord, u_hat, _t, f_bend = self._stem_state()
        tension = self.rest_force + t_dyn
        # the pendulum's pivot is fixed, so bend_frame "spur" and "world" coincide here
        theta = chord_angle_deg(chord, -up_world(n, dev))
        f_d = bend_pull_detach_force(theta, self._detach_scale, self._reward_cfg.detach)
        u = tension / f_d
        junction_force = tension.unsqueeze(-1) * u_hat - f_bend  # what the stem pulls the junction with
        junction_moment = torch.cross(-chord, -f_bend, dim=-1)
        target_wrench = torch.cat([junction_force, junction_moment], -1)
        down = -up_world(n, dev)
        woody = {self.TARGET_JUNCTION_NAME: target_wrench}
        for j in JUNCTION_NAMES:
            if j == self.TARGET_JUNCTION_NAME:
                continue
            mag = self.coll_rest[j] + self.coll_pull[j] * torch.clamp(t_dyn, min=0.0) + self.coll_twist[j] * bend_mag
            woody[j] = torch.cat([mag.unsqueeze(-1) * down, torch.zeros(n, 3, device=dev)], -1)
        apple_weight = torch.tensor([0.0, 0.0, -0.15 * 9.81], device=dev).expand(n, 3)
        ft_raw = torch.cat([f_plant + apple_weight, t_plant], -1)
        obs = self._build_obs(ft_raw)
        info = {
            "obs_layout": "surrogate_vic_harvest_bend_pull",
            "step_count": self._step_count,
            "invalid_env": self._invalid.clone(),
            "woody_part_force": woody,
            # [D17q] the surrogate's junction moments are already about the joint (no COM offset)
            "woody_part_torque_anchor": {j: w[:, 3:6] for j, w in woody.items() if j != self.TARGET_JUNCTION_NAME},
            "target_junction_force": target_wrench,
            "target_junction_wrench": target_wrench,
            "target_junction_axis": u_hat.clone(),
            "theta_b_deg": theta,
            "stem_tension": tension,
            "detach_force": f_d,
            "detach_scale": self._detach_scale.clone(),
            "detach_index": u,
            "detach_utilization": u,
            "ft_wrist": ft_raw,
        }
        return obs, info

    def plant_geometry(self) -> dict[str, torch.Tensor]:
        geo = super().plant_geometry()
        geo["stem_length_m"] = self.stem_len.reshape(-1, 1).clone()
        geo["apple_radius_m"] = self.apple_r.reshape(-1, 1).clone()
        geo["grasp_roll"] = self.roll.reshape(-1, 1).clone()
        return geo


def up_world(n: int, device: Any) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)


def _rotate_z_to(local: torch.Tensor, pole: torch.Tensor) -> torch.Tensor:
    """Rotate ``(N, 3)`` vectors given about +z so that +z maps to the unit ``pole``."""
    p = pole / torch.linalg.norm(pole)
    z = torch.tensor([0.0, 0.0, 1.0])
    v = torch.cross(z, p, dim=-1)
    c = float(torch.dot(z, p))
    if float(torch.linalg.norm(v)) < 1e-9:
        return local if c > 0 else local * torch.tensor([1.0, -1.0, -1.0])
    vx = torch.tensor([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    rot = torch.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    return local @ rot.T


def _matrix_to_quat_wxyz(m: torch.Tensor) -> torch.Tensor:
    """``(N, 3, 3)`` rotation matrices (columns = frame axes) -> ``(N, 4)`` wxyz quaternions."""
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    w = torch.sqrt(torch.clamp(1.0 + tr, min=1e-12)) / 2.0
    x = torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=0.0)) / 2.0
    y = torch.sqrt(torch.clamp(1.0 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], min=0.0)) / 2.0
    z = torch.sqrt(torch.clamp(1.0 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2], min=0.0)) / 2.0
    x = torch.copysign(x, m[:, 2, 1] - m[:, 1, 2])
    y = torch.copysign(y, m[:, 0, 2] - m[:, 2, 0])
    z = torch.copysign(z, m[:, 1, 0] - m[:, 0, 1])
    q = torch.stack([w, x, y, z], -1)
    return q / torch.linalg.norm(q, dim=-1, keepdim=True)
