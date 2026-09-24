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
    integrate_delta_pose,
    leash_target_pose,
    split_harvest_action,
)
from apple_pick_gym.batched_envs.harvest_detach import detach_index, envelope_thresholds
from apple_pick_gym.batched_envs.harvest_episode import EpisodeConfig, FreezeMask, SuccessStreakTracker
from apple_pick_gym.batched_envs.harvest_obs import _PRIVILEGED_FIELDS
from apple_pick_gym.batched_envs.harvest_outcome import evaluate_harvest_step
from apple_pick_gym.batched_envs.harvest_privileged import PLANT_GEOMETRY_FIELDS
from apple_pick_gym.batched_envs.harvest_reward import HarvestRewardConfig, compute_progress_reward
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
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self._max_episode_steps = int(max_episode_steps)
        self._dt = 1.0 / float(control_hz)
        self._substeps = int(substeps)
        self._action_bounds = action_bounds or HarvestActionBounds()
        self._reward_cfg = reward_config or HarvestRewardConfig()
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
        self._q = torch.zeros(n, 4, device=dev)
        self._q[:, 0] = 1.0  # wxyz; rest orientation is the identity for every env
        self._w = torch.zeros(n, 3, device=dev)
        self._target = torch.cat([self._x, self._q], -1)
        self._last_action = torch.zeros(n, _ACTION_DIM, device=dev)
        self._kp = torch.zeros(n, 6, device=dev)
        self._kd = torch.zeros(n, 6, device=dev)
        self._resample_arm()
        self._thresholds = envelope_thresholds(self._reward_cfg.detach, n, self._rng, device=dev)  # [D7]
        self._ft_sensor.reset()
        self._tracker.reset()
        self._freeze.reset()
        self._freeze.update(self._invalid)
        obs, info = self._observe()
        self._collateral_baseline = {
            j: torch.linalg.norm(w[:, :3], dim=-1).clone() for j, w in info["woody_part_force"].items() if j != self.TARGET_JUNCTION_NAME
        }
        info["collateral_baseline_norm"] = self._collateral_baseline
        self._peak_collateral = None  # [D13] running peak collateral, reset per episode
        self._progress_prev = compute_progress_reward(
            info["target_junction_wrench"],
            self._reward_cfg,
            stem_axis=info["target_junction_axis"],
            thresholds=info["detach_thresholds"],
        )
        return obs, info

    def step(self, action: torch.Tensor):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.shape != (self.num_envs, _ACTION_DIM):
            raise ValueError(f"actions shape must be ({self.num_envs}, {_ACTION_DIM}), got {tuple(action.shape)}")
        action = self._freeze.apply_to_delta_action(action, self._last_action, delta_dims=6)
        split = split_harvest_action(action, self._action_bounds)
        b = self._action_bounds
        target = integrate_delta_pose(self._target, split.delta)
        if b.max_target_pos_offset_m is not None or b.max_target_rot_offset_rad is not None:
            target = leash_target_pose(
                target,
                torch.cat([self._x, self._q], -1),
                max_pos_offset_m=b.max_target_pos_offset_m,
                max_rot_offset_rad=b.max_target_rot_offset_rad,
            )
        self._target = target
        self._kp = torch.cat([split.linear_k, split.angular_k], -1)
        self._kd = 2.0 * split.zeta * torch.sqrt(self._kp)
        self._last_action = action.clone()
        self._integrate()
        self._step_count += 1

        obs, info = self._observe()
        info["collateral_baseline_norm"] = self._collateral_baseline
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
        for _ in range(self._substeps):
            f_plant, t_plant, _, _ = self._plant_wrench_on_tcp()
            f_spring = self._kp[:, :3] * (self._target[:, :3] - self._x)
            rot_err = _quat_to_rotvec_wxyz(_quat_mul_wxyz(self._target[:, 3:], _quat_conj_wxyz(self._q)))
            t_spring = self._kp[:, 3:] * rot_err
            # Damping is integrated implicitly: D = 2 zeta sqrt(K) is sized for unit inertia, and
            # explicit Euler diverges once h*D/I > 2 (zeta = 2 at I = 0.02 kg m^2 is h*D/I ~ 3).
            self._v = (self._v + h * (f_spring + f_plant) / m) / (1.0 + h * d_lin / m)
            self._w = (self._w + h * (t_spring + t_plant) / inertia) / (1.0 + h * d_ang / inertia)
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
        rot = _quat_to_rotvec_wxyz(self._q)
        pose6 = torch.cat([self._x - self.tcp_rest, rot], -1)
        obs = {
            "tcp_pos": self._x.clone(),
            "tcp_quat": self._q[:, [1, 2, 3, 0]].clone(),  # xyzw, as in the real env's obs
            "tcp_velocity": torch.cat([self._v, self._w], -1),
            "ft_wrist": self._ft_sensor.step(ft_raw),
            "robot_joint_q": self._q_rest + pose6 @ self._joint_map,
            "last_action": self._last_action.clone(),
            "step_frac": torch.full((n, 1), self._step_count / self._max_episode_steps, device=dev),
        }
        info = {
            "obs_layout": "surrogate_vic_harvest",
            "step_count": self._step_count,
            "invalid_env": self._invalid.clone(),
            "woody_part_force": woody,
            "target_junction_force": target_wrench,
            "target_junction_wrench": target_wrench,
            "target_junction_axis": self.weld.clone(),  # the surrogate's stem axis is the grasp axis
            "detach_thresholds": self._thresholds,
            "detach_index": detach_index(target_wrench, self._reward_cfg.detach, stem_axis=self.weld, thresholds=self._thresholds),
            "ft_wrist": ft_raw,
        }
        return obs, info

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


def _rotvec_to_quat_wxyz(r: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(r, dim=-1, keepdim=True)
    half = 0.5 * angle
    k = torch.where(angle > 1e-9, torch.sin(half) / angle.clamp_min(1e-9), torch.full_like(angle, 0.5))
    return torch.cat([torch.cos(half), r * k], -1)
