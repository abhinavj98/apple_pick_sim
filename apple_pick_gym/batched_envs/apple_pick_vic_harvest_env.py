"""Post-grasp harvest env: 13-D delta-pose action -> 19-D vic_pose, arm+plant DR,
sensor-realistic obs, privileged info.

Builds on `ApplePickBatchedBaseEnv` with `ControllerConfig(mode="vic_pose",
action_dim=19)` -- the real rig's own pose-PD interface (H2
`docs/handbook-variable-impedance.md` sec 3), not the twist-mode `vic` used by
the sibling ``ApplePickBatchedVicEnv``. The policy commands a bounded delta
around a per-env target pose the env itself integrates (Task 1
``harvest_action.py``); the target is packed into the 19-D action every step.

Domain randomization sources (see the design spec and Task 2/3 for full
rationale):

- **Plant** (spur/stem moduli, damping ratios, primary density, support
  kp/roll_kp/zeta) is baked at build from
  ``fruiting_system_ranges_rl_harvest_variance.json`` and the new
  ``sim_build.support_dr`` block. Fixed for the run, matching the plan's
  "plant DR cannot be re-randomized; baked at build" note.
- **Arm joint dynamics** (armature, friction, damping) resample on every
  ``reset()`` -- cheap, safe, and exactly what Task 2's ``apply_joint_dynamics_dr``
  supports (it always derives from the true nominal, so repeated resampling
  never compounds).
- **Arm link-mass/inertia and EE-payload DR** are applied **once**, right
  after build, not resampled per reset. Task 2's ``apply_link_mass_inertia_dr``/
  ``apply_ee_payload_dr`` scale the model's *current* value, so calling them
  again on an already-scaled value would compound rather than resample --
  correct per-episode resampling for these two axes needs a reset-to-nominal
  entry point that Task 2 does not yet have. Tracked as a follow-up.
- **Grasp direction** is sampled once per env at build (`per_env_grippers`),
  reusing the existing sys-ID Fibonacci-hemisphere sampler for diversity
  rather than inventing new geometry. Diversity comes from N, not resampling
  (grasps are fixed for the run, matching the design decision that per-env
  grasp placement happens once via IK at build time).

**IK convergence gate (measured, Task 6 Step 5).** Per-env grasps are placed
by IK against `IK_TELEOP_POS_TOL_M = 0.005` m tolerance for the target-pose
*orientation frame* used during teleop; the coarser build-time bootstrap
tolerance is 0.05 m, and warnings above it are what this default cone was
tuned against. At the FULL hemisphere (``max_polar_angle_rad=pi/2``,
sys-ID's own default), measured convergence was only **56% (14/32 envs
missed, up to 0.128 m error)** -- consistent with a background investigation
that separately found `IKBootstrapConvergenceWarning`s at N=512 up to
0.158 m. Narrowing the default cone to ``pi/6`` (30 degrees around
straight-down) improved this to **81% (6/32 missed, up to 0.126 m)**.
**The residual ~19% failure is not eliminated by cone angle alone** -- it is
consistent with the separately-documented finding that per-env IK bootstrap
placement is not perfectly reproducible/convex even for a fixed, reachable
target (see the Task 0a report). Rejection-sampling failed grasps at build
time would close this further but needs a build-path change outside this
task's file list; tracked as a follow-up. At N in the hundreds, ~19% lost
envs is a real but survivable cost, not a training blocker.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

try:
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise ImportError(
        "apple_pick_gym requires gymnasium. Install from repo root (uv sync --extra gym)."
    ) from e

from apple_pick_gym.batched_envs.apple_pick_batched_base_env import ApplePickBatchedBaseEnv
from apple_pick_gym.batched_envs.harvest_action import (
    HarvestActionBounds,
    integrate_delta_pose,
    pack_vic_pose_action,
    split_harvest_action,
)
from apple_pick_gym.batched_envs.sensor_realism import FtSensorConfig, FtSensorModel
from apple_pick_gym.batched_envs.support_joint_dr import apply_support_joint_dr, sample_support_joint_dr

_ACTION_DIM = 13
_VIC_POSE_ACTION_DIM = 19
_N_ARM_DOF = 7

_RL_HARVEST_RANGES_FIXTURE = (
    Path(__file__).resolve().parent.parent.parent
    / "apple_pick_sim"
    / "fixtures"
    / "fruiting_system_ranges_rl_harvest_variance.json"
)


def _xyzw_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    return quat_xyzw[:, [3, 0, 1, 2]]


class ApplePickVicHarvestEnv(ApplePickBatchedBaseEnv):
    """Post-grasp harvest env: 13-D delta-pose action, `vic_pose` (19D) controller."""

    TARGET_JUNCTION_NAME = "spur_stem"

    def __init__(
        self,
        *,
        num_envs: int = 1,
        render_mode: str | None = None,
        max_episode_steps: int = 240,
        max_woody_parts: int = 64,
        device: str | None = None,
        sim_config: Any | None = None,
        ranges_path: Path | str | None = None,
        topology_seed: int = 42,
        use_settle_cache: bool = False,
        per_env_params: Any | None = None,
        per_env_grippers: Any | None = None,
        action_bounds: HarvestActionBounds | None = None,
        arm_dr_ranges: Any | None = None,
        ft_sensor_config: FtSensorConfig | None = None,
        target_junction_name: str | None = None,
        dr_seed: int = 0,
        grasp_hemisphere_pole: tuple[float, float, float] = (0.0, 0.0, -1.0),
        grasp_max_polar_angle_rad: float = np.pi / 6.0,
    ) -> None:
        from apple_pick_sim.robot.fr3_robot.arm_domain_randomization import (
            ArmDomainRandomizationRanges,
        )

        self._action_bounds = action_bounds or HarvestActionBounds()
        self._arm_dr_ranges = arm_dr_ranges or ArmDomainRandomizationRanges()
        self._ft_sensor_config = ft_sensor_config or FtSensorConfig()
        self._target_junction_name = target_junction_name or self.TARGET_JUNCTION_NAME
        self._dr_rng = np.random.default_rng(dr_seed)
        self._last_full_obs: dict[str, Any] | None = None
        self._last_arm_dr_sample = None

        if sim_config is None:
            from apple_pick_sim.coupled_fruiting import BatchedHeterogeneousCoupledSimConfig
            from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
                ControllerConfig,
            )

            sim_config = BatchedHeterogeneousCoupledSimConfig.gym_defaults(num_envs=int(num_envs))
            sim_config = dataclasses.replace(
                sim_config,
                controller=ControllerConfig(mode="vic_pose", action_dim=_VIC_POSE_ACTION_DIM),
            )
        if ranges_path is None:
            ranges_path = _RL_HARVEST_RANGES_FIXTURE

        if per_env_grippers is None:
            per_env_grippers = self._sample_default_grasps(
                num_envs=int(num_envs),
                pole=grasp_hemisphere_pole,
                max_polar_angle_rad=grasp_max_polar_angle_rad,
                seed=dr_seed,
            )

        super().__init__(
            num_envs=num_envs,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            max_woody_parts=max_woody_parts,
            device=device,
            sim_config=sim_config,
            ranges_path=ranges_path,
            topology_seed=topology_seed,
            use_settle_cache=use_settle_cache,
            per_env_params=per_env_params,
            per_env_grippers=per_env_grippers,
        )

        if self._target_junction_name not in self._junction_names:
            raise ValueError(
                f"target_junction_name={self._target_junction_name!r} not found in "
                f"junction_names={self._junction_names}"
            )
        self._target_junction_idx = self._junction_names.index(self._target_junction_name)

        b = self._action_bounds
        self.action_space = spaces.Box(
            low=np.array(
                [-b.linear_delta_m] * 3
                + [-b.angular_delta_rad] * 3
                + [b.k_lin_min] * 3
                + [b.k_ang_min] * 3
                + [b.zeta_min],
                dtype=np.float32,
            ),
            high=np.array(
                [b.linear_delta_m] * 3
                + [b.angular_delta_rad] * 3
                + [b.k_lin_max] * 3
                + [b.k_ang_max] * 3
                + [b.zeta_max],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.observation_space = self._harvest_observation_space()

        self._last_action = torch.zeros(
            (self.num_envs, _ACTION_DIM), dtype=torch.float32, device=self.device
        )
        self._target_pose = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)
        self._ft_sensor = FtSensorModel(
            num_envs=self.num_envs, device=self.device, config=self._ft_sensor_config
        )

        self._apply_build_time_dr()

    @staticmethod
    def _sample_default_grasps(
        *, num_envs: int, pole: tuple[float, float, float], max_polar_angle_rad: float, seed: int
    ) -> list[Any]:
        """Per-env grasp approach-direction diversity via the existing sys-ID sampler.

        Diversity comes from N (fixed per env at build), reusing
        ``apple_pick_sim.system_id.fibonacci_hemisphere.sample_fibonacci_hemisphere``
        rather than inventing new grasp geometry. This is a simple default
        (direction only, no roll); a caller wanting more control should pass
        ``per_env_grippers`` explicitly.
        """
        from apple_pick_sim.fruiting_system import GripperProxyConfig, PLACEHOLDER_EE_MASS_KG
        from apple_pick_sim.system_id.fibonacci_hemisphere import sample_fibonacci_hemisphere

        directions = sample_fibonacci_hemisphere(
            int(num_envs), np.asarray(pole), max_polar_angle=float(max_polar_angle_rad)
        )
        return [
            GripperProxyConfig(
                mass=PLACEHOLDER_EE_MASS_KG,
                fix_to_apple=True,
                robot_facing_weld=False,
                weld_direction=tuple(float(x) for x in directions[i]),
            )
            for i in range(int(num_envs))
        ]

    def _apply_build_time_dr(self) -> None:
        """Support-joint plant DR (Task 3) + arm link-mass/EE-payload DR (Task 2), once."""
        from apple_pick_sim.fruiting_system.joint_kd_scaling import support_dowel_length_m
        from apple_pick_sim.fruiting_system.params import parse_sim_build
        from apple_pick_sim.robot.fr3_robot.arm_domain_randomization import (
            apply_ee_payload_dr,
            apply_link_mass_inertia_dr,
            sample_arm_domain_randomization,
        )

        ranges = self._sim.ranges
        sim_build = parse_sim_build(ranges)
        if sim_build is not None and sim_build.support_dr is not None:
            scene = self._sim.scene
            support_sample = sample_support_joint_dr(
                sim_build.support_dr, num_envs=self.num_envs, rng=self._dr_rng
            )
            apply_support_joint_dr(
                scene,
                support_sample,
                num_envs=self.num_envs,
                joints_per_world=scene.layout.joints_per_world,
                per_env_params=self._sim.per_env_params,
            )
            self._last_support_dr_sample = support_sample

        layout = self._sim.scene.layout
        arm_sample = sample_arm_domain_randomization(
            self._arm_dr_ranges, num_envs=self.num_envs, rng=self._dr_rng
        )
        apply_link_mass_inertia_dr(
            self._sim.scene.robot_model,
            self._sim.scene.mj_solver,
            arm_sample,
            robot_bodies_per_world=layout.robot_bodies_per_world,
            num_envs=self.num_envs,
        )
        apply_ee_payload_dr(
            self._sim.scene.robot_model,
            self._sim.scene.mj_solver,
            arm_sample,
            robot_bodies_per_world=layout.robot_bodies_per_world,
            num_envs=self.num_envs,
        )

    def _resample_joint_dynamics_dr(self) -> None:
        """Arm joint dynamics (armature/friction/damping) resample every reset."""
        from apple_pick_sim.robot.fr3_robot.arm_domain_randomization import (
            apply_joint_dynamics_dr,
            sample_arm_domain_randomization,
        )

        sample = sample_arm_domain_randomization(
            self._arm_dr_ranges, num_envs=self.num_envs, rng=self._dr_rng
        )
        layout = self._sim.scene.layout
        apply_joint_dynamics_dr(
            self._sim.scene.robot_model,
            self._sim.scene.mj_solver,
            sample,
            dofs_per_world=layout.joint_dof_count_per_world,
        )
        self._last_arm_dr_sample = sample

    def _harvest_observation_space(self) -> spaces.Dict:
        inf_box = lambda shape: spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
        junction_pos_space = spaces.Dict({name: inf_box((3,)) for name in self._junction_names})
        return spaces.Dict(
            {
                "tcp_pos": inf_box((3,)),
                "tcp_quat": inf_box((4,)),
                "tcp_velocity": inf_box((6,)),
                "ft_wrist": inf_box((6,)),
                "apple_pos": inf_box((3,)),
                "apple_quat": inf_box((4,)),
                "robot_joint_q": inf_box((7,)),
                "woody_part_start_pos": junction_pos_space,
                "woody_part_end_pos": junction_pos_space,
                "last_action": inf_box((_ACTION_DIM,)),
                "step_frac": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

    def _apple_quat_tensor(self) -> torch.Tensor:
        layout = self._sim.layout
        if layout is None:
            raise RuntimeError("batched scene missing layout")
        cable = self._sim.scene.cable
        bq = cable.state_0.body_q.numpy().reshape(-1, 7)
        quats = np.stack(
            [bq[int(layout.apple_body_indices[w]), 3:7] for w in range(self.num_envs)], axis=0
        )
        return torch.as_tensor(quats, dtype=torch.float32, device=self.device)

    def _gather_obs(self) -> dict[str, Any]:
        full = super()._gather_obs()
        bufs = self._sim.obs_bufs
        if bufs is None:
            raise RuntimeError("sim observation buffers not allocated")

        tcp_pose = wp.to_torch(bufs.tcp_pose).to(device=self.device, dtype=torch.float32)
        step_frac = torch.full(
            (self.num_envs, 1),
            float(self._step_count) / float(self._max_episode_steps),
            dtype=torch.float32,
            device=self.device,
        )
        woody_start = {
            name: full["woody_part_info"][name]["anchors_pos"][:, :3] for name in self._junction_names
        }
        woody_end = {
            name: full["woody_part_info"][name]["anchors_pos"][:, 3:6] for name in self._junction_names
        }

        sensor_ft = self._ft_sensor.step(full["ft_wrist"])

        obs = {
            "tcp_pos": tcp_pose[:, :3],
            "tcp_quat": tcp_pose[:, 3:7],  # native (xyzw) -- matches the established v3 contract
            "tcp_velocity": full["tcp_velocity"],
            "ft_wrist": sensor_ft,
            "apple_pos": full["apple_pos"],
            "apple_quat": self._apple_quat_tensor(),
            "robot_joint_q": wp.to_torch(bufs.joint_q).to(device=self.device, dtype=torch.float32),
            "woody_part_start_pos": woody_start,
            "woody_part_end_pos": woody_end,
            "last_action": self._last_action.clone(),
            "step_frac": step_frac,
        }
        self._last_full_obs = full
        return obs

    def _woody_part_force(self) -> dict[str, torch.Tensor]:
        if self._last_full_obs is None:
            raise RuntimeError("call reset() or step() before _woody_part_force()")
        return {
            name: self._last_full_obs["woody_part_info"][name]["anchor_force"]
            for name in self._junction_names
        }

    def _make_info(self) -> dict[str, Any]:
        info = super()._make_info()
        info["obs_layout"] = "batched_vic_harvest"
        woody_part_force = self._woody_part_force()
        info["woody_part_force"] = woody_part_force
        info["target_junction_force"] = woody_part_force[self._target_junction_name]
        return info

    def _actions_tensor(self, action: Any) -> torch.Tensor:
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        action = action.to(device=self.device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0).expand(self.num_envs, _ACTION_DIM)
        if action.shape != (self.num_envs, _ACTION_DIM):
            raise ValueError(
                f"actions shape must be ({self.num_envs}, {_ACTION_DIM}), got {tuple(action.shape)}"
            )
        action = action.contiguous()

        split = split_harvest_action(action, self._action_bounds)
        gains = torch.cat([split.linear_k, split.angular_k], dim=-1)
        self._target_pose = integrate_delta_pose(self._target_pose, split.delta)
        packed = pack_vic_pose_action(
            self._target_pose, split.linear_k, split.angular_k, split.zeta
        )

        cfg = self._sim.config
        validated = cfg.controller.validate_actions(
            packed, num_envs=self.num_envs, device=str(self.device), robot_step_mode=cfg.robot.step_mode
        )
        self._last_action = action.clone()
        return validated

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Restore the whole-batch snapshot, then reset all episode-scoped state
        BEFORE the single gather this returns.

        Deliberately does not call ``super().reset()`` (which restores AND
        gathers in one call): that would force a gather before the sensor
        model and arm DR are reset, then a second, redundant gather after --
        wasteful, and the first gather would read the *previous* episode's
        stale sensor bias/EMA state.
        """
        import gymnasium as gym

        gym.Env.reset(self, seed=seed)
        del options
        self._last_action = torch.zeros(
            (self.num_envs, _ACTION_DIM), dtype=torch.float32, device=self.device
        )
        self._sim.restore_episode_snapshot()
        self._step_count = 0

        # Refresh obs_bufs from the just-restored state BEFORE reading tcp_pose --
        # restore_episode_snapshot() does not itself update obs_bufs, so without
        # this call bufs.tcp_pose would still hold the PREVIOUS episode's stale
        # value. This is a cheap buffer refresh only (no dict built yet), kept
        # separate from the final self._gather_obs() below so the sensor model
        # reset (next) does not have to happen before this read.
        self._sim.gather_obs()
        bufs = self._sim.obs_bufs
        tcp_pose = wp.to_torch(bufs.tcp_pose).to(device=self.device, dtype=torch.float32)
        self._target_pose = torch.cat([tcp_pose[:, :3], _xyzw_to_wxyz(tcp_pose[:, 3:7])], dim=-1)

        self._ft_sensor.reset()
        self._resample_joint_dynamics_dr()

        obs = self._gather_obs()
        info = self._make_info()
        return obs, info

    def compute_reward(self, obs: dict[str, Any], info: dict[str, Any]) -> torch.Tensor:
        del obs, info
        return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)

    def compute_terminated(self, obs: dict[str, Any], info: dict[str, Any]) -> torch.Tensor:
        del obs, info
        return torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
