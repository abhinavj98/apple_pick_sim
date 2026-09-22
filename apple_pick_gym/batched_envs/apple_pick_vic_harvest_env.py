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
  ``fruiting_system_ranges_rl_harvest_real_g05_m1.json`` (real sys-ID data) and the
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
from apple_pick_gym.batched_envs.harvest_episode import (
    EpisodeConfig,
    FreezeMask,
    SuccessStreakTracker,
    check_safety_violation,
    compute_terminal_reward,
)
from apple_pick_gym.batched_envs.harvest_reward import (
    HarvestRewardConfig,
    compute_dense_reward_terms,
    weight_dense_reward_terms,
)
from apple_pick_gym.batched_envs.sensor_realism import FtSensorConfig, FtSensorModel
from apple_pick_gym.batched_envs.support_joint_dr import sample_support_joint_dr

_ACTION_DIM = 13
_VIC_POSE_ACTION_DIM = 19


@dataclasses.dataclass(frozen=True)
class HoldSettleConfig:
    """Robot-stepped settle run once after build, before the episode snapshot is stored.

    The arm holds its current TCP pose with a fixed impedance while the plant relaxes, so
    the stored snapshot is a force-balanced coupled equilibrium (arm sag = load / linear_k).
    """

    enabled: bool = True
    linear_k: float = 100.0
    linear_d: float = 15.0
    angular_k: float = 10.0
    angular_d: float = 3.0
    max_frames: int = 240
    check_every: int = 10
    force_tol_n: float = 0.5
    speed_tol_m_s: float = 1e-3
    # Envs whose settled TCP sits farther than this from the hold target are flagged
    # invalid (failed IK grasp: the apple is welded to a TCP centimetres away and the
    # weld drags the plant with hundreds of newtons). Healthy envs settle within a few mm.
    max_rest_pos_err_m: float = 0.02
    # ...or whose settled wrist force exceeds this (a failed grasp can settle within
    # the position tolerance while the weld still loads the wrist with >100 N; healthy
    # envs rest at a few newtons).
    max_rest_wrist_force_n: float = 20.0


_N_ARM_DOF = 7

_RL_HARVEST_RANGES_FIXTURE = (
    Path(__file__).resolve().parent.parent.parent
    / "apple_pick_sim"
    / "fixtures"
    / "fruiting_system_ranges_rl_harvest_real_g05_m1.json"
)
# Real sys-ID data (make_rl_harvest_real_fixture.py): CMA materials mu +- 1 sigma, real
# structure geometry mu +- 0.5 sigma, stem angles uniform over the real range, and the robot
# base 0.2 m toward the plant so every default grasp is IK-reachable. The older
# fruiting_system_ranges_rl_harvest_variance.json put CMA moduli (identified on thin real
# rods) onto 2-4x thicker rods, which made the plant far too stiff and unstable.


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
        max_episode_steps: int = 500,
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
        reward_config: HarvestRewardConfig | None = None,
        episode_config: EpisodeConfig | None = None,
        hold_settle: HoldSettleConfig | None = None,
        support_dr_sample: Any | None = None,
        arm_build_dr_scales: dict[str, Any] | None = None,
    ) -> None:
        """``support_dr_sample`` / ``arm_build_dr_scales`` (with ``per_env_params`` and
        ``per_env_grippers``) rebuild exact worlds from a screened world set -- see
        :func:`apple_pick_gym.batched_envs.world_set.world_specs_to_env_kwargs`."""
        from apple_pick_sim.robot.fr3_robot.arm_domain_randomization import (
            ArmDomainRandomizationRanges,
        )

        self._action_bounds = action_bounds or HarvestActionBounds()
        self._arm_dr_ranges = arm_dr_ranges or ArmDomainRandomizationRanges()
        self._ft_sensor_config = ft_sensor_config or FtSensorConfig()
        self._target_junction_name = target_junction_name or self.TARGET_JUNCTION_NAME
        self._reward_cfg = reward_config or HarvestRewardConfig()
        self._episode_cfg = episode_config or EpisodeConfig()
        self._hold_cfg = hold_settle or HoldSettleConfig()
        self._hold_target_pose: torch.Tensor | None = None
        self._invalid_env_mask: torch.Tensor | None = None
        self._dr_rng = np.random.default_rng(dr_seed)
        self._last_full_obs: dict[str, Any] | None = None
        self._last_arm_dr_sample = None
        self._arm_build_dr_scales = arm_build_dr_scales
        self._arm_build_dr_sample = None
        self._pending_terminated: torch.Tensor | None = None

        if sim_config is None:
            from apple_pick_sim.coupled_fruiting import BatchedHeterogeneousCoupledSimConfig
            from apple_pick_gym.batched_envs.real_batched_replay_build import (
                _REAL_OSC_KD_NULL,
                _REAL_OSC_KP_NULL,
                _REAL_OSC_SEP_ORI,
            )
            from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
                ControllerConfig,
            )

            sim_config = BatchedHeterogeneousCoupledSimConfig.gym_defaults(num_envs=int(num_envs))
            # Dynamic apple welded to the proxy, weld-reaction TCP harvest, and the real
            # collection OSC (rotation without Lambda, real kd_null) -- same as the sys-ID
            # real-replay build (real_replay_sim_config, dynamic_apple=True), so K_ang
            # means the same thing here as on the rig.
            sim_config = dataclasses.replace(
                sim_config,
                controller=ControllerConfig(
                    mode="vic_pose",
                    action_dim=_VIC_POSE_ACTION_DIM,
                    kp_null=_REAL_OSC_KP_NULL,
                    kd_null=_REAL_OSC_KD_NULL,
                    sep_ori=_REAL_OSC_SEP_ORI,
                ),
                robot=dataclasses.replace(
                    sim_config.robot,
                    gripper=dataclasses.replace(sim_config.robot.gripper, dynamic_apple=True),
                ),
                fruiting_system=dataclasses.replace(
                    sim_config.fruiting_system, tcp_harvest_source="weld"
                ),
            )
        if ranges_path is None:
            ranges_path = _RL_HARVEST_RANGES_FIXTURE

        # Support-joint DR must be in the config BEFORE the build so the pre- and
        # post-grasp settles and the episode snapshot all see the randomized
        # kp/roll_kp/zeta. Applying it after build (and after the snapshot) is
        # silently reverted by restore_episode_snapshot() (it restores
        # joint_penalty_k) and never gets a settle.
        self._last_support_dr_sample = None
        sim_config = self._with_support_dr(
            sim_config,
            ranges_path=ranges_path,
            num_envs=int(num_envs),
            sample=support_dr_sample,
        )

        if per_env_grippers is None:
            per_env_grippers = self._sample_default_grasps(
                num_envs=int(num_envs),
                pole=grasp_hemisphere_pole,
                max_polar_angle_rad=grasp_max_polar_angle_rad,
                seed=dr_seed,
            )
        self._per_env_grippers = list(per_env_grippers)

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
        self._success_tracker = SuccessStreakTracker(num_envs=self.num_envs, device=self.device)
        self._freeze_mask = FreezeMask(num_envs=self.num_envs, device=self.device)

        self._apply_build_time_dr()
        if self._hold_cfg.enabled:
            self._hold_settle()
        self._invalid_env_mask = self._detect_invalid_envs()

    def _detect_invalid_envs(self) -> torch.Tensor:
        """Flag envs whose stored rest state is a failed IK grasp.

        Criteria: rest TCP far from the hold target, or rest wrist force too high.

        These envs start the episode with the weld dragging the plant (wrist and
        junction forces in the hundreds of newtons), so they are frozen with zero
        reward from every reset and reported via ``info["invalid_env"]``.
        """
        from apple_pick_gym.batched_envs.apple_pick_batched_base_env import ApplePickBatchedBaseEnv

        cfg = self._hold_cfg
        full = ApplePickBatchedBaseEnv._gather_obs(self)
        tcp = wp.to_torch(self._sim.obs_bufs.tcp_pose).to(device=self.device, dtype=torch.float32)
        wrist = torch.linalg.norm(full["ft_wrist"][:, :3], dim=-1)
        if self._hold_target_pose is None:
            err = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        else:
            err = torch.linalg.norm(tcp[:, :3] - self._hold_target_pose[:, :3], dim=-1)
        invalid = (err > float(cfg.max_rest_pos_err_m)) | (wrist > float(cfg.max_rest_wrist_force_n))
        if bool(invalid.any()):
            import warnings

            idx = torch.nonzero(invalid).flatten().tolist()
            warnings.warn(
                f"{len(idx)}/{self.num_envs} envs flagged invalid (rest TCP error "
                f"{[round(float(err[i]), 3) for i in idx]} m, rest wrist "
                f"{[round(float(wrist[i]), 1) for i in idx]} N; limits "
                f"{cfg.max_rest_pos_err_m} m / {cfg.max_rest_wrist_force_n} N); "
                f"frozen with zero reward: {idx}",
                UserWarning,
                stacklevel=3,
            )
        return invalid

    def _hold_settle(self) -> None:
        """Step arm + plant with the arm holding its pose, then re-store the snapshot.

        The post-grasp settle in the build is cable-only (the arm is not stepped), so the
        stored state is not a coupled equilibrium. Here the VIC target is the TCP pose at
        the start; the arm settles to ``target - load / linear_k`` and the plant to its
        balanced state. ``reset()`` then restores that snapshot and starts the episode
        target at the *stored target* (not the sagged TCP), keeping the state balanced.
        """
        from apple_pick_gym.batched_envs.apple_pick_batched_base_env import ApplePickBatchedBaseEnv

        cfg = self._hold_cfg
        sim = self._sim
        n = self.num_envs
        sim.gather_obs()
        tcp_pose = wp.to_torch(sim.obs_bufs.tcp_pose).to(device=self.device, dtype=torch.float32)
        target = torch.cat([tcp_pose[:, :3], _xyzw_to_wxyz(tcp_pose[:, 3:7])], dim=-1)
        kp = torch.tensor(
            [cfg.linear_k] * 3 + [cfg.angular_k] * 3, dtype=torch.float32, device=self.device
        ).expand(n, 6)
        kd = torch.tensor(
            [cfg.linear_d] * 3 + [cfg.angular_d] * 3, dtype=torch.float32, device=self.device
        ).expand(n, 6)
        sim_cfg = sim.config
        action = sim_cfg.controller.validate_actions(
            torch.cat([target, kp, kd], dim=-1),
            num_envs=n,
            device=str(self.device),
            robot_step_mode=sim_cfg.robot.step_mode,
        )

        prev_force = None
        converged = False
        for frame in range(1, int(cfg.max_frames) + 1):
            sim.step(action)
            if frame % int(cfg.check_every) != 0:
                continue
            full = ApplePickBatchedBaseEnv._gather_obs(self)
            force = torch.linalg.norm(
                full["woody_part_info"][self._target_junction_name]["anchor_force"][:, :3], dim=-1
            )
            speed = torch.linalg.norm(full["tcp_velocity"][:, :3], dim=-1)
            if (
                prev_force is not None
                and float((force - prev_force).abs().max()) < cfg.force_tol_n
                and float(speed.max()) < cfg.speed_tol_m_s
            ):
                converged = True
                break
            prev_force = force
        if not converged:
            import warnings

            warnings.warn(
                f"hold settle not converged after {cfg.max_frames} frames "
                f"(target-junction force {prev_force.tolist() if prev_force is not None else None})",
                UserWarning,
                stacklevel=2,
            )
        self._hold_target_pose = target.clone()
        sim.capture_episode_snapshot()

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
                dynamic_apple=True,
                robot_facing_weld=False,
                weld_direction=tuple(float(x) for x in directions[i]),
            )
            for i in range(int(num_envs))
        ]

    def _with_support_dr(
        self, sim_config: Any, *, ranges_path: Path | str, num_envs: int, sample: Any | None = None
    ) -> Any:
        """Sample (or take the given) per-env support kp/roll_kp/zeta and bake them into ``sim_config``."""
        from apple_pick_sim.fruiting_system import load_ranges
        from apple_pick_sim.fruiting_system.params import parse_sim_build

        if sample is None:
            sim_build = parse_sim_build(load_ranges(Path(ranges_path)))
            if sim_build is None or sim_build.support_dr is None:
                return sim_config
            sample = sample_support_joint_dr(sim_build.support_dr, num_envs=num_envs, rng=self._dr_rng)
        elif len(sample.kp) != num_envs:
            raise ValueError(f"support_dr_sample has {len(sample.kp)} envs, expected {num_envs}")
        self._last_support_dr_sample = sample
        return dataclasses.replace(
            sim_config,
            fruiting_system=dataclasses.replace(
                sim_config.fruiting_system,
                support_kp_per_env=tuple(float(x) for x in sample.kp),
                support_roll_kp_per_env=tuple(float(x) for x in sample.roll_kp),
                support_zeta_per_env=tuple(float(x) for x in sample.zeta),
            ),
        )

    def _apply_build_time_dr(self) -> None:
        """Arm link-mass/EE-payload DR, applied once after build (arm state is not
        part of what the plant snapshot must keep consistent)."""
        from apple_pick_sim.robot.fr3_robot.arm_domain_randomization import (
            apply_ee_payload_dr,
            apply_link_mass_inertia_dr,
            sample_arm_domain_randomization,
        )

        layout = self._sim.scene.layout
        arm_sample = sample_arm_domain_randomization(
            self._arm_dr_ranges, num_envs=self.num_envs, rng=self._dr_rng
        )
        if self._arm_build_dr_scales is not None:
            overrides = {k: np.asarray(v, dtype=np.float32) for k, v in self._arm_build_dr_scales.items()}
            for k, v in overrides.items():
                if v.shape != (self.num_envs,):
                    raise ValueError(f"arm_build_dr_scales[{k!r}] shape {v.shape}, expected ({self.num_envs},)")
            arm_sample = dataclasses.replace(arm_sample, **overrides)
        self._arm_build_dr_sample = arm_sample
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

    def export_world_specs(self, *, prefix: str) -> list[Any]:
        """One :class:`WorldSpec` per env (plant, grasp, build-time DR) for a world set."""
        from apple_pick_gym.batched_envs.world_set import WorldSpec
        from apple_pick_sim.fruiting_system.params import fruiting_params_to_json

        sup, arm = self._last_support_dr_sample, self._arm_build_dr_sample
        if sup is None or arm is None:
            raise RuntimeError("support/arm build-time DR samples are not recorded")
        return [
            WorldSpec(
                world_id=f"{prefix}_e{i}",
                params_json=fruiting_params_to_json(params),
                weld_direction=tuple(float(x) for x in self._per_env_grippers[i].weld_direction),
                support_kp=float(sup.kp[i]),
                support_roll_kp=float(sup.roll_kp[i]),
                support_zeta=float(sup.zeta[i]),
                arm_link_mass_scale=float(arm.link_mass_scale[i]),
                arm_link_inertia_scale=float(arm.link_inertia_scale[i]),
                arm_ee_mass_scale=float(arm.ee_mass_scale[i]),
                arm_ee_inertia_scale=float(arm.ee_inertia_scale[i]),
            )
            for i, params in enumerate(self._sim.per_env_params)
        ]

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
        """Proprioception + F/T only -- no vision-tracked geometry.

        ``apple_pos``/``apple_quat``/``woody_part_start_pos``/``woody_part_end_pos``
        were part of the sys-ID observation contract (v3), which assumes vision
        tracking of the apple and junction geometry. The pick policy is
        deliberately scoped to rely on F/T (``ft_wrist``) and proprioception
        (``tcp_pos``/``tcp_quat``/``tcp_velocity``/``robot_joint_q``) only, so
        these four fields are exposed via ``info`` instead (see
        :meth:`_make_info`) -- available for logging, reward shaping, or a
        future vision-augmented variant, but never fed to the policy.
        """
        inf_box = lambda shape: spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
        return spaces.Dict(
            {
                "tcp_pos": inf_box((3,)),
                "tcp_quat": inf_box((4,)),
                "tcp_velocity": inf_box((6,)),
                "ft_wrist": inf_box((6,)),
                "robot_joint_q": inf_box((7,)),
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

        sensor_ft = self._ft_sensor.step(full["ft_wrist"])

        # Proprioception + F/T only -- apple/junction geometry is NOT fed to
        # the policy (see _harvest_observation_space's docstring); it is
        # still available via info (see _make_info), derived from this same
        # `full` gather cached on self._last_full_obs.
        obs = {
            "tcp_pos": tcp_pose[:, :3],
            "tcp_quat": tcp_pose[:, 3:7],  # native (xyzw) -- matches the established v3 contract
            "tcp_velocity": full["tcp_velocity"],
            "ft_wrist": sensor_ft,
            "robot_joint_q": wp.to_torch(bufs.joint_q).to(device=self.device, dtype=torch.float32),
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
        info["invalid_env"] = (
            self._invalid_env_mask.clone()
            if self._invalid_env_mask is not None
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        woody_part_force = self._woody_part_force()
        info["woody_part_force"] = woody_part_force
        info["target_junction_force"] = woody_part_force[self._target_junction_name]
        # Raw (privileged, un-sensor-filtered) ft_wrist for reward computation --
        # distinct from obs["ft_wrist"], which is what the policy actually
        # observes. Reward is train-time only, so this privilege is legitimate.
        info["ft_wrist"] = self._last_full_obs["ft_wrist"]
        # Vision-tracked geometry (sys-ID's v3 contract), NOT fed to the policy
        # (see _harvest_observation_space) -- kept here for logging, reward
        # shaping, or a future vision-augmented variant.
        full = self._last_full_obs
        info["apple_pos"] = full["apple_pos"]
        info["apple_quat"] = self._apple_quat_tensor()
        info["woody_part_start_pos"] = {
            name: full["woody_part_info"][name]["anchors_pos"][:, :3]
            for name in self._junction_names
        }
        info["woody_part_end_pos"] = {
            name: full["woody_part_info"][name]["anchors_pos"][:, 3:6]
            for name in self._junction_names
        }
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
        # Frozen (already-succeeded) envs hold their target with their last gains
        # rather than taking a fresh action -- see harvest_episode.py's module
        # docstring for why this keeps LSTM hidden-state resets batch-uniform. The
        # pose delta is zeroed: replaying it would keep integrating the target away.
        action = self._freeze_mask.apply_to_delta_action(action, self._last_action, delta_dims=6)

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
        if self._hold_target_pose is not None:
            self._target_pose = self._hold_target_pose.clone()
        else:
            tcp_pose = wp.to_torch(bufs.tcp_pose).to(device=self.device, dtype=torch.float32)
            self._target_pose = torch.cat(
                [tcp_pose[:, :3], _xyzw_to_wxyz(tcp_pose[:, 3:7])], dim=-1
            )

        self._ft_sensor.reset()
        self._resample_joint_dynamics_dr()
        self._success_tracker.reset()
        self._freeze_mask.reset()
        if self._invalid_env_mask is not None:
            self._freeze_mask.update(self._invalid_env_mask)
        self._pending_terminated = None

        obs = self._gather_obs()
        info = self._make_info()
        return obs, info

    def compute_reward(self, obs: dict[str, Any], info: dict[str, Any]) -> torch.Tensor:
        """Dense shaping + terminal bonus/penalty, masked to 0 for already-frozen envs.

        Also updates the success streak and the freeze mask, and caches the
        resulting ``terminated`` for :meth:`compute_terminated` to return --
        the base env's ``step()`` always calls ``compute_reward`` first, so
        this ordering is safe and avoids double-incrementing the streak
        tracker by calling ``.update()`` from both methods.
        """
        raw_terms = compute_dense_reward_terms(
            obs, info, target_junction_name=self._target_junction_name, cfg=self._reward_cfg
        )
        weighted_terms = weight_dense_reward_terms(raw_terms, self._reward_cfg)
        dense = (
            weighted_terms["progress"] + weighted_terms["pullout"] + weighted_terms["collateral"]
        ).unsqueeze(-1)

        force_norm = torch.linalg.norm(info["target_junction_force"][:, :3], dim=-1)
        success_this_step = force_norm >= float(self._reward_cfg.f_threshold_n)
        success_achieved = self._success_tracker.update(success_this_step, self._episode_cfg)

        # Safety caps: the target junction's wrench is uncapped (privileged
        # debug gather, not the stem-harvest transfer), so this is a real
        # check. ft_wrist is already hard-capped by the stem-harvest transfer
        # at the same 40 N / 10 N*m default, so that half is a no-op safety
        # net today, kept for robustness if the caps ever diverge.
        safety_junction = check_safety_violation(info["target_junction_force"], self._episode_cfg)
        safety_wrist = check_safety_violation(info["ft_wrist"], self._episode_cfg)
        safety_violation = safety_junction | safety_wrist

        terminal = compute_terminal_reward(success_achieved, safety_violation, self._reward_cfg)
        reward = self._freeze_mask.apply_to_reward(dense + terminal)

        self._pending_terminated = success_achieved | safety_violation
        self._freeze_mask.update(self._pending_terminated)

        # Debug/logging surface (reward decomposition + termination reasons). Values are
        # this step's; "total" is the returned (freeze-masked) reward.
        info["reward_terms"] = {
            "raw": raw_terms,
            "weighted": weighted_terms,
            "dense": dense.squeeze(-1),
            "terminal": terminal.squeeze(-1),
            "total": reward.reshape(self.num_envs),
        }
        info["episode"] = {
            "success_this_step": success_this_step,
            "success_achieved": success_achieved,
            "success_streak": self._success_tracker.streak.clone(),
            "safety_junction": safety_junction,
            "safety_wrist": safety_wrist,
            "frozen": self._freeze_mask.done_mask.clone(),
        }
        info["target_pose"] = self._target_pose.clone()
        return reward

    def compute_terminated(self, obs: dict[str, Any], info: dict[str, Any]) -> torch.Tensor:
        del obs, info
        if self._pending_terminated is None:
            return torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        return self._pending_terminated.unsqueeze(-1)
