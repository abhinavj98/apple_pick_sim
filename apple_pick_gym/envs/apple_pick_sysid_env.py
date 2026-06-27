"""Gymnasium env for system-ID excitation (§2.1 quasi-static stiffness mapping)."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from apple_pick_gym.envs.apple_pick_vic_env import ApplePickVicEnv
from apple_pick_sim.system_id.excitation_state import ExcitationContext


_EXCITATION_TYPE_TO_INT: dict[str, int] = {
    "quasi_static": 0,
    "translational_chirp": 1,
    "torsional": 2,
}


class ApplePickSysIdEnv(ApplePickVicEnv):
    """VIC env with continuous EE velocity actions and excitation metadata obs.

    Extends :class:`ApplePickVicEnv` for post-grasp quasi-static stepping.
    ``tcp_pos`` is the **actual** TCP body position (not the VIC target) so
    stiffness estimates from ``ft_wrist`` displacement are unbiased by compliance.
    """

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        max_episode_steps: int = 240,
        enable_self_collisions: bool = False,
        fix_to_apple: bool = True,
        fix_to_apple_warmup_substeps: int = 1800,
        max_woody_parts: int = 64,
        mujoco_solver_kwargs: dict[str, Any] | None = None,
        control_hz: float = 60.0,
        vic_linear_k: float = 2000.0,
        vic_linear_d: float = 80.0,
        vic_angular_k: float = 40.0,
        vic_angular_d: float = 4.0,
        vic_use_joint_torques: bool = True,
        stem_force_cap_n: float | None = 100.0,
        stem_torque_cap_nm: float | None = 100.0,
        max_linear_vel: float = 1.0,
        max_angular_vel: float = 1.0,
        robot_facing_weld: bool = True,
        n_weld_hemisphere_samples: int = 10,
        device: str | None = None,
    ) -> None:
        self._stem_force_cap_n = (
            None if stem_force_cap_n is None else float(stem_force_cap_n)
        )
        self._stem_torque_cap_nm = (
            None if stem_torque_cap_nm is None else float(stem_torque_cap_nm)
        )
        self._max_linear_vel = float(max_linear_vel)
        self._max_angular_vel = float(max_angular_vel)
        self._robot_facing_weld = bool(robot_facing_weld)
        self._n_weld_hemisphere_samples = int(n_weld_hemisphere_samples)
        self._weld_reset_count = 0
        self._weld_direction_override: tuple[float, float, float] | None = None
        self._weld_reference_pos_override: tuple[float, float, float] | None = None
        self._weld_reference_quat_override: tuple[float, float, float, float] | None = None
        self._last_weld_direction: tuple[float, float, float] | None = None
        self._grasp_robot_body_q: np.ndarray | None = None
        self._grasp_robot_body_qd: np.ndarray | None = None
        self._grasp_robot_joint_q: np.ndarray | None = None
        self._grasp_robot_joint_qd: np.ndarray | None = None
        self._grasp_cable_body_q: np.ndarray | None = None
        self._grasp_cable_body_qd: np.ndarray | None = None
        self._grasp_cable_state_1_body_q: np.ndarray | None = None
        self._grasp_cable_state_1_body_qd: np.ndarray | None = None
        self._grasp_target_tf: Any | None = None
        self._excitation_context = ExcitationContext(
            type="quasi_static",
            f_inst=0.0,
            direction=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            enable_self_collisions=enable_self_collisions,
            fix_to_apple=fix_to_apple,
            fix_to_apple_warmup_substeps=fix_to_apple_warmup_substeps,
            max_woody_parts=max_woody_parts,
            mujoco_solver_kwargs=mujoco_solver_kwargs,
            control_hz=control_hz,
            vic_linear_k=vic_linear_k,
            vic_linear_d=vic_linear_d,
            vic_angular_k=vic_angular_k,
            vic_angular_d=vic_angular_d,
            vic_use_joint_torques=vic_use_joint_torques,
            device=device,
        )

    def set_excitation_context(self, ctx: ExcitationContext) -> None:
        self._excitation_context = ctx

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        options = options or {}
        override = options.get("weld_direction")
        if override is not None:
            arr = np.asarray(override, dtype=np.float64).reshape(3)
            norm = float(np.linalg.norm(arr))
            if norm < 1e-12:
                raise ValueError("weld_direction must be non-zero")
            self._weld_direction_override = (
                float(arr[0] / norm),
                float(arr[1] / norm),
                float(arr[2] / norm),
            )
        else:
            self._weld_direction_override = None
        ref_pos = options.get("weld_reference_pos")
        self._weld_reference_pos_override = (
            None
            if ref_pos is None
            else tuple(float(x) for x in np.asarray(ref_pos, dtype=np.float64).reshape(3))
        )
        ref_quat = options.get("weld_reference_quat")
        self._weld_reference_quat_override = (
            None
            if ref_quat is None
            else tuple(float(x) for x in np.asarray(ref_quat, dtype=np.float64).reshape(4))
        )

        obs, info = super().reset(seed=seed, options=options)
        self.snapshot_grasp_pose()
        if self._last_weld_direction is not None:
            info["weld_direction"] = np.asarray(self._last_weld_direction, dtype=np.float32)
        return obs, info

    def snapshot_grasp_pose(self) -> None:
        """Save initial post-reset state for later :meth:`restore_grasp_pose`."""
        if self._scene is None or self._controller is None:
            raise RuntimeError("Environment must be reset() before snapshot_grasp_pose().")
        scene = self._scene
        self._grasp_robot_body_q = scene.robot_state_0.body_q.numpy().copy()
        self._grasp_robot_body_qd = scene.robot_state_0.body_qd.numpy().copy()
        self._grasp_robot_joint_q = scene.robot_state_0.joint_q.numpy().copy()
        self._grasp_robot_joint_qd = scene.robot_state_0.joint_qd.numpy().copy()
        self._grasp_cable_body_q = scene.cable.state_0.body_q.numpy().copy()
        self._grasp_cable_body_qd = scene.cable.state_0.body_qd.numpy().copy()
        self._grasp_cable_state_1_body_q = scene.cable.state_1.body_q.numpy().copy()
        self._grasp_cable_state_1_body_qd = scene.cable.state_1.body_qd.numpy().copy()
        self._grasp_target_tf = self._controller.target_tf

    def restore_grasp_pose(self) -> None:
        """Teleport state back to the snapshotted grasp pose."""
        if self._scene is None or self._controller is None:
            raise RuntimeError("Environment must be reset() before restore_grasp_pose().")
        if self._grasp_robot_body_q is None:
            raise RuntimeError("Call reset() or snapshot_grasp_pose() before restore_grasp_pose().")

        from apple_pick_sim.coupled_fruiting.proxy_coupling import sync_solver_body_q_prev_from_state
        from apple_pick_sim.coupled_fruiting.scene import init_robot_mujoco_step_buffers
        from apple_pick_sim.robot import fr3_robot

        scene = self._scene
        scene.robot_state_0.body_q.assign(self._grasp_robot_body_q)
        scene.robot_state_0.body_qd.assign(self._grasp_robot_body_qd)
        scene.robot_state_0.joint_q.assign(self._grasp_robot_joint_q)
        scene.robot_state_0.joint_qd.assign(self._grasp_robot_joint_qd)
        scene.robot_model.joint_q.assign(self._grasp_robot_joint_q)
        scene.robot_model.joint_qd.assign(self._grasp_robot_joint_qd)
        init_robot_mujoco_step_buffers(scene)
        fr3_robot.hold_mujoco_actuator_targets_at_state(
            scene.robot_model, scene.robot_state_0, scene.robot_control
        )
        if getattr(scene, "vic_jt_default_dof_pos", None) is not None:
            default_q = np.asarray(self._grasp_robot_joint_q, dtype=np.float32).reshape(-1).copy()
            if default_q.shape[0] > 6:
                default_q[6] = 0.0
            scene.vic_jt_default_dof_pos.assign(default_q)

        cable = scene.cable
        cable.state_0.body_q.assign(self._grasp_cable_body_q)
        cable.state_0.body_qd.assign(self._grasp_cable_body_qd)
        if self._grasp_cable_state_1_body_q is not None:
            cable.state_1.body_q.assign(self._grasp_cable_state_1_body_q)
        else:
            cable.state_1.body_q.assign(self._grasp_cable_body_q)
        if self._grasp_cable_state_1_body_qd is not None:
            cable.state_1.body_qd.assign(self._grasp_cable_state_1_body_qd)
        else:
            cable.state_1.body_qd.assign(self._grasp_cable_body_qd)
        sync_solver_body_q_prev_from_state(cable, cable.state_0.body_q)

        self._controller.target_tf = self._grasp_target_tf
        scene.vic_target_tf = self._grasp_target_tf
        scene.vic_target_twist = fr3_robot.EEVelocity()
        if scene.proxy_forces is not None:
            scene.proxy_forces.zero_()
        if scene.coupling_forces_cache is not None:
            scene.coupling_forces_cache.zero_()

    def _make_gripper_proxy_config(self, *, fix_to_apple: bool):
        import dataclasses

        import apple_pick_sim.fruiting_system as fs

        cfg = super()._make_gripper_proxy_config(fix_to_apple=fix_to_apple)
        if fix_to_apple and self._robot_facing_weld:
            return dataclasses.replace(
                cfg,
                fix_to_apple=True,
                robot_facing_weld=True,
                weld_direction=self._pending_weld_direction,
                weld_reference_pos=self._pending_weld_reference_pos,
                weld_reference_quat=self._pending_weld_reference_quat,
            )
        return cfg

    def _coupled_build_kwargs(self, *, params: Any | None = None) -> dict[str, Any]:
        kw = super()._coupled_build_kwargs(params=params)
        kw["stem_force_cap_N"] = self._stem_force_cap_n
        kw["stem_torque_cap_Nm"] = self._stem_torque_cap_nm
        if self._cfg.fix_to_apple and self._robot_facing_weld:
            # Weld hemisphere uses the fixture robot base; arm root tracks the proxy for reach.
            from apple_pick_sim.coupled_fruiting.defaults import COUPLED_ROBOT_BASE_POS

            kw["robot_base_pos"] = self._reset_robot_base_pos or COUPLED_ROBOT_BASE_POS
            kw["robot_base_from_proxy"] = True
            kw["ik_bootstrap_iterations"] = 512
        return kw

    def _weld_direction_before_fix_to_apple_build(
        self, probe_scene: Any
    ) -> tuple[float, float, float] | None:
        if not self._robot_facing_weld:
            self._last_weld_direction = None
            self._pending_weld_reference_pos = None
            self._pending_weld_reference_quat = None
            return None

        from apple_pick_sim.system_id import sample_fibonacci_hemisphere, stem_perpendicular_robot_pole
        from apple_pick_sim.coupled_fruiting.defaults import COUPLED_ROBOT_BASE_POS

        cable = probe_scene.cable
        apple_body = cable.apple_body
        if apple_body is None:
            self._last_weld_direction = None
            self._pending_weld_reference_pos = None
            self._pending_weld_reference_quat = None
            return None

        apple_q7 = probe_scene.cable.state_0.body_q.numpy().reshape(-1, 7)[int(apple_body)]
        apple_pos = (
            np.asarray(self._weld_reference_pos_override, dtype=np.float64)
            if self._weld_reference_pos_override is not None
            else apple_q7[:3]
        )
        self._pending_weld_reference_pos = (
            float(apple_pos[0]),
            float(apple_pos[1]),
            float(apple_pos[2]),
        )
        apple_quat = (
            np.asarray(self._weld_reference_quat_override, dtype=np.float64)
            if self._weld_reference_quat_override is not None
            else apple_q7[3:7]
        )
        self._pending_weld_reference_quat = (
            float(apple_quat[0]),
            float(apple_quat[1]),
            float(apple_quat[2]),
            float(apple_quat[3]),
        )
        robot_base_pos = self._reset_robot_base_pos or COUPLED_ROBOT_BASE_POS
        robot_vec = np.asarray(robot_base_pos, dtype=np.float64) - apple_pos

        stem_bodies = cable.stem_bodies
        assert len(stem_bodies) >= 2
        body_q = cable.state_0.body_q.numpy().reshape(-1, 7)
        stem_tip = body_q[int(stem_bodies[-1]), :3]
        stem_base = body_q[int(stem_bodies[-2]), :3]
        physical_stem = stem_tip - stem_base
        physical_stem /= np.linalg.norm(physical_stem)

        pole = stem_perpendicular_robot_pole(physical_stem, robot_vec)

        if self._weld_direction_override is not None:
            weld = self._weld_direction_override
        else:
            directions = sample_fibonacci_hemisphere(
                self._n_weld_hemisphere_samples,
                pole,
            )
            idx = self._weld_reset_count % self._n_weld_hemisphere_samples
            self._weld_reset_count += 1
            picked = directions[idx]
            weld = (float(picked[0]), float(picked[1]), float(picked[2]))

        self._last_weld_direction = weld
        return weld

    def _setup_action_space(self) -> None:
        lin = self._max_linear_vel
        ang = self._max_angular_vel
        self.action_space = spaces.Box(
            low=np.array([-lin, -lin, -lin, -ang, -ang, -ang], dtype=np.float32),
            high=np.array([lin, lin, lin, ang, ang, ang], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = self._observation_space_for(self._cfg.max_woody_parts)

    @staticmethod
    def _observation_space_for(
        n_woody: int, junction_names: list[str] | None = None
    ) -> spaces.Dict:
        base = ApplePickVicEnv._observation_space_for(n_woody, junction_names)
        return spaces.Dict(
            {
                **dict(base.spaces),
                "excitation_type": spaces.Discrete(3),
                "excitation_f_inst": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(), dtype=np.float32
                ),
                "excitation_direction": spaces.Box(
                    low=-1.0, high=1.0, shape=(3,), dtype=np.float32
                ),
                "tcp_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "tcp_quat": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
                ),
                "apple_quat": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
                ),
                "robot_joint_q": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
                "raw_ft_wrist": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                ),
            }
        )

    def _tcp_pos(self) -> np.ndarray:
        assert self._scene is not None
        tcp = int(self._scene.tcp_body_index)
        bq = self._scene.robot_state_0.body_q.numpy().reshape(-1, 7)
        return np.asarray(bq[tcp, :3], dtype=np.float32)

    def _tcp_quat(self) -> np.ndarray:
        assert self._scene is not None
        tcp = int(self._scene.tcp_body_index)
        bq = self._scene.robot_state_0.body_q.numpy().reshape(-1, 7)
        return np.asarray(bq[tcp, 3:7], dtype=np.float32)

    def _robot_joint_q(self) -> np.ndarray:
        assert self._scene is not None
        return np.asarray(self._scene.robot_state_0.joint_q.numpy(), dtype=np.float32).reshape(-1)

    def _raw_ft_wrist(self) -> np.ndarray:
        """Uncapped stem-harvest TCP wrench for diagnostics; ``ft_wrist`` is applied/capped."""
        assert self._scene is not None
        scene = self._scene
        cable = scene.cable
        if scene.stem_apple_joint_index is None:
            return np.zeros(6, dtype=np.float32)

        from apple_pick_sim.coupled_fruiting.explicit_load import (
            apple_com_from_tcp_grasp_offset,
            apple_explicit_wrench_about_tcp,
            body_com_position_world,
            body_orientation_world,
        )
        from apple_pick_sim.vbd_fixed_joint_wrenches import fixed_joint_wrenches_child_com_vbd

        _frame_dt, _substeps_per_step, sub_dt = self._timing_constants()
        records = fixed_joint_wrenches_child_com_vbd(
            cable.model,
            cable.solver,
            body_q=cable.state_0.body_q,
            body_q_prev=cable.state_1.body_q,
            dt=sub_dt,
            joint_pairs=[(int(scene.stem_apple_joint_index), "stem_apple")],
        )
        if not records:
            return np.zeros(6, dtype=np.float32)

        rec = records[0]
        f_stem_at_com = np.asarray(rec.force_world, dtype=np.float64)
        tau_total_tcp = np.asarray(rec.torque_at_child_com_world, dtype=np.float64)
        f_total_tcp = f_stem_at_com.copy()

        apple_body = cable.apple_body
        if apple_body is not None:
            tcp = int(scene.tcp_body_index)
            p_tcp = body_com_position_world(scene.robot_state_0.body_q, tcp)
            if cable.gripper_proxy_offset_in_apple_frame is not None:
                tcp_rot = body_orientation_world(scene.robot_state_0.body_q, tcp)
                p_apple = apple_com_from_tcp_grasp_offset(
                    p_tcp,
                    tcp_rot,
                    cable.gripper_proxy_offset_in_apple_frame,
                )
            else:
                p_apple = body_com_position_world(cable.state_0.body_q, int(apple_body))

            tau_total_tcp = tau_total_tcp + np.cross(p_apple - p_tcp, f_stem_at_com)
            if scene.stem_harvest_explicit_apple_weight and scene.apple_mass_kg > 0.0:
                f_apple, tau_apple = apple_explicit_wrench_about_tcp(
                    float(scene.apple_mass_kg),
                    scene.gravity_vec,
                    p_tcp,
                    apple_pos_world=p_apple,
                )
                f_total_tcp = f_total_tcp + f_apple
                tau_total_tcp = tau_total_tcp + tau_apple

        gain = float(scene.stem_coupling_gain)
        return np.concatenate([f_total_tcp * gain, tau_total_tcp * gain]).astype(np.float32)

    def _excitation_obs(self) -> dict[str, Any]:
        ctx = self._excitation_context
        type_int = _EXCITATION_TYPE_TO_INT.get(ctx.type)
        if type_int is None:
            raise ValueError(f"Unknown excitation type {ctx.type!r}")
        return {
            "excitation_type": int(type_int),
            "excitation_f_inst": np.float32(ctx.f_inst),
            "excitation_direction": np.asarray(ctx.direction, dtype=np.float32),
        }

    def _make_info(self) -> dict[str, Any]:
        info = super()._make_info()
        info["rod_radii"] = self._rod_radii()
        base = self._primary_base_world_pos(self._scene)
        if base is not None:
            info["fruiting_base_pos"] = np.asarray(base, dtype=np.float32)
        if self._reset_robot_base_pos is not None:
            info["robot_base_pos"] = np.asarray(self._reset_robot_base_pos, dtype=np.float32)
        if self._pending_weld_reference_pos is not None:
            info["weld_reference_pos"] = np.asarray(
                self._pending_weld_reference_pos, dtype=np.float32
            )
        if self._pending_weld_reference_quat is not None:
            info["weld_reference_quat"] = np.asarray(
                self._pending_weld_reference_quat, dtype=np.float32
            )
        if self._last_weld_direction is not None:
            info["weld_direction"] = np.asarray(self._last_weld_direction, dtype=np.float32)
        return info

    def _make_obs(self) -> dict[str, Any]:
        obs = super()._make_obs()
        obs.update(self._excitation_obs())
        obs["tcp_pos"] = self._tcp_pos()
        obs["tcp_quat"] = self._tcp_quat()
        obs["apple_quat"] = self._apple_quat()
        obs["robot_joint_q"] = self._robot_joint_q()
        obs["raw_ft_wrist"] = self._raw_ft_wrist()
        return obs

    def _action_to_command(self, action):
        from apple_pick_sim.robot import fr3_robot

        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.shape != (6,):
            raise ValueError(f"Expected action shape (6,), got {arr.shape}")
        lin = np.clip(arr[:3], -self._max_linear_vel, self._max_linear_vel)
        ang = np.clip(arr[3:], -self._max_angular_vel, self._max_angular_vel)
        return fr3_robot.EEVelocity(
            linear=(float(lin[0]), float(lin[1]), float(lin[2])),
            angular=(float(ang[0]), float(ang[1]), float(ang[2])),
        )

    @staticmethod
    def log_movement_direction_arrow(
        viewer,
        obs: dict[str, Any],
        *,
        scene: Any | None = None,
        linear_velocity: tuple[float, float, float] | np.ndarray | None = None,
        length_m: float = 0.4,
        velocity_threshold: float = 1e-6,
    ) -> None:
        """Draw commanded push direction at the TCP (bright cyan arrow).

        Uses non-zero ``linear_velocity`` when provided. When velocity is zero
        (hold phases), falls back to ``excitation_direction`` in ``obs`` so the
        quasi-static push axis stays visible.
        """
        from apple_pick_sim.tcp_force_viz import log_direction_arrow, tcp_origin_world

        name = "/gym/movement_direction"
        if scene is None:
            log_direction_arrow(viewer, name, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), length_m=length_m)
            return

        direction: np.ndarray | None = None
        if linear_velocity is not None:
            vel = np.asarray(linear_velocity, dtype=np.float64).reshape(3)
            if float(np.linalg.norm(vel)) >= velocity_threshold:
                direction = vel

        if direction is None:
            excitation = obs.get("excitation_direction")
            if excitation is None:
                log_direction_arrow(
                    viewer, name, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), length_m=length_m
                )
                return
            direction = np.asarray(excitation, dtype=np.float64).reshape(3)

        if float(np.linalg.norm(direction)) < velocity_threshold:
            log_direction_arrow(viewer, name, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), length_m=length_m)
            return

        origin = tcp_origin_world(scene)
        log_direction_arrow(viewer, name, origin, direction, length_m=length_m)
