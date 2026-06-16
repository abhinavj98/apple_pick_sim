"""Dynamic-arm VIC env for post-grasp pulling in the coupled sim."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp
from gymnasium import spaces

from apple_pick_gym.envs.apple_pick_coupled_env import ApplePickCoupledEnv


class ApplePickVicEnv(ApplePickCoupledEnv):
    """Coupled FR3 env with variable-impedance control (dynamic arm).

    Uses ``Fr3EEImpedanceController`` + ``update_fr3_ee_teleop`` with joint-torque
    VIC (default) or wrench-only VIC. Intended for post-grasp pulling with
    ``fix_to_apple=True`` so stem load feeds back through the lagged harvest path.

    Action/observation contract extends :class:`ApplePickCoupledEnv` (``Discrete(13)``
    keyboard-style commands; woody geometry, per-link forces, apple pose, fresh VBD
    harvest at TCP) with ``ft_wrist``: lagged **plant** wrench last applied to TCP
    ``body_f`` — a proxy for a wrist F/T sensor (world frame, excludes VIC).

    Per-junction branch forces (primary→secondary, secondary→spur, spur→stem,
    stem→apple) are in ``obs["woody_part_force"]``; use :attr:`~ApplePickBaseEnv.junction_names`
    and :meth:`~ApplePickBaseEnv.junction_forces_dict` for named access.
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
        vic_linear_k: float = 4000.0,
        vic_linear_d: float = 80.0,
        vic_angular_k: float = 40.0,
        vic_angular_d: float = 4.0,
        vic_use_joint_torques: bool = True,
    ) -> None:
        from apple_pick_sim.robot import fr3_robot

        self._vic_gains = fr3_robot.ImpedanceGains(
            linear_k=float(vic_linear_k),
            linear_d=float(vic_linear_d),
            angular_k=float(vic_angular_k),
            angular_d=float(vic_angular_d),
        )
        self._vic_use_joint_torques = bool(vic_use_joint_torques)
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            enable_self_collisions=enable_self_collisions,
            fix_to_apple=fix_to_apple,
            fix_to_apple_warmup_substeps=fix_to_apple_warmup_substeps,
            max_woody_parts=max_woody_parts,
            mujoco_solver_kwargs=mujoco_solver_kwargs,
            control_hz=control_hz,
        )
        print(f"vic_linear_k: {vic_linear_k}")
        print(f"vic_linear_d: {vic_linear_d}")
        print(f"vic_angular_k: {vic_angular_k}")
        print(f"vic_angular_d: {vic_angular_d}")
        print(f"vic_use_joint_torques: {vic_use_joint_torques}")
        print(f"max_episode_steps: {max_episode_steps}")
        print(f"enable_self_collisions: {enable_self_collisions}")
        print(f"fix_to_apple: {fix_to_apple}")

    def _create_controller(self):
        from apple_pick_sim.robot import fr3_robot

        return fr3_robot.Fr3EEImpedanceController(
            tcp_body_index=int(self._scene.tcp_body_index),
        )

    def _finalize_scene(self) -> None:
        from apple_pick_sim.coupled_fruiting import vic_joint_torques
        from apple_pick_sim.robot import fr3_robot

        if self._vic_use_joint_torques:
            vic_joint_torques._require_torch()

        self._scene.robot_kinematic_mode = False
        fr3_robot.init_mujoco_actuator_targets_from_model(
            self._scene.robot_model, self._scene.robot_control
        )

        self._scene.vic_use_joint_torques = self._vic_use_joint_torques
        self._controller = self._create_controller()
        self._scene.vic_controller = self._controller
        self._scene.vic_gains = self._vic_gains

        if self._vic_use_joint_torques:
            fr3_robot.configure_vic_joint_torques_arm(
                self._scene.robot_model,
                self._scene.robot_state_0,
                self._scene.robot_control,
                self._scene.mj_solver,
                scene=self._scene,
            )
            self._scene.vic_joint_torques_configured = True
        else:
            fr3_robot.configure_vic_wrench_only_arm(
                self._scene.robot_model,
                self._scene.robot_state_0,
                self._scene.robot_control,
                self._scene.mj_solver,
            )

        tcp = int(self._scene.tcp_body_index)
        self._controller.sync_target_from_state(self._scene.robot_state_0, tcp)
        self._scene.vic_target_tf = self._controller.target_tf
        self._scene.vic_target_twist = fr3_robot.EEVelocity()

    def _update_teleop(self, frame_dt: float, vel) -> None:
        self._scene.update_fr3_ee_teleop(frame_dt, self._controller, velocity=vel)

    @staticmethod
    def _observation_space_for(n_woody: int) -> spaces.Dict:
        base = ApplePickCoupledEnv._observation_space_for(n_woody)
        return spaces.Dict(
            {
                **dict(base.spaces),
                "ft_wrist": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                ),
            }
        )

    def _ft_wrist(self) -> np.ndarray:
        """Lagged plant wrench at TCP — wrist F/T sensor proxy (``coupling_forces_cache``)."""
        from apple_pick_sim.coupling_force_debug import read_tcp_wrench

        assert self._scene is not None and self._scene.coupling_forces_cache is not None
        scene = self._scene
        tcp = int(scene.tcp_body_index)
        w = read_tcp_wrench(scene.coupling_forces_cache, tcp).astype(np.float64)
        # Joint-torque VIC keeps plant load on body_f only; wrench-only VIC adds to cache.
        if (
            not scene.robot_kinematic_mode
            and not getattr(scene, "vic_use_joint_torques", True)
            and getattr(scene, "vic_controller", None) is not None
        ):
            bq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp]
            bqd = scene.robot_state_0.body_qd.numpy().reshape(-1, 6)[tcp]
            vic = scene.vic_controller.compute_applied_wrench(
                target_tf=scene.vic_target_tf,
                target_twist=scene.vic_target_twist,
                tcp_body_q=bq,
                tcp_body_qd=bqd,
                gains=getattr(scene, "vic_gains", None),
            )
            w = w - vic
        return w.astype(np.float32)

    def _make_obs(self) -> dict[str, Any]:
        obs = super()._make_obs()
        obs["ft_wrist"] = self._ft_wrist()
        return obs

    @staticmethod
    def _primary_base_world_pos(scene: Any) -> np.ndarray | None:
        """World position of the pinned primary-chain root (not in woody-part obs)."""
        cable = getattr(scene, "cable", None)
        if cable is None or not cable.primary_bodies:
            return None
        bq = cable.state_0.body_q.numpy().reshape(-1, 7)
        return np.asarray(bq[int(cable.primary_bodies[0]), :3], dtype=np.float32)

    @staticmethod
    def log_woody_part_markers(
        viewer,
        obs: dict[str, Any],
        *,
        scene: Any | None = None,
        radius: float = 0.025,
    ) -> None:
        """Draw debug spheres at woody fixed-joint anchors from an observation dict.

        Uses ``woody_part_start_pos`` (flat ``N*3`` world positions [m]) for inter-rod
        and rod–apple fixed joints. The pinned primary-chain root is not a fixed joint
        in ``fruiting_fixed_joints``; pass ``scene`` to also draw it (blue marker).
        Intended for manual visual verification in examples.
        """
        log_points = getattr(viewer, "log_points", None)
        if log_points is None:
            return

        device = getattr(viewer, "device", None)

        def _emit(name: str, positions: np.ndarray, rgb: tuple[float, float, float]) -> None:
            positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
            if positions.size == 0:
                log_points(name, None)
                return
            n = len(positions)
            points = wp.array(
                [wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in positions],
                dtype=wp.vec3,
                device=device,
            )
            log_points(
                name,
                points,
                radii=wp.full(n, float(radius), dtype=wp.float32, device=device),
                colors=wp.full(n, wp.vec3(*rgb), dtype=wp.vec3, device=device),
            )

        start = np.asarray(obs["woody_part_start_pos"], dtype=np.float32).reshape(-1, 3)
        _emit("/gym/woody_parts", start, (0.95, 0.15, 0.15))

        base = ApplePickVicEnv._primary_base_world_pos(scene) if scene is not None else None
        if base is not None:
            _emit("/gym/primary_base", base.reshape(1, 3), (0.15, 0.45, 0.95))
        else:
            log_points("/gym/primary_base", None)

    @staticmethod
    def _junction_labels_for_obs(
        n_junctions: int,
        *,
        scene: Any | None = None,
        junction_names: list[str] | None = None,
    ) -> list[str]:
        if junction_names is not None:
            return list(junction_names)
        cable = getattr(scene, "cable", None) if scene is not None else None
        fixed = getattr(cable, "fruiting_fixed_joints", None) if cable is not None else None
        if fixed is not None and len(fixed) == n_junctions:
            return [label.removeprefix("joint_") for _, label in fixed]
        return [f"joint_{i}" for i in range(n_junctions)]

    @staticmethod
    def log_junction_force_arrows(
        viewer,
        obs: dict[str, Any],
        *,
        scene: Any | None = None,
        junction_names: list[str] | None = None,
        scale_per_newton: float = 0.02,
        gain: float = 1.0,
        min_length: float = 0.0,
        max_length: float = 0.0,
        force_threshold: float = 1e-6,
    ) -> None:
        """Draw linear force at each branch junction as a world-frame arrow.

        Uses the midpoint of ``woody_part_start_pos`` / ``woody_part_end_pos`` as the
        arrow origin and ``woody_part_force[i*6:i*6+3]`` as direction. Paths are
        ``/gym/junction_forces/<label>`` (e.g. ``stem_apple``). Labels come from
        ``junction_names``, else ``scene.cable.fruiting_fixed_joints``, else
        ``joint_0``, ``joint_1``, …
        """
        from apple_pick_sim.tcp_force_viz import log_tcp_force_arrow

        force = obs.get("woody_part_force")
        start = obs.get("woody_part_start_pos")
        end = obs.get("woody_part_end_pos")
        if force is None or start is None or end is None:
            return

        forces = np.asarray(force, dtype=np.float32).reshape(-1, 6)
        starts = np.asarray(start, dtype=np.float32).reshape(-1, 3)
        ends = np.asarray(end, dtype=np.float32).reshape(-1, 3)
        n = len(forces)
        if n == 0 or len(starts) != n or len(ends) != n:
            return

        labels = ApplePickVicEnv._junction_labels_for_obs(
            n, scene=scene, junction_names=junction_names
        )
        origins = 0.5 * (starts + ends)
        for i, label in enumerate(labels):
            log_tcp_force_arrow(
                viewer,
                f"/gym/junction_forces/{label}",
                origins[i],
                forces[i],
                scale_per_newton=scale_per_newton,
                gain=gain,
                min_length=min_length,
                max_length=max_length,
                force_threshold=force_threshold,
            )

    @staticmethod
    def log_ft_wrist_arrow(
        viewer,
        obs: dict[str, Any],
        *,
        scene: Any | None = None,
        scale_per_newton: float = 0.02,
        gain: float = 1.0,
        min_length: float = 0.0,
        max_length: float = 0.0,
        force_threshold: float = 1e-6,
    ) -> None:
        """Draw ``ft_wrist`` linear force as an arrow at the TCP (world frame)."""
        from apple_pick_sim.tcp_force_viz import log_tcp_force_arrow, tcp_origin_world

        wrench = obs.get("ft_wrist")
        if wrench is None or scene is None:
            return
        origin = tcp_origin_world(scene)
        log_tcp_force_arrow(
            viewer,
            "/gym/ft_wrist",
            origin,
            np.asarray(wrench, dtype=np.float32),
            scale_per_newton=scale_per_newton,
            gain=gain,
            min_length=min_length,
            max_length=max_length,
            force_threshold=force_threshold,
        )
