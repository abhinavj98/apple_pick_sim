"""Dynamic-arm VIC env for post-grasp pulling in the coupled sim."""

from __future__ import annotations

from typing import Any

from apple_pick_gym.envs.apple_pick_coupled_env import ApplePickCoupledEnv


class ApplePickVicEnv(ApplePickCoupledEnv):
    """Coupled FR3 env with variable-impedance control (dynamic arm).

    Uses ``Fr3EEImpedanceController`` + ``update_fr3_ee_teleop`` with joint-torque
    VIC (default) or wrench-only VIC. Intended for post-grasp pulling with
    ``fix_to_apple=True`` so stem load feeds back through the lagged harvest path.

    Action/observation contract matches :class:`ApplePickCoupledEnv` (``Discrete(13)``
    keyboard-style commands; woody geometry, per-link forces, apple pose, TCP wrench).
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
        vic_linear_k: float = 800.0,
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
