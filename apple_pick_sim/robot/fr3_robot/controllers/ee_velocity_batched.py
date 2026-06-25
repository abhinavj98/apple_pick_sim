"""Batched multi-env FR3 end-effector velocity IK teleop."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import warp as wp

import newton

from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.robot.fr3_robot.batched_template_ik import BatchedTemplateIK
from apple_pick_sim.robot.fr3_robot.controllers.keyboard import (
    EEVelocity,
    _KeyViewer,
    integrate_tcp_target,
    read_keyboard_ee_velocity,
)
from apple_pick_sim.robot.fr3_robot.placement import (
    IK_TELEOP_POS_TOL_M,
    IK_TELEOP_ROT_TOL_RAD,
    raise_if_ik_teleop_not_converged,
)
from apple_pick_sim.robot.fr3_robot.setup import (
    sync_mujoco_actuator_targets_from_joint_q,
)


class Fr3BatchedEEVelocityController:
    """Per-env TCP velocity IK on a single-world template; scatter to batched ``robot_model``."""

    def __init__(
        self,
        robot_model: newton.Model,
        layout: BatchedEnvLayout,
        ik_robot_model: newton.Model,
        ik_tcp_body_index: int,
        *,
        velocity_for_world: Callable[[int], EEVelocity] | None = None,
        linear_speed: float = 0.2,
        angular_speed: float = 1.0,
        ik_iterations: int = 48,
        joint_limit_weight: float = 10.0,
        ik_pos_tol_m: float = IK_TELEOP_POS_TOL_M,
        ik_rot_tol_rad: float = IK_TELEOP_ROT_TOL_RAD,
        print_ik_teleop_error_each_step: bool = False,
    ) -> None:
        self.robot_model = robot_model
        self.layout = layout
        self.tcp_body_index = int(layout.tcp_body_indices[0])
        self._velocity_for_world = velocity_for_world
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.ik_iterations = ik_iterations
        self.ik_pos_tol_m = ik_pos_tol_m
        self.ik_rot_tol_rad = ik_rot_tol_rad
        self.print_ik_teleop_error_each_step = print_ik_teleop_error_each_step
        self.target_tf = [
            wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
            for _ in range(layout.num_envs)
        ]
        self._ik = BatchedTemplateIK(
            ik_model=ik_robot_model,
            sim_model=robot_model,
            layout=layout,
            tcp_body_index=ik_tcp_body_index,
            joint_limit_weight=joint_limit_weight,
        )
        self.joint_q = self._ik.joint_q

    def _velocity_for(self, world: int, velocity: EEVelocity | None) -> EEVelocity:
        if self._velocity_for_world is not None:
            return self._velocity_for_world(world)
        if velocity is not None:
            return velocity
        return EEVelocity()

    def sync_target_from_state(self, state: Any) -> None:
        for w in range(self.layout.num_envs):
            self.target_tf[w] = self._ik.sim_tcp_pose_from_model(w)
        self._ik.set_target_from_fk(state)

    def seed_ik_from_state(self, state: Any) -> None:
        self._ik.seed_from_state(state)

    def advance_target(
        self,
        dt: float,
        *,
        velocity: EEVelocity | None = None,
        viewer: _KeyViewer | None = None,
        poll_events: bool = True,
        lock_angular: bool = False,
    ) -> EEVelocity:
        if velocity is None and self._velocity_for_world is None:
            velocity = read_keyboard_ee_velocity(
                viewer,
                linear_speed=self.linear_speed,
                angular_speed=self.angular_speed,
                poll_events=poll_events,
            )
        for w in range(self.layout.num_envs):
            v = self._velocity_for(w, velocity)
            if lock_angular:
                v = EEVelocity(linear=v.linear, angular=(0.0, 0.0, 0.0))
            self.target_tf[w] = integrate_tcp_target(
                self.target_tf[w],
                linear_vel=v.linear_vec,
                angular_vel=v.angular_vec,
                dt=dt,
            )
            self._ik.set_target_world(w, self.target_tf[w])
        return velocity if velocity is not None else self._velocity_for(0, None)

    def solve_ik(self, state: Any | None = None) -> None:
        if state is not None:
            self.seed_ik_from_state(state)
        self._ik.step(self.ik_iterations)

    def measure_ik_target_error(self, state: Any) -> tuple[float, float]:
        del state
        return self._ik.max_pose_error(self.target_tf)

    def measure_ik_target_error_per_world(self, state: Any) -> list[tuple[float, float]]:
        del state
        return self._ik.pose_errors_per_world(self.target_tf)

    def command_velocity_for_world(
        self, world: int, *, fallback: EEVelocity | None = None
    ) -> EEVelocity:
        """Return the teleop twist that would be applied to ``world`` this frame."""
        return self._velocity_for(world, fallback)

    def tcp_world_pose(self, world: int) -> wp.transform:
        """World-frame TCP pose from batched ``joint_q`` FK."""
        return self._ik.sim_tcp_pose_from_model(world)

    def run_ik_teleop_frame(
        self,
        dt: float,
        state: Any,
        *,
        velocity: EEVelocity | None = None,
        viewer: _KeyViewer | None = None,
        poll_events: bool = True,
        lock_angular: bool = False,
        after_advance: Any | None = None,
    ) -> EEVelocity:
        self.sync_target_from_state(state)
        velocity = self.advance_target(
            dt,
            velocity=velocity,
            viewer=viewer,
            poll_events=poll_events,
            lock_angular=lock_angular,
        )
        if after_advance is not None:
            after_advance()
        self.solve_ik(state)
        pos_err: float | None = None
        rot_err: float | None = None
        if self.print_ik_teleop_error_each_step or not velocity.is_zero():
            pos_err, rot_err = self.measure_ik_target_error(state)
        if self.print_ik_teleop_error_each_step and pos_err is not None and rot_err is not None:
            self._print_ik_teleop_error(pos_err, rot_err, velocity=velocity)
        if not velocity.is_zero() and pos_err is not None and rot_err is not None:
            self._raise_if_ik_not_converged(pos_err, rot_err)
        return velocity

    def _print_ik_teleop_error(
        self,
        pos_err_m: float,
        rot_err_rad: float,
        *,
        velocity: EEVelocity,
    ) -> None:
        pos_ok = pos_err_m < self.ik_pos_tol_m
        rot_ok = rot_err_rad < self.ik_rot_tol_rad
        status = "OK" if pos_ok and rot_ok else "FAIL"
        cmd = "hold"
        if not velocity.is_zero():
            lin = velocity.linear
            ang = velocity.angular
            cmd = (
                f"v=({lin[0]:+.3f},{lin[1]:+.3f},{lin[2]:+.3f}) "
                f"w=({ang[0]:+.3f},{ang[1]:+.3f},{ang[2]:+.3f})"
            )
        print(
            f"IK teleop [{status}] {cmd} "
            f"pos_err={pos_err_m * 1000.0:.2f} mm (tol {self.ik_pos_tol_m * 1000.0:.2f}) "
            f"rot_err={rot_err_rad:.4f} rad (tol {self.ik_rot_tol_rad:.4f})",
            flush=True,
        )

    def _raise_if_ik_not_converged(self, pos_err_m: float, rot_err_rad: float) -> None:
        target_pos = wp.transform_get_translation(self.target_tf[0])
        raise_if_ik_teleop_not_converged(
            pos_err_m,
            rot_err_rad,
            pos_tol_m=self.ik_pos_tol_m,
            rot_tol_rad=self.ik_rot_tol_rad,
            target_pos=(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])),
        )

    def apply_to_model_and_state(self, state: Any) -> None:
        self._ik.scatter_to_model(state, eval_fk=True)

    def apply_ik_to_mujoco_control(
        self,
        state: Any,
        control: Any,
        frame_dt: float,
        *,
        command_velocity: EEVelocity | None = None,
    ) -> None:
        full_jq = self._ik.scatter_to_model(state, eval_fk=False)
        sync_mujoco_actuator_targets_from_joint_q(
            self.robot_model,
            state,
            control,
            full_jq,
            frame_dt=frame_dt,
            command_velocity=command_velocity,
        )

    def run_coupled_teleop_frame(
        self,
        state: Any,
        control: Any,
        mj_solver: Any,
        dt: float,
        *,
        viewer: _KeyViewer | None = None,
        velocity: EEVelocity | None = None,
    ) -> EEVelocity:
        del mj_solver
        velocity = self.run_ik_teleop_frame(
            dt,
            state,
            velocity=velocity,
            viewer=viewer,
            poll_events=True,
        )
        self.apply_ik_to_mujoco_control(
            state,
            control,
            frame_dt=dt,
            command_velocity=velocity,
        )
        return velocity
