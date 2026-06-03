"""FR3 end-effector velocity IK teleop controller."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

import newton

from apple_pick_sim.robot.fr3_robot.controllers.keyboard import (
    EEVelocity,
    _KeyViewer,
    poll_viewer_events,
    read_keyboard_ee_velocity,
)
from apple_pick_sim.robot.fr3_robot.controllers.keyboard import integrate_tcp_target
from apple_pick_sim.robot.fr3_robot.placement import (
    IK_TELEOP_POS_TOL_M,
    IK_TELEOP_ROT_TOL_RAD,
    raise_if_ik_teleop_not_converged,
    tcp_ik_target_pose_errors,
)
from apple_pick_sim.robot.fr3_robot.setup import (
    init_mujoco_actuator_targets_from_model,
    sync_mujoco_actuator_targets_from_joint_q,
    sync_mujoco_visual_state,
)

def _quat_to_ik_vec4(q: wp.quat) -> wp.vec4:
    return wp.vec4(q[0], q[1], q[2], q[3])


def _make_tcp_ik_solver(
    robot_model: newton.Model,
    tcp_body_index: int,
    target_tf: wp.transform,
    *,
    joint_limit_weight: float = 10.0,
    lambda_initial: float = 0.1,
):
    """Build position + rotation + joint-limit IK objectives for ``tcp``."""
    import newton.ik as ik

    dev = robot_model.device
    target_pos = wp.transform_get_translation(target_tf)
    target_rot = wp.transform_get_rotation(target_tf)
    pos_obj = ik.IKObjectivePosition(
        link_index=tcp_body_index,
        link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=wp.array([target_pos], dtype=wp.vec3, device=dev),
    )
    rot_obj = ik.IKObjectiveRotation(
        link_index=tcp_body_index,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([_quat_to_ik_vec4(target_rot)], dtype=wp.vec4, device=dev),
    )
    limits = ik.IKObjectiveJointLimit(
        joint_limit_lower=robot_model.joint_limit_lower,
        joint_limit_upper=robot_model.joint_limit_upper,
        weight=joint_limit_weight,
    )
    joint_q = robot_model.joint_q.reshape((1, int(robot_model.joint_coord_count)))
    solver = ik.IKSolver(
        model=robot_model,
        n_problems=1,
        objectives=[pos_obj, rot_obj, limits],
        lambda_initial=lambda_initial,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    return pos_obj, rot_obj, joint_q, solver


class Fr3EEVelocityController:
    """Integrate a TCP velocity command, solve IK, and write ``joint_q`` on the robot model."""

    def __init__(
        self,
        robot_model: newton.Model,
        tcp_body_index: int,
        *,
        linear_speed: float = 0.2,
        angular_speed: float = 1.0,
        ik_iterations: int = 48,
        joint_limit_weight: float = 10.0,
        ik_pos_tol_m: float = IK_TELEOP_POS_TOL_M,
        ik_rot_tol_rad: float = IK_TELEOP_ROT_TOL_RAD,
        print_ik_teleop_error_each_step: bool = False,
    ) -> None:
        self.robot_model = robot_model
        self.tcp_body_index = tcp_body_index
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.ik_iterations = ik_iterations
        self.ik_pos_tol_m = ik_pos_tol_m
        self.ik_rot_tol_rad = ik_rot_tol_rad
        self.print_ik_teleop_error_each_step = print_ik_teleop_error_each_step
        self.target_tf = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        self._pos_obj, self._rot_obj, self.joint_q, self._ik_solver = _make_tcp_ik_solver(
            robot_model,
            tcp_body_index,
            self.target_tf,
            joint_limit_weight=joint_limit_weight,
        )

    def sync_target_from_state(self, state: Any) -> None:
        """Set the integrated TCP target to the current FK pose of ``tcp``."""
        bq = state.body_q.numpy().reshape(-1, 7)[self.tcp_body_index]
        self.target_tf = wp.transform(
            wp.vec3(float(bq[0]), float(bq[1]), float(bq[2])),
            wp.quat(float(bq[3]), float(bq[4]), float(bq[5]), float(bq[6])),
        )
        self._push_target_to_ik()

    def _push_target_to_ik(self) -> None:
        pos = wp.transform_get_translation(self.target_tf)
        q = wp.transform_get_rotation(self.target_tf)
        self._pos_obj.set_target_position(0, pos)
        self._rot_obj.set_target_rotation(0, _quat_to_ik_vec4(q))

    def seed_ik_from_state(self, state: Any) -> None:
        """Copy simulated ``state.joint_q`` into the IK seed before :meth:`solve_ik`."""
        jq = state.joint_q.numpy().reshape(1, int(self.robot_model.joint_coord_count))
        self.joint_q.assign(jq.astype(self.robot_model.joint_q.dtype))

    def advance_target(
        self,
        dt: float,
        *,
        velocity: EEVelocity | None = None,
        viewer: _KeyViewer | None = None,
        poll_events: bool = True,
        lock_angular: bool = False,
    ) -> EEVelocity:
        """Integrate the TCP target twist on the host (safe to call outside a CUDA graph capture)."""
        if velocity is None:
            velocity = read_keyboard_ee_velocity(
                viewer,
                linear_speed=self.linear_speed,
                angular_speed=self.angular_speed,
                poll_events=poll_events,
            )
        if lock_angular:
            velocity = EEVelocity(linear=velocity.linear, angular=(0.0, 0.0, 0.0))
        self.target_tf = integrate_tcp_target(
            self.target_tf,
            linear_vel=velocity.linear_vec,
            angular_vel=velocity.angular_vec,
            dt=dt,
        )
        self._push_target_to_ik()
        return velocity

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
        """Re-anchor to FK, integrate one velocity step, solve IK, and verify tolerance."""
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

    def measure_ik_target_error(self, state: Any) -> tuple[float, float]:
        """Return TCP position [m] and orientation [rad] error vs the integrated IK target."""
        return tcp_ik_target_pose_errors(
            self.robot_model,
            state,
            tcp_body_index=self.tcp_body_index,
            target_tf=self.target_tf,
            joint_q=self.joint_q.numpy().reshape(-1),
        )

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
        """Raise :class:`~apple_pick_sim.robot.fr3_robot.placement.IKTeleopConvergenceError` on miss."""
        target_pos = wp.transform_get_translation(self.target_tf)
        raise_if_ik_teleop_not_converged(
            pos_err_m,
            rot_err_rad,
            pos_tol_m=self.ik_pos_tol_m,
            rot_tol_rad=self.ik_rot_tol_rad,
            target_pos=(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])),
        )

    def solve_ik(self, state: Any | None = None) -> None:
        """Run the IK solver for the current target (may be CUDA-graph captured).

        Pass ``state`` (or call :meth:`seed_ik_from_state` first) so the seed matches the
        simulated arm rather than a stale ``robot_model.joint_q``.
        """
        if state is not None:
            self.seed_ik_from_state(state)
        self._ik_solver.step(self.joint_q, self.joint_q, iterations=self.ik_iterations)

    def step(
        self,
        dt: float,
        *,
        velocity: EEVelocity | None = None,
        viewer: _KeyViewer | None = None,
        poll_events: bool = True,
    ) -> EEVelocity:
        """Advance the TCP target, run IK, and leave the solution in ``joint_q``."""
        velocity = self.advance_target(
            dt, velocity=velocity, viewer=viewer, poll_events=poll_events
        )
        self.solve_ik()
        return velocity

    def apply_to_model_and_state(self, state: Any) -> None:
        """Copy the latest IK ``joint_q`` into the model and ``state``, then refresh FK.

        Kinematic teleop only (e.g. ``example_fr3_keyboard`` without MuJoCo stepping).
        For the coupled stack use :meth:`apply_ik_to_mujoco_control` instead.
        """
        import numpy as np

        jq = self.joint_q.numpy().reshape(-1).astype(self.robot_model.joint_q.dtype)
        jqd = np.zeros(int(self.robot_model.joint_dof_count), dtype=np.float32)
        self.robot_model.joint_q.assign(jq)
        self.robot_model.joint_qd.assign(jqd)
        state.joint_q.assign(jq)
        state.joint_qd.assign(jqd)
        newton.eval_fk(self.robot_model, self.robot_model.joint_q, self.robot_model.joint_qd, state)

    def apply_ik_to_mujoco_control(
        self,
        state: Any,
        control: Any,
        frame_dt: float,
        *,
        command_velocity: EEVelocity | None = None,
    ) -> None:
        """Push the latest IK solution to MuJoCo PD actuators (``joint_target_pos`` / ``vel``)."""
        sync_mujoco_actuator_targets_from_joint_q(
            self.robot_model,
            state,
            control,
            self.joint_q.numpy().reshape(-1),
            frame_dt=frame_dt,
            command_velocity=command_velocity,
        )

