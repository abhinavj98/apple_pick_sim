"""Variable-impedance wrench at the FR3 TCP (world frame)."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import warp as wp

from apple_pick_sim.robot.fr3_robot.controllers.keyboard import (
    EEVelocity,
    _KeyViewer,
    integrate_tcp_target,
    read_keyboard_ee_velocity,
)


@dataclasses.dataclass(frozen=True)
class ImpedanceGains:
    """Isotropic stiffness and damping for linear and angular TCP impedance."""

    linear_k: float = 800.0
    linear_d: float = 80.0
    angular_k: float = 40.0
    angular_d: float = 4.0


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _orientation_error_axis_angle(q_des: np.ndarray, q_act: np.ndarray) -> np.ndarray:
    """Return axis-angle orientation error vector (world frame) for ``q_des * q_act^{-1}``."""
    qd = _normalize_quat_wxyz(q_des)
    qa = _normalize_quat_wxyz(q_act)
    q_err = _quat_mul_wxyz(qd, _quat_conjugate_wxyz(qa))
    q_err = _normalize_quat_wxyz(q_err)
    q_wp = wp.quat(float(q_err[0]), float(q_err[1]), float(q_err[2]), float(q_err[3]))
    axis, angle = wp.quat_to_axis_angle(q_wp)
    ax = wp.vec3(axis)
    mag = float(wp.length(ax))
    if mag < 1e-12 or abs(angle) < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return np.array(
        [float(ax[0] / mag * angle), float(ax[1] / mag * angle), float(ax[2] / mag * angle)],
        dtype=np.float64,
    )


class Fr3EEImpedanceController:
    """Post-grasp TCP impedance teleop: target pose integration + applied wrench."""

    def __init__(
        self,
        *,
        tcp_body_index: int | None = None,
        linear_speed: float = 0.2,
        angular_speed: float = 1.0,
    ) -> None:
        self.tcp_body_index = tcp_body_index
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.target_tf: wp.transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

    def sync_target_from_state(self, state: Any, tcp_body_index: int | None = None) -> None:
        """Set the integrated TCP target to the current FK pose of ``tcp``."""
        idx = tcp_body_index if tcp_body_index is not None else self.tcp_body_index
        if idx is None:
            raise ValueError("tcp_body_index required for sync_target_from_state")
        bq = state.body_q.numpy().reshape(-1, 7)[idx]
        self.target_tf = wp.transform(
            wp.vec3(float(bq[0]), float(bq[1]), float(bq[2])),
            wp.quat(float(bq[3]), float(bq[4]), float(bq[5]), float(bq[6])),
        )

    def advance_target(
        self,
        velocity: EEVelocity | None,
        dt: float,
        *,
        viewer: _KeyViewer | None = None,
        poll_events: bool = True,
        lock_angular: bool = False,
    ) -> EEVelocity:
        """Integrate ``target_tf`` by one teleop frame using a constant world-frame twist."""
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
        return velocity

    def run_tcp_target_teleop_frame(
        self,
        dt: float,
        state: Any,
        *,
        velocity: EEVelocity | None = None,
        viewer: _KeyViewer | None = None,
        poll_events: bool = True,
        lock_angular: bool = False,
        tcp_body_index: int | None = None,
    ) -> EEVelocity:
        """Re-anchor to FK and integrate TCP target only (no IK; for VIC teleop)."""
        self.sync_target_from_state(state, tcp_body_index)
        return self.advance_target(
            velocity,
            dt,
            viewer=viewer,
            poll_events=poll_events,
            lock_angular=lock_angular,
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
        """Per-frame coupled VIC teleop: TCP target integration only (PD gains zeroed at setup)."""
        del control, mj_solver  # unused; VIC configure zeros joint PD
        return self.run_tcp_target_teleop_frame(
            dt,
            state,
            velocity=velocity,
            viewer=viewer,
            poll_events=True,
        )

    def compute_applied_wrench(
        self,
        *,
        target_tf: wp.transform,
        target_twist: EEVelocity,
        tcp_body_q: np.ndarray,
        tcp_body_qd: np.ndarray,
        gains: ImpedanceGains | None = None,
    ) -> np.ndarray:
        """Return spatial wrench ``[fx,fy,fz,tx,ty,tz]`` in world frame at TCP COM."""
        g = gains if gains is not None else ImpedanceGains()
        q7 = np.asarray(tcp_body_q, dtype=np.float64).reshape(7)
        qd = np.asarray(tcp_body_qd, dtype=np.float64).reshape(6)

        p_des = wp.transform_get_translation(target_tf)
        q_des = wp.transform_get_rotation(target_tf)
        p_act = q7[:3]
        q_act = q7[3:7]

        e_p = np.array(
            [float(p_des[0]) - p_act[0], float(p_des[1]) - p_act[1], float(p_des[2]) - p_act[2]],
            dtype=np.float64,
        )
        e_r = _orientation_error_axis_angle(
            np.array([float(q_des[0]), float(q_des[1]), float(q_des[2]), float(q_des[3])]),
            q_act,
        )

        v_des = np.array(target_twist.linear, dtype=np.float64)
        w_des = np.array(target_twist.angular, dtype=np.float64)
        v_act = qd[:3]
        w_act = qd[3:6]

        force = g.linear_k * e_p + g.linear_d * (v_des - v_act)
        torque = g.angular_k * e_r + g.angular_d * (w_des - w_act)
        return np.concatenate([force, torque]).astype(np.float64)
