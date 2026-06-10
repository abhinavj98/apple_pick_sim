"""Device-side variable-impedance wrench at the TCP (no ``body_q`` host sync)."""

from __future__ import annotations

from typing import Any

import warp as wp

from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import ImpedanceGains
from apple_pick_sim.robot.fr3_robot.controllers.keyboard import EEVelocity


@wp.func
def _normalize_quat(q: wp.quat) -> wp.quat:
    n = wp.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1.0e-12:
        return wp.quat(0.0, 0.0, 0.0, 1.0)
    inv = 1.0 / n
    return wp.quat(q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv)


@wp.func
def _quat_conj(q: wp.quat) -> wp.quat:
    return wp.quat(-q[0], -q[1], -q[2], q[3])


@wp.func
def _quat_mul(a: wp.quat, b: wp.quat) -> wp.quat:
    ax, ay, az, aw = a[0], a[1], a[2], a[3]
    bx, by, bz, bw = b[0], b[1], b[2], b[3]
    return wp.quat(
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


@wp.func
def _orientation_error_axis_angle(q_des: wp.quat, q_act: wp.quat) -> wp.vec3:
    qd = _normalize_quat(q_des)
    qa = _normalize_quat(q_act)
    q_err = _normalize_quat(_quat_mul(qd, _quat_conj(qa)))
    axis, angle = wp.quat_to_axis_angle(q_err)
    ax = wp.vec3(axis[0], axis[1], axis[2])
    mag = wp.length(ax)
    if mag < 1.0e-12 or wp.abs(angle) < 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0)
    scale = angle / mag
    return wp.vec3(ax[0] * scale, ax[1] * scale, ax[2] * scale)


@wp.func
def compute_vic_spatial_wrench(
    tcp_tf: wp.transform,
    tcp_qd: wp.spatial_vector,
    target_tf: wp.transform,
    v_des: wp.vec3,
    w_des: wp.vec3,
    linear_k: float,
    linear_d: float,
    angular_k: float,
    angular_d: float,
) -> wp.spatial_vector:
    """Impedance wrench in world frame at TCP COM (matches ``Fr3EEImpedanceController``)."""
    p_des = wp.transform_get_translation(target_tf)
    q_des = wp.transform_get_rotation(target_tf)
    p_act = wp.transform_get_translation(tcp_tf)
    q_act = wp.transform_get_rotation(tcp_tf)

    e_p = p_des - p_act
    e_r = _orientation_error_axis_angle(q_des, q_act)

    v_act = wp.spatial_top(tcp_qd)
    w_act = wp.spatial_bottom(tcp_qd)

    force = linear_k * e_p + linear_d * (v_des - v_act)
    torque = angular_k * e_r + angular_d * (w_des - w_act)
    return wp.spatial_vector(force[0], force[1], force[2], torque[0], torque[1], torque[2])


@wp.kernel
def _add_vic_wrench_at_tcp_kernel(
    wrenches: wp.array(dtype=wp.spatial_vector),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    tcp_index: int,
    target_tf: wp.transform,
    v_des: wp.vec3,
    w_des: wp.vec3,
    linear_k: float,
    linear_d: float,
    angular_k: float,
    angular_d: float,
):
    delta = compute_vic_spatial_wrench(
        body_q[tcp_index],
        body_qd[tcp_index],
        target_tf,
        v_des,
        w_des,
        linear_k,
        linear_d,
        angular_k,
        angular_d,
    )
    wrenches[tcp_index] = wrenches[tcp_index] + delta


def launch_apply_vic_to_coupling_cache(
    scene: Any,
    *,
    target_tf: wp.transform,
    target_twist: EEVelocity,
    gains: ImpedanceGains | None = None,
) -> None:
    """Add VIC wrench to ``coupling_forces_cache[tcp]`` using device ``body_q`` / ``body_qd``."""
    if scene.robot_state_0 is None or scene.coupling_forces_cache is None:
        return
    g = gains if gains is not None else ImpedanceGains()
    tcp = int(scene.tcp_body_index)
    dev = scene.coupling_forces_cache.device
    wp.launch(
        _add_vic_wrench_at_tcp_kernel,
        dim=1,
        inputs=[
            scene.coupling_forces_cache,
            scene.robot_state_0.body_q,
            scene.robot_state_0.body_qd,
            tcp,
            target_tf,
            wp.vec3(*target_twist.linear),
            wp.vec3(*target_twist.angular),
            float(g.linear_k),
            float(g.linear_d),
            float(g.angular_k),
            float(g.angular_d),
        ],
        device=dev,
    )


def apply_vic_to_coupling_cache(scene: Any) -> None:
    """Add fresh VIC wrench when ``vic_controller`` and targets are configured."""
    if getattr(scene, "vic_controller", None) is None:
        return
    target_tf = getattr(scene, "vic_target_tf", None)
    target_twist = getattr(scene, "vic_target_twist", None)
    if target_tf is None or target_twist is None:
        return
    launch_apply_vic_to_coupling_cache(
        scene,
        target_tf=target_tf,
        target_twist=target_twist,
        gains=getattr(scene, "vic_gains", None),
    )
