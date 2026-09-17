"""Explicit apple weight and inertia for stem-harvest TCP wrench transfer.

When ``fix_to_apple`` prescribes the apple (``inv_mass == 0``), VBD does not integrate
gravity on that link. Stem joint gather still reflects stem deformation and kinematic
coupling, but may under-represent fruit load at hold. This module adds env-on-robot
apple weight ``m_apple * g`` and torque ``(p_apple - p_tcp) × F``, and (when enabled)
inertial reaction ``F = -m a_com``, ``τ = r × F - I α`` before gain/caps on the
harvested wrench.

With ``fix_to_apple``, apple pose follows the robot TCP via
``gripper_proxy_offset_in_apple_frame`` (same convention as
``mirror_robot_tcp_to_proxy_and_apple_kernel``). Prefer that offset for the lever arm
so TCP moment stays correct when the wrist rotates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp


def _gravity_components(gravity: wp.vec3 | tuple[float, float, float]) -> tuple[float, float, float]:
    """Normalize ``wp.vec3`` or a 3-tuple to ``(gx, gy, gz)`` floats.

    Internal helper so public functions accept either Warp vectors (from scene
    ``gravity_vec``) or plain tuples in tests. Used by
    :func:`apple_support_force_world` and downstream stem-harvest explicit-weight
    paths in :mod:`proxy_coupling` and :mod:`tests.test_explicit_apple_load`.
    """
    if isinstance(gravity, wp.vec3):
        return float(gravity[0]), float(gravity[1]), float(gravity[2])
    gx, gy, gz = gravity
    return float(gx), float(gy), float(gz)


def apple_support_force_world(
    mass_kg: float,
    gravity: wp.vec3 | tuple[float, float, float],
) -> np.ndarray:
    """Apple weight on the robot TCP [N] (world frame, env-on-robot).

    With ``gravity = (0, 0, -9.81)``, returns ``(0, 0, -m g)`` (downward payload).
    """
    m = float(mass_kg)
    if m <= 0.0:
        return np.zeros(3, dtype=np.float64)
    gx, gy, gz = _gravity_components(gravity)
    return np.array([m * gx, m * gy, m * gz], dtype=np.float64)


def body_com_position_world(body_q: Any, body_index: int) -> np.ndarray:
    """COM position from a ``body_q`` buffer (7-vector transforms per body)."""
    arr = body_q.numpy() if hasattr(body_q, "numpy") else np.asarray(body_q)
    pos = arr.reshape(-1, 7)[int(body_index), :3]
    return np.asarray(pos, dtype=np.float64)


def body_orientation_world(body_q: Any, body_index: int) -> wp.quat:
    """Body orientation (w, x, y, z) from a ``body_q`` buffer."""
    arr = body_q.numpy() if hasattr(body_q, "numpy") else np.asarray(body_q)
    q = arr.reshape(-1, 7)[int(body_index), 3:7]
    return wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def apple_com_from_tcp_grasp_offset(
    tcp_pos_world: np.ndarray,
    tcp_orientation_world: wp.quat,
    grasp_offset_in_apple_frame: tuple | np.ndarray,
) -> np.ndarray:
    """Apple COM in world frame from TCP pose and apple-frame grasp offset.

    Matches ``mirror_robot_tcp_to_proxy_and_apple_kernel`` /
    ``_co_teleport_apples_from_proxies``:

    ``X_apple = X_tcp * X_offset^{-1}``
    """
    p_tcp = np.asarray(tcp_pos_world, dtype=np.float64).reshape(3)
    go = grasp_offset_in_apple_frame
    if len(go) == 7:
        offset_tf = wp.transform(
            wp.vec3(float(go[0]), float(go[1]), float(go[2])),
            wp.quat(float(go[3]), float(go[4]), float(go[5]), float(go[6])),
        )
    else:
        offset_tf = wp.transform(
            wp.vec3(float(go[0]), float(go[1]), float(go[2])), wp.quat_identity()
        )
    tcp_tf = wp.transform(
        wp.vec3(float(p_tcp[0]), float(p_tcp[1]), float(p_tcp[2])),
        tcp_orientation_world,
    )
    apple_tf = wp.transform_multiply(tcp_tf, wp.transform_inverse(offset_tf))
    apple_pos = wp.transform_get_translation(apple_tf)
    return np.array([apple_pos[0], apple_pos[1], apple_pos[2]], dtype=np.float64)


def apple_explicit_wrench_about_tcp(
    mass_kg: float,
    gravity: wp.vec3 | tuple[float, float, float],
    tcp_pos_world: np.ndarray,
    apple_pos_world: np.ndarray | None = None,
    *,
    grasp_offset_in_apple_frame: tuple | np.ndarray | None = None,
    tcp_orientation_world: wp.quat | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit apple-weight wrench reported about the TCP origin (world frame).

    Force is apple payload ``m * g`` (same as :func:`apple_support_force_world`).
    Torque is ``(p_apple - p_tcp) × F`` so offset COM weight produces moment at the flange.

    When ``grasp_offset_in_apple_frame`` and ``tcp_orientation_world`` are both set,
    ``p_apple`` is derived from the TCP pose (kinematic ``fix_to_apple`` placement) and
    ``apple_pos_world`` is ignored. Otherwise ``apple_pos_world`` is required.
    """
    f = apple_support_force_world(mass_kg, gravity)
    p_tcp = np.asarray(tcp_pos_world, dtype=np.float64).reshape(3)
    if grasp_offset_in_apple_frame is not None and tcp_orientation_world is not None:
        p_apple = apple_com_from_tcp_grasp_offset(
            p_tcp, tcp_orientation_world, grasp_offset_in_apple_frame
        )
    elif apple_pos_world is not None:
        p_apple = np.asarray(apple_pos_world, dtype=np.float64).reshape(3)
    else:
        raise ValueError(
            "apple_explicit_wrench_about_tcp needs apple_pos_world or "
            "(grasp_offset_in_apple_frame, tcp_orientation_world)"
        )
    r = p_apple - p_tcp
    tau = np.cross(r, f)
    return f, tau


def apple_inertia_kgm2_from_mass_radius(
    mass_kg: float,
    apple_radius_m: float | None,
) -> float:
    """Solid-sphere scalar inertia ``(2/5) m r²`` about apple COM [kg·m²]."""
    m = float(mass_kg)
    if m <= 0.0:
        return 0.0
    r = 0.0 if apple_radius_m is None else float(apple_radius_m)
    if r <= 0.0:
        return 0.0
    return 0.4 * m * r * r


def _spatial_vector_linear_angular(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 6-vector ``[vx,vy,vz, wx,wy,wz]`` into linear and angular parts."""
    arr = np.asarray(row, dtype=np.float64).reshape(6)
    return arr[:3], arr[3:6]


def tcp_twist_finite_difference(
    qd: Any,
    qd_prev: Any,
    tcp_body_index: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(v, ω, a_tcp, α)`` from TCP ``body_qd`` buffers (world frame)."""
    if float(dt) <= 0.0:
        z = np.zeros(3, dtype=np.float64)
        return z.copy(), z.copy(), z.copy(), z.copy()
    qd_arr = qd.numpy() if hasattr(qd, "numpy") else np.asarray(qd)
    qd_prev_arr = qd_prev.numpy() if hasattr(qd_prev, "numpy") else np.asarray(qd_prev)
    v, w = _spatial_vector_linear_angular(qd_arr.reshape(-1, 6)[int(tcp_body_index)])
    v_prev, w_prev = _spatial_vector_linear_angular(
        qd_prev_arr.reshape(-1, 6)[int(tcp_body_index)]
    )
    inv_dt = 1.0 / float(dt)
    return v, w, (v - v_prev) * inv_dt, (w - w_prev) * inv_dt


def apple_com_acceleration_world(
    v_tcp_world: np.ndarray,
    w_tcp_world: np.ndarray,
    a_tcp_world: np.ndarray,
    alpha_world: np.ndarray,
    r_tcp_to_apple_world: np.ndarray,
) -> np.ndarray:
    """Rigid weld: ``a_com = a_tcp + α × r + ω × (ω × r)`` (world frame)."""
    w = np.asarray(w_tcp_world, dtype=np.float64).reshape(3)
    a_tcp = np.asarray(a_tcp_world, dtype=np.float64).reshape(3)
    alpha = np.asarray(alpha_world, dtype=np.float64).reshape(3)
    r = np.asarray(r_tcp_to_apple_world, dtype=np.float64).reshape(3)
    return a_tcp + np.cross(alpha, r) + np.cross(w, np.cross(w, r))


def apple_inertial_reaction_wrench_about_tcp(
    mass_kg: float,
    inertia_kgm2: float,
    a_com_world: np.ndarray,
    alpha_world: np.ndarray,
    r_tcp_to_apple_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Env-on-robot inertial reaction: ``F = -m a_com``, ``τ = r × F - I α``."""
    m = float(mass_kg)
    if m <= 0.0:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    a_com = np.asarray(a_com_world, dtype=np.float64).reshape(3)
    alpha = np.asarray(alpha_world, dtype=np.float64).reshape(3)
    r = np.asarray(r_tcp_to_apple_world, dtype=np.float64).reshape(3)
    f = -m * a_com
    tau = np.cross(r, f) - float(inertia_kgm2) * alpha
    return f, tau


def explicit_apple_wrench_for_stem_harvest(
    *,
    mass_kg: float,
    gravity: wp.vec3 | tuple[float, float, float],
    robot_body_q: Any,
    cable_body_q: Any,
    tcp_body_index: int,
    apple_body_index: int,
    grasp_offset_in_apple_frame: tuple | np.ndarray | None = None,
    robot_body_qd: Any | None = None,
    robot_body_qd_prev: Any | None = None,
    dt: float = 0.0,
    inertia_kgm2: float = 0.0,
    explicit_apple_weight: bool = True,
    explicit_apple_inertia: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apple payload + optional inertial wrench for stem harvest (world, about TCP)."""
    p_tcp = body_com_position_world(robot_body_q, tcp_body_index)
    if grasp_offset_in_apple_frame is not None:
        tcp_rot = body_orientation_world(robot_body_q, tcp_body_index)
        p_apple = apple_com_from_tcp_grasp_offset(
            p_tcp, tcp_rot, grasp_offset_in_apple_frame
        )
    else:
        p_apple = body_com_position_world(cable_body_q, apple_body_index)
    r = p_apple - p_tcp
    f = np.zeros(3, dtype=np.float64)
    tau = np.zeros(3, dtype=np.float64)
    if explicit_apple_weight:
        if grasp_offset_in_apple_frame is not None:
            tcp_rot = body_orientation_world(robot_body_q, tcp_body_index)
            f, tau = apple_explicit_wrench_about_tcp(
                mass_kg,
                gravity,
                p_tcp,
                grasp_offset_in_apple_frame=grasp_offset_in_apple_frame,
                tcp_orientation_world=tcp_rot,
            )
        else:
            f, tau = apple_explicit_wrench_about_tcp(mass_kg, gravity, p_tcp, p_apple)
    if (
        explicit_apple_inertia
        and robot_body_qd is not None
        and robot_body_qd_prev is not None
        and float(mass_kg) > 0.0
        and float(dt) > 0.0
    ):
        _v, w, a_tcp, alpha = tcp_twist_finite_difference(
            robot_body_qd, robot_body_qd_prev, tcp_body_index, dt
        )
        a_com = apple_com_acceleration_world(_v, w, a_tcp, alpha, r)
        f_i, tau_i = apple_inertial_reaction_wrench_about_tcp(
            mass_kg, inertia_kgm2, a_com, alpha, r
        )
        f = f + f_i
        tau = tau + tau_i
    return f, tau


def apple_mass_kg_from_model(model: Any, apple_body_index: int | None) -> float:
    """Read ``model.body_mass[apple]``; return 0 when the index is missing."""
    if apple_body_index is None or int(apple_body_index) < 0:
        return 0.0
    bid = int(apple_body_index)
    masses = model.body_mass.numpy()
    if bid >= masses.shape[0]:
        return 0.0
    return float(masses[bid])
