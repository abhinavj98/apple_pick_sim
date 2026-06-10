"""Explicit quasi-static apple weight for stem-harvest TCP wrench transfer.

When ``fix_to_apple`` prescribes the apple (``inv_mass == 0``), VBD does not integrate
gravity on that link. Stem joint gather still reflects stem deformation and kinematic
coupling, but may under-represent fruit weight at hold. This module adds support force
``-m_apple * g`` and torque ``(p_apple - p_tcp) × F`` (world frame, about TCP) before
gain/caps on the harvested wrench.

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
    """Support force on the apple [N] (world frame).

    Weight on the apple is ``mass_kg * gravity``; stem support is the opposite:
    ``-mass_kg * gravity``. With ``gravity = (0, 0, -9.81)``, support is ``(0, 0, +m g)``.
    """
    m = float(mass_kg)
    if m <= 0.0:
        return np.zeros(3, dtype=np.float64)
    gx, gy, gz = _gravity_components(gravity)
    return np.array([-m * gx, -m * gy, -m * gz], dtype=np.float64)


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

    Force is support ``-m * g`` (same as :func:`apple_support_force_world`).
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


def explicit_apple_wrench_for_stem_harvest(
    *,
    mass_kg: float,
    gravity: wp.vec3 | tuple[float, float, float],
    robot_body_q: Any,
    cable_body_q: Any,
    tcp_body_index: int,
    apple_body_index: int,
    grasp_offset_in_apple_frame: tuple | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Support force and TCP moment for stem harvest (world frame, about TCP)."""
    p_tcp = body_com_position_world(robot_body_q, tcp_body_index)
    if grasp_offset_in_apple_frame is not None:
        tcp_rot = body_orientation_world(robot_body_q, tcp_body_index)
        return apple_explicit_wrench_about_tcp(
            mass_kg,
            gravity,
            p_tcp,
            grasp_offset_in_apple_frame=grasp_offset_in_apple_frame,
            tcp_orientation_world=tcp_rot,
        )
    p_apple = body_com_position_world(cable_body_q, apple_body_index)
    return apple_explicit_wrench_about_tcp(mass_kg, gravity, p_tcp, p_apple)


def apple_mass_kg_from_model(model: Any, apple_body_index: int | None) -> float:
    """Read ``model.body_mass[apple]``; return 0 when the index is missing."""
    if apple_body_index is None or int(apple_body_index) < 0:
        return 0.0
    bid = int(apple_body_index)
    masses = model.body_mass.numpy()
    if bid >= masses.shape[0]:
        return 0.0
    return float(masses[bid])
