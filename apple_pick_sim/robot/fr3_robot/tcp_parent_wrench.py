"""World-frame env-on-robot TCP wrench from EE ``body_parent_f``.

Newton ``State.body_parent_f`` is the MuJoCo ``cfrc_int`` analog: parent-on-child
joint wrench in world frame about the child COM. For the FR3 tool body ``/fr3/ee``
that is the flange reaction holding the EE mass. Gym ``ft_wrist`` (plant harvest)
is a different signal; this helper is for arm-only / unloaded diagnostics.

Env-on-robot (match converted real ``ft_wrist`` / H3) is the negation. Transport
from EE COM to the TCP tip:

.. math::

    F_{tcp} = -F_{ee}

    \\tau_{tcp} = -\\bigl(\\tau_{ee} + (p_{ee,com} - p_{tcp}) \\times F_{ee}\\bigr)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp


def ee_com_world_from_body_q(
    body_q: Any,
    body_com_local: Any,
    body_index: int,
) -> np.ndarray:
    """World-frame COM from ``body_q`` origin + local ``body_com`` [m]."""
    bq = body_q.numpy() if hasattr(body_q, "numpy") else np.asarray(body_q)
    com = (
        body_com_local.numpy()
        if hasattr(body_com_local, "numpy")
        else np.asarray(body_com_local)
    )
    row = bq.reshape(-1, 7)[int(body_index)]
    origin = np.asarray(row[:3], dtype=np.float64)
    # Newton / Warp transform quat is (x, y, z, w).
    q = wp.quat(float(row[3]), float(row[4]), float(row[5]), float(row[6]))
    local = np.asarray(com.reshape(-1, 3)[int(body_index)], dtype=np.float64)
    offset = wp.quat_rotate(q, wp.vec3(float(local[0]), float(local[1]), float(local[2])))
    return origin + np.array([offset[0], offset[1], offset[2]], dtype=np.float64)


def env_on_robot_tcp_wrench_from_ee_parent_f(
    parent_f_ee: Any,
    *,
    p_ee_com_world: Any,
    p_tcp_world: Any,
) -> np.ndarray:
    """Transport EE parent-on-child wrench to TCP and negate (env-on-robot).

    Returns shape ``(6,)`` float64: ``[Fx, Fy, Fz, Tx, Ty, Tz]`` about TCP, world
    frame, env-on-robot.
    """
    w = np.asarray(parent_f_ee, dtype=np.float64).reshape(6)
    f = w[:3]
    tau = w[3:]
    p_ee = np.asarray(p_ee_com_world, dtype=np.float64).reshape(3)
    p_tcp = np.asarray(p_tcp_world, dtype=np.float64).reshape(3)
    r = p_ee - p_tcp
    tau_at_tcp = tau + np.cross(r, f)
    out = np.empty(6, dtype=np.float64)
    out[:3] = -f
    out[3:] = -tau_at_tcp
    return out


def read_ee_parent_f_world(state: Any, ee_body_index: int) -> np.ndarray:
    """Copy ``state.body_parent_f[ee]`` as a host ``(6,)`` float64 array."""
    buf = getattr(state, "body_parent_f", None)
    if buf is None:
        raise ValueError(
            "state.body_parent_f is None; build FR3 with request_body_parent_f=True"
        )
    arr = buf.numpy() if hasattr(buf, "numpy") else np.asarray(buf)
    return np.asarray(arr.reshape(-1, 6)[int(ee_body_index)], dtype=np.float64).copy()


def tcp_world_wrench_from_scene(scene: Any) -> np.ndarray:
    """Env-on-robot TCP wrench from a coupled / mujoco_only FR3 scene."""
    from apple_pick_sim.robot.fr3_robot.setup import resolve_ee_body_index

    model = scene.robot_model
    state = scene.robot_state_0
    ee_idx = resolve_ee_body_index(model)
    tcp_idx = int(scene.tcp_body_index)
    parent_f = read_ee_parent_f_world(state, ee_idx)
    p_ee = ee_com_world_from_body_q(state.body_q, model.body_com, ee_idx)
    p_tcp = state.body_q.numpy().reshape(-1, 7)[tcp_idx, :3]
    return env_on_robot_tcp_wrench_from_ee_parent_f(
        parent_f, p_ee_com_world=p_ee, p_tcp_world=p_tcp
    )
