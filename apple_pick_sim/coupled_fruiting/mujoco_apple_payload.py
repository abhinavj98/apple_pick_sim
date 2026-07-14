"""Inertia-only MuJoCo apple payload (mass / COM / I) for welded FR3 builds.

Model A keeps ``gravity = 0``; stem harvest still supplies ``-m · g``. The FIXED
child of the TCP only contributes reflected inertia, with properties taken from
the AVBD apple (mass, radius → solid-sphere ``I``, grasp offset → COM).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import warp as wp

import newton

from apple_pick_sim.coupled_fruiting.explicit_load import apple_mass_kg_from_model
from apple_pick_sim.fruiting_system.params import FruitingSystemParams, analytic_apple_mass_kg

APPLE_PAYLOAD_BODY_LABEL = "apple_payload"
APPLE_PAYLOAD_JOINT_LABEL = "tcp_apple_payload"


def solid_sphere_inertia_diag(mass_kg: float, radius_m: float) -> wp.mat33:
    """Return ``(2/5) m r²`` as a diagonal inertia about the COM [kg·m²]."""
    m = float(mass_kg)
    r = float(radius_m)
    if m <= 0.0 or r <= 0.0:
        return wp.mat33()
    i = 0.4 * m * r * r
    return wp.mat33(i, 0.0, 0.0, 0.0, i, 0.0, 0.0, 0.0, i)


def _offset_transform(offset_7d: tuple | np.ndarray | Sequence[float]) -> wp.transform:
    go = offset_7d
    if len(go) == 7:
        return wp.transform(
            wp.vec3(float(go[0]), float(go[1]), float(go[2])),
            wp.quat(float(go[3]), float(go[4]), float(go[5]), float(go[6])),
        )
    return wp.transform(
        wp.vec3(float(go[0]), float(go[1]), float(go[2])),
        wp.quat_identity(),
    )


def apple_com_in_tcp_frame(offset_7d: tuple | np.ndarray | Sequence[float]) -> np.ndarray:
    """Apple COM in the TCP / dummy-body frame: translation of ``inv(X_offset)``.

    Matches ``X_apple = X_tcp · X_offset^{-1}`` when the dummy body origin coincides
    with the TCP (identity FIXED joint).
    """
    inv = wp.transform_inverse(_offset_transform(offset_7d))
    t = wp.transform_get_translation(inv)
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)


def payload_props_from_params(
    params: FruitingSystemParams,
    offset_7d: tuple | np.ndarray | Sequence[float],
) -> tuple[float, wp.mat33, np.ndarray]:
    """Return ``(mass_kg, inertia, com_in_tcp)`` from fruiting params + grasp offset."""
    m = analytic_apple_mass_kg(params)
    if m is None:
        m = 0.0
    r = 0.0 if params.apple_radius is None else float(params.apple_radius)
    return float(m), solid_sphere_inertia_diag(float(m), r), apple_com_in_tcp_frame(offset_7d)


def resolve_apple_payload_body_index(model: newton.Model | newton.ModelBuilder) -> int | None:
    """Return a payload body index labeled ``apple_payload``, or ``None`` if absent.

    For batched models with one payload per world, returns the **lowest** index
    (world 0). Use :attr:`BatchedEnvLayout.mj_apple_payload_body_indices` for all
    worlds.
    """
    labels = list(model.body_label)
    hits = [i for i, lbl in enumerate(labels) if str(lbl) == APPLE_PAYLOAD_BODY_LABEL]
    if not hits:
        return None
    return int(min(hits))


def append_apple_payload_link(builder: newton.ModelBuilder, tcp_body: int) -> int:
    """Append a mass-only FIXED child of ``tcp_body`` and extend its articulation.

    Returns the payload body index. Caller must later set mass / inertia / COM.
    """
    tcp = int(tcp_body)
    parent_art: int | None = None
    for ji in range(int(builder.joint_count)):
        if int(builder.joint_child[ji]) == tcp:
            parent_art = int(builder.joint_articulation[ji])
            break
    if parent_art is None or parent_art < 0:
        raise ValueError(
            f"TCP body {tcp} has no articulated parent joint; cannot attach apple payload"
        )

    payload = builder.add_link(
        mass=0.0,
        lock_inertia=True,
        label=APPLE_PAYLOAD_BODY_LABEL,
        inertia=wp.mat33(),
        com=wp.vec3(0.0, 0.0, 0.0),
    )
    joint = builder.add_joint_fixed(
        parent=tcp,
        child=payload,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
        label=APPLE_PAYLOAD_JOINT_LABEL,
    )
    builder.articulation_end[parent_art] = int(joint) + 1
    builder.joint_articulation[joint] = parent_art
    return int(payload)


def _set_body_inertial_props(
    model: newton.Model,
    body_index: int,
    *,
    mass_kg: float,
    inertia: wp.mat33,
    com_in_body: np.ndarray,
) -> None:
    idx = int(body_index)
    m = float(mass_kg)
    com = np.asarray(com_in_body, dtype=np.float32).reshape(3)

    mass_np = model.body_mass.numpy().copy()
    inv_mass_np = model.body_inv_mass.numpy().copy()
    inertia_np = model.body_inertia.numpy().copy()
    inv_inertia_np = model.body_inv_inertia.numpy().copy()
    com_np = model.body_com.numpy().copy()

    mass_np[idx] = m
    inv_mass_np[idx] = (1.0 / m) if m > 0.0 else 0.0
    I = np.array(
        [
            [float(inertia[0, 0]), float(inertia[0, 1]), float(inertia[0, 2])],
            [float(inertia[1, 0]), float(inertia[1, 1]), float(inertia[1, 2])],
            [float(inertia[2, 0]), float(inertia[2, 1]), float(inertia[2, 2])],
        ],
        dtype=np.float32,
    )
    inertia_np[idx] = I
    if m > 0.0 and float(I[0, 0]) > 0.0:
        inv_inertia_np[idx] = np.diag(1.0 / np.diag(I)).astype(np.float32)
    else:
        inv_inertia_np[idx] = np.zeros((3, 3), dtype=np.float32)
    com_np[idx] = com

    model.body_mass.assign(mass_np)
    model.body_inv_mass.assign(inv_mass_np)
    model.body_inertia.assign(inertia_np)
    model.body_inv_inertia.assign(inv_inertia_np)
    model.body_com.assign(com_np)


def apply_mujoco_apple_payload_inertias(scene: Any) -> None:
    """Write per-env AVBD apple mass/inertia/COM onto MuJoCo payload bodies and notify."""
    robot_model = getattr(scene, "robot_model", None)
    mj_solver = getattr(scene, "mj_solver", None)
    cable = getattr(scene, "cable", None)
    if robot_model is None or mj_solver is None or cable is None:
        return

    layout = getattr(scene, "layout", None)
    tpl_payload = None
    if layout is not None:
        tpl_payload = getattr(layout, "template_mj_apple_payload_body", None)
    if tpl_payload is None:
        tpl_payload = getattr(scene, "mj_apple_payload_body_index", None)
    if tpl_payload is None:
        tpl_payload = resolve_apple_payload_body_index(robot_model)
    if tpl_payload is None:
        return

    num_envs = 1
    if layout is not None and int(getattr(layout, "num_envs", 1)) > 1:
        num_envs = int(layout.num_envs)
        payload_indices = tuple(layout.mj_apple_payload_body_indices)
        apple_indices = tuple(layout.apple_body_indices)
    else:
        payload_indices = (int(tpl_payload),)
        apple = cable.apple_body
        apple_indices = (int(apple) if apple is not None else -1,)

    per_params: Sequence[Any] | None = getattr(scene, "per_env_params", None)
    per_offsets = getattr(scene, "per_world_proxy_offsets", None)
    default_offset = cable.gripper_proxy_offset_in_apple_frame
    default_params = cable.params

    for w in range(num_envs):
        payload_i = int(payload_indices[w])
        apple_i = int(apple_indices[w])
        if payload_i < 0 or apple_i < 0:
            continue

        params_w = (
            per_params[w]
            if per_params is not None and w < len(per_params)
            else default_params
        )
        offset_w = default_offset
        if per_offsets is not None and w < len(per_offsets) and per_offsets[w] is not None:
            offset_w = per_offsets[w]
        if offset_w is None:
            offset_w = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        m_model = apple_mass_kg_from_model(cable.model, apple_i)
        m_analytic = analytic_apple_mass_kg(params_w)
        if m_analytic is not None and abs(float(m_analytic) - float(m_model)) > 1e-4 * max(
            1.0, abs(float(m_model))
        ):
            # Prefer AVBD body_mass (shape/builder authority) when they diverge.
            pass
        m = float(m_model)
        r = 0.0 if params_w.apple_radius is None else float(params_w.apple_radius)
        I = solid_sphere_inertia_diag(m, r)
        com = apple_com_in_tcp_frame(offset_w)
        _set_body_inertial_props(
            robot_model,
            payload_i,
            mass_kg=m,
            inertia=I,
            com_in_body=com,
        )

    mj_solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)


def clear_mujoco_apple_payload_inertias(scene: Any) -> None:
    """Zero MuJoCo apple-payload mass/inertia (keep the FIXED body topology).

    Used to A/B compare VIC response with vs without reflected fruit inertia
    without rebuilding the robot model.
    """
    robot_model = getattr(scene, "robot_model", None)
    mj_solver = getattr(scene, "mj_solver", None)
    if robot_model is None or mj_solver is None:
        return

    layout = getattr(scene, "layout", None)
    if layout is not None and int(getattr(layout, "num_envs", 1)) > 1:
        indices = tuple(int(i) for i in layout.mj_apple_payload_body_indices if int(i) >= 0)
    else:
        idx = getattr(scene, "mj_apple_payload_body_index", None)
        if idx is None:
            idx = resolve_apple_payload_body_index(robot_model)
        indices = (int(idx),) if idx is not None else ()

    zero_I = wp.mat33()
    zero_com = np.zeros(3, dtype=np.float64)
    for payload_i in indices:
        _set_body_inertial_props(
            robot_model,
            payload_i,
            mass_kg=0.0,
            inertia=zero_I,
            com_in_body=zero_com,
        )
    if indices:
        mj_solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
