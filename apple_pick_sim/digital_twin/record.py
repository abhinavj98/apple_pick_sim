"""Record digital-twin observations from an initialized simulation scene."""

from __future__ import annotations

from typing import Any

import numpy as np

from apple_pick_sim.digital_twin.obs_io import DigitalTwinObs
from apple_pick_sim.fruiting_system.scene import fixed_joint_anchors_world

_FRUITING_JUNCTION_SUFFIXES = ("_gripper_proxy",)


def fruiting_tree_fixed_joints(scene: Any) -> tuple[tuple[int, str], ...]:
    """Return fruiting-chain fixed joints only (exclude gripper-proxy welds)."""
    return tuple(
        pair
        for pair in scene.fruiting_fixed_joints
        if not any(pair[1].endswith(suffix) for suffix in _FRUITING_JUNCTION_SUFFIXES)
    )


def default_weld_direction_from_scene(
    scene: Any,
    robot_base_pos: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Unit vector from apple COM toward ``robot_base_pos`` (robot-facing hemisphere pole)."""
    if scene.apple_body is None:
        raise ValueError("scene has no apple body; pass weld_direction explicitly")
    bq = scene.state_0.body_q.numpy().reshape(-1, 7)
    apple_com = bq[int(scene.apple_body), :3]
    direction = np.asarray(robot_base_pos, dtype=np.float64) - apple_com
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError("robot_base_pos coincides with apple COM")
    unit = direction / norm
    return (float(unit[0]), float(unit[1]), float(unit[2]))


def _rod_radii_from_scene(scene: Any) -> dict[str, float] | None:
    params = getattr(scene, "params", None)
    if params is None:
        return None
    radii: dict[str, float] = {}
    for name in ("primary", "secondary", "spur", "stem"):
        segment = getattr(params, name, None)
        radius = getattr(segment, "radius", None)
        if radius is not None:
            radii[name] = float(radius)
    return radii or None


def record_obs_from_scene(
    scene: Any,
    *,
    fruiting_base_pos: tuple[float, float, float],
    weld_direction: tuple[float, float, float] | None = None,
    robot_base_pos: tuple[float, float, float] | None = None,
    apple_radius: float | None = None,
) -> DigitalTwinObs:
    """Build :class:`DigitalTwinObs` from a built cable or fruiting scene at rest.

    Junction labels and anchor arrays match the gym observation contract
    (``woody_part_start_pos`` / ``woody_part_end_pos``). Gripper-proxy welds are
  omitted so the file can seed a fresh ``build_digital_twin_scene`` call.
    """
    joints = fruiting_tree_fixed_joints(scene)
    if not joints:
        raise ValueError("scene has no fruiting fixed joints to record")

    parent, child = fixed_joint_anchors_world(scene.model, scene.state_0.body_q, joints)
    junction_names = [label.removeprefix("joint_") for _, label in joints]

    if weld_direction is None:
        if robot_base_pos is None:
            raise ValueError("pass weld_direction or robot_base_pos")
        weld_direction = default_weld_direction_from_scene(scene, robot_base_pos)

    if apple_radius is None and scene.params.apple_radius is not None:
        apple_radius = float(scene.params.apple_radius)

    return DigitalTwinObs(
        fruiting_base_pos=fruiting_base_pos,
        weld_direction=weld_direction,
        junction_names=junction_names,
        woody_part_start_pos=parent,
        woody_part_end_pos=child,
        apple_radius=apple_radius,
        rod_radii=_rod_radii_from_scene(scene),
    )
