"""FR3 + custom end-effector import for M1 ``robot_model``."""

from __future__ import annotations

from apple_pick_sim.robot.fr3_robot.controllers.ee_direct_joint import Fr3EEDirectJointController
from apple_pick_sim.robot.fr3_robot.controllers.ee_velocity import Fr3EEVelocityController
from apple_pick_sim.robot.fr3_robot.controllers.keyboard import (
    EEVelocity,
    FR3_KEYBOARD_BINDINGS,
    _KeyViewer,
    integrate_tcp_target,
    poll_viewer_events,
    print_fr3_keyboard_bindings,
    read_keyboard_ee_velocity,
)
from apple_pick_sim.robot.fr3_robot.paths import (
    EE_BOX_HALF_EXTENTS,
    EE_MASS_KG,
    OMNIVERSE_FR3_SCHEMA,
    OMNIVERSE_FR3_USD,
    TESTFR3_SCENE_USD,
    fr3_assets_available,
)
from apple_pick_sim.robot.fr3_robot.placement import (
    bootstrap_tcp_ik_from_proxy,
    placement_xform_for_proxy,
)
from apple_pick_sim.robot.fr3_robot.setup import (
    build_fr3_robot_model_from_usd,
    init_mujoco_actuator_targets_from_model,
    np_zeros_like_joint_qd,
    resolve_ee_body_index,
    resolve_tcp_body_index,
    sync_mujoco_actuator_targets_from_joint_q,
    sync_mujoco_visual_state,
    sync_robot_gravity_to_mujoco,
)

__all__ = [
    "EE_BOX_HALF_EXTENTS",
    "EE_MASS_KG",
    "EEVelocity",
    "FR3_KEYBOARD_BINDINGS",
    "Fr3EEDirectJointController",
    "Fr3EEVelocityController",
    "OMNIVERSE_FR3_SCHEMA",
    "OMNIVERSE_FR3_USD",
    "TESTFR3_SCENE_USD",
    "_KeyViewer",
    "bootstrap_tcp_ik_from_proxy",
    "build_fr3_robot_model_from_usd",
    "fr3_assets_available",
    "init_mujoco_actuator_targets_from_model",
    "integrate_tcp_target",
    "np_zeros_like_joint_qd",
    "placement_xform_for_proxy",
    "poll_viewer_events",
    "print_fr3_keyboard_bindings",
    "read_keyboard_ee_velocity",
    "resolve_ee_body_index",
    "resolve_tcp_body_index",
    "sync_mujoco_actuator_targets_from_joint_q",
    "sync_mujoco_visual_state",
    "sync_robot_gravity_to_mujoco",
]
