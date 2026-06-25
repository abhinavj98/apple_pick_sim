"""FR3 + custom end-effector import for M1 ``robot_model``."""

from __future__ import annotations

from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import (
    Fr3EEImpedanceController,
    ImpedanceGains,
)
from apple_pick_sim.robot.fr3_robot.controllers.ee_direct_joint import Fr3EEDirectJointController
from apple_pick_sim.robot.fr3_robot.controllers.ee_direct_joint_batched import (
    Fr3BatchedEEDirectJointController,
)
from apple_pick_sim.robot.fr3_robot.controllers.ee_velocity import Fr3EEVelocityController
from apple_pick_sim.robot.fr3_robot.controllers.ee_velocity_batched import (
    Fr3BatchedEEVelocityController,
)
from apple_pick_sim.robot.fr3_robot.controllers.keyboard import (
    EEVelocity,
    FR3_KEYBOARD_BINDINGS,
    _KeyViewer,
    add_gaussian_noise_to_ee_velocity,
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
    IKBootstrapConvergenceError,
    IKBootstrapConvergenceWarning,
    IKTeleopConvergenceError,
    IK_TELEOP_POS_TOL_M,
    IK_TELEOP_ROT_TOL_RAD,
    bootstrap_tcp_ik_from_proxy,
    enable_ik_bootstrap_warnings_for_examples,
    batched_ik_teleop_kwargs,
    placement_xform_for_proxy,
    root_world_translation_for_proxy,
    raise_if_ik_bootstrap_not_converged,
    warn_ik_bootstrap_for_fr3_scene,
)
from apple_pick_sim.robot.fr3_robot.setup import (
    build_fr3_robot_builder,
    build_fr3_robot_model_from_usd,
    configure_vic_joint_torques_arm,
    configure_vic_wrench_only_arm,
    hold_mujoco_actuator_targets_at_state,
    init_mujoco_actuator_targets_from_model,
    np_zeros_like_joint_qd,
    resolve_ee_body_index,
    resolve_tcp_body_index,
    scale_mujoco_joint_pd,
    sync_mujoco_actuator_targets_from_joint_q,
    sync_mujoco_visual_state,
    sync_robot_gravity_to_mujoco,
    zero_mujoco_joint_pd,
)

__all__ = [
    "EE_BOX_HALF_EXTENTS",
    "EE_MASS_KG",
    "EEVelocity",
    "FR3_KEYBOARD_BINDINGS",
    "Fr3BatchedEEDirectJointController",
    "Fr3BatchedEEVelocityController",
    "Fr3EEDirectJointController",
    "Fr3EEImpedanceController",
    "Fr3EEVelocityController",
    "ImpedanceGains",
    "OMNIVERSE_FR3_SCHEMA",
    "OMNIVERSE_FR3_USD",
    "TESTFR3_SCENE_USD",
    "_KeyViewer",
    "IKBootstrapConvergenceError",
    "IKBootstrapConvergenceWarning",
    "IKTeleopConvergenceError",
    "IK_TELEOP_POS_TOL_M",
    "IK_TELEOP_ROT_TOL_RAD",
    "bootstrap_tcp_ik_from_proxy",
    "add_gaussian_noise_to_ee_velocity",
    "enable_ik_bootstrap_warnings_for_examples",
    "build_fr3_robot_builder",
    "build_fr3_robot_model_from_usd",
    "configure_vic_joint_torques_arm",
    "configure_vic_wrench_only_arm",
    "fr3_assets_available",
    "hold_mujoco_actuator_targets_at_state",
    "init_mujoco_actuator_targets_from_model",
    "integrate_tcp_target",
    "np_zeros_like_joint_qd",
    "placement_xform_for_proxy",
    "root_world_translation_for_proxy",
    "raise_if_ik_bootstrap_not_converged",
    "warn_ik_bootstrap_for_fr3_scene",
    "poll_viewer_events",
    "print_fr3_keyboard_bindings",
    "read_keyboard_ee_velocity",
    "resolve_ee_body_index",
    "resolve_tcp_body_index",
    "scale_mujoco_joint_pd",
    "sync_mujoco_actuator_targets_from_joint_q",
    "zero_mujoco_joint_pd",
    "sync_mujoco_visual_state",
    "sync_robot_gravity_to_mujoco",
    "batched_ik_teleop_kwargs",
]
