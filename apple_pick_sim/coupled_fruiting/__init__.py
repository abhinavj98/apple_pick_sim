"""MuJoCo + VBD coupled fruiting scene (M1 staggered coupling)."""

from __future__ import annotations

from apple_pick_sim.coupled_fruiting.apply_wrench import _apply_spatial_wrench_to_body_f
from apple_pick_sim.coupled_fruiting.bootstrap import (
    bootstrap_articulated_tcp_from_proxy,
    bootstrap_tcp_joint_from_proxy,
)
from apple_pick_sim.coupled_fruiting.builders import (
    build_coupled_fruiting_fr3,
    build_coupled_fruiting_placeholder,
    build_placeholder_tcp_robot_model,
)
from apple_pick_sim.coupled_fruiting.scene import (
    DEFAULT_FR3_MUJOCO_SOLVER_KWARGS,
    DEFAULT_MUJOCO_SOLVER_KWARGS,
    DEFAULT_STEM_COUPLING_GAIN,
    DEFAULT_STEM_FORCE_CAP_N,
    DEFAULT_STEM_TORQUE_CAP_NM,
    CoupledFruitingScene,
)

__all__ = [
    "CoupledFruitingScene",
    "DEFAULT_FR3_MUJOCO_SOLVER_KWARGS",
    "DEFAULT_MUJOCO_SOLVER_KWARGS",
    "DEFAULT_STEM_COUPLING_GAIN",
    "DEFAULT_STEM_FORCE_CAP_N",
    "DEFAULT_STEM_TORQUE_CAP_NM",
    "_apply_spatial_wrench_to_body_f",
    "bootstrap_articulated_tcp_from_proxy",
    "bootstrap_tcp_joint_from_proxy",
    "build_coupled_fruiting_fr3",
    "build_coupled_fruiting_placeholder",
    "build_placeholder_tcp_robot_model",
]
