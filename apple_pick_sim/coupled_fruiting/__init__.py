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
    build_mega_coupled_fruiting_fr3,
    build_placeholder_tcp_robot_model,
)
from apple_pick_sim.coupled_fruiting.scene import (
    DEFAULT_FR3_MUJOCO_SOLVER_KWARGS,
    DEFAULT_MUJOCO_SOLVER_KWARGS,
    DEFAULT_STEM_COUPLING_GAIN,
    DEFAULT_STEM_FORCE_CAP_N,
    DEFAULT_STEM_TORQUE_CAP_NM,
    CoupledFruitingScene,
    MegaCoupledFruitingScene,
    mega_ghost_position_offsets_wp,
)
from apple_pick_sim.coupled_fruiting.settle_then_weld import (
    seed_fix_to_apple_from_settled,
    seed_mega_fix_to_apple_from_settled,
    settle_vbd_substeps,
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
    "build_mega_coupled_fruiting_fr3",
    "MegaCoupledFruitingScene",
    "build_placeholder_tcp_robot_model",
    "seed_fix_to_apple_from_settled",
    "seed_mega_fix_to_apple_from_settled",
    "settle_vbd_substeps",
]
