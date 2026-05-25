"""Variational fruiting-system generator (P0) and M1 coupled cable scenes."""

from __future__ import annotations

from apple_pick_sim.fruiting_system.build import make_fruiting_solver_vbd
from apple_pick_sim.fruiting_system.coupled import (
    CoupledCableScene,
    generate_coupled_cable_scene,
    geometry_fingerprint_coupled,
)
from apple_pick_sim.fruiting_system.params import (
    FruitingSystemParams,
    GripperProxyConfig,
    RodParams,
    load_ranges,
    params_fingerprint,
    sample_params,
)
from apple_pick_sim.fruiting_system.scene import (
    FruitingSystemScene,
    _build_scene,
    example_collision_pipeline,
    generate_scene,
    geometry_fingerprint,
    iter_fruiting_fixed_joint_indices,
    measure_fruiting_forces,
    run_rollout,
)

from apple_pick_sim.vbd_fixed_joint_wrenches import (
    FixedJointWrenchRecord,
    fixed_joint_wrenches_child_com_vbd,
    iter_fixed_joint_indices,
)

__all__ = [
    "CoupledCableScene",
    "FruitingSystemParams",
    "FruitingSystemScene",
    "GripperProxyConfig",
    "RodParams",
    "_build_scene",
    "example_collision_pipeline",
    "generate_coupled_cable_scene",
    "generate_scene",
    "geometry_fingerprint",
    "geometry_fingerprint_coupled",
    "FixedJointWrenchRecord",
    "fixed_joint_wrenches_child_com_vbd",
    "iter_fruiting_fixed_joint_indices",
    "load_ranges",
    "make_fruiting_solver_vbd",
    "measure_fruiting_forces",
    "params_fingerprint",
    "run_rollout",
    "sample_params",
]
