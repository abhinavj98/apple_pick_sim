"""Variational fruiting-system generator (P0) and M1 coupled cable scenes."""

from __future__ import annotations

from apple_pick_sim.fruiting_system.build import make_fruiting_solver_vbd
from apple_pick_sim.fruiting_system.coupled import (
    CoupledCableScene,
    generate_coupled_cable_scene,
    geometry_fingerprint_coupled,
)
from apple_pick_sim.fruiting_system.mega import (
    FruitingInstanceLayout,
    MegaCoupledCableScene,
    generate_mega_coupled_cable_scene,
)
from apple_pick_sim.fruiting_system.mega_fd import (
    MegaFdStepResult,
    copy_coupled_scene_from_nominal,
    copy_mega_instance_state,
    coupled_vbd_substep,
    default_mega_fd_features,
    extract_mega_fd_jacobian,
    instance_body_ids,
    mega_fd_step,
    mega_vbd_substep,
    reset_perturbed_instances_to_nominal,
    sync_all_instances_from_nominal,
)
from apple_pick_sim.fruiting_system.params import (
    FruitingSystemParams,
    GripperProxyConfig,
    RodParams,
    analytic_apple_mass_kg,
    copy_fruiting_params,
    enabled_rod_segments,
    fd_stiffness_param_columns,
    load_ranges,
    params_fingerprint,
    perturb_rod_stiffness,
    sample_params,
    set_rod_bend_stiffness,
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
    "FruitingInstanceLayout",
    "FruitingSystemParams",
    "FruitingSystemScene",
    "GripperProxyConfig",
    "MegaCoupledCableScene",
    "MegaFdStepResult",
    "RodParams",
    "analytic_apple_mass_kg",
    "copy_coupled_scene_from_nominal",
    "copy_mega_instance_state",
    "coupled_vbd_substep",
    "default_mega_fd_features",
    "extract_mega_fd_jacobian",
    "instance_body_ids",
    "mega_fd_step",
    "mega_vbd_substep",
    "reset_perturbed_instances_to_nominal",
    "sync_all_instances_from_nominal",
    "_build_scene",
    "copy_fruiting_params",
    "enabled_rod_segments",
    "example_collision_pipeline",
    "fd_stiffness_param_columns",
    "generate_coupled_cable_scene",
    "generate_mega_coupled_cable_scene",
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
    "perturb_rod_stiffness",
    "set_rod_bend_stiffness",
    "run_rollout",
    "sample_params",
]
