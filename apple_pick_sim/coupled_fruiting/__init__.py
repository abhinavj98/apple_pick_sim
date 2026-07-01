"""MuJoCo + VBD coupled fruiting scene (M1 staggered coupling)."""

from __future__ import annotations

from apple_pick_sim.coupled_fruiting.apply_wrench import (
    _add_tcp_spatial_wrench_inplace,
    _apply_registry_spatial_wrenches_to_body_f,
    _apply_spatial_wrench_to_body_f,
)
from apple_pick_sim.coupled_fruiting.bootstrap import (
    bootstrap_articulated_tcp_from_proxy,
    bootstrap_tcp_joint_from_proxy,
)
from apple_pick_sim.coupled_fruiting.builders import (
    build_batched_coupled_fruiting_fr3,
    build_batched_coupled_fruiting_placeholder,
    build_coupled_fruiting_fr3,
    build_coupled_fruiting_placeholder,
    build_heterogeneous_coupled_fruiting_fr3,
    build_heterogeneous_coupled_fruiting_placeholder,
    build_placeholder_tcp_robot_model,
)
from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.coupled_fruiting.broadcast_actions import broadcast_joint_q_from_world0
from apple_pick_sim.coupled_fruiting.scene import (
    DEFAULT_FR3_MUJOCO_SOLVER_KWARGS,
    DEFAULT_MUJOCO_SOLVER_KWARGS,
    DEFAULT_STEM_COUPLING_GAIN,
    DEFAULT_STEM_FORCE_CAP_N,
    DEFAULT_STEM_TORQUE_CAP_NM,
    CoupledFruitingScene,
)
from apple_pick_sim.coupled_fruiting.settle_quasi_static import (
    SettleQuasiStaticReport,
    SettleStabilityReport,
    count_apples_outside_envelope,
    count_non_quasi_static_from_cable,
    nominal_spur_stem_apple_length_m,
    print_envelope_coverage_report,
    print_settle_quasi_static_summary,
    print_settle_stability_report,
    settle_stability_reports_from_cable,
)
from apple_pick_sim.coupled_fruiting.settle_then_weld import (
    quiet_all_cable_bodies,
    seed_fix_to_apple_from_settled,
    settle_vbd_substeps,
)

__all__ = [
    "BatchedEnvLayout",
    "CoupledFruitingScene",
    "DEFAULT_FR3_MUJOCO_SOLVER_KWARGS",
    "DEFAULT_MUJOCO_SOLVER_KWARGS",
    "DEFAULT_STEM_COUPLING_GAIN",
    "DEFAULT_STEM_FORCE_CAP_N",
    "DEFAULT_STEM_TORQUE_CAP_NM",
    "_add_tcp_spatial_wrench_inplace",
    "_apply_registry_spatial_wrenches_to_body_f",
    "_apply_spatial_wrench_to_body_f",
    "bootstrap_articulated_tcp_from_proxy",
    "bootstrap_tcp_joint_from_proxy",
    "broadcast_joint_q_from_world0",
    "build_batched_coupled_fruiting_fr3",
    "build_batched_coupled_fruiting_placeholder",
    "build_coupled_fruiting_fr3",
    "build_coupled_fruiting_placeholder",
    "build_heterogeneous_coupled_fruiting_fr3",
    "build_heterogeneous_coupled_fruiting_placeholder",
    "build_placeholder_tcp_robot_model",
    "SettleQuasiStaticReport",
    "SettleStabilityReport",
    "count_apples_outside_envelope",
    "count_non_quasi_static_from_cable",
    "nominal_spur_stem_apple_length_m",
    "print_envelope_coverage_report",
    "print_settle_quasi_static_summary",
    "print_settle_stability_report",
    "settle_stability_reports_from_cable",
    "quiet_all_cable_bodies",
    "seed_fix_to_apple_from_settled",
    "settle_vbd_substeps",
]
