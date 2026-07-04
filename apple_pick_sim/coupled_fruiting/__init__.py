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
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_build import (
    BatchedHeterogeneousBuildResult,
    build_batched_heterogeneous_scene,
    print_per_env_params,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_coupled_sim import (
    BatchedHeterogeneousCoupledSim,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ControllerConfig,
    DomainRandomizationConfig,
    FruitingSystemConfig,
    MujocoConfig,
    ObsConfig,
    RobotConfig,
    RuntimeConfig,
    SceneSettleCollisionConfig,
    SettleDiagnosticsConfig,
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
from apple_pick_sim.coupled_fruiting.episode_state_snapshot import EpisodeStateSnapshot
from apple_pick_sim.coupled_fruiting.settled_checkpoint import (
    SettledCheckpoint,
    settle_cache_path_for,
)
from apple_pick_sim.coupled_fruiting.settle_ke_decay import (
    SettleKeAnalysisConfig,
    SettleKeDecayReport,
    SettleKeRecorder,
    branch_linear_kinetic_energy_j,
    print_settle_ke_decay_report,
    settle_ke_decay_reports_from_recorder,
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
    seed_fix_to_apple_from_settled_body_q,
    settle_vbd_substeps,
)

__all__ = [
    "BatchedEnvLayout",
    "BatchedHeterogeneousBuildResult",
    "BatchedHeterogeneousCoupledSim",
    "BatchedHeterogeneousCoupledSimConfig",
    "EpisodeStateSnapshot",
    "SettledCheckpoint",
    "build_batched_heterogeneous_scene",
    "print_per_env_params",
    "settle_cache_path_for",
    "ControllerConfig",
    "DomainRandomizationConfig",
    "FruitingSystemConfig",
    "MujocoConfig",
    "ObsConfig",
    "RobotConfig",
    "RuntimeConfig",
    "SceneSettleCollisionConfig",
    "SettleDiagnosticsConfig",
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
    "SettleKeAnalysisConfig",
    "SettleKeDecayReport",
    "SettleKeRecorder",
    "SettleQuasiStaticReport",
    "SettleStabilityReport",
    "branch_linear_kinetic_energy_j",
    "count_apples_outside_envelope",
    "count_non_quasi_static_from_cable",
    "nominal_spur_stem_apple_length_m",
    "print_envelope_coverage_report",
    "print_settle_ke_decay_report",
    "print_settle_quasi_static_summary",
    "print_settle_stability_report",
    "settle_ke_decay_reports_from_recorder",
    "settle_stability_reports_from_cable",
    "quiet_all_cable_bodies",
    "seed_fix_to_apple_from_settled",
    "settle_vbd_substeps",
]
