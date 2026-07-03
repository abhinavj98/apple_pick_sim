"""Config-driven build orchestration for batched heterogeneous coupled fruiting (V.3.1 step A)."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
)
from apple_pick_sim.coupled_fruiting.builders import (
    build_heterogeneous_coupled_fruiting_fr3,
    build_heterogeneous_coupled_fruiting_placeholder,
)
from apple_pick_sim.coupled_fruiting.scene import CoupledFruitingScene
from apple_pick_sim.coupled_fruiting.settled_checkpoint import SettledCheckpoint
from apple_pick_sim.coupled_fruiting.settle_ke_decay import (
    SettleKeDecayReport,
    SettleKeRecorder,
)
from apple_pick_sim.coupled_fruiting.settle_quasi_static import (
    SettleStabilityReport,
    settle_stability_reports_from_cable,
)
from apple_pick_sim.coupled_fruiting.settle_then_weld import (
    apply_settle_gravity_for_substep,
    quiet_all_cable_bodies,
    seed_fix_to_apple_from_settled,
    seed_fix_to_apple_from_settled_body_q,
)
from apple_pick_sim.fruiting_system import (
    FruitingSystemParams,
    GripperProxyConfig,
    set_fruiting_joint_angular_kd_batched,
)
from apple_pick_sim.robot import fr3_robot

# Heterogeneous example: sag under gravity can exceed straight rest length.
_SETTLE_PATH_RTOL = 0.05


@dataclasses.dataclass(frozen=True)
class BatchedHeterogeneousBuildResult:
    """Output of :func:`build_batched_heterogeneous_scene`."""

    scene: CoupledFruitingScene
    per_env_params: tuple[FruitingSystemParams, ...]
    joint_angular_kd_overrides: dict[str, float]
    settled_body_q: np.ndarray | None = None
    settle_stability_reports: tuple[SettleStabilityReport, ...] | None = None
    settle_ke_decay_reports: tuple[SettleKeDecayReport, ...] | None = None
    ik_envelope_results: tuple[tuple[float, float, bool], ...] | None = None


def build_batched_heterogeneous_scene(
    config: BatchedHeterogeneousCoupledSimConfig,
    per_env_params: Sequence[FruitingSystemParams],
    ranges: dict,
    *,
    viewer: Any | None = None,
    settled_checkpoint: SettledCheckpoint | None = None,
) -> BatchedHeterogeneousBuildResult:
    """Build a batched heterogeneous coupled scene from config and pre-sampled params."""
    config.validate()
    params = tuple(per_env_params)
    if len(params) != config.runtime.num_envs:
        raise ValueError(
            f"per_env_params length ({len(params)}) must match "
            f"runtime.num_envs ({config.runtime.num_envs})"
        )

    robot_kind = _resolve_robot_kind(config.robot.kind)
    build_fn = (
        build_heterogeneous_coupled_fruiting_fr3
        if robot_kind == "fr3"
        else build_heterogeneous_coupled_fruiting_placeholder
    )

    vbd_only = config.robot.step_mode == "vbd_only"
    fix_to_apple = config.robot.fix_to_apple
    settle_substeps = int(config.scene.settle_substeps)
    sim_dt = float(config.runtime.sub_dt)
    collect_diag = config.settle_diagnostics is not None

    stability_reports: list[SettleStabilityReport] = []
    ke_decay_reports: list[SettleKeDecayReport] = []
    ik_results: list[tuple[float, float, bool]] | None = None
    settled_body_q: np.ndarray | None = None

    if fix_to_apple and not vbd_only:
        gripper_weld = _gripper_proxy(config, fix_to_apple=True)
        if settled_checkpoint is not None:
            weld_kw = _builder_kwargs(
                config, gripper=gripper_weld, vbd_only=False, robot_kind=robot_kind
            )
            if robot_kind == "fr3":
                weld_kw["skip_ik_bootstrap"] = True
                weld_kw["defer_template_robot_bootstrap"] = True
            scene = build_fn(ranges, params, **weld_kw)
            seed_fix_to_apple_from_settled_body_q(
                welded_scene=scene,
                settled_body_q=settled_checkpoint.body_q,
                quiet_apple_proxy=True,
                per_env_ik=True,
                per_world_proxy_offsets=scene.per_world_proxy_offsets,
            )
            ik_raw = getattr(scene, "settle_ik_envelope_results", None)
            ik_results = list(ik_raw) if ik_raw else None
        else:
            gripper_free = _gripper_proxy(config, fix_to_apple=False)
            settled = build_fn(
                ranges,
                params,
                **_builder_kwargs(
                    config, gripper=gripper_free, vbd_only=True, robot_kind=robot_kind
                ),
            )
            stability_reports, ke_decay_reports = _run_vbd_settle(
                settled,
                config=config,
                per_env_params=params,
                substeps=settle_substeps,
                sim_dt=sim_dt,
                viewer=viewer,
                collect_diagnostics=collect_diag,
            )
            settled_body_q = settled.cable.state_0.body_q.numpy().reshape(-1, 7).copy()
            weld_kw = _builder_kwargs(
                config, gripper=gripper_weld, vbd_only=False, robot_kind=robot_kind
            )
            if robot_kind == "fr3":
                weld_kw["skip_ik_bootstrap"] = True
                weld_kw["defer_template_robot_bootstrap"] = True
            scene = build_fn(ranges, params, **weld_kw)
            seed_fix_to_apple_from_settled(
                welded_scene=scene,
                settled_scene=settled,
                quiet_apple_proxy=True,
                per_env_ik=True,
                per_world_proxy_offsets=scene.per_world_proxy_offsets,
            )
            ik_raw = getattr(scene, "settle_ik_envelope_results", None)
            ik_results = list(ik_raw) if ik_raw else None
    else:
        scene = build_fn(
            ranges,
            params,
            **_builder_kwargs(
                config,
                gripper=_gripper_proxy(config),
                vbd_only=vbd_only,
                robot_kind=robot_kind,
            ),
        )
        if settle_substeps > 0:
            stability_reports, ke_decay_reports = _run_vbd_settle(
                scene,
                config=config,
                per_env_params=params,
                substeps=settle_substeps,
                sim_dt=sim_dt,
                viewer=viewer,
                collect_diagnostics=collect_diag,
            )

    kd_overrides = dict(config.fruiting_system.joint_angular_kd_overrides)
    applied_kd = _apply_joint_kd_overrides(scene, kd_overrides)

    if not collect_diag:
        return BatchedHeterogeneousBuildResult(
            scene=scene,
            per_env_params=params,
            joint_angular_kd_overrides=applied_kd,
            settled_body_q=settled_body_q,
        )

    return BatchedHeterogeneousBuildResult(
        scene=scene,
        per_env_params=params,
        joint_angular_kd_overrides=applied_kd,
        settled_body_q=settled_body_q,
        settle_stability_reports=tuple(stability_reports),
        settle_ke_decay_reports=tuple(ke_decay_reports),
        ik_envelope_results=tuple(ik_results) if ik_results is not None else None,
    )


def _resolve_robot_kind(kind: str) -> str:
    if kind == "fr3" and not fr3_robot.fr3_assets_available():
        warnings.warn(
            "FR3 assets not found; building with placeholder TCP.",
            UserWarning,
            stacklevel=3,
        )
        return "placeholder"
    return kind


def _gripper_proxy(
    config: BatchedHeterogeneousCoupledSimConfig,
    *,
    fix_to_apple: bool | None = None,
) -> GripperProxyConfig:
    fix = config.robot.fix_to_apple if fix_to_apple is None else fix_to_apple
    return dataclasses.replace(
        config.robot.gripper,
        fix_to_apple=fix,
        robot_facing_weld=fix,
    )


def _builder_kwargs(
    config: BatchedHeterogeneousCoupledSimConfig,
    *,
    gripper: GripperProxyConfig,
    vbd_only: bool,
    robot_kind: str,
) -> dict[str, Any]:
    scene_cfg = config.scene
    robot_cfg = config.robot
    fruiting = config.fruiting_system
    mujoco = config.mujoco
    kw: dict[str, Any] = {
        "device": config.resolve_device(),
        "env_spacing": config.runtime.env_spacing,
        "enable_self_collisions": scene_cfg.enable_self_collisions,
        "enable_apple_woody_collisions": scene_cfg.enable_apple_woody_collisions,
        "enable_proxy_woody_collisions": scene_cfg.enable_proxy_woody_collisions,
        "gripper_proxy": gripper,
        "vbd_only": vbd_only,
        "base_pos": scene_cfg.fruiting_base_pos,
        "robot_base_pos": robot_cfg.robot_base_pos,
        "stem_coupling_gain": fruiting.stem_coupling_gain,
        "stem_force_cap_N": fruiting.stem_force_cap_N,
        "stem_torque_cap_Nm": fruiting.stem_torque_cap_Nm,
        "mujoco_solver_kwargs": dict(mujoco.solver_kwargs),
        "mujoco_use_cpu": mujoco.use_cpu,
        "skip_ik_bootstrap": robot_cfg.skip_ik_bootstrap,
    }
    if robot_kind == "fr3":
        kw["ik_bootstrap_iterations"] = robot_cfg.ik_bootstrap_iterations
        kw["defer_template_robot_bootstrap"] = robot_cfg.defer_template_robot_bootstrap
    return kw


def _matching_kd_overrides(
    fruiting_fixed_joints: Sequence[tuple[int, str]],
    kd_overrides: dict[str, float],
) -> dict[str, float]:
    """Keep override keys that match exactly one fixed-joint label in the template."""
    if not kd_overrides:
        return {}
    keys = list(kd_overrides.keys())
    used_keys: set[str] = set()
    for _joint_index, label in fruiting_fixed_joints:
        matching = [k for k in keys if k in label]
        if len(matching) == 1:
            used_keys.add(matching[0])
    return {k: kd_overrides[k] for k in sorted(used_keys)}


def _apply_joint_kd_overrides(
    scene: CoupledFruitingScene,
    kd_overrides: dict[str, float],
) -> dict[str, float]:
    layout = scene.layout
    if layout is None or not kd_overrides:
        return {}
    filtered = _matching_kd_overrides(scene.cable.fruiting_fixed_joints, kd_overrides)
    if not filtered:
        return {}
    set_fruiting_joint_angular_kd_batched(
        scene.cable.solver,
        scene.cable.fruiting_fixed_joints,
        filtered,
        num_envs=layout.num_envs,
        joints_per_world=layout.joints_per_world,
    )
    return dict(filtered)


def _run_vbd_settle(
    scene: CoupledFruitingScene,
    *,
    config: BatchedHeterogeneousCoupledSimConfig,
    per_env_params: tuple[FruitingSystemParams, ...],
    substeps: int,
    sim_dt: float,
    viewer: Any | None,
    collect_diagnostics: bool,
) -> tuple[list[SettleStabilityReport], list[SettleKeDecayReport]]:
    n = int(substeps)
    if n <= 0:
        return [], []

    scene_cfg = config.scene
    diag = config.settle_diagnostics
    ke_enabled = collect_diagnostics and diag is not None and diag.enabled
    recorder = (
        SettleKeRecorder(
            num_envs=config.runtime.num_envs,
            sample_every=int(diag.ke_sample_every),
        )
        if ke_enabled
        else None
    )
    h = float(sim_dt)
    gravity_ramp = bool(scene_cfg.settle_gravity_ramp)

    for substep_idx in range(n):
        apply_settle_gravity_for_substep(
            scene,
            substep_idx,
            n,
            gravity_ramp=gravity_ramp,
        )
        scene.vbd_substep(h)
        if recorder is not None:
            recorder.record_substep(
                scene.cable,
                per_env_params,
                substep_idx,
                h,
                sample_every=int(diag.ke_sample_every),
            )
        _maybe_render_settle(viewer, scene, float(substep_idx + 1) * h)

    quiet_all_cable_bodies(scene.cable)

    if not collect_diagnostics:
        return [], []

    stability_reports = settle_stability_reports_from_cable(
        scene.cable,
        per_env_params,
        max_branch_speed_m_s=float(scene_cfg.settle_max_speed_m_s),
        path_rtol=_SETTLE_PATH_RTOL,
    )
    ke_decay_reports: list[SettleKeDecayReport] = []
    if recorder is not None and diag is not None:
        ke_decay_reports = recorder.reports(config=diag.ke_analysis)
    return stability_reports, ke_decay_reports


def _maybe_render_settle(
    viewer: Any | None,
    scene: CoupledFruitingScene,
    sim_time_s: float,
) -> None:
    if viewer is None:
        return
    is_running = getattr(viewer, "is_running", None)
    if is_running is not None and not is_running():
        return
    if hasattr(viewer, "begin_frame"):
        viewer.begin_frame(sim_time_s)
    if hasattr(viewer, "log_state"):
        viewer.log_state(scene.cable.state_0)
    if hasattr(viewer, "end_frame"):
        viewer.end_frame()


def print_per_env_params(params_list: Sequence[FruitingSystemParams]) -> None:
    """Print per-env continuous θ (topology shared) for CLI diagnostics."""
    print("Per-env fruiting params (topology shared, continuous θ differs):")
    for w, p in enumerate(params_list):
        print(f"  env{w}:")
        for seg_name in ("primary", "secondary", "spur", "stem"):
            rod = getattr(p, seg_name)
            if rod is None:
                continue
            print(
                f"    {seg_name}: E={rod.youngs_modulus_pa:.4g} Pa  "
                f"zeta={rod.damping_ratio:.4g}  "
                f"k_bend={rod.bend_stiffness:.4g} N·m/rad  "
                f"c_bend={rod.bend_damping:.4g} N·m·s/rad  "
                f"k_stretch={rod.stretch_stiffness:.4g} N/m  "
                f"c_stretch={rod.stretch_damping:.4g} N·s/m"
            )
        radius = float(p.apple_radius) if p.apple_radius is not None else float("nan")
        density = float(p.apple_density) if p.apple_density is not None else float("nan")
        print(f"    apple: r={radius:.4g} m  rho={density:.4g} kg/m³")
