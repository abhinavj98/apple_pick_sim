"""Config-driven build orchestration for batched heterogeneous coupled fruiting (V.3.1 step A)."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
)
from apple_pick_sim.coupled_fruiting.builders import build_heterogeneous_coupled_fruiting_fr3
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
from apple_pick_sim.coupled_fruiting.settle_seed_device import capture_body_q_numpy
from apple_pick_sim.coupled_fruiting.settle_then_weld import (
    _bootstrap_tcp_at_fixed_origin,
    _bootstrap_tcp_per_env,
    apply_settle_gravity_for_substep,
    quiet_all_cable_bodies,
    seed_fix_to_apple_from_settled,
    seed_fix_to_apple_from_settled_body_q,
    should_quiet_cable_bodies_at_settle_substep,
    warn_settle_quiet_every_alignment,
)
from apple_pick_sim.coupled_fruiting.proxy_coupling import (
    align_proxy_body_q_prev_for_vbd,
    prepare_batched_stem_harvest_arrays,
    sync_model_body_q_rest_from_state,
)
from apple_pick_sim.coupled_fruiting.broadcast_actions import broadcast_joint_q_from_world0
from apple_pick_sim.digital_twin.record import fruiting_tree_fixed_joints
from apple_pick_sim.fruiting_system import (
    FruitingSystemParams,
    GripperProxyConfig,
    set_fruiting_joint_angular_kd_batched,
    set_fruiting_joint_angular_kp_batched,
    set_fruiting_joint_linear_kd_batched,
    set_fruiting_joint_linear_kp_batched,
)
from apple_pick_sim.fruiting_system.joint_kd_scaling import joint_kd_from_damping_ratio
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.system_id.pre_weld_obs import capture_pre_weld_tree_obs_all_worlds

# Heterogeneous example: sag under gravity can exceed straight rest length.
_SETTLE_PATH_RTOL = 0.05
_SETTLE_RENDER_TARGET_FRAMES = 200
_SETTLE_RENDER_FRAME_DT_S = 1.0 / 30.0


@dataclasses.dataclass
class _SettleViewerState:
    model_bound: bool = False


def _settle_render_stride(substeps: int) -> int:
    n = int(substeps)
    if n <= 1:
        return 1
    return max(1, n // _SETTLE_RENDER_TARGET_FRAMES)


def _should_render_settle_substep(substep_idx: int, substeps: int, stride: int) -> bool:
    if stride <= 1:
        return True
    if substep_idx == 0 or substep_idx == substeps - 1:
        return True
    return (substep_idx + 1) % stride == 0


@dataclasses.dataclass(frozen=True)
class BatchedHeterogeneousBuildResult:
    """Output of :func:`build_batched_heterogeneous_scene`."""

    scene: CoupledFruitingScene
    per_env_params: tuple[FruitingSystemParams, ...]
    joint_angular_kd_overrides: dict[str, float]
    joint_linear_kd_overrides: dict[str, float]
    joint_angular_kp_overrides: dict[str, float]
    joint_linear_kp_overrides: dict[str, float]
    settled_body_q: np.ndarray | None = None
    pre_weld_tree_obs: tuple[dict[str, Any], ...] | None = None
    settle_stability_reports: tuple[SettleStabilityReport, ...] | None = None
    settle_ke_decay_reports: tuple[SettleKeDecayReport, ...] | None = None
    ik_envelope_results: tuple[tuple[float, float, bool], ...] | None = None


def build_batched_heterogeneous_scene(
    config: BatchedHeterogeneousCoupledSimConfig,
    per_env_params: Sequence[FruitingSystemParams],
    ranges: dict,
    *,
    per_env_grippers: Sequence[GripperProxyConfig] | None = None,
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

    _require_fr3_assets()
    build_fn = build_heterogeneous_coupled_fruiting_fr3

    vbd_only = config.robot.step_mode == "vbd_only"
    fix_to_apple = config.robot.fix_to_apple
    weld_grippers = _normalize_per_env_grippers(
        config.robot.gripper,
        per_env_grippers,
        num_envs=config.runtime.num_envs,
        fix_to_apple=True,
    )
    free_grippers = _normalize_per_env_grippers(
        config.robot.gripper,
        per_env_grippers,
        num_envs=config.runtime.num_envs,
        fix_to_apple=False,
    )
    settle_substeps = int(config.scene.settle_substeps)
    post_grasp_settle_substeps = int(config.scene.post_grasp_settle_substeps)
    sim_dt = float(config.runtime.sub_dt)
    collect_diag = config.settle_diagnostics is not None

    stability_reports: list[SettleStabilityReport] = []
    ke_decay_reports: list[SettleKeDecayReport] = []
    ik_results: list[tuple[float, float, bool]] | None = None
    settled_body_q: np.ndarray | None = None
    pre_weld_tree_obs: tuple[dict[str, Any], ...] | None = None

    def _capture_pre_weld_tree_obs(scene: Any) -> tuple[dict[str, Any], ...] | None:
        layout = getattr(scene, "layout", None)
        cable = getattr(scene, "cable", None)
        if layout is None or cable is None:
            return None
        joint_pairs = list(fruiting_tree_fixed_joints(cable))
        if not joint_pairs:
            return None
        junction_names = [label.removeprefix("joint_") for _, label in joint_pairs]
        return capture_pre_weld_tree_obs_all_worlds(
            cable,
            layout,
            junction_names=junction_names,
        )

    def _apply_post_grasp_settle(welded: Any) -> tuple[list[SettleStabilityReport], list[SettleKeDecayReport]]:
        if post_grasp_settle_substeps <= 0:
            return [], []
        post_stab, post_ke = _run_vbd_settle(
            welded,
            config=config,
            per_env_params=params,
            substeps=post_grasp_settle_substeps,
            sim_dt=sim_dt,
            viewer=viewer,
            collect_diagnostics=collect_diag,
        )
        _rebootstrap_fr3_after_post_grasp_settle(welded, config=config)
        return post_stab, post_ke

    angular_kd_overrides = dict(config.fruiting_system.joint_angular_kd_overrides)
    linear_kd_overrides = dict(config.fruiting_system.joint_linear_kd_overrides)
    angular_kp_overrides = dict(config.fruiting_system.joint_angular_kp_overrides)
    linear_kp_overrides = dict(config.fruiting_system.joint_linear_kp_overrides)
    joint_damping_ratio = config.fruiting_system.joint_damping_ratio

    if fix_to_apple and not vbd_only:
        gripper_weld = weld_grippers[0]
        if settled_checkpoint is not None:
            gripper_free = free_grippers[0]
            settled = build_fn(
                ranges,
                params,
                **_builder_kwargs(
                    config,
                    gripper=gripper_free,
                    per_env_grippers=free_grippers,
                    vbd_only=True,
                ),
            )
            settled.cable.state_0.body_q.assign(settled_checkpoint.body_q)
            settled.cable.state_1.body_q.assign(settled_checkpoint.body_q)
            pre_weld_tree_obs = _capture_pre_weld_tree_obs(settled)
            weld_kw = _builder_kwargs(
                config,
                gripper=gripper_weld,
                per_env_grippers=weld_grippers,
                vbd_only=False,
            )
            weld_kw["skip_ik_bootstrap"] = True
            weld_kw["defer_template_robot_bootstrap"] = True
            scene = build_fn(ranges, params, **weld_kw)
            seed_fix_to_apple_from_settled_body_q(
                welded_scene=scene,
                settled_body_q=settled_checkpoint.body_q,
                quiet_apple_proxy=True,
                per_env_ik=config.robot.per_env_ik,
                per_world_proxy_offsets=scene.per_world_proxy_offsets,
                ik_bootstrap_iterations=config.robot.ik_bootstrap_iterations,
                bootstrap_joint_q=config.robot.bootstrap_joint_q,
                per_world_bootstrap_joint_q=config.robot.per_world_bootstrap_joint_q,
            )
            ik_raw = getattr(scene, "settle_ik_envelope_results", None)
            ik_results = list(ik_raw) if ik_raw else None
            post_stab, post_ke = _apply_post_grasp_settle(scene)
            if collect_diag:
                stability_reports = list(post_stab)
                ke_decay_reports = list(post_ke)
        else:
            gripper_free = free_grippers[0]
            settled = build_fn(
                ranges,
                params,
                **_builder_kwargs(
                    config,
                    gripper=gripper_free,
                    per_env_grippers=free_grippers,
                    vbd_only=True,
                ),
            )
            _apply_joint_penalty_overrides(
                settled,
                angular_kd_overrides=angular_kd_overrides,
                linear_kd_overrides=linear_kd_overrides,
                angular_kp_overrides=angular_kp_overrides,
                linear_kp_overrides=linear_kp_overrides,
                joint_damping_ratio=joint_damping_ratio,
                per_env_params=params,
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
            settled_body_q = capture_body_q_numpy(settled.cable.state_0.body_q)
            pre_weld_tree_obs = _capture_pre_weld_tree_obs(settled)
            weld_kw = _builder_kwargs(
                config,
                gripper=gripper_weld,
                per_env_grippers=weld_grippers,
                vbd_only=False,
            )
            weld_kw["skip_ik_bootstrap"] = True
            weld_kw["defer_template_robot_bootstrap"] = True
            scene = build_fn(ranges, params, **weld_kw)
            seed_fix_to_apple_from_settled(
                welded_scene=scene,
                settled_scene=settled,
                quiet_apple_proxy=True,
                per_env_ik=config.robot.per_env_ik,
                per_world_proxy_offsets=scene.per_world_proxy_offsets,
                ik_bootstrap_iterations=config.robot.ik_bootstrap_iterations,
                bootstrap_joint_q=config.robot.bootstrap_joint_q,
                per_world_bootstrap_joint_q=config.robot.per_world_bootstrap_joint_q,
            )
            ik_raw = getattr(scene, "settle_ik_envelope_results", None)
            ik_results = list(ik_raw) if ik_raw else None
            post_stab, post_ke = _apply_post_grasp_settle(scene)
            if collect_diag and post_stab:
                # Prefer post-grasp residual motion when that phase ran.
                stability_reports = list(post_stab)
                ke_decay_reports = list(post_ke)
    else:
        active_grippers = weld_grippers if fix_to_apple else free_grippers
        scene = build_fn(
            ranges,
            params,
            **_builder_kwargs(
                config,
                gripper=active_grippers[0],
                per_env_grippers=active_grippers,
                vbd_only=vbd_only,
            ),
        )
        if settle_substeps > 0:
            _apply_joint_penalty_overrides(
                scene,
                angular_kd_overrides=angular_kd_overrides,
                linear_kd_overrides=linear_kd_overrides,
                angular_kp_overrides=angular_kp_overrides,
                linear_kp_overrides=linear_kp_overrides,
                joint_damping_ratio=joint_damping_ratio,
                per_env_params=params,
            )
            stability_reports, ke_decay_reports = _run_vbd_settle(
                scene,
                config=config,
                per_env_params=params,
                substeps=settle_substeps,
                sim_dt=sim_dt,
                viewer=viewer,
                collect_diagnostics=collect_diag,
            )

    (
        applied_angular_kd,
        applied_linear_kd,
        applied_angular_kp,
        applied_linear_kp,
    ) = _apply_joint_penalty_overrides(
        scene,
        angular_kd_overrides=angular_kd_overrides,
        linear_kd_overrides=linear_kd_overrides,
        angular_kp_overrides=angular_kp_overrides,
        linear_kp_overrides=linear_kp_overrides,
        joint_damping_ratio=joint_damping_ratio,
        per_env_params=params,
    )

    if not collect_diag:
        return BatchedHeterogeneousBuildResult(
            scene=scene,
            per_env_params=params,
            joint_angular_kd_overrides=applied_angular_kd,
            joint_linear_kd_overrides=applied_linear_kd,
            joint_angular_kp_overrides=applied_angular_kp,
            joint_linear_kp_overrides=applied_linear_kp,
            settled_body_q=settled_body_q,
            pre_weld_tree_obs=pre_weld_tree_obs,
        )

    return BatchedHeterogeneousBuildResult(
        scene=scene,
        per_env_params=params,
        joint_angular_kd_overrides=applied_angular_kd,
        joint_linear_kd_overrides=applied_linear_kd,
        joint_angular_kp_overrides=applied_angular_kp,
        joint_linear_kp_overrides=applied_linear_kp,
        settled_body_q=settled_body_q,
        pre_weld_tree_obs=pre_weld_tree_obs,
        settle_stability_reports=tuple(stability_reports),
        settle_ke_decay_reports=tuple(ke_decay_reports),
        ik_envelope_results=tuple(ik_results) if ik_results is not None else None,
    )


def _require_fr3_assets() -> None:
    if not fr3_robot.fr3_assets_available():
        raise FileNotFoundError(
            "Bundled FR3 assets missing; see assets/fr3/README.md"
        )


_GRIPPER_STRUCTURAL_FIELDS = (
    "mass",
    "shape",
    "cylinder_radius",
    "cylinder_half_height",
    "box_half_extents",
    "label",
    "fix_to_apple",
)


def _gripper_with_fix_mode(
    gripper: GripperProxyConfig,
    *,
    fix_to_apple: bool,
) -> GripperProxyConfig:
    if not fix_to_apple:
        return dataclasses.replace(
            gripper,
            fix_to_apple=False,
            robot_facing_weld=False,
            weld_direction=None,
            weld_reference_pos=None,
            weld_reference_quat=None,
            weld_reference_stem_dir=None,
            weld_proxy_offset_in_apple_frame=None,
        )
    robot_facing_weld = gripper.weld_direction is None
    return dataclasses.replace(
        gripper,
        fix_to_apple=True,
        robot_facing_weld=robot_facing_weld,
    )


def _normalize_per_env_grippers(
    base: GripperProxyConfig,
    per_env_grippers: Sequence[GripperProxyConfig] | None,
    *,
    num_envs: int,
    fix_to_apple: bool,
) -> tuple[GripperProxyConfig, ...]:
    raw = (
        tuple(per_env_grippers)
        if per_env_grippers is not None
        else tuple(base for _ in range(int(num_envs)))
    )
    if len(raw) != int(num_envs):
        raise ValueError(
            f"per_env_grippers length ({len(raw)}) must match num_envs ({num_envs})"
        )
    normalized = tuple(
        _gripper_with_fix_mode(item, fix_to_apple=bool(fix_to_apple))
        for item in raw
    )
    reference = normalized[0]
    for env_idx, item in enumerate(normalized[1:], start=1):
        mismatched = [
            name
            for name in _GRIPPER_STRUCTURAL_FIELDS
            if getattr(item, name) != getattr(reference, name)
        ]
        if mismatched:
            raise ValueError(
                "structural gripper mismatch at env "
                f"{env_idx}: {', '.join(mismatched)}"
            )
    return normalized


def _gripper_proxy(
    config: BatchedHeterogeneousCoupledSimConfig,
    *,
    fix_to_apple: bool | None = None,
) -> GripperProxyConfig:
    fix = config.robot.fix_to_apple if fix_to_apple is None else fix_to_apple
    return _gripper_with_fix_mode(config.robot.gripper, fix_to_apple=bool(fix))


def _builder_kwargs(
    config: BatchedHeterogeneousCoupledSimConfig,
    *,
    gripper: GripperProxyConfig,
    per_env_grippers: Sequence[GripperProxyConfig] | None = None,
    vbd_only: bool,
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
        "per_env_gripper_proxies": per_env_grippers,
        "vbd_only": vbd_only,
        "base_pos": scene_cfg.fruiting_base_pos,
        "robot_base_pos": robot_cfg.robot_base_pos,
        "stem_coupling_gain": fruiting.stem_coupling_gain,
        "stem_force_cap_N": fruiting.stem_force_cap_N,
        "stem_torque_cap_Nm": fruiting.stem_torque_cap_Nm,
        "mujoco_solver_kwargs": dict(mujoco.solver_kwargs),
        "mujoco_use_cpu": mujoco.use_cpu,
        "skip_ik_bootstrap": robot_cfg.skip_ik_bootstrap,
        "ik_bootstrap_iterations": robot_cfg.ik_bootstrap_iterations,
        "defer_template_robot_bootstrap": robot_cfg.defer_template_robot_bootstrap,
        "force_batched_layout": robot_cfg.force_batched_layout,
        "reuse_replicated_mujoco": bool(robot_cfg.reuse_replicated_mujoco),
    }
    return kw


def _matching_label_overrides(
    fruiting_fixed_joints: Sequence[tuple[int, str]],
    overrides: dict[str, float],
) -> dict[str, float]:
    """Keep override keys that match exactly one fixed-joint label in the template."""
    if not overrides:
        return {}
    keys = list(overrides.keys())
    used_keys: set[str] = set()
    for _joint_index, label in fruiting_fixed_joints:
        matching = [k for k in keys if k in label]
        if len(matching) == 1:
            used_keys.add(matching[0])
    return {k: overrides[k] for k in sorted(used_keys)}


def _apply_joint_angular_kd_overrides(
    scene: CoupledFruitingScene,
    kd_overrides: dict[str, float],
) -> dict[str, float]:
    layout = scene.layout
    if layout is None or not kd_overrides:
        return {}
    filtered = _matching_label_overrides(scene.cable.fruiting_fixed_joints, kd_overrides)
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


def _apply_joint_linear_kd_overrides(
    scene: CoupledFruitingScene,
    kd_overrides: dict[str, float],
) -> dict[str, float]:
    layout = scene.layout
    if layout is None or not kd_overrides:
        return {}
    filtered = _matching_label_overrides(scene.cable.fruiting_fixed_joints, kd_overrides)
    if not filtered:
        return {}
    set_fruiting_joint_linear_kd_batched(
        scene.cable.solver,
        scene.cable.fruiting_fixed_joints,
        filtered,
        num_envs=layout.num_envs,
        joints_per_world=layout.joints_per_world,
    )
    return dict(filtered)


def _apply_joint_angular_kp_overrides(
    scene: CoupledFruitingScene,
    kp_overrides: dict[str, float],
) -> dict[str, float]:
    layout = scene.layout
    if layout is None or not kp_overrides:
        return {}
    filtered = _matching_label_overrides(scene.cable.fruiting_fixed_joints, kp_overrides)
    if not filtered:
        return {}
    set_fruiting_joint_angular_kp_batched(
        scene.cable.solver,
        scene.cable.fruiting_fixed_joints,
        filtered,
        num_envs=layout.num_envs,
        joints_per_world=layout.joints_per_world,
    )
    return dict(filtered)


def _apply_joint_linear_kp_overrides(
    scene: CoupledFruitingScene,
    kp_overrides: dict[str, float],
) -> dict[str, float]:
    layout = scene.layout
    if layout is None or not kp_overrides:
        return {}
    filtered = _matching_label_overrides(scene.cable.fruiting_fixed_joints, kp_overrides)
    if not filtered:
        return {}
    set_fruiting_joint_linear_kp_batched(
        scene.cable.solver,
        scene.cable.fruiting_fixed_joints,
        filtered,
        num_envs=layout.num_envs,
        joints_per_world=layout.joints_per_world,
    )
    return dict(filtered)


def _apply_joint_penalty_overrides(
    scene: CoupledFruitingScene,
    *,
    angular_kd_overrides: dict[str, float],
    linear_kd_overrides: dict[str, float],
    angular_kp_overrides: dict[str, float],
    linear_kp_overrides: dict[str, float],
    joint_damping_ratio: float | None = None,
    per_env_params: Sequence[FruitingSystemParams] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    # kp first so weld stiffness is in place before kd patches.
    applied_angular_kp = _apply_joint_angular_kp_overrides(scene, angular_kp_overrides)
    applied_linear_kp = _apply_joint_linear_kp_overrides(scene, linear_kp_overrides)

    ang_kd = dict(angular_kd_overrides)
    lin_kd = dict(linear_kd_overrides)
    if joint_damping_ratio is not None:
        layout = scene.layout
        if layout is None:
            raise ValueError("joint_damping_ratio requires a batched scene layout")
        model = scene.cable.model
        body_mass = model.body_mass.numpy()
        body_inertia = model.body_inertia.numpy()
        joint_child = model.joint_child.numpy()
        joints = list(scene.cable.fruiting_fixed_joints)
        # Env-0 map is the reported "requested" base; per-env uses that env's m/I.
        ang_kd, lin_kd = joint_kd_from_damping_ratio(
            zeta=float(joint_damping_ratio),
            fruiting_fixed_joints=joints,
            body_mass=body_mass,
            body_inertia=body_inertia,
            joint_child=joint_child,
            angular_kp_by_role=angular_kp_overrides,
            linear_kp_by_role=linear_kp_overrides,
            body_offset=0,
        )
        if per_env_params is not None:
            per_env_ang: list[dict[str, float]] = []
            per_env_lin: list[dict[str, float]] = []
            for w, _params in enumerate(per_env_params):
                a, l = joint_kd_from_damping_ratio(
                    zeta=float(joint_damping_ratio),
                    fruiting_fixed_joints=joints,
                    body_mass=body_mass,
                    body_inertia=body_inertia,
                    joint_child=joint_child,
                    angular_kp_by_role=angular_kp_overrides,
                    linear_kp_by_role=linear_kp_overrides,
                    body_offset=int(w) * int(layout.bodies_per_world),
                )
                per_env_ang.append(a)
                per_env_lin.append(l)
            if per_env_ang and any(per_env_ang[0].values()):
                set_fruiting_joint_angular_kd_batched(
                    scene.cable.solver,
                    joints,
                    label_kd_per_env=per_env_ang,
                    num_envs=layout.num_envs,
                    joints_per_world=layout.joints_per_world,
                )
            if per_env_lin and any(per_env_lin[0].values()):
                set_fruiting_joint_linear_kd_batched(
                    scene.cable.solver,
                    joints,
                    label_kd_per_env=per_env_lin,
                    num_envs=layout.num_envs,
                    joints_per_world=layout.joints_per_world,
                )
            return ang_kd, lin_kd, applied_angular_kp, applied_linear_kp

    applied_angular_kd = _apply_joint_angular_kd_overrides(scene, ang_kd)
    applied_linear_kd = _apply_joint_linear_kd_overrides(scene, lin_kd)
    return applied_angular_kd, applied_linear_kd, applied_angular_kp, applied_linear_kp


def _rebootstrap_fr3_after_post_grasp_settle(
    scene: CoupledFruitingScene,
    *,
    config: BatchedHeterogeneousCoupledSimConfig,
) -> None:
    """Re-align FR3 TCP to the cable proxy after post-grasp VBD settle."""
    cable = scene.cable
    body_count = int(cable.model.body_count)
    align_proxy_body_q_prev_for_vbd(cable, tuple(range(body_count)))
    sync_model_body_q_rest_from_state(cable)

    if config.robot.per_world_bootstrap_joint_q is not None:
        _bootstrap_tcp_at_fixed_origin(
            scene, per_world_bootstrap_joint_q=config.robot.per_world_bootstrap_joint_q
        )
        return
    if config.robot.bootstrap_joint_q is not None:
        # Keep open-loop recorded joints after plant settle (no IK).
        _bootstrap_tcp_at_fixed_origin(
            scene, bootstrap_joint_q=config.robot.bootstrap_joint_q
        )
        return

    layout = getattr(scene, "layout", None)
    ik_iters = config.robot.ik_bootstrap_iterations
    if config.robot.per_env_ik and layout is not None and layout.num_envs > 1:
        _bootstrap_tcp_per_env(scene, layout, ik_iterations=ik_iters)
        prepare_batched_stem_harvest_arrays(scene, layout)
        return

    from apple_pick_sim.robot.fr3_robot.placement import IK_BOOTSTRAP_DEFAULT_ITERATIONS

    bootstrap_iters = (
        int(ik_iters) if ik_iters is not None else IK_BOOTSTRAP_DEFAULT_ITERATIONS
    )
    _bootstrap_tcp_at_fixed_origin(scene, ik_iterations=bootstrap_iters)
    if layout is not None:
        broadcast_joint_q_from_world0(scene, layout)


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
    quiet_every = scene_cfg.settle_quiet_every
    warn_settle_quiet_every_alignment(n, quiet_every)
    render_stride = _settle_render_stride(n) if viewer is not None else 1
    viewer_state = _SettleViewerState()

    for substep_idx in range(n):
        apply_settle_gravity_for_substep(
            scene,
            substep_idx,
            n,
            gravity_ramp=gravity_ramp,
        )
        scene.vbd_substep(h)
        if should_quiet_cable_bodies_at_settle_substep(substep_idx + 1, quiet_every):
            quiet_all_cable_bodies(scene.cable)
        if recorder is not None:
            recorder.record_substep(
                scene.cable,
                per_env_params,
                substep_idx,
                h,
                sample_every=int(diag.ke_sample_every),
            )
        if _should_render_settle_substep(substep_idx, n, render_stride):
            _maybe_render_settle(
                viewer,
                scene,
                float(substep_idx + 1) * h,
                config=config,
                viewer_state=viewer_state,
                frame_sleep_s=_SETTLE_RENDER_FRAME_DT_S,
            )

    if not collect_diagnostics:
        quiet_all_cable_bodies(scene.cable)
        return [], []

    # Measure residual motion before the final quiet so |v|_max is meaningful.
    stability_reports = settle_stability_reports_from_cable(
        scene.cable,
        per_env_params,
        max_branch_speed_m_s=float(scene_cfg.settle_max_speed_m_s),
        path_rtol=_SETTLE_PATH_RTOL,
    )
    ke_decay_reports: list[SettleKeDecayReport] = []
    if recorder is not None and diag is not None:
        ke_decay_reports = recorder.reports(config=diag.ke_analysis)
    quiet_all_cable_bodies(scene.cable)
    return stability_reports, ke_decay_reports


def _maybe_render_settle(
    viewer: Any | None,
    scene: CoupledFruitingScene,
    sim_time_s: float,
    *,
    config: BatchedHeterogeneousCoupledSimConfig | None = None,
    viewer_state: _SettleViewerState | None = None,
    frame_sleep_s: float = 0.0,
) -> None:
    if viewer is None:
        return
    is_running = getattr(viewer, "is_running", None)
    if is_running is not None and not is_running():
        return
    state = viewer_state if viewer_state is not None else _SettleViewerState()
    if not state.model_bound:
        if hasattr(viewer, "set_model"):
            viewer.set_model(scene.cable.model)
        if (
            config is not None
            and config.runtime.num_envs > 1
            and hasattr(viewer, "set_world_offsets")
        ):
            viewer.set_world_offsets(tuple(config.runtime.env_spacing))
        if hasattr(viewer, "hide_loading_splash"):
            viewer.hide_loading_splash()
        state.model_bound = True
    if hasattr(viewer, "begin_frame"):
        viewer.begin_frame(sim_time_s)
    if hasattr(viewer, "log_state"):
        viewer.log_state(scene.cable.state_0)
    if hasattr(viewer, "end_frame"):
        viewer.end_frame()
    if frame_sleep_s > 0.0:
        time.sleep(frame_sleep_s)


def print_per_env_params(
    params_list: Sequence[FruitingSystemParams],
    *,
    env_indices: Sequence[int] | None = None,
    heading: str | None = None,
) -> None:
    """Print per-env continuous θ (topology shared) for CLI diagnostics.

    When ``env_indices`` is set, only those worlds are printed (e.g. unstable settle envs).
    """
    if env_indices is None:
        worlds = list(range(len(params_list)))
        title = heading or (
            "Per-env fruiting params (topology shared, continuous θ differs):"
        )
    else:
        worlds = [int(i) for i in env_indices]
        title = heading or (
            f"Fruiting params for selected envs {worlds} "
            "(topology shared, continuous θ differs):"
        )
    if not worlds:
        return
    print(title)
    for w in worlds:
        p = params_list[w]
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
