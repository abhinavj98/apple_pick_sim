"""Build coupled cable + robot scenes (placeholder TCP and FR3)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import warp as wp

import newton

from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.coupled_fruiting.bootstrap import (
    bootstrap_articulated_tcp_from_proxy,
    bootstrap_tcp_joint_from_proxy,
    mirror_tcp_to_welded_cable_after_bootstrap,
)
from apple_pick_sim.coupled_fruiting.scene import (
    DEFAULT_FR3_MUJOCO_SOLVER_KWARGS,
    DEFAULT_MUJOCO_SOLVER_KWARGS,
    DEFAULT_STEM_COUPLING_GAIN,
    DEFAULT_STEM_FORCE_CAP_N,
    DEFAULT_STEM_TORQUE_CAP_NM,
    CoupledFruitingScene,
    init_robot_mujoco_step_buffers,
)
from apple_pick_sim.coupled_fruiting.explicit_load import apple_mass_kg_from_model
from apple_pick_sim.coupled_fruiting.stem import _find_stem_apple_joint
from apple_pick_sim.sim_device import resolve_sim_device
from apple_pick_sim.sim_mujoco_device import resolve_mujoco_use_cpu
from apple_pick_sim.fruiting_system import (
    CoupledCableScene,
    FruitingSystemParams,
    GripperProxyConfig,
    example_collision_pipeline,
    generate_coupled_cable_scene,
    resolve_fruiting_base_pos,
    resolve_robot_base_pos,
)
from apple_pick_sim.coupled_fruiting.proxy_coupling import ProxyBodyRegistry
from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.coupled_fruiting.batched_build import (
    build_heterogeneous_coupled_cable_scene,
    build_replicated_coupled_cable_scene,
    build_replicated_robot_model,
)


def _validate_batched_options(
    *,
    num_envs: int,
    fix_to_apple: bool,
    vbd_only: bool,
    mujoco_only: bool,
) -> None:
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")
    if num_envs == 1:
        return
    if mujoco_only:
        raise ValueError("num_envs > 1 does not support mujoco_only")


def _batched_proxy_registry(layout: BatchedEnvLayout) -> ProxyBodyRegistry:
    mapping = {
        layout.tcp_body_indices[w]: layout.proxy_body_indices[w]
        for w in range(layout.num_envs)
    }
    return ProxyBodyRegistry.from_mapping(mapping)


def _mj_kw_for_batch(mj_kw: dict[str, Any], num_envs: int) -> dict[str, Any]:
    out = dict(mj_kw)
    if num_envs > 1:
        out["separate_worlds"] = True
    return out


def _cached_apple_mass_kg(cable: CoupledCableScene) -> float:
    """Return the apple body mass in kg, or 0.0 when no apple is present."""
    if cable.apple_body is None:
        return 0.0
    return apple_mass_kg_from_model(cable.model, cable.apple_body)


def _resolve_stem_harvest_explicit_apple_weight(
    gripper_proxy: GripperProxyConfig | None,
    *,
    override: bool | None = None,
) -> bool:
    """Enable explicit apple-weight correction only for prescribed (welded) apples."""
    if override is True:
        if gripper_proxy is not None and not gripper_proxy.fix_to_apple:
            raise ValueError(
                "stem_harvest_explicit_apple_weight=True with fix_to_apple=False is not "
                "supported: VBD already integrates apple gravity; explicit correction "
                "double-counts apple weight at the TCP."
            )
        return True
    if override is False:
        return False
    return bool(gripper_proxy is not None and gripper_proxy.fix_to_apple)


def _robot_root_xform(
    ranges: dict,
    proxy_body_q7: Any,
    *,
    robot_base_pos: tuple[float, float, float] | None = None,
    anchor_robot_root_at_world_origin: bool = False,
) -> wp.transform:
    """World root transform for the FR3 USD import."""
    resolved = resolve_robot_base_pos(ranges, override=robot_base_pos)
    if resolved is not None:
        return wp.transform(wp.vec3(*resolved), wp.quat_identity())
    if anchor_robot_root_at_world_origin:
        return wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
    return fr3_robot.placement_xform_for_proxy(proxy_body_q7)


def _fr3_root_world_pos(
    ranges: dict,
    proxy_body_q7: Any,
    *,
    robot_base_pos: tuple[float, float, float] | None = None,
    robot_base_from_proxy: bool = False,
) -> tuple[float, float, float]:
    """Resolve FR3 base world translation from ranges, override, or proxy pose."""
    if robot_base_from_proxy:
        return fr3_robot.root_world_translation_for_proxy(proxy_body_q7)
    resolved = resolve_robot_base_pos(ranges, override=robot_base_pos)
    if resolved is not None:
        return resolved
    return fr3_robot.root_world_translation_for_proxy(proxy_body_q7)


def _assemble_coupled_robot_scene(
    cable: CoupledCableScene,
    *,
    device: str,
    pipe: Any,
    gravity_vec: wp.vec3,
    robot_model: newton.Model,
    tcp_body: int,
    mj_solver: newton.solvers.SolverMuJoCo,
    mj_kw: dict[str, Any],
    bootstrap_fn: Callable[..., None],
    mujoco_only: bool,
    stem_coupling_gain: float,
    stem_force_cap_N: float | None,
    stem_torque_cap_Nm: float | None,
    init_mujoco_actuator_targets: bool = False,
    qd_synced: wp.array | None = None,
    mirror_welded_cable_after_bootstrap: bool = False,
    skip_ik_bootstrap: bool = False,
    stem_harvest_explicit_apple_weight: bool | None = None,
    proxy_registry: ProxyBodyRegistry | None = None,
    layout: BatchedEnvLayout | None = None,
    env_spacing: tuple[float, float, float] | None = None,
    ik_template_robot_model: Any | None = None,
) -> CoupledFruitingScene:
    """Wire cable scene, robot model, bootstrap, and coupling buffers into a ``CoupledFruitingScene``."""
    robot_state_0 = robot_model.state()
    robot_state_1 = robot_model.state()
    robot_control = robot_model.control()

    if skip_ik_bootstrap:
        newton.eval_fk(
            robot_model,
            robot_model.joint_q,
            robot_model.joint_qd,
            robot_state_0,
        )
    else:
        bootstrap_fn(cable, robot_model, tcp_body, robot_state_0)
    if mirror_welded_cable_after_bootstrap and cable.gripper_proxy_apple_joint is not None:
        mirror_tcp_to_welded_cable_after_bootstrap(
            cable,
            robot_model,
            tcp_body,
            robot_state_0,
            gravity=gravity_vec,
        )
    if init_mujoco_actuator_targets and not skip_ik_bootstrap:
        fr3_robot.init_mujoco_actuator_targets_from_model(robot_model, robot_control)

    proxy_registry = (
        proxy_registry if proxy_registry is not None else cable.proxy_registry(tcp_body)
    )
    proxy_forces = wp.zeros(robot_model.body_count, dtype=wp.spatial_vector, device=device)
    coupling_cache = wp.zeros_like(proxy_forces)

    use_mc = bool(mj_kw["use_mujoco_contacts"])
    robot_dc = bool(mj_kw.get("disable_contacts", False))
    fr3_robot.sync_robot_gravity_to_mujoco(robot_model, mj_solver)
    if use_mc:
        mj_contacts = newton.Contacts(mj_solver.get_max_contact_count(), 0)
    else:
        mj_contacts = robot_model.contacts()

    stem_joint = (
        _find_stem_apple_joint(cable) if cable.apple_body is not None else None
    )
    grip_cfg = cable.gripper_proxy_config
    explicit_apple_weight = _resolve_stem_harvest_explicit_apple_weight(
        grip_cfg,
        override=stem_harvest_explicit_apple_weight,
    )

    scene = CoupledFruitingScene(
        cable=cable,
        cable_collision_pipeline=pipe,
        vbd_only=False,
        mujoco_only=mujoco_only,
        robot_model=robot_model,
        tcp_body_index=tcp_body,
        mj_solver=mj_solver,
        robot_state_0=robot_state_0,
        robot_state_1=robot_state_1,
        robot_control=robot_control,
        mj_contacts=mj_contacts,
        proxy_registry=proxy_registry,
        proxy_forces=proxy_forces,
        coupling_forces_cache=coupling_cache,
        gravity_vec=gravity_vec,
        use_mujoco_contacts=use_mc,
        robot_disable_contacts=robot_dc,
        stem_apple_joint_index=stem_joint,
        stem_coupling_gain=stem_coupling_gain,
        stem_force_cap_N=stem_force_cap_N,
        stem_torque_cap_Nm=stem_torque_cap_Nm,
        apple_mass_kg=_cached_apple_mass_kg(cable),
        qd_synced=qd_synced,
        stem_harvest_explicit_apple_weight=explicit_apple_weight,
        layout=layout,
        env_spacing=env_spacing,
        ik_template_robot_model=ik_template_robot_model,
    )
    init_robot_mujoco_step_buffers(scene)
    return scene


def build_placeholder_tcp_robot_builder(
    *,
    gripper_cfg: GripperProxyConfig,
) -> tuple[newton.ModelBuilder, int]:
    """Populate placeholder box-TCP robot on a builder without ``finalize``."""
    hx, hy, hz = gripper_cfg.box_half_extents
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    tcp_body = builder.add_link(
        mass=gripper_cfg.mass,
        label=f"{gripper_cfg.label}_robot_tcp",
    )
    shape_cfg = builder.default_shape_cfg.copy()
    shape_cfg.density = 0.0
    builder.add_shape_box(body=tcp_body, hx=hx, hy=hy, hz=hz, cfg=shape_cfg)
    j_free = builder.add_joint_free(parent=-1, child=tcp_body)
    builder.add_articulation([j_free])
    return builder, tcp_body


def build_placeholder_tcp_robot_model(
    *,
    gripper_cfg: GripperProxyConfig,
    device: str,
    mujoco_solver_kwargs: dict[str, Any] | None = None,
) -> tuple[newton.Model, int, newton.solvers.SolverMuJoCo]:
    """Build a free-floating box TCP robot model with MuJoCo solver (no articulated arm)."""
    builder, tcp_body = build_placeholder_tcp_robot_builder(gripper_cfg=gripper_cfg)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    mj_kw = dict(DEFAULT_MUJOCO_SOLVER_KWARGS)
    mj_kw.update(mujoco_solver_kwargs or {})
    mj_kw["use_mujoco_cpu"] = resolve_mujoco_use_cpu(device, mj_kw.get("use_mujoco_cpu"))

    solver = newton.solvers.SolverMuJoCo(
        model,
        njmax=80,
        nconmax=80,
        **mj_kw,
    )
    return model, tcp_body, solver


def build_coupled_fruiting_placeholder(
    ranges: dict,
    seed: int,
    *,
    base_pos: tuple[float, float, float] | None = None,
    device: str | None = None,
    omit: Any | None = None,
    enable_self_collisions: bool = False,
    gripper_proxy: GripperProxyConfig | None = None,
    cable_collision_pipeline: Any | None = None,
    mujoco_solver_kwargs: dict[str, Any] | None = None,
    stem_coupling_gain: float = DEFAULT_STEM_COUPLING_GAIN,
    stem_force_cap_N: float | None = DEFAULT_STEM_FORCE_CAP_N,
    stem_torque_cap_Nm: float | None = DEFAULT_STEM_TORQUE_CAP_NM,
    vbd_only: bool = False,
    mujoco_only: bool = False,
    mujoco_use_cpu: bool | None = None,
    robot_base_pos: tuple[float, float, float] | None = None,
) -> CoupledFruitingScene:
    """Build cable + placeholder box-TCP scene for VBD-only or full staggered coupling."""
    if vbd_only and mujoco_only:
        raise ValueError("vbd_only and mujoco_only are mutually exclusive")
    device = resolve_sim_device(device)
    use_mujoco_cpu = resolve_mujoco_use_cpu(device, mujoco_use_cpu)

    if gripper_proxy is None:
        gripper_proxy = GripperProxyConfig()

    cable = generate_coupled_cable_scene(
        ranges,
        seed,
        base_pos=resolve_fruiting_base_pos(ranges, (0.0, 0.2, 1.0), override=base_pos),
        device=device,
        omit=omit,
        enable_self_collisions=enable_self_collisions,
        gripper_proxy=gripper_proxy,
        robot_base_pos=resolve_robot_base_pos(ranges, override=robot_base_pos),
    )

    newton.eval_fk(cable.model, cable.model.joint_q, cable.model.joint_qd, cable.state_0)
    wp.synchronize()

    pipe = (
        cable_collision_pipeline
        if cable_collision_pipeline is not None
        else example_collision_pipeline(cable.model, args=None)
    )

    gravity_vec = wp.vec3(0.0, 0.0, -9.81)

    if vbd_only:
        return CoupledFruitingScene(
            cable=cable,
            cable_collision_pipeline=pipe,
            vbd_only=True,
            gravity_vec=gravity_vec,
        )

    grip_cfg = cable.gripper_proxy_config

    mj_kw = dict(DEFAULT_MUJOCO_SOLVER_KWARGS)
    if mujoco_solver_kwargs:
        mj_kw.update(mujoco_solver_kwargs)
    mj_kw["use_mujoco_cpu"] = use_mujoco_cpu

    robot_model, tcp_body, mj_solver = build_placeholder_tcp_robot_model(
        gripper_cfg=grip_cfg,
        device=device,
        mujoco_solver_kwargs=mj_kw,
    )

    qd_synced = wp.empty_like(cable.state_0.body_qd)

    return _assemble_coupled_robot_scene(
        cable,
        device=device,
        pipe=pipe,
        gravity_vec=gravity_vec,
        robot_model=robot_model,
        tcp_body=tcp_body,
        mj_solver=mj_solver,
        mj_kw=mj_kw,
        bootstrap_fn=bootstrap_tcp_joint_from_proxy,
        mujoco_only=mujoco_only,
        stem_coupling_gain=stem_coupling_gain,
        stem_force_cap_N=stem_force_cap_N,
        stem_torque_cap_Nm=stem_torque_cap_Nm,
        qd_synced=qd_synced,
    )


def build_coupled_fruiting_fr3(
    ranges: dict,
    seed: int,
    *,
    params: FruitingSystemParams | None = None,
    base_pos: tuple[float, float, float] | None = None,
    device: str | None = None,
    omit: Any | None = None,
    enable_self_collisions: bool = False,
    gripper_proxy: GripperProxyConfig | None = None,
    cable_collision_pipeline: Any | None = None,
    mujoco_solver_kwargs: dict[str, Any] | None = None,
    usd_path: str | Path | None = None,
    stem_coupling_gain: float = DEFAULT_STEM_COUPLING_GAIN,
    stem_force_cap_N: float | None = DEFAULT_STEM_FORCE_CAP_N,
    stem_torque_cap_Nm: float | None = DEFAULT_STEM_TORQUE_CAP_NM,
    vbd_only: bool = False,
    mujoco_only: bool = False,
    ik_bootstrap_iterations: int = 96,
    mujoco_use_cpu: bool | None = None,
    robot_base_pos: tuple[float, float, float] | None = None,
    robot_base_from_proxy: bool = False,
    skip_ik_bootstrap: bool = False,
) -> CoupledFruitingScene:
    """Build cable + FR3 arm scene; IK-bootstrap TCP to gripper proxy at construction."""
    if vbd_only and mujoco_only:
        raise ValueError("vbd_only and mujoco_only are mutually exclusive")
    if not fr3_robot.fr3_assets_available():
        raise FileNotFoundError(
            "Bundled FR3 assets missing; see assets/fr3/README.md"
        )
    device = resolve_sim_device(device)
    use_mujoco_cpu = resolve_mujoco_use_cpu(device, mujoco_use_cpu)

    if gripper_proxy is None:
        gripper_proxy = GripperProxyConfig(
            mass=fr3_robot.EE_MASS_KG,
            box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
        )

    cable = generate_coupled_cable_scene(
        ranges,
        seed,
        params=params,
        base_pos=resolve_fruiting_base_pos(ranges, (0.0, 0.2, 1.0), override=base_pos),
        device=device,
        omit=omit,
        enable_self_collisions=enable_self_collisions,
        gripper_proxy=gripper_proxy,
        robot_base_pos=resolve_robot_base_pos(ranges, override=robot_base_pos),
    )
    newton.eval_fk(cable.model, cable.model.joint_q, cable.model.joint_qd, cable.state_0)
    wp.synchronize()

    pipe = (
        cable_collision_pipeline
        if cable_collision_pipeline is not None
        else example_collision_pipeline(cable.model, args=None)
    )
    gravity_vec = wp.vec3(0.0, 0.0, -9.81)

    if vbd_only:
        return CoupledFruitingScene(
            cable=cable,
            cable_collision_pipeline=pipe,
            vbd_only=True,
            gravity_vec=gravity_vec,
        )

    mj_kw = dict(DEFAULT_FR3_MUJOCO_SOLVER_KWARGS)
    if mujoco_solver_kwargs:
        mj_kw.update(mujoco_solver_kwargs)
    mj_kw["use_mujoco_cpu"] = use_mujoco_cpu

    proxy_bq = cable.state_0.body_q.numpy().reshape(-1, 7)[cable.gripper_proxy_body]
    if robot_base_from_proxy:
        root_xform = fr3_robot.placement_xform_for_proxy(proxy_bq)
    else:
        root_xform = _robot_root_xform(
            ranges,
            proxy_bq,
            robot_base_pos=robot_base_pos,
        )

    robot_model, tcp_body, mj_solver = fr3_robot.build_fr3_robot_model_from_usd(
        device=device,
        usd_path=usd_path,
        root_xform=root_xform,
        mujoco_solver_kwargs=mj_kw,
    )

    def _bootstrap(
        cable_scene: CoupledCableScene,
        model: newton.Model,
        tcp_idx: int,
        state_0: Any,
    ) -> None:
        """IK-align articulated FR3 TCP with the cable gripper proxy."""
        bootstrap_articulated_tcp_from_proxy(
            cable_scene,
            model,
            tcp_idx,
            state_0,
            ik_iterations=ik_bootstrap_iterations,
        )

    scene = _assemble_coupled_robot_scene(
        cable,
        device=device,
        pipe=pipe,
        gravity_vec=gravity_vec,
        robot_model=robot_model,
        tcp_body=tcp_body,
        mj_solver=mj_solver,
        mj_kw=mj_kw,
        bootstrap_fn=_bootstrap,
        mujoco_only=mujoco_only,
        stem_coupling_gain=stem_coupling_gain,
        stem_force_cap_N=stem_force_cap_N,
        stem_torque_cap_Nm=stem_torque_cap_Nm,
        init_mujoco_actuator_targets=True,
        qd_synced=wp.empty_like(cable.state_0.body_qd),
        mirror_welded_cable_after_bootstrap=mujoco_only,
        skip_ik_bootstrap=skip_ik_bootstrap,
    )
    scene.fr3_root_world_pos = _fr3_root_world_pos(
        ranges,
        proxy_bq,
        robot_base_pos=robot_base_pos if not robot_base_from_proxy else None,
        robot_base_from_proxy=robot_base_from_proxy,
    )
    return scene


def build_batched_coupled_fruiting_placeholder(
    ranges: dict,
    seed: int,
    *,
    num_envs: int = 4,
    env_spacing: tuple[float, float, float] = (2.5, 2.5, 0.0),
    base_pos: tuple[float, float, float] | None = None,
    device: str | None = None,
    omit: Any | None = None,
    enable_self_collisions: bool = False,
    gripper_proxy: GripperProxyConfig | None = None,
    cable_collision_pipeline: Any | None = None,
    mujoco_solver_kwargs: dict[str, Any] | None = None,
    stem_coupling_gain: float = DEFAULT_STEM_COUPLING_GAIN,
    stem_force_cap_N: float | None = DEFAULT_STEM_FORCE_CAP_N,
    stem_torque_cap_Nm: float | None = DEFAULT_STEM_TORQUE_CAP_NM,
    mujoco_use_cpu: bool | None = None,
    robot_base_pos: tuple[float, float, float] | None = None,
    skip_ik_bootstrap: bool = True,
    vbd_only: bool = False,
    defer_template_robot_bootstrap: bool = False,
) -> CoupledFruitingScene:
    """Build ``num_envs`` identical placeholder-TCP coupled scenes via ``replicate()``."""
    _ = defer_template_robot_bootstrap  # no-op; placeholder has no template IK to defer
    if gripper_proxy is None:
        gripper_proxy = GripperProxyConfig()
    fix = bool(gripper_proxy.fix_to_apple)
    _validate_batched_options(
        num_envs=num_envs, fix_to_apple=fix, vbd_only=vbd_only, mujoco_only=False
    )
    if num_envs == 1:
        return build_coupled_fruiting_placeholder(
            ranges,
            seed,
            base_pos=base_pos,
            device=device,
            omit=omit,
            enable_self_collisions=enable_self_collisions,
            gripper_proxy=gripper_proxy,
            cable_collision_pipeline=cable_collision_pipeline,
            mujoco_solver_kwargs=mujoco_solver_kwargs,
            stem_coupling_gain=stem_coupling_gain,
            stem_force_cap_N=stem_force_cap_N,
            stem_torque_cap_Nm=stem_torque_cap_Nm,
            mujoco_use_cpu=mujoco_use_cpu,
            robot_base_pos=robot_base_pos,
            vbd_only=vbd_only,
        )

    device = resolve_sim_device(device)
    resolved_base = resolve_fruiting_base_pos(ranges, (0.0, 0.2, 1.0), override=base_pos)
    resolved_robot_base = resolve_robot_base_pos(ranges, override=robot_base_pos)

    template_cable = generate_coupled_cable_scene(
        ranges,
        seed,
        base_pos=resolved_base,
        device=device,
        omit=omit,
        enable_self_collisions=enable_self_collisions,
        gripper_proxy=gripper_proxy,
        robot_base_pos=resolved_robot_base,
    )
    newton.eval_fk(
        template_cable.model,
        template_cable.model.joint_q,
        template_cable.model.joint_qd,
        template_cable.state_0,
    )
    wp.synchronize()

    cable = build_replicated_coupled_cable_scene(
        template_cable,
        num_envs=num_envs,
        env_spacing=env_spacing,
        device=device,
        enable_self_collisions=enable_self_collisions,
        base_pos=resolved_base,
        robot_base_pos=resolved_robot_base,
    )
    pipe = (
        cable_collision_pipeline
        if cable_collision_pipeline is not None
        else example_collision_pipeline(cable.model, args=None)
    )
    gravity_vec = wp.vec3(0.0, 0.0, -9.81)
    if vbd_only:
        return CoupledFruitingScene(
            cable=cable,
            cable_collision_pipeline=pipe,
            vbd_only=True,
            gravity_vec=gravity_vec,
        )

    grip_cfg = template_cable.gripper_proxy_config
    tpl_mj_kw = dict(DEFAULT_MUJOCO_SOLVER_KWARGS)
    if mujoco_solver_kwargs:
        tpl_mj_kw.update(mujoco_solver_kwargs)
    tpl_mj_kw["use_mujoco_cpu"] = resolve_mujoco_use_cpu(device, mujoco_use_cpu)
    mj_kw = _mj_kw_for_batch(dict(tpl_mj_kw), num_envs)

    tpl_robot_model, tpl_tcp, _ = build_placeholder_tcp_robot_model(
        gripper_cfg=grip_cfg,
        device=device,
        mujoco_solver_kwargs=tpl_mj_kw,
    )
    tpl_state = tpl_robot_model.state()
    bootstrap_tcp_joint_from_proxy(template_cable, tpl_robot_model, tpl_tcp, tpl_state)

    robot_model, template_tcp, mj_solver = build_replicated_robot_model(
        tpl_robot_model,
        tpl_tcp,
        num_envs=num_envs,
        env_spacing=env_spacing,
        device=device,
        template_builder_factory=lambda: build_placeholder_tcp_robot_builder(
            gripper_cfg=grip_cfg
        ),
        mujoco_solver_kwargs=mj_kw,
    )
    layout = BatchedEnvLayout.from_template_scene(
        template_cable,
        cable.model,
        robot_model,
        template_tcp_body=template_tcp,
        env_spacing=(0.0, 0.0, 0.0),
    )
    registry = _batched_proxy_registry(layout)
    tcp_world0 = layout.tcp_body_indices[0]

    pipe = (
        cable_collision_pipeline
        if cable_collision_pipeline is not None
        else example_collision_pipeline(cable.model, args=None)
    )
    gravity_vec = wp.vec3(0.0, 0.0, -9.81)
    scene = _assemble_coupled_robot_scene(
        cable,
        device=device,
        pipe=pipe,
        gravity_vec=gravity_vec,
        robot_model=robot_model,
        tcp_body=tcp_world0,
        mj_solver=mj_solver,
        mj_kw=mj_kw,
        bootstrap_fn=bootstrap_tcp_joint_from_proxy,
        mujoco_only=False,
        stem_coupling_gain=stem_coupling_gain,
        stem_force_cap_N=stem_force_cap_N,
        stem_torque_cap_Nm=stem_torque_cap_Nm,
        qd_synced=wp.empty_like(cable.state_0.body_qd),
        proxy_registry=registry,
        layout=layout,
        env_spacing=env_spacing,
        skip_ik_bootstrap=skip_ik_bootstrap,
        ik_template_robot_model=tpl_robot_model,
    )
    newton.eval_fk(
        robot_model,
        robot_model.joint_q,
        robot_model.joint_qd,
        scene.robot_state_0,
    )
    init_robot_mujoco_step_buffers(scene)
    return scene


def build_batched_coupled_fruiting_fr3(
    ranges: dict,
    seed: int,
    *,
    num_envs: int = 4,
    env_spacing: tuple[float, float, float] = (2.5, 2.5, 0.0),
    params: FruitingSystemParams | None = None,
    base_pos: tuple[float, float, float] | None = None,
    device: str | None = None,
    omit: Any | None = None,
    enable_self_collisions: bool = False,
    gripper_proxy: GripperProxyConfig | None = None,
    cable_collision_pipeline: Any | None = None,
    mujoco_solver_kwargs: dict[str, Any] | None = None,
    usd_path: str | Path | None = None,
    stem_coupling_gain: float = DEFAULT_STEM_COUPLING_GAIN,
    stem_force_cap_N: float | None = DEFAULT_STEM_FORCE_CAP_N,
    stem_torque_cap_Nm: float | None = DEFAULT_STEM_TORQUE_CAP_NM,
    ik_bootstrap_iterations: int = 96,
    mujoco_use_cpu: bool | None = None,
    robot_base_pos: tuple[float, float, float] | None = None,
    robot_base_from_proxy: bool = False,
    skip_ik_bootstrap: bool = True,
    vbd_only: bool = False,
    defer_template_robot_bootstrap: bool = False,
) -> CoupledFruitingScene:
    """Build ``num_envs`` identical FR3 coupled scenes via ``replicate()``."""
    if not fr3_robot.fr3_assets_available():
        raise FileNotFoundError(
            "Bundled FR3 assets missing; see assets/fr3/README.md"
        )
    if gripper_proxy is None:
        gripper_proxy = GripperProxyConfig(
            mass=fr3_robot.EE_MASS_KG,
            box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
        )
    fix = bool(gripper_proxy.fix_to_apple)
    _validate_batched_options(
        num_envs=num_envs, fix_to_apple=fix, vbd_only=vbd_only, mujoco_only=False
    )
    if num_envs == 1:
        return build_coupled_fruiting_fr3(
            ranges,
            seed,
            params=params,
            base_pos=base_pos,
            device=device,
            omit=omit,
            enable_self_collisions=enable_self_collisions,
            gripper_proxy=gripper_proxy,
            cable_collision_pipeline=cable_collision_pipeline,
            mujoco_solver_kwargs=mujoco_solver_kwargs,
            usd_path=usd_path,
            stem_coupling_gain=stem_coupling_gain,
            stem_force_cap_N=stem_force_cap_N,
            stem_torque_cap_Nm=stem_torque_cap_Nm,
            ik_bootstrap_iterations=ik_bootstrap_iterations,
            mujoco_use_cpu=mujoco_use_cpu,
            robot_base_pos=robot_base_pos,
            robot_base_from_proxy=robot_base_from_proxy,
            skip_ik_bootstrap=skip_ik_bootstrap,
            vbd_only=vbd_only,
        )

    device = resolve_sim_device(device)
    resolved_base = resolve_fruiting_base_pos(ranges, (0.0, 0.2, 1.0), override=base_pos)
    resolved_robot_base = resolve_robot_base_pos(ranges, override=robot_base_pos)

    template_cable = generate_coupled_cable_scene(
        ranges,
        seed,
        params=params,
        base_pos=resolved_base,
        device=device,
        omit=omit,
        enable_self_collisions=enable_self_collisions,
        gripper_proxy=gripper_proxy,
        robot_base_pos=resolved_robot_base,
    )
    newton.eval_fk(
        template_cable.model,
        template_cable.model.joint_q,
        template_cable.model.joint_qd,
        template_cable.state_0,
    )
    wp.synchronize()

    cable = build_replicated_coupled_cable_scene(
        template_cable,
        num_envs=num_envs,
        env_spacing=env_spacing,
        device=device,
        enable_self_collisions=enable_self_collisions,
        base_pos=resolved_base,
        robot_base_pos=resolved_robot_base,
    )
    pipe = (
        cable_collision_pipeline
        if cable_collision_pipeline is not None
        else example_collision_pipeline(cable.model, args=None)
    )
    gravity_vec = wp.vec3(0.0, 0.0, -9.81)
    if vbd_only:
        return CoupledFruitingScene(
            cable=cable,
            cable_collision_pipeline=pipe,
            vbd_only=True,
            gravity_vec=gravity_vec,
        )

    proxy_bq = template_cable.state_0.body_q.numpy().reshape(-1, 7)[
        template_cable.gripper_proxy_body
    ]
    if robot_base_from_proxy:
        root_xform = fr3_robot.placement_xform_for_proxy(proxy_bq)
    else:
        root_xform = _robot_root_xform(ranges, proxy_bq, robot_base_pos=robot_base_pos)

    mj_kw = dict(DEFAULT_FR3_MUJOCO_SOLVER_KWARGS)
    if mujoco_solver_kwargs:
        mj_kw.update(mujoco_solver_kwargs)
    mj_kw["use_mujoco_cpu"] = resolve_mujoco_use_cpu(device, mujoco_use_cpu)
    tpl_mj_kw = dict(mj_kw)
    batched_mj_kw = _mj_kw_for_batch(dict(mj_kw), num_envs)

    tpl_robot_model, tpl_tcp, _ = fr3_robot.build_fr3_robot_model_from_usd(
        device=device,
        usd_path=usd_path,
        root_xform=root_xform,
        mujoco_solver_kwargs=tpl_mj_kw,
    )
    tpl_state = tpl_robot_model.state()
    if not defer_template_robot_bootstrap:
        bootstrap_articulated_tcp_from_proxy(
            template_cable,
            tpl_robot_model,
            tpl_tcp,
            tpl_state,
            ik_iterations=ik_bootstrap_iterations,
        )

    def _robot_builder_factory() -> tuple[newton.ModelBuilder, int]:
        return fr3_robot.build_fr3_robot_builder(
            usd_path=usd_path,
            root_xform=root_xform,
        )

    robot_model, template_tcp, mj_solver = build_replicated_robot_model(
        tpl_robot_model,
        tpl_tcp,
        num_envs=num_envs,
        env_spacing=env_spacing,
        device=device,
        template_builder_factory=_robot_builder_factory,
        mujoco_solver_kwargs=batched_mj_kw,
    )
    layout = BatchedEnvLayout.from_template_scene(
        template_cable,
        cable.model,
        robot_model,
        template_tcp_body=template_tcp,
        env_spacing=(0.0, 0.0, 0.0),
    )
    registry = _batched_proxy_registry(layout)
    tcp_world0 = layout.tcp_body_indices[0]

    pipe = (
        cable_collision_pipeline
        if cable_collision_pipeline is not None
        else example_collision_pipeline(cable.model, args=None)
    )
    gravity_vec = wp.vec3(0.0, 0.0, -9.81)

    scene = _assemble_coupled_robot_scene(
        cable,
        device=device,
        pipe=pipe,
        gravity_vec=gravity_vec,
        robot_model=robot_model,
        tcp_body=tcp_world0,
        mj_solver=mj_solver,
        mj_kw=batched_mj_kw,
        bootstrap_fn=bootstrap_articulated_tcp_from_proxy,
        mujoco_only=False,
        stem_coupling_gain=stem_coupling_gain,
        stem_force_cap_N=stem_force_cap_N,
        stem_torque_cap_Nm=stem_torque_cap_Nm,
        init_mujoco_actuator_targets=True,
        qd_synced=wp.empty_like(cable.state_0.body_qd),
        proxy_registry=registry,
        layout=layout,
        env_spacing=env_spacing,
        skip_ik_bootstrap=skip_ik_bootstrap,
        ik_template_robot_model=tpl_robot_model,
    )
    newton.eval_fk(
        robot_model,
        robot_model.joint_q,
        robot_model.joint_qd,
        scene.robot_state_0,
    )
    init_robot_mujoco_step_buffers(scene)
    scene.fr3_root_world_pos = _fr3_root_world_pos(
        ranges,
        proxy_bq,
        robot_base_pos=robot_base_pos if not robot_base_from_proxy else None,
        robot_base_from_proxy=robot_base_from_proxy,
    )
    return scene


def build_heterogeneous_coupled_fruiting_placeholder(
    ranges: dict,
    params_list: Sequence[FruitingSystemParams],
    *,
    env_spacing: tuple[float, float, float] = (2.5, 2.5, 0.0),
    base_pos: tuple[float, float, float] | None = None,
    device: str | None = None,
    omit: Any | None = None,
    enable_self_collisions: bool = False,
    gripper_proxy: GripperProxyConfig | None = None,
    cable_collision_pipeline: Any | None = None,
    mujoco_solver_kwargs: dict[str, Any] | None = None,
    stem_coupling_gain: float = DEFAULT_STEM_COUPLING_GAIN,
    stem_force_cap_N: float | None = DEFAULT_STEM_FORCE_CAP_N,
    stem_torque_cap_Nm: float | None = DEFAULT_STEM_TORQUE_CAP_NM,
    mujoco_use_cpu: bool | None = None,
    robot_base_pos: tuple[float, float, float] | None = None,
    skip_ik_bootstrap: bool = True,
    vbd_only: bool = False,
    defer_template_robot_bootstrap: bool = False,
) -> CoupledFruitingScene:
    """Build heterogeneous placeholder-TCP coupled scenes via ``add_world``."""
    _ = defer_template_robot_bootstrap
    _ = omit
    params = list(params_list)
    num_envs = len(params)
    if gripper_proxy is None:
        gripper_proxy = GripperProxyConfig()
    fix = bool(gripper_proxy.fix_to_apple)
    _validate_batched_options(
        num_envs=num_envs, fix_to_apple=fix, vbd_only=vbd_only, mujoco_only=False
    )
    if num_envs == 1:
        return build_coupled_fruiting_placeholder(
            ranges,
            0,
            params=params[0],
            base_pos=base_pos,
            device=device,
            enable_self_collisions=enable_self_collisions,
            gripper_proxy=gripper_proxy,
            cable_collision_pipeline=cable_collision_pipeline,
            mujoco_solver_kwargs=mujoco_solver_kwargs,
            stem_coupling_gain=stem_coupling_gain,
            stem_force_cap_N=stem_force_cap_N,
            stem_torque_cap_Nm=stem_torque_cap_Nm,
            mujoco_use_cpu=mujoco_use_cpu,
            robot_base_pos=robot_base_pos,
            vbd_only=vbd_only,
        )

    device = resolve_sim_device(device)
    resolved_base = resolve_fruiting_base_pos(ranges, (0.0, 0.2, 1.0), override=base_pos)
    resolved_robot_base = resolve_robot_base_pos(ranges, override=robot_base_pos)

    cable, per_world_offsets = build_heterogeneous_coupled_cable_scene(
        params,
        env_spacing=env_spacing,
        device=device,
        enable_self_collisions=enable_self_collisions,
        base_pos=resolved_base,
        robot_base_pos=resolved_robot_base,
        gripper_proxy=gripper_proxy,
    )
    pipe = (
        cable_collision_pipeline
        if cable_collision_pipeline is not None
        else example_collision_pipeline(cable.model, args=None)
    )
    gravity_vec = wp.vec3(0.0, 0.0, -9.81)
    if vbd_only:
        return CoupledFruitingScene(
            cable=cable,
            cable_collision_pipeline=pipe,
            vbd_only=True,
            gravity_vec=gravity_vec,
            env_spacing=env_spacing,
            per_env_params=params,
            per_world_proxy_offsets=per_world_offsets,
        )

    grip_cfg = cable.gripper_proxy_config
    tpl_mj_kw = dict(DEFAULT_MUJOCO_SOLVER_KWARGS)
    if mujoco_solver_kwargs:
        tpl_mj_kw.update(mujoco_solver_kwargs)
    tpl_mj_kw["use_mujoco_cpu"] = resolve_mujoco_use_cpu(device, mujoco_use_cpu)
    mj_kw = _mj_kw_for_batch(dict(tpl_mj_kw), num_envs)

    tpl_robot_model, tpl_tcp, _ = build_placeholder_tcp_robot_model(
        gripper_cfg=grip_cfg,
        device=device,
        mujoco_solver_kwargs=tpl_mj_kw,
    )
    tpl_state = tpl_robot_model.state()
    bootstrap_tcp_joint_from_proxy(cable, tpl_robot_model, tpl_tcp, tpl_state)

    robot_model, template_tcp, mj_solver = build_replicated_robot_model(
        tpl_robot_model,
        tpl_tcp,
        num_envs=num_envs,
        env_spacing=env_spacing,
        device=device,
        template_builder_factory=lambda: build_placeholder_tcp_robot_builder(
            gripper_cfg=grip_cfg
        ),
        mujoco_solver_kwargs=mj_kw,
    )
    layout = BatchedEnvLayout.from_template_scene(
        cable,
        cable.model,
        robot_model,
        template_tcp_body=template_tcp,
        env_spacing=(0.0, 0.0, 0.0),
    )
    registry = _batched_proxy_registry(layout)
    tcp_world0 = layout.tcp_body_indices[0]

    scene = _assemble_coupled_robot_scene(
        cable,
        device=device,
        pipe=pipe,
        gravity_vec=gravity_vec,
        robot_model=robot_model,
        tcp_body=tcp_world0,
        mj_solver=mj_solver,
        mj_kw=mj_kw,
        bootstrap_fn=bootstrap_tcp_joint_from_proxy,
        mujoco_only=False,
        stem_coupling_gain=stem_coupling_gain,
        stem_force_cap_N=stem_force_cap_N,
        stem_torque_cap_Nm=stem_torque_cap_Nm,
        qd_synced=wp.empty_like(cable.state_0.body_qd),
        proxy_registry=registry,
        layout=layout,
        env_spacing=env_spacing,
        skip_ik_bootstrap=skip_ik_bootstrap,
        ik_template_robot_model=tpl_robot_model,
    )
    scene.per_env_params = params
    scene.per_world_proxy_offsets = per_world_offsets
    newton.eval_fk(
        robot_model,
        robot_model.joint_q,
        robot_model.joint_qd,
        scene.robot_state_0,
    )
    init_robot_mujoco_step_buffers(scene)
    return scene


def build_heterogeneous_coupled_fruiting_fr3(
    ranges: dict,
    params_list: Sequence[FruitingSystemParams],
    *,
    env_spacing: tuple[float, float, float] = (2.5, 2.5, 0.0),
    base_pos: tuple[float, float, float] | None = None,
    device: str | None = None,
    omit: Any | None = None,
    enable_self_collisions: bool = False,
    gripper_proxy: GripperProxyConfig | None = None,
    cable_collision_pipeline: Any | None = None,
    mujoco_solver_kwargs: dict[str, Any] | None = None,
    usd_path: str | Path | None = None,
    stem_coupling_gain: float = DEFAULT_STEM_COUPLING_GAIN,
    stem_force_cap_N: float | None = DEFAULT_STEM_FORCE_CAP_N,
    stem_torque_cap_Nm: float | None = DEFAULT_STEM_TORQUE_CAP_NM,
    ik_bootstrap_iterations: int = 96,
    mujoco_use_cpu: bool | None = None,
    robot_base_pos: tuple[float, float, float] | None = None,
    robot_base_from_proxy: bool = False,
    skip_ik_bootstrap: bool = True,
    vbd_only: bool = False,
    defer_template_robot_bootstrap: bool = False,
) -> CoupledFruitingScene:
    """Build heterogeneous FR3 coupled scenes via ``add_world`` (uniform topology)."""
    if not fr3_robot.fr3_assets_available():
        raise FileNotFoundError(
            "Bundled FR3 assets missing; see assets/fr3/README.md"
        )
    if robot_base_from_proxy:
        raise ValueError(
            "robot_base_from_proxy is not supported for heterogeneous batched builds "
            "(each env's proxy starts at a different position)"
        )
    _ = omit
    params = list(params_list)
    num_envs = len(params)
    if gripper_proxy is None:
        gripper_proxy = GripperProxyConfig(
            mass=fr3_robot.EE_MASS_KG,
            box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
        )
    fix = bool(gripper_proxy.fix_to_apple)
    _validate_batched_options(
        num_envs=num_envs, fix_to_apple=fix, vbd_only=vbd_only, mujoco_only=False
    )
    if num_envs == 1:
        return build_coupled_fruiting_fr3(
            ranges,
            0,
            params=params[0],
            base_pos=base_pos,
            device=device,
            enable_self_collisions=enable_self_collisions,
            gripper_proxy=gripper_proxy,
            cable_collision_pipeline=cable_collision_pipeline,
            mujoco_solver_kwargs=mujoco_solver_kwargs,
            usd_path=usd_path,
            stem_coupling_gain=stem_coupling_gain,
            stem_force_cap_N=stem_force_cap_N,
            stem_torque_cap_Nm=stem_torque_cap_Nm,
            ik_bootstrap_iterations=ik_bootstrap_iterations,
            mujoco_use_cpu=mujoco_use_cpu,
            robot_base_pos=robot_base_pos,
            skip_ik_bootstrap=skip_ik_bootstrap,
            vbd_only=vbd_only,
        )

    device = resolve_sim_device(device)
    resolved_base = resolve_fruiting_base_pos(ranges, (0.0, 0.2, 1.0), override=base_pos)
    resolved_robot_base = resolve_robot_base_pos(ranges, override=robot_base_pos)

    cable, per_world_offsets = build_heterogeneous_coupled_cable_scene(
        params,
        env_spacing=env_spacing,
        device=device,
        enable_self_collisions=enable_self_collisions,
        base_pos=resolved_base,
        robot_base_pos=resolved_robot_base,
        gripper_proxy=gripper_proxy,
    )
    pipe = (
        cable_collision_pipeline
        if cable_collision_pipeline is not None
        else example_collision_pipeline(cable.model, args=None)
    )
    gravity_vec = wp.vec3(0.0, 0.0, -9.81)
    if vbd_only:
        return CoupledFruitingScene(
            cable=cable,
            cable_collision_pipeline=pipe,
            vbd_only=True,
            gravity_vec=gravity_vec,
            env_spacing=env_spacing,
            per_env_params=params,
            per_world_proxy_offsets=per_world_offsets,
        )

    proxy_bq = cable.state_0.body_q.numpy().reshape(-1, 7)[cable.gripper_proxy_body]
    root_xform = _robot_root_xform(ranges, proxy_bq, robot_base_pos=robot_base_pos)

    mj_kw = dict(DEFAULT_FR3_MUJOCO_SOLVER_KWARGS)
    if mujoco_solver_kwargs:
        mj_kw.update(mujoco_solver_kwargs)
    mj_kw["use_mujoco_cpu"] = resolve_mujoco_use_cpu(device, mujoco_use_cpu)
    tpl_mj_kw = dict(mj_kw)
    batched_mj_kw = _mj_kw_for_batch(dict(mj_kw), num_envs)

    tpl_robot_model, tpl_tcp, _ = fr3_robot.build_fr3_robot_model_from_usd(
        device=device,
        usd_path=usd_path,
        root_xform=root_xform,
        mujoco_solver_kwargs=tpl_mj_kw,
    )
    tpl_state = tpl_robot_model.state()
    if not defer_template_robot_bootstrap:
        bootstrap_articulated_tcp_from_proxy(
            cable,
            tpl_robot_model,
            tpl_tcp,
            tpl_state,
            ik_iterations=ik_bootstrap_iterations,
        )

    def _robot_builder_factory() -> tuple[newton.ModelBuilder, int]:
        return fr3_robot.build_fr3_robot_builder(
            usd_path=usd_path,
            root_xform=root_xform,
        )

    robot_model, template_tcp, mj_solver = build_replicated_robot_model(
        tpl_robot_model,
        tpl_tcp,
        num_envs=num_envs,
        env_spacing=env_spacing,
        device=device,
        template_builder_factory=_robot_builder_factory,
        mujoco_solver_kwargs=batched_mj_kw,
    )
    layout = BatchedEnvLayout.from_template_scene(
        cable,
        cable.model,
        robot_model,
        template_tcp_body=template_tcp,
        env_spacing=(0.0, 0.0, 0.0),
    )
    registry = _batched_proxy_registry(layout)
    tcp_world0 = layout.tcp_body_indices[0]

    scene = _assemble_coupled_robot_scene(
        cable,
        device=device,
        pipe=pipe,
        gravity_vec=gravity_vec,
        robot_model=robot_model,
        tcp_body=tcp_world0,
        mj_solver=mj_solver,
        mj_kw=batched_mj_kw,
        bootstrap_fn=bootstrap_articulated_tcp_from_proxy,
        mujoco_only=False,
        stem_coupling_gain=stem_coupling_gain,
        stem_force_cap_N=stem_force_cap_N,
        stem_torque_cap_Nm=stem_torque_cap_Nm,
        init_mujoco_actuator_targets=True,
        qd_synced=wp.empty_like(cable.state_0.body_qd),
        proxy_registry=registry,
        layout=layout,
        env_spacing=env_spacing,
        skip_ik_bootstrap=skip_ik_bootstrap,
        ik_template_robot_model=tpl_robot_model,
    )
    scene.per_env_params = params
    scene.per_world_proxy_offsets = per_world_offsets
    newton.eval_fk(
        robot_model,
        robot_model.joint_q,
        robot_model.joint_qd,
        scene.robot_state_0,
    )
    init_robot_mujoco_step_buffers(scene)
    scene.fr3_root_world_pos = _fr3_root_world_pos(
        ranges,
        proxy_bq,
        robot_base_pos=robot_base_pos,
        robot_base_from_proxy=False,
    )
    return scene
