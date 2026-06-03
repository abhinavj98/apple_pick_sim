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
)
from apple_pick_sim.coupled_fruiting.stem import _find_stem_apple_joint
from apple_pick_sim.sim_device import resolve_sim_device
from apple_pick_sim.sim_mujoco_device import resolve_mujoco_use_cpu
from apple_pick_sim.fruiting_system import (
    CoupledCableScene,
    FruitingSystemParams,
    GripperProxyConfig,
    example_collision_pipeline,
    generate_coupled_cable_scene,
    generate_mega_coupled_cable_scene,
    resolve_fruiting_base_pos,
    resolve_robot_base_pos,
)
from apple_pick_sim.coupled_fruiting.mega_scene import (
    MegaCoupledFruitingScene,
    mega_ghost_position_offsets_wp,
)
from apple_pick_sim.coupled_fruiting.explicit_load import apple_mass_kg_from_model
from apple_pick_sim.coupled_fruiting.proxy_coupling import ProxyBodyRegistry


def _cached_apple_mass_kg(cable: CoupledCableScene) -> float:
    if cable.apple_body is None:
        return 0.0
    return apple_mass_kg_from_model(cable.model, cable.apple_body)


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
) -> tuple[float, float, float]:
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
) -> CoupledFruitingScene:
    robot_state_0 = robot_model.state()
    robot_state_1 = robot_model.state()
    robot_control = robot_model.control()

    bootstrap_fn(cable, robot_model, tcp_body, robot_state_0)
    if mirror_welded_cable_after_bootstrap and cable.gripper_proxy_apple_joint is not None:
        mirror_tcp_to_welded_cable_after_bootstrap(
            cable,
            robot_model,
            tcp_body,
            robot_state_0,
            gravity=gravity_vec,
        )
    if init_mujoco_actuator_targets:
        fr3_robot.init_mujoco_actuator_targets_from_model(robot_model, robot_control)
    robot_state_1.joint_q.assign(robot_model.joint_q)
    robot_state_1.joint_qd.assign(robot_model.joint_qd)
    newton.eval_fk(robot_model, robot_model.joint_q, robot_model.joint_qd, robot_state_1)
    mj_solver._update_mjc_data(mj_solver.mj_data, robot_model, robot_state_0)

    proxy_registry = cable.proxy_registry(tcp_body)
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

    return CoupledFruitingScene(
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
    )


def build_placeholder_tcp_robot_model(
    *,
    gripper_cfg: GripperProxyConfig,
    device: str,
    mujoco_solver_kwargs: dict[str, Any] | None = None,
) -> tuple[newton.Model, int, newton.solvers.SolverMuJoCo]:
    hx, hy, hz = gripper_cfg.box_half_extents
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    # builder.add_ground_plane()

    tcp_body = builder.add_link(
        mass=gripper_cfg.mass,
        label=f"{gripper_cfg.label}_robot_tcp",
    )
    shape_cfg = builder.default_shape_cfg.copy()
    shape_cfg.density = 0.0
    builder.add_shape_box(body=tcp_body, hx=hx, hy=hy, hz=hz, cfg=shape_cfg)

    j_free = builder.add_joint_free(parent=-1, child=tcp_body)
    builder.add_articulation([j_free])

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
    enable_self_collisions: bool = True,
    gripper_proxy: GripperProxyConfig | None = None,
    cable_collision_pipeline: Any | None = None,
    mujoco_solver_kwargs: dict[str, Any] | None = None,
    stem_coupling_gain: float = DEFAULT_STEM_COUPLING_GAIN,
    stem_force_cap_N: float | None = DEFAULT_STEM_FORCE_CAP_N,
    stem_torque_cap_Nm: float | None = DEFAULT_STEM_TORQUE_CAP_NM,
    vbd_only: bool = False,
    mujoco_only: bool = False,
    mujoco_use_cpu: bool | None = None,
) -> CoupledFruitingScene:
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
    )

    newton.eval_fk(cable.model, cable.model.joint_q, cable.model.joint_qd, cable.state_0)

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
    base_pos: tuple[float, float, float] | None = None,
    device: str | None = None,
    omit: Any | None = None,
    enable_self_collisions: bool = True,
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
) -> CoupledFruitingScene:
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
        base_pos=resolve_fruiting_base_pos(ranges, (0.0, 0.2, 1.0), override=base_pos),
        device=device,
        omit=omit,
        enable_self_collisions=enable_self_collisions,
        gripper_proxy=gripper_proxy,
    )
    newton.eval_fk(cable.model, cable.model.joint_q, cable.model.joint_qd, cable.state_0)

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
    )
    scene.fr3_root_world_pos = _fr3_root_world_pos(
        ranges,
        proxy_bq,
        robot_base_pos=robot_base_pos,
    )
    return scene


def build_mega_coupled_fruiting_fr3(
    ranges: dict,
    seed: int,
    *,
    base_pos: tuple[float, float, float] | None = None,
    instance_spacing: tuple[float, float, float] = (0.0, 1.5, 0.0),
    stiffness_epsilon: float | None = 0.02,
    params_list: Sequence[FruitingSystemParams] | None = None,
    nominal_index: int = 0,
    device: str | None = None,
    omit: Any | None = None,
    enable_self_collisions: bool = True,
    gripper_proxy: GripperProxyConfig | None = None,
    cable_collision_pipeline: Any | None = None,
    mujoco_solver_kwargs: dict[str, Any] | None = None,
    usd_path: str | Path | None = None,
    ik_bootstrap_iterations: int = 96,
    anchor_robot_root_at_world_origin: bool = False,
    mujoco_use_cpu: bool | None = None,
    robot_kinematic_mode: bool = True,
    stem_coupling_gain: float = DEFAULT_STEM_COUPLING_GAIN,
    stem_force_cap_N: float | None = DEFAULT_STEM_FORCE_CAP_N,
    stem_torque_cap_Nm: float | None = DEFAULT_STEM_TORQUE_CAP_NM,
    robot_base_pos: tuple[float, float, float] | None = None,
) -> MegaCoupledFruitingScene:
    """One FR3 + mega VBD plant (fd_ghost: offset mirror all proxies, harvest nominal only)."""
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

    mega_kw: dict[str, Any] = dict(
        base_pos=resolve_fruiting_base_pos(ranges, (0.0, 0.2, 1.0), override=base_pos),
        instance_spacing=instance_spacing,
        device=device,
        enable_self_collisions=enable_self_collisions,
        gripper_proxy=gripper_proxy,
    )
    if params_list is not None:
        from apple_pick_sim.fruiting_system.mega import MegaCoupledCableScene

        cable = MegaCoupledCableScene.build(params_list, **mega_kw)
    else:
        cable = generate_mega_coupled_cable_scene(
            ranges,
            seed,
            params_list=None,
            stiffness_epsilon=stiffness_epsilon,
            omit=omit,
            **mega_kw,
        )

    newton.eval_fk(cable.model, cable.model.joint_q, cable.model.joint_qd, cable.state_0)

    pipe = (
        cable_collision_pipeline
        if cable_collision_pipeline is not None
        else example_collision_pipeline(cable.model, args=None)
    )
    gravity_vec = wp.vec3(0.0, 0.0, -9.81)

    mj_kw = dict(DEFAULT_FR3_MUJOCO_SOLVER_KWARGS)
    if mujoco_solver_kwargs:
        mj_kw.update(mujoco_solver_kwargs)
    mj_kw["use_mujoco_cpu"] = use_mujoco_cpu

    nom_cable = cable.as_single_instance_coupled(nominal_index)
    proxy_bq = nom_cable.state_0.body_q.numpy().reshape(-1, 7)[nom_cable.gripper_proxy_body]
    root_xform = _robot_root_xform(
        ranges,
        proxy_bq,
        robot_base_pos=robot_base_pos,
        anchor_robot_root_at_world_origin=anchor_robot_root_at_world_origin,
    )

    robot_model, tcp_body, mj_solver = fr3_robot.build_fr3_robot_model_from_usd(
        device=device,
        usd_path=usd_path,
        root_xform=root_xform,
        mujoco_solver_kwargs=mj_kw,
    )

    robot_state_0 = robot_model.state()
    robot_state_1 = robot_model.state()
    robot_control = robot_model.control()

    bootstrap_articulated_tcp_from_proxy(
        nom_cable,
        robot_model,
        tcp_body,
        robot_state_0,
        ik_iterations=ik_bootstrap_iterations,
    )
    fr3_robot.init_mujoco_actuator_targets_from_model(robot_model, robot_control)
    robot_state_1.joint_q.assign(robot_model.joint_q)
    robot_state_1.joint_qd.assign(robot_model.joint_qd)
    newton.eval_fk(robot_model, robot_model.joint_q, robot_model.joint_qd, robot_state_1)
    mj_solver._update_mjc_data(mj_solver.mj_data, robot_model, robot_state_0)

    proxy_ids = cable.all_gripper_proxy_body_ids()
    ghost_registry = ProxyBodyRegistry.from_repeated_robot(tcp_body, proxy_ids)
    harvest_registry = ProxyBodyRegistry.from_mapping(
        {tcp_body: nom_cable.gripper_proxy_body}
    )
    position_offsets_wp = mega_ghost_position_offsets_wp(
        cable, nominal_index=nominal_index, device=device
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
        _find_stem_apple_joint(nom_cable) if nom_cable.apple_body is not None else None
    )

    scene = MegaCoupledFruitingScene(
        cable=cable,
        cable_collision_pipeline=pipe,
        ghost_registry=ghost_registry,
        harvest_registry=harvest_registry,
        position_offsets_wp=position_offsets_wp,
        nominal_index=nominal_index,
        robot_model=robot_model,
        tcp_body_index=tcp_body,
        mj_solver=mj_solver,
        robot_state_0=robot_state_0,
        robot_state_1=robot_state_1,
        robot_control=robot_control,
        mj_contacts=mj_contacts,
        proxy_forces=proxy_forces,
        coupling_forces_cache=coupling_cache,
        gravity_vec=gravity_vec,
        use_mujoco_contacts=use_mc,
        robot_disable_contacts=robot_dc,
        robot_kinematic_mode=robot_kinematic_mode,
        qd_synced=wp.empty_like(cable.state_0.body_qd),
        stem_apple_joint_index=stem_joint,
        stem_coupling_gain=stem_coupling_gain,
        stem_force_cap_N=stem_force_cap_N,
        stem_torque_cap_Nm=stem_torque_cap_Nm,
        apple_mass_kg=_cached_apple_mass_kg(nom_cable),
    )
    scene.fr3_root_world_pos = _fr3_root_world_pos(
        ranges,
        proxy_bq,
        robot_base_pos=robot_base_pos,
    )
    return scene
