"""Replicate helpers for batched coupled cable/robot builds."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import warp as wp

import newton

from apple_pick_sim.fruiting_system.build import (
    _FruitingChainArtifacts,
    _apply_all_chain_collision_filters,
    _register_fruiting_articulations,
    _new_fruiting_builder,
    _scene_states_from_model,
)
from apple_pick_sim.fruiting_system.coupled import (
    CoupledCableScene,
    CoupledCablePopulateResult,
    _populate_coupled_cable_builder,
)
from apple_pick_sim.fruiting_system.params import FruitingSystemParams, GripperProxyConfig
from apple_pick_sim.coupled_fruiting.proxy_coupling import align_proxy_body_q_prev_for_vbd


def _prepare_cable_template_builder_for_replicate(
    builder: newton.ModelBuilder,
    artifacts: _FruitingChainArtifacts,
    *,
    enable_self_collisions: bool,
    proxy_free_joint: int | None,
) -> None:
    """Add articulations and collision filters on the template before ``replicate()``."""
    proxy_joints = (int(proxy_free_joint),) if proxy_free_joint is not None else ()
    _register_fruiting_articulations(
        builder,
        all_joints=artifacts.all_joints,
        chain_bodies=artifacts.chain_bodies,
        enable_self_collisions=enable_self_collisions,
        seg_bodies=artifacts.seg_bodies,
        apple_body=artifacts.apple_body,
        gripper_proxy_joints=proxy_joints,
        world_root_joints=artifacts.world_root_joints,
    )


def build_replicated_coupled_cable_scene(
    template_cable: CoupledCableScene,
    *,
    num_envs: int,
    env_spacing: tuple[float, float, float],
    device: str,
    enable_self_collisions: bool,
    base_pos: tuple[float, float, float],
    robot_base_pos: tuple[float, float, float] | None,
) -> CoupledCableScene:
    """Replicate a cable template into ``num_envs`` VBD worlds and run ``eval_fk``."""
    if num_envs < 2:
        raise ValueError("build_replicated_coupled_cable_scene requires num_envs >= 2")

    tpl_builder = _new_fruiting_builder()
    populated = _populate_coupled_cable_builder(
        tpl_builder,
        template_cable.params,
        base_pos,
        gripper_proxy=template_cable.gripper_proxy_config,
        robot_base_pos=robot_base_pos,
    )
    _prepare_cable_template_builder_for_replicate(
        tpl_builder,
        populated.artifacts,
        enable_self_collisions=enable_self_collisions,
        proxy_free_joint=populated.proxy_free_joint,
    )
    bodies_per_world = int(tpl_builder.body_count)

    outer = _new_fruiting_builder()
    outer.replicate(tpl_builder, num_envs, spacing=(0.0, 0.0, 0.0))
    outer.color()
    model = outer.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    if bool(template_cable.gripper_proxy_config.fix_to_apple):
        from apple_pick_sim.fruiting_system.build import prescribe_body_vbd_on_model

        tpl_apple = template_cable.apple_body
        tpl_proxy = template_cable.gripper_proxy_body
        if tpl_apple is not None and tpl_proxy is not None:
            prescribed: list[int] = []
            for w in range(num_envs):
                off = w * bodies_per_world
                prescribed.append(off + int(tpl_apple))
                prescribed.append(off + int(tpl_proxy))
            prescribe_body_vbd_on_model(model, *prescribed)

    state_0, state_1, control, solver = _scene_states_from_model(model)
    tpl = template_cable
    scene = CoupledCableScene(
        model=model,
        state_0=state_0,
        state_1=state_1,
        control=control,
        solver=solver,
        params=tpl.params,
        primary_bodies=tpl.primary_bodies,
        secondary_bodies=tpl.secondary_bodies,
        spur_bodies=tpl.spur_bodies,
        stem_bodies=tpl.stem_bodies,
        apple_body=tpl.apple_body,
        fruiting_fixed_joints=tpl.fruiting_fixed_joints,
        cable_joint_indices=tpl.cable_joint_indices,
        gripper_proxy_body=tpl.gripper_proxy_body,
        gripper_proxy_config=tpl.gripper_proxy_config,
        gripper_proxy_apple_joint=tpl.gripper_proxy_apple_joint,
        gripper_proxy_offset_in_apple_frame=tpl.gripper_proxy_offset_in_apple_frame,
        gripper_proxy_vis_offset=tpl.gripper_proxy_vis_offset,
    )
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_1)
    wp.synchronize()
    tpl_proxy_ids = tuple(
        int(tpl.gripper_proxy_body) + w * bodies_per_world for w in range(num_envs)
    )
    align_proxy_body_q_prev_for_vbd(scene, tpl_proxy_ids)
    return scene


def _rod_segments_match(a: FruitingSystemParams, b: FruitingSystemParams) -> bool:
    for seg in ("primary", "secondary", "spur", "stem"):
        ra = getattr(a, seg)
        rb = getattr(b, seg)
        if (ra is None) != (rb is None):
            return False
        if ra is not None and rb is not None and ra.num_segments != rb.num_segments:
            return False
    if (a.apple_radius is None) != (b.apple_radius is None):
        return False
    return True


def _assert_uniform_topology(params_list: list[FruitingSystemParams]) -> None:
    if not params_list:
        raise ValueError("params_list must be non-empty")
    topo0 = params_list[0]
    for w, params_w in enumerate(params_list[1:], start=1):
        if not _rod_segments_match(topo0, params_w):
            raise ValueError(
                f"heterogeneous build topology mismatch at env {w}: "
                "segment enablement or num_segments differs across envs"
            )


def _assert_uniform_world_entity_gaps(model: newton.Model, num_envs: int) -> int:
    """Return bodies_per_world after verifying uniform body/joint/shape counts."""
    world_count = int(model.world_count)
    if world_count != num_envs:
        raise ValueError(
            f"expected world_count={num_envs} after heterogeneous build, got {world_count}"
        )

    def _uniform_per_world(arr: wp.array, label: str) -> int:
        starts = arr.numpy()
        per = int(starts[1] - starts[0])
        for w in range(world_count):
            gap = int(starts[w + 1] - starts[w])
            if gap != per:
                raise ValueError(
                    f"heterogeneous build produced non-uniform world {label} counts "
                    f"(world {w} gap {gap} != {per})"
                )
        return per

    bodies_per = _uniform_per_world(model.body_world_start, "body")
    _uniform_per_world(model.joint_world_start, "joint")
    _uniform_per_world(model.shape_world_start, "shape")
    return bodies_per


def build_heterogeneous_coupled_cable_scene(
    params_list: list[FruitingSystemParams],
    *,
    env_spacing: tuple[float, float, float],
    device: str,
    enable_self_collisions: bool,
    base_pos: tuple[float, float, float],
    robot_base_pos: tuple[float, float, float] | None,
    gripper_proxy: GripperProxyConfig,
) -> tuple[CoupledCableScene, tuple[tuple[float, float, float, float, float, float, float] | None, ...]]:
    """Build ``len(params_list)`` heterogeneous VBD worlds via ``add_world`` (co-located)."""
    del env_spacing  # viewer-only; physics worlds share the origin
    num_envs = len(params_list)
    if num_envs < 1:
        raise ValueError("build_heterogeneous_coupled_cable_scene requires num_envs >= 1")
    _assert_uniform_topology(params_list)

    outer = _new_fruiting_builder()
    populate_results: list[CoupledCablePopulateResult] = []
    for params_w in params_list:
        sub = _new_fruiting_builder()
        pop = _populate_coupled_cable_builder(
            sub,
            params_w,
            base_pos,
            gripper_proxy=gripper_proxy,
            robot_base_pos=robot_base_pos,
        )
        _prepare_cable_template_builder_for_replicate(
            sub,
            pop.artifacts,
            enable_self_collisions=enable_self_collisions,
            proxy_free_joint=pop.proxy_free_joint,
        )
        outer.add_world(sub)
        populate_results.append(pop)

    outer.color()
    model = outer.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    bodies_per_world = _assert_uniform_world_entity_gaps(model, num_envs)
    tpl = populate_results[0]

    if gripper_proxy.fix_to_apple and tpl.artifacts.apple_body is not None:
        from apple_pick_sim.fruiting_system.build import prescribe_body_vbd_on_model

        prescribed: list[int] = []
        for w in range(num_envs):
            off = w * bodies_per_world
            prescribed.append(off + int(tpl.artifacts.apple_body))
            prescribed.append(off + int(tpl.proxy_body))
        prescribe_body_vbd_on_model(model, *prescribed)

    state_0, state_1, control, solver = _scene_states_from_model(model)
    scene = CoupledCableScene(
        model=model,
        state_0=state_0,
        state_1=state_1,
        control=control,
        solver=solver,
        params=params_list[0],
        primary_bodies=tpl.artifacts.seg_bodies.get("primary", []),
        secondary_bodies=tpl.artifacts.seg_bodies.get("secondary", []),
        spur_bodies=tpl.artifacts.seg_bodies.get("spur", []),
        stem_bodies=tpl.artifacts.seg_bodies.get("stem", []),
        apple_body=tpl.artifacts.apple_body,
        fruiting_fixed_joints=tuple(tpl.artifacts.fruiting_fixed_joints),
        cable_joint_indices=tuple(tpl.artifacts.cable_joint_indices),
        gripper_proxy_body=tpl.proxy_body,
        gripper_proxy_config=gripper_proxy,
        gripper_proxy_apple_joint=tpl.proxy_apple_joint,
        gripper_proxy_offset_in_apple_frame=tpl.proxy_offset_in_apple,
        gripper_proxy_vis_offset=tpl.vis_offset,
    )
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_1)
    wp.synchronize()
    proxy_ids = tuple(
        int(tpl.proxy_body) + w * bodies_per_world for w in range(num_envs)
    )
    align_proxy_body_q_prev_for_vbd(scene, proxy_ids)

    per_world_offsets = tuple(pop.proxy_offset_in_apple for pop in populate_results)
    return scene, per_world_offsets


def broadcast_settled_cable_state_to_batched_worlds(
    settled_cable: CoupledCableScene,
    welded_cable: CoupledCableScene,
    layout: Any,
    env_spacing: tuple[float, float, float],
) -> None:
    """Legacy: copy a single-world settled cable state into every replicated world.

    Used only when ``settled_cable.model.world_count == 1`` and the welded scene
    has ``num_envs > 1``. The canonical batched path settles all N worlds in parallel
    and copies world *i* → world *i* directly (see ``seed_fix_to_apple_from_settled``).
    """
    tpl_bq = settled_cable.state_0.body_q.numpy().reshape(-1, 7)
    tpl_bqd = settled_cable.state_0.body_qd.numpy().reshape(-1, 6)
    tpl_n = int(tpl_bq.shape[0])
    bodies_per = int(layout.bodies_per_world)
    world_offsets = newton.utils.compute_world_offsets(
        int(layout.num_envs),
        tuple(float(v) for v in env_spacing),
        up_axis=newton.Axis.Z,
    )

    for state in (welded_cable.state_0, welded_cable.state_1):
        bq = state.body_q.numpy().reshape(-1, 7).copy()
        bqd = state.body_qd.numpy().reshape(-1, 6).copy()
        for w in range(int(layout.num_envs)):
            off = w * bodies_per
            shift = np.asarray(world_offsets[w], dtype=np.float32).reshape(3)
            bq[off : off + tpl_n, :3] = tpl_bq[:tpl_n, :3] + shift
            bq[off : off + tpl_n, 3:] = tpl_bq[:tpl_n, 3:]
            bqd[off : off + tpl_n] = tpl_bqd[:tpl_n]
        state.body_q.assign(bq.ravel())
        state.body_qd.assign(bqd.ravel())


def build_replicated_robot_model(
    template_model: newton.Model,
    template_tcp: int,
    *,
    num_envs: int,
    env_spacing: tuple[float, float, float],
    device: str,
    template_builder_factory: Callable[[], tuple[newton.ModelBuilder, int]],
    mujoco_solver_kwargs: dict[str, Any],
) -> tuple[newton.Model, int, newton.solvers.SolverMuJoCo]:
    """Replicate robot template builder; return batched model and template TCP body index.

    Robot worlds are co-located (``spacing=(0, 0, 0)``) to match the batched cable
    replicate. ``env_spacing`` is retained for API compatibility but applies only to
    viewer offsets via :attr:`~apple_pick_sim.coupled_fruiting.scene.CoupledFruitingScene.env_spacing`.
    """
    del env_spacing
    tpl_builder, _ = template_builder_factory()
    outer = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(outer)
    outer.replicate(tpl_builder, num_envs, spacing=(0.0, 0.0, 0.0))
    model = outer.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))

    from apple_pick_sim.sim_mujoco_device import resolve_mujoco_use_cpu

    mj_kw = dict(mujoco_solver_kwargs)
    mj_kw["separate_worlds"] = True
    mj_kw["use_mujoco_cpu"] = resolve_mujoco_use_cpu(device, mj_kw.get("use_mujoco_cpu"))

    solver = newton.solvers.SolverMuJoCo(
        model,
        njmax=max(200, 80 * num_envs),
        nconmax=max(200, 80 * num_envs),
        **mj_kw,
    )
    _broadcast_robot_state_from_template(template_model, model)
    return model, int(template_tcp), solver


def _broadcast_robot_state_from_template(
    template_model: newton.Model,
    batched_model: newton.Model,
) -> None:
    """Copy template robot ``joint_q`` / ``joint_qd`` into every world."""
    num_envs = int(batched_model.world_count)
    jcs = batched_model.joint_coord_world_start.numpy()
    coord_per = int(jcs[1] - jcs[0])
    dps = batched_model.joint_dof_world_start.numpy()
    dof_per = int(dps[1] - dps[0])
    tpl_jc = int(template_model.joint_coord_count)
    tpl_dof = int(template_model.joint_dof_count)
    tpl_jq = template_model.joint_q.numpy()[:tpl_jc]
    tpl_jqd = template_model.joint_qd.numpy()[:tpl_dof]
    jq = batched_model.joint_q.numpy().copy()
    jqd = batched_model.joint_qd.numpy().copy()
    for w in range(num_envs):
        c0 = w * coord_per
        jq[c0 : c0 + tpl_jc] = tpl_jq
        d0 = w * dof_per
        jqd[d0 : d0 + tpl_dof] = tpl_jqd
    batched_model.joint_q.assign(jq)
    batched_model.joint_qd.assign(jqd)
