"""Fruiting chain ModelBuilder helpers."""

from __future__ import annotations

import dataclasses
import inspect
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any, Iterable

import numpy as np
import warp as wp

import newton

from apple_pick_sim.fruiting_system.gripper_proxy_shape import (
    add_gripper_proxy_collision_shape,
    gripper_proxy_clearance,
)
from apple_pick_sim.fruiting_system.params import (
    FruitingSystemParams,
    GripperProxyConfig,
    RodParams,
    TOPOLOGY_LINEAR_CHAIN,
    TOPOLOGY_T_JUNCTION,
)

def newton_solver_vbd_default_rigid_joint_kd() -> tuple[float, float]:
    """Return ``(linear_kd, angular_kd)`` from ``newton.solvers.SolverVBD`` constructor defaults."""
    sig = inspect.signature(newton.solvers.SolverVBD.__init__)
    linear_kd = float(sig.parameters["rigid_joint_linear_kd"].default)
    angular_kd = float(sig.parameters["rigid_joint_angular_kd"].default)
    return linear_kd, angular_kd


_NEWTON_DEFAULT_LINEAR_KD, _NEWTON_DEFAULT_ANGULAR_KD = newton_solver_vbd_default_rigid_joint_kd()

# Passed to ``SolverVBD(rigid_joint_*_kd=...)`` in :func:`make_fruiting_solver_vbd`.
# Newton's constructor defaults are 0.0 for both slots; keep overrides aligned via the
# same constants in ``batched_heterogeneous_config`` / example scripts.
FRUITING_VBD_RIGID_JOINT_LINEAR_KD = _NEWTON_DEFAULT_LINEAR_KD  # N·s/m
FRUITING_VBD_RIGID_JOINT_ANGULAR_KD = _NEWTON_DEFAULT_ANGULAR_KD  # N·m·s/rad

# Paul Tol bright palette (matches Newton ``ModelBuilder._SHAPE_COLOR_PALETTE``).
_ROD_DISPLAY_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "primary": (68, 119, 170),  # blue
    "secondary": (102, 204, 238),  # cyan
    "spur": (34, 136, 51),  # green
    "stem": (238, 153, 51),  # orange
}


def _rod_display_color(segment_name: str) -> tuple[float, float, float]:
    """Viewer RGB in ``[0, 1]`` for a fruiting rod segment type."""
    try:
        rgb = _ROD_DISPLAY_COLORS_RGB[segment_name]
    except KeyError as exc:
        raise ValueError(f"unknown rod segment name {segment_name!r}") from exc
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


def _prescribe_body_vbd_integration(builder: newton.ModelBuilder, body_id: int) -> None:
    """Disable VBD integration for a body (``inv_mass == 0``) while keeping ``body_mass``.

    Pose/velocity are overwritten each MuJoCo substep by staggered coupling sync.
    ``BodyFlags.KINEMATIC`` is not set because Newton only allows kinematic bodies on
    world-root joints; the apple remains a dynamic-flag child of the stem.
    """
    builder.body_inv_mass[body_id] = 0.0
    builder.body_inv_inertia[body_id] = wp.mat33(0.0)


def _pin_body_vbd_prescribed(builder: newton.ModelBuilder, body_id: int) -> None:
    """Alias for :func:`_prescribe_body_vbd_integration` (legacy name)."""
    _prescribe_body_vbd_integration(builder, body_id)


def prescribe_body_vbd_on_model(model: newton.Model, *body_ids: int) -> None:
    """Apply post-finalize VBD prescription (``finalize`` recomputes ``inv_mass`` from mass)."""
    if not body_ids:
        return
    inv = model.body_inv_mass.numpy().copy()
    inert = model.body_inv_inertia.numpy().copy()
    zero33 = np.zeros((3, 3), dtype=np.float32)
    for bid in body_ids:
        inv[int(bid)] = 0.0
        inert[int(bid)] = zero33
    model.body_inv_mass.assign(inv)
    model.body_inv_inertia.assign(inert)

@dataclasses.dataclass
class _FruitingChainArtifacts:
    """Mutable builder state for the fruiting rod chain before ``finalize``."""

    cable_joint_indices: list[int]
    fruiting_fixed_joints: list[tuple[int, str]]
    seg_bodies: dict[str, list[int]]
    apple_body: int | None
    all_joints: list[int]
    chain_bodies: list[int]
    proxy_placement_origin: wp.vec3
    proxy_placement_dir: wp.vec3
    world_root_joints: tuple[int, ...] = ()
    t_junction_support_ctx: tuple[list[wp.vec3], list[int], float] | None = None

def _filter_shape_collisions_within_group(
    builder: newton.ModelBuilder,
    bodies: list[int],
) -> None:
    """Add collision filter pairs for every distinct body pair in ``bodies``."""
    for i, body_a in enumerate(bodies):
        for body_b in bodies[i + 1 :]:
            for shape_a in builder.body_shapes.get(body_a, []):
                for shape_b in builder.body_shapes.get(body_b, []):
                    builder.add_shape_collision_filter_pair(shape_a, shape_b)


def _filter_shape_collisions_between_groups(
    builder: newton.ModelBuilder,
    group_a: list[int],
    group_b: list[int],
) -> None:
    """Add collision filter pairs for every body in ``group_a`` vs every body in ``group_b``."""
    for body_a in group_a:
        for body_b in group_b:
            for shape_a in builder.body_shapes.get(body_a, []):
                for shape_b in builder.body_shapes.get(body_b, []):
                    builder.add_shape_collision_filter_pair(shape_a, shape_b)


def _apply_default_fruiting_collision_filters(
    builder: newton.ModelBuilder,
    seg_bodies: dict[str, list[int]],
    apple_body: int | None,
    gripper_proxy_body: int | None = None,
    *,
    enable_apple_woody_collisions: bool = True,
    enable_proxy_woody_collisions: bool = True,
) -> None:
    """Apply default cable collision filters (``enable_self_collisions=False``).

    Four cases:

    1. **Self** — filter same-segment contacts (woody↔woody, stem↔stem).
    2. **Within chain** — filter cross-segment cable contacts (stem↔woody).
    3. **Apple ↔ woody** — collidable when ``enable_apple_woody_collisions`` (default on).
    4. **Proxy ↔ woody** — collidable when ``enable_proxy_woody_collisions`` (default on).
    """
    woody = (
        list(seg_bodies.get("primary", []))
        + list(seg_bodies.get("secondary", []))
        + list(seg_bodies.get("spur", []))
    )
    stem = list(seg_bodies.get("stem", []))

    # 1. Self-collision within each segment group.
    if woody:
        _filter_shape_collisions_within_group(builder, woody)
    if stem:
        _filter_shape_collisions_within_group(builder, stem)

    # 2. Within-chain cross-segment collision (cable bodies only).
    if woody and stem:
        _filter_shape_collisions_between_groups(builder, woody, stem)

    # 3. Apple ↔ woody — optional filter when disabled.
    if not enable_apple_woody_collisions and apple_body is not None and woody:
        _filter_shape_collisions_between_groups(builder, [apple_body], woody)

    # 4. Proxy ↔ woody — optional filter when disabled.
    if not enable_proxy_woody_collisions and gripper_proxy_body is not None and woody:
        _filter_shape_collisions_between_groups(builder, [gripper_proxy_body], woody)


def _apply_all_chain_collision_filters(
    builder: newton.ModelBuilder,
    chain_body_indices: list[int],
) -> None:
    """Add shape collision filter pairs for every distinct pair of chain bodies."""
    _filter_shape_collisions_within_group(builder, chain_body_indices)



def _new_fruiting_builder() -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e3  # 1.0e1 × ke(1.0e2), absolute VBD damping (#2877)
    builder.default_shape_cfg.mu = 1
    return builder


def _build_fruiting_chain_into_builder(
    builder: newton.ModelBuilder,
    params: FruitingSystemParams,
    base_pos: tuple[float, float, float],
) -> _FruitingChainArtifacts:
    """Populate rod chain + apple on ``builder`` (no articulation / finalize yet)."""
    if params.topology == TOPOLOGY_T_JUNCTION:
        return _build_t_junction_into_builder(builder, params, base_pos)
    if params.topology != TOPOLOGY_LINEAR_CHAIN:
        raise ValueError(f"unsupported topology {params.topology!r}")
    return _build_linear_chain_into_builder(builder, params, base_pos)


def _build_linear_chain_into_builder(
    builder: newton.ModelBuilder,
    params: FruitingSystemParams,
    base_pos: tuple[float, float, float],
) -> _FruitingChainArtifacts:
    """Serial primary→…→apple chain with a zero-mass pin at the first primary body."""
    if not any((params.primary, params.secondary, params.spur, params.stem)):
        raise ValueError(
            "At least one rod segment (primary, secondary, spur, or stem) must be non-None."
        )
        
    cable_joint_indices: list[int] = []
    fruiting_fixed_joints: list[tuple[int, str]] = []

    origin = wp.vec3(*base_pos)

    rod_specs: list[tuple[str, RodParams]] = [
        (n, rp)
        for n, rp in (
            ("primary", params.primary),
            ("secondary", params.secondary),
            ("spur", params.spur),
            ("stem", params.stem),
        )
        if rp is not None
    ]

    seg_bodies: dict[str, list[int]] = {
        "primary": [],
        "secondary": [],
        "spur": [],
        "stem": [],
    }

    all_joints: list[int] = []
    prev_bodies: list[int] | None = None
    prev_name: str | None = None
    prev_rod: RodParams | None = None
    prev_points: list[wp.vec3] | None = None
    prev_quats: list[wp.quat] | None = None

    for name, rod in rod_specs:
        start = origin if prev_bodies is None else prev_points[-1]
        points, quats = _make_rod_geometry(
            start, rod.direction, rod.length, rod.num_segments
        )
        rod_cfg = builder.ShapeConfig(density=rod.density)
        bodies, joints = builder.add_rod(
            positions=points,
            quaternions=quats,
            radius=rod.radius,
            cfg=rod_cfg,
            bend_stiffness=rod.bend_stiffness,
            bend_damping=rod.bend_damping,
            stretch_stiffness=rod.stretch_stiffness,
            stretch_damping=rod.stretch_damping,
            wrap_in_articulation=False,
            body_frame_origin="start",
            label=name,
            color=_rod_display_color(name),
        )
        all_joints.extend(joints)
        cable_joint_indices.extend(joints)

        if prev_bodies is None:
            builder.body_mass[bodies[0]] = 0.0
            builder.body_inv_mass[bodies[0]] = 0.0
        else:
            assert prev_rod is not None and prev_name is not None
            parent_seg_len = prev_rod.length / prev_rod.num_segments
            j_link = _connect_rod_tip_to_base(
                builder,
                parent_body=prev_bodies[-1],
                parent_seg_length=parent_seg_len,
                child_body=bodies[0],
                key=f"joint_{prev_name}_{name}",
            )
            all_joints.append(j_link)
            fruiting_fixed_joints.append((j_link, f"joint_{prev_name}_{name}"))

        seg_bodies[name] = bodies
        prev_bodies = bodies
        prev_name = name
        prev_rod = rod
        prev_points = points
        prev_quats = quats

    assert prev_bodies is not None and prev_rod is not None
    assert prev_points is not None and prev_quats is not None
    last_name = prev_name
    assert last_name is not None
    tip_seg_len = prev_rod.length / prev_rod.num_segments

    stem_tip_pt = prev_points[-1]
    stem_base_pt = prev_points[-2]
    last_seg_dir = wp.normalize(stem_tip_pt - stem_base_pt)
    proxy_placement_origin = stem_tip_pt
    proxy_placement_dir = last_seg_dir

    apple_body: int | None = None
    if params.apple_radius is not None and params.apple_density is not None:
        apple_pos = stem_tip_pt + last_seg_dir * params.apple_radius
        proxy_placement_origin = apple_pos
        apple_mass = (4.0 / 3.0) * math.pi * params.apple_radius**3 * params.apple_density
        apple_quat = wp.quat_identity()
        apple_body = builder.add_link(
            xform=wp.transform(apple_pos, apple_quat),
            mass=apple_mass,
            label="apple",
        )
        apple_shape_cfg = builder.default_shape_cfg.copy()
        apple_shape_cfg.density = 0.0
        builder.add_shape_sphere(
            body=apple_body, radius=params.apple_radius, cfg=apple_shape_cfg
        )
        j_st2apple = _connect_rod_tip_to_apple(
            builder,
            stem_tip_body=prev_bodies[-1],
            stem_tip_quat=prev_quats[-1],
            stem_seg_length=tip_seg_len,
            segment_dir_world=last_seg_dir,
            apple_radius=params.apple_radius,
            apple_body=apple_body,
            apple_quat=apple_quat,
            key=f"joint_{last_name}_apple",
        )
        all_joints.append(j_st2apple)
        fruiting_fixed_joints.append((j_st2apple, f"joint_{last_name}_apple"))

    chain_bodies: list[int] = (
        seg_bodies["primary"]
        + seg_bodies["secondary"]
        + seg_bodies["spur"]
        + seg_bodies["stem"]
    )
    if apple_body is not None:
        chain_bodies.append(apple_body)

    return _FruitingChainArtifacts(
        cable_joint_indices=cable_joint_indices,
        fruiting_fixed_joints=fruiting_fixed_joints,
        seg_bodies=seg_bodies,
        apple_body=apple_body,
        all_joints=all_joints,
        chain_bodies=chain_bodies,
        proxy_placement_origin=proxy_placement_origin,
        proxy_placement_dir=proxy_placement_dir,
    )


def _build_t_junction_into_builder(
    builder: newton.ModelBuilder,
    params: FruitingSystemParams,
    base_pos: tuple[float, float, float],
) -> _FruitingChainArtifacts:
    """T topology: primary simply supported at both ends; spur branch at mid-span."""
    if params.secondary is not None:
        raise ValueError("T-junction topology does not support secondary segment")
    if params.primary is None:
        raise ValueError("T-junction topology requires primary segment")

    cable_joint_indices: list[int] = []
    fruiting_fixed_joints: list[tuple[int, str]] = []
    all_joints: list[int] = []
    seg_bodies: dict[str, list[int]] = {
        "primary": [],
        "secondary": [],
        "spur": [],
        "stem": [],
    }

    center = wp.vec3(*base_pos)
    primary = params.primary
    direction = wp.normalize(wp.vec3(*primary.direction))
    half_len = primary.length / 2.0
    start = center - direction * half_len
    primary_points, primary_quats = _make_rod_geometry(
        start, primary.direction, primary.length, primary.num_segments
    )
    rod_cfg = builder.ShapeConfig(density=primary.density)
    primary_bodies, primary_joints = builder.add_rod(
        positions=primary_points,
        quaternions=primary_quats,
        radius=primary.radius,
        cfg=rod_cfg,
        bend_stiffness=primary.bend_stiffness,
        bend_damping=primary.bend_damping,
        stretch_stiffness=primary.stretch_stiffness,
        stretch_damping=primary.stretch_damping,
        wrap_in_articulation=False,
        body_frame_origin="start",
        label="primary",
        color=_rod_display_color("primary"),
    )
    all_joints.extend(primary_joints)
    cable_joint_indices.extend(primary_joints)
    seg_bodies["primary"] = primary_bodies

    branch_specs: list[tuple[str, RodParams]] = [
        (n, rp)
        for n, rp in (("spur", params.spur), ("stem", params.stem))
        if rp is not None
    ]

    prev_bodies: list[int] = primary_bodies
    prev_name = "primary"
    prev_rod: RodParams = primary
    prev_points: list[wp.vec3] = primary_points
    prev_quats: list[wp.quat] = primary_quats
    parent_idx = _primary_spur_parent_body_index(
        primary.num_segments, params.spur_attach_fraction
    )

    for branch_i, (name, rod) in enumerate(branch_specs):
        radial_world = wp.vec3(0.0, 0.0, 0.0)
        if branch_i == 0:
            if params.spur_surface_offset:
                radial_world = _primary_radial_surface_offset_world(
                    primary.direction, rod.direction, primary.radius
                )
            branch_start = primary_points[parent_idx + 1] + radial_world
        else:
            branch_start = prev_points[-1]
        points, quats = _make_rod_geometry(
            branch_start, rod.direction, rod.length, rod.num_segments
        )
        rod_cfg = builder.ShapeConfig(density=rod.density)
        bodies, joints = builder.add_rod(
            positions=points,
            quaternions=quats,
            radius=rod.radius,
            cfg=rod_cfg,
            bend_stiffness=rod.bend_stiffness,
            bend_damping=rod.bend_damping,
            stretch_stiffness=rod.stretch_stiffness,
            stretch_damping=rod.stretch_damping,
            wrap_in_articulation=False,
            body_frame_origin="start",
            label=name,
            color=_rod_display_color(name),
        )
        all_joints.extend(joints)
        cable_joint_indices.extend(joints)

        if branch_i == 0:
            parent_seg_len = prev_rod.length / prev_rod.num_segments
            parent_local_offset = wp.quat_rotate_inv(
                primary_quats[parent_idx], radial_world
            )
            j_link = _connect_rod_tip_to_base(
                builder,
                parent_body=prev_bodies[parent_idx],
                parent_seg_length=parent_seg_len,
                child_body=bodies[0],
                key=f"joint_{prev_name}_{name}",
                parent_local_offset=parent_local_offset,
            )
        else:
            parent_seg_len = prev_rod.length / prev_rod.num_segments
            j_link = _connect_rod_tip_to_base(
                builder,
                parent_body=prev_bodies[-1],
                parent_seg_length=parent_seg_len,
                child_body=bodies[0],
                key=f"joint_{prev_name}_{name}",
            )
        all_joints.append(j_link)
        fruiting_fixed_joints.append((j_link, f"joint_{prev_name}_{name}"))

        seg_bodies[name] = bodies
        prev_bodies = bodies
        prev_name = name
        prev_rod = rod
        prev_points = points
        prev_quats = quats

    if not branch_specs:
        raise ValueError(
            "T-junction topology requires at least one branch segment (spur or stem)."
        )

    assert prev_rod is not None and prev_points is not None and prev_quats is not None
    last_name = prev_name
    tip_seg_len = prev_rod.length / prev_rod.num_segments
    stem_tip_pt = prev_points[-1]
    stem_base_pt = prev_points[-2]
    last_seg_dir = wp.normalize(stem_tip_pt - stem_base_pt)
    proxy_placement_origin = stem_tip_pt
    proxy_placement_dir = last_seg_dir

    apple_body: int | None = None
    if params.apple_radius is not None and params.apple_density is not None:
        apple_pos = stem_tip_pt + last_seg_dir * params.apple_radius
        proxy_placement_origin = apple_pos
        apple_mass = (4.0 / 3.0) * math.pi * params.apple_radius**3 * params.apple_density
        apple_quat = wp.quat_identity()
        apple_body = builder.add_link(
            xform=wp.transform(apple_pos, apple_quat),
            mass=apple_mass,
            label="apple",
        )
        apple_shape_cfg = builder.default_shape_cfg.copy()
        apple_shape_cfg.density = 0.0
        builder.add_shape_sphere(
            body=apple_body, radius=params.apple_radius, cfg=apple_shape_cfg
        )
        j_st2apple = _connect_rod_tip_to_apple(
            builder,
            stem_tip_body=prev_bodies[-1],
            stem_tip_quat=prev_quats[-1],
            stem_seg_length=tip_seg_len,
            segment_dir_world=last_seg_dir,
            apple_radius=params.apple_radius,
            apple_body=apple_body,
            apple_quat=apple_quat,
            key=f"joint_{last_name}_apple",
        )
        all_joints.append(j_st2apple)
        fruiting_fixed_joints.append((j_st2apple, f"joint_{last_name}_apple"))

    chain_bodies: list[int] = (
        seg_bodies["primary"]
        + seg_bodies["spur"]
        + seg_bodies["stem"]
    )
    if apple_body is not None:
        chain_bodies.append(apple_body)

    primary_seg_len = primary.length / primary.num_segments
    return _FruitingChainArtifacts(
        cable_joint_indices=cable_joint_indices,
        fruiting_fixed_joints=fruiting_fixed_joints,
        seg_bodies=seg_bodies,
        apple_body=apple_body,
        all_joints=all_joints,
        chain_bodies=chain_bodies,
        proxy_placement_origin=proxy_placement_origin,
        proxy_placement_dir=proxy_placement_dir,
        t_junction_support_ctx=(primary_points, primary_bodies, primary_seg_len),
    )


def _attach_t_junction_world_supports(
    builder: newton.ModelBuilder,
    artifacts: _FruitingChainArtifacts,
) -> None:
    """Add world-fixed primary endpoint supports (must run after all other joints)."""
    ctx = artifacts.t_junction_support_ctx
    if ctx is None:
        return
    primary_points, primary_bodies, primary_seg_len = ctx
    j_left = _connect_world_to_rod_base(
        builder,
        world_pos=primary_points[0],
        body=primary_bodies[0],
        key="joint_primary_support_left",
    )
    j_right = _connect_world_to_rod_tip(
        builder,
        world_pos=primary_points[-1],
        body=primary_bodies[-1],
        seg_length=primary_seg_len,
        key="joint_primary_support_right",
    )
    artifacts.all_joints.extend((j_left, j_right))
    artifacts.fruiting_fixed_joints.extend(
        (
            (j_left, "joint_primary_support_left"),
            (j_right, "joint_primary_support_right"),
        )
    )
    artifacts.world_root_joints = (j_left, j_right)
    artifacts.t_junction_support_ctx = None


def _apply_collision_filters_between_chain_groups(
    builder: newton.ModelBuilder,
    chain_a: list[int],
    chain_b: list[int],
) -> None:
    """Disable collisions between every body in ``chain_a`` and every body in ``chain_b``."""
    _filter_shape_collisions_between_groups(builder, chain_a, chain_b)


def _register_fruiting_articulations(
    builder: newton.ModelBuilder,
    *,
    all_joints: list[int],
    chain_bodies: list[int],
    enable_self_collisions: bool,
    seg_bodies: dict[str, list[int]] | None = None,
    apple_body: int | None = None,
    gripper_proxy_body: int | None = None,
    gripper_proxy_joints: tuple[int, ...] = (),
    world_root_joints: tuple[int, ...] = (),
    extra_chain_groups_for_filters: tuple[list[int], ...] = (),
    enable_apple_woody_collisions: bool = True,
    enable_proxy_woody_collisions: bool = True,
) -> None:
    """Register articulations and optional collision filters (no ``finalize``)."""
    joint_list = sorted(all_joints)
    isolated = sorted(set(gripper_proxy_joints) | set(world_root_joints))
    if isolated:
        isolated_set = set(isolated)
        chain_joints = [j for j in joint_list if j not in isolated_set]
        builder.add_articulation(chain_joints)
        for ij in isolated:
            builder.add_articulation([ij])
    else:
        builder.add_articulation(joint_list)
    if not enable_self_collisions:
        if seg_bodies is not None:
            _apply_default_fruiting_collision_filters(
                builder,
                seg_bodies,
                apple_body,
                gripper_proxy_body=gripper_proxy_body,
                enable_apple_woody_collisions=enable_apple_woody_collisions,
                enable_proxy_woody_collisions=enable_proxy_woody_collisions,
            )
        else:
            _apply_all_chain_collision_filters(builder, chain_bodies)
    for other in extra_chain_groups_for_filters:
        _apply_collision_filters_between_chain_groups(builder, chain_bodies, other)


def _finalize_fruiting_builder_joints(
    builder: newton.ModelBuilder,
    *,
    all_joints: list[int],
    chain_bodies: list[int],
    device: str,
    enable_self_collisions: bool,
    seg_bodies: dict[str, list[int]] | None = None,
    apple_body: int | None = None,
    gripper_proxy_body: int | None = None,
    gripper_proxy_joints: tuple[int, ...] = (),
    world_root_joints: tuple[int, ...] = (),
    extra_chain_groups_for_filters: tuple[list[int], ...] = (),
    enable_apple_woody_collisions: bool = True,
    enable_proxy_woody_collisions: bool = True,
) -> newton.Model:
    """Finalize a cable ``Model``; world-root and FREE proxy joints use separate articulations."""
    _register_fruiting_articulations(
        builder,
        all_joints=all_joints,
        chain_bodies=chain_bodies,
        enable_self_collisions=enable_self_collisions,
        seg_bodies=seg_bodies,
        apple_body=apple_body,
        gripper_proxy_body=gripper_proxy_body,
        gripper_proxy_joints=gripper_proxy_joints,
        world_root_joints=world_root_joints,
        extra_chain_groups_for_filters=extra_chain_groups_for_filters,
        enable_apple_woody_collisions=enable_apple_woody_collisions,
        enable_proxy_woody_collisions=enable_proxy_woody_collisions,
    )
    # builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    return model


def _finalize_fruiting_builder(
    builder: newton.ModelBuilder,
    artifacts: _FruitingChainArtifacts,
    *,
    device: str,
    enable_self_collisions: bool,
    gripper_proxy_joint: int | None = None,
    gripper_proxy_body: int | None = None,
    enable_apple_woody_collisions: bool = True,
    enable_proxy_woody_collisions: bool = True,
) -> newton.Model:
    """Finalize the cable model; keep world-root proxy FREE joints out of the tree articulation."""
    proxy_joints = (gripper_proxy_joint,) if gripper_proxy_joint is not None else ()
    return _finalize_fruiting_builder_joints(
        builder,
        all_joints=artifacts.all_joints,
        chain_bodies=artifacts.chain_bodies,
        device=device,
        enable_self_collisions=enable_self_collisions,
        seg_bodies=artifacts.seg_bodies,
        apple_body=artifacts.apple_body,
        gripper_proxy_body=gripper_proxy_body,
        gripper_proxy_joints=proxy_joints,
        world_root_joints=artifacts.world_root_joints,
        enable_apple_woody_collisions=enable_apple_woody_collisions,
        enable_proxy_woody_collisions=enable_proxy_woody_collisions,
    )
def _approach_dir_from_robot_base(
    apple_pos: wp.vec3,
    robot_base_pos: tuple[float, float, float],
) -> wp.vec3:
    """Unit vector from apple center toward the robot base."""
    robot_vec = wp.vec3(*robot_base_pos) - apple_pos
    return wp.normalize(robot_vec)


def _stem_perpendicular_robot_pole(
    stem_dir: wp.vec3,
    robot_vec: wp.vec3,
) -> wp.vec3:
    """Unit pole in the stem-perpendicular plane facing the robot."""
    stem = wp.normalize(stem_dir)
    perp = robot_vec - wp.dot(robot_vec, stem) * stem
    perp_len = float(wp.length(perp))
    if perp_len < 1e-12:
        ref = wp.vec3(1.0, 0.0, 0.0)
        perp = wp.cross(stem, ref)
        perp_len = float(wp.length(perp))
        if perp_len < 1e-12:
            ref = wp.vec3(0.0, 1.0, 0.0)
            perp = wp.cross(stem, ref)
            perp_len = float(wp.length(perp))
        if perp_len < 1e-12:
            raise ValueError("cannot construct perpendicular to stem_dir")
        pole = perp / perp_len
        if wp.dot(pole, robot_vec) < 0.0:
            pole = -pole
        return pole
    return perp / perp_len


def _weld_apple_center(
    config: GripperProxyConfig,
    artifacts: _FruitingChainArtifacts,
) -> wp.vec3:
    """Apple center used for robot-facing weld validation and surface placement."""
    if config.weld_reference_pos is not None:
        return wp.vec3(*config.weld_reference_pos)
    return artifacts.proxy_placement_origin


def _approach_dir_in_apple_frame(
    approach_world: wp.vec3,
    apple_quat: wp.quat | None,
) -> wp.vec3:
    """Map a world-frame weld approach direction into the apple body frame."""
    if apple_quat is None:
        return wp.normalize(approach_world)
    return wp.normalize(wp.quat_rotate_inv(apple_quat, approach_world))


def _resolve_robot_facing_approach_dir(
    config: GripperProxyConfig,
    apple_pos: wp.vec3,
    robot_base_pos: tuple[float, float, float],
    stem_dir: wp.vec3,
) -> wp.vec3:
    """Approach direction for robot-facing weld (stem⊥ pole or explicit ``weld_direction``)."""
    if config.weld_reference_stem_dir is not None:
        stem_dir = wp.normalize(wp.vec3(*config.weld_reference_stem_dir))
    robot_vec = wp.vec3(*robot_base_pos) - apple_pos
    pole = _stem_perpendicular_robot_pole(stem_dir, robot_vec)
    if config.weld_direction is None:
        return pole

    weld_dir = wp.normalize(wp.vec3(*config.weld_direction))
    if wp.dot(weld_dir, pole) < 0.0:
        raise ValueError(
            "weld_direction must be on the stem-perpendicular robot-facing hemisphere "
            "(dot product with stem⊥ pole ≥ 0)"
        )
    return weld_dir


def _add_gripper_proxy(
    builder: newton.ModelBuilder,
    artifacts: _FruitingChainArtifacts,
    config: GripperProxyConfig,
    *,
    apple_radius: float | None,
    robot_base_pos: tuple[float, float, float] | None = None,
) -> tuple[int, int | None, tuple[float, float, float, float, float, float, float] | None]:
    """Add gripper proxy link, shape, and joint(s).

    Returns ``(body_id, apple_fixed_joint, proxy_offset_in_apple_frame)``.

    Default: ``add_joint_free`` so the proxy is a dynamic body whose pose/velocity are
    overwritten each MuJoCo substep by ``sync_proxy_state`` (not integrated from cable
    gravity alone). With ``fix_to_apple``, a FIXED joint welds the proxy to the apple
    at the exterior pole (same placement as the free proxy, via ``parent_xform`` /
    ``child_xform``). The apple and proxy use ``inv_mass == 0`` so VBD does not integrate
    them; staggered coupling teleports their poses from the robot TCP each substep while
    the stem supplies the harvested wrench.
    """
    if config.robot_facing_weld and not config.fix_to_apple:
        raise ValueError("robot_facing_weld requires fix_to_apple=True")
    if config.weld_direction is not None and not config.fix_to_apple:
        raise ValueError("weld_direction requires fix_to_apple=True")
    if config.weld_reference_quat is not None and config.weld_reference_pos is None:
        raise ValueError("weld_reference_quat requires weld_reference_pos")

    weld_apple_center = _weld_apple_center(config, artifacts)
    weld_apple_quat = (
        None
        if config.weld_reference_quat is None
        else wp.quat(*config.weld_reference_quat)
    )
    clearance = gripper_proxy_clearance(config)
    robot_facing_approach_dir: wp.vec3 | None = None

    vis_offset: wp.vec3 = wp.vec3(0.0, 0.0, 0.0)
    if apple_radius is not None:
        if config.fix_to_apple and config.robot_facing_weld:
            if robot_base_pos is None:
                raise ValueError("robot_facing_weld requires robot_base_pos")
            robot_facing_approach_dir = _resolve_robot_facing_approach_dir(
                config,
                weld_apple_center,
                robot_base_pos,
                artifacts.proxy_placement_dir,
            )
            # IK target is on the apple surface; clearance is stored in vis_offset.
            proxy_pos = weld_apple_center + robot_facing_approach_dir * apple_radius
            vis_offset = robot_facing_approach_dir * clearance
        elif config.fix_to_apple and config.weld_direction is not None:
            weld_dir = wp.normalize(wp.vec3(*config.weld_direction))
            proxy_pos = weld_apple_center + weld_dir * apple_radius
            vis_offset = weld_dir * clearance
        else:
            # Free proxy or legacy random weld.
            proxy_pos = artifacts.proxy_placement_origin + artifacts.proxy_placement_dir * apple_radius
            vis_offset = artifacts.proxy_placement_dir * clearance
    elif config.fix_to_apple:
        if artifacts.apple_body is None:
            raise ValueError("fix_to_apple requires an apple body in the scene")
        proxy_pos = artifacts.proxy_placement_origin
        vis_offset = wp.vec3(0.0, 0.0, 0.0)
    else:
        proxy_pos = artifacts.proxy_placement_origin
        vis_offset = artifacts.proxy_placement_dir * clearance

    proxy_body = builder.add_link(
        xform=wp.transform(proxy_pos, wp.quat_identity()),
        mass=config.mass,
        label=config.label,
    )
    proxy_shape_cfg = builder.default_shape_cfg.copy()
    proxy_shape_cfg.density = 0.0
    add_gripper_proxy_collision_shape(
        builder,
        proxy_body,
        config,
        shape_cfg=proxy_shape_cfg,
    )

    apple_fixed_joint: int | None = None
    proxy_free_joint: int | None = None
    proxy_offset_in_apple_frame: tuple[float, float, float, float, float, float, float] | None = None
    if config.fix_to_apple:
        assert artifacts.apple_body is not None
        if apple_radius is not None:
            if config.robot_facing_weld:
                assert robot_facing_approach_dir is not None
                approach_dir = robot_facing_approach_dir
            elif config.weld_direction is not None:
                approach_dir = wp.normalize(wp.vec3(*config.weld_direction))
            else:
                # Random approach direction (proxy Z-axis will point along this toward apple center)
                theta = random.uniform(0.0, 2 * math.pi)
                phi = math.acos(random.uniform(-1.0, 1.0))
                approach_dir = wp.vec3(
                    math.sin(phi) * math.cos(theta),
                    math.sin(phi) * math.sin(theta),
                    math.cos(phi),
                )

            # Position offset in apple frame: robot-facing weld uses the exterior pole
            # toward the robot; legacy random weld keeps the opposite convention.
            approach_local = _approach_dir_in_apple_frame(approach_dir, weld_apple_quat)
            if config.robot_facing_weld or config.weld_direction is not None:
                off = approach_local * (apple_radius + clearance)
            else:
                off = approach_local * -(apple_radius + clearance)

            # Construct look-at rotation (proxy local Z → approach_local in apple frame)
            z_axis = approach_local
            up = wp.vec3(0.0, 0.0, 1.0)
            if abs(wp.dot(z_axis, up)) > 0.99:
                up = wp.vec3(1.0, 0.0, 0.0)
            x_axis = wp.normalize(wp.cross(up, z_axis))
            y_axis = wp.cross(z_axis, x_axis)
            R = wp.mat33(
                x_axis[0], y_axis[0], z_axis[0],
                x_axis[1], y_axis[1], z_axis[1],
                x_axis[2], y_axis[2], z_axis[2],
            )
            base_rot = wp.quat_from_matrix(R)

            if config.robot_facing_weld or config.weld_direction is not None:
                final_rot = base_rot
            else:
                # Optional: randomize roll around the approach axis
                roll_angle = random.uniform(0.0, 2 * math.pi)
                roll_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), roll_angle)
                final_rot = wp.mul(base_rot, roll_rot)


            # 4. Full rigid pose: parent anchor in apple frame, child at identity
            parent_xform = wp.transform(off, final_rot)
            child_xform = wp.transform_identity()

            # 5. Export 7D transform (pos_x, pos_y, pos_z, qi, qj, qk, qw)
            proxy_offset_in_apple_frame = (
                float(off[0]), float(off[1]), float(off[2]),
                float(final_rot[0]), float(final_rot[1]), float(final_rot[2]), float(final_rot[3]),
            )
        else:
            parent_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
            child_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        apple_fixed_joint = builder.add_joint_fixed(
            parent=artifacts.apple_body,
            child=proxy_body,
            parent_xform=parent_xform,
            child_xform=child_xform,
            label="joint_apple_gripper_proxy",
        )
        artifacts.all_joints.append(apple_fixed_joint)
        artifacts.fruiting_fixed_joints.append(
            (apple_fixed_joint, "joint_apple_gripper_proxy")
        )
        _prescribe_body_vbd_integration(builder, artifacts.apple_body)
        _prescribe_body_vbd_integration(builder, proxy_body)
    else:
        proxy_free_joint = builder.add_joint_free(parent=-1, child=proxy_body)
        artifacts.all_joints.append(proxy_free_joint)

    return proxy_body, apple_fixed_joint, proxy_offset_in_apple_frame, proxy_free_joint, vis_offset

def make_fruiting_solver_vbd(model: newton.Model, **overrides: Any) -> newton.solvers.SolverVBD:
    kwargs: dict[str, Any] = {
        "iterations": 25,
        "friction_epsilon": 1e-2,
        "rigid_contact_k_start": 1.0e4,
        "rigid_joint_linear_k_start": 1.0e8,
        "rigid_joint_angular_k_start": 1.0e6,
        "rigid_joint_linear_kd": FRUITING_VBD_RIGID_JOINT_LINEAR_KD,
        "rigid_joint_angular_kd": FRUITING_VBD_RIGID_JOINT_ANGULAR_KD,
    }
    kwargs.update(overrides)
    return newton.solvers.SolverVBD(model, **kwargs)


def _match_fruiting_joint_labels(
    fruiting_fixed_joints: Iterable[tuple[int, str]],
    label_values: dict[str, float],
    *,
    param_name: str = "label_kd",
    value_label: str = "angular kd",
) -> dict[str, list[int]]:
    """Match label override keys to template joint indices; validate inputs."""
    if not label_values:
        return {}

    for key, val in label_values.items():
        if val < 0.0:
            raise ValueError(
                f"{param_name}[{key!r}]={val} is negative; {value_label} must be >= 0."
            )

    joint_pairs = list(fruiting_fixed_joints)
    keys = list(label_values.keys())

    joint_match: dict[int, tuple[str, str]] = {}
    for joint_index, label in joint_pairs:
        matching_keys = [k for k in keys if k in label]
        if len(matching_keys) > 1:
            raise ValueError(
                f"ambiguous {param_name} match for joint_index={joint_index} "
                f"label={label!r}: keys {matching_keys!r} all match."
            )
        if len(matching_keys) == 1:
            joint_match[joint_index] = (label, matching_keys[0])

    unmatched_keys = [k for k in keys if k not in {m[1] for m in joint_match.values()}]
    if unmatched_keys:
        raise ValueError(
            f"{param_name} key(s) matched no fruiting fixed joint: {unmatched_keys!r}."
        )

    matched_by_key: dict[str, list[int]] = {k: [] for k in keys}
    for joint_index, (_label, key) in joint_match.items():
        matched_by_key[key].append(joint_index)

    return {k: sorted(v) for k, v in matched_by_key.items() if v}


def _patch_k_constraint_slot(
    k_np: np.ndarray,
    k_min_np: np.ndarray,
    k_max_np: np.ndarray,
    constraint_index: int,
    kp_val: float,
) -> None:
    """Write one penalty-k slot and widen AVBD k bounds to include ``kp_val``."""
    k_np[constraint_index] = kp_val
    k_min_np[constraint_index] = min(float(k_min_np[constraint_index]), kp_val)
    k_max_np[constraint_index] = max(float(k_max_np[constraint_index]), kp_val)


def _validate_template_joint_angular_slots(
    solver: newton.solvers.SolverVBD,
    template_joint_indices: Iterable[int],
) -> None:
    """Ensure each template joint has an angular constraint slot."""
    ang_slot = newton.solvers.SolverVBD.JointSlot.ANGULAR
    jc_dim = solver.joint_constraint_dim.numpy()
    for joint_index in template_joint_indices:
        cdim = int(jc_dim[joint_index])
        if cdim <= ang_slot:
            raise ValueError(
                f"joint_index={joint_index} has constraint dimension {cdim}; "
                f"cannot set angular kd (slot {ang_slot})."
            )


def _validate_template_joint_linear_slots(
    solver: newton.solvers.SolverVBD,
    template_joint_indices: Iterable[int],
) -> None:
    """Ensure each template joint has a linear constraint slot."""
    lin_slot = newton.solvers.SolverVBD.JointSlot.LINEAR
    jc_dim = solver.joint_constraint_dim.numpy()
    for joint_index in template_joint_indices:
        cdim = int(jc_dim[joint_index])
        if cdim <= lin_slot:
            raise ValueError(
                f"joint_index={joint_index} has constraint dimension {cdim}; "
                f"cannot set linear kd (slot {lin_slot})."
            )


@wp.kernel(enable_backward=False)
def _apply_batched_joint_kd_kernel(
    joint_constraint_start: wp.array(dtype=wp.int32),
    template_joint_indices: wp.array(dtype=wp.int32),
    kd_values: wp.array(dtype=wp.float32),
    joints_per_world: int,
    n_templates: int,
    kd_slot: int,
    joint_penalty_kd: wp.array(dtype=wp.float32),
):
    """Patch one penalty-kd slot for one (env, matched-template-joint) pair.

    ``kd_values`` is length ``num_envs * n_templates``, indexed as
    ``w * n_templates + k`` (broadcast callers expand a single role map).
    """
    w, k = wp.tid()
    global_joint = w * joints_per_world + template_joint_indices[k]
    c0 = joint_constraint_start[global_joint]
    joint_penalty_kd[c0 + kd_slot] = kd_values[w * n_templates + k]


def _normalize_batched_label_kd(
    label_kd: Mapping[str, float] | None,
    label_kd_per_env: Sequence[Mapping[str, float]] | None,
    *,
    num_envs: int,
) -> list[dict[str, float]]:
    """Return per-env label->kd maps (broadcast ``label_kd`` when given)."""
    if (label_kd is None) == (label_kd_per_env is None):
        raise ValueError("provide exactly one of label_kd or label_kd_per_env")
    if label_kd is not None:
        return [{str(k): float(v) for k, v in label_kd.items()} for _ in range(int(num_envs))]
    if len(label_kd_per_env) != int(num_envs):
        raise ValueError(
            f"label_kd_per_env length {len(label_kd_per_env)} != num_envs={num_envs}"
        )
    keys0 = set(label_kd_per_env[0].keys())
    out: list[dict[str, float]] = []
    for env_idx, mapping in enumerate(label_kd_per_env):
        if set(mapping.keys()) != keys0:
            raise ValueError(
                "label_kd_per_env maps must share the same keys; "
                f"env0={sorted(keys0)} env{env_idx}={sorted(mapping.keys())}"
            )
        out.append({str(k): float(v) for k, v in mapping.items()})
    return out


def _batched_kd_values_from_per_env(
    matched_by_key: dict[str, list[int]],
    per_env_label_kd: Sequence[Mapping[str, float]],
    template_idx_np: np.ndarray,
) -> np.ndarray:
    """Pack ``(num_envs, n_templates)`` kd values in row-major order."""
    n_templates = int(len(template_idx_np))
    num_envs = int(len(per_env_label_kd))
    kd_np = np.empty(num_envs * n_templates, dtype=np.float32)
    for w, label_kd in enumerate(per_env_label_kd):
        kd_by_template = {
            int(j): float(label_kd[key])
            for key, indices in matched_by_key.items()
            for j in indices
        }
        base = w * n_templates
        for k, template_j in enumerate(template_idx_np):
            kd_np[base + k] = kd_by_template[int(template_j)]
    return kd_np


def set_fruiting_joint_angular_kd(
    solver: newton.solvers.SolverVBD,
    fruiting_fixed_joints: Iterable[tuple[int, str]],
    label_kd: dict[str, float],
) -> dict[str, list[int]]:
    """Patch per-joint angular AVBD damping on an existing ``SolverVBD``.

    ``SolverVBD`` seeds one global ``rigid_joint_angular_kd`` into every
    structural angular constraint slot at construction time.  Child-body inertia
    varies ~1000× across the fruiting chain, so a single scalar is often wrong
    everywhere at once.  This helper overwrites ``solver.joint_penalty_kd`` at
    the angular slot (``JointSlot.ANGULAR``) for selected FIXED joints.

    Each ``label_kd`` key is matched as a **substring** of the joint label
    recorded in ``fruiting_fixed_joints`` (e.g. ``"stem_apple"`` matches
    ``"joint_stem_apple"``).  One key may match multiple joints (e.g.
    ``"support"`` matches both T-junction world supports).  Joints not matched
    by any key retain the solver's default ``rigid_joint_angular_kd``.

    Call after ``make_fruiting_solver_vbd(model)`` and before ``solver.step()``.
    The solver must have rigid-joint state initialized (``model.body_count > 0``,
    not ``integrate_with_external_rigid_solver``).

    Args:
        solver: Constructed VBD solver whose ``joint_penalty_kd`` array will be
            patched in place.
        fruiting_fixed_joints: ``(joint_index, label)`` pairs from scene build
            (``FruitingSystemScene.fruiting_fixed_joints``).
        label_kd: Substring label -> absolute angular damping [N·m·s/rad].

    Returns:
        ``{label_kd_key: [joint_index, ...]}`` with sorted joint indices per key.

    Raises:
        ValueError: Negative ``kd``, ambiguous multi-key match on one joint,
            or a key that matches no joint.
    """
    matched_by_key = _match_fruiting_joint_labels(fruiting_fixed_joints, label_kd)
    if not matched_by_key:
        return {}

    if not hasattr(solver, "joint_penalty_kd"):
        raise RuntimeError(
            "SolverVBD joint_penalty_kd is not initialized; "
            "ensure the solver was constructed with rigid bodies present."
        )

    template_indices = sorted(
        {j for indices in matched_by_key.values() for j in indices}
    )
    _validate_template_joint_angular_slots(solver, template_indices)

    jc_start = solver.joint_constraint_start.numpy()
    kd_np = solver.joint_penalty_kd.numpy().copy()
    ang_slot = newton.solvers.SolverVBD.JointSlot.ANGULAR

    for key, joint_indices in matched_by_key.items():
        kd_val = float(label_kd[key])
        for joint_index in joint_indices:
            c0 = int(jc_start[joint_index])
            kd_np[c0 + ang_slot] = kd_val

    solver.joint_penalty_kd.assign(kd_np)
    return matched_by_key


def set_fruiting_joint_angular_kd_batched(
    solver: newton.solvers.SolverVBD,
    template_fruiting_fixed_joints: Iterable[tuple[int, str]],
    label_kd: dict[str, float] | None = None,
    *,
    label_kd_per_env: Sequence[Mapping[str, float]] | None = None,
    num_envs: int,
    joints_per_world: int,
) -> dict[str, list[int]]:
    """Vectorized per-role angular kd patch across every env of a batched SolverVBD.

    Same substring-match semantics as :func:`set_fruiting_joint_angular_kd`, applied to
    ``template_fruiting_fixed_joints`` (world-0 labels) via a single ``wp.launch``.
    Pass either ``label_kd`` (broadcast to all envs) or ``label_kd_per_env`` (one map
    per env, same keys). Requires uniform topology across envs.

    Args:
        solver: Constructed VBD solver whose ``joint_penalty_kd`` array will be patched.
        template_fruiting_fixed_joints: World-0 ``(joint_index, label)`` pairs.
        label_kd: Substring label -> absolute angular damping [N·m·s/rad] (broadcast).
        label_kd_per_env: Per-env label->kd maps (mutually exclusive with ``label_kd``).
        num_envs: Number of VBD worlds in the batched model.
        joints_per_world: Joint count per world (from ``BatchedEnvLayout.joints_per_world``
            or ``model.joint_world_start`` gap).

    Returns:
        ``{label_kd_key: [global_joint_index, ...]}`` with sorted indices per key
        (includes every env's copy of each matched template joint).

    Raises:
        ValueError: Negative ``kd``, ambiguous/unmatched keys, wrong batch dimensions,
            or joints without an angular constraint slot.
    """
    per_env_label_kd = _normalize_batched_label_kd(
        label_kd, label_kd_per_env, num_envs=num_envs
    )
    matched_by_key = _match_fruiting_joint_labels(
        template_fruiting_fixed_joints, per_env_label_kd[0]
    )
    if not matched_by_key:
        return {}
    # Validate non-negative kd on every env (env0 already checked by match).
    for env_idx, env_kd in enumerate(per_env_label_kd[1:], start=1):
        _match_fruiting_joint_labels(
            template_fruiting_fixed_joints,
            env_kd,
            param_name=f"label_kd_per_env[{env_idx}]",
        )

    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}.")
    if joints_per_world < 1:
        raise ValueError(f"joints_per_world must be >= 1, got {joints_per_world}.")

    model = solver.model
    if num_envs * joints_per_world != int(model.joint_count):
        raise ValueError(
            f"batched joint layout mismatch: num_envs={num_envs} * "
            f"joints_per_world={joints_per_world} != model.joint_count="
            f"{model.joint_count}."
        )

    if not hasattr(solver, "joint_penalty_kd"):
        raise RuntimeError(
            "SolverVBD joint_penalty_kd is not initialized; "
            "ensure the solver was constructed with rigid bodies present."
        )

    template_indices = sorted(
        {j for indices in matched_by_key.values() for j in indices}
    )
    _validate_template_joint_angular_slots(solver, template_indices)

    template_idx_np = np.asarray(template_indices, dtype=np.int32)
    n_templates = int(len(template_idx_np))
    kd_np = _batched_kd_values_from_per_env(
        matched_by_key, per_env_label_kd, template_idx_np
    )

    device = solver.joint_penalty_kd.device
    ang_slot = newton.solvers.SolverVBD.JointSlot.ANGULAR
    template_idx_wp = wp.array(template_idx_np, dtype=wp.int32, device=device)
    kd_wp = wp.array(kd_np, dtype=wp.float32, device=device)

    wp.launch(
        _apply_batched_joint_kd_kernel,
        dim=(int(num_envs), n_templates),
        inputs=[
            solver.joint_constraint_start,
            template_idx_wp,
            kd_wp,
            int(joints_per_world),
            n_templates,
            int(ang_slot),
            solver.joint_penalty_kd,
        ],
        device=device,
    )

    return _global_matched_joint_indices(
        matched_by_key, num_envs=num_envs, joints_per_world=joints_per_world
    )


def _global_matched_joint_indices(
    matched_by_key: dict[str, list[int]],
    *,
    num_envs: int,
    joints_per_world: int,
) -> dict[str, list[int]]:
    global_matched: dict[str, list[int]] = {}
    for key, indices in matched_by_key.items():
        global_indices: list[int] = []
        for w in range(int(num_envs)):
            base = w * int(joints_per_world)
            global_indices.extend(base + int(j) for j in indices)
        global_matched[key] = sorted(global_indices)
    return global_matched


def set_fruiting_joint_linear_kd(
    solver: newton.solvers.SolverVBD,
    fruiting_fixed_joints: Iterable[tuple[int, str]],
    label_kd: dict[str, float],
) -> dict[str, list[int]]:
    """Patch per-joint linear AVBD damping on an existing ``SolverVBD``.

    Same substring-match semantics as :func:`set_fruiting_joint_angular_kd`, but
    overwrites ``solver.joint_penalty_kd`` at the linear slot
    (``JointSlot.LINEAR``) for selected FIXED joints.

    Args:
        solver: Constructed VBD solver whose ``joint_penalty_kd`` array will be
            patched in place.
        fruiting_fixed_joints: ``(joint_index, label)`` pairs from scene build.
        label_kd: Substring label -> absolute linear damping [N·s/m].

    Returns:
        ``{label_kd_key: [joint_index, ...]}`` with sorted joint indices per key.
    """
    matched_by_key = _match_fruiting_joint_labels(
        fruiting_fixed_joints,
        label_kd,
        value_label="linear kd",
    )
    if not matched_by_key:
        return {}

    if not hasattr(solver, "joint_penalty_kd"):
        raise RuntimeError(
            "SolverVBD joint_penalty_kd is not initialized; "
            "ensure the solver was constructed with rigid bodies present."
        )

    template_indices = sorted(
        {j for indices in matched_by_key.values() for j in indices}
    )
    _validate_template_joint_linear_slots(solver, template_indices)

    jc_start = solver.joint_constraint_start.numpy()
    kd_np = solver.joint_penalty_kd.numpy().copy()
    lin_slot = newton.solvers.SolverVBD.JointSlot.LINEAR

    for key, joint_indices in matched_by_key.items():
        kd_val = float(label_kd[key])
        for joint_index in joint_indices:
            c0 = int(jc_start[joint_index])
            kd_np[c0 + lin_slot] = kd_val

    solver.joint_penalty_kd.assign(kd_np)
    return matched_by_key


def set_fruiting_joint_linear_kd_batched(
    solver: newton.solvers.SolverVBD,
    template_fruiting_fixed_joints: Iterable[tuple[int, str]],
    label_kd: dict[str, float] | None = None,
    *,
    label_kd_per_env: Sequence[Mapping[str, float]] | None = None,
    num_envs: int,
    joints_per_world: int,
) -> dict[str, list[int]]:
    """Vectorized per-role linear kd patch across every env of a batched SolverVBD."""
    per_env_label_kd = _normalize_batched_label_kd(
        label_kd, label_kd_per_env, num_envs=num_envs
    )
    matched_by_key = _match_fruiting_joint_labels(
        template_fruiting_fixed_joints,
        per_env_label_kd[0],
        value_label="linear kd",
    )
    if not matched_by_key:
        return {}
    for env_idx, env_kd in enumerate(per_env_label_kd[1:], start=1):
        _match_fruiting_joint_labels(
            template_fruiting_fixed_joints,
            env_kd,
            param_name=f"label_kd_per_env[{env_idx}]",
            value_label="linear kd",
        )

    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}.")
    if joints_per_world < 1:
        raise ValueError(f"joints_per_world must be >= 1, got {joints_per_world}.")

    model = solver.model
    if num_envs * joints_per_world != int(model.joint_count):
        raise ValueError(
            f"batched joint layout mismatch: num_envs={num_envs} * "
            f"joints_per_world={joints_per_world} != model.joint_count="
            f"{model.joint_count}."
        )

    if not hasattr(solver, "joint_penalty_kd"):
        raise RuntimeError(
            "SolverVBD joint_penalty_kd is not initialized; "
            "ensure the solver was constructed with rigid bodies present."
        )

    template_indices = sorted(
        {j for indices in matched_by_key.values() for j in indices}
    )
    _validate_template_joint_linear_slots(solver, template_indices)

    template_idx_np = np.asarray(template_indices, dtype=np.int32)
    n_templates = int(len(template_idx_np))
    kd_np = _batched_kd_values_from_per_env(
        matched_by_key, per_env_label_kd, template_idx_np
    )

    device = solver.joint_penalty_kd.device
    lin_slot = newton.solvers.SolverVBD.JointSlot.LINEAR
    template_idx_wp = wp.array(template_idx_np, dtype=wp.int32, device=device)
    kd_wp = wp.array(kd_np, dtype=wp.float32, device=device)

    wp.launch(
        _apply_batched_joint_kd_kernel,
        dim=(int(num_envs), n_templates),
        inputs=[
            solver.joint_constraint_start,
            template_idx_wp,
            kd_wp,
            int(joints_per_world),
            n_templates,
            int(lin_slot),
            solver.joint_penalty_kd,
        ],
        device=device,
    )

    return _global_matched_joint_indices(
        matched_by_key, num_envs=num_envs, joints_per_world=joints_per_world
    )


@wp.kernel(enable_backward=False)
def _apply_batched_joint_angular_kp_kernel(
    joint_constraint_start: wp.array(dtype=wp.int32),
    template_joint_indices: wp.array(dtype=wp.int32),
    kp_values: wp.array(dtype=wp.float32),
    joints_per_world: int,
    angular_slot: int,
    joint_penalty_k: wp.array(dtype=wp.float32),
    joint_penalty_k_min: wp.array(dtype=wp.float32),
    joint_penalty_k_max: wp.array(dtype=wp.float32),
):
    """Patch angular kp (and AVBD k bounds) for one (env, matched-template-joint) pair."""
    w, k = wp.tid()
    global_joint = w * joints_per_world + template_joint_indices[k]
    c0 = joint_constraint_start[global_joint]
    idx = c0 + angular_slot
    kp = kp_values[k]
    joint_penalty_k[idx] = kp
    joint_penalty_k_min[idx] = wp.min(joint_penalty_k_min[idx], kp)
    joint_penalty_k_max[idx] = wp.max(joint_penalty_k_max[idx], kp)


def set_fruiting_joint_angular_kp(
    solver: newton.solvers.SolverVBD,
    fruiting_fixed_joints: Iterable[tuple[int, str]],
    label_kp: dict[str, float],
) -> dict[str, list[int]]:
    """Patch per-joint angular AVBD stiffness on an existing ``SolverVBD``.

    Same substring-match semantics as :func:`set_fruiting_joint_angular_kd`, but
    overwrites ``solver.joint_penalty_k`` at the angular slot and widens
    ``joint_penalty_k_min`` / ``joint_penalty_k_max`` so AVBD maintenance does not
    immediately clamp the new value back to the constructor ceiling.

    Call after ``make_fruiting_solver_vbd(model)`` and before ``solver.step()``.

    Args:
        solver: Constructed VBD solver whose joint penalty-k arrays will be patched.
        fruiting_fixed_joints: ``(joint_index, label)`` pairs from scene build.
        label_kp: Substring label -> absolute angular penalty stiffness [N·m/rad].

    Returns:
        ``{label_kp_key: [joint_index, ...]}`` with sorted joint indices per key.

    Raises:
        ValueError: Negative ``kp``, ambiguous multi-key match on one joint,
            or a key that matches no joint.
    """
    matched_by_key = _match_fruiting_joint_labels(
        fruiting_fixed_joints,
        label_kp,
        param_name="label_kp",
        value_label="angular kp",
    )
    if not matched_by_key:
        return {}

    for name in ("joint_penalty_k", "joint_penalty_k_min", "joint_penalty_k_max"):
        if not hasattr(solver, name):
            raise RuntimeError(
                f"SolverVBD {name} is not initialized; "
                "ensure the solver was constructed with rigid bodies present."
            )

    template_indices = sorted(
        {j for indices in matched_by_key.values() for j in indices}
    )
    _validate_template_joint_angular_slots(solver, template_indices)

    jc_start = solver.joint_constraint_start.numpy()
    k_np = solver.joint_penalty_k.numpy().copy()
    k_min_np = solver.joint_penalty_k_min.numpy().copy()
    k_max_np = solver.joint_penalty_k_max.numpy().copy()
    ang_slot = newton.solvers.SolverVBD.JointSlot.ANGULAR

    for key, joint_indices in matched_by_key.items():
        kp_val = float(label_kp[key])
        for joint_index in joint_indices:
            c0 = int(jc_start[joint_index])
            _patch_k_constraint_slot(
                k_np, k_min_np, k_max_np, c0 + ang_slot, kp_val
            )

    solver.joint_penalty_k.assign(k_np)
    solver.joint_penalty_k_min.assign(k_min_np)
    solver.joint_penalty_k_max.assign(k_max_np)
    return matched_by_key


def set_fruiting_joint_angular_kp_batched(
    solver: newton.solvers.SolverVBD,
    template_fruiting_fixed_joints: Iterable[tuple[int, str]],
    label_kp: dict[str, float],
    *,
    num_envs: int,
    joints_per_world: int,
) -> dict[str, list[int]]:
    """Vectorized per-role angular kp patch across every env of a batched SolverVBD.

    Same substring-match semantics as :func:`set_fruiting_joint_angular_kp`, applied to
    world-0 template labels and broadcast via a single ``wp.launch``.

    Args:
        solver: Constructed VBD solver whose joint penalty-k arrays will be patched.
        template_fruiting_fixed_joints: World-0 ``(joint_index, label)`` pairs.
        label_kp: Substring label -> absolute angular penalty stiffness [N·m/rad].
        num_envs: Number of VBD worlds in the batched model.
        joints_per_world: Joint count per world.

    Returns:
        ``{label_kp_key: [global_joint_index, ...]}`` with sorted indices per key.

    Raises:
        ValueError: Negative ``kp``, ambiguous/unmatched keys, wrong batch dimensions,
            or joints without an angular constraint slot.
    """
    matched_by_key = _match_fruiting_joint_labels(
        template_fruiting_fixed_joints,
        label_kp,
        param_name="label_kp",
        value_label="angular kp",
    )
    if not matched_by_key:
        return {}

    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}.")
    if joints_per_world < 1:
        raise ValueError(f"joints_per_world must be >= 1, got {joints_per_world}.")

    model = solver.model
    if num_envs * joints_per_world != int(model.joint_count):
        raise ValueError(
            f"batched joint layout mismatch: num_envs={num_envs} * "
            f"joints_per_world={joints_per_world} != model.joint_count="
            f"{model.joint_count}."
        )

    for name in ("joint_penalty_k", "joint_penalty_k_min", "joint_penalty_k_max"):
        if not hasattr(solver, name):
            raise RuntimeError(
                f"SolverVBD {name} is not initialized; "
                "ensure the solver was constructed with rigid bodies present."
            )

    template_indices = sorted(
        {j for indices in matched_by_key.values() for j in indices}
    )
    _validate_template_joint_angular_slots(solver, template_indices)

    template_idx_np = np.asarray(template_indices, dtype=np.int32)
    kp_by_template = {
        j: float(label_kp[key])
        for key, indices in matched_by_key.items()
        for j in indices
    }
    kp_np = np.asarray(
        [kp_by_template[int(j)] for j in template_idx_np], dtype=np.float32
    )

    device = solver.joint_penalty_k.device
    ang_slot = newton.solvers.SolverVBD.JointSlot.ANGULAR
    template_idx_wp = wp.array(template_idx_np, dtype=wp.int32, device=device)
    kp_wp = wp.array(kp_np, dtype=wp.float32, device=device)

    wp.launch(
        _apply_batched_joint_angular_kp_kernel,
        dim=(int(num_envs), int(len(template_idx_np))),
        inputs=[
            solver.joint_constraint_start,
            template_idx_wp,
            kp_wp,
            int(joints_per_world),
            int(ang_slot),
            solver.joint_penalty_k,
            solver.joint_penalty_k_min,
            solver.joint_penalty_k_max,
        ],
        device=device,
    )

    global_matched: dict[str, list[int]] = {}
    for key, indices in matched_by_key.items():
        global_indices: list[int] = []
        for w in range(int(num_envs)):
            base = w * int(joints_per_world)
            global_indices.extend(base + int(j) for j in indices)
        global_matched[key] = sorted(global_indices)

    return global_matched


def set_fruiting_joint_linear_kp(
    solver: newton.solvers.SolverVBD,
    fruiting_fixed_joints: Iterable[tuple[int, str]],
    label_kp: dict[str, float],
) -> dict[str, list[int]]:
    """Patch per-joint linear AVBD stiffness on an existing ``SolverVBD``.

    Same substring-match semantics as :func:`set_fruiting_joint_angular_kp`, but
    overwrites ``solver.joint_penalty_k`` at the linear slot
    (``JointSlot.LINEAR``) for selected FIXED joints.

    Call after ``make_fruiting_solver_vbd(model)`` and before ``solver.step()``.

    Args:
        solver: Constructed VBD solver whose joint penalty-k arrays will be patched.
        fruiting_fixed_joints: ``(joint_index, label)`` pairs from scene build.
        label_kp: Substring label -> absolute linear penalty stiffness [N/m].

    Returns:
        ``{label_kp_key: [joint_index, ...]}`` with sorted joint indices per key.
    """
    matched_by_key = _match_fruiting_joint_labels(
        fruiting_fixed_joints,
        label_kp,
        param_name="label_kp",
        value_label="linear kp",
    )
    if not matched_by_key:
        return {}

    for name in ("joint_penalty_k", "joint_penalty_k_min", "joint_penalty_k_max"):
        if not hasattr(solver, name):
            raise RuntimeError(
                f"SolverVBD {name} is not initialized; "
                "ensure the solver was constructed with rigid bodies present."
            )

    template_indices = sorted(
        {j for indices in matched_by_key.values() for j in indices}
    )
    _validate_template_joint_linear_slots(solver, template_indices)

    jc_start = solver.joint_constraint_start.numpy()
    k_np = solver.joint_penalty_k.numpy().copy()
    k_min_np = solver.joint_penalty_k_min.numpy().copy()
    k_max_np = solver.joint_penalty_k_max.numpy().copy()
    lin_slot = newton.solvers.SolverVBD.JointSlot.LINEAR

    for key, joint_indices in matched_by_key.items():
        kp_val = float(label_kp[key])
        for joint_index in joint_indices:
            c0 = int(jc_start[joint_index])
            _patch_k_constraint_slot(
                k_np, k_min_np, k_max_np, c0 + lin_slot, kp_val
            )

    solver.joint_penalty_k.assign(k_np)
    solver.joint_penalty_k_min.assign(k_min_np)
    solver.joint_penalty_k_max.assign(k_max_np)
    return matched_by_key


def set_fruiting_joint_linear_kp_batched(
    solver: newton.solvers.SolverVBD,
    template_fruiting_fixed_joints: Iterable[tuple[int, str]],
    label_kp: dict[str, float],
    *,
    num_envs: int,
    joints_per_world: int,
) -> dict[str, list[int]]:
    """Vectorized per-role linear kp patch across every env of a batched SolverVBD.

    Same substring-match semantics as :func:`set_fruiting_joint_linear_kp`, applied to
    world-0 template labels and broadcast via a single ``wp.launch``.
    """
    matched_by_key = _match_fruiting_joint_labels(
        template_fruiting_fixed_joints,
        label_kp,
        param_name="label_kp",
        value_label="linear kp",
    )
    if not matched_by_key:
        return {}

    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}.")
    if joints_per_world < 1:
        raise ValueError(f"joints_per_world must be >= 1, got {joints_per_world}.")

    model = solver.model
    if num_envs * joints_per_world != int(model.joint_count):
        raise ValueError(
            f"batched joint layout mismatch: num_envs={num_envs} * "
            f"joints_per_world={joints_per_world} != model.joint_count="
            f"{model.joint_count}."
        )

    for name in ("joint_penalty_k", "joint_penalty_k_min", "joint_penalty_k_max"):
        if not hasattr(solver, name):
            raise RuntimeError(
                f"SolverVBD {name} is not initialized; "
                "ensure the solver was constructed with rigid bodies present."
            )

    template_indices = sorted(
        {j for indices in matched_by_key.values() for j in indices}
    )
    _validate_template_joint_linear_slots(solver, template_indices)

    template_idx_np = np.asarray(template_indices, dtype=np.int32)
    kp_by_template = {
        j: float(label_kp[key])
        for key, indices in matched_by_key.items()
        for j in indices
    }
    kp_np = np.asarray(
        [kp_by_template[int(j)] for j in template_idx_np], dtype=np.float32
    )

    device = solver.joint_penalty_k.device
    lin_slot = newton.solvers.SolverVBD.JointSlot.LINEAR
    template_idx_wp = wp.array(template_idx_np, dtype=wp.int32, device=device)
    kp_wp = wp.array(kp_np, dtype=wp.float32, device=device)

    wp.launch(
        _apply_batched_joint_angular_kp_kernel,
        dim=(int(num_envs), int(len(template_idx_np))),
        inputs=[
            solver.joint_constraint_start,
            template_idx_wp,
            kp_wp,
            int(joints_per_world),
            int(lin_slot),
            solver.joint_penalty_k,
            solver.joint_penalty_k_min,
            solver.joint_penalty_k_max,
        ],
        device=device,
    )

    global_matched: dict[str, list[int]] = {}
    for key, indices in matched_by_key.items():
        global_indices: list[int] = []
        for w in range(int(num_envs)):
            base = w * int(joints_per_world)
            global_indices.extend(base + int(j) for j in indices)
        global_matched[key] = sorted(global_indices)

    return global_matched


def _scene_states_from_model(model: newton.Model) -> tuple[Any, Any, Any, newton.solvers.SolverVBD]:
    solver = make_fruiting_solver_vbd(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    return state_0, state_1, control, solver

def _make_rod_geometry(
    start: wp.vec3,
    direction: tuple[float, float, float],
    length: float,
    num_segments: int,
) -> tuple[list[wp.vec3], list[wp.quat]]:
    """Build straight-rod positions and per-segment quaternions."""
    d = wp.normalize(wp.vec3(*direction))
    points = [start + d * (length * i / num_segments) for i in range(num_segments + 1)]
    quats = newton.utils.create_parallel_transport_cable_quaternions(points)
    return points, quats


def _primary_radial_surface_offset_world(
    primary_direction: tuple[float, float, float],
    spur_direction: tuple[float, float, float],
    primary_radius: float,
) -> wp.vec3:
    """World-frame vector from the primary centerline to its surface toward the spur."""
    axis = wp.normalize(wp.vec3(*primary_direction))
    d = wp.normalize(wp.vec3(*spur_direction))
    radial = d - axis * wp.dot(d, axis)
    radial_len = wp.length(radial)
    if radial_len < 1e-6:
        return wp.vec3(0.0, 0.0, 0.0)
    return (radial / radial_len) * float(primary_radius)


def _primary_spur_parent_body_index(num_segments: int, fraction: float) -> int:
    """Index of the primary body whose tip hosts the spur branch."""
    point_index = int(round(float(fraction) * num_segments))
    point_index = max(1, min(num_segments, point_index))
    return point_index - 1


def _connect_world_to_rod_base(
    builder: newton.ModelBuilder,
    world_pos: wp.vec3,
    body: int,
    key: str,
) -> int:
    """World-fixed support at a rod segment base (local ``(0,0,0)``)."""
    return builder.add_joint_fixed(
        parent=-1,
        child=body,
        parent_xform=wp.transform(world_pos, wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        label=key,
    )


def _connect_world_to_rod_tip(
    builder: newton.ModelBuilder,
    world_pos: wp.vec3,
    body: int,
    seg_length: float,
    key: str,
) -> int:
    """World-fixed support at a rod segment tip (local ``(0,0,seg_length)``)."""
    return builder.add_joint_fixed(
        parent=-1,
        child=body,
        parent_xform=wp.transform(world_pos, wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, seg_length), wp.quat_identity()),
        label=key,
    )


def _connect_rod_tip_to_base(
    builder: newton.ModelBuilder,
    parent_body: int,
    parent_seg_length: float,
    child_body: int,
    key: str,
    *,
    parent_local_offset: wp.vec3 | None = None,
) -> int:
    """Add a fixed joint connecting the tip of one rod to the base of the next.

    In rod convention, the tip of a segment body is at local (0, 0, seg_length).
    The base of the next segment body is at local (0, 0, 0).
    """
    offset = (
        parent_local_offset if parent_local_offset is not None else wp.vec3(0.0, 0.0, 0.0)
    )
    anchor = wp.vec3(0.0, 0.0, parent_seg_length) + offset
    return builder.add_joint_fixed(
        parent=parent_body,
        child=child_body,
        parent_xform=wp.transform(anchor, wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        label=key,
    )


def _connect_rod_tip_to_apple(
    builder: newton.ModelBuilder,
    stem_tip_body: int,
    stem_tip_quat: wp.quat,
    stem_seg_length: float,
    segment_dir_world: wp.vec3,
    apple_radius: float,
    apple_body: int,
    apple_quat: wp.quat,
    key: str,
) -> int:
    """Add a fixed joint attaching the apple to the distal end of the last rod segment.

    The parent anchor is the capsule tip in the stem-tip body's local frame
    (same construction as ``example_apple_stem.py``). The child anchor is offset
    from the apple COM toward the stem by one radius so the stem-side pole of
    the sphere meets the rod tip instead of burying the COM at the junction.
    The child offset is expressed in the apple body's local frame (rotated by
    ``apple_quat`` from world).
    """
    segment_vector_world = segment_dir_world * stem_seg_length
    parent_local_anchor = wp.quat_rotate(
        wp.quat_inverse(stem_tip_quat), segment_vector_world
    )
    stem_toward_apple_world = segment_dir_world * (-apple_radius)
    child_local_anchor = wp.quat_rotate(wp.quat_inverse(apple_quat), stem_toward_apple_world)
    return builder.add_joint_fixed(
        parent=stem_tip_body,
        child=apple_body,
        parent_xform=wp.transform(parent_local_anchor, wp.quat_identity()),
        child_xform=wp.transform(child_local_anchor, wp.quat_identity()),
        label=key,
    )
