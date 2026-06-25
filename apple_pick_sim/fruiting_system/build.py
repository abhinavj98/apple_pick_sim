"""Fruiting chain ModelBuilder helpers."""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any

import numpy as np
import warp as wp

import newton

from apple_pick_sim.fruiting_system.params import (
    FruitingSystemParams,
    GripperProxyConfig,
    RodParams,
)

FRUITING_VBD_RIGID_JOINT_KD = 5.0e-4


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
def _apply_all_chain_collision_filters(
    builder: newton.ModelBuilder,
    chain_body_indices: list[int],
) -> None:
    """Add shape collision filter pairs for every distinct pair of chain bodies."""
    bodies = chain_body_indices
    for i, body1 in enumerate(bodies):
        for body2 in bodies[i + 1 :]:
            for shape1 in builder.body_shapes.get(body1, []):
                for shape2 in builder.body_shapes.get(body2, []):
                    builder.add_shape_collision_filter_pair(shape1, shape2)



def _new_fruiting_builder() -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 0.8
    return builder


def _build_fruiting_chain_into_builder(
    builder: newton.ModelBuilder,
    params: FruitingSystemParams,
    base_pos: tuple[float, float, float],
) -> _FruitingChainArtifacts:
    """Populate rod chain + apple on ``builder`` (no articulation / finalize yet)."""
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
            stretch_damping=0.0,
            wrap_in_articulation=False,
            label=name,
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


def _apply_collision_filters_between_chain_groups(
    builder: newton.ModelBuilder,
    chain_a: list[int],
    chain_b: list[int],
) -> None:
    """Disable collisions between every body in ``chain_a`` and every body in ``chain_b``."""
    for body1 in chain_a:
        for body2 in chain_b:
            for shape1 in builder.body_shapes.get(body1, []):
                for shape2 in builder.body_shapes.get(body2, []):
                    builder.add_shape_collision_filter_pair(shape1, shape2)


def _finalize_fruiting_builder_joints(
    builder: newton.ModelBuilder,
    *,
    all_joints: list[int],
    chain_bodies: list[int],
    device: str,
    enable_self_collisions: bool,
    gripper_proxy_joints: tuple[int, ...] = (),
    extra_chain_groups_for_filters: tuple[list[int], ...] = (),
) -> newton.Model:
    """Finalize a cable ``Model``; FREE proxy joints live in separate articulations."""
    joint_list = sorted(all_joints)
    proxy_set = set(gripper_proxy_joints)
    if proxy_set:
        chain_joints = [j for j in joint_list if j not in proxy_set]
        builder.add_articulation(chain_joints)
        for pj in gripper_proxy_joints:
            builder.add_articulation([pj])
    else:
        builder.add_articulation(joint_list)
    if not enable_self_collisions:
        _apply_all_chain_collision_filters(builder, chain_bodies)
    for other in extra_chain_groups_for_filters:
        _apply_collision_filters_between_chain_groups(builder, chain_bodies, other)
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
) -> newton.Model:
    """Finalize the cable model; keep world-root proxy FREE joints out of the tree articulation."""
    proxy_joints = (gripper_proxy_joint,) if gripper_proxy_joint is not None else ()
    return _finalize_fruiting_builder_joints(
        builder,
        all_joints=artifacts.all_joints,
        chain_bodies=artifacts.chain_bodies,
        device=device,
        enable_self_collisions=enable_self_collisions,
        gripper_proxy_joints=proxy_joints,
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

    hx, hy, hz = config.box_half_extents
    weld_apple_center = _weld_apple_center(config, artifacts)
    weld_apple_quat = (
        None
        if config.weld_reference_quat is None
        else wp.quat(*config.weld_reference_quat)
    )
    clearance = max(hx, hy, hz)
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
    builder.add_shape_box(
        body=proxy_body,
        hx=hx,
        hy=hy,
        hz=hz,
        cfg=proxy_shape_cfg,
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
        "iterations": 50,
        "friction_epsilon": 0.1,
        "rigid_contact_k_start": 1.0e4,
        "rigid_joint_linear_k_start": 1.0e8,
        "rigid_joint_angular_k_start": 1.0e6,
        "rigid_joint_linear_kd": FRUITING_VBD_RIGID_JOINT_KD,
        "rigid_joint_angular_kd": FRUITING_VBD_RIGID_JOINT_KD,
    }
    kwargs.update(overrides)
    return newton.solvers.SolverVBD(model, **kwargs)


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


def _connect_rod_tip_to_base(
    builder: newton.ModelBuilder,
    parent_body: int,
    parent_seg_length: float,
    child_body: int,
    key: str,
) -> int:
    """Add a fixed joint connecting the tip of one rod to the base of the next.

    In rod convention, the tip of a segment body is at local (0, 0, seg_length).
    The base of the next segment body is at local (0, 0, 0).
    """
    return builder.add_joint_fixed(
        parent=parent_body,
        child=child_body,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, parent_seg_length), wp.quat_identity()),
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
