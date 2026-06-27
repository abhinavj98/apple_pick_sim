"""Debug visualization for batched coupled fruiting scenes."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

import newton

from apple_pick_sim.coupling_force_debug import read_tcp_wrench
from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.digital_twin.record import fruiting_tree_fixed_joints
from apple_pick_sim.fruiting_system.gripper_proxy_shape import gripper_proxy_clearance
from apple_pick_sim.fruiting_system.params import GripperProxyConfig
from apple_pick_sim.fruiting_system.scene import fixed_joint_anchors_world
from apple_pick_sim.tcp_force_viz import (
    _clear_logged_arrow,
    _viewer_device,
    force_arrow_segment_arrays,
)

_ENDPOINT_COLOR = (1.0, 0.0, 0.0)


def _viewer_env_spacing(scene: Any, layout: BatchedEnvLayout) -> tuple[float, float, float]:
    """Spacing used by ``viewer.set_world_offsets`` (stored on scene, not layout)."""
    spacing = getattr(scene, "env_spacing", None)
    if spacing is not None:
        return tuple(float(v) for v in spacing)
    return layout.env_spacing


def _viewer_world_origin(scene: Any, layout: BatchedEnvLayout, world: int) -> np.ndarray:
    """Per-env translation applied by the Newton viewer for replicated worlds."""
    spacing = _viewer_env_spacing(scene, layout)
    offsets = newton.utils.compute_world_offsets(
        layout.num_envs,
        spacing,
        up_axis=newton.Axis.Z,
    )
    return np.asarray(offsets[int(world)], dtype=np.float64).reshape(3)


def _world_position(
    scene: Any,
    layout: BatchedEnvLayout,
    world: int,
    local_pos: np.ndarray | tuple[float, float, float],
) -> np.ndarray:
    """Map template/sim body position to viewer world coordinates."""
    pos = np.asarray(local_pos, dtype=np.float64).reshape(3)
    return pos + _viewer_world_origin(scene, layout, world)


def _apple_marker_position(
    scene: Any,
    layout: BatchedEnvLayout,
    world: int,
    cable_bq: np.ndarray,
) -> np.ndarray:
    """Apple-surface marker in viewer coordinates (offset from COM so it is visible)."""
    cable = scene.cable
    idx = layout.apple_body_indices[world]
    com = _world_position(scene, layout, world, cable_bq[idx, :3])
    params = getattr(cable, "params", None)
    apple_r = float(getattr(params, "apple_radius", 0.04) or 0.04)
    return com + np.array([0.0, 0.0, apple_r * 1.05], dtype=np.float64)


def _proxy_marker_position(
    scene: Any,
    layout: BatchedEnvLayout,
    world: int,
    cable_bq: np.ndarray,
) -> np.ndarray:
    """Proxy / grasp-surface marker in viewer coordinates."""
    cable = scene.cable
    idx = layout.proxy_body_indices[world]
    com = _world_position(scene, layout, world, cable_bq[idx, :3])
    vis = getattr(cable, "gripper_proxy_vis_offset", (0.0, 0.0, 0.0))
    off = np.asarray(vis, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(off)) > 1e-9:
        return com + off
    cfg = getattr(cable, "gripper_proxy_config", None)
    clearance = 0.04
    if isinstance(cfg, GripperProxyConfig):
        clearance = gripper_proxy_clearance(cfg)
    elif cfg is not None:
        ext = getattr(cfg, "box_half_extents", None)
        if ext is not None:
            clearance = float(ext[2])
    return com + np.array([0.0, 0.0, clearance * 1.05], dtype=np.float64)


def _cross_segment_arrays(
    origin: np.ndarray,
    *,
    half_extent: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Three orthogonal line segments centered on ``origin``."""
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    h = float(half_extent)
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for axis in range(3):
        delta = np.zeros(3, dtype=np.float64)
        delta[axis] = h
        starts.append(o - delta)
        ends.append(o + delta)
    return np.stack(starts), np.stack(ends)


def _log_endpoint_crosses(
    viewer: Any,
    name: str,
    origins: np.ndarray,
    *,
    half_extent: float,
    device: str,
    hidden: bool = False,
) -> None:
    """Draw red XYZ crosses via ``log_lines`` (same path as force arrows)."""
    log_lines = getattr(viewer, "log_lines", None)
    if log_lines is None:
        return

    starts_list: list[np.ndarray] = []
    ends_list: list[np.ndarray] = []
    for origin in origins:
        starts, ends = _cross_segment_arrays(origin, half_extent=half_extent)
        starts_list.append(starts)
        ends_list.append(ends)

    if not starts_list:
        return

    starts_np = np.vstack(starts_list)
    ends_np = np.vstack(ends_list)
    colors_np = np.tile(_ENDPOINT_COLOR, (starts_np.shape[0], 1)).astype(np.float32)
    wp_starts = wp.array(starts_np, dtype=wp.vec3, device=device)
    wp_ends = wp.array(ends_np, dtype=wp.vec3, device=device)
    wp_colors = wp.array(colors_np, dtype=wp.vec3, device=device)
    log_lines(name, wp_starts, wp_ends, wp_colors, hidden=hidden)


def log_batched_woody_part_endpoints(
    viewer: Any,
    scene: Any,
    layout: BatchedEnvLayout,
    *,
    name: str = "/debug/batched_woody_endpoints",
    radius: float = 0.05,
    hidden: bool = False,
) -> None:
    """Draw red spheres at woody fixed-joint parent anchors for each env."""
    log_points = getattr(viewer, "log_points", None)
    if log_points is None:
        return

    cable = scene.cable
    if not getattr(cable, "fruiting_fixed_joints", None):
        return
    joints = fruiting_tree_fixed_joints(cable)
    if not joints:
        return

    fallback = str(getattr(getattr(cable, "model", None), "device", "cpu"))
    dev = _viewer_device(viewer, fallback=fallback)
    body_q = cable.state_0.body_q
    model = cable.model

    positions_list: list[np.ndarray] = []
    for w in range(layout.num_envs):
        pairs = [(layout.joint_index(w, ji), label) for ji, label in joints]
        parent_flat, _child_flat = fixed_joint_anchors_world(model, body_q, pairs)
        parent = parent_flat.reshape(-1, 3)
        for pos in parent:
            positions_list.append(_world_position(scene, layout, w, pos))

    if not positions_list:
        return

    positions_np = np.stack(positions_list, axis=0).astype(np.float32)
    n = positions_np.shape[0]
    points = wp.array(positions_np, dtype=wp.vec3, device=dev)
    radii = wp.full(n, float(radius), dtype=wp.float32, device=dev)
    colors = wp.full(n, wp.vec3(*_ENDPOINT_COLOR), dtype=wp.vec3, device=dev)
    log_points(name, points, radii=radii, colors=colors, hidden=hidden)


def log_batched_tcp_force_arrows(
    viewer: Any,
    scene: Any,
    layout: BatchedEnvLayout,
    *,
    name: str = "/debug/tcp_force_arrow",
    scale_per_newton: float = 0.02,
    gain: float = 1.0,
    min_length: float = 0.0,
    max_length: float = 0.0,
    force_threshold: float = 1e-6,
    hidden: bool = False,
) -> None:
    """Draw harvested TCP force arrows for every env in a batched coupled scene."""
    log_lines = getattr(viewer, "log_lines", None)
    if log_lines is None:
        return

    if scene.proxy_forces is None:
        _clear_logged_arrow(viewer, name, hidden=hidden)
        return

    fallback = None
    model = getattr(scene, "robot_model", None)
    if model is not None:
        fallback = str(model.device)
    dev = _viewer_device(viewer, fallback=fallback)

    robot_bq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)
    starts_list: list[np.ndarray] = []
    ends_list: list[np.ndarray] = []
    colors_list: list[np.ndarray] = []

    for w in range(layout.num_envs):
        tcp_idx = layout.tcp_body_indices[w]
        wrench = read_tcp_wrench(scene.proxy_forces, tcp_idx)
        origin = _world_position(scene, layout, w, robot_bq[tcp_idx, :3])
        segments = force_arrow_segment_arrays(
            origin,
            wrench[:3],
            scale_per_newton=scale_per_newton,
            gain=gain,
            min_length=min_length,
            max_length=max_length,
            force_threshold=force_threshold,
        )
        if segments is None:
            continue
        starts, ends, colors = segments
        starts_list.append(starts)
        ends_list.append(ends)
        colors_list.append(colors)

    if not starts_list:
        _clear_logged_arrow(viewer, name, hidden=hidden)
        return

    starts_np = np.vstack(starts_list)
    ends_np = np.vstack(ends_list)
    colors_np = np.vstack(colors_list)
    wp_starts = wp.array(starts_np, dtype=wp.vec3, device=dev)
    wp_ends = wp.array(ends_np, dtype=wp.vec3, device=dev)
    wp_colors = wp.array(colors_np, dtype=wp.vec3, device=dev)
    log_arrows = getattr(viewer, "log_arrows", None)
    if log_arrows is not None:
        log_arrows(name, wp_starts, wp_ends, wp_colors, hidden=hidden)
    else:
        log_lines(name, wp_starts, wp_ends, wp_colors, hidden=hidden)


def log_batched_endpoints(
    viewer: Any,
    scene: Any,
    layout: BatchedEnvLayout,
    *,
    name_apple: str = "/debug/batched_apple_endpoints",
    name_proxy: str = "/debug/batched_proxy_endpoints",
    radius: float = 0.05,
    hidden: bool = False,
) -> None:
    """Draw red XYZ crosses at apple/proxy markers and red dots at woody parent anchors."""
    cable = scene.cable
    fallback = str(getattr(getattr(cable, "model", None), "device", "cpu"))
    dev = _viewer_device(viewer, fallback=fallback)
    cable_bq = cable.state_0.body_q.numpy().reshape(-1, 7)
    half_extent = max(float(radius), 0.02)

    if layout.template_apple_body is not None:
        apple_origins = np.stack(
            [_apple_marker_position(scene, layout, w, cable_bq) for w in range(layout.num_envs)],
            axis=0,
        )
        _log_endpoint_crosses(
            viewer,
            name_apple,
            apple_origins,
            half_extent=half_extent,
            device=dev,
            hidden=hidden,
        )

    proxy_origins = np.stack(
        [_proxy_marker_position(scene, layout, w, cable_bq) for w in range(layout.num_envs)],
        axis=0,
    )
    _log_endpoint_crosses(
        viewer,
        name_proxy,
        proxy_origins,
        half_extent=half_extent,
        device=dev,
        hidden=hidden,
    )

    log_batched_woody_part_endpoints(
        viewer,
        scene,
        layout,
        radius=float(radius),
        hidden=hidden,
    )
