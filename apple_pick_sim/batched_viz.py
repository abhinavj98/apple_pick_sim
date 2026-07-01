"""Debug visualization for batched coupled fruiting scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from apple_pick_sim.batched_obs import BatchedObsBuffers

MARKER_APPLE_COM_TO_GRASP_COLOR = (1.0, 0.20, 0.75)
_SUPPORT_JUNCTION_PREFIX = "primary_support_"
# Defaults when ``--mark-endpoints`` is enabled (no separate CLI knobs).
MARK_ENDPOINT_ARROW_MIN_LENGTH_M = 0.20
MARK_ENDPOINT_ARROW_LENGTH_GAIN = 3.0
MARK_ENDPOINT_WOODY_POINT_RADIUS_M = 0.035
MARK_ENDPOINT_WOODY_FORCE_SCALE_PER_NEWTON = 0.02
MARK_ENDPOINT_WOODY_FORCE_MIN_LENGTH_M = 0.08
MARK_ENDPOINT_WOODY_FORCE_MAX_LENGTH_M = 1.5
_JUNCTION_POINT_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.0, 1.0, 0.55),   # e.g. primary_spur
    (0.0, 0.78, 1.0),   # spur_stem
    (1.0, 0.55, 0.0),   # stem_apple
    (0.85, 0.0, 1.0),
    (0.0, 0.85, 0.35),
    (1.0, 0.85, 0.0),
)


def _junction_label_short(label: str) -> str:
    return str(label).removeprefix("joint_")


def _is_viz_woody_junction_label(label: str) -> bool:
    short = _junction_label_short(label)
    if short.endswith("_gripper_proxy"):
        return False
    return not short.startswith(_SUPPORT_JUNCTION_PREFIX)


def fruiting_viz_joints(cable: Any) -> tuple[tuple[int, str], ...]:
    """Fruiting fixed joints for debug viz (excludes gripper-proxy welds and primary supports)."""
    return tuple(
        pair
        for pair in fruiting_tree_fixed_joints(cable)
        if _is_viz_woody_junction_label(pair[1])
    )


def junction_viz_color(label: str) -> tuple[float, float, float]:
    """Stable RGB per junction label for woody start-point coloring."""
    short = _junction_label_short(label)
    idx = abs(hash(short)) % len(_JUNCTION_POINT_COLORS)
    return _JUNCTION_POINT_COLORS[idx]


def format_mark_endpoints_color_legend(cable: Any) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    """Ordered (label, rgb) pairs for all ``--mark-endpoints`` markers."""
    entries: list[tuple[str, tuple[float, float, float]]] = [
        ("apple_com→grasp", MARKER_APPLE_COM_TO_GRASP_COLOR),
    ]
    for _ji, label in fruiting_viz_joints(cable):
        short = _junction_label_short(label)
        rgb = junction_viz_color(label)
        entries.append((f"{short} (woody start)", rgb))
        entries.append((f"{short} (woody force)", rgb))
    return tuple(entries)


def format_mark_endpoints_color_legend_line(cable: Any) -> str:
    """One-line ``label=(r,g,b)`` legend for console."""
    parts: list[str] = []
    for name, (r, g, b) in format_mark_endpoints_color_legend(cable):
        parts.append(f"{name}=({r:.2f},{g:.2f},{b:.2f})")
    return "; ".join(parts)


def print_mark_endpoints_color_legend(cable: Any) -> None:
    """Print color → label mapping (GL viewer has no on-screen text)."""
    print("  mark-endpoints color → label:", flush=True)
    for name, (r, g, b) in format_mark_endpoints_color_legend(cable):
        print(f"    ({r:.2f}, {g:.2f}, {b:.2f}) → {name}", flush=True)


def format_mark_endpoints_legend(cable: Any) -> str:
    """Deprecated alias for woody-only one-liner; prefer :func:`format_mark_endpoints_color_legend_line`."""
    return format_mark_endpoints_color_legend_line(cable)


def print_mark_endpoints_startup(cable: Any, *, status_every: int) -> None:
    """Print what ``--mark-endpoints`` draws and how console obs dumps are scheduled."""
    arrow_min = MARK_ENDPOINT_ARROW_MIN_LENGTH_M
    arrow_gain = MARK_ENDPOINT_ARROW_LENGTH_GAIN
    radius = MARK_ENDPOINT_WOODY_POINT_RADIUS_M
    print(
        "Endpoint markers (--mark-endpoints): "
        f"apple COM→grasp arrow (min {arrow_min:.2f} m, ×{arrow_gain:g} span); "
        f"colored spheres + matching force arrows at woody branch junctions "
        f"(point radius {radius:.3f} m; force scale "
        f"{MARK_ENDPOINT_WOODY_FORCE_SCALE_PER_NEWTON:.3f} m/N; primary supports hidden). "
        "GL viewer has no on-screen labels — color map below.",
        flush=True,
    )
    print_mark_endpoints_color_legend(cable)
    if status_every > 0:
        print(
            f"  Console obs dump: gather_batched_obs every {status_every} frame(s) "
            f"(same interval as --status-every).",
            flush=True,
        )


def print_batched_obs_debug(
    bufs: BatchedObsBuffers,
    *,
    frame: int,
    sim_time: float,
    cable: Any | None = None,
) -> None:
    """Host readback of gathered obs (debug only; not for RL hot path)."""
    tcp_pose = bufs.tcp_pose.numpy()
    tcp_vel = bufs.tcp_velocity.numpy()
    joint_q = bufs.joint_q.numpy()
    joint_qd = bufs.joint_qd.numpy()
    woody_force = bufs.woody_force.numpy()
    print(f"mark-endpoints obs frame={frame} t={sim_time:.2f}s", flush=True)
    if cable is not None:
        print_mark_endpoints_color_legend(cable)
    all_joints: tuple[tuple[int, str], ...] = ()
    if cable is not None:
        all_joints = fruiting_tree_fixed_joints(cable)
    apple_np = bufs.apple_pos.numpy()
    proxy_np = bufs.proxy_pos.numpy()
    for w in range(bufs.num_envs):
        pos = tcp_pose[w, :3]
        quat = tcp_pose[w, 3:7]
        vel = tcp_vel[w]
        apple_com = apple_np[w]
        proxy_com = proxy_np[w]
        grasp = _proxy_grasp_local(cable, proxy_com) if cable is not None else proxy_com
        print(
            f"  env{w}: tcp_pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
            f"quat=({quat[0]:+.3f},{quat[1]:+.3f},{quat[2]:+.3f},{quat[3]:+.3f}) "
            f"tcp_vel=[{vel[0]:+.3f},{vel[1]:+.3f},{vel[2]:+.3f}, "
            f"{vel[3]:+.3f},{vel[4]:+.3f},{vel[5]:+.3f}]",
            flush=True,
        )
        print(
            f"         apple_com=({apple_com[0]:.3f},{apple_com[1]:.3f},{apple_com[2]:.3f}) "
            f"grasp=({grasp[0]:.3f},{grasp[1]:.3f},{grasp[2]:.3f}) "
            f"proxy_com=({proxy_com[0]:.3f},{proxy_com[1]:.3f},{proxy_com[2]:.3f})",
            flush=True,
        )
        print(
            f"         joint_q={np.array2string(joint_q[w], precision=3, suppress_small=True)} "
            f"joint_qd={np.array2string(joint_qd[w], precision=3, suppress_small=True)}",
            flush=True,
        )
        if bufs.num_junctions > 0 and woody_force.shape[0] >= bufs.num_junctions:
            base = w * bufs.num_junctions
            for j_local in range(bufs.num_junctions):
                f = woody_force[base + j_local]
                if j_local < len(all_joints):
                    label = _junction_label_short(all_joints[j_local][1])
                else:
                    label = f"woody[{j_local}]"
                print(
                    f"         {label}_force=({f[0]:+.3f},{f[1]:+.3f},{f[2]:+.3f}) N",
                    flush=True,
                )


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


def _viewer_has_line_overlay(viewer: Any) -> bool:
    return getattr(viewer, "log_lines", None) is not None or getattr(
        viewer, "log_arrows", None
    ) is not None


def _viewer_has_points_overlay(viewer: Any) -> bool:
    return getattr(viewer, "log_points", None) is not None


def _apple_com_world(
    scene: Any,
    layout: BatchedEnvLayout,
    world: int,
    com: np.ndarray | tuple[float, float, float],
) -> np.ndarray:
    """Apple body COM in viewer world coordinates."""
    return _world_position(scene, layout, world, com)


def _proxy_grasp_local(
    cable: Any,
    proxy_com: np.ndarray | tuple[float, float, float],
) -> np.ndarray:
    """Grasp surface position in sim-local coordinates (proxy COM + vis offset)."""
    com = np.asarray(proxy_com, dtype=np.float64).reshape(3)
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


def _proxy_grasp_world(
    scene: Any,
    layout: BatchedEnvLayout,
    world: int,
    proxy_com: np.ndarray | tuple[float, float, float],
) -> np.ndarray:
    """Gripper-proxy grasp surface in viewer world coordinates."""
    local = _proxy_grasp_local(scene.cable, proxy_com)
    return _world_position(scene, layout, world, local)


def _apple_marker_from_com(
    scene: Any,
    layout: BatchedEnvLayout,
    world: int,
    com: np.ndarray | tuple[float, float, float],
) -> np.ndarray:
    """Apple-surface marker in viewer coordinates (offset from COM so it is visible)."""
    cable = scene.cable
    marker = _world_position(scene, layout, world, com)
    params = getattr(cable, "params", None)
    apple_r = float(getattr(params, "apple_radius", 0.04) or 0.04)
    return marker + np.array([0.0, 0.0, apple_r * 1.05], dtype=np.float64)


def _proxy_marker_from_com(
    scene: Any,
    layout: BatchedEnvLayout,
    world: int,
    com: np.ndarray | tuple[float, float, float],
) -> np.ndarray:
    """Proxy / grasp-surface marker in viewer coordinates."""
    return _proxy_grasp_world(scene, layout, world, com)


def _apple_marker_position(
    scene: Any,
    layout: BatchedEnvLayout,
    world: int,
    cable_bq: np.ndarray,
) -> np.ndarray:
    """Apple-surface marker in viewer coordinates (offset from COM so it is visible)."""
    idx = layout.apple_body_indices[world]
    return _apple_marker_from_com(scene, layout, world, cable_bq[idx, :3])


def _proxy_marker_position(
    scene: Any,
    layout: BatchedEnvLayout,
    world: int,
    cable_bq: np.ndarray,
) -> np.ndarray:
    """Proxy / grasp-surface marker in viewer coordinates."""
    idx = layout.proxy_body_indices[world]
    return _proxy_grasp_world(scene, layout, world, cable_bq[idx, :3])


def _log_force_arrow_batch(
    viewer: Any,
    name: str,
    starts_np: np.ndarray,
    ends_np: np.ndarray,
    colors_np: np.ndarray,
    *,
    device: str,
    hidden: bool,
) -> None:
    wp_starts = wp.array(starts_np, dtype=wp.vec3, device=device)
    wp_ends = wp.array(ends_np, dtype=wp.vec3, device=device)
    wp_colors = wp.array(colors_np, dtype=wp.vec3, device=device)
    log_arrows = getattr(viewer, "log_arrows", None)
    if log_arrows is not None:
        log_arrows(name, wp_starts, wp_ends, wp_colors, hidden=hidden)
    else:
        log_lines = getattr(viewer, "log_lines", None)
        if log_lines is not None:
            log_lines(name, wp_starts, wp_ends, wp_colors, hidden=hidden)


def _endpoint_arrow_segment(
    parent: np.ndarray,
    child: np.ndarray,
    *,
    min_length: float,
    length_gain: float = MARK_ENDPOINT_ARROW_LENGTH_GAIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale parent→child segment and enforce ``min_length`` for visibility."""
    p = np.asarray(parent, dtype=np.float64).reshape(3)
    c = np.asarray(child, dtype=np.float64).reshape(3)
    delta = c - p
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        direction = delta / length
    target = max(float(min_length), length * max(float(length_gain), 1.0))
    c = p + direction * target
    return p, c


_woody_arrow_segment = _endpoint_arrow_segment  # test / backward-compat alias


def log_batched_woody_start_points(
    viewer: Any,
    scene: Any,
    layout: BatchedEnvLayout,
    *,
    name: str = "/debug/batched_woody_start_points",
    radius: float = MARK_ENDPOINT_WOODY_POINT_RADIUS_M,
    hidden: bool = False,
    bufs: BatchedObsBuffers | None = None,
) -> None:
    """Draw colored spheres at woody fixed-joint parent (start) anchors."""
    log_points = getattr(viewer, "log_points", None)
    if log_points is None:
        return

    cable = scene.cable
    if not getattr(cable, "fruiting_fixed_joints", None):
        return
    all_joints = fruiting_tree_fixed_joints(cable)
    if not all_joints:
        return

    fallback = str(getattr(getattr(cable, "model", None), "device", "cpu"))
    dev = _viewer_device(viewer, fallback=fallback)
    point_radius = max(float(radius), 1e-4)

    positions_list: list[np.ndarray] = []
    colors_list: list[tuple[float, float, float]] = []

    if bufs is not None and bufs.num_junctions > 0:
        parent_all = bufs.woody_parent_pos.numpy().reshape(-1, 3)
        for w in range(layout.num_envs):
            for j_local, (_ji, label) in enumerate(all_joints):
                if not _is_viz_woody_junction_label(label):
                    continue
                flat = w * bufs.num_junctions + j_local
                positions_list.append(
                    _world_position(scene, layout, w, parent_all[flat])
                )
                colors_list.append(junction_viz_color(label))
    else:
        body_q = cable.state_0.body_q
        model = cable.model
        for w in range(layout.num_envs):
            pairs = [(layout.joint_index(w, ji), label) for ji, label in all_joints]
            parent_flat, _child_flat = fixed_joint_anchors_world(model, body_q, pairs)
            parent = parent_flat.reshape(-1, 3)
            for j_local, (_ji, label) in enumerate(all_joints):
                if not _is_viz_woody_junction_label(label):
                    continue
                positions_list.append(
                    _world_position(scene, layout, w, parent[j_local])
                )
                colors_list.append(junction_viz_color(label))

    if not positions_list:
        return

    positions_np = np.stack(positions_list, axis=0).astype(np.float32)
    n = positions_np.shape[0]
    points = wp.array(positions_np, dtype=wp.vec3, device=dev)
    radii = wp.full(n, point_radius, dtype=wp.float32, device=dev)
    colors = wp.array(
        np.asarray(colors_list, dtype=np.float32),
        dtype=wp.vec3,
        device=dev,
    )
    log_points(name, points, radii=radii, colors=colors, hidden=hidden)


def log_batched_woody_junction_force_arrows(
    viewer: Any,
    scene: Any,
    layout: BatchedEnvLayout,
    *,
    name: str = "/debug/batched_woody_junction_forces",
    scale_per_newton: float = MARK_ENDPOINT_WOODY_FORCE_SCALE_PER_NEWTON,
    gain: float = 1.0,
    min_length: float = MARK_ENDPOINT_WOODY_FORCE_MIN_LENGTH_M,
    max_length: float = MARK_ENDPOINT_WOODY_FORCE_MAX_LENGTH_M,
    force_threshold: float = 1e-6,
    hidden: bool = False,
    bufs: BatchedObsBuffers | None = None,
) -> None:
    """Draw junction-colored force arrows at woody start anchors (world-frame child COM force)."""
    if not _viewer_has_line_overlay(viewer) or bufs is None or bufs.num_junctions <= 0:
        return

    cable = scene.cable
    all_joints = fruiting_tree_fixed_joints(cable)
    if not all_joints:
        return

    fallback = str(getattr(getattr(cable, "model", None), "device", "cpu"))
    dev = _viewer_device(viewer, fallback=fallback)

    parent_all = bufs.woody_parent_pos.numpy().reshape(-1, 3)
    force_all = bufs.woody_force.numpy().reshape(-1, 3)

    starts_list: list[np.ndarray] = []
    ends_list: list[np.ndarray] = []
    colors_list: list[np.ndarray] = []

    for w in range(layout.num_envs):
        for j_local, (_ji, label) in enumerate(all_joints):
            if not _is_viz_woody_junction_label(label):
                continue
            flat = w * bufs.num_junctions + j_local
            origin = _world_position(scene, layout, w, parent_all[flat])
            segments = force_arrow_segment_arrays(
                origin,
                force_all[flat],
                scale_per_newton=scale_per_newton,
                gain=gain,
                min_length=min_length,
                max_length=max_length,
                force_threshold=force_threshold,
            )
            if segments is None:
                continue
            starts, ends, _colors = segments
            rgb = junction_viz_color(label)
            starts_list.append(starts)
            ends_list.append(ends)
            colors_list.append(np.array([rgb], dtype=np.float32))

    if not starts_list:
        _clear_logged_arrow(viewer, name, hidden=hidden)
        return

    _log_force_arrow_batch(
        viewer,
        name,
        np.vstack(starts_list),
        np.vstack(ends_list),
        np.vstack(colors_list),
        device=dev,
        hidden=hidden,
    )


def log_batched_woody_part_endpoints(
    viewer: Any,
    scene: Any,
    layout: BatchedEnvLayout,
    *,
    name: str = "/debug/batched_woody_start_points",
    radius: float = MARK_ENDPOINT_WOODY_POINT_RADIUS_M,
    hidden: bool = False,
    bufs: BatchedObsBuffers | None = None,
) -> None:
    """Alias for :func:`log_batched_woody_start_points`."""
    log_batched_woody_start_points(
        viewer,
        scene,
        layout,
        name=name,
        radius=radius,
        hidden=hidden,
        bufs=bufs,
    )


def log_batched_apple_com_grasp_arrow(
    viewer: Any,
    scene: Any,
    layout: BatchedEnvLayout,
    *,
    name: str = "/debug/batched_apple_com_to_grasp",
    min_length: float = MARK_ENDPOINT_ARROW_MIN_LENGTH_M,
    hidden: bool = False,
    bufs: BatchedObsBuffers | None = None,
) -> None:
    """Draw apple COM→grasp arrow for each env (tail=COM, head=grasp surface)."""
    if not _viewer_has_line_overlay(viewer):
        return
    if layout.template_apple_body is None:
        return

    cable = scene.cable
    fallback = str(getattr(getattr(cable, "model", None), "device", "cpu"))
    dev = _viewer_device(viewer, fallback=fallback)
    min_len = max(float(min_length), 1e-4)

    starts_list: list[np.ndarray] = []
    ends_list: list[np.ndarray] = []
    colors_list: list[np.ndarray] = []

    if bufs is not None:
        apple_np = bufs.apple_pos.numpy()
        proxy_np = bufs.proxy_pos.numpy()
        for w in range(layout.num_envs):
            apple_com = apple_np[w]
            proxy_com = proxy_np[w]
            com_w = _apple_com_world(scene, layout, w, apple_com)
            grasp_w = _proxy_grasp_world(scene, layout, w, proxy_com)
            p_seg, c_seg = _endpoint_arrow_segment(com_w, grasp_w, min_length=min_len)
            starts_list.append(p_seg.reshape(1, 3))
            ends_list.append(c_seg.reshape(1, 3))
            colors_list.append(
                np.array([MARKER_APPLE_COM_TO_GRASP_COLOR], dtype=np.float32)
            )
    else:
        cable_bq = cable.state_0.body_q.numpy().reshape(-1, 7)
        for w in range(layout.num_envs):
            apple_idx = layout.apple_body_indices[w]
            proxy_idx = layout.proxy_body_indices[w]
            apple_com = cable_bq[apple_idx, :3]
            proxy_com = cable_bq[proxy_idx, :3]
            com_w = _apple_com_world(scene, layout, w, apple_com)
            grasp_w = _proxy_grasp_world(scene, layout, w, proxy_com)
            p_seg, c_seg = _endpoint_arrow_segment(com_w, grasp_w, min_length=min_len)
            starts_list.append(p_seg.reshape(1, 3))
            ends_list.append(c_seg.reshape(1, 3))
            colors_list.append(
                np.array([MARKER_APPLE_COM_TO_GRASP_COLOR], dtype=np.float32)
            )

    if not starts_list:
        return

    _log_force_arrow_batch(
        viewer,
        name,
        np.vstack(starts_list),
        np.vstack(ends_list),
        np.vstack(colors_list),
        device=dev,
        hidden=hidden,
    )


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
    bufs: BatchedObsBuffers | None = None,
) -> None:
    """Draw harvested TCP force arrows for every env in a batched coupled scene."""
    if not _viewer_has_line_overlay(viewer):
        return

    if bufs is None and scene.proxy_forces is None:
        _clear_logged_arrow(viewer, name, hidden=hidden)
        return

    fallback = None
    model = getattr(scene, "robot_model", None)
    if model is not None:
        fallback = str(model.device)
    dev = _viewer_device(viewer, fallback=fallback)

    starts_list: list[np.ndarray] = []
    ends_list: list[np.ndarray] = []
    colors_list: list[np.ndarray] = []

    if bufs is not None:
        tcp_pose = bufs.tcp_pose.numpy()
        tcp_force = bufs.tcp_force.numpy()
        for w in range(layout.num_envs):
            origin = _world_position(scene, layout, w, tcp_pose[w, :3])
            segments = force_arrow_segment_arrays(
                origin,
                tcp_force[w, :3],
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
    else:
        robot_bq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)
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

    _log_force_arrow_batch(
        viewer,
        name,
        np.vstack(starts_list),
        np.vstack(ends_list),
        np.vstack(colors_list),
        device=dev,
        hidden=hidden,
    )


def log_batched_endpoints(
    viewer: Any,
    scene: Any,
    layout: BatchedEnvLayout,
    *,
    hidden: bool = False,
    bufs: BatchedObsBuffers | None = None,
    woody_force_scale: float = MARK_ENDPOINT_WOODY_FORCE_SCALE_PER_NEWTON,
    woody_force_gain: float = 1.0,
    woody_force_min_length: float = MARK_ENDPOINT_WOODY_FORCE_MIN_LENGTH_M,
    woody_force_max_length: float = MARK_ENDPOINT_WOODY_FORCE_MAX_LENGTH_M,
    woody_force_threshold: float = 1e-6,
) -> None:
    """Draw apple COM→grasp arrow, woody start points, and woody junction force arrows."""
    log_batched_apple_com_grasp_arrow(
        viewer,
        scene,
        layout,
        hidden=hidden,
        bufs=bufs,
    )
    log_batched_woody_start_points(
        viewer,
        scene,
        layout,
        hidden=hidden,
        bufs=bufs,
    )
    log_batched_woody_junction_force_arrows(
        viewer,
        scene,
        layout,
        hidden=hidden,
        bufs=bufs,
        scale_per_newton=woody_force_scale,
        gain=woody_force_gain,
        min_length=woody_force_min_length,
        max_length=woody_force_max_length,
        force_threshold=woody_force_threshold,
    )
