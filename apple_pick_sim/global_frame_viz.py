"""Debug coordinate frames in Newton viewers (world / instance / body)."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from apple_pick_sim.fruiting_system.mega import MegaCoupledCableScene

# RGB = XYZ (same convention as newton.examples.basic_viewer).
_AXIS_COLORS = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
_UNIT_AXES = np.eye(3, dtype=np.float64)


def _viewer_device(viewer: Any, *, fallback: str | None = None) -> str:
    dev = getattr(viewer, "device", None)
    if dev is not None:
        return str(dev)
    if fallback is not None:
        return str(fallback)
    return "cpu"


def axis_segment_arrays(
    origin: tuple[float, float, float] | np.ndarray,
    *,
    quat_wxyz: tuple[float, float, float, float] | np.ndarray | None = None,
    length: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return line segment arrays for X/Y/Z axes (shape ``(3, 3)`` each)."""
    if length <= 0.0:
        raise ValueError("length must be positive")
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    starts = np.broadcast_to(o, (3, 3)).copy()
    dirs = _UNIT_AXES * float(length)
    if quat_wxyz is not None:
        q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        wq = wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        dirs = np.stack(
            [
                np.asarray(wp.quat_rotate(wq, wp.vec3(*row)), dtype=np.float64)
                for row in dirs
            ],
            axis=0,
        )
    ends = starts + dirs
    return starts, ends, _AXIS_COLORS.copy()


def log_coordinate_frame(
    viewer: Any,
    name: str,
    origin: tuple[float, float, float] | np.ndarray,
    *,
    quat_wxyz: tuple[float, float, float, float] | np.ndarray | None = None,
    length: float = 0.35,
    device: str | None = None,
    hidden: bool = False,
) -> None:
    """Draw RGB XYZ axes at ``origin`` (optional body orientation via ``quat_wxyz``)."""
    log_lines = getattr(viewer, "log_lines", None)
    if log_lines is None:
        return
    dev = device if device is not None else _viewer_device(viewer)
    starts, ends, colors = axis_segment_arrays(origin, quat_wxyz=quat_wxyz, length=length)
    wp_starts = wp.array(starts, dtype=wp.vec3, device=dev)
    wp_ends = wp.array(ends, dtype=wp.vec3, device=dev)
    wp_colors = wp.array(colors, dtype=wp.vec3, device=dev)
    log_arrows = getattr(viewer, "log_arrows", None)
    if log_arrows is not None:
        log_arrows(name, wp_starts, wp_ends, wp_colors, hidden=hidden)
    else:
        log_lines(name, wp_starts, wp_ends, wp_colors, hidden=hidden)


def log_mega_global_frames(
    viewer: Any,
    mega: MegaCoupledCableScene,
    *,
    nominal_index: int = 0,
    axis_length: float = 0.35,
    show_world: bool = True,
    show_instance_bases: bool = True,
    show_nominal_bodies: bool = True,
) -> None:
    """World frame, per-column ``base_pos`` frames, and nominal apple/proxy body frames."""
    device = _viewer_device(viewer, fallback=str(mega.model.device))

    if show_world:
        log_coordinate_frame(
            viewer,
            "/debug/world_frame",
            (0.0, 0.0, 0.0),
            length=axis_length,
            device=device,
        )

    if show_instance_bases:
        for inst in mega.instances:
            log_coordinate_frame(
                viewer,
                f"/debug/instance_{inst.index}/base_frame",
                inst.base_pos,
                length=axis_length * 0.85,
                device=device,
            )

    if not show_nominal_bodies:
        return

    inst = mega.instance(nominal_index)
    bq = mega.state_0.body_q.numpy().reshape(-1, 7)
    bodies: list[tuple[str, int]] = [("proxy", inst.gripper_proxy_body)]
    if inst.apple_body is not None:
        bodies.insert(0, ("apple", inst.apple_body))

    for label, bid in bodies:
        row = bq[bid]
        origin = row[:3]
        quat = row[3:7]
        log_coordinate_frame(
            viewer,
            f"/debug/nominal/{label}_frame",
            origin,
            quat_wxyz=quat,
            length=axis_length,
            device=device,
        )
