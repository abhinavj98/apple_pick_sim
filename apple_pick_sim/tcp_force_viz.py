"""Draw TCP coupling force as a world-frame arrow in Newton viewers."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from apple_pick_sim.coupling_force_debug import read_tcp_wrench

# Distinct from RGB axis colors in ``global_frame_viz``.
_FORCE_ARROW_COLOR = np.array([[1.0, 0.85, 0.1]], dtype=np.float32)


def _viewer_device(viewer: Any, *, fallback: str | None = None) -> str:
    dev = getattr(viewer, "device", None)
    if dev is not None:
        return str(dev)
    if fallback is not None:
        return str(fallback)
    return "cpu"


def arrow_length_m(
    force_magnitude: float,
    *,
    scale_per_newton: float,
    gain: float = 1.0,
    min_length: float = 0.0,
    max_length: float = 0.0,
) -> float:
    """Arrow length [m] from ``|F|`` with optional gain and clamp (``max_length`` 0 = no cap)."""
    if scale_per_newton <= 0.0:
        raise ValueError("scale_per_newton must be positive")
    if gain <= 0.0:
        raise ValueError("gain must be positive")
    if min_length < 0.0:
        raise ValueError("min_length must be >= 0")
    if max_length < 0.0:
        raise ValueError("max_length must be >= 0")
    length = float(force_magnitude) * scale_per_newton * gain
    if min_length > 0.0:
        length = max(length, min_length)
    if max_length > 0.0:
        length = min(length, max_length)
    return length


def force_arrow_segment_arrays(
    origin: tuple[float, float, float] | np.ndarray,
    force: tuple[float, float, float] | np.ndarray,
    *,
    scale_per_newton: float = 0.02,
    gain: float = 1.0,
    min_length: float = 0.0,
    max_length: float = 0.0,
    force_threshold: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """One segment from ``origin`` along ``force`` (length from :func:`arrow_length_m`).

    Returns ``None`` when ``|F| < force_threshold`` (caller should clear the batch).
    """
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    f = np.asarray(force, dtype=np.float64).reshape(3)
    fmag = float(np.linalg.norm(f))
    if fmag < force_threshold:
        return None
    length = arrow_length_m(
        fmag,
        scale_per_newton=scale_per_newton,
        gain=gain,
        min_length=min_length,
        max_length=max_length,
    )
    tip = o + (f / fmag) * length
    starts = o.reshape(1, 3)
    ends = tip.reshape(1, 3)
    return starts, ends, _FORCE_ARROW_COLOR.copy()


def log_tcp_force_arrow(
    viewer: Any,
    name: str,
    origin: tuple[float, float, float] | np.ndarray,
    wrench_6: np.ndarray,
    *,
    scale_per_newton: float = 0.02,
    gain: float = 1.0,
    min_length: float = 0.0,
    max_length: float = 0.0,
    force_threshold: float = 1e-6,
    device: str | None = None,
    hidden: bool = False,
) -> None:
    """Draw force ``wrench_6[:3]`` as an arrow from ``origin`` (world frame)."""
    log_lines = getattr(viewer, "log_lines", None)
    if log_lines is None:
        return

    segments = force_arrow_segment_arrays(
        origin,
        wrench_6[:3],
        scale_per_newton=scale_per_newton,
        gain=gain,
        min_length=min_length,
        max_length=max_length,
        force_threshold=force_threshold,
    )
    dev = device if device is not None else _viewer_device(viewer)

    if segments is None:
        log_lines(name, None, None, None, hidden=hidden)
        return

    starts, ends, colors = segments
    wp_starts = wp.array(starts, dtype=wp.vec3, device=dev)
    wp_ends = wp.array(ends, dtype=wp.vec3, device=dev)
    wp_colors = wp.array(colors, dtype=wp.vec3, device=dev)
    log_arrows = getattr(viewer, "log_arrows", None)
    if log_arrows is not None:
        log_arrows(name, wp_starts, wp_ends, wp_colors, hidden=hidden)
    else:
        log_lines(name, wp_starts, wp_ends, wp_colors, hidden=hidden)


def tcp_origin_world(scene: Any) -> np.ndarray:
    """TCP body COM / transform position from ``scene.robot_state_0.body_q``."""
    tcp = int(scene.tcp_body_index)
    bq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)
    return bq[tcp, :3].astype(np.float64, copy=False)


def log_coupled_scene_tcp_force(
    viewer: Any,
    scene: Any,
    *,
    name: str = "/debug/tcp_force_arrow",
    scale_per_newton: float = 0.02,
    gain: float = 1.0,
    min_length: float = 0.0,
    max_length: float = 0.0,
    force_threshold: float = 1e-6,
    hidden: bool = False,
) -> None:
    """Harvested TCP wrench from ``scene.proxy_forces`` at the robot TCP pose."""
    if scene.proxy_forces is None:
        log_lines = getattr(viewer, "log_lines", None)
        if log_lines is not None:
            log_lines(name, None, None, None, hidden=hidden)
        return

    wrench = read_tcp_wrench(scene.proxy_forces, scene.tcp_body_index)
    origin = tcp_origin_world(scene)
    fallback = None
    model = getattr(scene, "robot_model", None)
    if model is not None:
        fallback = str(model.device)
    log_tcp_force_arrow(
        viewer,
        name,
        origin,
        wrench,
        scale_per_newton=scale_per_newton,
        gain=gain,
        min_length=min_length,
        max_length=max_length,
        force_threshold=force_threshold,
        device=_viewer_device(viewer, fallback=fallback),
        hidden=hidden,
    )
