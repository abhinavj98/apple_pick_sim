"""Collect and render Fibonacci pull-direction geometry from ApplePickSysIdEnv."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from apple_pick_sim.system_id.fibonacci_hemisphere import (
    sample_robot_facing_pull_directions,
    stem_perpendicular_robot_pole,
)
from apple_pick_sim.tests.conftest import COUPLED_ROBOT_BASE_POS

DEFAULT_PULL_MAX_POLAR_ANGLE_RAD = np.pi / 2.0


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero vector cannot be normalized")
    return (v / n).astype(np.float64)


@dataclass(frozen=True)
class PullDirectionGeometry:
    """Geometry extracted from a live sys-ID env after ``reset()``."""

    apple_pos: np.ndarray
    tcp_target: np.ndarray
    proxy_pos: np.ndarray | None
    robot_base: np.ndarray
    stem_dir: np.ndarray
    physical_stem_dir: np.ndarray
    robot_dir: np.ndarray
    weld_dir: np.ndarray | None
    pull_directions: np.ndarray
    apple_radius: float
    weld_robot_dot: float | None
    proxy_robot_dot: float | None
    stem_robot_dot: float
    min_pull_robot_dot: float
    max_pull_polar_angle_rad: float
    reset_index: int = 0


def collect_pull_direction_geometry(
    env: Any,
    obs: dict[str, Any],
    *,
    n_directions: int,
    reset_index: int = 0,
    max_polar_angle: float = DEFAULT_PULL_MAX_POLAR_ANGLE_RAD,
) -> PullDirectionGeometry:
    """Sample robot-facing pull directions on a reset env."""
    tcp_target = np.asarray(env._controller.target_tf[:3], dtype=np.float64)
    apple_pos = np.asarray(obs["apple_pos"], dtype=np.float64)
    stem_dir = _normalize(apple_pos - tcp_target)

    robot_base = np.asarray(COUPLED_ROBOT_BASE_POS, dtype=np.float64)
    robot_vec = robot_base - apple_pos
    robot_base_dir = _normalize(robot_vec)

    physical_stem_dir = robot_base_dir
    scene = env._scene
    if scene is not None:
        stem_bodies = scene.cable.stem_bodies
        if len(stem_bodies) >= 2:
            body_q = scene.cable.state_0.body_q.numpy().reshape(-1, 7)
            physical_stem_dir = _normalize(
                body_q[int(stem_bodies[-1]), :3] - body_q[int(stem_bodies[-2]), :3]
            )

    pole = stem_perpendicular_robot_pole(physical_stem_dir, robot_vec)
    pull_directions = sample_robot_facing_pull_directions(
        n_directions,
        physical_stem_dir,
        robot_vec,
        max_polar_angle=max_polar_angle,
    )

    weld_dir: np.ndarray | None = None
    proxy_pos: np.ndarray | None = None
    apple_radius = 0.04
    weld_robot_dot: float | None = None
    proxy_robot_dot: float | None = None

    scene = env._scene
    if scene is not None:
        cable = scene.cable
        apple_radius = float(cable.params.apple_radius or apple_radius)
        if cable.gripper_proxy_body is not None:
            body_q = cable.state_0.body_q.numpy().reshape(-1, 7)
            proxy_pos = body_q[int(cable.gripper_proxy_body), :3].astype(np.float64)
            proxy_vec = proxy_pos - apple_pos
            proxy_robot_dot = float(np.dot(_normalize(proxy_vec), pole))

    last_weld = getattr(env, "_last_weld_direction", None)
    if last_weld is not None:
        weld_dir = _normalize(np.asarray(last_weld, dtype=np.float64))
        weld_robot_dot = float(np.dot(weld_dir, pole))

    pull_dots = pull_directions @ pole
    return PullDirectionGeometry(
        apple_pos=apple_pos,
        tcp_target=tcp_target,
        proxy_pos=proxy_pos,
        robot_base=robot_base,
        stem_dir=stem_dir,
        physical_stem_dir=physical_stem_dir,
        robot_dir=pole,
        weld_dir=weld_dir,
        pull_directions=pull_directions,
        apple_radius=apple_radius,
        weld_robot_dot=weld_robot_dot,
        proxy_robot_dot=proxy_robot_dot,
        stem_robot_dot=float(np.dot(stem_dir, pole)),
        min_pull_robot_dot=float(np.min(pull_dots)),
        max_pull_polar_angle_rad=float(max_polar_angle),
        reset_index=reset_index,
    )


def _set_equal_aspect_3d(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    if span < 1e-6:
        span = 1.0
    half = 0.55 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def _draw_apple_circle_2d(ax, center: np.ndarray, radius: float, i: int, j: int) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 48)
    xs = center[i] + radius * np.cos(theta)
    ys = center[j] + radius * np.sin(theta)
    ax.plot(xs, ys, color="orange", linewidth=1.5, alpha=0.8)


def _draw_directions_2d(
    ax,
    origin: np.ndarray,
    directions: np.ndarray,
    *,
    length: float,
    cmap_name: str = "tab10",
    label_indices: bool = True,
    fontsize: int = 8,
) -> None:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    for idx, direction in enumerate(directions):
        color = cmap(idx % 10)
        ax.quiver(
            origin[0],
            origin[1],
            direction[0] * length,
            direction[1] * length,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            width=0.004,
            alpha=0.85,
        )
        if label_indices:
            tip = origin + direction * length * 1.08
            ax.text(tip[0], tip[1], str(idx), fontsize=fontsize, color=color)


def _arrow_len_for_geom(geom: PullDirectionGeometry) -> float:
    return max(0.12, geom.apple_radius * 1.5)


def _sanity_check_text(geom: PullDirectionGeometry) -> str:
    weld_dot_str = "n/a" if geom.weld_robot_dot is None else f"{geom.weld_robot_dot:+.3f}"
    proxy_dot_str = "n/a" if geom.proxy_robot_dot is None else f"{geom.proxy_robot_dot:+.3f}"
    return (
        f"dot(weld, pole) = {weld_dot_str}  (expect >= 0)\n"
        f"dot(proxy-apple, pole) = {proxy_dot_str}  (expect >= 0)\n"
        f"min dot(pull, pole) = {geom.min_pull_robot_dot:+.3f}  "
        f"(expect >= cos({np.degrees(geom.max_pull_polar_angle_rad):.0f}°))"
    )


def _draw_3d_panel(ax3, geom: PullDirectionGeometry, *, arrow_len: float, show_legend: bool) -> None:
    import matplotlib.pyplot as plt

    apple = geom.apple_pos
    r = geom.apple_radius
    points = [apple, geom.tcp_target, geom.robot_base]
    if geom.proxy_pos is not None:
        points.append(geom.proxy_pos)

    u = np.linspace(0.0, 2.0 * np.pi, 24)
    v = np.linspace(0.0, np.pi, 12)
    xs = apple[0] + r * np.outer(np.cos(u), np.sin(v))
    ys = apple[1] + r * np.outer(np.sin(u), np.sin(v))
    zs = apple[2] + r * np.outer(np.ones_like(u), np.cos(v))
    ax3.plot_surface(xs, ys, zs, color="orange", alpha=0.2, linewidth=0)
    ax3.scatter(*apple, color="black", s=40, label="apple")
    ax3.scatter(*geom.tcp_target, color="gray", marker="x", s=50, label="TCP target")
    ax3.scatter(*geom.robot_base, color="dimgray", marker="^", s=55, label="robot base")
    if geom.proxy_pos is not None:
        ax3.scatter(*geom.proxy_pos, color="magenta", s=45, label="proxy COM")

    ax3.quiver(
        *apple,
        *geom.physical_stem_dir,
        length=arrow_len,
        color="red",
        linewidth=2,
        label="physical stem (base→tip)",
    )
    ax3.quiver(
        *apple,
        *geom.stem_dir,
        length=arrow_len * 0.85,
        color="salmon",
        linewidth=1.5,
        label="grasp axis (TCP→apple)",
    )
    ax3.quiver(
        *apple,
        *geom.robot_dir,
        length=arrow_len,
        color="blue",
        linewidth=1.5,
        label="pull pole (stem⊥, robot-facing)",
    )
    if geom.weld_dir is not None:
        ax3.quiver(
            *apple,
            *geom.weld_dir,
            length=arrow_len * 0.9,
            color="magenta",
            linewidth=2,
            label="weld dir (env)",
        )

    cmap = plt.get_cmap("tab10")
    for idx, direction in enumerate(geom.pull_directions):
        color = cmap(idx % 10)
        ax3.quiver(
            *apple,
            *direction,
            length=arrow_len * 0.75,
            color=color,
            alpha=0.8,
            linewidth=1.0,
        )
        tip = apple + direction * arrow_len * 0.82
        ax3.text(tip[0], tip[1], tip[2], str(idx), fontsize=7, color=color)

    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")
    ax3.set_zlabel("z [m]")
    if show_legend:
        ax3.legend(loc="upper left", fontsize=7)
    _set_equal_aspect_3d(ax3, np.vstack(points + [apple + d * arrow_len for d in geom.pull_directions]))


def _draw_xz_panel(ax_xz, geom: PullDirectionGeometry, *, arrow_len: float) -> None:
    apple = geom.apple_pos
    r = geom.apple_radius
    _draw_apple_circle_2d(ax_xz, apple, r, 0, 2)
    ax_xz.scatter(apple[0], apple[2], color="black", s=30)
    ax_xz.scatter(geom.tcp_target[0], geom.tcp_target[2], color="gray", marker="x", s=40)
    ax_xz.scatter(geom.robot_base[0], geom.robot_base[2], color="dimgray", marker="^", s=45)
    if geom.proxy_pos is not None:
        ax_xz.scatter(geom.proxy_pos[0], geom.proxy_pos[2], color="magenta", s=35)
    dirs_xz = np.stack(
        [geom.pull_directions[:, 0], geom.pull_directions[:, 2]],
        axis=1,
    )
    norms = np.linalg.norm(dirs_xz, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    dirs_xz = dirs_xz / norms
    _draw_directions_2d(ax_xz, np.array([apple[0], apple[2]]), dirs_xz, length=arrow_len, fontsize=7)
    ax_xz.quiver(
        apple[0],
        apple[2],
        geom.robot_dir[0] * arrow_len,
        geom.robot_dir[2] * arrow_len,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="blue",
        width=0.005,
    )
    if geom.weld_dir is not None:
        ax_xz.quiver(
            apple[0],
            apple[2],
            geom.weld_dir[0] * arrow_len,
            geom.weld_dir[2] * arrow_len,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="magenta",
            width=0.005,
        )
    ax_xz.set_xlabel("x [m]")
    ax_xz.set_ylabel("z [m]")
    ax_xz.set_title("XZ side view")
    ax_xz.set_aspect("equal", adjustable="box")


def _draw_xy_panel(ax_xy, geom: PullDirectionGeometry, *, arrow_len: float) -> None:
    apple = geom.apple_pos
    r = geom.apple_radius
    _draw_apple_circle_2d(ax_xy, apple, r, 0, 1)
    ax_xy.scatter(apple[0], apple[1], color="black", s=30)
    ax_xy.scatter(geom.tcp_target[0], geom.tcp_target[1], color="gray", marker="x", s=40)
    ax_xy.scatter(geom.robot_base[0], geom.robot_base[1], color="dimgray", marker="^", s=45)
    if geom.proxy_pos is not None:
        ax_xy.scatter(geom.proxy_pos[0], geom.proxy_pos[1], color="magenta", s=35)
    dirs_xy = np.stack(
        [geom.pull_directions[:, 0], geom.pull_directions[:, 1]],
        axis=1,
    )
    norms_xy = np.linalg.norm(dirs_xy, axis=1, keepdims=True)
    norms_xy = np.maximum(norms_xy, 1e-12)
    dirs_xy = dirs_xy / norms_xy
    _draw_directions_2d(ax_xy, np.array([apple[0], apple[1]]), dirs_xy, length=arrow_len, fontsize=7)
    ax_xy.quiver(
        apple[0],
        apple[1],
        geom.robot_dir[0] * arrow_len,
        geom.robot_dir[1] * arrow_len,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="blue",
        width=0.005,
    )
    if geom.weld_dir is not None:
        ax_xy.quiver(
            apple[0],
            apple[1],
            geom.weld_dir[0] * arrow_len,
            geom.weld_dir[1] * arrow_len,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="magenta",
            width=0.005,
        )
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("XY top-down")
    ax_xy.set_aspect("equal", adjustable="box")


def _save_or_show_figure(fig, output_path: Path | str | None, *, show: bool) -> None:
    import matplotlib.pyplot as plt

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Saved {out.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def render_pull_direction_figure(
    geom: PullDirectionGeometry,
    output_path: Path | str | None,
    *,
    show: bool = False,
    title_suffix: str = "",
) -> None:
    """Save a 3-panel figure of pull directions and weld/grasp geometry."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 5))
    title = "Fibonacci pull directions (ApplePickSysIdEnv)"
    if title_suffix:
        title = f"{title} — {title_suffix}"
    fig.suptitle(title, fontsize=12)

    arrow_len = _arrow_len_for_geom(geom)
    _draw_3d_panel(fig.add_subplot(1, 3, 1, projection="3d"), geom, arrow_len=arrow_len, show_legend=True)
    _draw_xz_panel(fig.add_subplot(1, 3, 2), geom, arrow_len=arrow_len)
    _draw_xy_panel(fig.add_subplot(1, 3, 3), geom, arrow_len=arrow_len)

    fig.text(0.5, 0.01, _sanity_check_text(geom), ha="center", va="bottom", fontsize=9, family="monospace")
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    _save_or_show_figure(fig, output_path, show=show)


def render_multi_reset_figure(
    geoms: list[PullDirectionGeometry],
    output_path: Path | str | None,
    *,
    show: bool = False,
    title_suffix: str = "",
) -> None:
    """Save an N-row x 3-column figure, one row per env reset / weld."""
    import matplotlib.pyplot as plt

    if not geoms:
        raise ValueError("geoms must be non-empty")

    n_rows = len(geoms)
    fig = plt.figure(figsize=(14, 4.5 * n_rows))
    title = f"Fibonacci pull directions — {n_rows} resets (ApplePickSysIdEnv)"
    if title_suffix:
        title = f"{title} — {title_suffix}"
    fig.suptitle(title, fontsize=12)

    for row, geom in enumerate(geoms):
        arrow_len = _arrow_len_for_geom(geom)
        ax3 = fig.add_subplot(n_rows, 3, row * 3 + 1, projection="3d")
        _draw_3d_panel(ax3, geom, arrow_len=arrow_len, show_legend=(row == 0))
        weld_label = "n/a"
        if geom.weld_dir is not None:
            w = geom.weld_dir
            weld_label = f"({w[0]:+.2f},{w[1]:+.2f},{w[2]:+.2f})"
        ax3.set_title(f"reset {geom.reset_index}: weld {weld_label}", fontsize=9)

        _draw_xz_panel(fig.add_subplot(n_rows, 3, row * 3 + 2), geom, arrow_len=arrow_len)
        _draw_xy_panel(fig.add_subplot(n_rows, 3, row * 3 + 3), geom, arrow_len=arrow_len)

    fig.text(
        0.5,
        0.01,
        _sanity_check_text(geoms[-1]),
        ha="center",
        va="bottom",
        fontsize=9,
        family="monospace",
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.96))
    _save_or_show_figure(fig, output_path, show=show)
