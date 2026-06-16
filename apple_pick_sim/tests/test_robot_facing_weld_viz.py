"""Robot-facing weld sampling + optional 3D visualization.

This module exercises the **production build path** — ``generate_coupled_cable_scene``
with ``GripperProxyConfig(fix_to_apple=True, robot_facing_weld=True)`` — not a
standalone geometry mock. Each test rebuilds a full VBD cable scene and checks that
the welded gripper proxy lands on the stem-perpendicular hemisphere that faces
``robot_base_pos``.

**What "robot-facing" means geometrically**

Given apple center ``A``, robot base ``R``, and physical stem direction ``s``,
the expected exterior weld pole is ``stem_perpendicular_robot_pole(s, R-A)``:
perpendicular to the stem and on the side toward the robot. The proxy COM should
sit at distance ``apple_radius + proxy_clearance`` from ``A`` in that direction.

**Why two direction checks (``weld_dir`` and ``offset_approach``)**

- ``weld_dir``: read from ``body_q`` positions (apple COM → proxy COM). This is what
  you see in the viewer after scene build.
- ``offset_approach``: read from ``gripper_proxy_offset_in_apple_frame``, the 7D
  fixed-joint anchor stored at build time in ``_add_gripper_proxy``. We rotate the
  offset position from apple frame to world and normalize.

We assert both align with the stem-perpendicular pole because the proxy link is added
with identity quaternion at build time; the welded orientation lives in the joint
offset, not in ``body_q[proxy, 3:7]`` until a full FK pass (which coupled cable build
deliberately skips for the proxy articulation).

**Visualization**

Set ``APPLE_PICK_SIM_SAVE_VIZ`` to persist PNGs outside pytest's ephemeral ``tmp_path``::

    APPLE_PICK_SIM_SAVE_VIZ=/tmp/robot_facing_weld \\
    uv run --env-file pytest.env python -m pytest \\
        apple_pick_sim/tests/test_robot_facing_weld_viz.py -k visualization -s
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from apple_pick_sim.system_id import stem_perpendicular_robot_pole
from apple_pick_sim.tests.conftest import (
    COUPLED_ROBOT_BASE_POS,
    COUPLED_VBD_SCENE_KW,
    RANGES_FIXTURE,
)

# Fruiting seeds to sweep: each seed changes rod/apple layout in world frame.
SAMPLE_SEEDS = tuple(range(12))

# Robot-base offsets relative to a reference apple position (seed 17 test).
# Mostly below the apple (negative Z) so the robot "looks up" at the fruit.
ROBOT_BASE_OFFSETS = (
    (0.0, 0.0, -0.85),
    (0.45, 0.0, -0.75),
    (-0.35, 0.25, -0.80),
    (0.15, -0.40, -0.70),
    (0.55, 0.35, -0.60),
    (-0.50, -0.30, -0.90),
)


def _import_fs():
    """Deferred import so Warp init fixture runs first."""
    import apple_pick_sim.fruiting_system as fs

    return fs


@pytest.fixture(scope="module", autouse=True)
def _warp_init():
    wp.init()


def _build_robot_facing_scene(
    fs,
    *,
    seed: int,
    robot_base_pos: tuple[float, float, float],
    weld_direction: tuple[float, float, float] | None = None,
):
    """Build one coupled cable scene with robot-facing weld enabled."""
    return fs.generate_coupled_cable_scene(
        fs.load_ranges(RANGES_FIXTURE),
        seed=seed,
        gripper_proxy=fs.GripperProxyConfig(
            fix_to_apple=True,
            robot_facing_weld=True,
            weld_direction=weld_direction,
        ),
        robot_base_pos=robot_base_pos,
        **COUPLED_VBD_SCENE_KW,
    )


def _stem_direction_world(scene) -> np.ndarray:
    """Unit vector from distal stem segment base toward tip (apple pole)."""
    stem = scene.stem_bodies
    assert len(stem) >= 2
    body_q = scene.state_0.body_q.to("cpu").numpy()
    tip = body_q[stem[-1], :3]
    base = body_q[stem[-2], :3]
    d = tip - base
    return d / np.linalg.norm(d)


def _fibonacci_hemisphere_directions(
    pole_dir: np.ndarray,
    count: int = 8,
) -> list[np.ndarray]:
    """Unit directions on the forward hemisphere w.r.t. ``pole_dir`` (golden-ratio lattice)."""
    pole = pole_dir / np.linalg.norm(pole_dir)
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    out: list[np.ndarray] = []
    i = 0
    max_samples = count * 20
    while len(out) < count and i < max_samples:
        theta = 2.0 * math.pi * i / golden_ratio
        z = 1.0 - (2.0 * (i + 0.5) / max_samples)
        r = math.sqrt(max(0.0, 1.0 - z * z))
        d = np.array([r * math.cos(theta), r * math.sin(theta), z], dtype=np.float64)
        d /= np.linalg.norm(d)
        if float(np.dot(d, pole)) >= 0.0:
            out.append(d)
        i += 1
    assert len(out) == count
    return out


def _offset_approach_in_world(scene) -> np.ndarray | None:
    """Unit approach direction implied by the apple↔proxy FIXED joint offset.

    ``gripper_proxy_offset_in_apple_frame`` stores ``(pos, quat)`` of the parent
    anchor in the apple body frame. For robot-facing weld, the position component
    points from apple COM toward the exterior pole (same line as the expected pole).
    """
    offset = scene.gripper_proxy_offset_in_apple_frame
    if offset is None or scene.apple_body is None:
        return None
    off_pos = np.asarray(offset[:3], dtype=np.float64)
    norm = float(np.linalg.norm(off_pos))
    if norm < 1e-9:
        return None
    approach_apple = off_pos / norm

    # Rotate apple-frame direction into world using the apple body orientation.
    body_q = scene.state_0.body_q.to("cpu").numpy()
    apple_quat = body_q[scene.apple_body, 3:7].astype(np.float64)
    wq = wp.quat(
        float(apple_quat[0]),
        float(apple_quat[1]),
        float(apple_quat[2]),
        float(apple_quat[3]),
    )
    return np.asarray(wp.quat_rotate(wq, wp.vec3(*approach_apple)), dtype=np.float64)


def _weld_geometry(
    scene,
    robot_base_pos: tuple[float, float, float],
) -> dict[str, np.ndarray | float | None]:
    """Extract positions and alignment metrics for one built scene."""
    body_q = scene.state_0.body_q.to("cpu").numpy()
    apple_pos = body_q[scene.apple_body, :3].astype(np.float64)
    proxy_pos = body_q[scene.gripper_proxy_body, :3].astype(np.float64)

    robot_vec = np.asarray(robot_base_pos, dtype=np.float64) - apple_pos
    robot_dir = robot_vec / np.linalg.norm(robot_vec)
    stem_dir = _stem_direction_world(scene)
    expected_pole = stem_perpendicular_robot_pole(stem_dir, robot_vec)

    weld_vec = proxy_pos - apple_pos
    weld_dir = weld_vec / np.linalg.norm(weld_vec)

    offset_approach = _offset_approach_in_world(scene)

    clearance = float(max(scene.gripper_proxy_config.box_half_extents))
    expected_radius = float(scene.params.apple_radius) + clearance

    return {
        "apple_pos": apple_pos,
        "proxy_pos": proxy_pos,
        "robot_dir": robot_dir,
        "expected_pole": expected_pole,
        "stem_dir": stem_dir,
        "weld_dir": weld_dir,
        "offset_approach": offset_approach,
        "weld_pole_dot": float(np.dot(weld_dir, expected_pole)),
        "offset_pole_dot": (
            None
            if offset_approach is None
            else float(np.dot(offset_approach, expected_pole))
        ),
        "weld_stem_dot": float(np.dot(weld_dir, stem_dir)),
        "surface_error_m": abs(float(np.linalg.norm(weld_vec)) - expected_radius),
    }


def _robot_bases_on_ring(
    apple_pos: np.ndarray,
    *,
    radius: float = 0.85,
    count: int = 8,
    elevation_deg: float = -35.0,
) -> list[tuple[float, float, float]]:
    """Place robot bases on a ring below the apple (negative elevation).

    Samples compass headings at fixed depression angle so the robot approaches
    from below/oblique angles — the typical pick workspace for a floor-mounted arm.
    """
    el = math.radians(elevation_deg)
    cos_el = math.cos(el)
    sin_el = math.sin(el)
    out: list[tuple[float, float, float]] = []
    for i in range(count):
        az = 2.0 * math.pi * i / count
        direction = np.array(
            [cos_el * math.cos(az), cos_el * math.sin(az), sin_el],
            dtype=np.float64,
        )
        base = apple_pos + radius * direction
        out.append((float(base[0]), float(base[1]), float(base[2])))
    return out


def _viz_output_path(tmp_path: Path, filename: str) -> Path:
    """Write under pytest tmp_path, or under APPLE_PICK_SIM_SAVE_VIZ when set."""
    override = os.environ.get("APPLE_PICK_SIM_SAVE_VIZ")
    if override:
        out_dir = Path(override)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / filename
    return tmp_path / filename


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_robot_facing_weld_samples_align_with_robot_base(seed: int):
    """Many fruiting layouts: weld pole tracks stem⊥ robot-facing direction."""
    fs = _import_fs()
    scene = _build_robot_facing_scene(fs, seed=seed, robot_base_pos=COUPLED_ROBOT_BASE_POS)
    geom = _weld_geometry(scene, COUPLED_ROBOT_BASE_POS)
    assert geom["weld_pole_dot"] > 0.99
    assert geom["offset_pole_dot"] is not None
    assert geom["offset_pole_dot"] > 0.99
    assert abs(float(geom["weld_stem_dot"])) < 0.05
    assert geom["surface_error_m"] < 0.02


@pytest.mark.parametrize("offset", ROBOT_BASE_OFFSETS)
def test_robot_facing_weld_tracks_robot_base_ring_offsets(offset):
    """One fruiting layout, several robot bases: weld follows each stem⊥ pole."""
    fs = _import_fs()
    seed = 17
    scene0 = _build_robot_facing_scene(fs, seed=seed, robot_base_pos=COUPLED_ROBOT_BASE_POS)
    apple_pos = _weld_geometry(scene0, COUPLED_ROBOT_BASE_POS)["apple_pos"]
    robot_base = (
        float(apple_pos[0] + offset[0]),
        float(apple_pos[1] + offset[1]),
        float(apple_pos[2] + offset[2]),
    )
    scene = _build_robot_facing_scene(fs, seed=seed, robot_base_pos=robot_base)
    geom = _weld_geometry(scene, robot_base)
    assert geom["weld_pole_dot"] > 0.99
    assert geom["offset_pole_dot"] is not None
    assert geom["offset_pole_dot"] > 0.99
    assert abs(float(geom["weld_stem_dot"])) < 0.05


def test_weld_direction_places_proxy_at_specified_direction():
    """Fibonacci samples on the stem⊥ robot-facing hemisphere land at the requested direction."""
    fs = _import_fs()
    seed = 17
    ref = _build_robot_facing_scene(fs, seed=seed, robot_base_pos=COUPLED_ROBOT_BASE_POS)
    expected_pole = _weld_geometry(ref, COUPLED_ROBOT_BASE_POS)["expected_pole"]
    cos_1deg = math.cos(math.radians(1.0))

    for direction in _fibonacci_hemisphere_directions(expected_pole, count=8):
        scene = _build_robot_facing_scene(
            fs,
            seed=seed,
            robot_base_pos=COUPLED_ROBOT_BASE_POS,
            weld_direction=(float(direction[0]), float(direction[1]), float(direction[2])),
        )
        geom = _weld_geometry(scene, COUPLED_ROBOT_BASE_POS)
        assert float(np.dot(geom["weld_dir"], direction)) > cos_1deg
        assert geom["offset_approach"] is not None
        assert float(np.dot(geom["offset_approach"], direction)) > cos_1deg
        assert float(geom["weld_pole_dot"]) > 0.0
        assert geom["surface_error_m"] < 0.02


def test_weld_direction_not_on_robot_hemisphere_raises():
    """Opposite hemisphere weld_direction is rejected when robot_facing_weld=True."""
    fs = _import_fs()
    ref = _build_robot_facing_scene(fs, seed=17, robot_base_pos=COUPLED_ROBOT_BASE_POS)
    expected_pole = _weld_geometry(ref, COUPLED_ROBOT_BASE_POS)["expected_pole"]
    away = -expected_pole

    with pytest.raises(ValueError, match="stem-perpendicular"):
        _build_robot_facing_scene(
            fs,
            seed=17,
            robot_base_pos=COUPLED_ROBOT_BASE_POS,
            weld_direction=(float(away[0]), float(away[1]), float(away[2])),
        )


def test_weld_direction_falls_back_to_pole_when_none():
    """Omitting weld_direction keeps exact stem⊥ robot-facing pole placement."""
    fs = _import_fs()
    scene = _build_robot_facing_scene(fs, seed=5, robot_base_pos=COUPLED_ROBOT_BASE_POS)
    geom = _weld_geometry(scene, COUPLED_ROBOT_BASE_POS)
    assert geom["weld_pole_dot"] > 0.99
    assert geom["offset_pole_dot"] is not None
    assert geom["offset_pole_dot"] > 0.99
    assert abs(float(geom["weld_stem_dot"])) < 0.05


def test_robot_facing_weld_ring_samples_around_apple():
    """Eight evenly spaced robot headings below one apple all weld correctly."""
    fs = _import_fs()
    seed = 23
    ref = _build_robot_facing_scene(fs, seed=seed, robot_base_pos=COUPLED_ROBOT_BASE_POS)
    apple_pos = _weld_geometry(ref, COUPLED_ROBOT_BASE_POS)["apple_pos"]
    for robot_base in _robot_bases_on_ring(apple_pos):
        scene = _build_robot_facing_scene(fs, seed=seed, robot_base_pos=robot_base)
        geom = _weld_geometry(scene, robot_base)
        assert geom["weld_pole_dot"] > 0.99
        assert geom["offset_pole_dot"] is not None
        assert geom["offset_pole_dot"] > 0.99
        assert abs(float(geom["weld_stem_dot"])) < 0.05


def test_robot_facing_weld_visualization(tmp_path: Path):
    """3D matplotlib plots of weld vs robot directions (optional manual QA)."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    fs = _import_fs()

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Robot-facing weld: green=weld dir, red=pole dir, cyan=offset approach",
        fontsize=12,
    )

    viz_seeds = SAMPLE_SEEDS[:8]
    for idx, seed in enumerate(viz_seeds):
        ax = fig.add_subplot(2, 4, idx + 1, projection="3d")
        scene = _build_robot_facing_scene(
            fs, seed=seed, robot_base_pos=COUPLED_ROBOT_BASE_POS
        )
        geom = _weld_geometry(scene, COUPLED_ROBOT_BASE_POS)
        apple = geom["apple_pos"]
        proxy = geom["proxy_pos"]
        r = float(scene.params.apple_radius)
        robot_base = np.asarray(COUPLED_ROBOT_BASE_POS, dtype=np.float64)

        u = np.linspace(0.0, 2.0 * np.pi, 24)
        v = np.linspace(0.0, np.pi, 12)
        xs = apple[0] + r * np.outer(np.cos(u), np.sin(v))
        ys = apple[1] + r * np.outer(np.sin(u), np.sin(v))
        zs = apple[2] + r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs, color="orange", alpha=0.25, linewidth=0)

        arrow_len = max(0.15, r * 1.5)
        ax.quiver(
            apple[0], apple[1], apple[2],
            geom["weld_dir"][0], geom["weld_dir"][1], geom["weld_dir"][2],
            length=arrow_len, color="green", linewidth=2, label="weld",
        )
        pole = geom["expected_pole"]
        ax.quiver(
            apple[0], apple[1], apple[2],
            pole[0], pole[1], pole[2],
            length=arrow_len, color="red", linewidth=1.5, linestyle="dashed",
        )
        if geom["offset_approach"] is not None:
            off = geom["offset_approach"]
            ax.quiver(
                apple[0], apple[1], apple[2],
                off[0], off[1], off[2],
                length=arrow_len * 0.85, color="cyan", linewidth=1.5,
            )
        ax.scatter(*apple, color="black", s=20)
        ax.scatter(*proxy, color="green", s=30)
        ax.scatter(*robot_base, color="red", marker="^", s=40)

        ax.set_title(f"seed={seed} dot={geom['weld_pole_dot']:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        _set_equal_aspect_3d(ax, np.vstack([apple, proxy, robot_base]))

    fig.tight_layout()
    out_multi = _viz_output_path(tmp_path, "robot_facing_weld_seeds.png")
    fig.savefig(out_multi, dpi=150)
    plt.close(fig)

    fig2 = plt.figure(figsize=(12, 10))
    fig2.suptitle("One apple, robot bases on a lower ring (seed=23)", fontsize=12)
    seed = 23
    ref = _build_robot_facing_scene(fs, seed=seed, robot_base_pos=COUPLED_ROBOT_BASE_POS)
    apple = _weld_geometry(ref, COUPLED_ROBOT_BASE_POS)["apple_pos"]
    r = float(ref.params.apple_radius)
    ring_bases = _robot_bases_on_ring(apple)

    ax2 = fig2.add_subplot(1, 1, 1, projection="3d")
    u = np.linspace(0.0, 2.0 * np.pi, 24)
    v = np.linspace(0.0, np.pi, 12)
    xs = apple[0] + r * np.outer(np.cos(u), np.sin(v))
    ys = apple[1] + r * np.outer(np.sin(u), np.sin(v))
    zs = apple[2] + r * np.outer(np.ones_like(u), np.cos(v))
    ax2.plot_surface(xs, ys, zs, color="orange", alpha=0.2, linewidth=0)
    ax2.scatter(*apple, color="black", s=40, label="apple")

    for i, robot_base in enumerate(ring_bases):
        scene = _build_robot_facing_scene(fs, seed=seed, robot_base_pos=robot_base)
        geom = _weld_geometry(scene, robot_base)
        proxy = geom["proxy_pos"]
        ax2.scatter(*proxy, s=35)
        ax2.plot(
            [apple[0], proxy[0]],
            [apple[1], proxy[1]],
            [apple[2], proxy[2]],
            linewidth=1.5,
        )
        ax2.plot(
            [apple[0], robot_base[0]],
            [apple[1], robot_base[1]],
            [apple[2], robot_base[2]],
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )
        ax2.text(proxy[0], proxy[1], proxy[2], str(i), fontsize=8)

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    _set_equal_aspect_3d(ax2, np.vstack([apple, *ring_bases]))
    fig2.tight_layout()
    out_ring = _viz_output_path(tmp_path, "robot_facing_weld_ring.png")
    fig2.savefig(out_ring, dpi=150)
    plt.close(fig2)

    fig3 = plt.figure(figsize=(10, 9))
    fig3.suptitle(
        "Fibonacci hemisphere weld directions on one apple (seed=17)",
        fontsize=12,
    )
    seed = 17
    ref = _build_robot_facing_scene(fs, seed=seed, robot_base_pos=COUPLED_ROBOT_BASE_POS)
    geom_ref = _weld_geometry(ref, COUPLED_ROBOT_BASE_POS)
    apple = geom_ref["apple_pos"]
    expected_pole = geom_ref["expected_pole"]
    r = float(ref.params.apple_radius)
    fib_dirs = _fibonacci_hemisphere_directions(expected_pole, count=8)
    cmap = plt.get_cmap("tab10")

    ax3 = fig3.add_subplot(1, 1, 1, projection="3d")
    u = np.linspace(0.0, 2.0 * np.pi, 24)
    v = np.linspace(0.0, np.pi, 12)
    xs = apple[0] + r * np.outer(np.cos(u), np.sin(v))
    ys = apple[1] + r * np.outer(np.sin(u), np.sin(v))
    zs = apple[2] + r * np.outer(np.ones_like(u), np.cos(v))
    ax3.plot_surface(xs, ys, zs, color="orange", alpha=0.2, linewidth=0)
    ax3.scatter(*apple, color="black", s=40, label="apple")

    arrow_len = max(0.12, r * 1.2)
    ax3.quiver(
        apple[0], apple[1], apple[2],
        expected_pole[0], expected_pole[1], expected_pole[2],
        length=arrow_len, color="red", linewidth=2, label="stem⊥ pole",
    )

    proxy_points: list[np.ndarray] = []
    for i, direction in enumerate(fib_dirs):
        scene = _build_robot_facing_scene(
            fs,
            seed=seed,
            robot_base_pos=COUPLED_ROBOT_BASE_POS,
            weld_direction=(float(direction[0]), float(direction[1]), float(direction[2])),
        )
        geom = _weld_geometry(scene, COUPLED_ROBOT_BASE_POS)
        proxy = geom["proxy_pos"]
        proxy_points.append(proxy)
        color = cmap(i % 10)
        ax3.scatter(*proxy, color=color, s=45)
        ax3.plot(
            [apple[0], proxy[0]],
            [apple[1], proxy[1]],
            [apple[2], proxy[2]],
            color=color,
            linewidth=1.5,
        )
        ax3.quiver(
            apple[0], apple[1], apple[2],
            direction[0], direction[1], direction[2],
            length=arrow_len * 0.7,
            color=color,
            alpha=0.7,
            linewidth=1.0,
        )
        ax3.text(proxy[0], proxy[1], proxy[2], str(i), fontsize=8)

    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    _set_equal_aspect_3d(ax3, np.vstack([apple, *proxy_points]))
    fig3.tight_layout()
    out_fib = _viz_output_path(tmp_path, "robot_facing_weld_fibonacci.png")
    fig3.savefig(out_fib, dpi=150)
    plt.close(fig3)

    print(f"Saved {out_multi}")
    print(f"Saved {out_ring}")
    print(f"Saved {out_fib}")
    assert out_multi.exists()
    assert out_ring.exists()
    assert out_fib.exists()


def _set_equal_aspect_3d(ax, points: np.ndarray) -> None:
    """Matplotlib 3D axes default to distorted aspect; rescale to a cube."""
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
