"""Fibonacci-lattice sampling on the forward hemisphere."""

from __future__ import annotations

import numpy as np


def _fibonacci_sphere(n_samples: int) -> np.ndarray:
    """Return ``(n_samples, 3)`` unit vectors on the full sphere."""
    if n_samples <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    if n_samples == 1:
        return np.array([[0.0, 0.0, 1.0]], dtype=np.float64)

    indices = np.arange(n_samples, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n_samples)
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    theta = 2.0 * np.pi * indices / golden_ratio
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    pts = np.stack([x, y, z], axis=1)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _fibonacci_hemisphere_local(
    n_samples: int,
    *,
    max_polar_angle: float = 0.5 * np.pi,
) -> np.ndarray:
    """Approximately uniform unit directions on a polar cap around ``+Z``.

    ``max_polar_angle`` is the largest allowed angle from ``+Z`` (radians).
    ``pi/2`` yields the full northern hemisphere; smaller values tighten the
    cap toward the pole.
    """
    if n_samples <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    if n_samples == 1:
        return np.array([[0.0, 0.0, 1.0]], dtype=np.float64)

    max_polar_angle = float(max_polar_angle)
    if not (0.0 < max_polar_angle <= 0.5 * np.pi + 1e-9):
        raise ValueError("max_polar_angle must be in (0, pi/2]")

    cos_min = float(np.cos(max_polar_angle))
    indices = np.arange(n_samples, dtype=np.float64) + 0.5
    # ``z`` uniform on [cos_min, 1] gives area-uniform coverage on the cap.
    z = cos_min + (1.0 - cos_min) * (1.0 - indices / n_samples)
    phi = np.arccos(np.clip(z, -1.0, 1.0))
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    theta = 2.0 * np.pi * indices / golden_ratio
    sin_phi = np.sin(phi)
    x = np.cos(theta) * sin_phi
    y = np.sin(theta) * sin_phi
    pts = np.stack([x, y, z], axis=1)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _rotation_matrix_from_z_to(direction: np.ndarray) -> np.ndarray:
    """Return ``R`` with ``R @ [0, 0, 1] == direction`` (both unit)."""
    z_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    d_norm = float(np.linalg.norm(d))
    if d_norm < 1e-12:
        raise ValueError("direction must be non-zero")
    d = d / d_norm

    dot = float(np.dot(z_hat, d))
    if dot > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-12:
        return np.diag([1.0, -1.0, -1.0]).astype(np.float64)

    v = np.cross(z_hat, d)
    s = float(np.linalg.norm(v))
    c = dot
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + vx @ vx * ((1.0 - c) / (s * s))


def stem_perpendicular_robot_pole(
    stem_dir: np.ndarray,
    robot_vec: np.ndarray,
) -> np.ndarray:
    """Unit pole perpendicular to stem and facing robot.

    Projects ``robot_vec`` onto the plane normal to ``stem_dir`` and normalizes.
    Falls back to an arbitrary perpendicular when ``robot_vec`` is nearly
    parallel to the stem.
    """
    stem = np.asarray(stem_dir, dtype=np.float64).reshape(3)
    stem_norm = float(np.linalg.norm(stem))
    if stem_norm < 1e-12:
        raise ValueError("stem_dir must be non-zero")
    stem = stem / stem_norm

    robot = np.asarray(robot_vec, dtype=np.float64).reshape(3)
    perp = robot - float(np.dot(robot, stem)) * stem
    perp_norm = float(np.linalg.norm(perp))
    if perp_norm < 1e-12:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        perp = np.cross(stem, ref)
        perp_norm = float(np.linalg.norm(perp))
        if perp_norm < 1e-12:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            perp = np.cross(stem, ref)
            perp_norm = float(np.linalg.norm(perp))
        if perp_norm < 1e-12:
            raise ValueError("cannot construct perpendicular to stem_dir")
        perp = perp / perp_norm
        if float(np.dot(perp, robot)) < 0.0:
            perp = -perp
        return perp.astype(np.float64)

    return (perp / perp_norm).astype(np.float64)


def _filter_horizontal_toward_pole(
    directions: np.ndarray,
    pole: np.ndarray,
    *,
    min_horizontal_dot: float,
) -> np.ndarray:
    """Keep directions whose XY projection aligns with the pole's horizontal part."""
    pole_xy = np.asarray(pole[:2], dtype=np.float64)
    pole_xy_norm = float(np.linalg.norm(pole_xy))
    if pole_xy_norm < 1e-9:
        return directions

    pole_xy_hat = pole_xy / pole_xy_norm
    keep: list[np.ndarray] = []
    for d in directions:
        d_xy = np.asarray(d[:2], dtype=np.float64)
        d_xy_norm = float(np.linalg.norm(d_xy))
        if d_xy_norm < 1e-9:
            continue
        if float(np.dot(d_xy / d_xy_norm, pole_xy_hat)) >= min_horizontal_dot - 1e-9:
            keep.append(np.asarray(d, dtype=np.float64))
    if not keep:
        return directions
    return np.stack(keep, axis=0)


def sample_robot_facing_pull_directions(
    n: int,
    physical_stem: np.ndarray,
    robot_vec: np.ndarray,
    *,
    max_polar_angle: float = 0.5 * np.pi,
    min_horizontal_dot: float | None = None,
) -> np.ndarray:
    """Sample ``n`` pull directions on a pole-centered cap toward the robot.

    The cap center (pole) is ``stem_perpendicular_robot_pole(physical_stem,
    robot_vec)``: perpendicular to the physical stem and facing the fixture
    robot base. Default ``max_polar_angle=π/2`` yields a full hemisphere around
    that pole (same path as ``example_gym_sysid.py`` data collection).
    """
    pole = stem_perpendicular_robot_pole(physical_stem, robot_vec)
    return sample_fibonacci_hemisphere(
        n,
        pole,
        max_polar_angle=max_polar_angle,
        min_horizontal_dot=min_horizontal_dot,
    )


def sample_fibonacci_hemisphere(
    n: int,
    stem_dir: np.ndarray,
    *,
    max_polar_angle: float = 0.5 * np.pi,
    min_horizontal_dot: float | None = None,
) -> np.ndarray:
    """Sample ``n`` unit directions on a polar cap centered at ``stem_dir``.

    Builds a golden-ratio Fibonacci lattice on the ``+Z`` cap, rotates the cap
    so ``+Z`` aligns with ``stem_dir``, and returns world-frame directions.
    Every output satisfies ``dot(direction, stem_dir) >= cos(max_polar_angle)``.

    When ``min_horizontal_dot`` is set (e.g. ``0.0``), directions must also
    lie in the horizontal half-plane toward ``stem_dir`` (positive dot between
    their XY projection and the pole's XY projection).
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    stem = np.asarray(stem_dir, dtype=np.float64).reshape(3)
    stem_norm = float(np.linalg.norm(stem))
    if stem_norm < 1e-12:
        raise ValueError("stem_dir must be non-zero")
    stem = stem / stem_norm

    pool_n = max(n * 4, 64) if min_horizontal_dot is not None else n
    local = _fibonacci_hemisphere_local(pool_n, max_polar_angle=max_polar_angle)
    rot = _rotation_matrix_from_z_to(stem)
    world = local @ rot.T
    norms = np.linalg.norm(world, axis=1, keepdims=True)
    world = world / norms

    if min_horizontal_dot is not None:
        world = _filter_horizontal_toward_pole(
            world,
            stem,
            min_horizontal_dot=float(min_horizontal_dot),
        )

    if len(world) == 0:
        world = np.array([stem], dtype=np.float64)

    selected: list[np.ndarray] = []
    for i in range(n):
        selected.append(world[i % len(world)].copy())

    out = np.stack(selected, axis=0)
    out_norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / out_norms).astype(np.float64)
