"""CPU-only tests for §2.1 quasi-static stiffness mapping utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from apple_pick_sim.system_id.excitation_state import ExcitationContext
from apple_pick_sim.system_id.fibonacci_hemisphere import sample_fibonacci_hemisphere
from apple_pick_sim.system_id.quasi_static_trajectory import (
    QuasiStaticStepConfig,
    QuasiStaticTrajectory,
)


def test_fibonacci_hemisphere_unit_norms():
    stem = np.array([0.0, 0.0, 1.0])
    dirs = sample_fibonacci_hemisphere(10, stem)
    norms = np.linalg.norm(dirs, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-6, atol=1e-6)


def test_fibonacci_hemisphere_forward_facing():
    stem = np.array([0.3, -0.2, 0.9])
    stem = stem / np.linalg.norm(stem)
    dirs = sample_fibonacci_hemisphere(10, stem)
    dots = dirs @ stem
    assert np.all(dots >= -1e-9)


def test_fibonacci_hemisphere_count():
    stem = np.array([1.0, 0.0, 0.0])
    for n in (1, 5, 10, 20):
        dirs = sample_fibonacci_hemisphere(n, stem)
        assert dirs.shape == (n, 3)


def test_fibonacci_hemisphere_approx_uniform():
    stem = np.array([0.0, 0.0, 1.0])
    dirs = sample_fibonacci_hemisphere(10, stem)
    min_angle_rad = math.radians(15.0)
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            dot = float(np.clip(dirs[i] @ dirs[j], -1.0, 1.0))
            angle = math.acos(dot)
            assert angle >= min_angle_rad - 1e-6, f"dirs {i},{j} too close: {math.degrees(angle):.1f}°"


def _collect_trajectory(traj: QuasiStaticTrajectory) -> list[tuple[str, object]]:
    return list(traj.iter_frames())


def test_quasi_static_trajectory_phase_sequence():
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    config = QuasiStaticStepConfig(n_steps=2, hold_duration_s=0.1, control_hz=10.0)
    traj = QuasiStaticTrajectory(directions, config)

    phases: list[str] = []
    for phase, _ in _collect_trajectory(traj):
        if not phases or phases[-1] != phase:
            phases.append(phase)
    per_dir = ["move_out", "hold", "return"]
    expected = per_dir + per_dir
    assert phases == expected


def test_quasi_static_trajectory_returns_to_center():
    directions = np.array([[1.0, 0.0, 0.0]])
    config = QuasiStaticStepConfig(
        step_size_m=0.02,
        n_steps=3,
        move_speed_mps=0.05,
        hold_duration_s=0.0,
        control_hz=60.0,
    )
    traj = QuasiStaticTrajectory(directions, config)

    displacement = np.zeros(3, dtype=np.float64)
    frame_dt = 1.0 / config.control_hz
    for phase, vel in _collect_trajectory(traj):
        if phase in ("move_out", "return"):
            displacement += np.asarray(vel.linear, dtype=np.float64) * frame_dt

    np.testing.assert_allclose(displacement, 0.0, atol=1e-9)


def test_quasi_static_trajectory_hold_frame_count():
    config = QuasiStaticStepConfig(hold_duration_s=1.5, control_hz=60.0, n_steps=1)
    directions = np.array([[0.0, 0.0, 1.0]])
    traj = QuasiStaticTrajectory(directions, config)

    expected = int(math.ceil(config.hold_duration_s * config.control_hz))
    hold_frames = [v for phase, v in _collect_trajectory(traj) if phase == "hold"]
    assert len(hold_frames) == expected
    for vel in hold_frames:
        assert vel.is_zero()


def test_quasi_static_trajectory_move_speed():
    config = QuasiStaticStepConfig(
        step_size_m=0.02,
        n_steps=2,
        move_speed_mps=0.05,
        hold_duration_s=0.0,
        control_hz=60.0,
    )
    directions = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    traj = QuasiStaticTrajectory(directions, config)

    eps = 1e-9
    for phase, vel in _collect_trajectory(traj):
        if phase in ("move_out", "return"):
            speed = float(np.linalg.norm(vel.linear))
            assert speed <= config.move_speed_mps + eps
            if speed > eps:
                assert abs(speed - config.move_speed_mps) < 1e-6


def test_excitation_context_frozen():
    ctx = ExcitationContext(type="quasi_static", f_inst=0.0, direction=np.array([1.0, 0.0, 0.0]))
    with pytest.raises(Exception):
        ctx.f_inst = 1.0  # type: ignore[misc]
