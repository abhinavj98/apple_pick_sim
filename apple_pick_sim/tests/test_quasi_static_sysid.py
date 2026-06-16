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
    derive_n_steps,
    estimate_trajectory_frames,
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


def test_quasi_static_config_derives_n_steps():
    assert derive_n_steps(movement_per_step_m=0.05, total_movement_m=0.10) == 2
    assert derive_n_steps(movement_per_step_m=0.05, total_movement_m=0.05) == 1
    with pytest.raises(ValueError, match="integer multiple"):
        derive_n_steps(movement_per_step_m=0.05, total_movement_m=0.12)


def test_quasi_static_trajectory_phase_sequence():
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    config = QuasiStaticStepConfig(
        movement_per_step_m=0.05,
        total_movement_m=0.10,
        hold_duration_s=0.1,
        control_hz=10.0,
        skip_return=False,
    )
    traj = QuasiStaticTrajectory(directions, config)

    phases: list[str] = []
    for phase, _ in _collect_trajectory(traj):
        if not phases or phases[-1] != phase:
            phases.append(phase)
    per_dir = ["move_out", "hold", "move_out", "hold", "return"]
    expected = per_dir + per_dir
    assert phases == expected


def test_quasi_static_trajectory_returns_to_center():
    directions = np.array([[1.0, 0.0, 0.0]])
    config = QuasiStaticStepConfig(
        movement_per_step_m=0.02,
        total_movement_m=0.06,
        move_speed_mps=0.2,
        hold_duration_s=0.0,
        control_hz=60.0,
        skip_return=False,
    )
    traj = QuasiStaticTrajectory(directions, config)

    displacement = np.zeros(3, dtype=np.float64)
    frame_dt = 1.0 / config.control_hz
    for phase, vel in _collect_trajectory(traj):
        if phase in ("move_out", "return"):
            displacement += np.asarray(vel.linear, dtype=np.float64) * frame_dt

    np.testing.assert_allclose(displacement, 0.0, atol=1e-9)


def test_quasi_static_trajectory_hold_frame_count():
    config = QuasiStaticStepConfig(
        movement_per_step_m=0.05,
        total_movement_m=0.10,
        hold_duration_s=1.5,
        control_hz=60.0,
    )
    directions = np.array([[0.0, 0.0, 1.0]])
    traj = QuasiStaticTrajectory(directions, config)

    expected_per_hold = int(math.ceil(config.hold_duration_s * config.control_hz))
    hold_frames = [v for phase, v in _collect_trajectory(traj) if phase == "hold"]
    assert len(hold_frames) == 2 * expected_per_hold
    for vel in hold_frames:
        assert vel.is_zero()


def test_quasi_static_trajectory_amplitude_at_holds():
    config = QuasiStaticStepConfig(
        movement_per_step_m=0.05,
        total_movement_m=0.10,
        hold_duration_s=0.1,
        control_hz=10.0,
    )
    directions = np.array([[1.0, 0.0, 0.0]])
    traj = QuasiStaticTrajectory(directions, config)

    hold_amplitudes: list[float] = []
    prev_phase: str | None = None
    for phase, _ in traj.iter_frames():
        if phase == "hold" and prev_phase != "hold":
            hold_amplitudes.append(traj.current_amplitude_m)
        prev_phase = phase

    np.testing.assert_allclose(
        hold_amplitudes,
        [0.05, 0.10],
        rtol=1e-6,
        atol=1e-9,
    )


def test_quasi_static_trajectory_move_burst_duration():
    config = QuasiStaticStepConfig(
        movement_per_step_m=0.05,
        total_movement_m=0.10,
        move_speed_mps=0.2,
        hold_duration_s=0.1,
        control_hz=60.0,
    )
    directions = np.array([[0.0, 0.0, 1.0]])
    traj = QuasiStaticTrajectory(directions, config)

    expected_burst = max(
        1, int(math.ceil(config.movement_per_step_m / config.move_speed_mps * config.control_hz))
    )
    burst_lengths: list[int] = []
    run_len = 0
    prev_phase: str | None = None
    for phase, _ in traj.iter_frames():
        if phase == "move_out":
            run_len += 1
        elif prev_phase == "move_out":
            burst_lengths.append(run_len)
            run_len = 0
        prev_phase = phase

    assert burst_lengths == [expected_burst, expected_burst]


def test_quasi_static_trajectory_move_speed():
    config = QuasiStaticStepConfig(
        movement_per_step_m=0.05,
        total_movement_m=0.10,
        move_speed_mps=0.2,
        hold_duration_s=0.0,
        control_hz=60.0,
        skip_return=False,
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


def test_estimate_trajectory_frames():
    config = QuasiStaticStepConfig(
        movement_per_step_m=0.05,
        total_movement_m=0.10,
        hold_duration_s=1.5,
        move_speed_mps=0.2,
        control_hz=60.0,
        skip_return=False,
    )
    assert estimate_trajectory_frames(config, n_directions=2) == 2 * (
        2
        * (
            max(1, int(math.ceil(0.05 / 0.2 * 60.0)))
            + int(math.ceil(1.5 * 60.0))
        )
        + max(1, int(math.ceil(0.10 / 0.2 * 60.0)))
    )


def test_skip_return_no_return_frames():
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    config = QuasiStaticStepConfig(
        movement_per_step_m=0.05,
        total_movement_m=0.10,
        hold_duration_s=0.1,
        control_hz=10.0,
        skip_return=True,
    )
    traj = QuasiStaticTrajectory(directions, config)
    phases = [phase for phase, _ in _collect_trajectory(traj)]
    assert "return" not in phases


def test_skip_return_estimate_trajectory_frames():
    config = QuasiStaticStepConfig(
        movement_per_step_m=0.05,
        total_movement_m=0.10,
        hold_duration_s=1.5,
        move_speed_mps=0.2,
        control_hz=60.0,
        skip_return=True,
    )
    with_return = QuasiStaticStepConfig(
        movement_per_step_m=0.05,
        total_movement_m=0.10,
        hold_duration_s=1.5,
        move_speed_mps=0.2,
        control_hz=60.0,
        skip_return=False,
    )
    return_frames = max(1, int(math.ceil(0.10 / 0.2 * 60.0)))
    n_directions = 3
    assert estimate_trajectory_frames(config, n_directions=n_directions) == (
        estimate_trajectory_frames(with_return, n_directions=n_directions)
        - return_frames * n_directions
    )


def test_excitation_context_frozen():
    ctx = ExcitationContext(type="quasi_static", f_inst=0.0, direction=np.array([1.0, 0.0, 0.0]))
    with pytest.raises(Exception):
        ctx.f_inst = 1.0  # type: ignore[misc]
