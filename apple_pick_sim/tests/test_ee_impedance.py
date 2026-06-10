"""Unit tests for FR3 TCP variable-impedance wrench law."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import (
    Fr3EEImpedanceController,
    ImpedanceGains,
)
from apple_pick_sim.robot.fr3_robot.controllers.keyboard import EEVelocity


def _tcp_q7(
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> np.ndarray:
    return np.array([*pos, *quat], dtype=np.float64)


def _tcp_qd(
    linear: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    return np.array([*linear, *angular], dtype=np.float64)


def _target_tf(
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> wp.transform:
    return wp.transform(wp.vec3(*pos), wp.quat(*quat))


@pytest.fixture
def vic() -> Fr3EEImpedanceController:
    return Fr3EEImpedanceController()


@pytest.fixture
def unit_gains() -> ImpedanceGains:
    return ImpedanceGains(linear_k=100.0, linear_d=10.0, angular_k=5.0, angular_d=0.5)


def test_vic_zero_wrench_at_target_pose_and_twist(vic: Fr3EEImpedanceController):
    target = _target_tf(pos=(0.5, 0.2, 1.0))
    tcp_q = _tcp_q7(pos=(0.5, 0.2, 1.0))
    w = vic.compute_applied_wrench(
        target_tf=target,
        target_twist=EEVelocity(),
        tcp_body_q=tcp_q,
        tcp_body_qd=_tcp_qd(),
    )
    np.testing.assert_allclose(w, 0.0, atol=1e-5)


def test_vic_linear_force_proportional_to_position_error(
    vic: Fr3EEImpedanceController, unit_gains: ImpedanceGains
):
    target = _target_tf(pos=(0.0, 0.0, 0.0))
    tcp_q = _tcp_q7(pos=(0.01, 0.0, 0.0))
    w = vic.compute_applied_wrench(
        target_tf=target,
        target_twist=EEVelocity(),
        tcp_body_q=tcp_q,
        tcp_body_qd=_tcp_qd(),
        gains=unit_gains,
    )
    assert w[0] == pytest.approx(-unit_gains.linear_k * 0.01, rel=1e-6)
    assert abs(w[1]) < 1e-9
    assert abs(w[2]) < 1e-9


def test_vic_damping_opposes_velocity(vic: Fr3EEImpedanceController, unit_gains: ImpedanceGains):
    target = _target_tf()
    tcp_q = _tcp_q7()
    tcp_qd = _tcp_qd(linear=(0.1, 0.0, 0.0))
    w = vic.compute_applied_wrench(
        target_tf=target,
        target_twist=EEVelocity(),
        tcp_body_q=tcp_q,
        tcp_body_qd=tcp_qd,
        gains=unit_gains,
    )
    assert w[0] == pytest.approx(-unit_gains.linear_d * 0.1, rel=1e-6)
    assert abs(w[1]) < 1e-9
    assert abs(w[2]) < 1e-9
