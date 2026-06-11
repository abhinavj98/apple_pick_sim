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


def test_vic_sync_target_from_state():
    vic = Fr3EEImpedanceController()

    class _BodyQ:
        @staticmethod
        def numpy() -> np.ndarray:
            return np.array([[0.5, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

    class _State:
        body_q = _BodyQ()

    vic.sync_target_from_state(_State(), tcp_body_index=0)
    pos = wp.transform_get_translation(vic.target_tf)
    assert float(pos[0]) == pytest.approx(0.5)
    assert float(pos[1]) == pytest.approx(0.2)
    assert float(pos[2]) == pytest.approx(1.0)


def test_vic_advance_target_integrates_linear_velocity():
    vic = Fr3EEImpedanceController()
    vic.target_tf = _target_tf(pos=(0.0, 0.0, 0.0))
    dt = 0.1
    vel = EEVelocity(linear=(0.5, 0.0, 0.0))
    vic.advance_target(vel, dt)
    pos = wp.transform_get_translation(vic.target_tf)
    assert float(pos[0]) == pytest.approx(0.05, rel=1e-6)
    assert float(pos[1]) == pytest.approx(0.0, abs=1e-9)
    assert float(pos[2]) == pytest.approx(0.0, abs=1e-9)


def test_run_coupled_teleop_frame_holds_setpoint_when_arm_sags():
    """Zero command must not re-anchor the setpoint to a sagged actual pose."""
    vic = Fr3EEImpedanceController(tcp_body_index=0)
    anchor_pos = (0.5, 0.2, 1.0)
    sagged_pos = (0.5, 0.2, 0.9)

    class _BodyQ:
        def __init__(self, pos: tuple[float, float, float]) -> None:
            self._pos = pos

        def numpy(self) -> np.ndarray:
            return np.array(
                [[*self._pos, 0.0, 0.0, 0.0, 1.0]],
                dtype=np.float64,
            )

    class _State:
        def __init__(self, pos: tuple[float, float, float]) -> None:
            self.body_q = _BodyQ(pos)

    vic.sync_target_from_state(_State(anchor_pos))
    anchor = wp.transform_get_translation(vic.target_tf)

    vic.run_coupled_teleop_frame(
        _State(sagged_pos),
        control=None,
        mj_solver=None,
        dt=1.0 / 15.0,
        velocity=EEVelocity(),
    )

    pos = wp.transform_get_translation(vic.target_tf)
    np.testing.assert_allclose(
        [float(pos[0]), float(pos[1]), float(pos[2])],
        [float(anchor[0]), float(anchor[1]), float(anchor[2])],
        rtol=0,
        atol=1e-9,
    )


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
