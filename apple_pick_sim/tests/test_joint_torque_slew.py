"""Unit tests for VIC joint-torque slew (Continuous_Force_RL 0.2 N·m/ms)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from apple_pick_sim.coupled_fruiting.vic_joint_torques import (
    apply_joint_torque_slew_to_scene,
    apply_vic_joint_torques_to_scene,
    slew_joint_torques,
)
from apple_pick_sim.coupled_fruiting.vic_joint_torques_batched import (
    apply_vic_joint_torques_batched_to_scene,
)

_RATE = 200.0  # N·m/s — match Continuous_Force_RL _MAX_TORQUE_DELTA = 0.2 at 1 kHz
_N_ARM = 7


def test_slew_reaches_step_in_fifteen_ms():
    """3 N·m OSC step at 200 N·m/s and dt=1 ms needs 15 steps."""
    tau_prev = np.zeros(7, dtype=np.float64)
    tau_osc = np.full(7, 3.0, dtype=np.float64)
    dt = 0.001
    for _ in range(14):
        tau_prev = slew_joint_torques(tau_osc, tau_prev, dt=dt, rate_nm_s=_RATE)
        assert float(np.max(np.abs(tau_prev))) < 3.0 - 1e-9
    tau_sent = slew_joint_torques(tau_osc, tau_prev, dt=dt, rate_nm_s=_RATE)
    np.testing.assert_allclose(tau_sent, tau_osc, atol=1e-12)


def test_slew_rate_zero_is_identity():
    tau_prev = np.zeros(7, dtype=np.float64)
    tau_osc = np.linspace(-5.0, 5.0, 7, dtype=np.float64)
    tau_sent = slew_joint_torques(tau_osc, tau_prev, dt=0.001, rate_nm_s=0.0)
    np.testing.assert_allclose(tau_sent, tau_osc)


def test_slew_batched_envs_independent():
    tau_prev = np.zeros((2, 7), dtype=np.float64)
    tau_osc = np.zeros((2, 7), dtype=np.float64)
    tau_osc[0, :] = 3.0
    tau_osc[1, :] = -1.0
    dt = 0.001
    tau_sent = slew_joint_torques(tau_osc, tau_prev, dt=dt, rate_nm_s=_RATE)
    np.testing.assert_allclose(tau_sent[0], 0.2)
    np.testing.assert_allclose(tau_sent[1], -0.2)


def test_slew_per_joint_independent_not_shared_scale():
    """Large and small joint steps must not share one scale factor."""
    tau_prev = np.zeros(7, dtype=np.float64)
    tau_osc = np.array([3.0, 0.05, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
    tau_sent = slew_joint_torques(tau_osc, tau_prev, dt=0.001, rate_nm_s=_RATE)
    np.testing.assert_allclose(
        tau_sent,
        np.array([0.2, 0.05, 0.0, 0.0, 0.0, 0.0, -0.2], dtype=np.float64),
        atol=1e-12,
    )


def test_slew_decreasing_and_inside_band():
    tau_prev = np.full(7, 3.0, dtype=np.float64)
    tau_osc = np.zeros(7, dtype=np.float64)
    tau_sent = slew_joint_torques(tau_osc, tau_prev, dt=0.001, rate_nm_s=_RATE)
    np.testing.assert_allclose(tau_sent, np.full(7, 2.8), atol=1e-12)

    tau_prev = np.array([1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    tau_osc = np.array([1.05, -0.55, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    tau_sent = slew_joint_torques(tau_osc, tau_prev, dt=0.001, rate_nm_s=_RATE)
    np.testing.assert_allclose(tau_sent, tau_osc, atol=1e-12)


def test_slew_rejects_negative_rate():
    with pytest.raises(ValueError, match="rate_nm_s"):
        slew_joint_torques(
            np.zeros(7),
            np.zeros(7),
            dt=0.001,
            rate_nm_s=-1.0,
        )


def test_slew_rejects_non_positive_dt():
    with pytest.raises(ValueError, match="dt"):
        slew_joint_torques(
            np.zeros(7),
            np.zeros(7),
            dt=0.0,
            rate_nm_s=_RATE,
        )


def test_apply_slew_to_scene_ramps_then_matches_osc():
    """Scene helper ramps joint_f over substeps toward a fixed OSC request."""
    dof_per = 10
    num_envs = 2
    layout = SimpleNamespace(num_envs=num_envs, joint_dof_count_per_world=dof_per)
    joint_f = wp.zeros(num_envs * dof_per, dtype=float, device="cpu")
    sent = wp.zeros((num_envs, _N_ARM), dtype=float, device="cpu")
    control = SimpleNamespace(joint_f=joint_f)
    scene = SimpleNamespace(
        layout=layout,
        robot_control=control,
        vic_jt_sent_tau=sent,
        vic_jt_num_envs=num_envs,
        vic_jt_torque_slew_nm_s=_RATE,
    )
    osc = np.full((num_envs, _N_ARM), 3.0, dtype=np.float32)
    gripper = np.full((num_envs, dof_per - _N_ARM), 99.0, dtype=np.float32)
    dt = 0.001
    for step in range(15):
        jf = joint_f.numpy().reshape(num_envs, dof_per)
        jf[:, :_N_ARM] = osc
        jf[:, _N_ARM:] = gripper
        joint_f.assign(jf.reshape(-1))
        apply_joint_torque_slew_to_scene(scene, dt=dt)
        got = joint_f.numpy().reshape(num_envs, dof_per)
        np.testing.assert_allclose(got[:, _N_ARM:], gripper, atol=1e-6)
        if step < 14:
            assert float(np.max(np.abs(got[:, :_N_ARM]))) < 3.0 - 1e-6
    got = joint_f.numpy().reshape(num_envs, dof_per)
    np.testing.assert_allclose(got[:, :_N_ARM], osc, atol=1e-5)
    np.testing.assert_allclose(got[:, _N_ARM:], gripper, atol=1e-6)
    np.testing.assert_allclose(sent.numpy(), osc, atol=1e-5)


def test_apply_slew_rate_zero_leaves_osc():
    joint_f = wp.array(np.linspace(1.0, 7.0, 7).astype(np.float32), dtype=float, device="cpu")
    sent = wp.zeros(7, dtype=float, device="cpu")
    scene = SimpleNamespace(
        layout=None,
        robot_control=SimpleNamespace(joint_f=joint_f),
        vic_jt_sent_tau=sent,
        vic_jt_torque_slew_nm_s=0.0,
    )
    before = joint_f.numpy().copy()
    apply_joint_torque_slew_to_scene(scene, dt=0.001)
    np.testing.assert_allclose(joint_f.numpy(), before)


def test_apply_slew_rejects_negative_rate():
    joint_f = wp.zeros(7, dtype=float, device="cpu")
    sent = wp.zeros(7, dtype=float, device="cpu")
    scene = SimpleNamespace(
        layout=None,
        robot_control=SimpleNamespace(joint_f=joint_f),
        vic_jt_sent_tau=sent,
        vic_jt_torque_slew_nm_s=-1.0,
    )
    with pytest.raises(ValueError, match="rate_nm_s"):
        apply_joint_torque_slew_to_scene(scene, dt=0.001)


def test_apply_slew_matches_numpy_reference():
    """One scene-helper step matches NumPy slew_joint_torques."""
    dof_per = 10
    num_envs = 2
    layout = SimpleNamespace(num_envs=num_envs, joint_dof_count_per_world=dof_per)
    osc = np.array(
        [
            [3.0, 0.05, 0.0, 0.0, 0.0, 0.0, -1.0],
            [-2.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.5],
        ],
        dtype=np.float64,
    )
    prev = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.4, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    expected = slew_joint_torques(osc, prev, dt=0.001, rate_nm_s=_RATE)

    joint_f = wp.zeros(num_envs * dof_per, dtype=float, device="cpu")
    sent = wp.array(prev.astype(np.float32), dtype=float, device="cpu")
    jf = joint_f.numpy().reshape(num_envs, dof_per)
    jf[:, :_N_ARM] = osc.astype(np.float32)
    joint_f.assign(jf.reshape(-1))
    scene = SimpleNamespace(
        layout=layout,
        robot_control=SimpleNamespace(joint_f=joint_f),
        vic_jt_sent_tau=sent,
        vic_jt_num_envs=num_envs,
        vic_jt_torque_slew_nm_s=_RATE,
    )
    apply_joint_torque_slew_to_scene(scene, dt=0.001)
    got = joint_f.numpy().reshape(num_envs, dof_per)[:, :_N_ARM]
    np.testing.assert_allclose(got, expected, atol=1e-5)
    np.testing.assert_allclose(sent.numpy(), expected, atol=1e-5)


def test_apply_vic_joint_torques_to_scene_slews_when_dt_passed():
    """Wrapper with dt= slews even when launch_apply_* no-ops (robot_model None)."""
    joint_f = wp.zeros(7, dtype=float, device="cpu")
    sent = wp.zeros(7, dtype=float, device="cpu")
    jf = joint_f.numpy()
    jf[:] = 3.0
    joint_f.assign(jf)
    scene = SimpleNamespace(
        layout=None,
        robot_model=None,
        robot_state_0=None,
        robot_control=SimpleNamespace(joint_f=joint_f),
        vic_jt_sent_tau=sent,
        vic_jt_torque_slew_nm_s=_RATE,
        vic_controller=object(),
        vic_target_tf=object(),
        vic_target_twist=object(),
        vic_gains=None,
    )
    apply_vic_joint_torques_to_scene(scene, dt=0.001)
    np.testing.assert_allclose(joint_f.numpy(), np.full(7, 0.2), atol=1e-5)
    np.testing.assert_allclose(sent.numpy(), np.full(7, 0.2), atol=1e-5)


def test_apply_vic_joint_torques_batched_to_scene_slews_when_dt_passed():
    dof_per = 10
    num_envs = 2
    layout = SimpleNamespace(num_envs=num_envs, joint_dof_count_per_world=dof_per)
    joint_f = wp.zeros(num_envs * dof_per, dtype=float, device="cpu")
    sent = wp.zeros((num_envs, _N_ARM), dtype=float, device="cpu")
    jf = joint_f.numpy().reshape(num_envs, dof_per)
    jf[:, :_N_ARM] = 3.0
    joint_f.assign(jf.reshape(-1))
    scene = SimpleNamespace(
        layout=layout,
        robot_model=None,
        robot_state_0=None,
        robot_control=SimpleNamespace(joint_f=joint_f),
        vic_jt_sent_tau=sent,
        vic_jt_num_envs=num_envs,
        vic_jt_torque_slew_nm_s=_RATE,
        vic_controller=object(),
        vic_target_positions_wp=object(),
        vic_target_rotations_wp=object(),
        vic_gains=None,
    )
    apply_vic_joint_torques_batched_to_scene(scene, dt=0.001)
    got = joint_f.numpy().reshape(num_envs, dof_per)[:, :_N_ARM]
    np.testing.assert_allclose(got, 0.2, atol=1e-5)
    np.testing.assert_allclose(sent.numpy(), 0.2, atol=1e-5)
