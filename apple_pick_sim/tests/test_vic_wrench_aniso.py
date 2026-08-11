"""Tests for anisotropic (per-axis Kp/Kd) VIC pose wrench law."""

from __future__ import annotations

import numpy as np
import warp as wp

from apple_pick_sim.coupled_fruiting.vic_wrench import compute_vic_spatial_wrench_aniso


def test_zero_wrench_at_target_zero_velocity():
    tf = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_identity())
    qd = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    kp_lin = wp.vec3(100.0, 100.0, 100.0)
    kp_ang = wp.vec3(10.0, 10.0, 10.0)
    kd_lin = wp.vec3(5.0, 5.0, 5.0)
    kd_ang = wp.vec3(1.0, 1.0, 1.0)
    w = compute_vic_spatial_wrench_aniso(tf, qd, tf, kp_lin, kp_ang, kd_lin, kd_ang)
    for i in range(6):
        assert abs(float(w[i])) < 1e-6


def test_per_axis_gain_scales_only_that_axis():
    tcp_tf = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
    target_tf = wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity())
    qd = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    kp_lin = wp.vec3(100.0, 400.0, 100.0)
    kp_ang = wp.vec3(0.0, 0.0, 0.0)
    kd_lin = wp.vec3(0.0, 0.0, 0.0)
    kd_ang = wp.vec3(0.0, 0.0, 0.0)
    w = compute_vic_spatial_wrench_aniso(tcp_tf, qd, target_tf, kp_lin, kp_ang, kd_lin, kd_ang)
    assert abs(float(w[0]) - 10.0) < 1e-3  # 100 * 0.1
    assert abs(float(w[1])) < 1e-6  # error is on x only
    assert abs(float(w[2])) < 1e-6


def test_angular_error_torque_scales_with_kp_ang():
    """Torque about z from a yaw error scales linearly with ``kp_ang[2]``."""
    angle = 0.2
    half = 0.5 * angle
    q_des = wp.quat(0.0, 0.0, float(np.sin(half)), float(np.cos(half)))  # xyzw about +z
    tcp_tf = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
    target_tf = wp.transform(wp.vec3(0.0, 0.0, 0.0), q_des)
    qd = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    zero3 = wp.vec3(0.0, 0.0, 0.0)

    w_low = compute_vic_spatial_wrench_aniso(
        tcp_tf, qd, target_tf, zero3, wp.vec3(0.0, 0.0, 10.0), zero3, zero3
    )
    w_high = compute_vic_spatial_wrench_aniso(
        tcp_tf, qd, target_tf, zero3, wp.vec3(0.0, 0.0, 40.0), zero3, zero3
    )
    assert abs(float(w_low[5]) - 10.0 * angle) < 1e-4
    assert abs(float(w_high[5]) - 40.0 * angle) < 1e-4
    assert abs(float(w_low[3])) < 1e-6  # yaw-only error leaves x/y torque at zero
    assert abs(float(w_low[4])) < 1e-6
    assert abs(float(w_low[0])) < 1e-6  # no position error -> no force


def test_damping_opposes_velocity_with_v_des_zero():
    tf = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
    qd = wp.spatial_vector(0.2, 0.0, 0.0, 0.0, 0.0, 0.0)
    kp_lin = wp.vec3(0.0, 0.0, 0.0)
    kp_ang = wp.vec3(0.0, 0.0, 0.0)
    kd_lin = wp.vec3(10.0, 10.0, 10.0)
    kd_ang = wp.vec3(1.0, 1.0, 1.0)
    w = compute_vic_spatial_wrench_aniso(tf, qd, tf, kp_lin, kp_ang, kd_lin, kd_ang)
    assert float(w[0]) < -1.9  # -kd * v = -10 * 0.2
