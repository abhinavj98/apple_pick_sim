"""Tests for anisotropic (per-axis Kp/Kd) VIC pose wrench law."""

from __future__ import annotations

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


def test_damping_opposes_velocity_with_v_des_zero():
    tf = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
    qd = wp.spatial_vector(0.2, 0.0, 0.0, 0.0, 0.0, 0.0)
    kp_lin = wp.vec3(0.0, 0.0, 0.0)
    kp_ang = wp.vec3(0.0, 0.0, 0.0)
    kd_lin = wp.vec3(10.0, 10.0, 10.0)
    kd_ang = wp.vec3(1.0, 1.0, 1.0)
    w = compute_vic_spatial_wrench_aniso(tf, qd, tf, kp_lin, kp_ang, kd_lin, kd_ang)
    assert float(w[0]) < -1.9  # -kd * v = -10 * 0.2
