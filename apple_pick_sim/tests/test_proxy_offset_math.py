import numpy as np
import warp as wp
import pytest

from apple_pick_sim.coupled_fruiting.explicit_load import (
    apple_com_from_tcp_grasp_offset,
    apple_explicit_wrench_about_tcp,
)

wp.init()

def test_apple_com_from_tcp_grasp_offset_identity_rot():
    """Verify offset with purely translational grasp offset (old behavior)."""
    p_tcp = np.array([1.0, 2.0, 3.0])
    q_tcp = wp.quat_identity()
    
    # 7D offset: pos=(0, 0, -0.1), rot=identity
    offset_7d = (0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 1.0)
    
    p_apple = apple_com_from_tcp_grasp_offset(p_tcp, q_tcp, offset_7d)
    
    # If offset is -0.1 along Z, then proxy is at apple - 0.1 Z.
    # So apple is at proxy + 0.1 Z.
    expected = np.array([1.0, 2.0, 3.1])
    np.testing.assert_allclose(p_apple, expected)


def test_apple_com_from_tcp_grasp_offset_with_proxy_rotation():
    """Verify that a 7D offset correctly unwinds the orientation."""
    # TCP is rotated 90 degrees around Y. (Z points to world X).
    q_tcp = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2.0)
    p_tcp = np.array([1.0, 2.0, 3.0])
    
    # Proxy offset simulates a random approach direction.
    # Approach dir is Apple's +Y axis. So proxy offset is at -R along Y.
    # Proxy rotation must map local +Z to +Y (e.g. rotate 90 deg around +X).
    q_off = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2.0)
    off_pos = (0.0, -0.1, 0.0)
    
    offset_7d = (off_pos[0], off_pos[1], off_pos[2], q_off[0], q_off[1], q_off[2], q_off[3])
    
    p_apple = apple_com_from_tcp_grasp_offset(p_tcp, q_tcp, offset_7d)
    
    # Expected Math:
    # In proxy's local frame, apple is located at (0, 0, 0.1) because
    # X_off^-1 * (0,0,0) -> (0, 0, 0.1)
    # When TCP is rotated 90 deg around Y, the proxy's local +Z points to world +X.
    # So the apple is +0.1 along world X from the TCP.
    expected = np.array([1.1, 2.0, 3.0])
    np.testing.assert_allclose(p_apple, expected, atol=1e-5)
    
    # The lever arm from TCP to Apple COM is p_apple - p_tcp = (0.1, 0, 0).
    r = p_apple - p_tcp
    np.testing.assert_allclose(r, np.array([0.1, 0.0, 0.0]), atol=1e-5)


def test_apple_explicit_wrench_about_tcp_with_rotation():
    """Verify torque transfers correctly using the true unwound 7D lever arm."""
    mass_kg = 1.0
    gravity = wp.vec3(0.0, 0.0, -10.0)
    
    p_tcp = np.array([1.0, 2.0, 3.0])
    q_tcp = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2.0)
    
    q_off = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2.0)
    off_pos = (0.0, -0.1, 0.0)
    offset_7d = (off_pos[0], off_pos[1], off_pos[2], q_off[0], q_off[1], q_off[2], q_off[3])
    
    f, tau = apple_explicit_wrench_about_tcp(
        mass_kg, gravity, p_tcp,
        grasp_offset_in_apple_frame=offset_7d,
        tcp_orientation_world=q_tcp
    )
    
    # Support force by robot should be opposite to weight => (0, 0, +mg)
    expected_f = np.array([0.0, 0.0, 10.0])
    np.testing.assert_allclose(f, expected_f)
    
    # Using the true lever arm r = (0.1, 0, 0)
    # tau = r x f = (0.1, 0, 0) x (0, 0, 10) = (0, -1.0, 0)
    # Note: If the old logic were used, the lever arm would be incorrectly
    # calculated as (0, 0.1, 0), resulting in a bogus torque of (1.0, 0, 0).
    expected_tau = np.array([0.0, -1.0, 0.0])
    np.testing.assert_allclose(tau, expected_tau, atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])
