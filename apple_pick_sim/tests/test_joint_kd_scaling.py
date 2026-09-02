"""Unit tests for joint-kd expansion from damping ratio."""

from __future__ import annotations

import math

import pytest


def test_joint_kd_from_damping_ratio_uses_critical_formula():
    import numpy as np

    from apple_pick_sim.fruiting_system.joint_kd_scaling import (
        DEFAULT_RIGID_JOINT_KE,
        joint_kd_from_damping_ratio,
    )

    # One joint per role; child bodies 0..3 with known m / I.
    joints = [
        (0, "joint_primary_support_left"),
        (1, "joint_primary_spur"),
        (2, "joint_spur_stem"),
        (3, "joint_stem_apple"),
    ]
    body_mass = np.array([0.1, 0.01, 0.001, 0.25], dtype=np.float64)
    # Diagonal inertia → I_max = diag entry
    body_inertia = np.zeros((4, 3, 3), dtype=np.float64)
    for i, I in enumerate([1.0e-4, 1.0e-6, 1.0e-8, 1.0e-6]):
        body_inertia[i, 0, 0] = I
        body_inertia[i, 1, 1] = I * 0.5
        body_inertia[i, 2, 2] = I * 0.25
    joint_child = np.array([0, 1, 2, 3], dtype=np.int32)
    ang_kp = {"support": 1.0e4}
    lin_kp = {"support": 1.0e4}
    zeta = 0.2
    ang, lin = joint_kd_from_damping_ratio(
        zeta=zeta,
        fruiting_fixed_joints=joints,
        body_mass=body_mass,
        body_inertia=body_inertia,
        joint_child=joint_child,
        angular_kp_by_role=ang_kp,
        linear_kp_by_role=lin_kp,
    )
    assert set(ang) == {"support", "primary_spur", "spur_stem", "stem_apple"}
    assert set(lin) == set(ang)
    # support uses kp=1e4; others default ke
    assert ang["support"] == pytest.approx(zeta * 2.0 * math.sqrt(1.0e4 * 1.0e-4))
    assert lin["support"] == pytest.approx(zeta * 2.0 * math.sqrt(1.0e4 * 0.1))
    assert ang["primary_spur"] == pytest.approx(
        zeta * 2.0 * math.sqrt(DEFAULT_RIGID_JOINT_KE * 1.0e-6)
    )
    assert lin["stem_apple"] == pytest.approx(
        zeta * 2.0 * math.sqrt(DEFAULT_RIGID_JOINT_KE * 0.25)
    )


def test_joint_kd_from_damping_ratio_skips_unmatched_roles():
    import numpy as np

    from apple_pick_sim.fruiting_system.joint_kd_scaling import (
        joint_kd_from_damping_ratio,
    )

    joints = [(0, "joint_stem_apple")]
    body_mass = np.array([0.2], dtype=np.float64)
    body_inertia = np.eye(3, dtype=np.float64)[None, :, :] * 1.0e-6
    joint_child = np.array([0], dtype=np.int32)
    ang, lin = joint_kd_from_damping_ratio(
        zeta=0.2,
        fruiting_fixed_joints=joints,
        body_mass=body_mass,
        body_inertia=body_inertia,
        joint_child=joint_child,
        angular_kp_by_role={},
        linear_kp_by_role={},
    )
    assert set(ang) == {"stem_apple"}
    assert set(lin) == {"stem_apple"}


def test_support_angular_kp_from_linear_uses_three_quarters_dowel_length_squared():
    from apple_pick_sim.fruiting_system.joint_kd_scaling import (
        SUPPORT_ANGULAR_KP_LENGTH_FACTOR,
        map_support_angular_kp_overrides,
        support_angular_kp_from_linear,
    )

    assert SUPPORT_ANGULAR_KP_LENGTH_FACTOR == pytest.approx(0.75)
    assert support_angular_kp_from_linear(500.0, 0.4) == pytest.approx(
        0.75 * (0.4**2) * 500.0
    )
    assert map_support_angular_kp_overrides(
        {"support": 10000.0},
        {"support": 10000.0},
        dowel_length_m=0.4,
    )["support"] == pytest.approx(0.75 * (0.4**2) * 10000.0)
    assert map_support_angular_kp_overrides(
        {"spur_stem": 1.0e5},
        {},
        dowel_length_m=0.4,
    ) == {"spur_stem": 1.0e5}


def test_support_angular_kp_from_linear_rejects_nonpositive_dowel_length():
    from apple_pick_sim.fruiting_system.joint_kd_scaling import (
        support_angular_kp_from_linear,
    )

    with pytest.raises(ValueError, match="dowel_length"):
        support_angular_kp_from_linear(1000.0, 0.0)
