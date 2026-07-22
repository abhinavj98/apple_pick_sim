"""Unit tests for joint-kd scaling with Young's modulus."""

from __future__ import annotations

import math

import pytest

from apple_pick_sim.fruiting_system import params as fs
from apple_pick_sim.tests.conftest import RANGES_FIXTURE


def _base_primary_spur_stem(seed: int = 0) -> fs.FruitingSystemParams:
    ranges = fs.load_ranges(RANGES_FIXTURE)
    return fs.sample_params(ranges, seed=seed, omit=("secondary",))


def test_youngs_modulus_ref_from_ranges_is_geometric_mid():
    from apple_pick_sim.fruiting_system.joint_kd_scaling import (
        youngs_modulus_ref_from_ranges,
    )

    ranges = {
        "primary": {"youngs_modulus_pa": {"min": 1.0e9, "max": 4.0e9}},
        "spur": {"youngs_modulus_pa": {"min": 1.0e8, "max": 1.0e8}},
        "stem": {"youngs_modulus_pa": {"min": 4.0e8, "max": 1.6e9}},
    }
    ref = youngs_modulus_ref_from_ranges(ranges)
    assert ref["primary"] == pytest.approx(2.0e9)
    assert ref["spur"] == pytest.approx(1.0e8)
    assert ref["stem"] == pytest.approx(8.0e8)


def test_scale_joint_kd_leaves_support_unscaled_and_scales_distal_roles():
    from apple_pick_sim.fruiting_system.joint_kd_scaling import (
        scale_joint_kd_overrides,
    )

    base = _base_primary_spur_stem()
    assert base.spur is not None and base.stem is not None
    ref_e = {
        "primary": float(base.primary.youngs_modulus_pa),
        "spur": float(base.spur.youngs_modulus_pa),
        "stem": float(base.stem.youngs_modulus_pa),
    }
    kd = {
        "support": 100.0,
        "primary_spur": 1.0,
        "spur_stem": 2.0,
        "stem_apple": 3.0,
    }
    stiff = fs.set_rod_youngs_modulus(base, "stem", 10.0 * ref_e["stem"])
    stiff = fs.set_rod_youngs_modulus(stiff, "spur", 4.0 * ref_e["spur"])

    out = scale_joint_kd_overrides(kd, stiff, ref_e)
    assert out["support"] == pytest.approx(100.0)
    assert out["primary_spur"] == pytest.approx(1.0 * math.sqrt(4.0))
    assert out["spur_stem"] == pytest.approx(2.0 * math.sqrt(10.0))
    assert out["stem_apple"] == pytest.approx(3.0 * math.sqrt(10.0))


def test_scale_joint_kd_at_ref_e_is_identity():
    from apple_pick_sim.fruiting_system.joint_kd_scaling import (
        scale_joint_kd_overrides,
        youngs_modulus_ref_from_ranges,
    )

    ranges = fs.load_ranges(RANGES_FIXTURE)
    ref_e = youngs_modulus_ref_from_ranges(ranges)
    # Build params exactly at ref E so scale == 1.
    base = _base_primary_spur_stem()
    at_ref = base
    for seg, e in ref_e.items():
        if getattr(at_ref, seg) is not None:
            at_ref = fs.set_rod_youngs_modulus(at_ref, seg, e)
    kd = {
        "support": 10.0,
        "primary_spur": 0.5,
        "spur_stem": 0.25,
        "stem_apple": 1.5,
    }
    out = scale_joint_kd_overrides(kd, at_ref, ref_e)
    assert out == pytest.approx(kd)


def test_scale_joint_kd_skips_missing_rod_roles():
    from apple_pick_sim.fruiting_system.joint_kd_scaling import (
        scale_joint_kd_overrides,
    )

    rod = fs.rod_params_from_material(
        youngs_modulus_pa=1.0e9,
        damping_ratio=0.1,
        length=0.3,
        radius=0.02,
        density=400.0,
        num_segments=4,
        direction=(1.0, 0.0, 0.0),
    )
    params = fs.FruitingSystemParams(
        primary=rod,
        secondary=None,
        spur=None,
        stem=None,
        apple_radius=0.05,
        apple_density=500.0,
    )
    kd = {"support": 1.0, "primary_spur": 2.0, "spur_stem": 3.0}
    ref_e = {"primary": 1.0e9, "spur": 1.0e8, "stem": 1.0e8}
    out = scale_joint_kd_overrides(kd, params, ref_e)
    assert out["support"] == pytest.approx(1.0)
    assert out["primary_spur"] == pytest.approx(2.0)  # no spur rod → leave as-is
    assert out["spur_stem"] == pytest.approx(3.0)  # no stem rod → leave as-is


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
