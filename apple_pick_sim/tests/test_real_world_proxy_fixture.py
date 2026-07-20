"""Tests for real-world bench proxy fixture (nominal + variance)."""

from __future__ import annotations

import pytest

from apple_pick_sim.tests.conftest import FIXTURES_DIR

NOMINAL_PATH = FIXTURES_DIR / "fruiting_system_ranges_real_world_proxy.json"
VARIANCE_PATH = FIXTURES_DIR / "fruiting_system_ranges_real_world_proxy_variance.json"


def _import_fs():
    import apple_pick_sim.fruiting_system as fs

    return fs


@pytest.fixture(scope="module")
def nominal_ranges():
    return _import_fs().load_ranges(NOMINAL_PATH)


@pytest.fixture(scope="module")
def variance_ranges():
    return _import_fs().load_ranges(VARIANCE_PATH)


def test_nominal_loads_and_topology(nominal_ranges):
    """Nominal fixture: primary→spur→stem→apple; secondary omitted."""
    r = nominal_ranges
    assert r["secondary"] is None
    assert r["primary"] is not None
    assert r["spur"] is not None
    assert r["stem"] is not None
    assert r["apple"] is not None


def test_nominal_placement_args(nominal_ranges):
    fs = _import_fs()
    args = fs.parse_fixture_args(nominal_ranges)
    assert args.fruiting_base_pos == pytest.approx((0.0, 0.5, 0.95))
    assert args.robot_base_pos == pytest.approx((0.0, 0.0, 0.0))


def test_nominal_primary_direction(nominal_ranges):
    primary = nominal_ranges["primary"]
    assert primary["azimuth_deg"]["min"] == primary["azimuth_deg"]["max"] == 0
    assert primary["elevation_deg"]["min"] == primary["elevation_deg"]["max"] == 0


def test_nominal_stiffness_tier(nominal_ranges):
    e = nominal_ranges["primary"]["youngs_modulus_pa"]
    assert e["min"] > 0.0
    assert e["max"] > e["min"]


def test_nominal_spur_hang_angle(nominal_ranges):
    spur = nominal_ranges["spur"]
    assert spur["elevation_delta_deg"] == {"min": -90.0, "max": -90.0}


def test_nominal_defaults_to_t_topology(nominal_ranges):
    assert nominal_ranges.get("topology", "t_junction") == "t_junction"


def test_nominal_geometry_bounds(nominal_ranges):
    assert nominal_ranges["primary"]["length"] == {"min": 0.10, "max": 0.20}
    assert nominal_ranges["spur"]["length"] == {"min": 0.01, "max": 0.1}
    assert nominal_ranges["stem"]["length"] == {"min": 0.01, "max": 0.02}
    assert nominal_ranges["apple"]["radius"] == {"min": 0.04, "max": 0.08}


def test_variance_loads_and_topology(variance_ranges):
    r = variance_ranges
    assert r["secondary"] is None
    assert r["primary"] is not None
    assert r["spur"] is not None
    assert r["stem"] is not None
    assert r["apple"] is not None


def test_variance_sim_build_knobs(variance_ranges):
    fs = _import_fs()
    sb = fs.parse_sim_build(variance_ranges)
    assert sb is not None
    assert sb.vic_gains.linear_k == pytest.approx(100.0)
    assert sb.vic_gains.linear_d == pytest.approx(20.0)
    assert sb.vic_gains.angular_k == pytest.approx(10.0)
    assert sb.vic_gains.angular_d == pytest.approx(3.0)
    # Critical kd at midpoint geometry: support uses kp=1e4; other roles ke=1e5.
    ang = sb.joint_angular_kd_overrides
    lin = sb.joint_linear_kd_overrides
    assert ang["support"] == pytest.approx(130.48, rel=0.01)
    assert ang["primary_spur"] == pytest.approx(0.25651, rel=0.01)
    assert ang["spur_stem"] == pytest.approx(0.01527, rel=0.01)
    assert ang["stem_apple"] == pytest.approx(10.233, rel=0.01)
    assert lin["support"] == pytest.approx(4718.7, rel=0.01)
    assert lin["primary_spur"] == pytest.approx(31.134, rel=0.01)
    assert lin["spur_stem"] == pytest.approx(4.8541, rel=0.01)
    assert lin["stem_apple"] == pytest.approx(323.6, rel=0.01)
    assert sb.joint_angular_kp_overrides == {"support": 10000.0}
    assert sb.joint_linear_kp_overrides == {"support": 10000.0}


def test_nominal_has_no_sim_build(nominal_ranges):
    fs = _import_fs()
    assert fs.parse_sim_build(nominal_ranges) is None


def test_variance_stiffness_ordering(variance_ranges):
    spur_max = variance_ranges["spur"]["youngs_modulus_pa"]["max"]
    primary_min = variance_ranges["primary"]["youngs_modulus_pa"]["min"]
    assert spur_max < primary_min


def _variance_segment_midpoint(row: dict) -> tuple[float, float, float, int]:
    def _mid(band: dict) -> float:
        return 0.5 * (float(band["min"]) + float(band["max"]))

    return (
        _mid(row["length"]),
        _mid(row["radius"]),
        _mid(row["density"]),
        int(_mid(row["num_segments"])),
    )


def test_variance_vbd_stretch_fixed_critical_damping(variance_ranges):
    """``vbd_stretch_fixed.stretch_damping`` is critical (ζ=1) at midpoint geometry."""
    import math

    from apple_pick_sim.fruiting_system.params import _segment_material_geometry

    for seg in ("primary", "spur", "stem"):
        row = variance_ranges[seg]
        length, radius, density, num_segments = _variance_segment_midpoint(row)
        _area, _inertia, _l_seg, m_seg, _j_seg = _segment_material_geometry(
            radius, length, num_segments, density
        )
        fixed = row["vbd_stretch_fixed"]
        k = float(fixed["stretch_stiffness"])
        c_crit = 2.0 * math.sqrt(k * m_seg)
        assert fixed["stretch_damping"] == pytest.approx(c_crit, rel=0.01)


def test_variance_stretch_stiffness_decoupled_from_bend_e(variance_ranges):
    """Spur/stem axial k uses a fixed woodlike stretch, not the bend E band."""
    for seg in ("spur", "stem"):
        row = variance_ranges[seg]
        k_stretch = float(row["vbd_stretch_fixed"]["stretch_stiffness"])
        e_lo = float(row["youngs_modulus_pa"]["min"])
        e_hi = float(row["youngs_modulus_pa"]["max"])
        assert k_stretch > 0.0
        # Fixed stretch must not scale with the bend-E band endpoints.
        assert k_stretch != pytest.approx(e_lo)
        assert k_stretch != pytest.approx(e_hi)
        assert k_stretch == pytest.approx(1.0e7, rel=0.05)


def test_variance_youngs_modulus_hits_proxy_tip_tier_at_midpoint_geometry(variance_ranges):
    """E bands map to documented proxy k_tip=3EI/L^3 tiers at segment midpoint L,r."""
    import math

    def _I(r: float) -> float:
        return math.pi * r**4 / 4.0

    def _k_tip(e: float, length: float, radius: float) -> float:
        return 3.0 * e * _I(radius) / length**3

    def _mid(band: dict) -> float:
        return 0.5 * (float(band["min"]) + float(band["max"]))

    # Current variance fixture targets stiff primary/spur and a softer stem tip.
    tiers = {
        "primary": (14084.0, 14084.0),
        "spur": (14043.0, 28086.0),
        "stem": (1150.0, 1438.0),
    }
    for seg, (k_lo, k_hi) in tiers.items():
        row = variance_ranges[seg]
        length = _mid(row["length"])
        radius = _mid(row["radius"])
        e_lo = float(row["youngs_modulus_pa"]["min"])
        e_hi = float(row["youngs_modulus_pa"]["max"])
        assert _k_tip(e_lo, length, radius) == pytest.approx(k_lo, rel=0.01)
        assert _k_tip(e_hi, length, radius) == pytest.approx(k_hi, rel=0.01)


def test_proxy_material_damping_ratio_is_fraction_of_critical(nominal_ranges, variance_ranges):
    """Apple-branch ζ bands are dimensionless fractions of critical damping."""
    for ranges in (nominal_ranges, variance_ranges):
        for seg in ("primary", "spur", "stem"):
            zeta = ranges[seg]["damping_ratio"]
            assert zeta["min"] <= zeta["max"]
            assert 0.0 <= zeta["min"] <= 1.0
            assert 0.0 <= zeta["max"] <= 1.0


def test_nominal_spur_E_below_primary(nominal_ranges):
    spur_max = nominal_ranges["spur"]["youngs_modulus_pa"]["max"]
    primary_min = nominal_ranges["primary"]["youngs_modulus_pa"]["min"]
    assert spur_max < primary_min


def test_variance_spur_angular_ranges(variance_ranges):
    spur = variance_ranges["spur"]
    assert spur["elevation_delta_deg"] == {"min": -90.0, "max": 0.0}
    assert spur["lateral_delta_deg"] == {"min": -180.0, "max": 180.0}


def test_variance_stem_angular_ranges(variance_ranges):
    stem = variance_ranges["stem"]
    assert stem["elevation_delta_deg"] == {"min": -90.0, "max": 90.0}
    assert stem["lateral_delta_deg"] == {"min": -180.0, "max": 180.0}


def test_placeholder_ee_mass():
    fs = _import_fs()
    assert fs.PLACEHOLDER_EE_MASS_KG == 0.5
    assert fs.GripperProxyConfig().mass == 0.5


def test_default_gripper_proxy_cylinder_dims():
    fs = _import_fs()
    cfg = fs.GripperProxyConfig()
    assert cfg.shape == "cylinder"
    assert cfg.cylinder_radius == pytest.approx(0.05)
    assert cfg.cylinder_half_height == pytest.approx(0.07)


def test_coupled_scene_gripper_proxy_uses_cylinder():
    fs = _import_fs()
    import newton

    ranges = fs.load_ranges(NOMINAL_PATH)
    scene = fs.generate_coupled_cable_scene(
        ranges,
        seed=0,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True, robot_facing_weld=True),
        robot_base_pos=(0.0, 0.0, 0.0),
    )
    model = scene.model
    proxy_body = scene.gripper_proxy_body
    shape_body = model.shape_body.numpy()
    shape_type = model.shape_type.numpy()
    shape_scale = model.shape_scale.numpy()
    shape_xform = model.shape_transform.numpy().reshape(-1, 7)

    proxy_shapes = [
        i
        for i in range(model.shape_count)
        if int(shape_body[i]) == proxy_body and int(shape_type[i]) == newton.GeoType.CYLINDER
    ]
    assert len(proxy_shapes) == 1
    si = proxy_shapes[0]
    radius, half_height, _ = shape_scale[si]
    assert radius == pytest.approx(0.05)
    assert half_height == pytest.approx(0.07)
    assert float(shape_xform[si, 2]) == pytest.approx(0.07)


def test_cylinder_proxy_tip_on_apple_surface_not_inside():
    """Welded cylinder tip sits on the apple; tool bulk is on the robot side (+Z)."""
    import numpy as np
    import warp as wp

    wp.init()
    fs = _import_fs()

    from apple_pick_sim.fruiting_system.gripper_proxy_shape import gripper_proxy_clearance
    from apple_pick_sim.tests.conftest import COUPLED_ROBOT_BASE_POS, COUPLED_VBD_SCENE_KW

    ranges = fs.load_ranges(NOMINAL_PATH)
    scene = fs.generate_coupled_cable_scene(
        ranges,
        seed=11,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True, robot_facing_weld=True),
        robot_base_pos=COUPLED_ROBOT_BASE_POS,
        **COUPLED_VBD_SCENE_KW,
    )
    bq = scene.state_0.body_q.numpy().reshape(-1, 7)
    apple_pos = bq[scene.apple_body, :3]
    proxy_pos = bq[scene.gripper_proxy_body, :3]
    apple_r = float(scene.params.apple_radius)
    dist = float(np.linalg.norm(proxy_pos - apple_pos))
    assert dist == pytest.approx(apple_r, abs=1e-3)
    assert gripper_proxy_clearance(scene.gripper_proxy_config) == pytest.approx(0.0)

    # Back cap (toward apple, local -Z) and flange (local +2*hh) in world frame.
    pq = bq[scene.gripper_proxy_body]
    p = np.asarray(pq[:3], dtype=np.float64)
    q = wp.quat(float(pq[3]), float(pq[4]), float(pq[5]), float(pq[6]))
    hh = scene.gripper_proxy_config.cylinder_half_height
    tip = p
    back = np.asarray(wp.quat_rotate(q, wp.vec3(0.0, 0.0, -2.0 * hh)), dtype=np.float64) + p
    flange = np.asarray(wp.quat_rotate(q, wp.vec3(0.0, 0.0, 2.0 * hh)), dtype=np.float64) + p
    assert float(np.linalg.norm(tip - apple_pos)) >= apple_r - 1e-3
    assert float(np.linalg.norm(back - apple_pos)) >= apple_r - 1e-3
    assert float(np.linalg.norm(flange - apple_pos)) >= apple_r - 1e-3
