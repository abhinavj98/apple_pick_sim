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
    # Joint weld damping via ζ (kd = ζ·2·√(k·I)/√(k·m)); mutually exclusive with absolute kd maps.
    assert sb.joint_damping_ratio == pytest.approx(0.5)
    assert sb.joint_angular_kd_overrides == {}
    assert sb.joint_linear_kd_overrides == {}
    assert sb.joint_angular_kp_overrides == {"support": 10000.0}
    assert sb.joint_linear_kp_overrides == {"support": 10000.0}


def test_nominal_has_no_sim_build(nominal_ranges):
    fs = _import_fs()
    assert fs.parse_sim_build(nominal_ranges) is None


def test_variance_stiffness_ordering(variance_ranges):
    """Primary is wood-scale; stem min is peduncle-scale; spur may reach wood max."""
    primary = variance_ranges["primary"]["youngs_modulus_pa"]
    spur = variance_ranges["spur"]["youngs_modulus_pa"]
    stem = variance_ranges["stem"]["youngs_modulus_pa"]
    assert primary["min"] == pytest.approx(5.0e9, rel=0.01)
    assert stem["min"] < 1.5e9  # peduncle-like floor
    assert spur["min"] < primary["min"]
    assert stem["min"] < primary["min"]


def _variance_segment_midpoint(row: dict) -> tuple[float, float, float, int]:
    def _mid(band: dict) -> float:
        return 0.5 * (float(band["min"]) + float(band["max"]))

    return (
        _mid(row["length"]),
        _mid(row["radius"]),
        _mid(row["density"]),
        int(_mid(row["num_segments"])),
    )


def test_variance_vbd_stretch_force_derives_at_midpoint(variance_ranges):
    """Axial k/c from max_force_n + ζ_stretch at midpoint geometry."""
    import math

    from apple_pick_sim.fruiting_system.params import stretch_knobs_from_max_force

    zeta_by_seg = {"primary": 1.0, "spur": 1.5, "stem": 3.0}
    for seg, zeta in zeta_by_seg.items():
        row = variance_ranges[seg]
        force = row["vbd_stretch_force"]
        assert float(force["max_force_n"]) == pytest.approx(35.0)
        assert float(force["damping_ratio"]) == pytest.approx(zeta)
        length, radius, density, num_segments = _variance_segment_midpoint(row)
        k_exp, c_exp = stretch_knobs_from_max_force(
            float(force["max_force_n"]),
            float(force["damping_ratio"]),
            length,
            radius,
            density,
            num_segments,
        )
        # Spot-check δ = 0.05 L_seg policy.
        l_seg = length / num_segments
        assert k_exp == pytest.approx(35.0 / (0.05 * l_seg))
        assert c_exp == pytest.approx(
            2.0 * zeta * math.sqrt(k_exp * density * math.pi * radius**2 * l_seg)
        )


def test_variance_stretch_force_decoupled_from_bend_e(variance_ranges):
    """Axial force policy is independent of bend youngs_modulus_pa bands."""
    for seg in ("spur", "stem"):
        row = variance_ranges[seg]
        f_max = float(row["vbd_stretch_force"]["max_force_n"])
        e_lo = float(row["youngs_modulus_pa"]["min"])
        e_hi = float(row["youngs_modulus_pa"]["max"])
        assert f_max > 0.0
        assert f_max != pytest.approx(e_lo)
        assert f_max != pytest.approx(e_hi)


def test_sample_params_variance_stretch_matches_force_policy(variance_ranges):
    """sample_params derives stretch knobs from vbd_stretch_force + geometry."""
    from apple_pick_sim.fruiting_system.params import (
        sample_params,
        stretch_knobs_from_max_force,
    )

    params = sample_params(variance_ranges, seed=7)
    for seg in ("primary", "spur", "stem"):
        rod = getattr(params, seg)
        assert rod is not None
        force = variance_ranges[seg]["vbd_stretch_force"]
        k_exp, c_exp = stretch_knobs_from_max_force(
            float(force["max_force_n"]),
            float(force["damping_ratio"]),
            rod.length,
            rod.radius,
            rod.density,
            rod.num_segments,
        )
        assert rod.stretch_stiffness == pytest.approx(k_exp)
        assert rod.stretch_damping == pytest.approx(c_exp)


def test_variance_youngs_modulus_tip_stiffness_at_midpoint_geometry(variance_ranges):
    """Document cantilever tip stiffness k=3EI/L^3 implied by current E bands."""
    import math

    def _I(r: float) -> float:
        return math.pi * r**4 / 4.0

    def _k_tip(e: float, length: float, radius: float) -> float:
        return 3.0 * e * _I(radius) / length**3

    def _mid(band: dict) -> float:
        return 0.5 * (float(band["min"]) + float(band["max"]))

    # Regenerated from fixture geometry + E (wood/peduncle literature bands).
    tiers = {
        "primary": (70422.0, 70422.0),
        "spur": (25144.0, 502872.0),
        "stem": (1150.0, 7191.0),
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
    assert fs.PLACEHOLDER_EE_MASS_KG == 1.1
    assert fs.GripperProxyConfig().mass == 1.1


def test_fr3_ee_mass_matches_proxy_default():
    from apple_pick_sim.robot import fr3_robot

    fs = _import_fs()
    assert fr3_robot.EE_MASS_KG == pytest.approx(1.1)
    assert fr3_robot.EE_MASS_KG == pytest.approx(fs.PLACEHOLDER_EE_MASS_KG)


def test_default_gripper_proxy_cylinder_dims():
    fs = _import_fs()
    cfg = fs.GripperProxyConfig()
    assert cfg.shape == "cylinder"
    assert cfg.cylinder_radius == pytest.approx(0.05)
    assert cfg.cylinder_half_height == pytest.approx(0.09)


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
    assert half_height == pytest.approx(0.09)
    # Distal tip at body origin; bulk toward flange along local -Z (USD tip-out).
    assert float(shape_xform[si, 2]) == pytest.approx(-0.09)


def test_gripper_proxy_cylinder_tcp_at_distal_tip_bulk_neg_z():
    """Body origin is the distal tip face; cylinder center sits at local -half_height."""
    import warp as wp

    wp.init()
    from apple_pick_sim.fruiting_system.gripper_proxy_shape import gripper_proxy_cylinder_tcp_xform
    from apple_pick_sim.fruiting_system.params import GripperProxyConfig

    cfg = GripperProxyConfig(cylinder_half_height=0.07)
    tf = gripper_proxy_cylinder_tcp_xform(cfg)
    p = wp.transform_get_translation(tf)
    assert float(p[0]) == pytest.approx(0.0)
    assert float(p[1]) == pytest.approx(0.0)
    assert float(p[2]) == pytest.approx(-0.07)


def test_cylinder_proxy_tip_on_apple_surface_not_inside():
    """Welded cylinder tip sits on the apple; tool bulk is on the robot side (−Z)."""
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

    # Tip-out: local +Z toward apple; flange at local -2*hh (away from fruit).
    pq = bq[scene.gripper_proxy_body]
    p = np.asarray(pq[:3], dtype=np.float64)
    q = wp.quat(float(pq[3]), float(pq[4]), float(pq[5]), float(pq[6]))
    hh = scene.gripper_proxy_config.cylinder_half_height
    tip = p
    flange = np.asarray(wp.quat_rotate(q, wp.vec3(0.0, 0.0, -2.0 * hh)), dtype=np.float64) + p
    tip_d = float(np.linalg.norm(tip - apple_pos))
    flange_d = float(np.linalg.norm(flange - apple_pos))
    assert tip_d >= apple_r - 1e-3
    assert flange_d >= apple_r - 1e-3
    assert flange_d > tip_d + hh  # bulk away from apple
    # Proxy +Z points tip-out toward apple center.
    plus_z = np.asarray(wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)), dtype=np.float64)
    toward_apple = apple_pos - tip
    toward_apple /= np.linalg.norm(toward_apple)
    assert float(np.dot(plus_z, toward_apple)) > 0.9
