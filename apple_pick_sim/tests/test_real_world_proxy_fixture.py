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
    bs = nominal_ranges["primary"]["bend_stiffness"]
    assert bs["min"] >= 210.0
    assert bs["max"] <= 736.0


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


def test_variance_stiffness_ordering(variance_ranges):
    spur_max = variance_ranges["spur"]["bend_stiffness"]["max"]
    primary_min = variance_ranges["primary"]["bend_stiffness"]["min"]
    assert spur_max < primary_min


def test_variance_spur_angular_ranges(variance_ranges):
    spur = variance_ranges["spur"]
    assert spur["elevation_delta_deg"] == {"min": -90.0, "max": 90.0}
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
