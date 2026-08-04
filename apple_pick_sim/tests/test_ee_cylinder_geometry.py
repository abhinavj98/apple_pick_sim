from pathlib import Path

import pytest

from apple_pick_sim.robot.fr3_robot.ee_cylinder_geometry import (
    assert_tip_flange_tcp_contract,
    ee_cylinder_layout_from_authored,
    scrape_ee_cylinder_authored,
)
from apple_pick_sim.robot.fr3_robot.paths import (
    EE_CYLINDER_HALF_HEIGHT,
    EE_CYLINDER_RADIUS,
    TESTFR3_SCENE_USD,
)

REPO = Path(__file__).resolve().parents[2]
AUTHORING_USD = REPO / "assets" / "testfr3.usda"


def test_layout_math_tip_equals_tcp_when_authored_consistently():
    layout = ee_cylinder_layout_from_authored(
        ee_scale_xyz=(0.2, 0.2, 0.14),
        mesh_translate_xyz=(0.0, 0.0, 0.5),
        mesh_scale_xyz=(0.5, 0.5, 1.0),
        mesh_z_min=-0.5,
        mesh_z_max=0.5,
        tcp_translate_xyz=(0.0, 0.0, 1.0),
    )
    assert layout.length_m == pytest.approx(0.14, abs=1e-9)
    assert layout.radius_m == pytest.approx(0.05, abs=1e-9)
    assert_tip_flange_tcp_contract(layout)
    assert layout.flange_z_m == pytest.approx(0.0, abs=1e-9)
    assert layout.tip_z_m == pytest.approx(layout.tcp_z_m, abs=1e-9)


def test_resolved_usd_tip_flange_tcp_contract():
    authored = scrape_ee_cylinder_authored(TESTFR3_SCENE_USD)
    layout = ee_cylinder_layout_from_authored(**authored)
    assert layout.radius_m == pytest.approx(EE_CYLINDER_RADIUS, abs=1e-3)
    assert layout.length_m == pytest.approx(2.0 * EE_CYLINDER_HALF_HEIGHT, abs=1e-3)
    assert_tip_flange_tcp_contract(layout)


def test_authoring_usd_tip_flange_tcp_contract():
    authored = scrape_ee_cylinder_authored(AUTHORING_USD)
    layout = ee_cylinder_layout_from_authored(**authored)
    assert_tip_flange_tcp_contract(layout)
