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
    EE_TCP_ORIENT_WXYZ,
    TESTFR3_SCENE_USD,
)

REPO = Path(__file__).resolve().parents[2]
AUTHORING_USD = REPO / "assets" / "testfr3.usda"

# link7 visuals max-z ≈ 0.1068; fr3_joint8 localPos0.z ≈ 0.113 → ~6.2 mm gap.
# Cylinder proximal face is shifted into ee +Z by this gap so it meets link7 mesh.
_LINK7_JOINT_Z_M = 0.11299999
_LINK7_MESH_MAX_Z_M = 0.106800005
_LINK7_FLANGE_GAP_M = _LINK7_JOINT_Z_M - _LINK7_MESH_MAX_Z_M
_TOOL_LENGTH_M = 2.0 * EE_CYLINDER_HALF_HEIGHT


def test_layout_math_tip_equals_tcp_when_authored_consistently():
    """Tip-out is ee −Z; proximal face may sit slightly on +Z to meet link7 mesh."""
    gap = _LINK7_FLANGE_GAP_M
    tip_m = gap - _TOOL_LENGTH_M
    ee_sz = _TOOL_LENGTH_M
    mesh_t = (gap / ee_sz) - 0.5
    tcp_local = tip_m / ee_sz
    layout = ee_cylinder_layout_from_authored(
        ee_scale_xyz=(0.2, 0.2, ee_sz),
        mesh_translate_xyz=(0.0, 0.0, mesh_t),
        mesh_scale_xyz=(0.5, 0.5, 1.0),
        mesh_z_min=-0.5,
        mesh_z_max=0.5,
        tcp_translate_xyz=(0.0, 0.0, tcp_local),
    )
    assert layout.length_m == pytest.approx(_TOOL_LENGTH_M, abs=1e-9)
    assert layout.radius_m == pytest.approx(0.05, abs=1e-9)
    assert_tip_flange_tcp_contract(layout, expected_flange_z_m=gap)
    assert layout.tip_z_m == pytest.approx(tip_m, abs=1e-9)
    assert layout.tip_z_m == pytest.approx(layout.tcp_z_m, abs=1e-9)


def test_authoring_usd_tip_flange_tcp_contract():
    authored = scrape_ee_cylinder_authored(AUTHORING_USD)
    layout = ee_cylinder_layout_from_authored(**{
        k: v for k, v in authored.items() if k != "tcp_orient_wxyz"
    })
    assert_tip_flange_tcp_contract(
        layout, expected_flange_z_m=_LINK7_FLANGE_GAP_M, flange_tol_m=1e-3
    )
    assert layout.tip_z_m < 0.0
    assert layout.tcp_z_m < 0.0
    assert layout.length_m == pytest.approx(_TOOL_LENGTH_M, abs=1e-3)


def test_resolved_and_authoring_tcp_orient_is_rotx_180():
    """TCP body +Z must be tip-out (logged Franka); ee −Z geometry stays via joint8."""
    for path in (TESTFR3_SCENE_USD, AUTHORING_USD):
        authored = scrape_ee_cylinder_authored(path)
        q = authored["tcp_orient_wxyz"]
        # Accept ± double-cover of RotX(180) ≈ (0, ±1, 0, 0) in wxyz.
        target = EE_TCP_ORIENT_WXYZ
        err = min(
            sum((a - b) ** 2 for a, b in zip(q, target, strict=True)) ** 0.5,
            sum((a + b) ** 2 for a, b in zip(q, target, strict=True)) ** 0.5,
        )
        assert err < 1e-3, f"{path}: tcp orient {q} != RotX(180) {target}"


def test_resolved_usd_tip_flange_tcp_contract():
    authored = scrape_ee_cylinder_authored(TESTFR3_SCENE_USD)
    layout = ee_cylinder_layout_from_authored(**{
        k: v for k, v in authored.items() if k != "tcp_orient_wxyz"
    })
    assert layout.radius_m == pytest.approx(EE_CYLINDER_RADIUS, abs=1e-3)
    assert layout.length_m == pytest.approx(_TOOL_LENGTH_M, abs=1e-3)
    assert_tip_flange_tcp_contract(
        layout, expected_flange_z_m=_LINK7_FLANGE_GAP_M, flange_tol_m=1e-3
    )
    assert layout.tip_z_m < 0.0
    assert layout.tcp_z_m < 0.0