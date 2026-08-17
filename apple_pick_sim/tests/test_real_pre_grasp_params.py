# apple_pick_sim/tests/test_real_pre_grasp_params.py
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from apple_pick_sim.fruiting_system.params import fruiting_params_to_dict
from apple_pick_sim.system_id.real_pre_grasp_params import (
    PreGraspMappedGeometry,
    coerce_xyz,
    fruiting_params_from_pre_grasp_meta,
    fruiting_params_from_pre_grasp_parquet,
    load_dataset_metadata,
    map_pre_grasp_geometry,
    primary_direction_from_fixture,
    surface_to_centerline,
)

VARIANCE = Path("apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json")
PRIMARY_DIR = primary_direction_from_fixture(VARIANCE)


def test_coerce_xyz_list():
    np.testing.assert_allclose(coerce_xyz([1.0, 2.0, 3.0], field="p"), [1.0, 2.0, 3.0])


def test_coerce_xyz_numpy_string():
    out = coerce_xyz("[-0.00889757  0.94594489  0.40465398]", field="apple_pos")
    np.testing.assert_allclose(out, [-0.00889757, 0.94594489, 0.40465398], rtol=1e-6)


def _synthetic_pre_grasp_meta() -> dict:
    # Branch = T-junction (fruiting_base); Spur = spur end; Apple = fruit
    # Non-collinear hang so spur/stem directions differ.
    branch = [0.0, 0.0, 0.0]
    spur = [0.02, 0.0, -0.10]
    apple = [0.05, 0.0, -0.13]
    return {
        "topology": {
            "junction_names": ["Branch", "Spur", "Apple"],
            "start_nodes": ["Branch", "Branch", "Spur"],
            "end_nodes": ["Spur", "Apple", "Apple"],
            "shared_endpoints": True,
            "n_woody_parts": 3,
        },
        "pre_grasp_geometry": {
            "structure_name": "default_template",
            "parts": {
                "primary": {
                    "length_m": 0.2,
                    "radius_m": 0.0125,
                    "density_kg_m3": 660,
                    "shape": "cylinder",
                },
                "spur": {
                    "length_m": 0.1,
                    "radius_m": 0.0025,
                    "density_kg_m3": 1200,
                    "shape": "cylinder",
                },
                "stem": {
                    "length_m": 0.025,
                    "radius_m": 0.0005,
                    "density_kg_m3": 1000,
                    "shape": "cylinder",
                },
                "apple": {
                    "length_m": 0.08,
                    "radius_m": 0.04,
                    "density_kg_m3": 650,
                    "shape": "sphere",
                },
            },
            "snapshot": {
                "woody_part_start_pos": branch + branch + spur,
                "woody_part_end_pos": spur + apple + apple,
                "woody_bending_angles": [0.0, 0.0, 0.0],
                "apple_pos": apple,
            },
        },
    }


def test_map_pre_grasp_branch_is_fruiting_base_pos():
    mapped = map_pre_grasp_geometry(_synthetic_pre_grasp_meta(), primary_dir=PRIMARY_DIR)
    assert isinstance(mapped, PreGraspMappedGeometry)
    assert mapped.apple_quat_xyzw is None  # no pose/quat on synthetic snapshot
    expected_base = surface_to_centerline(
        (0.0, 0.0, 0.0), mapped.spur_direction, PRIMARY_DIR, 0.0125
    )
    np.testing.assert_allclose(mapped.fruiting_base_pos, expected_base, atol=1e-9)
    spur_u = np.array([0.02, 0.0, -0.10], dtype=np.float64)
    spur_u /= np.linalg.norm(spur_u)
    stem_u = np.array([0.03, 0.0, -0.03], dtype=np.float64)
    stem_u /= np.linalg.norm(stem_u)
    np.testing.assert_allclose(mapped.spur_direction, spur_u, atol=1e-6)
    np.testing.assert_allclose(mapped.stem_direction, stem_u, atol=1e-6)
    assert mapped.spur_direction != mapped.stem_direction
    assert mapped.rod_geometry["primary"]["length_m"] == pytest.approx(0.2)
    assert mapped.rod_geometry["spur"]["radius_m"] == pytest.approx(0.0025)
    assert mapped.rod_geometry["spur"]["density_kg_m3"] == pytest.approx(1200.0)
    assert mapped.apple_radius_m == pytest.approx(0.04)
    assert mapped.apple_density_kg_m3 == pytest.approx(650.0)
    assert mapped.diagnostics.get("rod_density", {}).get("spur") is None
    assert "spur_length_error" in mapped.diagnostics
    # Stem chord = ‖spur_end − apple_CoM‖ − apple_radius (CoM is sphere center).
    spur_to_com = float(np.linalg.norm(np.array([0.05, 0.0, -0.13]) - np.array([0.02, 0.0, -0.10])))
    assert mapped.diagnostics["stem_spur_to_com_m"] == pytest.approx(spur_to_com)
    assert mapped.diagnostics["stem_chord_length_m"] == pytest.approx(spur_to_com - 0.04)


def test_map_pre_grasp_overrides_spur_density_from_mass_kg():
    meta = _synthetic_pre_grasp_meta()
    spur = meta["pre_grasp_geometry"]["parts"]["spur"]
    spur["mass_kg"] = 0.026
    mapped = map_pre_grasp_geometry(meta, primary_dir=PRIMARY_DIR)
    length = float(spur["length_m"])
    radius = float(spur["radius_m"])
    expected_rho = 0.026 / (math.pi * radius * radius * length)
    assert mapped.rod_geometry["spur"]["radius_m"] == pytest.approx(radius)
    assert mapped.rod_geometry["spur"]["length_m"] == pytest.approx(length)
    assert mapped.rod_geometry["spur"]["density_kg_m3"] == pytest.approx(expected_rho)
    diag = mapped.diagnostics["rod_density"]["spur"]
    assert diag["source"] == "mass_kg"
    assert diag["mass_kg"] == pytest.approx(0.026)
    assert diag["catalog_density_kg_m3"] == pytest.approx(1200.0)
    assert diag["density_kg_m3"] == pytest.approx(expected_rho)


def test_map_pre_grasp_rejects_nonpositive_mass_kg():
    meta = _synthetic_pre_grasp_meta()
    meta["pre_grasp_geometry"]["parts"]["spur"]["mass_kg"] = 0.0
    with pytest.raises(ValueError, match="mass_kg"):
        map_pre_grasp_geometry(meta, primary_dir=PRIMARY_DIR)


def test_fruiting_params_uses_mass_derived_spur_density():
    meta = _synthetic_pre_grasp_meta()
    meta["pre_grasp_geometry"]["parts"]["spur"]["mass_kg"] = 0.026
    params, _base, diagnostics = fruiting_params_from_pre_grasp_meta(
        meta, fixture_path=VARIANCE
    )
    spur = meta["pre_grasp_geometry"]["parts"]["spur"]
    expected_rho = 0.026 / (
        math.pi * float(spur["radius_m"]) ** 2 * float(spur["length_m"])
    )
    assert params.spur is not None
    assert params.spur.radius == pytest.approx(float(spur["radius_m"]))
    assert params.spur.density == pytest.approx(expected_rho)
    assert diagnostics["rod_density"]["spur"]["source"] == "mass_kg"


def test_map_pre_grasp_strict_rejects_nonzero_bend():
    meta = _synthetic_pre_grasp_meta()
    meta["pre_grasp_geometry"]["snapshot"]["woody_bending_angles"] = [0.2, 0.0, 0.0]
    with pytest.raises(ValueError, match="bend"):
        map_pre_grasp_geometry(meta, primary_dir=PRIMARY_DIR, strict=True)


def test_map_pre_grasp_prefers_rest_snapshot_during_run():
    """New compiler: woody lives on rest_snapshot_during_run; snapshot is TCP-only."""
    meta = _synthetic_pre_grasp_meta()
    pre = meta["pre_grasp_geometry"]
    woody = pre.pop("snapshot")
    pre["snapshot"] = {
        "tcp_pos": [0.0, 0.9, 0.4],
        "tcp_pose_4x4": [1.0] * 16,
        "target_pose_4x4": [1.0] * 16,
    }
    # Distinct rest geometry so preference is observable.
    branch = [1.0, 2.0, 3.0]
    spur = [1.02, 2.0, 2.90]
    apple = [1.05, 2.0, 2.87]
    # 90° about +Z; translation ignored for quat extract.
    apple_pose_4x4 = [
        0.0,
        -1.0,
        0.0,
        1.05,
        1.0,
        0.0,
        0.0,
        2.0,
        0.0,
        0.0,
        1.0,
        2.87,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    pre["rest_snapshot_during_run"] = {
        "woody_part_start_pos": branch + branch + spur,
        "woody_part_end_pos": spur + apple + apple,
        "woody_bending_angles": [0.0, 0.0, 0.0],
        "apple_pos": "[1.05  2.0  2.87]",  # numpy-string quirk
        "apple_pose_4x4": apple_pose_4x4,
    }
    # Settled differs — must not win over rest.
    pre["settled_snapshot"] = {
        **woody,
        "woody_part_start_pos": [9.0] * 9,
        "woody_part_end_pos": [8.0] * 9,
        "apple_pos": [9.0, 8.0, 7.0],
        "apple_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    mapped = map_pre_grasp_geometry(meta, primary_dir=PRIMARY_DIR)
    expected_base = surface_to_centerline((1.0, 2.0, 3.0), mapped.spur_direction, PRIMARY_DIR, 0.0125)
    np.testing.assert_allclose(mapped.fruiting_base_pos, expected_base, atol=1e-9)
    assert mapped.diagnostics.get("pre_grasp_snapshot_source") == "rest_snapshot_during_run"
    assert mapped.apple_quat_xyzw is not None
    s = math.sqrt(0.5)
    np.testing.assert_allclose(mapped.apple_quat_xyzw, (0.0, 0.0, s, s), atol=1e-6)


def test_map_pre_grasp_prefers_explicit_apple_quat_xyzw():
    meta = _synthetic_pre_grasp_meta()
    snap = meta["pre_grasp_geometry"]["snapshot"]
    snap["apple_quat_xyzw"] = [0.0, 0.0, 0.0, 1.0]
    # Conflicting 4x4 (90° Z) must lose to explicit quat.
    snap["apple_pose_4x4"] = [
        0.0,
        -1.0,
        0.0,
        0.05,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        -0.13,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    mapped = map_pre_grasp_geometry(meta, primary_dir=PRIMARY_DIR)
    np.testing.assert_allclose(mapped.apple_quat_xyzw, (0.0, 0.0, 0.0, 1.0), atol=1e-9)


def test_fruiting_params_from_pre_grasp_meta():
    params, base, diagnostics = fruiting_params_from_pre_grasp_meta(
        _synthetic_pre_grasp_meta(),
        fixture_path=VARIANCE,
    )
    spur_u = np.array([0.02, 0.0, -0.10], dtype=np.float64)
    spur_u /= np.linalg.norm(spur_u)
    expected_base = surface_to_centerline((0.0, 0.0, 0.0), spur_u, PRIMARY_DIR, 0.0125)
    assert base == expected_base
    assert params.topology == "t_junction"
    assert params.spur_surface_offset is True
    assert params.secondary is None
    assert params.primary is not None and params.spur is not None and params.stem is not None
    np.testing.assert_allclose(params.primary.direction, (1.0, 0.0, 0.0), atol=1e-6)
    stem_u = np.array([0.03, 0.0, -0.03], dtype=np.float64)
    stem_u /= np.linalg.norm(stem_u)
    np.testing.assert_allclose(params.spur.direction, spur_u, atol=1e-6)
    np.testing.assert_allclose(params.stem.direction, stem_u, atol=1e-6)
    assert params.primary.length == pytest.approx(0.2)
    assert params.spur.density == pytest.approx(1200.0)
    assert params.apple_radius == pytest.approx(0.04)
    assert params.apple_density == pytest.approx(650.0)
    assert params.apple_quat_xyzw is None
    blob = fruiting_params_to_dict(params)
    assert blob["schema"] == "fruiting_system_params_v2"
    assert blob.get("apple_quat_xyzw") is None
    assert "youngs_modulus_pa" in blob["primary"]
    assert "spur_length_rel_error" in diagnostics


def test_fruiting_params_from_pre_grasp_meta_sets_apple_quat():
    meta = _synthetic_pre_grasp_meta()
    s = math.sqrt(0.5)
    meta["pre_grasp_geometry"]["snapshot"]["apple_quat_xyzw"] = [0.0, 0.0, s, s]
    params, _base, _diag = fruiting_params_from_pre_grasp_meta(meta, fixture_path=VARIANCE)
    assert params.apple_quat_xyzw is not None
    np.testing.assert_allclose(params.apple_quat_xyzw, (0.0, 0.0, s, s), atol=1e-6)
    blob = fruiting_params_to_dict(params)
    np.testing.assert_allclose(blob["apple_quat_xyzw"], [0.0, 0.0, s, s], atol=1e-6)


def test_map_pre_grasp_ignores_parquet_fruiting_base_pos():
    meta = _synthetic_pre_grasp_meta()
    meta["fruiting_base_pos"] = [99.0, 99.0, 99.0]
    mapped = map_pre_grasp_geometry(meta, primary_dir=PRIMARY_DIR)
    assert mapped.fruiting_base_pos != (99.0, 99.0, 99.0)


def test_surface_to_centerline_perpendicular_spur():
    center = surface_to_centerline(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (1.0, 0.0, 0.0),
        0.02,
    )
    np.testing.assert_allclose(center, (0.0, 0.0, 0.02), atol=1e-9)


@pytest.mark.parametrize("parquet", [Path("robot_replay/s00-d00.parquet")])
def test_s00_d00_pre_grasp_params_smoke(parquet: Path):
    if not parquet.is_file():
        pytest.skip("missing robot_replay/s00-d00.parquet")
    meta = load_dataset_metadata(parquet)
    assert "pre_grasp_geometry" in meta
    params, base, diagnostics = fruiting_params_from_pre_grasp_parquet(
        parquet, fixture_path=VARIANCE
    )
    assert params.primary is not None
    assert all(math.isfinite(x) for x in base)
    assert "spur_chord_length_m" in diagnostics
    pre = meta["pre_grasp_geometry"]
    if isinstance(pre.get("rest_snapshot_during_run"), dict) and (
        "woody_part_start_pos" in pre["rest_snapshot_during_run"]
    ):
        assert diagnostics["pre_grasp_snapshot_source"] == "rest_snapshot_during_run"
    rest = pre.get("rest_snapshot_during_run")
    if isinstance(rest, dict) and "apple_pose_4x4" in rest:
        assert params.apple_quat_xyzw is not None
        from apple_pick_sim.system_id.real_post_grasp_plan import pose_4x4_to_pos_quat

        _pos, q = pose_4x4_to_pos_quat(rest["apple_pose_4x4"])
        assert abs(float(np.dot(params.apple_quat_xyzw, q))) > 1.0 - 1e-5
