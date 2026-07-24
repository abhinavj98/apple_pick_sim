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
)

VARIANCE = Path("apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json")


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
    mapped = map_pre_grasp_geometry(_synthetic_pre_grasp_meta())
    assert isinstance(mapped, PreGraspMappedGeometry)
    np.testing.assert_allclose(mapped.fruiting_base_pos, (0.0, 0.0, 0.0), atol=1e-9)
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
    assert "spur_length_error" in mapped.diagnostics


def test_map_pre_grasp_strict_rejects_nonzero_bend():
    meta = _synthetic_pre_grasp_meta()
    meta["pre_grasp_geometry"]["snapshot"]["woody_bending_angles"] = [0.2, 0.0, 0.0]
    with pytest.raises(ValueError, match="bend"):
        map_pre_grasp_geometry(meta, strict=True)


def test_fruiting_params_from_pre_grasp_meta():
    params, base, diagnostics = fruiting_params_from_pre_grasp_meta(
        _synthetic_pre_grasp_meta(),
        fixture_path=VARIANCE,
    )
    assert base == (0.0, 0.0, 0.0)
    assert params.topology == "t_junction"
    assert params.secondary is None
    assert params.primary is not None and params.spur is not None and params.stem is not None
    np.testing.assert_allclose(params.primary.direction, (1.0, 0.0, 0.0), atol=1e-6)
    spur_u = np.array([0.02, 0.0, -0.10], dtype=np.float64)
    spur_u /= np.linalg.norm(spur_u)
    stem_u = np.array([0.03, 0.0, -0.03], dtype=np.float64)
    stem_u /= np.linalg.norm(stem_u)
    np.testing.assert_allclose(params.spur.direction, spur_u, atol=1e-6)
    np.testing.assert_allclose(params.stem.direction, stem_u, atol=1e-6)
    assert params.primary.length == pytest.approx(0.2)
    assert params.spur.density == pytest.approx(1200.0)
    assert params.apple_radius == pytest.approx(0.04)
    assert params.apple_density == pytest.approx(650.0)
    blob = fruiting_params_to_dict(params)
    assert blob["schema"] == "fruiting_system_params_v2"
    assert "youngs_modulus_pa" in blob["primary"]
    assert "spur_length_rel_error" in diagnostics


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
