from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from apple_pick_sim.fruiting_system.params import load_ranges
from apple_pick_sim.system_id.real_to_batched_sysid import (
    SIM_JUNCTION_NAMES,
    build_episode_metadata_from_real,
    build_fruiting_params_from_real,
    flat_woody_to_dicts,
    range_midpoint,
    rod_directions_from_woody,
    split_pregrasp_and_trajectory,
)

VARIANCE = Path("apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json")


def test_flat_woody_to_dicts_order():
    start9 = np.arange(9, dtype=np.float32)
    end9 = np.arange(9, 18, dtype=np.float32)
    starts, ends = flat_woody_to_dicts(start9, end9)
    assert list(starts) == list(SIM_JUNCTION_NAMES)
    np.testing.assert_array_equal(starts["primary_spur"], start9[0:3])
    np.testing.assert_array_equal(ends["stem_apple"], end9[6:9])


def test_rod_directions_from_woody_unit():
    starts = {
        "primary_spur": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "spur_stem": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "stem_apple": np.array([1.0, 1.0, 0.0], dtype=np.float64),
    }
    ends = {
        "primary_spur": np.array([2.0, 0.0, 0.0], dtype=np.float64),
        "spur_stem": np.array([1.0, 3.0, 0.0], dtype=np.float64),
        "stem_apple": np.array([1.0, 1.0, 4.0], dtype=np.float64),
    }
    dirs = rod_directions_from_woody(starts, ends)
    np.testing.assert_allclose(dirs["primary"], (1.0, 0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(dirs["spur"], (0.0, 1.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(dirs["stem"], (0.0, 0.0, 1.0), atol=1e-6)


def test_rod_directions_zero_chord_raises():
    z = np.zeros(3, dtype=np.float64)
    with pytest.raises(ValueError, match="zero"):
        rod_directions_from_woody(
            {"primary_spur": z, "spur_stem": z, "stem_apple": z},
            {"primary_spur": z, "spur_stem": z, "stem_apple": z},
        )


def test_split_pregrasp_and_trajectory():
    step_idx = np.array([-1, 0, 1, 2], dtype=np.int32)
    pre_i, grasp_i = split_pregrasp_and_trajectory(step_idx)
    assert pre_i == 0 and grasp_i == 1


def test_split_missing_pregrasp_raises():
    with pytest.raises(ValueError, match="pre-grasp"):
        split_pregrasp_and_trajectory(np.array([0, 1, 2], dtype=np.int32))


def test_range_midpoint():
    assert range_midpoint({"min": 2.0, "max": 4.0}) == 3.0


def test_build_fruiting_params_uses_measured_geometry_and_fixture_materials():
    ranges = load_ranges(VARIANCE)
    directions = {
        "primary": (1.0, 0.0, 0.0),
        "spur": (0.0, 0.0, -1.0),
        "stem": (0.0, 0.0, -1.0),
    }
    rod_geometry = {
        "primary": {"length_m": 0.31, "radius_m": 0.021},
        "spur": {"length_m": 0.08, "radius_m": 0.009},
        "stem": {"length_m": 0.04, "radius_m": 0.0025},
    }
    params = build_fruiting_params_from_real(
        ranges_path=VARIANCE,
        rod_geometry=rod_geometry,
        directions=directions,
        apple_radius_m=0.055,
    )
    assert params.topology == "t_junction"
    assert params.spur_attach_fraction == 0.5
    assert params.secondary is None
    assert params.primary is not None and params.spur is not None and params.stem is not None
    assert params.primary.length == 0.31
    assert params.primary.radius == 0.021
    assert params.primary.num_segments == 4
    assert params.apple_radius == 0.055
    assert params.primary.density == range_midpoint(ranges["primary"]["density"])
    assert params.primary.youngs_modulus_pa == range_midpoint(
        ranges["primary"]["youngs_modulus_pa"]
    )


def _write_synthetic_real(path: Path) -> None:
    n = 3
    step_idx = [-1, 0, 1]
    rows = []
    for i in range(n):
        rows.append(
            {
                "step_idx": step_idx[i],
                "tcp_pos": [0.0, 0.9, 0.4],
                "tcp_quat": [0.0, 0.0, 0.0, 1.0],
                "apple_pos": [0.0, 0.95, 0.38],
                "apple_quat": [0.0, 0.0, 0.0, 1.0],
                "robot_joint_q": [0.1 * j for j in range(7)],
                "woody_part_start_pos": [
                    0.0,
                    1.0,
                    0.6,
                    0.0,
                    1.0,
                    0.5,
                    0.0,
                    1.0,
                    0.4,
                ],
                "woody_part_end_pos": [
                    0.0,
                    1.0,
                    0.5,
                    0.0,
                    1.0,
                    0.4,
                    0.0,
                    0.95,
                    0.38,
                ],
                "excitation_direction": [0.0, -1.0, 0.0],
            }
        )
    table = pa.Table.from_pylist(rows)
    dataset_metadata = {
        "episode_id": "synthetic-real-ep",
        "fruiting_base_pos": [0.0, 1.0, 0.6],
        "rod_geometry": {
            "primary": {"length_m": 0.31, "radius_m": 0.021},
            "spur": {"length_m": 0.08, "radius_m": 0.009},
            "stem": {"length_m": 0.04, "radius_m": 0.0025},
        },
        "apple_radius_m": 0.055,
        "source_metadata": {"robot": {"control_hz": 15.0}},
    }
    meta = {b"dataset_metadata": json.dumps(dataset_metadata).encode("utf-8")}
    pq.write_table(table.replace_schema_metadata(meta), path)


def test_build_episode_metadata_from_real(tmp_path: Path):
    path = tmp_path / "real.parquet"
    _write_synthetic_real(path)
    meta = build_episode_metadata_from_real(path, fixture_path=VARIANCE)

    assert meta["structure_idx"] == 0
    assert meta["direction_idx"] == 0
    assert meta["env_idx"] == 0
    assert meta["excitation_type"] == "quasi_static"
    assert meta["control_hz"] == 15.0
    assert meta["n_woody_parts"] == 3
    assert meta["junction_names"] == list(SIM_JUNCTION_NAMES)
    assert meta["episode_id"] == "synthetic-real-ep"
    assert meta["fruiting_base_pos"] == [0.0, 1.0, 0.6]
    assert meta["apple_radius"] == 0.055
    assert meta["rod_radii"] == {"primary": 0.021, "spur": 0.009, "stem": 0.0025}
    assert meta["fixture_path"] == str(VARIANCE.resolve())

    assert meta["initial_tcp_pos"] == [0.0, 0.9, 0.4]
    assert meta["initial_tcp_quat"] == [0.0, 0.0, 0.0, 1.0]
    assert meta["initial_apple_pos"] == [0.0, 0.95, 0.38]
    assert meta["initial_apple_quat"] == [0.0, 0.0, 0.0, 1.0]
    assert meta["initial_robot_joint_q"] == [0.1 * j for j in range(7)]
    assert meta["weld_reference_pos"] == meta["initial_apple_pos"]
    assert meta["weld_reference_quat"] == meta["initial_apple_quat"]
    n = float(np.hypot(0.05, 0.02))
    np.testing.assert_allclose(
        meta["weld_direction"],
        [0.0, -0.05 / n, 0.02 / n],
        atol=1e-6,
    )
    assert meta["pull_direction"] == [0.0, -1.0, 0.0]

    params = meta["fruiting_system_params"]
    assert params["schema"] == "fruiting_system_params_v2"
    assert params["topology"] == "t_junction"
    assert params["primary"]["length"] == 0.31
    # primary chord: (0,0,-0.1) -> (0,0,-1)
    np.testing.assert_allclose(params["primary"]["direction"], [0.0, 0.0, -1.0], atol=1e-6)
    assert "params_fingerprint" in meta
    assert isinstance(meta["params_fingerprint"], dict)
