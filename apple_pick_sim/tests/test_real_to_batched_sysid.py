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
    assert params.primary.num_segments == int(
        round(range_midpoint(ranges["primary"]["num_segments"]))
    )
    assert params.apple_radius == 0.055
    assert params.primary.density == range_midpoint(ranges["primary"]["density"])
    assert params.primary.youngs_modulus_pa == range_midpoint(
        ranges["primary"]["youngs_modulus_pa"]
    )
    assert params.spur_surface_offset is True


def test_build_fruiting_params_honors_spur_surface_offset_fixture_flag(tmp_path: Path):
    import copy
    import json

    custom = copy.deepcopy(json.loads(VARIANCE.read_text()))
    custom["spur_surface_offset"] = False
    path = tmp_path / "ranges.json"
    path.write_text(json.dumps(custom))
    params = build_fruiting_params_from_real(
        ranges_path=path,
        rod_geometry={
            "primary": {"length_m": 0.31, "radius_m": 0.021},
            "spur": {"length_m": 0.08, "radius_m": 0.009},
            "stem": {"length_m": 0.04, "radius_m": 0.0025},
        },
        directions={
            "primary": (1.0, 0.0, 0.0),
            "spur": (0.0, 0.0, -1.0),
            "stem": (0.0, 0.0, -1.0),
        },
        apple_radius_m=0.055,
    )
    assert params.spur_surface_offset is False


def _identity_pose_4x4(pos: list[float]) -> list[float]:
    x, y, z = pos
    return [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z, 0.0, 0.0, 0.0, 1.0]


def _write_synthetic_real(path: Path) -> None:
    """Minimal real-episode-shaped parquet for native pre/post → batched meta."""
    # Woody: part0 Branch→Spur, part1 Branch unused chord, part2 Spur→Apple CoM.
    spur_start = [0.0, 1.0, 0.6]
    spur_end = [0.0, 1.0, 0.5]
    apple_pos = [0.0, 0.95, 0.38]
    tcp_pos = [0.0, 0.9, 0.4]
    woody_start = spur_start + [0.0, 1.0, 0.55] + spur_end
    woody_end = spur_end + [0.0, 1.0, 0.45] + apple_pos
    joint = [0.1 * j for j in range(7)]
    rows = [
        {
            "step_idx": 0,
            "joint_pos": joint,
            "tcp_pos": tcp_pos,
            "apple_pos": apple_pos,
            "woody_part_start_pos": woody_start,
            "woody_part_end_pos": woody_end,
            "excitation_direction": [0.0, -1.0, 0.0],
        },
        {
            "step_idx": 1,
            "joint_pos": joint,
            "tcp_pos": tcp_pos,
            "apple_pos": apple_pos,
            "woody_part_start_pos": woody_start,
            "woody_part_end_pos": woody_end,
            "excitation_direction": [0.0, -1.0, 0.0],
        },
    ]
    table = pa.Table.from_pylist(rows)
    snap = {
        "woody_part_start_pos": woody_start,
        "woody_part_end_pos": woody_end,
        "woody_bending_angles": [0.0, 0.0, 0.0],
        "apple_pos": apple_pos,
        "apple_pose_4x4": _identity_pose_4x4(apple_pos),
    }
    dataset_metadata = {
        "episode_id": "synthetic-real-ep",
        "topology": {
            "junction_names": ["Branch", "Spur", "Apple"],
            "shared_endpoints": True,
            "n_woody_parts": 3,
        },
        "pre_grasp_geometry": {
            "parts": {
                "primary": {
                    "length_m": 0.31,
                    "radius_m": 0.021,
                    "density_kg_m3": 750.0,
                },
                "spur": {
                    "length_m": 0.10,
                    "radius_m": 0.009,
                    "density_kg_m3": 750.0,
                },
                "stem": {
                    "length_m": 0.065,
                    "radius_m": 0.0025,
                    "density_kg_m3": 750.0,
                },
                "apple": {
                    "length_m": 0.11,
                    "radius_m": 0.055,
                    "density_kg_m3": 500.0,
                },
            },
            "rest_snapshot_during_run": snap,
        },
        "post_grasp_geometry": {
            "tcp_pos": tcp_pos,
            "tcp_pose_4x4": _identity_pose_4x4(tcp_pos),
            "snapshot": {
                "apple_pos": apple_pos,
                "apple_pose_4x4": _identity_pose_4x4(apple_pos),
            },
        },
        "dump": {"control_hz": 15.0, "episode_id": "synthetic-real-ep"},
    }
    meta = {b"dataset_metadata": json.dumps(dataset_metadata).encode("utf-8")}
    pq.write_table(table.replace_schema_metadata(meta), path)


def test_build_episode_metadata_from_real(tmp_path: Path):
    path = tmp_path / "real.parquet"
    _write_synthetic_real(path)

    from apple_pick_sim.fruiting_system.params import fruiting_params_to_dict, params_fingerprint
    from apple_pick_sim.system_id.real_post_grasp_plan import post_grasp_plan_from_metadata
    from apple_pick_sim.system_id.real_pre_grasp_params import (
        fruiting_params_from_pre_grasp_parquet,
        load_dataset_metadata,
    )

    params, base_pos, _ = fruiting_params_from_pre_grasp_parquet(
        path, fixture_path=VARIANCE
    )
    plan = post_grasp_plan_from_metadata(
        load_dataset_metadata(path),
        apple_radius_m=float(params.apple_radius),
        emit_warnings=False,
    )
    meta = build_episode_metadata_from_real(path, fixture_path=VARIANCE)

    assert meta["structure_idx"] == 0
    assert meta["direction_idx"] == 0
    assert meta["env_idx"] == 0
    assert meta["excitation_type"] == "quasi_static"
    assert meta["control_hz"] == 15.0
    assert meta["n_woody_parts"] == 3
    assert meta["junction_names"] == list(SIM_JUNCTION_NAMES)
    assert meta["episode_id"] == "synthetic-real-ep"
    assert meta["fixture_path"] == str(VARIANCE.resolve())
    assert meta["pull_direction"] == [0.0, -1.0, 0.0]
    assert meta["fruiting_system_params"] == fruiting_params_to_dict(params)
    assert meta["params_fingerprint"] == params_fingerprint(params)
    np.testing.assert_allclose(meta["fruiting_base_pos"], list(base_pos), atol=1e-9)
    np.testing.assert_allclose(meta["initial_tcp_pos"], list(plan.tcp_pos), atol=1e-9)
    _assert_quat_close(meta["initial_tcp_quat"], plan.tcp_quat_xyzw)
    np.testing.assert_allclose(
        meta["initial_apple_pos"], list(plan.apple_pos_welded), atol=1e-9
    )
    _assert_quat_close(meta["initial_apple_quat"], plan.apple_quat_xyzw)
    assert meta["weld_reference_pos"] == meta["initial_apple_pos"]
    assert meta["weld_reference_quat"] == meta["initial_apple_quat"]
    np.testing.assert_allclose(
        meta["weld_direction"], list(plan.weld_direction), atol=1e-9
    )
    assert meta["apple_radius"] == pytest.approx(float(params.apple_radius))
    assert meta["rod_radii"] == {
        "primary": float(params.primary.radius),
        "spur": float(params.spur.radius),
        "stem": float(params.stem.radius),
    }
    assert meta["initial_robot_joint_q"] == [0.1 * j for j in range(7)]
    assert meta["fruiting_system_params"]["schema"] == "fruiting_system_params_v2"
    assert meta["fruiting_system_params"]["topology"] == "t_junction"


def test_export_real_to_batched_dataset_loads(tmp_path: Path):
    """Exported dataset must load as batched_sysid_v1 with non-zero actions."""
    from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        export_real_episode_to_batched_dataset,
    )

    src = Path("robot_replay/s02-d00_action.parquet")
    if not src.is_file():
        pytest.skip("missing robot_replay/s02-d00_action.parquet")

    out = tmp_path / "batched_real"
    export_real_episode_to_batched_dataset(
        src, fixture_path=VARIANCE, output_dir=out, overwrite=True
    )
    ds = BatchedSysIdDataset(out)
    assert ds.manifest["schema_version"] == "batched_sysid_v1"
    assert len(ds.episode_entries()) == 1
    meta = ds.load_episode_metadata(0, 0)
    assert "fruiting_system_params" in meta
    assert isinstance(meta["fruiting_system_params"], str)
    assert meta["junction_names"] == list(SIM_JUNCTION_NAMES)
    from apple_pick_sim.system_id.batched_digital_twin_init import true_params_for_structure

    true_params = true_params_for_structure(ds, 0)
    assert true_params.apple_radius is not None
    arrays = ds.load_episode_obs_arrays(0, 0)
    assert arrays["action"].shape[1] == 6
    assert float(np.linalg.norm(arrays["action"], axis=1).max()) > 0.0
    assert arrays["tcp_pos"].shape[1] == 3
    assert arrays["tcp_pos"].shape[0] == arrays["action"].shape[0]
    assert ds.manifest["collection"]["seed"] == 0


def test_export_refuses_all_zero_action(tmp_path: Path):
    """Zero-action real episodes fail loud unless allow_zero_action / drive_fill."""
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        export_real_episode_to_batched_dataset,
    )

    src = Path("robot_replay/s00-d00.parquet")
    if not src.is_file():
        pytest.skip("missing robot_replay/s00-d00.parquet")

    out = tmp_path / "batched_zero"
    with pytest.raises(ValueError, match="real-replay-action-zero"):
        export_real_episode_to_batched_dataset(
            src, fixture_path=VARIANCE, output_dir=out, overwrite=True
        )


def _assert_quat_close(got, expected, *, atol: float = 1e-9) -> None:
    g = np.asarray(got, dtype=np.float64).reshape(4)
    e = np.asarray(expected, dtype=np.float64).reshape(4)
    assert abs(float(np.dot(g, e))) >= 1.0 - atol


@pytest.mark.parametrize("parquet", [Path("robot_replay/s00-d00.parquet")])
def test_s00_d00_convert_matches_native_pre_post(parquet: Path):
    """Converted metadata must match settle-viewer native pre/post builders."""
    if not parquet.is_file():
        pytest.skip(f"missing {parquet}")

    from apple_pick_sim.fruiting_system.params import fruiting_params_to_dict, params_fingerprint
    from apple_pick_sim.system_id.real_post_grasp_plan import post_grasp_plan_from_metadata
    from apple_pick_sim.system_id.real_pre_grasp_params import (
        fruiting_params_from_pre_grasp_parquet,
        load_dataset_metadata,
    )

    params, base_pos, _ = fruiting_params_from_pre_grasp_parquet(
        parquet, fixture_path=VARIANCE
    )
    dm = load_dataset_metadata(parquet)
    plan = post_grasp_plan_from_metadata(
        dm,
        apple_radius_m=float(params.apple_radius),
        emit_warnings=False,
    )

    meta = build_episode_metadata_from_real(parquet, fixture_path=VARIANCE)

    assert meta["fruiting_system_params"] == fruiting_params_to_dict(params)
    assert meta["params_fingerprint"] == params_fingerprint(params)
    np.testing.assert_allclose(meta["fruiting_base_pos"], list(base_pos), atol=1e-9)
    np.testing.assert_allclose(meta["initial_tcp_pos"], list(plan.tcp_pos), atol=1e-9)
    _assert_quat_close(meta["initial_tcp_quat"], plan.tcp_quat_xyzw)
    np.testing.assert_allclose(
        meta["initial_apple_pos"], list(plan.apple_pos_welded), atol=1e-9
    )
    _assert_quat_close(meta["initial_apple_quat"], plan.apple_quat_xyzw)
    np.testing.assert_allclose(
        meta["weld_reference_pos"], list(plan.apple_pos_welded), atol=1e-9
    )
    _assert_quat_close(meta["weld_reference_quat"], plan.apple_quat_xyzw)
    np.testing.assert_allclose(
        meta["weld_direction"], list(plan.weld_direction), atol=1e-9
    )
    assert meta["apple_radius"] == pytest.approx(float(params.apple_radius))
    assert meta["rod_radii"] == {
        "primary": float(params.primary.radius),
        "spur": float(params.spur.radius),
        "stem": float(params.stem.radius),
    }

    table = pq.read_table(parquet)
    joint = table.column("joint_pos")[0].as_py()
    np.testing.assert_allclose(meta["initial_robot_joint_q"], joint, atol=1e-9)
    assert isinstance(meta["control_hz"], float)
    assert meta["control_hz"] > 0.0
    assert str(meta["episode_id"]).strip() != ""
