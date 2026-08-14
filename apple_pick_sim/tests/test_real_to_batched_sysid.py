from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from apple_pick_sim.fruiting_system.params import load_ranges
from apple_pick_sim.system_id.mmd_features import CMA_WOODY_JUNCTIONS
from apple_pick_sim.system_id.real_to_batched_sysid import (
    SIM_JUNCTION_NAMES,
    build_episode_metadata_from_real,
    build_fruiting_params_from_real,
    flat_woody_to_dicts,
    range_midpoint,
    rod_directions_from_woody,
    split_pregrasp_and_trajectory,
    tag_poses_to_cma_woody,
)

VARIANCE = Path("apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json")


def test_tag_poses_to_cma_woody_maps_translations():
    branch = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    spur = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    apple = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    starts, apple_pos = tag_poses_to_cma_woody(
        _identity_pose_4x4(branch.tolist()),
        _identity_pose_4x4(spur.tolist()),
        _identity_pose_4x4(apple.tolist()),
    )
    assert set(starts) == {"primary_spur", "spur_stem"}
    np.testing.assert_allclose(starts["primary_spur"], branch)
    np.testing.assert_allclose(starts["spur_stem"], spur)
    np.testing.assert_allclose(apple_pos, apple)


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


def test_build_fruiting_params_applies_vbd_stretch_force_not_beam_ea_over_l():
    """Real converter must honor fixture vbd_stretch_force (same as sample_params)."""
    from apple_pick_sim.fruiting_system.params import stretch_knobs_from_max_force

    ranges = load_ranges(VARIANCE)
    directions = {
        "primary": (1.0, 0.0, 0.0),
        "spur": (0.0, 0.0, -1.0),
        "stem": (0.0, 0.0, -1.0),
    }
    # Short measured stem (real proxy scale) makes beam EA/L diverge sharply from F_max budget.
    rod_geometry = {
        "primary": {"length_m": 0.827, "radius_m": 0.0125, "density_kg_m3": 660.0},
        "spur": {"length_m": 0.12, "radius_m": 0.0025, "density_kg_m3": 1200.0},
        "stem": {"length_m": 0.005, "radius_m": 0.0005, "density_kg_m3": 1000.0},
    }
    params = build_fruiting_params_from_real(
        ranges_path=VARIANCE,
        rod_geometry=rod_geometry,
        directions=directions,
        apple_radius_m=0.04,
        apple_density_kg_m3=650.0,
        use_parts_density=True,
    )
    assert params.primary is not None and params.spur is not None and params.stem is not None
    for name, rod in (
        ("primary", params.primary),
        ("spur", params.spur),
        ("stem", params.stem),
    ):
        force = ranges[name]["vbd_stretch_force"]
        k_exp, c_exp = stretch_knobs_from_max_force(
            float(force["max_force_n"]),
            float(force["damping_ratio"]),
            float(rod.length),
            float(rod.radius),
            float(rod.density),
            int(rod.num_segments),
        )
        assert rod.stretch_stiffness == pytest.approx(k_exp, rel=1e-9)
        assert rod.stretch_damping == pytest.approx(c_exp, rel=1e-9)
        # Sanity: must not silently fall back to beam EA/L on this geometry.
        import math

        a = math.pi * float(rod.radius) ** 2
        l_seg = float(rod.length) / int(rod.num_segments)
        k_beam = float(rod.youngs_modulus_pa) * a / l_seg
        assert abs(rod.stretch_stiffness - k_beam) / k_beam > 0.1


def test_build_fruiting_params_honors_spur_attach_fraction_from_fixture(tmp_path: Path):
    import copy
    import json

    custom = copy.deepcopy(json.loads(VARIANCE.read_text()))
    # Canonical key matches sample_params / _spur_attach_fraction_from_ranges (top-level).
    custom["spur_attach_fraction"] = 0.4
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
    assert params.spur_attach_fraction == pytest.approx(0.4)


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


_TAG_POSE_COLUMNS = ("branch_pose_4x4", "spur_pose_4x4", "apple_pose_4x4")


def _skip_unless_tag_poses(src: Path) -> None:
    names = set(pq.read_schema(src).names)
    missing = [c for c in _TAG_POSE_COLUMNS if c not in names]
    if missing:
        pytest.skip(f"{src} missing tag pose columns {missing}")

_WRENCH_EXPORT_KW = dict(
    action=[1.0, -2.0, -3.0, -0.5, 0.1, -0.2],
    action_semantics=(
        "per-frame pose-control wrench [Fx, Fy, Fz, Tx, Ty, Tz] "
        "computed from the current pose error and velocity"
    ),
    action_order=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
    target_pose_4x4=_identity_pose_4x4([0.1, 0.2, 0.3]),
    controller_gains={
        "task_prop_gains": [100.0, 100.0, 100.0, 30.0, 30.0, 30.0],
        "task_deriv_gains": [17.5, 17.5, 17.5, 9.5, 9.5, 9.5],
    },
)


def _write_synthetic_real(
    path: Path,
    *,
    action: list[float] | None = None,
    action_semantics: str | None = None,
    action_order: list[str] | None = None,
    target_pose_4x4: list[float] | None = None,
    controller_gains: dict | None = None,
    camera_to_base_4x4: list[list[float]] | None = None,
    tcp_pose_4x4: list[float] | None = None,
    ft_wrist: list[float] | None = None,
    ft_wrist_raw: list[float] | None = None,
    skip_tcp_pose_4x4: bool = False,
    hold_index: int | None = None,
    hold_number: list[float] | None = None,
    include_tag_poses: bool = True,
    tag_pose_columns: tuple[str, ...] | None = None,
    include_woody_part_columns: bool = False,
    packed_woody_start: list[float] | None = None,
    packed_woody_end: list[float] | None = None,
) -> None:
    """Minimal real-episode-shaped parquet for native pre/post → batched meta."""
    branch = [0.0, 1.0, 0.6]
    spur = [0.0, 1.0, 0.5]
    apple_pos = [0.0, 0.95, 0.38]
    tcp_pos = [0.0, 0.9, 0.4]
    snap_woody_start = branch + branch + spur
    snap_woody_end = spur + apple_pos + apple_pos
    woody_start = packed_woody_start if packed_woody_start is not None else snap_woody_start
    woody_end = packed_woody_end if packed_woody_end is not None else snap_woody_end
    joint = [0.1 * j for j in range(7)]
    base_row: dict = {
        "step_idx": 0,
        "joint_pos": joint,
        "tcp_pos": tcp_pos,
        "apple_pos": apple_pos,
        "excitation_direction": [0.0, -1.0, 0.0],
        "tcp_velocity": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "ft_wrist": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    if include_woody_part_columns:
        base_row["woody_part_start_pos"] = list(woody_start)
        base_row["woody_part_end_pos"] = list(woody_end)
    if include_tag_poses:
        pose_cols = tag_pose_columns if tag_pose_columns is not None else _TAG_POSE_COLUMNS
        if "branch_pose_4x4" in pose_cols:
            base_row["branch_pose_4x4"] = _identity_pose_4x4(branch)
        if "spur_pose_4x4" in pose_cols:
            base_row["spur_pose_4x4"] = _identity_pose_4x4(spur)
        if "apple_pose_4x4" in pose_cols:
            base_row["apple_pose_4x4"] = _identity_pose_4x4(apple_pos)
    if action is not None:
        base_row["action"] = list(action)
    if target_pose_4x4 is not None:
        base_row["target_pose_4x4"] = list(target_pose_4x4)
    if tcp_pose_4x4 is not None:
        base_row["tcp_pose_4x4"] = list(tcp_pose_4x4)
    elif target_pose_4x4 is not None and not skip_tcp_pose_4x4:
        base_row["tcp_pose_4x4"] = _identity_pose_4x4(tcp_pos)
    if ft_wrist is not None:
        base_row["ft_wrist"] = list(ft_wrist)
    if ft_wrist_raw is not None:
        base_row["ft_wrist_raw"] = list(ft_wrist_raw)
    if hold_index is not None:
        base_row["hold_index"] = int(hold_index)
    if hold_number is not None:
        base_row["hold_number"] = list(hold_number)
    rows = [dict(base_row), {**base_row, "step_idx": 1}]
    table = pa.Table.from_pylist(rows)
    snap = {
        "woody_part_start_pos": snap_woody_start,
        "woody_part_end_pos": snap_woody_end,
        "woody_bending_angles": [0.0, 0.0, 0.0],
        "apple_pos": apple_pos,
        "apple_pose_4x4": _identity_pose_4x4(apple_pos),
    }
    dump: dict = {"control_hz": 15.0, "episode_id": "synthetic-real-ep"}
    if action_semantics is not None:
        dump["action_semantics"] = action_semantics
    if controller_gains is not None:
        dump["controller_gains"] = dict(controller_gains)
    dataset_metadata: dict = {
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
        "dump": dump,
    }
    if action_order is not None:
        dataset_metadata["field_layout"] = {
            "action": {
                "dim": 6,
                "order": list(action_order),
                "description": action_semantics or "action",
            }
        }
    if camera_to_base_4x4 is not None:
        dataset_metadata["camera_to_base_4x4_used"] = camera_to_base_4x4
    meta = {b"dataset_metadata": json.dumps(dataset_metadata).encode("utf-8")}
    pq.write_table(table.replace_schema_metadata(meta), path)


def test_camera_to_base_4x4_from_dataset_metadata_prefers_used():
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        camera_to_base_4x4_from_dataset_metadata,
    )

    used = [
        [1.0, 0.0, 0.0, 0.1],
        [0.0, 1.0, 0.0, 0.2],
        [0.0, 0.0, 1.0, 0.3],
        [0.0, 0.0, 0.0, 1.0],
    ]
    snap = [
        [1.0, 0.0, 0.0, 9.0],
        [0.0, 1.0, 0.0, 9.0],
        [0.0, 0.0, 1.0, 9.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    got = camera_to_base_4x4_from_dataset_metadata(
        {
            "camera_to_base_4x4_used": used,
            "pre_grasp_geometry": {
                "settled_snapshot": {"camera_to_base_4x4": snap},
            },
        }
    )
    assert got == used


def test_camera_to_base_4x4_from_dataset_metadata_falls_back_to_snapshot():
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        camera_to_base_4x4_from_dataset_metadata,
    )

    snap = [
        [1.0, 0.0, 0.0, 9.0],
        [0.0, 1.0, 0.0, 9.0],
        [0.0, 0.0, 1.0, 9.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    got = camera_to_base_4x4_from_dataset_metadata(
        {"pre_grasp_geometry": {"settled_snapshot": {"camera_to_base_4x4": snap}}}
    )
    assert got == snap


def test_build_episode_metadata_copies_camera_to_base_4x4(tmp_path: Path):
    path = tmp_path / "real.parquet"
    T = [
        [1.0, 0.0, 0.0, -0.3],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 0.0, 1.0, 0.4],
        [0.0, 0.0, 0.0, 1.0],
    ]
    _write_synthetic_real(path, camera_to_base_4x4=T)
    meta = build_episode_metadata_from_real(path, fixture_path=VARIANCE)
    assert meta["camera_to_base_4x4"] == T


def test_export_persists_camera_to_base_4x4_in_episode_parquet(tmp_path: Path):
    """EPISODE_METADATA_KEYS must allowlist camera_to_base_4x4 or save drops it."""
    from apple_pick_sim.system_id import BatchedSysIdDataset
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        export_real_episode_to_batched_dataset,
    )

    path = tmp_path / "real.parquet"
    T = [
        [1.0, 0.0, 0.0, -0.3],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 0.0, 1.0, 0.4],
        [0.0, 0.0, 0.0, 1.0],
    ]
    _write_synthetic_real(
        path,
        camera_to_base_4x4=T,
        action=[0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        action_semantics="EE twist",
        action_order=["vx", "vy", "vz", "wx", "wy", "wz"],
    )
    out = tmp_path / "batched"
    export_real_episode_to_batched_dataset(
        path, fixture_path=VARIANCE, output_dir=out, overwrite=True
    )
    loaded = BatchedSysIdDataset(out).load_episode_metadata(0, 0)
    assert loaded.get("camera_to_base_4x4") == T


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
    assert meta["n_woody_parts"] == 2
    assert meta["junction_names"] == list(CMA_WOODY_JUNCTIONS)
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


def test_scalar_hold_number_prefers_hold_index_over_onehot():
    from apple_pick_sim.system_id.real_to_batched_sysid import _scalar_hold_number

    assert _scalar_hold_number([0.0, 0.0, 1.0, 0.0], hold_index=2) == 2
    assert _scalar_hold_number([0.0, 1.0, 0.0, 0.0], hold_index=None) == 1
    assert _scalar_hold_number(None, hold_index=0) == 0
    assert _scalar_hold_number(None, hold_index=None) == -1


def test_export_hold_number_is_scalar(tmp_path: Path):
    from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        export_real_episode_to_batched_dataset,
    )

    src = tmp_path / "hold_scalar.parquet"
    _write_synthetic_real(
        src,
        hold_index=1,
        hold_number=[0.0, 1.0, 0.0, 0.0],
        action=[0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        action_semantics="EE twist",
        action_order=["vx", "vy", "vz", "wx", "wy", "wz"],
    )
    out = tmp_path / "batched_hold"
    export_real_episode_to_batched_dataset(
        src, fixture_path=VARIANCE, output_dir=out, overwrite=True
    )
    arrays = BatchedSysIdDataset(out).load_episode_obs_arrays(0, 0)
    hn = arrays["hold_number"]
    assert hn.dtype == np.int32
    assert hn.shape == (2,)
    np.testing.assert_array_equal(hn, np.array([1, 1], dtype=np.int32))


def test_export_writes_two_woody_starts_and_no_ends(tmp_path: Path):
    from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        export_real_episode_to_batched_dataset,
    )
    from apple_pick_sim.system_id.trajectory_store import woody_end_column, woody_start_column

    target = _identity_pose_4x4([0.1, 0.2, 0.3])
    kp = [100.0, 100.0, 100.0, 30.0, 30.0, 30.0]
    kd = [17.5, 17.5, 17.5, 9.5, 9.5, 9.5]
    src = tmp_path / "woody_export.parquet"
    _write_synthetic_real(
        src,
        action=[1.0, -2.0, -3.0, -0.5, 0.1, -0.2],
        action_semantics=(
            "per-frame pose-control wrench [Fx, Fy, Fz, Tx, Ty, Tz] "
            "computed from the current pose error and velocity"
        ),
        action_order=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        target_pose_4x4=target,
        controller_gains={"task_prop_gains": kp, "task_deriv_gains": kd},
    )
    out = tmp_path / "batched_woody"
    export_real_episode_to_batched_dataset(
        src, fixture_path=VARIANCE, output_dir=out, overwrite=True
    )
    ds = BatchedSysIdDataset(out)
    meta = ds.load_episode_metadata(0, 0)
    assert meta["junction_names"] == list(CMA_WOODY_JUNCTIONS)
    assert meta["n_woody_parts"] == 2

    table = ds.load_episode_frames(0, 0)
    col_names = set(table.column_names)
    assert woody_start_column("primary_spur") in col_names
    assert woody_start_column("spur_stem") in col_names
    assert woody_end_column("primary_spur") not in col_names
    assert woody_end_column("spur_stem") not in col_names
    assert woody_start_column("stem_apple") not in col_names
    assert woody_start_column("support") not in col_names

    arrays = ds.load_episode_obs_arrays(0, 0)
    assert set(arrays["woody_part_start_pos"]) == set(CMA_WOODY_JUNCTIONS)
    assert arrays["woody_part_end_pos"] == {}
    np.testing.assert_allclose(
        arrays["apple_pos"][0], [0.0, 0.95, 0.38], atol=1e-6
    )
    np.testing.assert_allclose(
        arrays["woody_part_start_pos"]["primary_spur"][0], [0.0, 1.0, 0.6], atol=1e-6
    )
    np.testing.assert_allclose(
        arrays["woody_part_start_pos"]["spur_stem"][0], [0.0, 1.0, 0.5], atol=1e-6
    )


def _export_wrench(tmp_path: Path, name: str, **extra):
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        export_real_episode_to_batched_dataset,
    )

    src = tmp_path / f"{name}.parquet"
    _write_synthetic_real(src, **_WRENCH_EXPORT_KW, **extra)
    out = tmp_path / f"{name}_out"
    export_real_episode_to_batched_dataset(
        src, fixture_path=VARIANCE, output_dir=out, overwrite=True
    )
    return src, out


def test_export_tag_poses_without_woody_part_columns(tmp_path: Path):
    from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset

    _src, out = _export_wrench(tmp_path, "poses_only")
    arrays = BatchedSysIdDataset(out).load_episode_obs_arrays(0, 0)
    np.testing.assert_allclose(
        arrays["woody_part_start_pos"]["primary_spur"][0], [0.0, 1.0, 0.6], atol=1e-6
    )
    np.testing.assert_allclose(
        arrays["woody_part_start_pos"]["spur_stem"][0], [0.0, 1.0, 0.5], atol=1e-6
    )
    np.testing.assert_allclose(arrays["apple_pos"][0], [0.0, 0.95, 0.38], atol=1e-6)


def test_export_ignores_woody_part_columns_when_tag_poses_present(tmp_path: Path):
    from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset

    _src, out = _export_wrench(
        tmp_path,
        "ignore_pack",
        include_woody_part_columns=True,
        packed_woody_start=[9.0] * 9,
        packed_woody_end=[8.0] * 9,
    )
    arrays = BatchedSysIdDataset(out).load_episode_obs_arrays(0, 0)
    np.testing.assert_allclose(
        arrays["woody_part_start_pos"]["primary_spur"][0], [0.0, 1.0, 0.6], atol=1e-6
    )
    np.testing.assert_allclose(
        arrays["woody_part_start_pos"]["spur_stem"][0], [0.0, 1.0, 0.5], atol=1e-6
    )
    np.testing.assert_allclose(arrays["apple_pos"][0], [0.0, 0.95, 0.38], atol=1e-6)


def test_export_missing_tag_poses_raises(tmp_path: Path):
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        export_real_episode_to_batched_dataset,
    )

    src = tmp_path / "no_poses.parquet"
    _write_synthetic_real(
        src,
        **_WRENCH_EXPORT_KW,
        include_tag_poses=False,
        include_woody_part_columns=True,
    )
    with pytest.raises(ValueError, match="branch_pose_4x4"):
        export_real_episode_to_batched_dataset(
            src, fixture_path=VARIANCE, output_dir=tmp_path / "out", overwrite=True
        )


def test_export_partial_tag_poses_raises(tmp_path: Path):
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        export_real_episode_to_batched_dataset,
    )

    src = tmp_path / "partial_poses.parquet"
    _write_synthetic_real(
        src,
        **_WRENCH_EXPORT_KW,
        tag_pose_columns=("spur_pose_4x4",),
    )
    with pytest.raises(ValueError, match="branch_pose_4x4"):
        export_real_episode_to_batched_dataset(
            src, fixture_path=VARIANCE, output_dir=tmp_path / "out", overwrite=True
        )


def test_export_s01_d01_spur_stem_differs_from_primary_spur(tmp_path: Path):
    """Real s01: skip unless tag poses exist; spur and branch translations must differ."""
    from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        export_real_episode_to_batched_dataset,
    )

    src = Path("robot_replay/new_data/s01-d01.parquet")
    if not src.is_file():
        pytest.skip("missing robot_replay/new_data/s01-d01.parquet")
    _skip_unless_tag_poses(src)

    out = tmp_path / "batched_s01"
    export_real_episode_to_batched_dataset(
        src, fixture_path=VARIANCE, output_dir=out, overwrite=True
    )
    arrays = BatchedSysIdDataset(out).load_episode_obs_arrays(0, 0)
    primary = arrays["woody_part_start_pos"]["primary_spur"]
    spur = arrays["woody_part_start_pos"]["spur_stem"]
    assert float(np.max(np.linalg.norm(primary - spur, axis=1))) > 0.05


def test_export_real_to_batched_dataset_loads(tmp_path: Path):
    """Exported wrench-semantics dataset packs 19D vic_pose actions."""
    from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        export_real_episode_to_batched_dataset,
    )

    src = Path("robot_replay/s02-d00.parquet")
    if not src.is_file():
        pytest.skip("missing robot_replay/s02-d00.parquet")
    _skip_unless_tag_poses(src)

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
    assert meta["junction_names"] == list(CMA_WOODY_JUNCTIONS)
    assert meta.get("action_dim") == 19
    assert meta.get("action_layout") == "vic_pose_v1"
    from apple_pick_sim.system_id.batched_digital_twin_init import true_params_for_structure

    true_params = true_params_for_structure(ds, 0)
    assert true_params.apple_radius is not None
    arrays = ds.load_episode_obs_arrays(0, 0)
    assert arrays["action"].shape[1] == 19
    assert float(np.linalg.norm(arrays["action"][:, :3], axis=1).max()) > 0.0
    assert arrays["tcp_pos"].shape[1] == 3
    assert arrays["tcp_pos"].shape[0] == arrays["action"].shape[0]
    assert ds.manifest["collection"]["seed"] == 0
    assert ds.manifest["collection"].get("action_dim") == 19


def test_export_s00_packs_vic_pose_from_target_pose(tmp_path: Path):
    """Pose-wrench logs without a usable ``action`` column still export via target_pose_4x4."""
    from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        export_real_episode_to_batched_dataset,
    )

    src = Path("robot_replay/s00-d00.parquet")
    if not src.is_file():
        pytest.skip("missing robot_replay/s00-d00.parquet")
    _skip_unless_tag_poses(src)

    out = tmp_path / "batched_s00_vic_pose"
    export_real_episode_to_batched_dataset(
        src, fixture_path=VARIANCE, output_dir=out, overwrite=True
    )
    ds = BatchedSysIdDataset(out)
    meta = ds.load_episode_metadata(0, 0)
    assert meta.get("action_layout") == "vic_pose_v1"
    assert meta.get("action_dim") == 19
    arrays = ds.load_episode_obs_arrays(0, 0)
    assert arrays["action"].shape[1] == 19
    assert float(np.linalg.norm(arrays["action"][:, :3], axis=1).max()) > 0.0


def test_detects_pose_control_wrench_action_semantics():
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        is_pose_control_wrench_semantics,
        real_action_semantics_label,
    )

    dm = {
        "dump": {
            "action_semantics": (
                "per-frame pose-control wrench [Fx, Fy, Fz, Tx, Ty, Tz] "
                "computed from the current pose error and velocity"
            )
        },
        "field_layout": {
            "action": {
                "order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                "description": "per-frame pose-control wrench",
            }
        },
    }
    label = real_action_semantics_label(dm)
    assert label is not None
    assert "wrench" in label.lower()
    assert is_pose_control_wrench_semantics(dm) is True
    assert is_pose_control_wrench_semantics({"dump": {"action_semantics": "EE twist"}}) is False
    assert is_pose_control_wrench_semantics({}) is False


def test_real_pose_control_gains_reads_task_gains():
    from apple_pick_sim.system_id.real_to_batched_sysid import real_pose_control_gains

    kp, kd = real_pose_control_gains(
        {
            "dump": {
                "controller_gains": {
                    "task_prop_gains": [100.0, 100.0, 100.0, 30.0, 30.0, 30.0],
                    "task_deriv_gains": [17.5, 17.5, 17.5, 9.5, 9.5, 9.5],
                }
            }
        }
    )
    assert kp == [100.0, 100.0, 100.0, 30.0, 30.0, 30.0]
    assert kd == [17.5, 17.5, 17.5, 9.5, 9.5, 9.5]


def test_real_pose_control_gains_raises_when_missing():
    from apple_pick_sim.system_id.real_to_batched_sysid import real_pose_control_gains

    with pytest.raises(ValueError, match="controller_gains"):
        real_pose_control_gains({"dump": {}})


def test_real_pose_control_gains_raises_on_wrong_length():
    from apple_pick_sim.system_id.real_to_batched_sysid import real_pose_control_gains

    with pytest.raises(ValueError, match="task_prop_gains"):
        real_pose_control_gains(
            {
                "dump": {
                    "controller_gains": {
                        "task_prop_gains": [1.0, 2.0, 3.0],
                        "task_deriv_gains": [1.0] * 6,
                    }
                }
            }
        )


def test_export_packs_wrench_semantics_into_19d_vic_pose_action(tmp_path: Path):
    """Wrench-semantics real action is packed as 19D pose+gains, not copied as twist."""
    from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        export_real_episode_to_batched_dataset,
    )

    target = _identity_pose_4x4([0.1, 0.2, 0.3])
    kp = [100.0, 100.0, 100.0, 30.0, 30.0, 30.0]
    kd = [17.5, 17.5, 17.5, 9.5, 9.5, 9.5]
    src = tmp_path / "wrench_real.parquet"
    _write_synthetic_real(
        src,
        action=[1.0, -2.0, -3.0, -0.5, 0.1, -0.2],
        action_semantics=(
            "per-frame pose-control wrench [Fx, Fy, Fz, Tx, Ty, Tz] "
            "computed from the current pose error and velocity"
        ),
        action_order=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        target_pose_4x4=target,
        controller_gains={"task_prop_gains": kp, "task_deriv_gains": kd},
    )
    out = tmp_path / "batched_wrench"
    export_real_episode_to_batched_dataset(
        src, fixture_path=VARIANCE, output_dir=out, overwrite=True
    )
    ds = BatchedSysIdDataset(out)
    meta = ds.load_episode_metadata(0, 0)
    assert meta.get("action_dim") == 19
    assert meta.get("action_layout") == "vic_pose_v1"
    assert meta.get("action_compatible_with_vic_twist") is False
    arrays = ds.load_episode_obs_arrays(0, 0)
    assert arrays["action"].shape == (2, 19)
    np.testing.assert_allclose(arrays["action"][0, 0:3], [0.1, 0.2, 0.3], atol=1e-5)
    # Action contract is wxyz; identity rotation → (1,0,0,0).
    np.testing.assert_allclose(arrays["action"][0, 3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(arrays["action"][0, 7:13], kp, atol=1e-5)
    np.testing.assert_allclose(arrays["action"][0, 13:19], kd, atol=1e-5)
    # Must not leave the raw wrench in the action column.
    assert not np.allclose(arrays["action"][0, :6], [1.0, -2.0, -3.0, -0.5, 0.1, -0.2])


def test_export_raises_when_target_pose_4x4_missing_for_wrench_semantics(tmp_path: Path):
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        export_real_episode_to_batched_dataset,
    )

    src = tmp_path / "wrench_no_pose.parquet"
    _write_synthetic_real(
        src,
        action=[1.0, -2.0, -3.0, -0.5, 0.1, -0.2],
        action_semantics="per-frame pose-control wrench [Fx, Fy, Fz, Tx, Ty, Tz]",
        action_order=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        controller_gains={
            "task_prop_gains": [100.0] * 6,
            "task_deriv_gains": [10.0] * 6,
        },
    )
    with pytest.raises(ValueError, match="target_pose_4x4"):
        export_real_episode_to_batched_dataset(
            src, fixture_path=VARIANCE, output_dir=tmp_path / "out", overwrite=True
        )


def test_export_raises_when_controller_gains_missing_for_wrench_semantics(tmp_path: Path):
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        export_real_episode_to_batched_dataset,
    )

    src = tmp_path / "wrench_no_gains.parquet"
    _write_synthetic_real(
        src,
        action=[1.0, -2.0, -3.0, -0.5, 0.1, -0.2],
        action_semantics="per-frame pose-control wrench [Fx, Fy, Fz, Tx, Ty, Tz]",
        action_order=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        target_pose_4x4=_identity_pose_4x4([0.1, 0.2, 0.3]),
    )
    with pytest.raises(ValueError, match="controller_gains"):
        export_real_episode_to_batched_dataset(
            src, fixture_path=VARIANCE, output_dir=tmp_path / "out", overwrite=True
        )


def test_load_episode_obs_arrays_reads_19d_action_column(tmp_path: Path):
    from apple_pick_sim.system_id.batched_trajectory_store import (
        BatchedEpisodeWriter,
        BatchedSysIdDataset,
        write_manifest,
    )

    ep = tmp_path / "episodes" / "s00_d00.parquet"
    writer = BatchedEpisodeWriter(episode_id="wide-action")
    action19 = np.arange(19, dtype=np.float32)
    obs = {
        "excitation_type": 0,
        "excitation_direction": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "tcp_velocity": np.zeros(6, dtype=np.float32),
        "ft_wrist": np.zeros(6, dtype=np.float32),
        "raw_ft_wrist": np.zeros(6, dtype=np.float32),
        "tcp_pos": np.zeros(3, dtype=np.float32),
        "apple_pos": np.zeros(3, dtype=np.float32),
        "tcp_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "apple_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot_joint_q": np.zeros(7, dtype=np.float32),
        "woody_part_start_pos": {
            "primary_spur": np.zeros(3, dtype=np.float32),
            "spur_stem": np.zeros(3, dtype=np.float32),
            "stem_apple": np.zeros(3, dtype=np.float32),
        },
        "woody_part_end_pos": {
            "primary_spur": np.ones(3, dtype=np.float32),
            "spur_stem": np.ones(3, dtype=np.float32),
            "stem_apple": np.ones(3, dtype=np.float32),
        },
        "woody_part_force": np.zeros(0, dtype=np.float32),
    }
    writer.record_step(
        step_idx=0,
        sim_time=0.0,
        phase="move_out",
        amplitude_m=0.0,
        action=action19,
        obs=obs,
    )
    writer.save(
        ep,
        {
            "schema_version": "batched_sysid_v1",
            "episode_id": "wide-action",
            "structure_idx": 0,
            "direction_idx": 0,
            "env_idx": 0,
            "junction_names": list(SIM_JUNCTION_NAMES),
            "n_woody_parts": 3,
            "action_dim": 19,
            "action_layout": "vic_pose_v1",
        },
    )
    write_manifest(
        tmp_path,
        command_argv=["test"],
        collection={"seed": 0, "num_structures": 1, "num_directions": 1},
        structures=[{"structure_idx": 0, "junction_names": list(SIM_JUNCTION_NAMES)}],
        episodes=[
            {
                "structure_idx": 0,
                "direction_idx": 0,
                "env_idx": 0,
                "filename": "episodes/s00_d00.parquet",
                "episode_id": "wide-action",
                "n_frames": 1,
                "excluded": False,
            }
        ],
        overwrite=True,
    )
    arrays = BatchedSysIdDataset(tmp_path).load_episode_obs_arrays(0, 0)
    assert arrays["action"].shape == (1, 19)
    np.testing.assert_allclose(arrays["action"][0], action19, atol=1e-6)


def test_world_wrench_from_ee_logged_rotates_force_and_torque():
    from apple_pick_sim.system_id.real_to_batched_sysid import world_wrench_from_ee_logged

    # 90° about Z: e1 -> e2
    pose = [
        0.0, -1.0, 0.0, 0.0,
        1.0,  0.0, 0.0, 0.0,
        0.0,  0.0, 1.0, 0.0,
        0.0,  0.0, 0.0, 1.0,
    ]
    ft_ee = np.array([1.0, 0.0, 0.0, 0.0, 2.0, 0.0], dtype=np.float32)
    got = world_wrench_from_ee_logged(ft_ee, pose)
    # R @ e1 = e2; R @ (2 e2) = -2 e1. Do not expect τ → e3.
    np.testing.assert_allclose(got[:3], [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(got[3:], [-2.0, 0.0, 0.0], atol=1e-6)


def test_world_wrench_from_ee_logged_does_not_negate():
    from apple_pick_sim.system_id.real_to_batched_sysid import world_wrench_from_ee_logged

    pose = _identity_pose_4x4([0.1, 0.2, 0.3])
    ft_ee = np.array([1.0, -2.0, 3.0, 4.0, -5.0, 6.0], dtype=np.float32)
    got = world_wrench_from_ee_logged(ft_ee, pose)
    np.testing.assert_allclose(got, ft_ee, atol=1e-6)


def test_export_rotates_ft_wrist_and_requires_tcp_pose(tmp_path: Path):
    from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        export_real_episode_to_batched_dataset,
    )

    rotz90 = [
        0.0, -1.0, 0.0, 0.0,
        1.0,  0.0, 0.0, 0.0,
        0.0,  0.0, 1.0, 0.0,
        0.0,  0.0, 0.0, 1.0,
    ]
    kp = [100.0] * 6
    kd = [10.0] * 6
    target = _identity_pose_4x4([0.1, 0.2, 0.3])
    ft_ee = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pull = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    src = tmp_path / "rot_ft.parquet"
    _write_synthetic_real(
        src,
        action=[0.0] * 6,
        action_semantics="per-frame pose-control wrench [Fx, Fy, Fz, Tx, Ty, Tz]",
        action_order=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        target_pose_4x4=target,
        controller_gains={"task_prop_gains": kp, "task_deriv_gains": kd},
        tcp_pose_4x4=rotz90,
        ft_wrist=ft_ee,
        ft_wrist_raw=[2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    out = tmp_path / "batched"
    export_real_episode_to_batched_dataset(
        src, fixture_path=VARIANCE, output_dir=out, overwrite=True
    )
    arrays = BatchedSysIdDataset(out).load_episode_obs_arrays(0, 0)
    ft_world = arrays["ft_wrist"][0]
    np.testing.assert_allclose(ft_world[:3], [0.0, 1.0, 0.0], atol=1e-5)
    assert not np.allclose(ft_world[:3], ft_ee[:3])
    F = ft_world[:3]
    cos_fp = float(np.dot(F, pull) / (np.linalg.norm(F) * np.linalg.norm(pull)))
    assert cos_fp < 0.0
    np.testing.assert_allclose(
        arrays["raw_ft_wrist"][0, :3], [0.0, 2.0, 0.0], atol=1e-5
    )

    src_no_tcp = tmp_path / "no_tcp.parquet"
    _write_synthetic_real(
        src_no_tcp,
        action=[0.0] * 6,
        action_semantics="per-frame pose-control wrench [Fx, Fy, Fz, Tx, Ty, Tz]",
        action_order=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        target_pose_4x4=target,
        controller_gains={"task_prop_gains": kp, "task_deriv_gains": kd},
        ft_wrist=ft_ee,
        skip_tcp_pose_4x4=True,
    )
    with pytest.raises(ValueError, match="tcp_pose_4x4"):
        export_real_episode_to_batched_dataset(
            src_no_tcp, fixture_path=VARIANCE, output_dir=tmp_path / "out2", overwrite=True
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
