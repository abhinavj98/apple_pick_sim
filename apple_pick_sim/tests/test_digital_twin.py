"""Tests for digital-twin scene construction from field observations."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from apple_pick_sim.tests.conftest import FIXTURES_DIR, NO_SELF_COLLISION_KW, RANGES_FIXTURE

PROXY_FIXTURE = FIXTURES_DIR / "fruiting_system_ranges_real_world_proxy.json"
STRAIGHT_ROD_FIXTURE = RANGES_FIXTURE


def _import_dt():
    from apple_pick_sim import digital_twin as dt

    return dt


def _import_fs():
    import apple_pick_sim.fruiting_system as fs

    return fs


def _median_params(fs, ranges: dict):
    """Build params from fixture range midpoints (deterministic reference for round-trip)."""
    from apple_pick_sim.digital_twin.from_obs import params_from_ranges_median

    return params_from_ranges_median(ranges)


def _obs_from_scene(
    scene,
    *,
    fruiting_base_pos: tuple[float, float, float],
    weld_direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    apple_radius: float | None = None,
):
    fs = _import_fs()
    dt = _import_dt()
    parent, child = fs.fixed_joint_anchors_world(
        scene.model,
        scene.state_0.body_q,
        scene.fruiting_fixed_joints,
    )
    junction_names = [
        label.removeprefix("joint_") for _, label in scene.fruiting_fixed_joints
    ]
    if apple_radius is None and scene.params.apple_radius is not None:
        apple_radius = float(scene.params.apple_radius)
    return dt.DigitalTwinObs(
        fruiting_base_pos=fruiting_base_pos,
        weld_direction=weld_direction,
        junction_names=junction_names,
        woody_part_start_pos=parent,
        woody_part_end_pos=child,
        apple_radius=apple_radius,
    )


def test_load_save_roundtrip():
    """JSON serialization preserves observation fields."""
    dt = _import_dt()
    obs = dt.DigitalTwinObs(
        fruiting_base_pos=(0.0, 0.2, 1.3),
        weld_direction=(0.0, 0.0, 1.0),
        junction_names=["primary_secondary", "secondary_spur", "spur_stem", "stem_apple"],
        woody_part_start_pos=np.array(
            [0.1, 0.2, 1.0, 0.1, 0.2, 0.9, 0.1, 0.2, 0.8, 0.1, 0.2, 0.7],
            dtype=np.float32,
        ),
        woody_part_end_pos=np.array(
            [0.1, 0.2, 0.95, 0.1, 0.2, 0.85, 0.1, 0.2, 0.75, 0.1, 0.2, 0.65],
            dtype=np.float32,
        ),
        apple_radius=0.04,
        rod_radii={"primary": 0.012, "secondary": 0.01, "spur": 0.008, "stem": 0.004},
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "obs.json"
        dt.save_digital_twin_obs(obs, path)
        loaded = dt.load_digital_twin_obs(path)
    assert loaded.fruiting_base_pos == obs.fruiting_base_pos
    assert loaded.weld_direction == obs.weld_direction
    assert loaded.junction_names == obs.junction_names
    np.testing.assert_allclose(loaded.woody_part_start_pos, obs.woody_part_start_pos)
    np.testing.assert_allclose(loaded.woody_part_end_pos, obs.woody_part_end_pos)
    assert loaded.apple_radius == obs.apple_radius
    assert loaded.rod_radii == obs.rod_radii


def test_params_from_ranges_median_honors_spur_surface_offset():
    """Median params read spur_surface_offset from ranges (default True when absent)."""
    import copy

    from apple_pick_sim.digital_twin.from_obs import params_from_ranges_median

    fs = _import_fs()
    ranges = fs.load_ranges(PROXY_FIXTURE)
    assert "spur_surface_offset" not in ranges
    assert params_from_ranges_median(ranges).spur_surface_offset is True

    overridden = copy.deepcopy(ranges)
    overridden["spur_surface_offset"] = False
    assert params_from_ranges_median(overridden).spur_surface_offset is False


def test_infer_params_from_obs_forwards_spur_surface_offset(tmp_path: Path):
    """infer_params_from_obs propagates spur_surface_offset from the base fixture."""
    import copy

    dt = _import_dt()
    fs = _import_fs()
    custom = copy.deepcopy(json.loads(PROXY_FIXTURE.read_text()))
    custom["spur_surface_offset"] = False
    path = tmp_path / "ranges.json"
    path.write_text(json.dumps(custom))

    ranges = fs.load_ranges(PROXY_FIXTURE)
    args = fs.parse_fixture_args(ranges)
    base_pos = args.fruiting_base_pos or (0.0, 0.5, 0.95)
    ref_params = _median_params(fs, ranges)
    ref_scene = fs.generate_coupled_cable_scene(
        ranges,
        seed=0,
        params=ref_params,
        base_pos=base_pos,
        device="cpu",
        **NO_SELF_COLLISION_KW,
    )
    obs = _obs_from_scene(ref_scene, fruiting_base_pos=base_pos, weld_direction=(0.0, 1.0, 0.0))
    inferred = dt.infer_params_from_obs(obs, path)
    assert inferred.spur_surface_offset is False


def test_infer_params_uses_observed_rod_radii():
    """Measured rod radii override range midpoints when present."""
    dt = _import_dt()
    obs = dt.DigitalTwinObs(
        fruiting_base_pos=(0.0, 0.2, 1.3),
        weld_direction=(0.0, 0.0, 1.0),
        junction_names=["primary_secondary", "secondary_spur", "spur_stem", "stem_apple"],
        woody_part_start_pos=np.array(
            [0.0, 0.2, 1.2, 0.0, 0.2, 1.0, 0.0, 0.2, 0.8, 0.0, 0.2, 0.6],
            dtype=np.float32,
        ),
        woody_part_end_pos=np.array(
            [0.0, 0.2, 1.1, 0.0, 0.2, 0.9, 0.0, 0.2, 0.7, 0.0, 0.2, 0.5],
            dtype=np.float32,
        ),
        apple_radius=0.04,
        rod_radii={"primary": 0.011, "secondary": 0.009, "spur": 0.007, "stem": 0.003},
    )
    inferred = dt.infer_params_from_obs(obs, STRAIGHT_ROD_FIXTURE)

    assert inferred.primary is not None
    assert inferred.secondary is not None
    assert inferred.spur is not None
    assert inferred.stem is not None
    assert inferred.primary.radius == pytest.approx(0.011)
    assert inferred.secondary.radius == pytest.approx(0.009)
    assert inferred.spur.radius == pytest.approx(0.007)
    assert inferred.stem.radius == pytest.approx(0.003)


def test_round_trip_junction_positions():
    """Obs recorded from a median-built scene rebuilds matching junction anchors (<1 mm)."""
    fs = _import_fs()
    dt = _import_dt()
    ranges = fs.load_ranges(STRAIGHT_ROD_FIXTURE)
    args = fs.parse_fixture_args(ranges)
    base_pos = args.fruiting_base_pos or (0.0, 0.2, 1.3)
    ref_params = _median_params(fs, ranges)
    ref_scene = fs.generate_coupled_cable_scene(
        ranges,
        seed=0,
        params=ref_params,
        base_pos=base_pos,
        device="cpu",
        **NO_SELF_COLLISION_KW,
    )
    obs = _obs_from_scene(ref_scene, fruiting_base_pos=base_pos, weld_direction=(0.0, 0.0, -1.0))
    twin_scene = dt.build_digital_twin_scene(
        obs,
        STRAIGHT_ROD_FIXTURE,
        device="cpu",
        fix_to_apple=False,
        **NO_SELF_COLLISION_KW,
    )
    ref_parent, ref_child = fs.fixed_joint_anchors_world(
        ref_scene.model,
        ref_scene.state_0.body_q,
        ref_scene.fruiting_fixed_joints,
    )
    twin_parent, twin_child = fs.fixed_joint_anchors_world(
        twin_scene.model,
        twin_scene.state_0.body_q,
        twin_scene.fruiting_fixed_joints,
    )
    np.testing.assert_allclose(twin_parent, ref_parent, atol=1e-3)
    np.testing.assert_allclose(twin_child, ref_child, atol=1e-3)


def test_topology_primary_only():
    """Primary-only chain with stem_apple junction infers a single rod segment."""
    fs = _import_fs()
    dt = _import_dt()
    ranges = fs.load_ranges(STRAIGHT_ROD_FIXTURE)
    args = fs.parse_fixture_args(ranges)
    base_pos = args.fruiting_base_pos or (0.0, 0.2, 1.3)
    ref_params = _median_params(fs, ranges)
    ref_params = fs.FruitingSystemParams(
        primary=ref_params.primary,
        secondary=None,
        spur=None,
        stem=None,
        apple_radius=ref_params.apple_radius,
        apple_density=ref_params.apple_density,
        topology=fs.TOPOLOGY_LINEAR_CHAIN,
    )
    ref_scene = fs.generate_coupled_cable_scene(
        ranges,
        seed=0,
        params=ref_params,
        base_pos=base_pos,
        device="cpu",
        **NO_SELF_COLLISION_KW,
    )
    obs = _obs_from_scene(ref_scene, fruiting_base_pos=base_pos)
    inferred = dt.infer_params_from_obs(obs, STRAIGHT_ROD_FIXTURE)
    assert inferred.primary is not None
    assert inferred.secondary is None
    assert inferred.spur is None
    assert inferred.stem is None
    assert inferred.apple_radius is not None


def test_topology_full_chain():
    """Full four-rod topology is recovered from junction names."""
    fs = _import_fs()
    dt = _import_dt()
    ranges = fs.load_ranges(STRAIGHT_ROD_FIXTURE)
    args = fs.parse_fixture_args(ranges)
    base_pos = args.fruiting_base_pos or (0.0, 0.2, 1.3)
    ref_params = _median_params(fs, ranges)
    ref_scene = fs.generate_coupled_cable_scene(
        ranges,
        seed=0,
        params=ref_params,
        base_pos=base_pos,
        device="cpu",
        **NO_SELF_COLLISION_KW,
    )
    obs = _obs_from_scene(ref_scene, fruiting_base_pos=base_pos)
    inferred = dt.infer_params_from_obs(obs, STRAIGHT_ROD_FIXTURE)
    assert inferred.primary is not None
    assert inferred.secondary is not None
    assert inferred.spur is not None
    assert inferred.stem is not None


def _import_example_digital_twin():
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "examples" / "example_digital_twin.py"
    spec = importlib.util.spec_from_file_location("example_digital_twin", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_example_digital_twin_registers_model_with_viewer():
    """GL viewer needs set_model before log_state can draw collision shapes."""
    import argparse
    from unittest.mock import MagicMock

    mod = _import_example_digital_twin()
    viewer = MagicMock()
    args = argparse.Namespace(
        obs=str(FIXTURES_DIR / "digital_twin_obs_straight_rod_initial.json"),
        base_fixture=str(STRAIGHT_ROD_FIXTURE),
        settle_substeps=0,
        fix_to_apple=False,
        enable_self_collision=False,
        device="cpu",
    )
    example = mod.ExampleDigitalTwin(viewer, args)
    viewer.set_model.assert_called_once_with(example.model)


def test_infer_params_rejects_mismatched_anchor_lengths():
    """Mismatched start/end flat lengths raise a clear validation error."""
    dt = _import_dt()
    with pytest.raises(ValueError, match="woody_part"):
        dt.DigitalTwinObs(
            fruiting_base_pos=(0.0, 0.0, 1.0),
            weld_direction=(1.0, 0.0, 0.0),
            junction_names=["primary_secondary"],
            woody_part_start_pos=np.zeros(3, dtype=np.float32),
            woody_part_end_pos=np.zeros(6, dtype=np.float32),
            apple_radius=0.04,
        )


def test_fixture_catalog_references_existing_assets():
    fs = _import_fs()
    dt = _import_dt()
    catalog_path = FIXTURES_DIR / "digital_twin_fixture_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    assert catalog["schema"] == "apple_pick_sim_fixture_catalog_v1"
    assert "straight_rod_test" in catalog["fixtures"]

    repo_root = FIXTURES_DIR.parents[1]
    for name, fixture in catalog["fixtures"].items():
        assert fixture["description"]
        ranges_path = repo_root / fixture["ranges_path"]
        assert ranges_path.exists()
        ranges = fs.load_ranges(ranges_path)
        args = fs.parse_fixture_args(ranges)
        assert tuple(fixture["fruiting_base_pos"]) == pytest.approx(args.fruiting_base_pos)
        assert tuple(fixture["robot_base_pos"]) == pytest.approx(args.robot_base_pos)

        observation_path = fixture.get("observation_path")
        if observation_path is not None:
            obs_path = repo_root / observation_path
            assert obs_path.exists()
            obs = dt.load_digital_twin_obs(obs_path)
            assert obs.fruiting_base_pos == pytest.approx(fixture["fruiting_base_pos"])

        assert len(fixture["fruiting_base_pos"]) == 3
        assert len(fixture["robot_base_pos"]) == 3
        assert fixture["smoke_commands"], name
        assert all(cmd.startswith("uv run ") for cmd in fixture["smoke_commands"])


def test_round_trip_t_junction_junction_positions():
    """T-junction obs with support labels rebuilds matching rod junction anchors."""
    fs = _import_fs()
    dt = _import_dt()
    ranges = fs.load_ranges(PROXY_FIXTURE)
    args = fs.parse_fixture_args(ranges)
    base_pos = args.fruiting_base_pos or (0.0, 0.5, 0.95)
    ref_params = _median_params(fs, ranges)
    ref_scene = fs.generate_coupled_cable_scene(
        ranges,
        seed=0,
        params=ref_params,
        base_pos=base_pos,
        device="cpu",
        **NO_SELF_COLLISION_KW,
    )
    obs = _obs_from_scene(ref_scene, fruiting_base_pos=base_pos, weld_direction=(0.0, 1.0, 0.0))
    inferred = dt.infer_params_from_obs(obs, PROXY_FIXTURE)
    assert inferred.topology == fs.TOPOLOGY_T_JUNCTION
    twin_scene = dt.build_digital_twin_scene(
        obs,
        PROXY_FIXTURE,
        device="cpu",
        fix_to_apple=False,
        **NO_SELF_COLLISION_KW,
    )
    rod_labels = {
        lab
        for lab in (label.removeprefix("joint_") for _, label in ref_scene.fruiting_fixed_joints)
        if not lab.startswith("primary_support_")
    }
    ref_parent, ref_child = fs.fixed_joint_anchors_world(
        ref_scene.model,
        ref_scene.state_0.body_q,
        [(j, l) for j, l in ref_scene.fruiting_fixed_joints if l.removeprefix("joint_") in rod_labels],
    )
    twin_parent, twin_child = fs.fixed_joint_anchors_world(
        twin_scene.model,
        twin_scene.state_0.body_q,
        [(j, l) for j, l in twin_scene.fruiting_fixed_joints if l.removeprefix("joint_") in rod_labels],
    )
    np.testing.assert_allclose(twin_parent, ref_parent, atol=1e-3)
    np.testing.assert_allclose(twin_child, ref_child, atol=1e-3)
