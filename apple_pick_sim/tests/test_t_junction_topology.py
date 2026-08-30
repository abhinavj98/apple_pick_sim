"""Tests for default T-junction fruiting topology (dual supports + mid-span spur)."""

from __future__ import annotations

import numpy as np
import pytest

from apple_pick_sim.tests.conftest import FIXTURES_DIR

PROXY_PATH = FIXTURES_DIR / "fruiting_system_ranges_real_world_proxy.json"
STRAIGHT_PATH = FIXTURES_DIR / "fruiting_system_ranges_straight_rod_test.json"
BASE_POS = (0.0, 0.5, 0.95)


def _import_fs():
    import apple_pick_sim.fruiting_system as fs

    return fs


def test_sample_params_defaults_to_t_junction():
    fs = _import_fs()
    ranges = fs.load_ranges(PROXY_PATH)
    params = fs.sample_params(ranges, seed=0)
    assert params.topology == fs.TOPOLOGY_T_JUNCTION
    assert params.spur_attach_fraction == pytest.approx(0.5)
    assert params.spur_surface_offset is True
    assert params.stem_surface_offset is True


def test_t_junction_stem_surface_offset_shifts_stem_anchor_by_spur_radius():
    from apple_pick_sim.fruiting_system.build import _rod_tip_surface_offset_world

    fs = _import_fs()
    ranges = fs.load_ranges(PROXY_PATH)
    params_on = fs.sample_params(ranges, seed=0)
    params_off = fs.sample_params({**ranges, "stem_surface_offset": False}, seed=0)
    assert params_on.stem_surface_offset is True
    assert params_off.stem_surface_offset is False
    scene_on = fs.generate_scene(
        ranges, seed=0, base_pos=BASE_POS, device="cpu", enable_self_collisions=False
    )
    scene_off = fs.generate_scene(
        {**ranges, "stem_surface_offset": False},
        seed=0,
        base_pos=BASE_POS,
        device="cpu",
        enable_self_collisions=False,
    )

    def _stem_anchor(scene):
        _, child_anchors = fs.fixed_joint_anchors_world(
            scene.model,
            scene.state_0.body_q,
            scene.fruiting_fixed_joints,
        )
        labels = [lab for _, lab in scene.fruiting_fixed_joints]
        stem_i = labels.index("joint_spur_stem")
        return child_anchors[stem_i * 3 : (stem_i + 1) * 3]

    stem_on = np.asarray(_stem_anchor(scene_on), dtype=np.float64)
    stem_off = np.asarray(_stem_anchor(scene_off), dtype=np.float64)
    assert params_on.spur is not None and params_on.stem is not None
    offset = np.array(
        _rod_tip_surface_offset_world(params_on.stem.direction, params_on.spur.radius),
        dtype=np.float64,
    )
    assert np.linalg.norm(offset) > 1e-6
    np.testing.assert_allclose(stem_on - stem_off, offset, atol=1e-3)


def test_t_junction_spur_surface_offset_false_legacy_centerline():
    from apple_pick_sim.fruiting_system.build import _primary_radial_surface_offset_world

    fs = _import_fs()
    ranges = fs.load_ranges(PROXY_PATH)
    params_on = fs.sample_params(ranges, seed=0)
    params_off = fs.sample_params({**ranges, "spur_surface_offset": False}, seed=0)
    assert params_on.spur_surface_offset is True
    assert params_off.spur_surface_offset is False
    scene_on = fs.generate_scene(
        ranges, seed=0, base_pos=BASE_POS, device="cpu", enable_self_collisions=False
    )
    scene_off = fs.generate_scene(
        {**ranges, "spur_surface_offset": False},
        seed=0,
        base_pos=BASE_POS,
        device="cpu",
        enable_self_collisions=False,
    )

    def _spur_anchor(scene):
        _, child_anchors = fs.fixed_joint_anchors_world(
            scene.model,
            scene.state_0.body_q,
            scene.fruiting_fixed_joints,
        )
        labels = [lab for _, lab in scene.fruiting_fixed_joints]
        spur_i = labels.index("joint_primary_spur")
        return child_anchors[spur_i * 3 : (spur_i + 1) * 3]

    spur_on = np.asarray(_spur_anchor(scene_on), dtype=np.float64)
    spur_off = np.asarray(_spur_anchor(scene_off), dtype=np.float64)
    radial = np.array(
        _primary_radial_surface_offset_world(
            params_on.primary.direction, params_on.spur.direction, params_on.primary.radius
        ),
        dtype=np.float64,
    )
    assert np.linalg.norm(radial) > 1e-6
    np.testing.assert_allclose(spur_on - spur_off, radial, atol=1e-3)


def test_linear_chain_fixture_opt_in():
    fs = _import_fs()
    ranges = fs.load_ranges(STRAIGHT_PATH)
    params = fs.sample_params(ranges, seed=0)
    assert params.topology == fs.TOPOLOGY_LINEAR_CHAIN


def test_t_junction_fixed_joint_labels():
    fs = _import_fs()
    ranges = fs.load_ranges(PROXY_PATH)
    scene = fs.generate_scene(
        ranges, seed=0, base_pos=BASE_POS, device="cpu", enable_self_collisions=False
    )
    labels = {lab for _, lab in scene.fruiting_fixed_joints}
    assert labels == {
        "joint_primary_support_left",
        "joint_primary_support_right",
        "joint_primary_spur",
        "joint_spur_stem",
        "joint_stem_apple",
    }


def test_t_junction_primary_endpoint_has_mass():
    fs = _import_fs()
    ranges = fs.load_ranges(PROXY_PATH)
    scene = fs.generate_scene(
        ranges, seed=0, base_pos=BASE_POS, device="cpu", enable_self_collisions=False
    )
    assert scene.primary_bodies
    assert float(scene.model.body_mass.numpy()[scene.primary_bodies[0]]) > 0.0


def test_t_junction_center_and_support_anchors():
    from apple_pick_sim.fruiting_system.build import _primary_radial_surface_offset_world

    fs = _import_fs()
    ranges = fs.load_ranges(PROXY_PATH)
    params = fs.sample_params(ranges, seed=0)
    scene = fs.generate_scene(
        ranges, seed=0, base_pos=BASE_POS, device="cpu", enable_self_collisions=False
    )
    parent_anchors, child_anchors = fs.fixed_joint_anchors_world(
        scene.model,
        scene.state_0.body_q,
        scene.fruiting_fixed_joints,
    )
    labels = [lab for _, lab in scene.fruiting_fixed_joints]
    left_i = labels.index("joint_primary_support_left")
    right_i = labels.index("joint_primary_support_right")
    spur_i = labels.index("joint_primary_spur")

    left = child_anchors[left_i * 3 : (left_i + 1) * 3]
    right = child_anchors[right_i * 3 : (right_i + 1) * 3]
    spur = child_anchors[spur_i * 3 : (spur_i + 1) * 3]
    center = np.asarray(BASE_POS, dtype=np.float64)

    half_span = params.primary.length / 2.0
    np.testing.assert_allclose(left, center - np.array([half_span, 0.0, 0.0]), atol=1e-3)
    np.testing.assert_allclose(right, center + np.array([half_span, 0.0, 0.0]), atol=1e-3)
    radial = np.array(
        _primary_radial_surface_offset_world(
            params.primary.direction, params.spur.direction, params.primary.radius
        ),
        dtype=np.float64,
    )
    scene_off = fs.generate_scene(
        {**ranges, "spur_surface_offset": False},
        seed=0,
        base_pos=BASE_POS,
        device="cpu",
        enable_self_collisions=False,
    )
    _, child_off = fs.fixed_joint_anchors_world(
        scene_off.model,
        scene_off.state_0.body_q,
        scene_off.fruiting_fixed_joints,
    )
    spur_off_i = [lab for _, lab in scene_off.fruiting_fixed_joints].index("joint_primary_spur")
    spur_off = child_off[spur_off_i * 3 : (spur_off_i + 1) * 3]
    np.testing.assert_allclose(spur, np.asarray(spur_off, dtype=np.float64) + radial, atol=1e-3)


def test_t_junction_rejects_secondary():
    fs = _import_fs()
    from apple_pick_sim.fruiting_system.scene import _build_scene

    ranges = fs.load_ranges(PROXY_PATH)
    params = fs.sample_params(ranges, seed=0)
    assert params.primary is not None
    secondary = fs.rod_params_from_vbd_targets(
        num_segments=2,
        length=0.05,
        radius=0.01,
        bend_stiffness=10.0,
        bend_damping=0.1,
        stretch_stiffness=1.0e6,
        density=300.0,
        direction=(1.0, 0.0, 0.0),
    )
    params = fs.FruitingSystemParams(
        primary=params.primary,
        secondary=secondary,
        spur=params.spur,
        stem=params.stem,
        apple_radius=params.apple_radius,
        apple_density=params.apple_density,
        topology=fs.TOPOLOGY_T_JUNCTION,
        spur_attach_fraction=params.spur_attach_fraction,
    )
    with pytest.raises(ValueError, match="secondary"):
        _build_scene(params, base_pos=BASE_POS, device="cpu", enable_self_collisions=False)


def test_t_junction_settles():
    fs = _import_fs()
    ranges = fs.load_ranges(PROXY_PATH)
    scene = fs.generate_scene(
        ranges, seed=1, base_pos=BASE_POS, device="cpu", enable_self_collisions=False
    )
    fs.run_rollout(scene, num_steps=5, sim_substeps=10, fps=60.0)
    bq = scene.state_0.body_q.numpy().reshape(-1, 7)
    assert np.isfinite(bq).all()
