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
    np.testing.assert_allclose(spur, center, atol=1e-3)


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
