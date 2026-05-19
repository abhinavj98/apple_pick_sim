"""Tests for M1 coupled cable scene generation (``generate_coupled_cable_scene``).

Validates P0-equivalent fruiting metadata plus a collision-equipped gripper proxy body
on the same VBD ``Model`` before robot ``Model`` integration (Slice 2a).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
RANGES_FIXTURE = FIXTURES_DIR / "fruiting_system_ranges_straight_rod_test.json"


def _import_fs():
    import apple_pick_sim.fruiting_system as fs

    return fs


@pytest.fixture(scope="module", autouse=True)
def _warp_init():
    import warp as wp

    wp.init()


def test_generate_coupled_cable_scene_returns_coupled_type():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_coupled_cable_scene(ranges, seed=0)
    assert isinstance(scene, fs.CoupledCableScene)
    assert scene.model is not None
    assert scene.solver is not None


def test_coupled_scene_has_one_extra_body_vs_p0():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    seed = 7
    p0 = fs.generate_scene(ranges, seed=seed)
    coupled = fs.generate_coupled_cable_scene(ranges, seed=seed)
    assert coupled.model.body_count == p0.model.body_count + 1


def test_coupled_fruiting_metadata_matches_p0_for_same_seed():
    """Rod/apple indices and fixed-joint list match P0; only the proxy body is added."""
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    seed = 11
    p0 = fs.generate_scene(ranges, seed=seed)
    coupled = fs.generate_coupled_cable_scene(ranges, seed=seed)

    assert coupled.params == p0.params
    assert coupled.primary_bodies == p0.primary_bodies
    assert coupled.secondary_bodies == p0.secondary_bodies
    assert coupled.spur_bodies == p0.spur_bodies
    assert coupled.stem_bodies == p0.stem_bodies
    assert coupled.apple_body == p0.apple_body
    assert coupled.cable_joint_indices == p0.cable_joint_indices
    assert len(coupled.fruiting_fixed_joints) == len(p0.fruiting_fixed_joints) + 1
    assert set(p0.fruiting_fixed_joints).issubset(set(coupled.fruiting_fixed_joints))


def test_coupled_geometry_fingerprint_matches_p0_apple_and_rod_counts():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    seed = 99
    coupled = fs.generate_coupled_cable_scene(ranges, seed=seed)
    fp_p0 = fs.geometry_fingerprint(fs.generate_scene(ranges, seed=seed))
    fp_coupled = fs.geometry_fingerprint_coupled(coupled)
    for key in (
        "primary_body_count",
        "secondary_body_count",
        "spur_body_count",
        "stem_body_count",
        "apple_pos",
        "primary_tip_pos",
    ):
        assert fp_coupled[key] == fp_p0[key]
    assert fp_coupled["gripper_proxy_body"] == coupled.gripper_proxy_body


def test_gripper_proxy_body_index_stable_across_rebuild():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    a = fs.generate_coupled_cable_scene(ranges, seed=3)
    b = fs.generate_coupled_cable_scene(ranges, seed=3)
    assert a.gripper_proxy_body == b.gripper_proxy_body


def test_gripper_proxy_near_apple_surface():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_coupled_cable_scene(ranges, seed=5)
    assert scene.apple_body is not None
    assert scene.params.apple_radius is not None

    body_q = scene.state_0.body_q.to("cpu").numpy()
    apple = body_q[scene.apple_body]
    proxy = body_q[scene.gripper_proxy_body]
    apple_pos = apple[:3]
    proxy_pos = proxy[:3]
    dist = float(np.linalg.norm(proxy_pos - apple_pos))
    r = scene.params.apple_radius
  # proxy COM sits outside the apple sphere by roughly one proxy half-width
    assert dist >= r * 0.5
    assert dist <= r * 3.0


def test_gripper_proxy_has_collision_shape():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_coupled_cable_scene(ranges, seed=2)
    shapes = scene.model.body_shapes.get(scene.gripper_proxy_body, [])
    assert len(shapes) >= 1


def test_proxy_registry_maps_robot_body_to_proxy():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_coupled_cable_scene(ranges, seed=0)
    reg = scene.proxy_registry(robot_body_id=42)
    assert reg.robot_to_proxy == ((42, scene.gripper_proxy_body),)


def test_fix_proxy_to_apple_adds_fixed_joint():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    cfg = fs.GripperProxyConfig(fix_to_apple=True)
    scene = fs.generate_coupled_cable_scene(ranges, seed=4, gripper_proxy=cfg)
    assert scene.gripper_proxy_apple_joint is not None
    labels = [label for _, label in scene.fruiting_fixed_joints]
    assert any("gripper_proxy" in label or "proxy" in label for label in labels)


def test_fix_to_apple_proxy_on_apple_surface_not_at_com():
    """Fixed proxy is welded at the apple surface, not coincident with the apple COM."""
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_coupled_cable_scene(
        ranges, seed=5, gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True)
    )
    assert scene.apple_body is not None
    assert scene.params.apple_radius is not None
    body_q = scene.state_0.body_q.to("cpu").numpy()
    apple_pos = body_q[scene.apple_body, :3]
    proxy_pos = body_q[scene.gripper_proxy_body, :3]
    dist = float(np.linalg.norm(proxy_pos - apple_pos))
    r = scene.params.apple_radius
    assert dist >= r * 0.5
    assert dist <= r * 3.0


def test_coupled_short_vbd_rollout_finite():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_coupled_cable_scene(ranges, seed=8)
    fs.run_rollout(scene, num_steps=3, sim_substeps=4)
    body_q = scene.state_0.body_q.to("cpu").numpy()
    assert np.isfinite(body_q).all()


def test_measure_fruiting_forces_works_on_coupled_scene():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_coupled_cable_scene(ranges, seed=3, device="cpu")
    fs.run_rollout(scene, num_steps=2, sim_substeps=5)
    q_prev = scene.state_1.body_q
    out = fs.measure_fruiting_forces(scene, scene.state_0.body_q, q_prev, dt=1.0e-3)
    assert len(out["fixed_joints"]) == len(scene.fruiting_fixed_joints)
