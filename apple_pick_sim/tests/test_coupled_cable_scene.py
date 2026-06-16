"""Tests for M1 coupled cable scene generation (``generate_coupled_cable_scene``).

Validates P0-equivalent fruiting metadata plus a collision-equipped gripper proxy body
on the same VBD ``Model`` before robot ``Model`` integration (Slice 2a).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apple_pick_sim.tests.conftest import (
    COUPLED_BASE_POS,
    COUPLED_ROBOT_BASE_POS,
    COUPLED_VBD_SCENE_KW,
    NO_SELF_COLLISION_KW,
    RANGES_FIXTURE,
)


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
    scene = fs.generate_coupled_cable_scene(
        ranges, seed=0, base_pos=COUPLED_BASE_POS, **NO_SELF_COLLISION_KW
    )
    assert isinstance(scene, fs.CoupledCableScene)
    assert scene.model is not None
    assert scene.solver is not None


def test_coupled_scene_has_one_extra_body_vs_p0():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    seed = 7
    p0 = fs.generate_scene(ranges, seed=seed, **COUPLED_VBD_SCENE_KW)
    coupled = fs.generate_coupled_cable_scene(ranges, seed=seed, **COUPLED_VBD_SCENE_KW)
    assert coupled.model.body_count == p0.model.body_count + 1


def test_coupled_fruiting_metadata_matches_p0_for_same_seed():
    """Rod/apple indices and fixed-joint list match P0; only the proxy body is added."""
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    seed = 11
    p0 = fs.generate_scene(ranges, seed=seed, **COUPLED_VBD_SCENE_KW)
    coupled = fs.generate_coupled_cable_scene(ranges, seed=seed, **COUPLED_VBD_SCENE_KW)

    assert coupled.params == p0.params
    assert coupled.primary_bodies == p0.primary_bodies
    assert coupled.secondary_bodies == p0.secondary_bodies
    assert coupled.spur_bodies == p0.spur_bodies
    assert coupled.stem_bodies == p0.stem_bodies
    assert coupled.apple_body == p0.apple_body
    assert coupled.cable_joint_indices == p0.cable_joint_indices
    assert set(p0.fruiting_fixed_joints).issubset(set(coupled.fruiting_fixed_joints))
    assert len(coupled.fruiting_fixed_joints) == len(p0.fruiting_fixed_joints)
    coupled_welded = fs.generate_coupled_cable_scene(
        ranges,
        seed=seed,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True),
        **COUPLED_VBD_SCENE_KW,
    )
    assert len(coupled_welded.fruiting_fixed_joints) == len(p0.fruiting_fixed_joints) + 1


def test_coupled_geometry_fingerprint_matches_p0_apple_and_rod_counts():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    seed = 99
    coupled = fs.generate_coupled_cable_scene(ranges, seed=seed, **COUPLED_VBD_SCENE_KW)
    fp_p0 = fs.geometry_fingerprint(
        fs.generate_scene(ranges, seed=seed, **COUPLED_VBD_SCENE_KW)
    )
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
    a = fs.generate_coupled_cable_scene(ranges, seed=3, **COUPLED_VBD_SCENE_KW)
    b = fs.generate_coupled_cable_scene(ranges, seed=3, **COUPLED_VBD_SCENE_KW)
    assert a.gripper_proxy_body == b.gripper_proxy_body


def test_gripper_proxy_near_apple_surface():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_coupled_cable_scene(ranges, seed=5, **COUPLED_VBD_SCENE_KW)
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
    scene = fs.generate_coupled_cable_scene(ranges, seed=2, **COUPLED_VBD_SCENE_KW)
    shapes = scene.model.body_shapes.get(scene.gripper_proxy_body, [])
    assert len(shapes) >= 1


def test_proxy_registry_maps_robot_body_to_proxy():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_coupled_cable_scene(
        ranges, seed=0, base_pos=COUPLED_BASE_POS, **NO_SELF_COLLISION_KW
    )
    reg = scene.proxy_registry(robot_body_id=42)
    assert reg.robot_to_proxy == ((42, scene.gripper_proxy_body),)


def test_fix_proxy_to_apple_adds_fixed_joint():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    cfg = fs.GripperProxyConfig(fix_to_apple=True)
    scene = fs.generate_coupled_cable_scene(
        ranges, seed=4, gripper_proxy=cfg, **COUPLED_VBD_SCENE_KW
    )
    assert scene.gripper_proxy_apple_joint is not None
    labels = [label for _, label in scene.fruiting_fixed_joints]
    assert any("gripper_proxy" in label or "proxy" in label for label in labels)


def test_fix_to_apple_proxy_on_apple_surface_not_at_com():
    """Fixed proxy is welded at the apple surface, not coincident with the apple COM."""
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_coupled_cable_scene(
        ranges,
        seed=5,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True),
        **COUPLED_VBD_SCENE_KW,
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


def _stem_direction_world(scene) -> np.ndarray:
    """Unit vector from distal stem segment base toward tip (apple pole)."""
    stem = scene.stem_bodies
    assert len(stem) >= 2
    body_q = scene.state_0.body_q.to("cpu").numpy()
    tip = body_q[stem[-1], :3]
    base = body_q[stem[-2], :3]
    d = tip - base
    return d / np.linalg.norm(d)


def test_robot_facing_weld_places_proxy_toward_robot_base():
    """Welded grasp sits on the stem-perpendicular robot-facing pole."""
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_coupled_cable_scene(
        ranges,
        seed=5,
        gripper_proxy=fs.GripperProxyConfig(
            fix_to_apple=True,
            robot_facing_weld=True,
        ),
        robot_base_pos=COUPLED_ROBOT_BASE_POS,
        **COUPLED_VBD_SCENE_KW,
    )
    body_q = scene.state_0.body_q.to("cpu").numpy()
    apple_pos = body_q[scene.apple_body, :3]
    proxy_pos = body_q[scene.gripper_proxy_body, :3]
    robot_vec = np.asarray(COUPLED_ROBOT_BASE_POS, dtype=np.float64) - apple_pos
    weld_vec = proxy_pos - apple_pos
    weld_vec /= np.linalg.norm(weld_vec)
    stem = _stem_direction_world(scene)
    from apple_pick_sim.system_id import stem_perpendicular_robot_pole

    pole = stem_perpendicular_robot_pole(stem, robot_vec)
    assert float(np.dot(weld_vec, pole)) > 0.99
    assert abs(float(np.dot(weld_vec, stem))) < 0.05


def test_weld_direction_replaces_pole():
    """Explicit weld_direction places proxy off the stem-perpendicular pole."""
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    ref = fs.generate_coupled_cable_scene(
        ranges,
        seed=5,
        gripper_proxy=fs.GripperProxyConfig(
            fix_to_apple=True,
            robot_facing_weld=True,
        ),
        robot_base_pos=COUPLED_ROBOT_BASE_POS,
        **COUPLED_VBD_SCENE_KW,
    )
    body_q = ref.state_0.body_q.to("cpu").numpy()
    apple_pos = body_q[ref.apple_body, :3]
    robot_vec = np.asarray(COUPLED_ROBOT_BASE_POS, dtype=np.float64) - apple_pos
    stem = _stem_direction_world(ref)
    from apple_pick_sim.system_id import stem_perpendicular_robot_pole

    pole = stem_perpendicular_robot_pole(stem, robot_vec)

    # Oblique direction on the stem-perpendicular hemisphere (offset in the stem plane).
    oblique = pole + np.array([0.3, 0.2, 0.0], dtype=np.float64)
    oblique /= np.linalg.norm(oblique)
    assert float(np.dot(oblique, pole)) > 0.0

    scene = fs.generate_coupled_cable_scene(
        ranges,
        seed=5,
        gripper_proxy=fs.GripperProxyConfig(
            fix_to_apple=True,
            robot_facing_weld=True,
            weld_direction=(float(oblique[0]), float(oblique[1]), float(oblique[2])),
        ),
        robot_base_pos=COUPLED_ROBOT_BASE_POS,
        **COUPLED_VBD_SCENE_KW,
    )
    body_q = scene.state_0.body_q.to("cpu").numpy()
    proxy_pos = body_q[scene.gripper_proxy_body, :3]
    weld_vec = proxy_pos - apple_pos
    weld_dir = weld_vec / np.linalg.norm(weld_vec)
    assert float(np.dot(weld_dir, oblique)) > 0.99
    assert float(np.dot(weld_dir, pole)) < 0.99


def test_weld_reference_pos_accepts_settled_hemisphere_direction():
    """Settled apple center override validates weld directions from post-settle pose."""
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    ref = fs.generate_coupled_cable_scene(
        ranges,
        seed=2345,
        gripper_proxy=fs.GripperProxyConfig(
            fix_to_apple=True,
            robot_facing_weld=True,
        ),
        robot_base_pos=COUPLED_ROBOT_BASE_POS,
        **COUPLED_VBD_SCENE_KW,
    )
    nominal_apple = ref.state_0.body_q.to("cpu").numpy()[ref.apple_body, :3]
    settled_apple = np.array([0.21797176, 0.33258533, 0.6142479], dtype=np.float64)
    settled_quat = np.array([0.01010303, 0.832637, -0.09918664, -0.5447711], dtype=np.float64)

    robot_vec_nom = np.asarray(COUPLED_ROBOT_BASE_POS, dtype=np.float64) - nominal_apple
    robot_vec_set = np.asarray(COUPLED_ROBOT_BASE_POS, dtype=np.float64) - settled_apple
    physical_stem = _stem_direction_world(ref)

    from apple_pick_sim.system_id import sample_fibonacci_hemisphere, stem_perpendicular_robot_pole

    pole_set = stem_perpendicular_robot_pole(physical_stem, robot_vec_set)
    pole_nom = stem_perpendicular_robot_pole(physical_stem, robot_vec_nom)

    samples = sample_fibonacci_hemisphere(64, pole_set)
    weld = next(
        (
            d
            for d in samples
            if float(np.dot(d, pole_set)) >= 0.0 and float(np.dot(d, pole_nom)) < 0.0
        ),
        None,
    )
    if weld is None:
        # Tilt toward the stem to leave the nominal cap while staying near the settled pole.
        weld = pole_set + 0.85 * physical_stem
        weld = weld / np.linalg.norm(weld)
    assert float(np.dot(weld, pole_set)) >= 0.0
    assert float(np.dot(weld, pole_nom)) < 0.0

    with pytest.raises(ValueError, match="stem-perpendicular"):
        fs.generate_coupled_cable_scene(
            ranges,
            seed=2345,
            gripper_proxy=fs.GripperProxyConfig(
                fix_to_apple=True,
                robot_facing_weld=True,
                weld_direction=(float(weld[0]), float(weld[1]), float(weld[2])),
            ),
            robot_base_pos=COUPLED_ROBOT_BASE_POS,
            **COUPLED_VBD_SCENE_KW,
        )

    scene = fs.generate_coupled_cable_scene(
        ranges,
        seed=2345,
        gripper_proxy=fs.GripperProxyConfig(
            fix_to_apple=True,
            robot_facing_weld=True,
            weld_direction=(float(weld[0]), float(weld[1]), float(weld[2])),
            weld_reference_pos=(
                float(settled_apple[0]),
                float(settled_apple[1]),
                float(settled_apple[2]),
            ),
            weld_reference_quat=(
                float(settled_quat[0]),
                float(settled_quat[1]),
                float(settled_quat[2]),
                float(settled_quat[3]),
            ),
        ),
        robot_base_pos=COUPLED_ROBOT_BASE_POS,
        **COUPLED_VBD_SCENE_KW,
    )
    assert scene.gripper_proxy_offset_in_apple_frame is not None


def test_robot_facing_weld_requires_robot_base_pos():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    with pytest.raises(ValueError, match="robot_facing_weld requires robot_base_pos"):
        fs.generate_coupled_cable_scene(
            ranges,
            seed=5,
            gripper_proxy=fs.GripperProxyConfig(
                fix_to_apple=True,
                robot_facing_weld=True,
            ),
            **COUPLED_VBD_SCENE_KW,
        )


def test_fix_to_apple_weld_direction_is_stem_pole():
    """Welded grasp stays on the stem-side exterior pole."""
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    cfg = fs.GripperProxyConfig(fix_to_apple=True)
    scene = fs.generate_coupled_cable_scene(ranges, seed=3, gripper_proxy=cfg, **COUPLED_VBD_SCENE_KW)
    stem = _stem_direction_world(scene)
    body_q = scene.state_0.body_q.to("cpu").numpy()
    apple_pos = body_q[scene.apple_body, :3]
    proxy_pos = body_q[scene.gripper_proxy_body, :3]
    weld_dir = (proxy_pos - apple_pos) / np.linalg.norm(proxy_pos - apple_pos)
    assert float(np.dot(weld_dir, stem)) > 0.99


def test_free_and_welded_proxy_share_stem_pole():
    """Free and welded builds place the proxy on the same stem pole."""
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    free = fs.generate_coupled_cable_scene(
        ranges, seed=7, gripper_proxy=fs.GripperProxyConfig(fix_to_apple=False), **COUPLED_VBD_SCENE_KW
    )
    welded = fs.generate_coupled_cable_scene(
        ranges, seed=7, gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True), **COUPLED_VBD_SCENE_KW
    )
    bq_free = free.state_0.body_q.to("cpu").numpy()
    bq_weld = welded.state_0.body_q.to("cpu").numpy()
    free_dir = bq_free[free.gripper_proxy_body, :3] - bq_free[free.apple_body, :3]
    weld_dir = bq_weld[welded.gripper_proxy_body, :3] - bq_weld[welded.apple_body, :3]
    free_dir /= np.linalg.norm(free_dir)
    weld_dir /= np.linalg.norm(weld_dir)
    assert float(np.dot(free_dir, weld_dir)) > 0.99


def test_free_proxy_remains_stem_aligned():
    """Non-welded proxy stays on the stem pole (unchanged placement)."""
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_coupled_cable_scene(
        ranges,
        seed=3,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=False),
        **COUPLED_VBD_SCENE_KW,
    )
    body_q = scene.state_0.body_q.to("cpu").numpy()
    apple_pos = body_q[scene.apple_body, :3]
    proxy_pos = body_q[scene.gripper_proxy_body, :3]
    grasp_dir = (proxy_pos - apple_pos) / np.linalg.norm(proxy_pos - apple_pos)
    stem = _stem_direction_world(scene)
    assert float(np.dot(grasp_dir, stem)) > 0.99


def test_coupled_short_vbd_rollout_finite():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_coupled_cable_scene(ranges, seed=8, **COUPLED_VBD_SCENE_KW)
    fs.run_rollout(scene, num_steps=3, sim_substeps=4)
    body_q = scene.state_0.body_q.to("cpu").numpy()
    assert np.isfinite(body_q).all()


def test_coupled_geometry_fingerprint_includes_proxy_fields():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_coupled_cable_scene(ranges, seed=6, **COUPLED_VBD_SCENE_KW)
    fp = fs.geometry_fingerprint_coupled(scene)
    assert "gripper_proxy_body" in fp
    assert "gripper_proxy_pos" in fp
    assert fp["gripper_proxy_body"] == scene.gripper_proxy_body
    p0_fp = fs.geometry_fingerprint(
        fs.generate_scene(ranges, seed=6, **COUPLED_VBD_SCENE_KW)
    )
    assert "gripper_proxy_pos" not in p0_fp


def test_params_fingerprint_stable_across_coupled_rebuild():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    a = fs.generate_coupled_cable_scene(ranges, seed=21, **COUPLED_VBD_SCENE_KW)
    b = fs.generate_coupled_cable_scene(ranges, seed=21, **COUPLED_VBD_SCENE_KW)
    assert fs.params_fingerprint(a.params) == fs.params_fingerprint(b.params)


def test_measure_fruiting_forces_works_on_coupled_scene():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_coupled_cable_scene(
        ranges, seed=3, device="cpu", **COUPLED_VBD_SCENE_KW
    )
    fs.run_rollout(scene, num_steps=2, sim_substeps=5)
    q_prev = scene.state_1.body_q
    out = fs.measure_fruiting_forces(scene, scene.state_0.body_q, q_prev, dt=1.0e-3)
    assert len(out["fixed_joints"]) == len(scene.fruiting_fixed_joints)
