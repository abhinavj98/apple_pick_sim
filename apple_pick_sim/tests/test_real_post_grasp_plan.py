# apple_pick_sim/tests/test_real_post_grasp_plan.py
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from apple_pick_sim.system_id.real_post_grasp_plan import (
    build_post_grasp_plan,
    pose_4x4_to_pos_quat,
    post_grasp_plan_from_metadata,
)


def _pose4(pos, R=None):
    if R is None:
        R = np.eye(3)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = pos
    return M.reshape(16).tolist()


def test_pose_4x4_to_pos_quat_identity():
    pos, quat = pose_4x4_to_pos_quat(_pose4([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(pos, (1.0, 2.0, 3.0))
    np.testing.assert_allclose(quat, (0.0, 0.0, 0.0, 1.0), atol=1e-6)


def test_build_plan_follows_measured_poses_no_surface_snap():
    tcp = np.array([0.0, 0.1, 0.0])
    apple_m = np.array([0.0, 0.0, 0.0])
    r = 0.04
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = build_post_grasp_plan(
            tcp_pose_4x4=_pose4(tcp.tolist()),
            apple_pose_4x4=_pose4(apple_m.tolist()),
            apple_radius_m=r,
            warn_tol_m=0.02,
        )
    assert plan.tcp_apple_distance_m == pytest.approx(0.1)
    assert plan.tcp_radius_residual_m == pytest.approx(0.06)
    assert plan.apple_shift_m == pytest.approx(0.0)
    np.testing.assert_allclose(plan.weld_direction, (0.0, 1.0, 0.0), atol=1e-6)
    # Follow data: welded apple = measured apple (no catalog r snap).
    np.testing.assert_allclose(plan.apple_pos_welded, apple_m, atol=1e-9)
    np.testing.assert_allclose(plan.apple_pos_measured, apple_m, atol=1e-9)
    np.testing.assert_allclose(plan.tcp_pos, tcp, atol=1e-9)
    msgs = " ".join(str(x.message).lower() for x in caught)
    assert "radius" in msgs or "tcp" in msgs
    assert "catalog-surface" not in msgs and "apple_welded−apple_meas" not in msgs


def test_no_warn_when_within_tol_and_aligned():
    # EE +Z points out tip toward apple: TCP→apple = −Y when tcp at +Y, apple at origin.
    # Columns: +Z = (0, -1, 0)
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=float)
    assert np.allclose(R[:, 2], (0.0, -1.0, 0.0))
    tcp = [0.0, 0.04, 0.0]
    apple = [0.0, 0.0, 0.0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = build_post_grasp_plan(
            tcp_pose_4x4=_pose4(tcp, R=R),
            apple_pose_4x4=_pose4(apple),
            apple_radius_m=0.04,
            warn_tol_m=0.02,
            approach_align_min_dot=0.9,
        )
    assert plan.tcp_radius_residual_m == pytest.approx(0.0, abs=1e-9)
    assert plan.apple_shift_m == pytest.approx(0.0, abs=1e-9)
    # tcp_approach_dot_weld = +Z · unit(apple − tcp)
    assert plan.tcp_approach_dot_weld == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(plan.apple_pos_welded, apple, atol=1e-9)
    assert caught == []


def test_warn_tcp_plus_z_misaligned_with_weld():
    # Identity R: +Z = (0,0,1); TCP→apple = −Y — diagnostic only; quat still used.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = build_post_grasp_plan(
            tcp_pose_4x4=_pose4([0.0, 0.04, 0.0]),
            apple_pose_4x4=_pose4([0.0, 0.0, 0.0]),
            apple_radius_m=0.04,
            warn_tol_m=0.02,
            approach_align_min_dot=0.9,
        )
    assert plan.tcp_approach_dot_weld == pytest.approx(0.0, abs=1e-6)
    msgs = [str(x.message).lower() for x in caught]
    assert any("tcp→apple" in m or "approach" in m or "+z" in m or "aligned" in m for m in msgs)
    assert not any("ignore" in m or "look-at" in m for m in msgs)
    np.testing.assert_allclose(plan.tcp_quat_xyzw, (0.0, 0.0, 0.0, 1.0), atol=1e-6)


def test_warn_when_plus_z_anti_aligned_with_tcp_to_apple():
    # Old wrong convention: +Z along apple→TCP (+Y) while tip should face apple (−Y).
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=float)
    assert np.allclose(R[:, 2], (0.0, 1.0, 0.0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = build_post_grasp_plan(
            tcp_pose_4x4=_pose4([0.0, 0.04, 0.0], R=R),
            apple_pose_4x4=_pose4([0.0, 0.0, 0.0]),
            apple_radius_m=0.04,
            warn_tol_m=0.02,
            approach_align_min_dot=0.9,
        )
    assert plan.tcp_approach_dot_weld == pytest.approx(-1.0, abs=1e-6)
    assert any("poorly aligned" in str(x.message).lower() for x in caught)


def test_warn_pose_translation_mismatch():
    tcp_pose = _pose4([0.0, 0.04, 0.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_post_grasp_plan(
            tcp_pose_4x4=tcp_pose,
            apple_pose_4x4=_pose4([0.0, 0.0, 0.0]),
            apple_radius_m=0.04,
            tcp_pos_override=(0.01, 0.04, 0.0),  # disagrees with pose t
            warn_tol_m=0.02,
            pose_pos_match_tol_m=1e-4,
        )
    assert any("tcp_pos" in str(x.message).lower() for x in caught)


@pytest.mark.parametrize("parquet", [Path("robot_replay/s00-d00.parquet")])
def test_s00_d00_plan_smoke(parquet: Path):
    if not parquet.is_file():
        pytest.skip("missing parquet")
    from apple_pick_sim.system_id.real_pre_grasp_params import load_dataset_metadata

    meta = load_dataset_metadata(parquet)
    r = float(meta["pre_grasp_geometry"]["parts"]["apple"]["radius_m"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = post_grasp_plan_from_metadata(meta, apple_radius_m=r, warn_tol_m=0.02)
    assert plan.tcp_radius_residual_m == pytest.approx(
        abs(plan.tcp_apple_distance_m - r), abs=1e-9
    )
    assert plan.apple_shift_m == pytest.approx(0.0, abs=1e-9)
    msgs = " ".join(str(x.message).lower() for x in caught)
    assert "ignore" not in msgs and "look-at" not in msgs
    assert "catalog-surface" not in msgs
    # Follow data: apple welded pose equals measured pose.
    np.testing.assert_allclose(plan.apple_pos_welded, plan.apple_pos_measured, atol=1e-9)
    # EE +Z out tip toward apple: +Z · unit(apple−tcp) should be strongly positive on this dump.
    assert plan.tcp_approach_dot_weld > 0.85
    # Logged TCP quat is present on the plan (true SE(3) contract).
    assert len(plan.tcp_quat_xyzw) == 4
    assert abs(float(np.linalg.norm(plan.tcp_quat_xyzw)) - 1.0) < 1e-5


def test_apply_post_grasp_after_settle_true_tcp_pose():
    from apple_pick_sim.fruiting_system import GripperProxyConfig, generate_coupled_cable_scene, load_ranges
    from apple_pick_sim.system_id.real_post_grasp_plan import apply_post_grasp_after_settle
    from apple_pick_sim.tests.conftest import NO_SELF_COLLISION_KW

    ranges = load_ranges(
        Path("apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json")
    )
    free = generate_coupled_cable_scene(
        ranges,
        seed=0,
        gripper_proxy=GripperProxyConfig(fix_to_apple=False),
        **NO_SELF_COLLISION_KW,
    )
    assert free.apple_body is not None
    bq0 = free.state_0.body_q.numpy().reshape(-1, 7)
    apple0 = bq0[free.apple_body, :3].copy()
    r = float(free.params.apple_radius)
    # Intentionally place TCP off the catalog surface (d != r).
    tcp = apple0 + np.array([0.0, r + 0.03, 0.0], dtype=np.float64)
    M_tcp = np.eye(4)
    M_tcp[:3, 3] = tcp
    M_ap = np.eye(4)
    M_ap[:3, 3] = apple0
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        plan = build_post_grasp_plan(
            tcp_pose_4x4=M_tcp.reshape(16).tolist(),
            apple_pose_4x4=M_ap.reshape(16).tolist(),
            apple_radius_m=r,
            warn_tol_m=0.02,
        )
    assert plan.apple_shift_m == pytest.approx(0.0, abs=1e-9)
    spur0 = bq0[free.spur_bodies[0], :3].copy() if free.spur_bodies else None

    welded = apply_post_grasp_after_settle(
        free,
        plan,
        ranges=ranges,
        params=free.params,
        base_pos=(0.5, 0.5, 0.5),
        robot_base_pos=(0.0, 0.0, 0.0),
    )
    wbq = welded.state_0.body_q.numpy().reshape(-1, 7)
    np.testing.assert_allclose(wbq[welded.apple_body, :3], plan.apple_pos_measured, atol=1e-4)
    np.testing.assert_allclose(wbq[welded.gripper_proxy_body, :3], plan.tcp_pos, atol=5e-3)
    pq = wbq[welded.gripper_proxy_body, 3:7]
    tq = np.asarray(plan.tcp_quat_xyzw, dtype=np.float64)
    q_dot = abs(float(np.dot(pq, tq)))
    assert q_dot > 1.0 - 1e-4, f"proxy quat {pq} vs tcp {tq} (abs-dot={q_dot})"
    if spur0 is not None and free.spur_bodies[0] < wbq.shape[0]:
        np.testing.assert_allclose(wbq[free.spur_bodies[0], :3], spur0, atol=1e-4)
    # Chord follows measured apple→TCP (no surface snap).
    chord = wbq[welded.gripper_proxy_body, :3] - wbq[welded.apple_body, :3]
    chord /= np.linalg.norm(chord)
    np.testing.assert_allclose(chord, plan.weld_direction, atol=5e-2)

    # USD tip-out: distal tip at body origin; cylinder bulk along local −Z (flange side).
    import newton
    import warp as wp

    model = welded.model
    proxy_body = welded.gripper_proxy_body
    shape_body = model.shape_body.numpy()
    shape_type = model.shape_type.numpy()
    shape_xform = model.shape_transform.numpy().reshape(-1, 7)
    cyl = [
        i
        for i in range(model.shape_count)
        if int(shape_body[i]) == proxy_body and int(shape_type[i]) == newton.GeoType.CYLINDER
    ]
    assert len(cyl) == 1
    hh = float(welded.gripper_proxy_config.cylinder_half_height)
    assert float(shape_xform[cyl[0], 2]) == pytest.approx(-hh)
    pq = wbq[welded.gripper_proxy_body]
    tip = pq[:3]
    q = wp.quat(float(pq[3]), float(pq[4]), float(pq[5]), float(pq[6]))
    flange = tip + np.asarray(wp.quat_rotate(q, wp.vec3(0.0, 0.0, -2.0 * hh)), dtype=np.float64)
    apple = wbq[welded.apple_body, :3]
    assert float(np.linalg.norm(flange - apple)) > float(np.linalg.norm(tip - apple))


def test_apply_post_grasp_keeps_settle_apple_orientation():
    """keep_apple_settle_orientation: snap apple pos only; ignore logged apple quat."""
    from apple_pick_sim.fruiting_system import GripperProxyConfig, generate_coupled_cable_scene, load_ranges
    from apple_pick_sim.system_id.real_post_grasp_plan import apply_post_grasp_after_settle
    from apple_pick_sim.tests.conftest import NO_SELF_COLLISION_KW

    ranges = load_ranges(
        Path("apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json")
    )
    free = generate_coupled_cable_scene(
        ranges,
        seed=0,
        gripper_proxy=GripperProxyConfig(fix_to_apple=False),
        **NO_SELF_COLLISION_KW,
    )
    assert free.apple_body is not None
    apple_id = int(free.apple_body)
    bq = free.state_0.body_q.numpy().reshape(-1, 7).astype(np.float32).copy()
    apple0 = bq[apple_id, :3].copy()
    # 90° about Z — distinct from logged identity quat below.
    settle_quat = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32)
    bq[apple_id, 0:3] = apple0 + np.array([0.01, -0.02, 0.005], dtype=np.float32)
    bq[apple_id, 3:7] = settle_quat
    zeros = np.zeros((bq.shape[0], 6), dtype=np.float32)
    free.state_0.body_q.assign(bq)
    free.state_0.body_qd.assign(zeros)
    free.state_1.body_q.assign(bq)
    free.state_1.body_qd.assign(zeros)

    r = float(free.params.apple_radius)
    tcp = apple0 + np.array([0.0, r + 0.03, 0.0], dtype=np.float64)
    M_tcp = np.eye(4)
    M_tcp[:3, 3] = tcp
    # Logged apple pose: measured center + identity rotation (≠ settle quat).
    M_ap = np.eye(4)
    M_ap[:3, 3] = apple0
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        plan = build_post_grasp_plan(
            tcp_pose_4x4=M_tcp.reshape(16).tolist(),
            apple_pose_4x4=M_ap.reshape(16).tolist(),
            apple_radius_m=r,
            warn_tol_m=0.02,
        )
    np.testing.assert_allclose(plan.apple_quat_xyzw, (0.0, 0.0, 0.0, 1.0), atol=1e-6)

    welded = apply_post_grasp_after_settle(
        free,
        plan,
        ranges=ranges,
        params=free.params,
        base_pos=(0.5, 0.5, 0.5),
        robot_base_pos=(0.0, 0.0, 0.0),
        keep_apple_settle_orientation=True,
    )
    wbq = welded.state_0.body_q.numpy().reshape(-1, 7)
    np.testing.assert_allclose(wbq[welded.apple_body, :3], plan.apple_pos_welded, atol=1e-4)
    aq = wbq[welded.apple_body, 3:7]
    # Same rotation up to sign flip of the quaternion.
    assert abs(float(np.dot(aq, settle_quat))) > 1.0 - 1e-4, (
        f"apple quat {aq} should keep settle {settle_quat}, not logged {plan.apple_quat_xyzw}"
    )
    np.testing.assert_allclose(wbq[welded.gripper_proxy_body, :3], plan.tcp_pos, atol=5e-3)
    pq = wbq[welded.gripper_proxy_body, 3:7]
    tq = np.asarray(plan.tcp_quat_xyzw, dtype=np.float64)
    assert abs(float(np.dot(pq, tq))) > 1.0 - 1e-4


def test_apply_post_grasp_stem_apple_joint_keeps_pre_grasp_apple_frame():
    """Welded rebuild bakes stem–apple child local from params.apple_quat (pre-grasp frame)."""
    import dataclasses
    import math

    import warp as wp

    from apple_pick_sim.fruiting_system import GripperProxyConfig, generate_coupled_cable_scene, load_ranges
    from apple_pick_sim.system_id.real_post_grasp_plan import apply_post_grasp_after_settle
    from apple_pick_sim.tests.conftest import NO_SELF_COLLISION_KW

    ranges = load_ranges(
        Path("apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json")
    )
    s = math.sqrt(0.5)
    pre_quat = (0.0, 0.0, s, s)
    free = generate_coupled_cable_scene(
        ranges,
        seed=0,
        gripper_proxy=GripperProxyConfig(fix_to_apple=False),
        **NO_SELF_COLLISION_KW,
    )
    # Re-build with pre-grasp apple frame on params (identity-sample fixture has no quat).
    params = dataclasses.replace(free.params, apple_quat_xyzw=pre_quat)
    free = generate_coupled_cable_scene(
        ranges,
        seed=0,
        params=params,
        gripper_proxy=GripperProxyConfig(fix_to_apple=False),
        **NO_SELF_COLLISION_KW,
    )
    assert free.apple_body is not None
    assert free.params.apple_quat_xyzw is not None
    bq0 = free.state_0.body_q.numpy().reshape(-1, 7)
    apple0 = bq0[free.apple_body, :3].copy()
    r = float(free.params.apple_radius)
    tcp = apple0 + np.array([0.0, r + 0.02, 0.0], dtype=np.float64)
    M_tcp = np.eye(4)
    M_tcp[:3, 3] = tcp
    # Post-grasp apple: same center, identity rotation (≠ pre-grasp 90° Z).
    M_ap = np.eye(4)
    M_ap[:3, 3] = apple0
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        plan = build_post_grasp_plan(
            tcp_pose_4x4=M_tcp.reshape(16).tolist(),
            apple_pose_4x4=M_ap.reshape(16).tolist(),
            apple_radius_m=r,
            warn_tol_m=0.05,
        )
    welded = apply_post_grasp_after_settle(
        free,
        plan,
        ranges=ranges,
        params=params,
        base_pos=(0.5, 0.5, 0.5),
        robot_base_pos=(0.0, 0.0, 0.0),
    )
    # Runtime apple pose is post-grasp (identity).
    aq = welded.state_0.body_q.numpy()[int(welded.apple_body), 3:7]
    assert abs(float(np.dot(aq, (0.0, 0.0, 0.0, 1.0)))) > 1.0 - 1e-4
    # Joint child local still from pre-grasp frame on params.
    labels = welded.model.joint_label
    ji = next(i for i, lab in enumerate(labels) if str(lab).endswith("_apple"))
    child_local = np.asarray(welded.model.joint_X_c.numpy()[ji, :3], dtype=np.float64)
    stem_dir = np.asarray(params.stem.direction, dtype=np.float64)
    stem_dir /= np.linalg.norm(stem_dir)
    world_attach = -r * stem_dir
    q = wp.quat(*pre_quat)
    expected = np.asarray(
        wp.quat_rotate(wp.quat_inverse(q), wp.vec3(*world_attach.tolist())),
        dtype=np.float64,
    )
    np.testing.assert_allclose(child_local, expected, atol=1e-4)
