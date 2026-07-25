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


def test_build_plan_forces_surface_and_invariant():
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
    assert plan.apple_shift_m == pytest.approx(0.06)
    np.testing.assert_allclose(plan.weld_direction, (0.0, 1.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(plan.apple_pos_welded, (0.0, 0.06, 0.0), atol=1e-6)
    aw = np.asarray(plan.apple_pos_welded)
    wdir = np.asarray(plan.weld_direction)
    np.testing.assert_allclose(aw + r * wdir, tcp, atol=1e-6)
    msgs = " ".join(str(x.message).lower() for x in caught)
    assert "radius" in msgs or "tcp" in msgs


def test_no_warn_when_within_tol_and_aligned():
    # TCP +Z = +Y, apple→TCP = +Y, d=r=0.04
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=float)
    # columns are axes: +Z = (0,1,0)
    assert np.allclose(R[:, 2], (0.0, 1.0, 0.0))
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
    assert plan.tcp_approach_dot_weld == pytest.approx(1.0, abs=1e-6)
    assert caught == []


def test_warn_tcp_plus_z_misaligned_with_weld():
    # Identity R: +Z = (0,0,1); apple→TCP = +Y
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = build_post_grasp_plan(
            tcp_pose_4x4=_pose4([0.0, 0.04, 0.0]),
            apple_pose_4x4=_pose4([0.0, 0.0, 0.0]),
            apple_radius_m=0.04,
            warn_tol_m=0.02,
            approach_align_min_dot=0.9,
        )
    assert abs(plan.tcp_approach_dot_weld) < 0.1
    assert any("approach" in str(x.message).lower() or "+z" in str(x.message).lower() for x in caught)


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
    assert plan.tcp_radius_residual_m == pytest.approx(0.0181, abs=5e-4)
    assert any(True for _ in caught)  # expect radius and/or +Z warns
    aw = np.asarray(plan.apple_pos_welded)
    w = np.asarray(plan.weld_direction)
    np.testing.assert_allclose(aw + r * w, plan.tcp_pos, atol=1e-6)


def test_apply_post_grasp_after_settle_look_at():
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
    # Place TCP along +Y from apple at exactly r
    r = float(free.params.apple_radius)
    tcp = apple0 + np.array([0.0, r, 0.0], dtype=np.float64)
    # TCP +Z = +Y for alignment
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=float)
    M_tcp = np.eye(4)
    M_tcp[:3, :3] = R
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
    np.testing.assert_allclose(wbq[welded.apple_body, :3], plan.apple_pos_welded, atol=1e-4)
    np.testing.assert_allclose(wbq[welded.gripper_proxy_body, :3], plan.tcp_pos, atol=5e-3)
    if spur0 is not None and free.spur_bodies[0] < wbq.shape[0]:
        np.testing.assert_allclose(wbq[free.spur_bodies[0], :3], spur0, atol=1e-4)
    # apple→proxy along ŵ
    chord = wbq[welded.gripper_proxy_body, :3] - wbq[welded.apple_body, :3]
    chord /= np.linalg.norm(chord)
    np.testing.assert_allclose(chord, plan.weld_direction, atol=5e-2)
