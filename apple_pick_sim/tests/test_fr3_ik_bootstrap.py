"""Tests for FR3 TCP IK bootstrap convergence reporting."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from apple_pick_sim.robot.fr3_robot.placement import (
    IK_BOOTSTRAP_DEFAULT_MAX_SEEDS,
    IK_BOOTSTRAP_POS_TOL_M,
    IK_BOOTSTRAP_ROT_TOL_RAD,
    IKBootstrapConvergenceError,
    IKBootstrapConvergenceWarning,
    bootstrap_tcp_ik_from_proxy,
    ik_bootstrap_joint_q_candidates,
    placement_xform_for_proxy,
    raise_if_ik_bootstrap_not_converged,
    tcp_proxy_pose_errors,
    warn_if_ik_bootstrap_not_converged,
)

def test_placement_xform_for_proxy_offsets_xy_and_lowers_z_by_reach():
    import warp as wp

    proxy_q7 = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    tf = placement_xform_for_proxy(proxy_q7, vertical_reach_m=0.85)
    pos = wp.transform_get_translation(tf)
    assert float(pos[0]) == pytest.approx(1.0)
    assert float(pos[1]) == pytest.approx(2.0)
    assert float(pos[2]) == pytest.approx(2.15)


def test_tcp_proxy_pose_errors():
    rq = np.zeros((2, 7), dtype=np.float32)
    pq = np.zeros((2, 7), dtype=np.float32)
    rq[1, :3] = (1.0, 0.0, 0.0)
    rq[1, 6] = 1.0
    pq[1, :3] = (0.0, 0.0, 0.0)
    pq[1, 6] = 1.0

    pos_err, rot_err = tcp_proxy_pose_errors(rq, pq, tcp_body_index=1, proxy_body_index=1)
    assert pos_err == pytest.approx(1.0)
    assert rot_err == pytest.approx(0.0, abs=1e-5)


def test_warn_if_ik_bootstrap_not_converged_ok():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert warn_if_ik_bootstrap_not_converged(
            0.01,
            0.01,
            pos_tol_m=IK_BOOTSTRAP_POS_TOL_M,
            rot_tol_rad=IK_BOOTSTRAP_ROT_TOL_RAD,
        )


def test_warn_if_ik_bootstrap_not_converged_position():
    with pytest.warns(IKBootstrapConvergenceWarning, match="position error"):
        assert not warn_if_ik_bootstrap_not_converged(
            IK_BOOTSTRAP_POS_TOL_M + 0.01,
            0.0,
        )


def test_warn_if_ik_bootstrap_not_converged_rotation():
    with pytest.warns(IKBootstrapConvergenceWarning, match="orientation error"):
        assert not warn_if_ik_bootstrap_not_converged(
            0.0,
            IK_BOOTSTRAP_ROT_TOL_RAD + 0.01,
        )


def test_raise_if_ik_bootstrap_not_converged_ok():
    raise_if_ik_bootstrap_not_converged(0.01, 0.01)


def test_raise_if_ik_bootstrap_not_converged_raises():
    with pytest.raises(IKBootstrapConvergenceError, match="position error"):
        raise_if_ik_bootstrap_not_converged(
            IK_BOOTSTRAP_POS_TOL_M + 0.01,
            0.0,
            target_pos=(1.0, 2.0, 3.0),
        )


def test_enable_ik_bootstrap_warnings_for_examples():
    from apple_pick_sim.robot.fr3_robot.placement import enable_ik_bootstrap_warnings_for_examples

    enable_ik_bootstrap_warnings_for_examples()
    with pytest.warns(IKBootstrapConvergenceWarning):
        warn_if_ik_bootstrap_not_converged(IK_BOOTSTRAP_POS_TOL_M + 0.01, 0.0)


def test_ik_bootstrap_joint_q_candidates_include_model_default_and_midpoint():
    from apple_pick_sim.robot.fr3_robot.paths import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    from apple_pick_sim.robot.fr3_robot.setup import build_fr3_robot_model_from_usd

    robot_model, _, _ = build_fr3_robot_model_from_usd(device="cpu")
    jc = int(robot_model.joint_coord_count)
    default = robot_model.joint_q.numpy().reshape(-1)[:jc]
    lower = robot_model.joint_limit_lower.numpy().reshape(-1)[:jc]
    upper = robot_model.joint_limit_upper.numpy().reshape(-1)[:jc]
    midpoint = lower + 0.5 * (upper - lower)

    seeds = ik_bootstrap_joint_q_candidates(robot_model, max_seeds=4)
    assert len(seeds) == 4
    assert np.allclose(seeds[0], default)
    assert np.allclose(seeds[1], midpoint)
    assert not np.allclose(seeds[0], seeds[1])


def test_ik_bootstrap_joint_q_candidates_support_large_max_seeds():
    from apple_pick_sim.robot.fr3_robot.paths import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    from apple_pick_sim.robot.fr3_robot.setup import build_fr3_robot_model_from_usd

    robot_model, _, _ = build_fr3_robot_model_from_usd(device="cpu")
    seeds = ik_bootstrap_joint_q_candidates(robot_model, max_seeds=40)
    assert len(seeds) == 40
    for i, seed in enumerate(seeds):
        for j in range(i):
            assert not np.allclose(seeds[i], seeds[j], atol=1e-4, rtol=0.0)


def test_ik_bootstrap_default_max_seeds_exceeds_legacy_four():
    assert IK_BOOTSTRAP_DEFAULT_MAX_SEEDS > 4


def test_bootstrap_tcp_ik_retries_alternate_joint_q_when_first_seed_poor(capsys):
    from pathlib import Path

    from apple_pick_sim.robot.fr3_robot.paths import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    import apple_pick_sim.coupled_fruiting as cf
    import apple_pick_sim.fruiting_system as fs
    from apple_pick_sim.coupled_fruiting.builders import build_coupled_fruiting_fr3

    ranges_fixture = (
        Path(__file__).resolve().parent.parent / "fixtures" / "fruiting_system_ranges_straight_rod_test.json"
    )
    ranges = fs.load_ranges(ranges_fixture)
    scene = build_coupled_fruiting_fr3(
        ranges,
        7,
        enable_self_collisions=False,
        base_pos=(0.2, 0.2, 0.5),
        robot_base_from_proxy=True,
        ik_bootstrap_iterations=256,
        skip_ik_bootstrap=True,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    jc = int(scene.robot_model.joint_coord_count)
    poor_seed = scene.robot_model.joint_limit_upper.numpy().reshape(-1)[:jc].astype(
        scene.robot_model.joint_q.dtype
    )
    scene.robot_model.joint_q.assign(poor_seed)

    with pytest.raises(IKBootstrapConvergenceError):
        bootstrap_tcp_ik_from_proxy(
            scene.cable,
            scene.robot_model,
            scene.tcp_body_index,
            scene.robot_state_0,
            ik_iterations=48,
            max_joint_q_seeds=1,
            raise_on_failure=True,
        )

    bootstrap_tcp_ik_from_proxy(
        scene.cable,
        scene.robot_model,
        scene.tcp_body_index,
        scene.robot_state_0,
        ik_iterations=48,
        max_joint_q_seeds=4,
        raise_on_failure=True,
    )
    pos_err, rot_err = tcp_proxy_pose_errors(
        scene.robot_state_0.body_q.numpy(),
        scene.cable.state_0.body_q.numpy(),
        tcp_body_index=scene.tcp_body_index,
        proxy_body_index=scene.cable.gripper_proxy_body,
    )
    assert pos_err < IK_BOOTSTRAP_POS_TOL_M
    assert rot_err < IK_BOOTSTRAP_ROT_TOL_RAD

    out = capsys.readouterr().out
    assert "joint_q seed" in out
