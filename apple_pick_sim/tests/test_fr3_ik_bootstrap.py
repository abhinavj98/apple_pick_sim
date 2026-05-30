"""Tests for FR3 TCP IK bootstrap convergence reporting."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from apple_pick_sim.robot.fr3_robot.placement import (
    IK_BOOTSTRAP_POS_TOL_M,
    IK_BOOTSTRAP_ROT_TOL_RAD,
    IKBootstrapConvergenceError,
    IKBootstrapConvergenceWarning,
    placement_xform_for_proxy,
    raise_if_ik_bootstrap_not_converged,
    tcp_proxy_pose_errors,
    warn_if_ik_bootstrap_not_converged,
)


def test_placement_xform_for_proxy_is_origin():
    import warp as wp

    proxy_q7 = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    tf = placement_xform_for_proxy(proxy_q7, vertical_reach_m=0.85)
    pos = wp.transform_get_translation(tf)
    assert float(pos[0]) == 0.0
    assert float(pos[1]) == 0.0
    assert float(pos[2]) == 0.0


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
