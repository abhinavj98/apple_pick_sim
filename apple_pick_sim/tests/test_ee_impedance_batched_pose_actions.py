"""Tests for Fr3BatchedEEImpedanceController pose-action unpack (vic_pose)."""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from apple_pick_sim.fruiting_system import load_ranges
from apple_pick_sim.robot import fr3_robot
from conftest import COUPLED_SCENE_KW, RANGES_FIXTURE, build_homogeneous_batched_fr3, requires_fr3

_NUM_ENVS = 2


def _build_ctrl():
    ranges = load_ranges(RANGES_FIXTURE)
    scene = build_homogeneous_batched_fr3(
        ranges, 42, device="cpu", num_envs=_NUM_ENVS, **COUPLED_SCENE_KW
    )
    ik_kw = fr3_robot.batched_ik_teleop_kwargs(scene)
    ctrl = fr3_robot.Fr3BatchedEEImpedanceController(scene.robot_model, **ik_kw)
    return scene, ctrl


def _pose_action_row(pos, quat_wxyz, kp, kd):
    return list(pos) + list(quat_wxyz) + list(kp) + list(kd)


@requires_fr3
@pytest.mark.slow
def test_unpack_pose_action_sets_targets_directly():
    scene, ctrl = _build_ctrl()
    row0 = _pose_action_row((0.3, 0.4, 0.5), (1.0, 0.0, 0.0, 0.0), [1.0] * 6, [2.0] * 6)
    row1 = _pose_action_row((0.1, 0.2, 0.3), (1.0, 0.0, 0.0, 0.0), [3.0] * 6, [4.0] * 6)
    actions = torch.tensor([row0, row1], dtype=torch.float32)
    ctrl.unpack_pose_action(actions)
    pos = ctrl._target_pos_wp.numpy()
    np.testing.assert_allclose(pos[0], [0.3, 0.4, 0.5], atol=1e-6)
    np.testing.assert_allclose(pos[1], [0.1, 0.2, 0.3], atol=1e-6)
    lin = ctrl._lin_vels_wp.numpy()
    ang = ctrl._ang_vels_wp.numpy()
    assert np.allclose(lin, 0.0)
    assert np.allclose(ang, 0.0)


@requires_fr3
@pytest.mark.slow
def test_unpack_pose_action_near_zero_quat_defaults_to_identity():
    scene, ctrl = _build_ctrl()
    row = _pose_action_row((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), [0.0] * 6, [0.0] * 6)
    actions = torch.tensor([row, row], dtype=torch.float32)
    ctrl.unpack_pose_action(actions)
    rot = ctrl._target_rot_wp.numpy()  # stored xyzw (Warp-native); identity is (0,0,0,1)
    for w in range(_NUM_ENVS):
        np.testing.assert_allclose(rot[w], [0.0, 0.0, 0.0, 1.0], atol=1e-6)


@requires_fr3
@pytest.mark.slow
def test_unpack_pose_action_converts_wxyz_to_warp_xyzw():
    """A 90 deg yaw action quat (wxyz) must land as the matching xyzw in Warp storage."""
    scene, ctrl = _build_ctrl()
    c = float(np.cos(np.pi / 4.0))  # cos(45 deg) = sin(45 deg)
    row = _pose_action_row((0.1, 0.2, 0.3), (c, 0.0, 0.0, c), [1.0] * 6, [1.0] * 6)
    actions = torch.tensor([row, row], dtype=torch.float32)
    ctrl.unpack_pose_action(actions)
    rot = ctrl._target_rot_wp.numpy()
    for w in range(_NUM_ENVS):
        np.testing.assert_allclose(rot[w], [0.0, 0.0, c, c], atol=1e-6)

    import warp as wp

    q0 = wp.transform_get_rotation(ctrl.target_tf[0])
    np.testing.assert_allclose([q0[0], q0[1], q0[2], q0[3]], [0.0, 0.0, c, c], atol=1e-6)


@requires_fr3
@pytest.mark.slow
def test_stage_pose_gains_to_scene_wires_buffers():
    scene, ctrl = _build_ctrl()
    row = _pose_action_row((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), [5.0, 6.0, 7.0, 8.0, 9.0, 10.0], [1.0] * 6)
    actions = torch.tensor([row, row], dtype=torch.float32)
    ctrl.unpack_pose_action(actions)
    ctrl.stage_pose_gains_to_scene(scene)
    assert scene.vic_kp_lin_wp is not None
    kp_lin = scene.vic_kp_lin_wp.numpy()
    np.testing.assert_allclose(kp_lin[0], [5.0, 6.0, 7.0], atol=1e-6)
    kp_ang = scene.vic_kp_ang_wp.numpy()
    np.testing.assert_allclose(kp_ang[0], [8.0, 9.0, 10.0], atol=1e-6)


@requires_fr3
@pytest.mark.slow
def test_run_coupled_teleop_frame_from_pose_actions_syncs_target_tf():
    scene, ctrl = _build_ctrl()
    row = _pose_action_row((0.7, 0.8, 0.9), (1.0, 0.0, 0.0, 0.0), [1.0] * 6, [1.0] * 6)
    actions = torch.tensor([row, row], dtype=torch.float32)
    ctrl.run_coupled_teleop_frame_from_pose_actions(
        scene.robot_state_0, scene.robot_control, scene.mj_solver, 1.0 / 15.0, actions
    )
    import warp as wp

    p0 = wp.transform_get_translation(ctrl.target_tf[0])
    np.testing.assert_allclose([p0[0], p0[1], p0[2]], [0.7, 0.8, 0.9], atol=1e-6)
