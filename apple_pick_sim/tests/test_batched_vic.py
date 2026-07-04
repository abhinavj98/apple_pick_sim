"""Tests for batched VIC joint-torque path (vectorized wrench + J^T Λ w)."""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import numpy as np
import pytest
import warp as wp

pytest.importorskip("torch")

from apple_pick_sim.coupled_fruiting.vic_joint_torques_batched import (
    launch_apply_vic_joint_torques_batched,
    launch_compute_vic_wrenches_batched,
)
from apple_pick_sim.fruiting_system import load_ranges
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import ImpedanceGains
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance_batched import (
    Fr3BatchedEEImpedanceController,
)
from apple_pick_sim.robot.fr3_robot.controllers.keyboard import EEVelocity
from conftest import COUPLED_SCENE_KW, RANGES_FIXTURE, build_homogeneous_batched_fr3, requires_fr3

_NUM_ENVS = 2
_N_ARM_DOF = 7


def _build_batched_scene():
    ranges = load_ranges(RANGES_FIXTURE)
    return build_homogeneous_batched_fr3(
        ranges,
        42,
        device="cpu",
        num_envs=_NUM_ENVS,
        **COUPLED_SCENE_KW,
    )


def _configure_batched_vic(scene) -> Fr3BatchedEEImpedanceController:
    ik_kw = fr3_robot.batched_ik_teleop_kwargs(scene)
    assert ik_kw
    scene.robot_kinematic_mode = False
    scene.vic_use_joint_torques = True
    ctrl = Fr3BatchedEEImpedanceController(
        scene.robot_model,
        linear_speed=0.2,
        angular_speed=1.0,
        **ik_kw,
    )
    scene.vic_controller = ctrl
    scene.vic_gains = ImpedanceGains(linear_k=800.0, linear_d=80.0)
    fr3_robot.init_mujoco_actuator_targets_from_model(scene.robot_model, scene.robot_control)
    fr3_robot.configure_vic_joint_torques_arm_batched(
        scene.robot_model,
        scene.robot_state_0,
        scene.robot_control,
        scene.mj_solver,
        scene=scene,
        layout=scene.layout,
    )
    scene.vic_joint_torques_configured = True
    ctrl.sync_target_from_state(scene.robot_state_0)
    ctrl.stage_targets_to_scene(scene)
    scene.vic_target_twist = EEVelocity()
    return ctrl


@requires_fr3
@pytest.mark.slow
def test_buffers_shaped_for_num_envs():
    scene = _build_batched_scene()
    _configure_batched_vic(scene)
    assert scene.vic_jt_J_buf.shape[0] == _NUM_ENVS
    assert scene.vic_jt_H_buf.shape[0] == _NUM_ENVS
    assert int(scene.vic_jt_num_envs) == _NUM_ENVS


@requires_fr3
@pytest.mark.slow
def test_batched_wrench_kernel_nonzero_off_target():
    scene = _build_batched_scene()
    ctrl = _configure_batched_vic(scene)
    pos = ctrl._target_pos_wp.numpy().copy()
    pos[:, 0] += 0.05
    ctrl._target_pos_wp.assign(pos.astype(np.float32))
    ctrl.stage_targets_to_scene(scene)
    launch_compute_vic_wrenches_batched(scene)
    wp.synchronize()
    wrenches = scene.vic_jt_wrench_buf.numpy()
    for w in range(_NUM_ENVS):
        assert float(np.linalg.norm(wrenches[w, :3])) > 1.0, f"world {w} expected nonzero force"


@requires_fr3
@pytest.mark.slow
def test_batched_vic_writes_joint_f_all_envs():
    scene = _build_batched_scene()
    ctrl = _configure_batched_vic(scene)
    pos = ctrl._target_pos_wp.numpy().copy()
    pos[:, 0] += 0.05
    ctrl._target_pos_wp.assign(pos.astype(np.float32))
    ctrl.stage_targets_to_scene(scene)
    launch_apply_vic_joint_torques_batched(scene)
    wp.synchronize()
    layout = scene.layout
    dof_per = int(layout.joint_dof_count_per_world)
    jf = scene.robot_control.joint_f.numpy().reshape(_NUM_ENVS, dof_per)
    for w in range(_NUM_ENVS):
        assert float(np.linalg.norm(jf[w, :_N_ARM_DOF])) > 0.1, f"world {w} joint_f near zero"


@requires_fr3
@pytest.mark.slow
def test_independent_targets_give_independent_torques():
    scene = _build_batched_scene()
    ctrl = _configure_batched_vic(scene)
    pos = ctrl._target_pos_wp.numpy().copy()
    pos[0, 0] += 0.08
    pos[1, 1] += 0.08
    ctrl._target_pos_wp.assign(pos.astype(np.float32))
    ctrl.stage_targets_to_scene(scene)
    launch_apply_vic_joint_torques_batched(scene)
    wp.synchronize()
    layout = scene.layout
    dof_per = int(layout.joint_dof_count_per_world)
    jf = scene.robot_control.joint_f.numpy().reshape(_NUM_ENVS, dof_per)
    assert not np.allclose(jf[0, :_N_ARM_DOF], jf[1, :_N_ARM_DOF], atol=1e-3)


@requires_fr3
@pytest.mark.slow
def test_zero_wrench_at_target():
    scene = _build_batched_scene()
    ctrl = _configure_batched_vic(scene)
    ctrl.sync_target_from_state(scene.robot_state_0)
    ctrl.stage_targets_to_scene(scene)
    launch_apply_vic_joint_torques_batched(scene)
    wp.synchronize()
    wrenches = scene.vic_jt_wrench_buf.numpy()
    for w in range(_NUM_ENVS):
        assert float(np.linalg.norm(wrenches[w])) < 1e-3, f"world {w} nonzero wrench at target"


@requires_fr3
@pytest.mark.slow
def test_per_env_twist_affects_wrench_damping():
    """D-term uses per-env v_des; worlds with different twists get different wrenches at zero pose error."""
    scene = _build_batched_scene()
    ctrl = _configure_batched_vic(scene)
    ctrl.sync_target_from_state(scene.robot_state_0)

    lin = np.zeros((_NUM_ENVS, 3), dtype=np.float32)
    lin[0, 0] = 0.05
    lin[1, 0] = 0.0
    ctrl._lin_vels_wp.assign(lin)
    ang = np.zeros((_NUM_ENVS, 3), dtype=np.float32)
    ctrl._ang_vels_wp.assign(ang)
    ctrl.stage_targets_to_scene(scene)

    launch_compute_vic_wrenches_batched(scene)
    wp.synchronize()
    wrenches = scene.vic_jt_wrench_buf.numpy()
    assert float(wrenches[0, 0]) > 1.0, "world 0 expected D-term force from v_des"
    assert abs(float(wrenches[1, 0])) < 1e-3, "world 1 expected zero D-term at v_des=0"
    assert not np.allclose(wrenches[0, :3], wrenches[1, :3], atol=1e-3)
