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
from apple_pick_sim.robot.fr3_robot.fr3_v21_props import load_fr3_v21_inertials, resolve_fr3_link_body_index
from apple_pick_sim.robot.fr3_robot.setup import FR3_REFLECTED_MOTOR_INERTIA_KGM2
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
def test_batched_configure_tiles_fr3_reflected_motor_inertia():
    scene = _build_batched_scene()
    _configure_batched_vic(scene)
    expected = np.asarray(FR3_REFLECTED_MOTOR_INERTIA_KGM2, dtype=np.float64)
    dof_per = int(scene.layout.joint_dof_count_per_world)
    arr = scene.robot_model.joint_armature.numpy().reshape(-1)
    for world in range(_NUM_ENVS):
        sl = slice(world * dof_per, world * dof_per + _N_ARM_DOF)
        np.testing.assert_allclose(arr[sl].astype(np.float64), expected, rtol=0.0, atol=1e-6)

    link1 = next(link for link in load_fr3_v21_inertials() if link.link_num == 1)
    tpl_link1 = resolve_fr3_link_body_index(scene.robot_model, 1)
    local = tpl_link1 % int(scene.layout.robot_bodies_per_world)
    masses = scene.robot_model.body_mass.numpy()
    for world in range(_NUM_ENVS):
        idx = world * int(scene.layout.robot_bodies_per_world) + local
        assert float(masses[idx]) == pytest.approx(link1.mass_kg, abs=1e-5)


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
def test_batched_vic_from_actions_holds_setpoint_when_arm_sags():
    """Zero actions must not re-anchor the integrated target to a sagged actual pose."""
    scene = _build_batched_scene()
    ctrl = _configure_batched_vic(scene)
    anchor = ctrl._target_pos_wp.numpy().copy()

    bq = scene.robot_state_0.body_q.numpy().reshape(-1, 7).copy()
    layout = scene.layout
    for w in range(_NUM_ENVS):
        tcp_idx = int(layout.tcp_body_indices[w])
        bq[tcp_idx, 2] -= 0.1
    scene.robot_state_0.body_q.assign(bq.reshape(-1))

    import torch

    actions = torch.zeros(_NUM_ENVS, 6, dtype=torch.float32, device="cpu")
    ctrl.run_coupled_teleop_frame_from_actions(
        scene.robot_state_0,
        scene.robot_control,
        scene.mj_solver,
        1.0 / 15.0,
        actions,
    )

    pos = ctrl._target_pos_wp.numpy()
    for w in range(_NUM_ENVS):
        np.testing.assert_allclose(
            pos[w],
            anchor[w],
            rtol=0,
            atol=1e-6,
            err_msg=f"world {w} target re-anchored to sagged TCP",
        )


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


@requires_fr3
@pytest.mark.slow
def test_aniso_wrench_used_when_gain_buffers_staged():
    """Per-env anisotropic Kp/Kd buffers override isotropic ImpedanceGains."""
    scene = _build_batched_scene()
    ctrl = _configure_batched_vic(scene)
    dev = scene.robot_model.device

    pos = ctrl._target_pos_wp.numpy().copy()
    pos[:, 0] += 0.1
    ctrl._target_pos_wp.assign(pos.astype(np.float32))
    ctrl.stage_targets_to_scene(scene)

    scene.vic_kp_lin_wp = wp.full(_NUM_ENVS, wp.vec3(1000.0, 0.0, 0.0), dtype=wp.vec3, device=dev)
    scene.vic_kp_ang_wp = wp.zeros(_NUM_ENVS, dtype=wp.vec3, device=dev)
    scene.vic_kd_lin_wp = wp.zeros(_NUM_ENVS, dtype=wp.vec3, device=dev)
    scene.vic_kd_ang_wp = wp.zeros(_NUM_ENVS, dtype=wp.vec3, device=dev)

    launch_compute_vic_wrenches_batched(scene)
    wp.synchronize()
    wrenches = scene.vic_jt_wrench_buf.numpy()
    for w in range(_NUM_ENVS):
        assert abs(float(wrenches[w, 0]) - 100.0) < 1.0, f"world {w} expected Kp_x*0.1=100"
        assert abs(float(wrenches[w, 1])) < 1e-3, f"world {w} expected zero on y (kp_lin_y=0)"


@requires_fr3
@pytest.mark.slow
def test_isotropic_path_unchanged_without_gain_buffers():
    """No aniso buffers staged -> existing isotropic kernel path still used (regression guard)."""
    scene = _build_batched_scene()
    ctrl = _configure_batched_vic(scene)
    assert getattr(scene, "vic_kp_lin_wp", None) is None
    pos = ctrl._target_pos_wp.numpy().copy()
    pos[:, 0] += 0.05
    ctrl._target_pos_wp.assign(pos.astype(np.float32))
    ctrl.stage_targets_to_scene(scene)
    launch_compute_vic_wrenches_batched(scene)
    wp.synchronize()
    wrenches = scene.vic_jt_wrench_buf.numpy()
    for w in range(_NUM_ENVS):
        assert float(np.linalg.norm(wrenches[w, :3])) > 1.0
