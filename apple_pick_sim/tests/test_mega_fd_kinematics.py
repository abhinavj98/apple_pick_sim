"""Welded mega FD on straight-rod fixture (zero gravity): forces and Jacobians.

Uses ``fruiting_system_ranges_straight_rod_test.json`` (branch ≈ −Z). Robot teleop
deflects the apple; stem–apple wrench features (indices 6–11) carry stiffness sensitivity
when ``fix_to_apple`` welds proxy to apple.

Checks:
  * Stem wrench in FD features matches coupled gather (``state_1.body_q`` pre-step).
  * Restoring force opposes apple displacement on the driven axis.
  * Jacobian columns equal forward differences ``(y_i - y_0) / ε``.
  * Dominant stiffness column on the driven force row is resolved and nonzero.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from apple_pick_sim import coupled_fruiting as cf
from apple_pick_sim import fruiting_system as fs
from apple_pick_sim.fruiting_system.mega_fd import (
    MegaFdStepResult,
    default_mega_fd_features,
    extract_mega_fd_jacobian,
    mega_fd_step,
    reset_perturbed_instances_to_nominal,
)
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.tests.conftest import (
    DEFAULT_MJ_KW,
    RANGES_FIXTURE,
    SUB_DT,
    SUBSTEPS_PER_FRAME,
    apply_direct_hold,
    new_direct_controller,
    requires_fr3,
)

EPS = 0.02
MIN_DISP = 0.05
MIN_FORCE = 5.0
MIN_J_ABS = 0.05
MIN_J_REL_FORCE = 1.0e-4
SETTLE_HOLD_FRAMES = 20
DRIVE_HOLD_FRAMES = 50
POST_DRIVE_HOLD_FRAMES = 25
SEGMENTS = ("primary", "secondary", "spur", "stem")
FORCE_ROW_BASE = 6

LATERAL_DRIVE_CASES = (
    pytest.param(1, (0.0, 0.15, 0.0), id="y_pos"),
    pytest.param(1, (0.0, -0.15, 0.0), id="y_neg"),
    pytest.param(0, (-0.25, 0.0, 0.0), id="x_neg"),
    pytest.param(0, (0.25, 0.0, 0.0), id="x_pos"),
)


@pytest.fixture(scope="module", autouse=True)
def _warp_init():
    wp.init()


def _zero_gravity(scene: cf.MegaCoupledFruitingScene) -> None:
    scene.gravity_vec = wp.vec3(0.0, 0.0, 0.0)
    scene.cable.model.set_gravity((0.0, 0.0, 0.0))


def _build_welded_zero_g_scene() -> cf.MegaCoupledFruitingScene:
    gripper = fs.GripperProxyConfig(fix_to_apple=True)
    scene = cf.build_mega_coupled_fruiting_fr3(
        fs.load_ranges(RANGES_FIXTURE),
        seed=42,
        stiffness_epsilon=EPS,
        enable_self_collisions=False,
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
        gripper_proxy=gripper,
        stem_coupling_gain=1.0,
        stem_force_cap_N=None,
        stem_torque_cap_Nm=None,
    )
    _zero_gravity(scene)
    scene.robot_kinematic_mode = True
    return scene


def _hold_substeps(
    scene: cf.MegaCoupledFruitingScene,
    ctrl: fr3_robot.Fr3EEDirectJointController,
    velocity: fr3_robot.EEVelocity,
    *,
    n_frames: int,
) -> None:
    for _ in range(n_frames):
        apply_direct_hold(scene, fr3_robot, ctrl, velocity=velocity)
        for _ in range(SUBSTEPS_PER_FRAME):
            scene.coupled_substep(SUB_DT)


def _stem_apple_wrench_coupled_gather(scene: cf.MegaCoupledFruitingScene) -> np.ndarray:
    """Raw stem→apple wrench (same inputs as ``harvest_stem_tension_for_tcp``)."""
    from apple_pick_sim.vbd_fixed_joint_wrenches import fixed_joint_wrenches_child_com_vbd

    cable = scene.cable
    assert scene.stem_apple_joint_index is not None
    records = fixed_joint_wrenches_child_com_vbd(
        cable.model,
        cable.solver,
        body_q=cable.state_0.body_q,
        body_q_prev=cable.state_1.body_q,
        dt=SUB_DT,
        joint_pairs=[(scene.stem_apple_joint_index, "stem_apple")],
    )
    assert len(records) == 1
    rec = records[0]
    return np.concatenate(
        [rec.force_world.astype(np.float64), rec.torque_at_child_com_world.astype(np.float64)]
    )


def _assert_jacobian_matches_features(
    result: MegaFdStepResult,
    epsilon: float,
    *,
    nominal_index: int = 0,
) -> None:
    y0 = result.features[nominal_index]
    n = result.features.shape[0]
    jacobian_col = 0
    for col in range(n):
        if col == nominal_index:
            continue
        gold = (result.features[col] - y0) / epsilon
        seg = SEGMENTS[jacobian_col] if jacobian_col < len(SEGMENTS) else str(jacobian_col)
        np.testing.assert_allclose(
            result.jacobian[:, jacobian_col],
            gold,
            rtol=1e-5,
            atol=1e-4,
            err_msg=f"Jacobian column {jacobian_col} ({seg})",
        )
        jacobian_col += 1


def _drive_to_deflected_state(
    scene: cf.MegaCoupledFruitingScene,
    ctrl: fr3_robot.Fr3EEDirectJointController,
    linear: tuple[float, float, float],
) -> np.ndarray:
    """Teleop hold → drive → hold; return apple displacement (world)."""
    apple = scene.cable.instance(0).apple_body
    assert apple is not None
    hold_zero = fr3_robot.EEVelocity()
    _hold_substeps(scene, ctrl, hold_zero, n_frames=SETTLE_HOLD_FRAMES)
    p0 = scene.cable.state_0.body_q.numpy().reshape(-1, 7)[apple, :3].copy()
    _hold_substeps(
        scene,
        ctrl,
        fr3_robot.EEVelocity(linear=linear),
        n_frames=DRIVE_HOLD_FRAMES,
    )
    _hold_substeps(scene, ctrl, hold_zero, n_frames=POST_DRIVE_HOLD_FRAMES)
    p1 = scene.cable.state_0.body_q.numpy().reshape(-1, 7)[apple, :3]
    return p1 - p0


def _fd_after_drive(
    scene: cf.MegaCoupledFruitingScene,
    ctrl: fr3_robot.Fr3EEDirectJointController,
    linear: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MegaFdStepResult]:
    """Drive, reset columns, one ``mega_fd_step``; return disp, force, Jacobian, result."""
    disp = _drive_to_deflected_state(scene, ctrl, linear)
    reset_perturbed_instances_to_nominal(scene.cable)
    result = mega_fd_step(
        scene.cable,
        EPS,
        dt=SUB_DT,
        collision_pipeline=scene.cable_collision_pipeline,
    )
    force = result.features[0, FORCE_ROW_BASE : FORCE_ROW_BASE + 3]
    return disp, force, result.jacobian, result


@requires_fr3
def test_cable_model_has_zero_gravity_when_configured():
    scene = _build_welded_zero_g_scene()
    g_scene = np.array(scene.gravity_vec, dtype=np.float64)
    g_model = scene.cable.model.gravity.numpy().reshape(3)
    np.testing.assert_allclose(g_scene, 0.0, atol=0.0)
    np.testing.assert_allclose(g_model, 0.0, atol=0.0)


@requires_fr3
def test_fd_stem_wrench_matches_coupled_gather_after_drive():
    """Feature wrench block uses the same gather convention as stem harvest."""
    scene = _build_welded_zero_g_scene()
    ctrl = new_direct_controller(scene, fr3_robot)
    _drive_to_deflected_state(scene, ctrl, (0.0, 0.15, 0.0))

    gather = _stem_apple_wrench_coupled_gather(scene)
    feat = default_mega_fd_features(scene.cable, 0, dt=SUB_DT)
    assert feat.size == 12
    np.testing.assert_allclose(
        feat[FORCE_ROW_BASE : FORCE_ROW_BASE + 6],
        gather,
        rtol=2e-4,
        atol=2.0,
        err_msg="FD feature wrench should match coupled stem gather",
    )


@requires_fr3
@pytest.mark.parametrize("axis,linear", LATERAL_DRIVE_CASES)
def test_welded_restoring_force_and_fd_jacobian(axis: int, linear: tuple[float, float, float]):
    """Mega FD: stem wrench is large on the driven axis; Jacobian matches FD."""
    scene = _build_welded_zero_g_scene()
    ctrl = new_direct_controller(scene, fr3_robot)
    disp, force, jacobian, result = _fd_after_drive(scene, ctrl, linear)

    assert abs(disp[axis]) >= MIN_DISP, (
        f"apple displacement too small along axis {axis}: {disp[axis]}"
    )
    assert abs(force[axis]) >= MIN_FORCE, (
        f"stem–apple force too small along axis {axis}: {force[axis]}"
    )

    row = FORCE_ROW_BASE + axis
    j_row = jacobian[row, :]
    j_dom = int(np.argmax(np.abs(j_row)))
    j_entry = float(j_row[j_dom])
    assert abs(j_entry) >= MIN_J_ABS or abs(j_entry) >= MIN_J_REL_FORCE * abs(force[axis]), (
        f"Jacobian force row {row} insensitive to stiffness on {SEGMENTS[j_dom]}: {j_row}"
    )

    _assert_jacobian_matches_features(result, EPS)


def _stem_force_after_zero_g_drive(linear: tuple[float, float, float]) -> np.ndarray:
    """Fresh mega scene: teleop drive then stem gather (before FD reset)."""
    scene = _build_welded_zero_g_scene()
    ctrl = new_direct_controller(scene, fr3_robot)
    _drive_to_deflected_state(scene, ctrl, linear)
    scene.coupled_substep(SUB_DT)
    return _stem_apple_wrench_coupled_gather(scene)[:3]


@requires_fr3
def test_jacobian_force_row_sign_flips_when_y_drive_reverses():
    """Reversing lateral Y teleop flips stem–apple force; some stiffness column flips ∂F_y/∂k."""
    f_pos = _stem_force_after_zero_g_drive((0.0, 0.15, 0.0))
    f_neg = _stem_force_after_zero_g_drive((0.0, -0.15, 0.0))

    assert np.sign(f_pos[1]) != np.sign(f_neg[1]), (
        f"expected opposite force_y: pos={f_pos[1]}, neg={f_neg[1]}"
    )

    scene = _build_welded_zero_g_scene()
    ctrl = new_direct_controller(scene, fr3_robot)
    _, _, j_pos, _ = _fd_after_drive(scene, ctrl, (0.0, 0.15, 0.0))
    _, _, j_neg, _ = _fd_after_drive(scene, ctrl, (0.0, -0.15, 0.0))
    row = FORCE_ROW_BASE + 1
    flipped_cols = [
        c
        for c in range(j_pos.shape[1])
        if abs(j_pos[row, c]) >= MIN_J_ABS
        and abs(j_neg[row, c]) >= MIN_J_ABS
        and int(np.sign(j_pos[row, c])) != int(np.sign(j_neg[row, c]))
    ]
    assert flipped_cols, (
        f"expected at least one stiffness column with opposite ∂force_y/∂k; "
        f"j_pos={j_pos[row, :]}, j_neg={j_neg[row, :]}"
    )


@requires_fr3
def test_extract_mega_fd_jacobian_matches_forward_difference():
    """``extract_mega_fd_jacobian`` matches explicit (y_i - y_0) / ε on current state."""
    scene = _build_welded_zero_g_scene()
    ctrl = new_direct_controller(scene, fr3_robot)
    _drive_to_deflected_state(scene, ctrl, (0.0, 0.15, 0.0))
    reset_perturbed_instances_to_nominal(scene.cable)

    from apple_pick_sim.fruiting_system.mega_fd import mega_vbd_substep

    mega_vbd_substep(scene.cable, SUB_DT, collision_pipeline=scene.cable_collision_pipeline)
    extracted = extract_mega_fd_jacobian(scene.cable, EPS, dt=SUB_DT)
    _assert_jacobian_matches_features(extracted, EPS)


@requires_fr3
def test_welded_features_include_stem_apple_wrench():
    """Feature vector is 12-D (apple, proxy, wrench) when welded."""
    scene = _build_welded_zero_g_scene()
    ctrl = new_direct_controller(scene, fr3_robot)
    _drive_to_deflected_state(scene, ctrl, (0.0, 0.15, 0.0))
    feat = default_mega_fd_features(scene.cable, 0, dt=SUB_DT)
    assert feat.size == 12
    assert np.linalg.norm(feat[FORCE_ROW_BASE : FORCE_ROW_BASE + 3]) >= MIN_FORCE
