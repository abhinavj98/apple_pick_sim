"""V.1 batched coupled fruiting: replicate(N), per-env IK scatter, multi-TCP wrench."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from conftest import (
    COUPLED_SCENE_KW,
    RANGES_FIXTURE,
    SUB_DT,
    requires_fr3,
)
from apple_pick_sim.fruiting_system import CoupledCableScene, GripperProxyConfig, load_ranges
from apple_pick_sim.coupled_fruiting import (
    broadcast_joint_q_from_world0,
    build_batched_coupled_fruiting_fr3,
    build_batched_coupled_fruiting_placeholder,
    build_coupled_fruiting_placeholder,
    seed_fix_to_apple_from_settled,
    settle_vbd_substeps,
)
from apple_pick_sim.coupled_fruiting.settle_then_weld import _proxy_world_pose_from_apple
from apple_pick_sim.coupled_fruiting.apply_wrench import (
    _apply_registry_spatial_wrenches_to_body_f,
)
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.robot.fr3_robot.placement import (
    IK_BOOTSTRAP_POS_TOL_M,
    IKBootstrapConvergenceError,
)


_NUM_ENVS = 2
_SETTLE_SUBSTEPS = 50


def _gripper_free() -> GripperProxyConfig:
    return GripperProxyConfig(
        mass=fr3_robot.EE_MASS_KG,
        fix_to_apple=False,
        robot_facing_weld=False,
    )


def _gripper_welded() -> GripperProxyConfig:
    return GripperProxyConfig(
        mass=fr3_robot.EE_MASS_KG,
        fix_to_apple=True,
        robot_facing_weld=True,
    )


def _cable_apple_indices(cable: CoupledCableScene, num_envs: int) -> tuple[int, ...]:
    """Global apple body index per replicated world (works without ``BatchedEnvLayout``)."""
    bws = cable.model.body_world_start.numpy()
    bodies_per = int(bws[1] - bws[0])
    tpl_apple = cable.apple_body
    if tpl_apple is None:
        return tuple(-1 for _ in range(num_envs))
    return tuple(int(tpl_apple) + w * bodies_per for w in range(num_envs))


def _cable_proxy_indices(cable: CoupledCableScene, num_envs: int) -> tuple[int, ...]:
    bws = cable.model.body_world_start.numpy()
    bodies_per = int(bws[1] - bws[0])
    tpl_proxy = cable.gripper_proxy_body
    return tuple(int(tpl_proxy) + w * bodies_per for w in range(num_envs))


def _make_batched_settle_then_weld(ranges, seed: int, *, settle_substeps: int = _SETTLE_SUBSTEPS):
    """Build N free worlds, settle in parallel, weld, seed world-i from world-i."""
    last_exc: Exception | None = None
    build_kw = dict(device="cpu", num_envs=_NUM_ENVS, **COUPLED_SCENE_KW)
    for try_seed in (seed, seed + 1, seed + 2, seed + 3):
        try:
            settled = build_batched_coupled_fruiting_fr3(
                ranges,
                try_seed,
                vbd_only=True,
                gripper_proxy=_gripper_free(),
                **build_kw,
            )
            settle_vbd_substeps(settled, substeps=settle_substeps, dt=SUB_DT)
            welded = build_batched_coupled_fruiting_fr3(
                ranges,
                try_seed,
                gripper_proxy=_gripper_welded(),
                skip_ik_bootstrap=True,
                defer_template_robot_bootstrap=True,
                **build_kw,
            )
            seed_fix_to_apple_from_settled(
                welded_scene=welded,
                settled_scene=settled,
                quiet_apple_proxy=True,
            )
            return welded, settled
        except IKBootstrapConvergenceError as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]


@pytest.fixture
def ranges():
    return load_ranges(RANGES_FIXTURE)


def test_build_num_envs_smoke(ranges):
    scene = build_batched_coupled_fruiting_placeholder(
        ranges,
        42,
        num_envs=4,
        device="cpu",
        **COUPLED_SCENE_KW,
    )
    assert scene.layout is not None
    assert scene.layout.num_envs == 4
    assert scene.cable.model.world_count == 4
    assert scene.robot_model is not None
    assert scene.robot_model.world_count == 4
    assert len(scene.proxy_registry.robot_to_proxy) == 4


def test_world0_parity_single_env(ranges):
    single = build_coupled_fruiting_placeholder(
        ranges, 7, device="cpu", **COUPLED_SCENE_KW
    )
    batch = build_batched_coupled_fruiting_placeholder(
        ranges, 7, num_envs=4, device="cpu", **COUPLED_SCENE_KW
    )
    layout = batch.layout
    assert layout is not None
    apple_t = single.cable.apple_body
    if apple_t is None:
        pytest.skip("fixture has no apple")
    single_apple = single.cable.state_0.body_q.numpy().reshape(-1, 7)[apple_t]
    batch_apple = batch.cable.state_0.body_q.numpy().reshape(-1, 7)[
        layout.apple_body_indices[0]
    ]
    np.testing.assert_allclose(batch_apple, single_apple, atol=1e-4)
    single_tcp = single.robot_state_0.body_q.numpy().reshape(-1, 7)[
        int(single.tcp_body_index)
    ]
    batch_tcp = batch.robot_state_0.body_q.numpy().reshape(-1, 7)[
        layout.tcp_body_indices[0]
    ]
    np.testing.assert_allclose(batch_tcp, single_tcp, atol=1e-3)


def test_coupled_substep_multi_env_stable(ranges):
    scene = build_batched_coupled_fruiting_placeholder(
        ranges,
        11,
        num_envs=4,
        device="cpu",
        **COUPLED_SCENE_KW,
    )
    scene.robot_kinematic_mode = True
    layout = scene.layout
    assert layout is not None
    for _ in range(60):
        scene.coupled_substep(SUB_DT)
    body_q = scene.cable.state_0.body_q.numpy().reshape(-1, 7)
    for w, apple_idx in enumerate(layout.apple_body_indices):
        if apple_idx < 0:
            continue
        z = float(body_q[apple_idx, 2])
        assert z > -0.05, f"world {w} apple fell: z={z}"


def test_broadcast_joint_q_copies_all_worlds(ranges):
    scene = build_batched_coupled_fruiting_placeholder(
        ranges,
        3,
        num_envs=3,
        device="cpu",
        **COUPLED_SCENE_KW,
    )
    layout = scene.layout
    assert layout is not None
    jq = scene.robot_model.joint_q.numpy().copy()
    jqd = scene.robot_model.joint_qd.numpy().copy()
    w0_slice = layout.joint_q_slice(0)
    w0_dof = layout.joint_qd_slice(0)
    jq[w0_slice] += 0.1
    jqd[w0_dof] += 0.05
    scene.robot_model.joint_q.assign(jq)
    scene.robot_model.joint_qd.assign(jqd)
    scene.robot_state_0.joint_q.assign(jq)
    scene.robot_state_0.joint_qd.assign(jqd)
    broadcast_joint_q_from_world0(scene, layout)
    jq_out = scene.robot_model.joint_q.numpy()
    jqd_out = scene.robot_model.joint_qd.numpy()
    for w in range(1, layout.num_envs):
        np.testing.assert_allclose(jq_out[layout.joint_q_slice(w)], jq_out[w0_slice])
        np.testing.assert_allclose(jqd_out[layout.joint_qd_slice(w)], jqd_out[w0_dof])


def test_apply_wrench_all_registry_tcps(ranges):
    scene = build_batched_coupled_fruiting_placeholder(
        ranges,
        5,
        num_envs=3,
        device="cpu",
        **COUPLED_SCENE_KW,
    )
    assert scene.proxy_registry is not None
    assert scene.robot_state_0 is not None
    dev = scene.robot_state_0.body_f.device
    wrenches = scene.proxy_forces
    assert wrenches is not None
    wrenches.zero_()
    robot_ids = scene.proxy_registry.robot_ids_wp(dev)
    w_np = wrenches.numpy().reshape(-1, 6)
    for i, tcp in enumerate(scene.proxy_registry.robot_body_ids):
        w_np[tcp] = (float(i + 1), 0.0, 0.0, 0.0, 0.0, 0.0)
    wrenches.assign(w_np.ravel())
    _apply_registry_spatial_wrenches_to_body_f(
        scene.robot_state_0, robot_ids, wrenches
    )
    body_f = scene.robot_state_0.body_f.numpy()
    for i, tcp in enumerate(scene.proxy_registry.robot_body_ids):
        f = body_f[tcp][:3]
        assert float(f[0]) == pytest.approx(float(i + 1))


@requires_fr3
@pytest.mark.slow
def test_build_batched_fr3_smoke(ranges):
    scene = build_batched_coupled_fruiting_fr3(
        ranges,
        42,
        device="cpu",
        num_envs=2,
        robot_base_from_proxy=True,
        **COUPLED_SCENE_KW,
    )
    assert scene.layout is not None
    assert scene.layout.num_envs == 2
    assert scene.cable.model.world_count == 2


@requires_fr3
@pytest.mark.slow
def test_parallel_free_settle_runs_on_all_worlds(ranges):
    """Free-proxy batched scene settles every replicated world in parallel."""
    build_kw = dict(device="cpu", num_envs=_NUM_ENVS, **COUPLED_SCENE_KW)
    settled = build_batched_coupled_fruiting_fr3(
        ranges,
        2,
        vbd_only=True,
        gripper_proxy=_gripper_free(),
        **build_kw,
    )
    assert int(settled.cable.model.world_count) == _NUM_ENVS

    apple_ids = _cable_apple_indices(settled.cable, _NUM_ENVS)
    bq_before = settled.cable.state_0.body_q.numpy().reshape(-1, 7)
    apple_before = {w: bq_before[apple_ids[w], :3].copy() for w in range(_NUM_ENVS)}

    settle_vbd_substeps(settled, substeps=_SETTLE_SUBSTEPS, dt=SUB_DT)

    bq_after = settled.cable.state_0.body_q.numpy().reshape(-1, 7)
    for w in range(_NUM_ENVS):
        apple_idx = apple_ids[w]
        if apple_idx < 0:
            continue
        z = float(bq_after[apple_idx, 2])
        assert z > -0.05, f"world {w} apple fell during parallel settle: z={z}"
        disp = float(np.linalg.norm(bq_after[apple_idx, :3] - apple_before[w]))
        assert disp > 1e-5, f"world {w} apple did not move during settle: disp={disp}"


@requires_fr3
@pytest.mark.slow
def test_parallel_settle_then_weld_seeds_each_world_apple(ranges):
    """Welded scene copies settled apple pose per world (not a single-world broadcast)."""
    welded, settled = _make_batched_settle_then_weld(ranges, seed=2)
    assert int(settled.cable.model.world_count) == _NUM_ENVS
    assert int(welded.cable.model.world_count) == _NUM_ENVS

    settled_bq = settled.cable.state_0.body_q.numpy().reshape(-1, 7)
    welded_bq = welded.cable.state_0.body_q.numpy().reshape(-1, 7)
    for w in range(_NUM_ENVS):
        apple_idx = _cable_apple_indices(settled.cable, _NUM_ENVS)[w]
        if apple_idx < 0:
            continue
        np.testing.assert_allclose(
            welded_bq[apple_idx],
            settled_bq[apple_idx],
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"world {w} apple not copied from parallel settled state",
        )


@requires_fr3
@pytest.mark.slow
def test_parallel_welded_quiet_proxy_offset_per_world(ranges):
    """After parallel settle-then-weld, each proxy sits at the grasp offset from its apple."""
    welded, _settled = _make_batched_settle_then_weld(ranges, seed=2)
    cable = welded.cable
    offset = cable.gripper_proxy_offset_in_apple_frame
    assert offset is not None
    bq = cable.state_0.body_q.numpy().reshape(-1, 7)
    bqd = cable.state_0.body_qd.numpy().reshape(-1, 6)
    apple_ids = _cable_apple_indices(cable, _NUM_ENVS)
    proxy_ids = _cable_proxy_indices(cable, _NUM_ENVS)
    for w in range(_NUM_ENVS):
        apple_idx = apple_ids[w]
        proxy_idx = proxy_ids[w]
        if apple_idx < 0:
            continue
        expected_pos, _ = _proxy_world_pose_from_apple(bq[apple_idx], offset)
        np.testing.assert_allclose(bq[proxy_idx, :3], expected_pos, atol=1e-4)
        np.testing.assert_allclose(bqd[apple_idx], 0.0, atol=1e-6)
        np.testing.assert_allclose(bqd[proxy_idx], 0.0, atol=1e-6)


@requires_fr3
@pytest.mark.slow
def test_parallel_welded_ik_bootstrap_aligns_batched_tcp(ranges):
    """After settle-then-weld seed, batched world-0 TCP matches proxy (not only template FK)."""
    welded, _settled = _make_batched_settle_then_weld(ranges, seed=2)
    layout = welded.layout
    assert layout is not None

    cable = welded.cable
    bq = cable.state_0.body_q.numpy().reshape(-1, 7)
    proxy_idx = layout.proxy_body_indices[0]
    proxy_pos = bq[proxy_idx, :3]

    tcp_pos = welded.robot_state_0.body_q.numpy().reshape(-1, 7)[
        layout.tcp_body_indices[0], :3
    ]
    assert float(np.linalg.norm(tcp_pos - proxy_pos)) < IK_BOOTSTRAP_POS_TOL_M


@requires_fr3
@pytest.mark.slow
def test_parallel_welded_ik_bootstrap_aligns_world0_tcp(ranges):
    """IK bootstrap solves on the template robot and broadcasts joint_q to all worlds."""
    import newton

    welded, _settled = _make_batched_settle_then_weld(ranges, seed=2)
    layout = welded.layout
    tpl = welded.ik_template_robot_model
    assert layout is not None
    assert tpl is not None

    cable = welded.cable
    bq = cable.state_0.body_q.numpy().reshape(-1, 7)
    proxy_idx = layout.proxy_body_indices[0]
    proxy_pos = bq[proxy_idx, :3]

    tpl_state = tpl.state()
    newton.eval_fk(tpl, tpl.joint_q, tpl.joint_qd, tpl_state)
    tcp_pos = tpl_state.body_q.numpy().reshape(-1, 7)[layout.template_tcp_body, :3]
    assert float(np.linalg.norm(tcp_pos - proxy_pos)) < IK_BOOTSTRAP_POS_TOL_M

    coord_per = int(tpl.joint_coord_count)
    batched_jq = welded.robot_model.joint_q.numpy()
    tpl_jq = tpl.joint_q.numpy()[:coord_per]
    np.testing.assert_allclose(batched_jq[:coord_per], tpl_jq, atol=1e-4)


@requires_fr3
@pytest.mark.slow
def test_batched_fr3_fix_to_apple_teleop_all_worlds_converge(ranges):
    welded, _settled = _make_batched_settle_then_weld(ranges, seed=2)
    ik_kw = fr3_robot.batched_ik_teleop_kwargs(welded)
    assert ik_kw, "batched FR3 scene should expose template IK kwargs"
    ctrl = fr3_robot.Fr3BatchedEEDirectJointController(
        welded.robot_model,
        ik_iterations=96,
        **ik_kw,
    )
    ctrl.sync_target_from_state(welded.robot_state_0)
    ctrl.run_ik_teleop_frame(
        1.0 / 60.0,
        welded.robot_state_0,
        velocity=fr3_robot.EEVelocity(),
        lock_angular=True,
    )
    ctrl.apply_direct_joints(welded.robot_state_0)
    ctrl.run_ik_teleop_frame(
        1.0 / 60.0,
        welded.robot_state_0,
        velocity=fr3_robot.EEVelocity(linear=(0.001, 0.0, 0.0)),
        lock_angular=True,
    )
    ctrl.apply_direct_joints(welded.robot_state_0)
    for w, (pos_err, rot_err) in enumerate(
        ctrl.measure_ik_target_error_per_world(welded.robot_state_0)
    ):
        assert pos_err < fr3_robot.IK_TELEOP_POS_TOL_M, f"world {w} pos_err={pos_err}"
        del rot_err  # fix_to_apple: rotation not tracked at bootstrap
    layout = welded.layout
    assert layout is not None
    rows = ctrl.joint_q.numpy()
    for w in range(1, layout.num_envs):
        np.testing.assert_allclose(rows[w], rows[0], atol=1e-4)


@requires_fr3
@pytest.mark.slow
def test_batched_fr3_per_env_velocity_diverges(ranges):
    welded, _settled = _make_batched_settle_then_weld(ranges, seed=2)
    ik_kw = fr3_robot.batched_ik_teleop_kwargs(welded)
    assert ik_kw
    ctrl = fr3_robot.Fr3BatchedEEDirectJointController(
        welded.robot_model,
        velocity_for_world=lambda w: (
            fr3_robot.EEVelocity(linear=(0.002, 0.0, 0.0)) if w == 0 else fr3_robot.EEVelocity()
        ),
        **ik_kw,
    )
    ctrl.sync_target_from_state(welded.robot_state_0)
    t0_init = wp.transform_get_translation(ctrl.target_tf[0])
    t1_init = wp.transform_get_translation(ctrl.target_tf[1])
    ctrl.advance_target(1.0 / 60.0, velocity=fr3_robot.EEVelocity())
    t0 = wp.transform_get_translation(ctrl.target_tf[0])
    t1 = wp.transform_get_translation(ctrl.target_tf[1])
    assert float(t0[0] - t0_init[0]) > float(t1[0] - t1_init[0]) + 1e-5


@requires_fr3
@pytest.mark.slow
def test_batched_fr3_fix_to_apple_substep_stable(ranges):
    welded, _settled = _make_batched_settle_then_weld(ranges, seed=3)
    layout = welded.layout
    assert layout is not None
    welded.robot_kinematic_mode = True
    for _ in range(30):
        welded.coupled_substep(SUB_DT)
    body_q = welded.cable.state_0.body_q.numpy().reshape(-1, 7)
    for w, apple_idx in enumerate(layout.apple_body_indices):
        if apple_idx < 0:
            continue
        z = float(body_q[apple_idx, 2])
        assert z > -0.05, f"world {w} apple fell: z={z}"
