"""Mega coupled FR3 + N-instance VBD plant (fd_ghost keyboard path)."""

from __future__ import annotations

import numpy as np
import pytest

from apple_pick_sim.tests.conftest import (
    COUPLED_SCENE_KW,
    DEFAULT_MJ_KW,
    NO_SELF_COLLISION_KW,
    RANGES_FIXTURE,
    SUB_DT,
    apply_direct_hold,
    build_coupled_fr3,
    new_direct_controller,
    requires_fr3,
)

INSTANCE_SPACING = (0.0, 1.5, 0.0)
SEED = 11
STIFFNESS_EPS = 0.02

pytestmark = requires_fr3


@pytest.fixture(scope="module", autouse=True)
def _warp_init():
    import warp as wp

    wp.init()


def _import_cf():
    import apple_pick_sim.coupled_fruiting as cf

    return cf


def _build_mega(cf, **kwargs):
    ranges = __import__("apple_pick_sim.fruiting_system", fromlist=["load_ranges"]).load_ranges(
        RANGES_FIXTURE
    )
    kw = dict(
        instance_spacing=INSTANCE_SPACING,
        stiffness_epsilon=STIFFNESS_EPS,
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
        device="cpu",
        **COUPLED_SCENE_KW,
    )
    kw.update(kwargs)
    return cf.build_mega_coupled_fruiting_fr3(ranges, SEED, **kw)


def test_mega_nominal_index_defaults_to_zero():
    cf = _import_cf()
    scene = _build_mega(cf)
    assert scene.nominal_index == 0
    assert scene.cable.instance(0).params == scene.cable.instance(scene.nominal_index).params


def test_build_two_instance_mega_coupled_finite():
    cf = _import_cf()
    scene = _build_mega(cf)
    assert scene.cable.num_instances >= 2
    dt = SUB_DT
    scene.coupled_substep(dt)
    bq = scene.cable.state_0.body_q.numpy()
    pf = scene.proxy_forces.numpy()
    assert np.all(np.isfinite(bq))
    assert np.all(np.isfinite(pf))


@pytest.mark.slow
def test_mega_coupled_tcp_stem_load_at_hold():
    """Nominal column: TCP stem harvest (gain=1) with |F| on the order of m_apple·g."""
    from apple_pick_sim.fruiting_system.params import analytic_apple_mass_kg
    from apple_pick_sim.robot import fr3_robot
    from apple_pick_sim.tests.test_coupled_fruiting_system import (
        _assert_tcp_stem_load_order_of_apple_weight,
        _run_coupled_hold_frames,
    )

    cf = _import_cf()
    scene = _build_mega(cf, stem_force_cap_N=None, stem_torque_cap_Nm=None)
    assert scene.stem_apple_joint_index is not None
    cf.settle_vbd_substeps(scene, substeps=600, dt=SUB_DT)
    _run_coupled_hold_frames(scene, fr3_robot, 100)
    scene.coupled_substep(SUB_DT)
    inst = scene.cable.instance(scene.nominal_index)
    m_apple = analytic_apple_mass_kg(inst.params)
    assert m_apple is not None and m_apple > 0.01
    _assert_tcp_stem_load_order_of_apple_weight(
        scene, scene.tcp_body_index, m_apple=m_apple, min_ratio=0.5, max_ratio=3.5
    )


def test_mega_fix_to_apple_co_teleport_twist_matches_proxy():
    """Ghost sync + co-teleport keep apple ``body_qd`` aligned with proxy (welded harvest)."""
    from apple_pick_sim.robot import fr3_robot

    cf = _import_cf()
    fs = __import__("apple_pick_sim.fruiting_system", fromlist=["load_ranges"])
    gripper = fs.GripperProxyConfig(
        mass=fr3_robot.EE_MASS_KG,
        box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
        fix_to_apple=True,
    )
    scene = _build_mega(cf, gripper_proxy=gripper, device="cpu")
    scene.robot_kinematic_mode = True
    ctrl = new_direct_controller(scene, fr3_robot)
    apply_direct_hold(
        scene,
        fr3_robot,
        ctrl,
        velocity=fr3_robot.EEVelocity(linear=(0.12, 0.0, 0.0)),
    )
    scene.mujoco_substep(SUB_DT)

    bqd = scene.cable.state_0.body_qd.numpy().reshape(-1, 6)
    for col, inst in enumerate(scene.cable.instances):
        if inst.apple_body is None:
            continue
        proxy = inst.gripper_proxy_body
        apple = inst.apple_body
        np.testing.assert_allclose(
            bqd[apple],
            bqd[proxy],
            rtol=0.0,
            atol=1e-6,
            err_msg=f"apple body_qd must match proxy after ghost co-teleport (col {col})",
        )


def test_ghost_mirror_offsets():
    cf = _import_cf()
    scene = _build_mega(cf)
    dt = SUB_DT
    scene.mujoco_substep(dt)

    tcp = scene.tcp_body_index
    bq_robot = scene.robot_state_0.body_q.numpy().reshape(-1, 7)
    tcp_pos = bq_robot[tcp, :3]

    inst0 = scene.cable.instance(0)
    inst1 = scene.cable.instance(1)
    bq = scene.cable.state_0.body_q.numpy().reshape(-1, 7)
    p0 = bq[inst0.gripper_proxy_body, :3]
    p1 = bq[inst1.gripper_proxy_body, :3]

    np.testing.assert_allclose(p0, tcp_pos, rtol=1e-4, atol=1e-4)
    expected_delta = np.array(INSTANCE_SPACING, dtype=np.float64)
    np.testing.assert_allclose(p1 - p0, expected_delta, rtol=1e-4, atol=1e-4)


@pytest.mark.slow
def test_nominal_stem_harvest_repeatable_without_teleop():
    """After settle + hold, consecutive stem harvests at TCP match (quasi-static)."""
    cf = _import_cf()
    from apple_pick_sim.coupling_force_debug import read_tcp_wrench
    from apple_pick_sim.robot import fr3_robot

    scene = _build_mega(cf)
    dt = SUB_DT
    cf.settle_vbd_substeps(scene, substeps=400, dt=dt)
    ctrl = new_direct_controller(scene, fr3_robot)
    hold = fr3_robot.EEVelocity()
    for _ in range(30):
        apply_direct_hold(scene, fr3_robot, ctrl, velocity=hold)
        scene.coupled_substep(dt)
    tcp = scene.tcp_body_index
    w0 = read_tcp_wrench(scene.proxy_forces, tcp).copy()
    scene.coupled_substep(dt)
    w1 = read_tcp_wrench(scene.proxy_forces, tcp)
    np.testing.assert_allclose(w1, w0, rtol=0.02, atol=0.5)
    assert float(np.linalg.norm(w0[:3])) > 0.5


def test_mega_instance0_parity_vs_1x1():
    """Nominal mega column matches a 1×1 coupled scene at the same fixture layout."""
    cf = _import_cf()
    from apple_pick_sim.robot import fr3_robot

    ranges = __import__("apple_pick_sim.fruiting_system", fromlist=["load_ranges"]).load_ranges(
        RANGES_FIXTURE
    )
    build_kw = dict(
        enable_self_collisions=False,
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
        device="cpu",
        ik_bootstrap_iterations=96,
    )
    mega = cf.build_mega_coupled_fruiting_fr3(
        ranges,
        SEED,
        instance_spacing=INSTANCE_SPACING,
        stiffness_epsilon=STIFFNESS_EPS,
        **build_kw,
    )
    single = build_coupled_fr3(cf, ranges, SEED, **build_kw)
    single.robot_kinematic_mode = True

    ctrl_m = new_direct_controller(mega, fr3_robot)
    ctrl_s = new_direct_controller(single, fr3_robot)

    dt = SUB_DT
    n_sub = 6
    for _ in range(n_sub):
        vel = fr3_robot.EEVelocity(linear=(0.05, 0.0, 0.0))
        mega.apply_fr3_ee_teleop_direct(dt, ctrl_m, velocity=vel)
        single.apply_fr3_ee_teleop_direct(dt, ctrl_s, velocity=vel)
        mega.coupled_substep(dt)
        single.coupled_substep(dt)

    inst0 = mega.cable.instance(0)
    mega_proxy = mega.cable.state_0.body_q.numpy().reshape(-1, 7)[
        inst0.gripper_proxy_body, :3
    ]
    single_proxy = single.cable.state_0.body_q.numpy().reshape(-1, 7)[
        single.cable.gripper_proxy_body, :3
    ]
    np.testing.assert_allclose(mega_proxy, single_proxy, rtol=0.02, atol=0.02)
