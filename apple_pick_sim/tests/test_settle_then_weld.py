"""Tests for settle-then-weld initialization (quiet fix_to_apple start)."""

from __future__ import annotations

import numpy as np
import pytest


from apple_pick_sim.tests.conftest import FRAME_DT, RANGES_FIXTURE, SUB_DT, fr3_assets_available


pytestmark = pytest.mark.skipif(
    not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core"
)


def test_settle_then_weld_quiet_start_bounds_first_harvest_wrench():
    import apple_pick_sim.coupled_fruiting as cf
    import apple_pick_sim.fruiting_system as fs
    from apple_pick_sim.robot import fr3_robot

    ranges = fs.load_ranges(RANGES_FIXTURE)
    seed = 0

    # --- Settle with a free apple (fix_to_apple=False) ---
    settled = cf.build_coupled_fruiting_fr3(
        ranges,
        seed,
        enable_self_collisions=False,
        mujoco_solver_kwargs={"disable_contacts": True},
        gripper_proxy=fs.GripperProxyConfig(
            mass=fr3_robot.EE_MASS_KG,
            box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
            fix_to_apple=False,
        ),
    )
    cf.settle_vbd_substeps(settled, substeps=40, dt=SUB_DT)

    # --- Build welded scene and seed from settled ---
    welded = cf.build_coupled_fruiting_fr3(
        ranges,
        seed,
        enable_self_collisions=False,
        mujoco_solver_kwargs={"disable_contacts": True},
        gripper_proxy=fs.GripperProxyConfig(
            mass=fr3_robot.EE_MASS_KG,
            box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
            fix_to_apple=True,
        ),
    )
    cf.seed_fix_to_apple_from_settled(welded_scene=welded, settled_scene=settled, quiet_apple_proxy=True)

    cable = welded.cable
    apple = cable.apple_body
    proxy = cable.gripper_proxy_body
    assert apple is not None and cable.gripper_proxy_offset_in_apple_frame is not None
    settled_bq = settled.cable.state_0.body_q.numpy().reshape(-1, 7)
    bq = cable.state_0.body_q.numpy().reshape(-1, 7)
    np.testing.assert_allclose(bq[apple], settled_bq[apple], rtol=1e-5, atol=1e-6)
    gap = float(np.linalg.norm(bq[proxy, :3] - bq[apple, :3]))
    expected = float(np.linalg.norm(cable.gripper_proxy_offset_in_apple_frame))
    np.testing.assert_allclose(gap, expected, rtol=0.02, atol=1e-3)

    tcp = welded.tcp_body_index
    proxy_pos = bq[proxy, :3]
    tcp_pos = welded.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp, :3]
    assert float(np.linalg.norm(tcp_pos - proxy_pos)) < 0.15

    welded.robot_kinematic_mode = True
    ctrl = fr3_robot.Fr3EEDirectJointController(welded.robot_model, welded.tcp_body_index)
    ctrl.sync_target_from_state(welded.robot_state_0)

    # One noop frame: apply no motion, then step once to harvest stem tension.
    welded.apply_fr3_ee_teleop_direct(FRAME_DT, ctrl, velocity=fr3_robot.EEVelocity())
    welded.coupled_substep(SUB_DT)

    tcp = welded.tcp_body_index
    w = welded.proxy_forces.numpy().reshape(-1, 6)[tcp]
    assert float(np.linalg.norm(w[:3])) < 200.0
    assert float(np.linalg.norm(w[3:])) < 80.0


def test_seed_quiet_zeros_apple_and_proxy_twists():
    import apple_pick_sim.coupled_fruiting as cf
    import apple_pick_sim.fruiting_system as fs
    from apple_pick_sim.robot import fr3_robot

    ranges = fs.load_ranges(RANGES_FIXTURE)
    settled = cf.build_coupled_fruiting_fr3(
        ranges,
        1,
        enable_self_collisions=False,
        mujoco_solver_kwargs={"disable_contacts": True},
        gripper_proxy=fs.GripperProxyConfig(
            mass=fr3_robot.EE_MASS_KG,
            box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
            fix_to_apple=False,
        ),
    )
    cf.settle_vbd_substeps(settled, substeps=30, dt=SUB_DT)
    welded = cf.build_coupled_fruiting_fr3(
        ranges,
        1,
        enable_self_collisions=False,
        mujoco_solver_kwargs={"disable_contacts": True},
        gripper_proxy=fs.GripperProxyConfig(
            mass=fr3_robot.EE_MASS_KG,
            box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
            fix_to_apple=True,
        ),
    )
    cf.seed_fix_to_apple_from_settled(
        welded_scene=welded, settled_scene=settled, quiet_apple_proxy=True
    )
    cable = welded.cable
    apple = cable.apple_body
    proxy = cable.gripper_proxy_body
    bqd = cable.state_0.body_qd.numpy().reshape(-1, 6)
    np.testing.assert_allclose(bqd[apple], 0.0, atol=1e-9)
    np.testing.assert_allclose(bqd[proxy], 0.0, atol=1e-9)


def test_seed_aligns_body_q_prev_for_apple_and_proxy():
    import apple_pick_sim.coupled_fruiting as cf
    import apple_pick_sim.fruiting_system as fs
    from apple_pick_sim.robot import fr3_robot

    ranges = fs.load_ranges(RANGES_FIXTURE)
    settled = cf.build_coupled_fruiting_fr3(
        ranges,
        2,
        enable_self_collisions=False,
        mujoco_solver_kwargs={"disable_contacts": True},
        gripper_proxy=fs.GripperProxyConfig(
            mass=fr3_robot.EE_MASS_KG,
            box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
            fix_to_apple=False,
        ),
    )
    cf.settle_vbd_substeps(settled, substeps=25, dt=SUB_DT)
    welded = cf.build_coupled_fruiting_fr3(
        ranges,
        2,
        enable_self_collisions=False,
        mujoco_solver_kwargs={"disable_contacts": True},
        gripper_proxy=fs.GripperProxyConfig(
            mass=fr3_robot.EE_MASS_KG,
            box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
            fix_to_apple=True,
        ),
    )
    cf.seed_fix_to_apple_from_settled(
        welded_scene=welded, settled_scene=settled, quiet_apple_proxy=True
    )
    cable = welded.cable
    apple = cable.apple_body
    proxy = cable.gripper_proxy_body
    bq = cable.state_0.body_q.numpy().reshape(-1, 7)
    bqp = cable.solver.body_q_prev.numpy().reshape(-1, 7)
    np.testing.assert_allclose(bqp[apple], bq[apple], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(bqp[proxy], bq[proxy], rtol=1e-6, atol=1e-6)


def test_seed_rebootstrap_clears_proxy_forces():
    import apple_pick_sim.coupled_fruiting as cf
    import apple_pick_sim.fruiting_system as fs
    from apple_pick_sim.robot import fr3_robot

    ranges = fs.load_ranges(RANGES_FIXTURE)
    settled = cf.build_coupled_fruiting_fr3(
        ranges,
        3,
        enable_self_collisions=False,
        mujoco_solver_kwargs={"disable_contacts": True},
        gripper_proxy=fs.GripperProxyConfig(
            mass=fr3_robot.EE_MASS_KG,
            box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
            fix_to_apple=False,
        ),
    )
    cf.settle_vbd_substeps(settled, substeps=20, dt=SUB_DT)
    welded = cf.build_coupled_fruiting_fr3(
        ranges,
        3,
        enable_self_collisions=False,
        mujoco_solver_kwargs={"disable_contacts": True},
        gripper_proxy=fs.GripperProxyConfig(
            mass=fr3_robot.EE_MASS_KG,
            box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
            fix_to_apple=True,
        ),
    )
    welded.proxy_forces.fill_(1.0)
    welded.coupling_forces_cache.fill_(2.0)
    cf.seed_fix_to_apple_from_settled(
        welded_scene=welded, settled_scene=settled, quiet_apple_proxy=True
    )
    assert bool(np.allclose(welded.proxy_forces.numpy(), 0.0, atol=1e-9))
    assert bool(np.allclose(welded.coupling_forces_cache.numpy(), 0.0, atol=1e-9))

