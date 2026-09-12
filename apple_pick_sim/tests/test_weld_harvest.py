"""TCP harvest from the proxy↔apple weld reaction (dynamic apple path)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from apple_pick_sim.coupled_fruiting.builders import (
    _resolve_tcp_harvest_source,
    build_coupled_fruiting_fr3,
)
from apple_pick_sim.tests.test_coupled_fruiting_system import (
    SUB_DT,
    _import_cf,
    _import_fs,
)
from conftest import (
    COUPLED_BASE_POS,
    COUPLED_ROBOT_BASE_POS,
    DEFAULT_MJ_KW,
    RANGES_FIXTURE,
    build_coupled_fr3,
    requires_fr3,
    run_coupled_substeps_direct_hold,
)

_BUILD_KW = dict(
    enable_self_collisions=False,
    base_pos=COUPLED_BASE_POS,
    robot_base_pos=COUPLED_ROBOT_BASE_POS,
    robot_base_from_proxy=False,
    mujoco_solver_kwargs=DEFAULT_MJ_KW,
    ik_bootstrap_iterations=256,
)


@requires_fr3
def test_weld_harvest_requires_dynamic_apple():
    fs = _import_fs()
    with pytest.raises(ValueError, match="dynamic_apple"):
        _resolve_tcp_harvest_source(
            fs.GripperProxyConfig(fix_to_apple=True, dynamic_apple=False),
            override="weld",
        )


@requires_fr3
def test_weld_harvest_requires_fix_to_apple():
    fs = _import_fs()
    with pytest.raises(ValueError, match="fix_to_apple"):
        _resolve_tcp_harvest_source(
            fs.GripperProxyConfig(fix_to_apple=False, dynamic_apple=True),
            override="weld",
        )


@requires_fr3
def test_dynamic_apple_defaults_to_weld_harvest():
    fs = _import_fs()
    assert (
        _resolve_tcp_harvest_source(
            fs.GripperProxyConfig(fix_to_apple=True, dynamic_apple=True),
            override=None,
        )
        == "weld"
    )


@requires_fr3
def test_dynamic_apple_rejects_explicit_stem_harvest():
    fs = _import_fs()
    with pytest.raises(ValueError, match="dynamic_apple"):
        _resolve_tcp_harvest_source(
            fs.GripperProxyConfig(fix_to_apple=True, dynamic_apple=True),
            override="stem",
        )


@requires_fr3
def test_build_dynamic_apple_auto_selects_weld_harvest():
    fs = _import_fs()
    scene = build_coupled_fr3(
        _import_cf(),
        fs.load_ranges(RANGES_FIXTURE),
        47,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True, dynamic_apple=True),
        stem_force_cap_N=None,
        stem_torque_cap_Nm=None,
    )
    assert scene.tcp_harvest_source == "weld"
    assert scene.stem_harvest_explicit_apple_weight is False


@requires_fr3
def test_build_with_weld_harvest_disables_explicit_apple_weight():
    fs = _import_fs()
    scene = build_coupled_fr3(
        _import_cf(),
        fs.load_ranges(RANGES_FIXTURE),
        47,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True, dynamic_apple=True),
        tcp_harvest_source="weld",
        stem_force_cap_N=None,
        stem_torque_cap_Nm=None,
    )
    assert scene.tcp_harvest_source == "weld"
    assert scene.stem_harvest_explicit_apple_weight is False
    assert scene.stem_harvest_explicit_apple_inertia is False
    apple = scene.cable.apple_body
    assert apple is not None
    assert float(scene.cable.model.body_inv_mass.numpy()[apple]) > 0.0


@requires_fr3
def test_weld_harvest_keeps_mujoco_apple_payload_mass_zero():
    """Weld reaction already carries apple weight; MuJoCo payload must stay mass 0.

    Otherwise gravity in M(q) and weld harvest F_weld≈F_stem+mg double-count mg
    at the TCP (~apple weight downward bias on every direction).
    """
    fs = _import_fs()
    scene = build_coupled_fr3(
        _import_cf(),
        fs.load_ranges(RANGES_FIXTURE),
        47,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True, dynamic_apple=True),
        tcp_harvest_source="weld",
        stem_force_cap_N=None,
        stem_torque_cap_Nm=None,
    )
    assert scene.tcp_harvest_source == "weld"
    assert scene.mj_apple_payload_body_index is not None
    payload = int(scene.mj_apple_payload_body_index)
    m_mj = float(scene.robot_model.body_mass.numpy()[payload])
    assert m_mj == pytest.approx(0.0, abs=1e-12), (
        f"weld harvest path must leave apple_payload mass 0, got {m_mj:.6f} kg"
    )


@requires_fr3
def test_coupled_substep_uses_weld_harvest_when_configured():
    """``coupled_substep`` writes the weld reaction into ``proxy_forces`` under the flag."""
    import apple_pick_sim.coupled_fruiting as cf
    from apple_pick_sim.coupled_fruiting import proxy_coupling as pc
    from apple_pick_sim.robot import fr3_robot
    from apple_pick_sim.robot.fr3_robot.placement import IKBootstrapConvergenceError

    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = None
    last_exc: Exception | None = None
    for try_seed in (2, 3, 4, 5, 6):
        try:
            settled = build_coupled_fruiting_fr3(
                ranges,
                try_seed,
                vbd_only=True,
                **_BUILD_KW,
                gripper_proxy=fs.GripperProxyConfig(
                    mass=fr3_robot.EE_MASS_KG,
                    fix_to_apple=False,
                ),
            )
            cf.settle_vbd_substeps(settled, substeps=40, dt=SUB_DT)
            scene = build_coupled_fruiting_fr3(
                ranges,
                try_seed,
                skip_ik_bootstrap=True,
                **_BUILD_KW,
                gripper_proxy=fs.GripperProxyConfig(
                    mass=fr3_robot.EE_MASS_KG,
                    fix_to_apple=True,
                    dynamic_apple=True,
                ),
                tcp_harvest_source="weld",
                stem_force_cap_N=None,
                stem_torque_cap_Nm=None,
            )
            cf.seed_fix_to_apple_from_settled(
                welded_scene=scene, settled_scene=settled, quiet_apple_proxy=True
            )
            break
        except IKBootstrapConvergenceError as exc:
            last_exc = exc
            scene = None
    if scene is None:
        raise last_exc  # type: ignore[misc]

    run_coupled_substeps_direct_hold(scene, fr3_robot, 20, sub_dt=SUB_DT)
    scene.coupled_substep(SUB_DT)

    cable = scene.cable
    out = wp.zeros(
        scene.robot_model.body_count, dtype=wp.spatial_vector, device=cable.model.device
    )
    pc.harvest_weld_tension_for_tcp(
        cable_model=cable.model,
        cable_solver=cable.solver,
        body_q_post=cable.state_0.body_q,
        body_q_prev=cable.state_1.body_q,
        dt=SUB_DT,
        weld_joint_index=int(cable.gripper_proxy_apple_joint),
        tcp_body_index=scene.tcp_body_index,
        out_robot_wrenches=out,
        coupling_gain=scene.stem_coupling_gain,
        force_cap_N=None,
        torque_cap_Nm=None,
    )
    tcp = scene.tcp_body_index
    np.testing.assert_allclose(
        scene.proxy_forces.numpy().reshape(-1, 6)[tcp],
        out.numpy().reshape(-1, 6)[tcp],
        rtol=1e-5,
        atol=1e-4,
    )
    assert float(np.linalg.norm(out.numpy().reshape(-1, 6)[tcp, :3])) > 0.5


def _build_dynamic_apple_weld_scene(try_seeds: tuple[int, ...] = (2, 3, 4, 5, 6)):
    """Settle→weld with ``dynamic_apple`` + weld harvest; retry on IK bootstrap failure."""
    import apple_pick_sim.coupled_fruiting as cf
    from apple_pick_sim.robot import fr3_robot
    from apple_pick_sim.robot.fr3_robot.placement import IKBootstrapConvergenceError

    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = None
    last_exc: Exception | None = None
    for try_seed in try_seeds:
        try:
            settled = build_coupled_fruiting_fr3(
                ranges,
                try_seed,
                vbd_only=True,
                **_BUILD_KW,
                gripper_proxy=fs.GripperProxyConfig(
                    mass=fr3_robot.EE_MASS_KG,
                    fix_to_apple=False,
                ),
            )
            cf.settle_vbd_substeps(settled, substeps=80, dt=SUB_DT)
            scene = build_coupled_fruiting_fr3(
                ranges,
                try_seed,
                skip_ik_bootstrap=True,
                **_BUILD_KW,
                gripper_proxy=fs.GripperProxyConfig(
                    mass=fr3_robot.EE_MASS_KG,
                    fix_to_apple=True,
                    dynamic_apple=True,
                ),
                tcp_harvest_source="weld",
                stem_force_cap_N=None,
                stem_torque_cap_Nm=None,
            )
            cf.seed_fix_to_apple_from_settled(
                welded_scene=scene, settled_scene=settled, quiet_apple_proxy=True
            )
            return scene
        except IKBootstrapConvergenceError as exc:
            last_exc = exc
            scene = None
    raise last_exc  # type: ignore[misc]


@requires_fr3
def test_weld_reaction_balances_stem_plus_apple_weight():
    """Dynamic apple statics: weld-on-proxy ≈ stem-on-apple + m·g after VBD settle.

    Absolute |F| may be large (woody rest-sync residual under a prescribed proxy);
    the free-body identity is what weld harvest must satisfy for TCP readout.
    """
    import apple_pick_sim.coupled_fruiting as cf
    from apple_pick_sim.coupled_fruiting import proxy_coupling as pc
    from apple_pick_sim.fruiting_system import analytic_apple_mass_kg
    from apple_pick_sim.robot import fr3_robot
    from apple_pick_sim.vbd_fixed_joint_wrenches import gather_joint_wrench_child_com_device

    scene = _build_dynamic_apple_weld_scene()
    cf.settle_vbd_substeps(scene, substeps=200, dt=SUB_DT)
    run_coupled_substeps_direct_hold(scene, fr3_robot, 40, sub_dt=SUB_DT)
    scene.coupled_substep(SUB_DT)

    cable = scene.cable
    m_apple = analytic_apple_mass_kg(cable.params)
    assert m_apple is not None and m_apple > 0.01
    mg = np.array([0.0, 0.0, -m_apple * 9.81], dtype=np.float64)

    out_weld = wp.zeros(
        scene.robot_model.body_count, dtype=wp.spatial_vector, device=cable.model.device
    )
    pc.harvest_weld_tension_for_tcp(
        cable_model=cable.model,
        cable_solver=cable.solver,
        body_q_post=cable.state_0.body_q,
        body_q_prev=cable.state_1.body_q,
        dt=SUB_DT,
        weld_joint_index=int(cable.gripper_proxy_apple_joint),
        tcp_body_index=scene.tcp_body_index,
        out_robot_wrenches=out_weld,
        coupling_gain=1.0,
        force_cap_N=None,
        torque_cap_Nm=None,
    )
    f_weld = out_weld.numpy().reshape(-1, 6)[scene.tcp_body_index, :3].astype(np.float64)

    stem_idx = wp.array([int(scene.stem_apple_joint_index)], dtype=int, device=cable.model.device)
    f_stem_arr, _ = gather_joint_wrench_child_com_device(
        cable.model,
        cable.solver,
        body_q=cable.state_0.body_q,
        body_q_prev=cable.state_1.body_q,
        joint_indices=stem_idx,
        dt=SUB_DT,
        control=cable.model.control(clone_variables=False),
        include_penalty_damping=False,
    )
    f_stem = f_stem_arr.numpy()[0].astype(np.float64)

    # Apple FBD: F_stem_on_apple + F_weld_on_apple + mg ≈ 0 with
    # F_weld_on_apple = −F_weld_on_proxy ⇒ F_weld_proxy ≈ F_stem + mg.
    np.testing.assert_allclose(f_weld, f_stem + mg, rtol=0.05, atol=1.0)
