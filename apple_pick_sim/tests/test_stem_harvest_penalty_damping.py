"""Penalty-weld kd is omitted from stem harvest gather, not from VBD."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import warp as wp

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from conftest import (
    COUPLED_VBD_SCENE_KW,
    RANGES_FIXTURE,
    SUB_DT,
    build_vbd_only,
    requires_fr3,
)


def _import_cf():
    from apple_pick_sim import coupled_fruiting as cf

    return cf


def _stem_linear_kd(solver, joint_index: int) -> float:
    c0 = int(solver.joint_constraint_start.numpy()[int(joint_index)])
    return float(solver.joint_penalty_kd.numpy()[c0])


def _resolve_stem_apple_joint_index(scene) -> int:
    if scene.stem_apple_joint_index is not None:
        return int(scene.stem_apple_joint_index)
    cable = scene.cable
    apple = cable.apple_body
    assert apple is not None
    jchild = cable.model.joint_child.numpy()
    for j_idx, _label in cable.fruiting_fixed_joints:
        if int(jchild[j_idx]) == apple:
            return int(j_idx)
    raise AssertionError("no stem→apple fixed joint found on cable model")


class _CableModel:
    def control(self, clone_variables: bool = False):
        del clone_variables
        return object()


@requires_fr3
def test_gather_without_penalty_damping_drops_kd_cdot():
    """Same body_q; apple-shifted body_q_prev ⇒ F_on - F_off ≈ -kd Ċ (child)."""
    from apple_pick_sim.fruiting_system import GripperProxyConfig, load_ranges
    from apple_pick_sim.vbd_fixed_joint_wrenches import (
        gather_joint_wrench_child_com_device,
    )

    cf = _import_cf()
    from apple_pick_sim import fruiting_system as fs

    scene = build_vbd_only(
        cf,
        load_ranges(RANGES_FIXTURE),
        seed=21,
        gripper_proxy=GripperProxyConfig(fix_to_apple=True),
        **COUPLED_VBD_SCENE_KW,
    )
    # Straight-rod fixture leaves weld kd at 0; set a known linear kd for the identity.
    known_kd = 50.0
    fs.set_fruiting_joint_linear_kd(
        scene.cable.solver,
        scene.cable.fruiting_fixed_joints,
        {"stem_apple": known_kd},
    )
    cf.settle_vbd_substeps(scene, substeps=80, dt=SUB_DT)
    cable = scene.cable
    apple = int(cable.apple_body)
    stem_j = _resolve_stem_apple_joint_index(scene)
    kd = _stem_linear_kd(cable.solver, stem_j)
    assert kd == pytest.approx(known_kd)

    q = cable.state_0.body_q.numpy().reshape(-1, 7).copy()
    q_prev = q.copy()
    delta = 1.0e-4
    q_prev[apple, 0] -= delta
    q_wp = wp.array(q, dtype=wp.transform, device=cable.solver.device)
    q_prev_wp = wp.array(q_prev, dtype=wp.transform, device=cable.solver.device)
    dt = float(SUB_DT)
    cdot = np.array([delta / dt, 0.0, 0.0], dtype=np.float64)

    f_on, _ = gather_joint_wrench_child_com_device(
        cable.model,
        cable.solver,
        body_q=q_wp,
        body_q_prev=q_prev_wp,
        joint_indices=[stem_j],
        dt=dt,
        include_penalty_damping=True,
    )
    f_off, _ = gather_joint_wrench_child_com_device(
        cable.model,
        cable.solver,
        body_q=q_wp,
        body_q_prev=q_prev_wp,
        joint_indices=[stem_j],
        dt=dt,
        include_penalty_damping=False,
    )
    diff = f_on.numpy()[0].astype(np.float64) - f_off.numpy()[0].astype(np.float64)
    np.testing.assert_allclose(diff, -kd * cdot, rtol=0.08, atol=0.05)


@requires_fr3
def test_stem_harvest_gather_omits_penalty_damping():
    """harvest_stem_tension_for_tcp calls gather with include_penalty_damping=False."""
    from apple_pick_sim.coupled_fruiting.proxy_coupling import harvest_stem_tension_for_tcp

    captured: dict = {}

    def _fake_gather(*_a, **kwargs):
        captured["include_penalty_damping"] = kwargs.get("include_penalty_damping", True)
        dev = kwargs["body_q"].device
        return (
            wp.zeros(1, dtype=wp.vec3, device=dev),
            wp.zeros(1, dtype=wp.vec3, device=dev),
        )

    out = wp.zeros(8, dtype=wp.spatial_vector, device="cpu")
    bq = wp.zeros(4, dtype=wp.transform, device="cpu")
    with patch(
        "apple_pick_sim.vbd_fixed_joint_wrenches.gather_joint_wrench_child_com_device",
        side_effect=_fake_gather,
    ):
        harvest_stem_tension_for_tcp(
            cable_model=_CableModel(),
            cable_solver=object(),
            body_q_post=bq,
            body_q_prev=bq,
            dt=SUB_DT,
            stem_apple_joint_index=0,
            tcp_body_index=0,
            out_robot_wrenches=out,
            explicit_apple_weight=False,
            explicit_apple_inertia=False,
        )
    assert captured["include_penalty_damping"] is False
