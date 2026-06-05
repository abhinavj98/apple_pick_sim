"""GPU VIC wrench kernel matches host ``Fr3EEImpedanceController``."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from apple_pick_sim.coupled_fruiting.vic_wrench import launch_apply_vic_to_coupling_cache
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import (
    Fr3EEImpedanceController,
    ImpedanceGains,
)
from apple_pick_sim.robot.fr3_robot.controllers.keyboard import EEVelocity
from apple_pick_sim.tests.conftest import (
    DEFAULT_MJ_KW,
    RANGES_FIXTURE,
    build_coupled_fr3,
    fr3_assets_available,
)

pytestmark = pytest.mark.skipif(
    not fr3_assets_available(),
    reason="Requires bundled assets/fr3 and usd-core",
)


def _import_cf():
    import apple_pick_sim.coupled_fruiting as cf

    return cf


def _build_mujoco_only_fr3():
    cf = _import_cf()
    import apple_pick_sim.fruiting_system as fs

    ranges = fs.load_ranges(RANGES_FIXTURE)
    return build_coupled_fr3(
        cf,
        ranges,
        11,
        mujoco_only=True,
        enable_self_collisions=False,
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
    )


@pytest.mark.parametrize(
    "pos,twist,gains",
    [
        ((0.5, 0.2, 1.0), EEVelocity(), ImpedanceGains()),
        ((0.51, 0.2, 1.0), EEVelocity(), ImpedanceGains(linear_k=100.0, linear_d=10.0)),
        (
            (0.5, 0.2, 1.0),
            EEVelocity(linear=(0.2, 0.0, 0.0), angular=(0.0, 0.1, 0.0)),
            ImpedanceGains(linear_k=500.0, linear_d=50.0, angular_k=20.0, angular_d=2.0),
        ),
    ],
)
def test_gpu_vic_wrench_matches_cpu(pos, twist, gains):
    scene = _build_mujoco_only_fr3()
    tcp = scene.tcp_body_index
    bq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp].copy()
    bqd = scene.robot_state_0.body_qd.numpy().reshape(-1, 6)[tcp].copy()

    target = wp.transform(
        wp.vec3(*pos),
        wp.quat(float(bq[3]), float(bq[4]), float(bq[5]), float(bq[6])),
    )

    vic = Fr3EEImpedanceController()
    w_cpu = vic.compute_applied_wrench(
        target_tf=target,
        target_twist=twist,
        tcp_body_q=bq,
        tcp_body_qd=bqd,
        gains=gains,
    )

    scene.coupling_forces_cache.zero_()
    launch_apply_vic_to_coupling_cache(
        scene,
        target_tf=target,
        target_twist=twist,
        gains=gains,
    )
    w_gpu = scene.coupling_forces_cache.numpy().reshape(-1, 6)[tcp]
    np.testing.assert_allclose(w_gpu, w_cpu, rtol=1e-5, atol=1e-4)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available")
def test_vic_substep_cuda_graph_smoke():
    """VIC on the dynamic path stays graph-capturable (no ``body_q`` host sync in substep)."""
    import apple_pick_sim.coupled_fruiting as cf
    import apple_pick_sim.fruiting_system as fs

    from apple_pick_sim.cuda_graph import capture_substep_loop
    from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import Fr3EEImpedanceController

    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(
        cf,
        ranges,
        20,
        device="cuda:0",
        mujoco_solver_kwargs={"disable_contacts": True},
        mujoco_use_cpu=False,
    )
    scene.robot_kinematic_mode = False
    scene.vic_controller = Fr3EEImpedanceController()
    scene.vic_gains = ImpedanceGains()
    tcp = scene.tcp_body_index
    bq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp]
    scene.vic_target_tf = wp.transform(
        wp.vec3(float(bq[0]), float(bq[1]), float(bq[2])),
        wp.quat(float(bq[3]), float(bq[4]), float(bq[5]), float(bq[6])),
    )
    scene.vic_target_twist = EEVelocity()

    dt = (1.0 / 60.0) / 30

    def _frame():
        for _ in range(3):
            scene.coupled_substep(dt)

    graph = capture_substep_loop(_frame, device="cuda:0", warmup=2)
    assert graph is not None
    for _ in range(3):
        wp.capture_launch(graph)
    wp.synchronize()
    cache = scene.coupling_forces_cache.numpy().reshape(-1, 6)[tcp]
    assert bool(np.isfinite(cache).all())
