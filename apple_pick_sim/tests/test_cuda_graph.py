"""CUDA graph capture smoke tests (optional CUDA hardware)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from apple_pick_sim.cuda_graph import can_capture_graph, capture_substep_loop


@pytest.fixture(scope="module", autouse=True)
def _warp_init():
    wp.init()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available")
def test_can_capture_graph_cuda():
    assert can_capture_graph("cuda:0")


def test_capture_substep_loop_returns_none_on_cpu():
    wp.init()
    calls = {"n": 0}

    def _body():
        calls["n"] += 1

    graph = capture_substep_loop(_body, device="cpu", warmup=0)
    assert graph is None
    assert calls["n"] == 0


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available")
def test_coupled_fruiting_cuda_graph_smoke():
    import apple_pick_sim.coupled_fruiting as cf
    import apple_pick_sim.fruiting_system as fs

    from conftest import build_coupled_fr3

    ranges = fs.load_ranges(
        Path(__file__).resolve().parent.parent / "fixtures" / "fruiting_system_ranges_straight_rod_test.json"
    )
    scene = build_coupled_fr3(
        cf,
        ranges,
        20,
        device="cuda:0",
        mujoco_solver_kwargs={"disable_contacts": True},
        mujoco_use_cpu=False,
    )
    dt = (1.0 / 60.0) / 30

    def _frame():
        for _ in range(3):
            scene.coupled_substep(dt)

    graph = capture_substep_loop(_frame, device="cuda:0", warmup=2)
    assert graph is not None
    for _ in range(5):
        wp.capture_launch(graph)
    wp.synchronize()
    bq = scene.robot_state_0.body_q.numpy()
    assert bool(np.isfinite(bq).all())


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available")
def test_coupled_cuda_graph_welded_explicit_stem_harvest_finite():
    """Captured loop with ``fix_to_apple`` + default explicit load stays finite at TCP."""
    import apple_pick_sim.coupled_fruiting as cf
    import apple_pick_sim.fruiting_system as fs

    from conftest import DEFAULT_MJ_KW, RANGES_FIXTURE, build_coupled_fr3, fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(
        cf,
        ranges,
        21,
        device="cuda:0",
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True),
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
        mujoco_use_cpu=False,
    )
    assert scene.stem_apple_joint_index is not None
    assert scene.apple_mass_kg > 0.0
    assert scene.stem_harvest_explicit_apple_weight is True
    dt = (1.0 / 60.0) / 30

    def _frame():
        for _ in range(3):
            scene.coupled_substep(dt)

    graph = capture_substep_loop(_frame, device="cuda:0", warmup=2)
    assert graph is not None
    for _ in range(4):
        wp.capture_launch(graph)
    wp.synchronize()
    tcp = scene.tcp_body_index
    w = scene.proxy_forces.numpy().reshape(-1, 6)[tcp]
    assert bool(np.isfinite(w).all())
    assert float(np.linalg.norm(w[:3])) < 5000.0


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available")
def test_example_coupled_fruiting_graph_flag():
    import newton.viewer

    import apple_pick_sim.example_coupled_fruiting as ex

    viewer = newton.viewer.ViewerNull()
    args = ex._make_parser().parse_args(
        ["--viewer", "null", "--cuda-graph", "--robot", "placeholder", "--seed", "1"]
    )
    example = ex.ExampleCoupledFruiting(viewer, args)
    assert example.graph is not None
    for _ in range(3):
        example.step()
    example.test_final()
