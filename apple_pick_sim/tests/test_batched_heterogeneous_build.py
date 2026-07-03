"""Tests for config-driven batched heterogeneous scene build (V.3.1 step A)."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    RobotConfig,
    RuntimeConfig,
    SceneSettleCollisionConfig,
    SettleDiagnosticsConfig,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_build import (
    BatchedHeterogeneousBuildResult,
    build_batched_heterogeneous_scene,
)
from apple_pick_sim.coupled_fruiting.defaults import COUPLED_BASE_POS, COUPLED_ROBOT_BASE_POS
from apple_pick_sim.fruiting_system import (
    GripperProxyConfig,
    load_ranges,
    sample_heterogeneous_params_list,
)
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.robot.fr3_robot.placement import IK_BOOTSTRAP_POS_TOL_M

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from conftest import COUPLED_SCENE_KW, requires_fr3  # noqa: E402
from test_heterogeneous_coupled_fruiting import _make_hetero_settle_then_weld  # noqa: E402

_TESTS_DIR = Path(__file__).resolve().parent
RANGES_FIXTURE = _TESTS_DIR.parent / "fixtures" / "fruiting_system_ranges_straight_rod_test.json"

_NUM_ENVS = 2
_PARITY_SEED = 54


@pytest.fixture
def ranges():
    return load_ranges(RANGES_FIXTURE)


@pytest.fixture
def per_env_params(ranges):
    return sample_heterogeneous_params_list(
        ranges, topology_seed=7, num_envs=_NUM_ENVS
    )


def _placeholder_vbd_config(*, settle_substeps: int = 0) -> BatchedHeterogeneousCoupledSimConfig:
    return dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=_NUM_ENVS),
        robot=RobotConfig(
            kind="placeholder",
            step_mode="vbd_only",
            fix_to_apple=False,
        ),
        scene=SceneSettleCollisionConfig(settle_substeps=settle_substeps),
    )


def test_build_minimal_smoke(ranges, per_env_params):
    cfg = _placeholder_vbd_config()
    result = build_batched_heterogeneous_scene(cfg, per_env_params, ranges)
    assert isinstance(result, BatchedHeterogeneousBuildResult)
    assert result.scene.layout is not None
    assert result.scene.layout.num_envs == _NUM_ENVS
    assert len(result.per_env_params) == _NUM_ENVS
    body_q = result.scene.cable.state_0.body_q.numpy().reshape(-1, 7)
    for w, apple_idx in enumerate(result.scene.layout.apple_body_indices):
        if apple_idx < 0:
            continue
        z = float(body_q[apple_idx, 2])
        assert z > -0.05, f"world {w} apple fell: z={z}"


def test_build_raises_on_params_length_mismatch(ranges, per_env_params):
    cfg = _placeholder_vbd_config()
    with pytest.raises(ValueError, match="per_env_params"):
        build_batched_heterogeneous_scene(cfg, per_env_params[:1], ranges)


def test_diagnostics_gated_off(ranges, per_env_params):
    cfg = _placeholder_vbd_config()
    result = build_batched_heterogeneous_scene(cfg, per_env_params, ranges)
    assert result.settle_stability_reports is None
    assert result.settle_ke_decay_reports is None
    assert result.ik_envelope_results is None


def test_diagnostics_gated_on(ranges, per_env_params):
    cfg = dataclasses.replace(
        _placeholder_vbd_config(settle_substeps=10),
        settle_diagnostics=SettleDiagnosticsConfig(),
    )
    result = build_batched_heterogeneous_scene(cfg, per_env_params, ranges)
    assert result.settle_stability_reports is not None
    assert len(result.settle_stability_reports) == _NUM_ENVS
    assert result.settle_ke_decay_reports is not None


def test_kd_overrides_on_result_and_applied(ranges, per_env_params):
    overrides = {"secondary_spur": 1.5, "stem_apple": 0.03}
    cfg = dataclasses.replace(
        _placeholder_vbd_config(),
        fruiting_system=dataclasses.replace(
            _placeholder_vbd_config().fruiting_system,
            joint_angular_kd_overrides=overrides,
        ),
    )
    result = build_batched_heterogeneous_scene(cfg, per_env_params, ranges)
    assert result.joint_angular_kd_overrides == overrides
    layout = result.scene.layout
    assert layout is not None
    solver = result.scene.cable.solver
    fruiting_joints = result.scene.cable.fruiting_fixed_joints
    spur_tpl = next(j for j, lab in fruiting_joints if "secondary_spur" in lab)
    kd_spur = _angular_kd_at_joint(solver, spur_tpl)
    assert kd_spur == pytest.approx(1.5)


def _angular_kd_at_joint(solver, global_joint_index: int) -> float:
    import newton

    jc_start = solver.joint_constraint_start.numpy()
    kd = solver.joint_penalty_kd.numpy()
    c0 = int(jc_start[global_joint_index])
    return float(kd[c0 + newton.solvers.SolverVBD.JointSlot.ANGULAR])


def _fr3_settle_weld_config(*, settle_substeps: int = 50) -> BatchedHeterogeneousCoupledSimConfig:
    """Config aligned with ``test_heterogeneous_coupled_fruiting._make_hetero_settle_then_weld``."""
    return dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=_NUM_ENVS),
        runtime=RuntimeConfig(
            num_envs=_NUM_ENVS,
            device="cpu",
            env_spacing=(2.5, 2.5, 0.0),
        ),
        robot=RobotConfig(
            kind="fr3",
            step_mode="coupled",
            fix_to_apple=True,
            gripper=GripperProxyConfig(
                mass=fr3_robot.EE_MASS_KG,
                fix_to_apple=False,
                robot_facing_weld=False,
            ),
            robot_base_pos=COUPLED_ROBOT_BASE_POS,
            skip_ik_bootstrap=True,
            defer_template_robot_bootstrap=True,
        ),
        scene=SceneSettleCollisionConfig(
            settle_substeps=settle_substeps,
            fruiting_base_pos=COUPLED_BASE_POS,
            enable_self_collisions=COUPLED_SCENE_KW["enable_self_collisions"],
        ),
        settle_diagnostics=None,
    )


def _assert_per_env_tcp_proxy_alignment(scene) -> None:
    layout = scene.layout
    assert layout is not None
    bq = scene.cable.state_0.body_q.numpy().reshape(-1, 7)
    tcp_bq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)
    for w in range(layout.num_envs):
        proxy_idx = layout.proxy_body_indices[w]
        tcp_idx = layout.tcp_body_indices[w]
        proxy_pos = bq[proxy_idx, :3]
        tcp_pos = tcp_bq[tcp_idx, :3]
        pos_err = float(np.linalg.norm(tcp_pos - proxy_pos))
        assert pos_err < IK_BOOTSTRAP_POS_TOL_M, f"world {w} pos_err={pos_err}"


@requires_fr3
@pytest.mark.slow
def test_build_parity_with_manual_settle_then_weld(ranges):
    """``build_batched_heterogeneous_scene`` matches manual settle-then-weld TCP alignment."""
    params = sample_heterogeneous_params_list(
        ranges, topology_seed=_PARITY_SEED, num_envs=_NUM_ENVS
    )
    cfg = _fr3_settle_weld_config(settle_substeps=50)
    batched = build_batched_heterogeneous_scene(cfg, params, ranges)
    manual_welded, _settled, _manual_params = _make_hetero_settle_then_weld(
        ranges, seed=_PARITY_SEED, settle_substeps=50
    )
    _assert_per_env_tcp_proxy_alignment(batched.scene)
    _assert_per_env_tcp_proxy_alignment(manual_welded)


def test_kd_overrides_filtered_to_matching_joint_labels(ranges, per_env_params):
    """Default kd keys for branched topology are dropped on straight-rod fixture."""
    cfg = dataclasses.replace(
        _placeholder_vbd_config(),
        fruiting_system=BatchedHeterogeneousCoupledSimConfig.defaults().fruiting_system,
    )
    result = build_batched_heterogeneous_scene(cfg, per_env_params, ranges)
    assert set(result.joint_angular_kd_overrides.keys()) == {"stem_apple"}
    assert result.joint_angular_kd_overrides["stem_apple"] == pytest.approx(5e-2)
    assert "support" not in result.joint_angular_kd_overrides
    assert "primary_spur" not in result.joint_angular_kd_overrides


@requires_fr3
def test_diagnostics_populated_on_weld_path(ranges, per_env_params):
    """Weld-path settle collects stability, KE decay, and IK envelope diagnostics."""
    cfg = dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=_NUM_ENVS),
        settle_diagnostics=SettleDiagnosticsConfig(),
        scene=SceneSettleCollisionConfig(settle_substeps=10),
    )
    result = build_batched_heterogeneous_scene(cfg, per_env_params, ranges)
    assert result.settle_stability_reports is not None
    assert len(result.settle_stability_reports) == _NUM_ENVS
    assert result.settle_ke_decay_reports is not None
    assert len(result.settle_ke_decay_reports) == _NUM_ENVS
    assert result.ik_envelope_results is not None
    assert len(result.ik_envelope_results) == _NUM_ENVS


def test_mock_viewer_render_hooks_called_during_settle(ranges, per_env_params):
    viewer = MagicMock()
    viewer.is_running.return_value = True
    settle_substeps = 4
    cfg = _placeholder_vbd_config(settle_substeps=settle_substeps)
    build_batched_heterogeneous_scene(cfg, per_env_params, ranges, viewer=viewer)
    assert viewer.begin_frame.call_count == settle_substeps
    assert viewer.log_state.call_count == settle_substeps
    assert viewer.end_frame.call_count == settle_substeps


@requires_fr3
@pytest.mark.slow
def test_settle_then_weld_fix_to_apple(ranges, per_env_params):
    cfg = dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=_NUM_ENVS),
        scene=SceneSettleCollisionConfig(settle_substeps=50),
    )
    result = build_batched_heterogeneous_scene(cfg, per_env_params, ranges)
    scene = result.scene
    assert scene.layout is not None
    assert scene.per_world_proxy_offsets is not None
    assert len(scene.per_world_proxy_offsets) == _NUM_ENVS
    assert all(off is not None for off in scene.per_world_proxy_offsets)
