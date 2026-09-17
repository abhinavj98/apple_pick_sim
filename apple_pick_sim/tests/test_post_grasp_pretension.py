"""Slow acceptance: post-grasp settle establishes plant pretension (load split)."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_build import (
    apply_post_grasp_vbd_settle,
    build_batched_heterogeneous_scene,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    RobotConfig,
    RuntimeConfig,
    SceneSettleCollisionConfig,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_coupled_sim import (
    BatchedHeterogeneousCoupledSim,
)
from apple_pick_sim.coupled_fruiting.defaults import (
    COUPLED_BASE_POS,
    COUPLED_ROBOT_BASE_POS,
)
from apple_pick_sim.coupled_fruiting.episode_state_snapshot import EpisodeStateSnapshot
from apple_pick_sim.coupled_fruiting.post_grasp_pretension import plant_load_split
from apple_pick_sim.fruiting_system import (
    GripperProxyConfig,
    load_ranges,
    sample_heterogeneous_params_list,
)
from apple_pick_sim.robot import fr3_robot
from conftest import requires_fr3

RANGES_FIXTURE = _TESTS_DIR.parent / "fixtures" / "fruiting_system_ranges_straight_rod_test.json"

pytestmark = [
    pytest.mark.slow,
    requires_fr3,
]


def _dynamic_apple_config(
    *,
    settle_substeps: int,
    post_grasp_settle_substeps: int,
) -> BatchedHeterogeneousCoupledSimConfig:
    base = BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=1)
    return dataclasses.replace(
        base,
        runtime=RuntimeConfig(num_envs=1, device="cpu", env_spacing=(2.5, 2.5, 0.0)),
        robot=RobotConfig(
            kind="fr3",
            step_mode="coupled",
            fix_to_apple=True,
            force_batched_layout=True,
            gripper=GripperProxyConfig(
                mass=fr3_robot.EE_MASS_KG,
                fix_to_apple=True,
                dynamic_apple=True,
                robot_facing_weld=False,
            ),
            robot_base_pos=COUPLED_ROBOT_BASE_POS,
            skip_ik_bootstrap=True,
            defer_template_robot_bootstrap=True,
        ),
        scene=SceneSettleCollisionConfig(
            settle_substeps=settle_substeps,
            post_grasp_settle_substeps=post_grasp_settle_substeps,
            settle_quiet_every=50,
            fruiting_base_pos=COUPLED_BASE_POS,
            enable_self_collisions=False,
        ),
        fruiting_system=dataclasses.replace(
            base.fruiting_system, tcp_harvest_source="weld"
        ),
        settle_diagnostics=None,
        domain_randomization=dataclasses.replace(
            base.domain_randomization,
            topology_seed=7,
        ),
    )


def test_post_grasp_settle_builds_stem_lambda_keeps_build_rest():
    """Post-grasp settle fills stem AVBD lambda while apple/woody rest stay frozen."""
    ranges = load_ranges(RANGES_FIXTURE)
    params = sample_heterogeneous_params_list(ranges, topology_seed=7, num_envs=1)
    cfg = _dynamic_apple_config(settle_substeps=200, post_grasp_settle_substeps=0)
    # Capture build rest after free settle→weld (no post-grasp yet).
    result = build_batched_heterogeneous_scene(cfg, params, ranges)
    scene = result.scene
    apple = int(scene.cable.apple_body)
    proxy = int(scene.cable.gripper_proxy_body)
    rest_before = scene.cable.model.body_q.numpy().reshape(-1, 7).copy()
    stem_j = int(scene.stem_apple_joint_index)
    lam0 = float(np.linalg.norm(scene.cable.solver.joint_lambda_lin.numpy()[stem_j]))

    settle_cfg = dataclasses.replace(
        cfg,
        scene=dataclasses.replace(cfg.scene, post_grasp_settle_substeps=400),
    )
    apply_post_grasp_vbd_settle(
        scene,
        config=settle_cfg,
        per_env_params=tuple(params),
        substeps=400,
    )
    rest_after = scene.cable.model.body_q.numpy().reshape(-1, 7)
    for bid in range(rest_before.shape[0]):
        if bid == proxy:
            continue
        np.testing.assert_allclose(
            rest_after[bid], rest_before[bid], rtol=1e-6, atol=1e-6
        )
    lam1 = float(np.linalg.norm(scene.cable.solver.joint_lambda_lin.numpy()[stem_j]))
    assert lam1 > lam0 + 1e-3, f"stem lambda did not grow: {lam0:.4e} -> {lam1:.4e}"
    splits = plant_load_split(scene, dt=float(cfg.runtime.sub_dt))
    assert splits is not None
    assert splits[0].apple_weight_N > 0.5
    # Stem reaction is non-trivial once lambda warm-starts.
    assert float(np.linalg.norm(splits[0].stem_force_world)) > 0.1 * splits[0].apple_weight_N


def test_snapshot_restore_preserves_post_grasp_load_split():
    """Episode snapshot restore reproduces stem/weld load split after settle."""
    ranges = load_ranges(RANGES_FIXTURE)
    params = sample_heterogeneous_params_list(ranges, topology_seed=7, num_envs=1)
    cfg = _dynamic_apple_config(settle_substeps=120, post_grasp_settle_substeps=0)
    sim = BatchedHeterogeneousCoupledSim(cfg, params, ranges, use_settle_cache=False)
    assert sim.layout is not None
    settle_cfg = dataclasses.replace(
        sim.config,
        scene=dataclasses.replace(sim.config.scene, post_grasp_settle_substeps=400),
    )
    apply_post_grasp_vbd_settle(
        sim.scene,
        config=settle_cfg,
        per_env_params=sim.per_env_params,
        substeps=400,
    )
    before = plant_load_split(sim.scene, dt=float(cfg.runtime.sub_dt))
    assert before is not None
    snap = EpisodeStateSnapshot.capture(sim)

    solver = sim.scene.cable.solver
    solver.joint_lambda_lin.zero_()
    solver.joint_lambda_ang.zero_()

    snap.restore(sim)
    after = plant_load_split(sim.scene, dt=float(cfg.runtime.sub_dt))
    assert after is not None
    np.testing.assert_allclose(
        after[0].stem_force_world,
        before[0].stem_force_world,
        rtol=1e-3,
        atol=1e-2,
    )
    np.testing.assert_allclose(
        after[0].weld_force_world,
        before[0].weld_force_world,
        rtol=1e-3,
        atol=1e-2,
    )
