"""Tests for BatchedHeterogeneousCoupledSim runtime (V.3.1 step B)."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import warp as wp

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ControllerConfig,
    ObsConfig,
    RobotConfig,
    SceneSettleCollisionConfig,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_coupled_sim import (
    BatchedHeterogeneousCoupledSim,
)
from apple_pick_sim.coupled_fruiting.settled_checkpoint import SettledCheckpoint
from apple_pick_sim.fruiting_system import load_ranges, sample_heterogeneous_params_list

from conftest import requires_fr3

_TESTS_DIR = Path(__file__).resolve().parent
RANGES_FIXTURE = _TESTS_DIR.parent / "fixtures" / "fruiting_system_ranges_straight_rod_test.json"
_NUM_ENVS = 2


def _require_torch():
    pytest.importorskip("torch")
    import torch

    return torch


@pytest.fixture
def ranges():
    return load_ranges(RANGES_FIXTURE)


@pytest.fixture
def per_env_params(ranges):
    return sample_heterogeneous_params_list(
        ranges, topology_seed=21, num_envs=_NUM_ENVS
    )


def _vbd_placeholder_config(*, settle_substeps: int = 0) -> BatchedHeterogeneousCoupledSimConfig:
    return dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=_NUM_ENVS),
        robot=RobotConfig(
            kind="placeholder",
            step_mode="vbd_only",
            fix_to_apple=False,
        ),
        scene=SceneSettleCollisionConfig(settle_substeps=settle_substeps),
        obs=ObsConfig(allocate_buffers=True),
    )


def _cacheable_config(*, settle_substeps: int = 8) -> BatchedHeterogeneousCoupledSimConfig:
    return dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=_NUM_ENVS),
        robot=RobotConfig(
            kind="fr3",
            step_mode="coupled",
            fix_to_apple=True,
            skip_ik_bootstrap=True,
            defer_template_robot_bootstrap=True,
        ),
        scene=SceneSettleCollisionConfig(settle_substeps=settle_substeps),
        domain_randomization=dataclasses.replace(
            BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=_NUM_ENVS).domain_randomization,
            topology_seed=21,
        ),
        settle_diagnostics=None,
        obs=ObsConfig(allocate_buffers=True),
    )


def _vic_cacheable_config(*, settle_substeps: int = 8) -> BatchedHeterogeneousCoupledSimConfig:
    return dataclasses.replace(
        _cacheable_config(settle_substeps=settle_substeps),
        controller=ControllerConfig(mode="vic"),
    )


def test_init_minimal_smoke(ranges, per_env_params):
    cfg = _vbd_placeholder_config()
    sim = BatchedHeterogeneousCoupledSim(cfg, per_env_params, ranges, use_settle_cache=False)
    assert sim.num_envs == _NUM_ENVS
    assert sim.layout is not None
    assert sim.layout.num_envs == _NUM_ENVS
    body_q = sim.scene.cable.state_0.body_q.numpy().reshape(-1, 7)
    for w, apple_idx in enumerate(sim.layout.apple_body_indices):
        if apple_idx < 0:
            continue
        z = float(body_q[apple_idx, 2])
        assert z > -0.05, f"world {w} apple fell: z={z}"


def test_placeholder_kind_warns(ranges, per_env_params):
    cfg = _vbd_placeholder_config()
    with pytest.warns(UserWarning, match="CPU host nudge"):
        BatchedHeterogeneousCoupledSim(cfg, per_env_params, ranges, use_settle_cache=False)


def test_fr3_missing_assets_warns_and_builds(ranges, per_env_params):
    cfg = dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=_NUM_ENVS),
        robot=RobotConfig(kind="fr3", step_mode="vbd_only", fix_to_apple=False),
        scene=SceneSettleCollisionConfig(settle_substeps=0),
    )
    with patch(
        "apple_pick_sim.coupled_fruiting.batched_heterogeneous_build.fr3_robot.fr3_assets_available",
        return_value=False,
    ):
        with pytest.warns(UserWarning, match="placeholder TCP"):
            with pytest.warns(UserWarning, match="CPU host nudge"):
                sim = BatchedHeterogeneousCoupledSim(
                    cfg, per_env_params, ranges, use_settle_cache=False
                )
    assert sim.layout is not None


@requires_fr3
def test_settle_cache_miss_then_hit(tmp_path, ranges, per_env_params):
    cfg = _cacheable_config(settle_substeps=8)
    sim1 = BatchedHeterogeneousCoupledSim(
        cfg,
        per_env_params,
        ranges,
        use_settle_cache=True,
        settle_cache_dir=tmp_path,
    )
    assert sim1.settle_cache_path is not None
    assert sim1.settle_cache_path.is_file()
    assert sim1.settled_checkpoint is not None
    bq1 = sim1.settled_checkpoint.body_q.copy()

    sim2 = BatchedHeterogeneousCoupledSim(
        cfg,
        per_env_params,
        ranges,
        use_settle_cache=True,
        settle_cache_dir=tmp_path,
    )
    assert sim2.build_result.settle_stability_reports is None
    np.testing.assert_array_equal(sim2.settled_checkpoint.body_q, bq1)


@requires_fr3
def test_force_settle_overwrites_cache(tmp_path, ranges, per_env_params):
    cfg = _cacheable_config(settle_substeps=8)
    sim1 = BatchedHeterogeneousCoupledSim(
        cfg, per_env_params, ranges, settle_cache_dir=tmp_path
    )
    path = sim1.settle_cache_path
    assert path is not None
    mtime1 = path.stat().st_mtime
    bq1 = sim1.settled_checkpoint.body_q.copy()

    time.sleep(0.02)
    sim2 = BatchedHeterogeneousCoupledSim(
        cfg,
        per_env_params,
        ranges,
        force_settle=True,
        settle_cache_dir=tmp_path,
    )
    assert path.stat().st_mtime >= mtime1
    assert sim2.settled_checkpoint is not None
    assert sim2.settled_checkpoint.body_q.shape == bq1.shape


@requires_fr3
def test_use_settle_cache_false_ignores_disk(tmp_path, ranges, per_env_params):
    cfg = _cacheable_config(settle_substeps=8)
    BatchedHeterogeneousCoupledSim(cfg, per_env_params, ranges, settle_cache_dir=tmp_path)
    path = list(tmp_path.glob("*.npz"))[0]
    tampered = SettledCheckpoint.load(path)
    bad_bq = np.zeros_like(tampered.body_q)
    tampered = SettledCheckpoint(body_q=bad_bq, metadata=tampered.metadata)
    tampered.save(path)

    sim = BatchedHeterogeneousCoupledSim(
        cfg,
        per_env_params,
        ranges,
        use_settle_cache=False,
        settle_cache_dir=tmp_path,
    )
    assert not np.allclose(sim.settled_checkpoint.body_q, bad_bq)


@requires_fr3
def test_checkpoint_validate_rejects_mismatch(tmp_path, ranges, per_env_params):
    cfg = _cacheable_config(settle_substeps=8)
    sim = BatchedHeterogeneousCoupledSim(cfg, per_env_params, ranges, settle_cache_dir=tmp_path)
    ckpt = sim.settled_checkpoint
    assert ckpt is not None
    bad_cfg = dataclasses.replace(
        cfg,
        scene=dataclasses.replace(cfg.scene, settle_substeps=cfg.scene.settle_substeps + 1),
    )
    with pytest.raises(ValueError, match="settle_substeps"):
        ckpt.validate_against(config=bad_cfg, ranges=ranges, per_env_params=per_env_params)


def test_step_clips_speed(ranges, per_env_params):
    torch = _require_torch()
    cfg = dataclasses.replace(
        _vbd_placeholder_config(),
        robot=RobotConfig(kind="placeholder", step_mode="coupled", fix_to_apple=False),
        scene=SceneSettleCollisionConfig(settle_substeps=0),
    )
    sim = BatchedHeterogeneousCoupledSim(cfg, per_env_params, ranges, use_settle_cache=False)
    big = torch.full((sim.num_envs, 6), 10.0, dtype=torch.float32, device=sim.device)
    clipped = sim._clip_actions(big)
    for i in range(sim.num_envs):
        assert float(torch.linalg.norm(clipped[i, :3])) <= cfg.controller.linear_speed + 1e-5
        assert float(torch.linalg.norm(clipped[i, 3:6])) <= cfg.controller.angular_speed + 1e-5


def test_step_vbd_only_rejects_actions(ranges, per_env_params):
    torch = _require_torch()
    cfg = _vbd_placeholder_config()
    sim = BatchedHeterogeneousCoupledSim(cfg, per_env_params, ranges, use_settle_cache=False)
    actions = torch.zeros(sim.num_envs, 6, dtype=torch.float32, device=sim.device)
    with pytest.raises(ValueError, match="vbd_only"):
        sim.step(actions)


def test_step_coupled_smoke(ranges, per_env_params):
    cfg = dataclasses.replace(
        _vbd_placeholder_config(),
        robot=RobotConfig(kind="placeholder", step_mode="coupled", fix_to_apple=False),
        scene=SceneSettleCollisionConfig(settle_substeps=0),
    )
    sim = BatchedHeterogeneousCoupledSim(cfg, per_env_params, ranges, use_settle_cache=False)
    for _ in range(3):
        sim.step(None)
    body_q = sim.scene.cable.state_0.body_q.numpy().reshape(-1, 7)
    assert np.isfinite(body_q).all()
    for w, apple_idx in enumerate(sim.layout.apple_body_indices):
        if apple_idx < 0:
            continue
        z = float(body_q[apple_idx, 2])
        assert z > -0.1


def test_gather_obs_keys(ranges, per_env_params):
    cfg = _vbd_placeholder_config()
    sim = BatchedHeterogeneousCoupledSim(cfg, per_env_params, ranges, use_settle_cache=False)
    sim.step(None)
    obs = sim.gather_obs()
    assert "apple_pos" in obs
    assert "proxy_pos" in obs
    assert obs["apple_pos"].shape == (sim.num_envs, 3)


@requires_fr3
@pytest.mark.slow
def test_fr3_per_env_actions_diverge_targets(ranges, per_env_params):
    cfg = _cacheable_config(settle_substeps=8)
    sim = BatchedHeterogeneousCoupledSim(cfg, per_env_params, ranges, use_settle_cache=False)
    ctrl = sim._ee_ctrl
    assert ctrl is not None

    torch = _require_torch()
    actions = torch.zeros(sim.num_envs, 6, dtype=torch.float32, device=sim.device)
    actions[0, 0] = 0.002

    t0_init = wp.transform_get_translation(ctrl.target_tf[0])
    t1_init = wp.transform_get_translation(ctrl.target_tf[1])
    sim.step(actions)
    t0 = wp.transform_get_translation(ctrl.target_tf[0])
    t1 = wp.transform_get_translation(ctrl.target_tf[1])
    assert float(t0[0] - t0_init[0]) > float(t1[0] - t1_init[0]) + 1e-5


@requires_fr3
@pytest.mark.slow
def test_vic_per_env_actions_differ_wrench_damping(ranges, per_env_params):
    cfg = _vic_cacheable_config(settle_substeps=8)
    sim = BatchedHeterogeneousCoupledSim(cfg, per_env_params, ranges, use_settle_cache=False)
    ctrl = sim._ee_ctrl
    assert ctrl is not None
    assert sim.scene.vic_use_joint_torques

    ctrl.sync_target_from_state(sim.scene.robot_state_0)

    torch = _require_torch()
    actions = torch.zeros(sim.num_envs, 6, dtype=torch.float32, device=sim.device)
    actions[0, 0] = 0.05

    sim.step(actions)
    wp.synchronize()
    wrenches = sim.scene.vic_jt_wrench_buf.numpy()
    assert float(wrenches[0, 0]) > 1.0, "world 0 expected D-term force from per-env action"
    assert abs(float(wrenches[1, 0])) < float(wrenches[0, 0]) * 0.5, (
        "world 1 should see weaker x force than world 0 with distinct actions"
    )
