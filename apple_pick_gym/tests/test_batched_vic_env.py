"""Tests for ApplePickBatchedVicEnv (V.3.3)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from apple_pick_gym.batched_envs import ApplePickBatchedVicEnv
from apple_pick_gym.batched_envs.obs_torch import legacy_v3_numpy_from_batched
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ControllerConfig,
    ObsConfig,
    RobotConfig,
    SceneSettleCollisionConfig,
)
from apple_pick_sim.tests.conftest import RANGES_FIXTURE, fr3_assets_available

_NUM_ENVS = 4
_SEED = 42


def _maybe_import_gymnasium():
    try:
        import gymnasium as gym  # noqa: F401

        return True
    except Exception:
        return False


gymnasium_available = pytest.mark.skipif(
    not _maybe_import_gymnasium(),
    reason="gymnasium not installed",
)

requires_fr3 = pytest.mark.skipif(
    not fr3_assets_available(),
    reason="Requires bundled assets/fr3 and usd-core",
)


def _test_sim_config(*, num_envs: int) -> BatchedHeterogeneousCoupledSimConfig:
    return dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=num_envs),
        robot=RobotConfig(
            kind="fr3",
            step_mode="coupled",
            fix_to_apple=True,
            skip_ik_bootstrap=False,
            defer_template_robot_bootstrap=False,
            force_batched_layout=True,
        ),
        scene=SceneSettleCollisionConfig(settle_substeps=8),
        controller=ControllerConfig(mode="vic"),
        domain_randomization=dataclasses.replace(
            BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=num_envs).domain_randomization,
            topology_seed=_SEED,
        ),
        obs=ObsConfig(allocate_buffers=True),
    )


def _make_env(*, num_envs: int = 1) -> ApplePickBatchedVicEnv:
    return ApplePickBatchedVicEnv(
        num_envs=num_envs,
        max_episode_steps=10,
        ranges_path=RANGES_FIXTURE,
        topology_seed=_SEED,
        use_settle_cache=False,
        sim_config=_test_sim_config(num_envs=num_envs),
    )


@gymnasium_available
@requires_fr3
def test_batched_vic_env_skrl_contract():
    pytest.importorskip("torch")
    env = _make_env(num_envs=_NUM_ENVS)
    try:
        assert env.num_envs == _NUM_ENVS
        assert env.device.type == "cpu"
        obs, info = env.reset(seed=_SEED)
        assert info["obs_schema"] == "v3"
        assert info["obs_layout"] == "batched_vic"
        assert obs["apple_pos"].shape == (_NUM_ENVS, 3)
        assert obs["ft_wrist"].shape == (_NUM_ENVS, 6)
        for name in env.junction_names:
            assert obs["woody_part_info"][name]["anchors_pos"].shape == (_NUM_ENVS, 6)

        actions = torch.zeros((_NUM_ENVS, 6), dtype=torch.float32, device=env.device)
        obs2, reward, terminated, truncated, info2 = env.step(actions)
        assert reward.shape == (_NUM_ENVS, 1)
        assert terminated.shape == (_NUM_ENVS, 1)
        assert truncated.shape == (_NUM_ENVS, 1)
        assert obs2["apple_pos"].shape == (_NUM_ENVS, 3)
        assert info2["step_count"] == 1
    finally:
        env.close()


@gymnasium_available
@requires_fr3
def test_reset_restores_episode_baseline():
    pytest.importorskip("torch")
    env = _make_env(num_envs=2)
    try:
        obs0, _ = env.reset(seed=_SEED)
        tcp0 = obs0["apple_pos"].clone()
        actions = torch.zeros((2, 6), dtype=torch.float32, device=env.device)
        actions[0, 0] = 0.08
        for _ in range(3):
            env.step(actions)
        obs1, _ = env.reset()
        np.testing.assert_allclose(
            obs1["apple_pos"].cpu().numpy(),
            tcp0.cpu().numpy(),
            rtol=1e-5,
            atol=1e-4,
        )
    finally:
        env.close()


@gymnasium_available
@requires_fr3
def test_batched_vic_obs_layout_after_reset():
    """Post-reset obs has woody_part_info layout and finite batched tensors."""
    pytest.importorskip("torch")
    env = _make_env(num_envs=1)
    try:
        obs, info = env.reset(seed=_SEED)
        assert info["obs_layout"] == "batched_vic"
        assert obs["apple_pos"].shape == (1, 3)
        assert torch.isfinite(obs["apple_pos"]).all()
        for name in env.junction_names:
            part = obs["woody_part_info"][name]
            assert part["anchors_pos"].shape == (1, 6)
            assert part["anchor_force"].shape == (1, 6)
            assert torch.isfinite(part["anchors_pos"]).all()
            assert torch.isfinite(part["anchor_force"]).all()
        mapped = legacy_v3_numpy_from_batched(obs, env.junction_names)
        assert "woody_part_start_pos" in mapped
        assert "ft_wrist" in mapped
        assert mapped["ft_wrist"].shape == (6,)
    finally:
        env.close()
