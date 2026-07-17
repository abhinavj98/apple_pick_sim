"""Tests for ApplePickBatchedSysIdEnv (V.4.2 collection env)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from apple_pick_gym.batched_envs import ApplePickBatchedSysIdEnv
from apple_pick_gym.batched_envs.apple_pick_batched_base_env import ApplePickBatchedBaseEnv
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ControllerConfig,
    ObsConfig,
    RobotConfig,
    SceneSettleCollisionConfig,
)
from apple_pick_sim.system_id.excitation_state import ExcitationContext
from apple_pick_sim.tests.conftest import RANGES_FIXTURE, fr3_assets_available

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
        controller=ControllerConfig(mode="vic", linear_speed=1.0, angular_speed=1.0),
        domain_randomization=dataclasses.replace(
            BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=num_envs).domain_randomization,
            topology_seed=_SEED,
        ),
        obs=ObsConfig(allocate_buffers=True),
    )


def _make_env(*, num_envs: int = 1) -> ApplePickBatchedSysIdEnv:
    return ApplePickBatchedSysIdEnv(
        num_envs=num_envs,
        max_episode_steps=10,
        ranges_path=RANGES_FIXTURE,
        topology_seed=_SEED,
        use_settle_cache=False,
        sim_config=_test_sim_config(num_envs=num_envs),
    )


_SYSID_OBS_KEYS = frozenset(
    {
        "woody_part_start_pos",
        "woody_part_end_pos",
        "woody_part_force",
        "apple_pos",
        "apple_quat",
        "tcp_pos",
        "tcp_quat",
        "tcp_velocity",
        "ft_wrist",
        "raw_ft_wrist",
        "robot_joint_q",
        "excitation_type",
        "excitation_f_inst",
        "excitation_direction",
    }
)


def test_batched_sysid_env_forwards_per_env_grippers(monkeypatch):
    per_env_grippers = (object(), object())
    captured: dict[str, object] = {}

    def _fake_base_init(self, **kwargs):
        captured.update(kwargs)
        self.num_envs = int(kwargs["num_envs"])
        self.device = torch.device("cpu")

    monkeypatch.setattr(ApplePickBatchedBaseEnv, "__init__", _fake_base_init)

    ApplePickBatchedSysIdEnv(
        num_envs=2,
        per_env_grippers=per_env_grippers,
        use_settle_cache=False,
    )

    assert captured["per_env_grippers"] is per_env_grippers


@gymnasium_available
@requires_fr3
def test_batched_sysid_obs_shapes_and_sysid_numpy_export():
    env = _make_env(num_envs=2)
    try:
        obs, info = env.reset(seed=_SEED)
        assert info["obs_layout"] == "batched_sysid"
        assert obs["tcp_pos"].shape == (2, 3)
        assert obs["tcp_quat"].shape == (2, 4)
        assert obs["apple_quat"].shape == (2, 4)
        assert obs["robot_joint_q"].shape[0] == 2
        assert obs["excitation_direction"].shape == (2, 3)
        assert len(info["per_env"]) == 2
        assert len(info["weld_direction"]) == 2

        exported = env.sysid_numpy_obs(1)
        assert _SYSID_OBS_KEYS <= frozenset(exported)
        assert exported["ft_wrist"].shape == (6,)
        assert exported["excitation_direction"].shape == (3,)
    finally:
        env.close()


@gymnasium_available
@requires_fr3
def test_batched_sysid_excitation_context_round_trip():
    env = _make_env(num_envs=2)
    try:
        env.reset(seed=_SEED)
        direction = np.array([0.2, -0.3, 0.9], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        ctx = ExcitationContext(type="quasi_static", f_inst=0.0, direction=direction)
        env.set_excitation_context(1, ctx)
        obs = env._gather_obs()
        np.testing.assert_allclose(
            obs["excitation_direction"][1].detach().cpu().numpy(),
            direction.astype(np.float32),
            rtol=0,
            atol=1e-6,
        )
        assert int(obs["excitation_type"][1].item()) == 0
        exported = env.sysid_numpy_obs(1)
        np.testing.assert_allclose(
            exported["excitation_direction"],
            direction.astype(np.float32),
            rtol=0,
            atol=1e-6,
        )
    finally:
        env.close()


@gymnasium_available
@requires_fr3
def test_batched_sysid_step_preserves_export_contract():
    env = _make_env(num_envs=1)
    try:
        env.reset(seed=_SEED)
        actions = torch.zeros((1, 6), dtype=torch.float32, device=env.device)
        actions[0, 0] = 0.05
        obs, *_rest = env.step(actions)
        exported = env.sysid_numpy_obs(0)
        assert _SYSID_OBS_KEYS <= frozenset(exported)
        assert np.isfinite(exported["ft_wrist"]).all()
        assert obs["tcp_pos"].shape == (1, 3)
    finally:
        env.close()
