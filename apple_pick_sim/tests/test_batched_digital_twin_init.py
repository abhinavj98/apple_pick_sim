"""Tests for batched digital-twin init helpers."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from apple_pick_gym.batched_envs import ApplePickBatchedSysIdEnv
from apple_pick_gym.batched_envs.batched_sysid_collect import (
    collect_batched_quasi_static_dataset,
    sample_and_broadcast_structure_params,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ControllerConfig,
    ObsConfig,
    RobotConfig,
    SceneSettleCollisionConfig,
)
from apple_pick_sim.fruiting_system.params import FruitingSystemParams
from apple_pick_sim.system_id import BatchedSysIdDataset, QuasiStaticStepConfig
from apple_pick_sim.system_id.batched_digital_twin_init import (
    digital_twin_obs_from_batched_episode,
    infer_base_params_for_structure,
    initialize_batched_env_from_dataset,
)
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


@pytest.fixture(scope="module")
def tiny_batched_dataset(tmp_path_factory) -> BatchedSysIdDataset:
    if not _maybe_import_gymnasium() or not fr3_assets_available():
        pytest.skip("requires gymnasium and FR3 assets")
    num_structures = 1
    num_directions = 1
    num_envs = num_structures * num_directions
    config = QuasiStaticStepConfig(
        movement_per_step_m=0.02,
        total_movement_m=0.04,
        move_speed_mps=0.2,
        hold_duration_s=0.1,
        skip_return=True,
    )
    per_env_params = sample_and_broadcast_structure_params(
        RANGES_FIXTURE,
        topology_seed=_SEED,
        num_structures=num_structures,
        num_directions=num_directions,
    )
    output_dir = tmp_path_factory.mktemp("batched_digital_twin")
    env = ApplePickBatchedSysIdEnv(
        num_envs=num_envs,
        max_episode_steps=120,
        ranges_path=RANGES_FIXTURE,
        topology_seed=_SEED,
        use_settle_cache=False,
        sim_config=_test_sim_config(num_envs=num_envs),
        per_env_params=per_env_params,
        control_hz=float(config.control_hz),
    )
    try:
        collect_batched_quasi_static_dataset(
            env,
            num_structures=num_structures,
            num_directions=num_directions,
            config=config,
            output_dir=output_dir,
            seed=_SEED,
            ranges_path=RANGES_FIXTURE,
            max_steps=20,
        )
    finally:
        env.close()
    return BatchedSysIdDataset(output_dir)


@gymnasium_available
@requires_fr3
def test_digital_twin_obs_from_batched_episode_has_junction_names(
    tiny_batched_dataset: BatchedSysIdDataset,
):
    obs = digital_twin_obs_from_batched_episode(tiny_batched_dataset, 0, 0)
    assert obs.junction_names
    assert len(obs.woody_part_start_pos) == 3 * len(obs.junction_names)
    assert len(obs.woody_part_end_pos) == 3 * len(obs.junction_names)
    assert obs.fruiting_base_pos is not None
    assert obs.weld_direction is not None


@gymnasium_available
@requires_fr3
def test_infer_base_params_for_structure_returns_fruiting_params(
    tiny_batched_dataset: BatchedSysIdDataset,
):
    params = infer_base_params_for_structure(tiny_batched_dataset, 0)
    assert isinstance(params, FruitingSystemParams)
    assert params.primary is not None
    assert params.primary.bend_stiffness > 0


@gymnasium_available
@requires_fr3
def test_initialize_batched_env_from_dataset_sets_joint_q_and_tcp(
    tiny_batched_dataset: BatchedSysIdDataset,
):
    structure_idx = 0
    num_directions = 1
    params = infer_base_params_for_structure(tiny_batched_dataset, structure_idx)
    arrays = tiny_batched_dataset.load_episode_obs_arrays(structure_idx, 0)
    recorded_joint_q = np.asarray(arrays["robot_joint_q"][0], dtype=np.float32).reshape(-1)
    recorded_tcp_pos = np.asarray(arrays["tcp_pos"][0], dtype=np.float32).reshape(3)

    env = ApplePickBatchedSysIdEnv(
        num_envs=1,
        max_episode_steps=10,
        ranges_path=RANGES_FIXTURE,
        topology_seed=_SEED,
        use_settle_cache=False,
        sim_config=_test_sim_config(num_envs=1),
        per_env_params=[params],
    )
    try:
        env.reset(seed=_SEED)
        initialize_batched_env_from_dataset(
            env,
            tiny_batched_dataset,
            structure_idx=structure_idx,
            num_directions=num_directions,
        )
        env._gather_obs()
        exported = env.sysid_numpy_obs(0)
        np.testing.assert_allclose(
            exported["robot_joint_q"],
            recorded_joint_q,
            rtol=0,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            exported["tcp_pos"],
            recorded_tcp_pos,
            rtol=0,
            atol=1e-3,
        )
    finally:
        env.close()
