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
from apple_pick_sim.fruiting_system.params import GripperProxyConfig
from apple_pick_sim.system_id.batched_trajectory_store import (
    FIRST_TRAJECTORY_STEP_IDX,
    frame_index_for_step,
)
from apple_pick_sim.fruiting_system import fruiting_params_from_json
from apple_pick_sim.system_id.batched_digital_twin_init import (
    digital_twin_obs_from_batched_episode,
    gripper_proxy_from_episode_metadata,
    infer_base_params_for_structure,
    initialize_batched_env_from_dataset,
    true_params_for_structure,
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


def test_gripper_proxy_from_episode_metadata_sets_weld_fields():
    meta = {
        "weld_direction": [0.1, 0.2, 0.97],
        "weld_reference_pos": [1.0, 2.0, 3.0],
        "weld_reference_quat": [0.0, 0.0, 0.0, 1.0],
    }
    base = GripperProxyConfig(mass=0.5)

    proxy = gripper_proxy_from_episode_metadata(meta, base=base)

    assert proxy.fix_to_apple is True
    assert proxy.robot_facing_weld is False
    assert proxy.mass == pytest.approx(0.5)
    assert proxy.weld_direction == pytest.approx((0.1, 0.2, 0.97))
    assert proxy.weld_reference_pos == pytest.approx((1.0, 2.0, 3.0))
    assert proxy.weld_reference_quat == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_gripper_proxy_from_episode_metadata_robot_facing_when_no_weld_direction():
    meta: dict = {}
    proxy = gripper_proxy_from_episode_metadata(meta)
    assert proxy.fix_to_apple is True
    assert proxy.robot_facing_weld is True
    assert proxy.weld_direction is None


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
def test_true_params_for_structure_returns_exact_sampled_params(
    tiny_batched_dataset: BatchedSysIdDataset,
):
    true_params = true_params_for_structure(tiny_batched_dataset, 0)
    meta = tiny_batched_dataset.load_episode_metadata(0, 0)
    from_metadata = fruiting_params_from_json(str(meta["fruiting_system_params"]))
    inferred = infer_base_params_for_structure(tiny_batched_dataset, 0)

    assert true_params.primary is not None
    assert from_metadata.primary is not None
    assert inferred.primary is not None

    assert true_params.primary.length == pytest.approx(from_metadata.primary.length)
    assert true_params.primary.direction == pytest.approx(from_metadata.primary.direction)
    assert true_params.primary.bend_stiffness == pytest.approx(
        from_metadata.primary.bend_stiffness
    )

    length_delta = abs(inferred.primary.length - true_params.primary.length)
    direction_delta = 1.0 - abs(
        float(np.dot(np.asarray(inferred.primary.direction), np.asarray(true_params.primary.direction)))
    )
    assert length_delta > 1e-6 or direction_delta > 1e-6


@gymnasium_available
@requires_fr3
def test_initialize_batched_env_from_dataset_sets_joint_q_and_tcp(
    tiny_batched_dataset: BatchedSysIdDataset,
):
    structure_idx = 0
    num_directions = 1
    params = infer_base_params_for_structure(tiny_batched_dataset, structure_idx)
    arrays = tiny_batched_dataset.load_episode_obs_arrays(structure_idx, 0)
    first_trajectory_frame = frame_index_for_step(arrays, FIRST_TRAJECTORY_STEP_IDX)
    recorded_joint_q = np.asarray(
        arrays["robot_joint_q"][first_trajectory_frame], dtype=np.float32
    ).reshape(-1)
    recorded_tcp_pos = np.asarray(arrays["tcp_pos"][first_trajectory_frame], dtype=np.float32).reshape(3)

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


def test_initialize_batched_env_from_dataset_raises_without_joint_q():
    from unittest.mock import MagicMock

    env = MagicMock()
    env.num_envs = 1
    env._sim.layout = MagicMock()
    dataset = MagicMock()
    dataset.load_episode_obs_arrays.return_value = {
        "excitation_direction": np.zeros((1, 3), dtype=np.float32),
    }
    dataset.load_episode_metadata.return_value = {}

    with pytest.raises(ValueError, match="missing initial robot_joint_q"):
        initialize_batched_env_from_dataset(
            env,
            dataset,
            structure_idx=0,
            num_directions=1,
        )
