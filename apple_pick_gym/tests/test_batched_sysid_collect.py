"""Tests for batched sys-ID Parquet collection helpers."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from apple_pick_gym.batched_envs import ApplePickBatchedSysIdEnv
from apple_pick_gym.batched_envs.batched_sysid_collect import (
    assign_pull_directions,
    broadcast_structure_params,
    collect_batched_quasi_static_dataset,
    sample_and_broadcast_structure_params,
    structure_and_direction_indices,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ControllerConfig,
    ObsConfig,
    RobotConfig,
    SceneSettleCollisionConfig,
)
from apple_pick_sim.fruiting_system import load_ranges, sample_heterogeneous_params_list
from apple_pick_sim.system_id import BatchedSysIdDataset, QuasiStaticStepConfig
from apple_pick_sim.system_id.batched_trajectory_store import (
    BATCHED_REQUIRED_FRAME_COLUMNS,
    episode_filename,
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


def test_structure_grid_index_helpers():
    assert structure_and_direction_indices(0, 3) == (0, 0)
    assert structure_and_direction_indices(4, 3) == (1, 1)
    assert structure_and_direction_indices(5, 3) == (1, 2)


def test_broadcast_structure_params_repeats_each_structure():
    ranges = load_ranges(RANGES_FIXTURE)
    structures = sample_heterogeneous_params_list(ranges, topology_seed=7, num_envs=2)
    out = broadcast_structure_params(structures, 3)
    assert len(out) == 6
    assert out[0] is structures[0]
    assert out[2] is structures[0]
    assert out[3] is structures[1]


def test_sample_and_broadcast_structure_params_count():
    out = sample_and_broadcast_structure_params(
        RANGES_FIXTURE,
        topology_seed=11,
        num_structures=2,
        num_directions=4,
    )
    assert len(out) == 8


@gymnasium_available
@requires_fr3
def test_assign_pull_directions_unit_vectors():
    env = ApplePickBatchedSysIdEnv(
        num_envs=4,
        max_episode_steps=4,
        ranges_path=RANGES_FIXTURE,
        topology_seed=_SEED,
        use_settle_cache=False,
        sim_config=_test_sim_config(num_envs=4),
        per_env_params=sample_and_broadcast_structure_params(
            RANGES_FIXTURE,
            topology_seed=_SEED,
            num_structures=2,
            num_directions=2,
        ),
    )
    try:
        env.reset(seed=_SEED)
        dirs = assign_pull_directions(env, num_structures=2, num_directions=2)
        assert len(dirs) == 4
        for d in dirs:
            assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-5
        assert not np.allclose(dirs[0], dirs[1], atol=1e-3)
        assert not np.allclose(dirs[2], dirs[3], atol=1e-3)
    finally:
        env.close()


@gymnasium_available
@requires_fr3
def test_collect_batched_quasi_static_dataset_writes_v1_layout(tmp_path: Path):
    num_structures = 2
    num_directions = 2
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
        out = collect_batched_quasi_static_dataset(
            env,
            num_structures=num_structures,
            num_directions=num_directions,
            config=config,
            output_dir=tmp_path / "dataset",
            seed=_SEED,
            ranges_path=RANGES_FIXTURE,
            max_steps=60,
            command_argv=["test_collect", "--num-structures", "2"],
        )
    finally:
        env.close()

    assert (out / "manifest.json").is_file()
    assert not (out / "metadata.parquet").exists()
    assert not (out / "frames").exists()
    assert (out / episode_filename(0, 0)).is_file()

    dataset = BatchedSysIdDataset(out)
    manifest = dataset.manifest
    assert manifest["command_argv"] == ["test_collect", "--num-structures", "2"]
    assert len(manifest["episodes"]) == num_envs
    assert len(manifest["structures"]) == num_structures

    ep00 = dataset.load_episode_metadata(0, 0)
    ep01 = dataset.load_episode_metadata(0, 1)
    ep10 = dataset.load_episode_metadata(1, 0)
    assert ep00["params_fingerprint"] == ep01["params_fingerprint"]
    assert ep00["params_fingerprint"] != ep10["params_fingerprint"]
    assert ep00["pull_direction"] != ep01["pull_direction"]

    frames = dataset.load_episode_frames(0, 0)
    for col in BATCHED_REQUIRED_FRAME_COLUMNS:
        assert col in frames.column_names
    assert frames.num_rows > 0
    arrays = dataset.load_episode_obs_arrays(0, 0)
    assert arrays["action"].shape[0] == frames.num_rows


@gymnasium_available
@requires_fr3
def test_collect_appends_timestamp_when_dataset_exists(tmp_path: Path):
    num_structures = 1
    num_directions = 1
    num_envs = 1
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
    base = tmp_path / "dataset"
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
        first = collect_batched_quasi_static_dataset(
            env,
            num_structures=num_structures,
            num_directions=num_directions,
            config=config,
            output_dir=base,
            seed=_SEED,
            ranges_path=RANGES_FIXTURE,
            max_steps=20,
        )
        assert first == base
        with pytest.warns(UserWarning, match="already exists"):
            second = collect_batched_quasi_static_dataset(
                env,
                num_structures=num_structures,
                num_directions=num_directions,
                config=config,
                output_dir=base,
                seed=_SEED,
                ranges_path=RANGES_FIXTURE,
                max_steps=20,
            )
    finally:
        env.close()

    assert second != base
    assert second.name.startswith("dataset_")
    assert BatchedSysIdDataset(base).episode_entries()
    assert BatchedSysIdDataset(second).episode_entries()


@gymnasium_available
@requires_fr3
def test_collect_batched_on_step_callback_invoked_and_can_stop(tmp_path: Path):
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
    seen: list[tuple[int, str]] = []
    stop_at = 3

    def on_step(*, step_idx: int, phase: str, **_kwargs) -> bool:
        seen.append((step_idx, phase))
        return step_idx < stop_at

    try:
        collect_batched_quasi_static_dataset(
            env,
            num_structures=num_structures,
            num_directions=num_directions,
            config=config,
            output_dir=tmp_path / "dataset",
            seed=_SEED,
            ranges_path=RANGES_FIXTURE,
            max_steps=60,
            on_step=on_step,
        )
    finally:
        env.close()

    assert seen[0][0] == -1
    assert seen[0][1] == "init"
    assert any(idx >= 0 for idx, _ in seen)
    assert max(idx for idx, _ in seen) == stop_at
