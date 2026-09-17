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
    cma_collect_junction_names,
    collect_batched_quasi_static_dataset,
    sample_and_broadcast_structure_params,
    structure_and_direction_indices,
)
from apple_pick_sim.system_id.mmd_features import CMA_WOODY_JUNCTIONS
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
    PRE_WELD_STEP_IDX,
    episode_filename,
    frame_index_for_step,
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


def test_cma_collect_junction_names_filters_t_junction_topology():
    """T-junction envs with the full CMA set collapse to the two CMA names."""
    env_names = [
        "primary_support_left",
        "primary_support_right",
        "primary_spur",
        "spur_stem",
        "stem_apple",
    ]
    assert cma_collect_junction_names(env_names) == list(CMA_WOODY_JUNCTIONS)


def test_cma_collect_junction_names_keeps_full_names_for_secondary_topology():
    """Secondary topologies missing primary_spur keep the full name list."""
    env_names = ["primary_support_left", "primary_support_right", "stem_apple"]
    assert cma_collect_junction_names(env_names) == env_names


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

        shared = assign_pull_directions(
            env,
            num_structures=2,
            num_directions=2,
            shared_across_structures=True,
        )
        assert np.allclose(shared[0], shared[2], atol=1e-12)
        assert np.allclose(shared[1], shared[3], atol=1e-12)
        assert not np.allclose(shared[0], shared[1], atol=1e-3)
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
    sim_config = manifest["collection"]["sim_config"]
    assert sim_config["controller"]["mode"] == "vic"
    assert sim_config["settle_substeps"] == 8
    assert "joint_angular_kd_overrides" in sim_config
    assert "joint_linear_kd_overrides" in sim_config

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
    step_idxs = frames.column("step_idx").to_pylist()
    assert PRE_WELD_STEP_IDX in step_idxs
    arrays = dataset.load_episode_obs_arrays(0, 0)
    assert arrays["action"].shape[0] == frames.num_rows
    assert frame_index_for_step(arrays, PRE_WELD_STEP_IDX) == 0


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
    assert seen[0][1] == "pre_weld"
    assert any(idx >= 0 for idx, _ in seen)
    assert max(idx for idx, _ in seen) == stop_at


def test_build_manifest_episodes_marks_excluded_envs():
    from apple_pick_gym.batched_envs.batched_sysid_collect import _build_manifest_episodes
    from apple_pick_sim.system_id import BatchedEpisodeWriter

    writers = [BatchedEpisodeWriter(episode_id="a"), BatchedEpisodeWriter(episode_id="b")]
    # one dummy stable frame each so n_frames > 0 is not required for builder
    meta = [
        {
            "structure_idx": 0,
            "direction_idx": 0,
            "env_idx": 0,
            "episode_id": "a",
            "pull_direction": [1.0, 0.0, 0.0],
        },
        {
            "structure_idx": 0,
            "direction_idx": 1,
            "env_idx": 1,
            "episode_id": "b",
            "pull_direction": [0.0, 1.0, 0.0],
        },
    ]
    episodes = _build_manifest_episodes(
        meta,
        writers,
        num_directions=2,
        excluded_env_indices={1},
    )
    assert episodes[0]["excluded"] is False
    assert episodes[0]["excluded_reason"] is None
    assert episodes[1]["excluded"] is True
    assert episodes[1]["excluded_reason"] == "stability_blowup"


def test_excluded_env_indices_only_sticky_disabled_not_force_cap_frames():
    """Force-cap (stable=False) frames must not exclude; sticky-disable must."""
    import torch

    from apple_pick_gym.batched_envs.batched_sysid_collect import _excluded_env_indices
    from apple_pick_gym.batched_envs.env_disable_controller import EnvDisableController
    from apple_pick_sim.system_id import BatchedEpisodeWriter

    writers = [
        BatchedEpisodeWriter(episode_id="force_cap_only"),
        BatchedEpisodeWriter(episode_id="sticky"),
    ]
    # Minimal obs for record_step
    obs = {
        "excitation_type": 0,
        "excitation_direction": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "tcp_velocity": np.zeros(6, dtype=np.float32),
        "woody_part_start_pos": {"j0": np.zeros(3, dtype=np.float32)},
        "woody_part_end_pos": {"j0": np.ones(3, dtype=np.float32)},
        "ft_wrist": np.zeros(6, dtype=np.float32),
        "raw_ft_wrist": np.zeros(6, dtype=np.float32),
        "tcp_pos": np.zeros(3, dtype=np.float32),
        "apple_pos": np.zeros(3, dtype=np.float32),
        "tcp_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "apple_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot_joint_q": np.zeros(7, dtype=np.float32),
        "woody_part_force": np.zeros(6, dtype=np.float32),
    }
    writers[0].record_step(
        step_idx=0,
        sim_time=0.0,
        phase="hold",
        amplitude_m=0.0,
        action=np.zeros(6, dtype=np.float32),
        obs=obs,
        stable=False,
    )
    writers[1].record_step(
        step_idx=0,
        sim_time=0.0,
        phase="hold",
        amplitude_m=0.0,
        action=np.zeros(6, dtype=np.float32),
        obs=obs,
        stable=True,
    )
    ctrl = EnvDisableController(2, device="cpu")
    ctrl.update(torch.tensor([False, True], dtype=torch.bool))

    excluded = _excluded_env_indices(ctrl, writers)
    assert excluded == {1}
