"""Tests for batched sys-ID MMD grid replay export to Parquet datasets."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from apple_pick_gym.batched_envs import batched_sysid_mmd_grid as grid
from apple_pick_sim.fruiting_system import fruiting_params_from_json, params as fs
from apple_pick_sim.system_id.batched_replay_export import (
    ReplayCandidateSpec,
    candidate_export_exists,
    candidate_replay_dir,
    export_replay_candidates_for_structure,
    write_replay_candidate_dataset,
)
from apple_pick_sim.system_id.batched_trajectory_store import (
    BatchedSysIdDataset,
    SCHEMA_VERSION,
    write_manifest,
)
from apple_pick_sim.tests.conftest import RANGES_FIXTURE


def _sample_params(seed: int = 0):
    ranges = fs.load_ranges(RANGES_FIXTURE)
    return fs.sample_params(ranges, seed=seed)


def _synthetic_replay_arrays(*, n_frames: int = 3, direction_idx: int = 0) -> dict:
    junction_names = ["joint_0", "joint_1"]
    return {
        "action": np.zeros((n_frames, 6), dtype=np.float32),
        "ft_wrist": np.ones((n_frames, 6), dtype=np.float32) * (direction_idx + 1),
        "tcp_velocity": np.zeros((n_frames, 6), dtype=np.float32),
        "tcp_pos": np.full((n_frames, 3), 0.1, dtype=np.float32),
        "apple_pos": np.full((n_frames, 3), 0.2, dtype=np.float32),
        "phase": np.array([0, 1, 1], dtype=np.int8),
        "dir_idx": np.full(n_frames, direction_idx, dtype=np.int32),
        "excitation_type": np.zeros(n_frames, dtype=np.int8),
        "excitation_direction": np.tile(
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            (n_frames, 1),
        ),
        "woody_part_start_pos": {
            name: np.full((n_frames, 3), float(i), dtype=np.float32)
            for i, name in enumerate(junction_names)
        },
        "woody_part_end_pos": {
            name: np.full((n_frames, 3), float(i) + 0.5, dtype=np.float32)
            for i, name in enumerate(junction_names)
        },
        "junction_names": junction_names,
    }


def _synthetic_gt_metadata(*, direction_idx: int = 0) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": str(uuid4()),
        "structure_idx": 0,
        "direction_idx": direction_idx,
        "env_idx": direction_idx,
        "pull_direction": [1.0, 0.0, 0.0],
        "params_fingerprint": json.dumps({"topology": "t_junction"}),
        "fruiting_system_params": json.dumps({"topology": "t_junction", "apple_radius": 0.04}),
        "excitation_type": "quasi_static",
        "control_hz": 30.0,
        "seed": 0,
        "n_woody_parts": 2,
        "junction_names": ["joint_0", "joint_1"],
        "initial_tcp_pos": [0.0, 0.0, 0.5],
        "initial_tcp_quat": [0.0, 0.0, 0.0, 1.0],
        "initial_apple_pos": [0.4, 0.5, 0.6],
        "initial_apple_quat": [0.0, 0.1, 0.0, 0.995],
        "initial_robot_joint_q": [0.0] * 7,
        "fixture_path": str(RANGES_FIXTURE),
        "fruiting_base_pos": None,
        "apple_radius": 0.04,
        "rod_radii": json.dumps({"primary": 0.01}),
        "weld_direction": [0.0, 0.0, 1.0],
        "weld_reference_pos": [0.4, 0.5, 0.6],
        "weld_reference_quat": [0.0, 0.1, 0.0, 0.995],
        "movement_per_step_m": 0.02,
        "total_movement_m": 0.10,
        "hold_duration_s": 2.0,
        "move_speed_mps": 0.05,
        "skip_return": True,
    }


def _mock_source_dataset(tmp_path: Path, *, num_directions: int = 2) -> BatchedSysIdDataset:
    out = tmp_path / "source_gt"
    episodes: list[dict] = []
    for direction_idx in range(num_directions):
        meta = _synthetic_gt_metadata(direction_idx=direction_idx)
        from apple_pick_sim.system_id.batched_trajectory_store import BatchedEpisodeWriter, episode_filename

        writer = BatchedEpisodeWriter(episode_id=meta["episode_id"])
        gt_arrays = _synthetic_replay_arrays(n_frames=3, direction_idx=direction_idx)
        for frame_idx in range(3):
            obs = {
                "excitation_type": 0,
                "excitation_direction": gt_arrays["excitation_direction"][frame_idx],
                "tcp_velocity": gt_arrays["tcp_velocity"][frame_idx],
                "ft_wrist": gt_arrays["ft_wrist"][frame_idx],
                "raw_ft_wrist": gt_arrays["ft_wrist"][frame_idx],
                "tcp_pos": gt_arrays["tcp_pos"][frame_idx],
                "apple_pos": gt_arrays["apple_pos"][frame_idx],
                "tcp_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "apple_quat": np.array([0.0, 0.1, 0.0, 0.995], dtype=np.float32),
                "robot_joint_q": np.zeros(7, dtype=np.float32),
                "woody_part_force": np.zeros(12, dtype=np.float32),
                "woody_part_start_pos": {
                    name: gt_arrays["woody_part_start_pos"][name][frame_idx]
                    for name in gt_arrays["junction_names"]
                },
                "woody_part_end_pos": {
                    name: gt_arrays["woody_part_end_pos"][name][frame_idx]
                    for name in gt_arrays["junction_names"]
                },
            }
            phase_name = ("move_out", "hold", "hold")[frame_idx]
            writer.record_step(
                step_idx=frame_idx,
                sim_time=float(frame_idx) / 30.0,
                phase=phase_name,
                amplitude_m=0.02 * float(frame_idx),
                action=gt_arrays["action"][frame_idx],
                obs=obs,
            )
        rel = episode_filename(0, direction_idx)
        writer.save(out / rel, meta)
        episodes.append(
            {
                "structure_idx": 0,
                "direction_idx": direction_idx,
                "env_idx": direction_idx,
                "filename": rel,
                "episode_id": meta["episode_id"],
                "pull_direction": meta["pull_direction"],
                "n_frames": writer.n_frames,
            }
        )
    write_manifest(
        out,
        command_argv=["test"],
        collection={
            "seed": 0,
            "topology_seed": 42,
            "num_structures": 1,
            "num_directions": num_directions,
            "control_hz": 30.0,
            "ranges_path": str(RANGES_FIXTURE),
        },
        structures=[
            {
                "structure_idx": 0,
                "params_fingerprint": "{}",
                "junction_names": ["joint_0", "joint_1"],
                "n_woody_parts": 2,
            }
        ],
        episodes=episodes,
        overwrite=True,
    )
    return BatchedSysIdDataset(out)


def test_candidate_replay_dir_and_exists(tmp_path: Path):
    root = candidate_replay_dir(tmp_path, structure_idx=1, candidate_index=7)
    assert root == tmp_path / "structure_001" / "candidates" / "c007"
    assert not candidate_export_exists(root)
    root.mkdir(parents=True)
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    assert candidate_export_exists(root)


def test_write_replay_candidate_dataset_round_trip(tmp_path: Path):
    source = _mock_source_dataset(tmp_path, num_directions=2)
    base = _sample_params(seed=0)
    candidate = grid.BendStiffnessCandidate(primary=11.0, secondary=1.0, spur=22.0, stem=33.0)
    candidate_params = candidate.apply_to(base)

    replay_eps = [
        _synthetic_replay_arrays(n_frames=3, direction_idx=d) for d in range(2)
    ]
    spec = ReplayCandidateSpec(
        candidate_index=2,
        params=candidate_params,
        stiffnesses={
            "primary": float(candidate.primary),
            "secondary": float(candidate.secondary),
            "spur": float(candidate.spur),
            "stem": float(candidate.stem),
        },
    )
    out_dir = candidate_replay_dir(tmp_path / "replay", structure_idx=0, candidate_index=2)
    written = write_replay_candidate_dataset(
        out_dir,
        source_dataset=source,
        source_structure_idx=0,
        spec=spec,
        replay_eps_by_direction=replay_eps,
        command_argv=["test-export"],
    )
    assert written == out_dir
    assert candidate_export_exists(out_dir)

    ds = BatchedSysIdDataset(out_dir)
    assert ds.manifest["collection"]["num_directions"] == 2
    assert ds.manifest["collection"]["num_structures"] == 1
    assert ds.manifest["replay"]["candidate_index"] == 2
    assert len(ds.episode_entries()) == 2

    meta = ds.load_episode_metadata(0, 0)
    loaded_params = fruiting_params_from_json(str(meta["fruiting_system_params"]))
    assert loaded_params.primary is not None
    assert loaded_params.primary.bend_stiffness == pytest.approx(11.0)
    assert loaded_params.spur is not None
    assert loaded_params.spur.bend_stiffness == pytest.approx(22.0)
    assert loaded_params.stem is not None
    assert loaded_params.stem.bend_stiffness == pytest.approx(33.0)

    arrays = ds.load_episode_obs_arrays(0, 1)
    assert arrays["ft_wrist"].shape == (3, 6)
    assert float(arrays["ft_wrist"][0, 0]) == pytest.approx(2.0)


def test_write_replay_candidate_dataset_skip_existing(tmp_path: Path):
    source = _mock_source_dataset(tmp_path, num_directions=1)
    base = _sample_params(seed=0)
    candidate_params = base
    spec = ReplayCandidateSpec(
        candidate_index=0,
        params=candidate_params,
        stiffnesses={"primary": 1.0, "secondary": 0.0, "spur": 1.0, "stem": 1.0},
    )
    out_dir = candidate_replay_dir(tmp_path / "replay", structure_idx=0, candidate_index=0)
    write_replay_candidate_dataset(
        out_dir,
        source_dataset=source,
        source_structure_idx=0,
        spec=spec,
        replay_eps_by_direction=[_synthetic_replay_arrays(n_frames=3, direction_idx=0)],
        command_argv=["test-export"],
    )
    first_mtime = (out_dir / "manifest.json").stat().st_mtime
    skipped = write_replay_candidate_dataset(
        out_dir,
        source_dataset=source,
        source_structure_idx=0,
        spec=spec,
        replay_eps_by_direction=[_synthetic_replay_arrays(n_frames=99, direction_idx=0)],
        command_argv=["test-export"],
        skip_existing=True,
    )
    assert skipped is None
    assert (out_dir / "manifest.json").stat().st_mtime == first_mtime
    ds = BatchedSysIdDataset(out_dir)
    assert ds.load_episode_obs_arrays(0, 0)["ft_wrist"].shape[0] == 3


def test_export_replay_candidates_for_structure_with_collectors(tmp_path: Path):
    source = _mock_source_dataset(tmp_path, num_directions=2)
    base = _sample_params(seed=0)
    candidates = [
        grid.BendStiffnessCandidate(5.0, 1.0, 6.0, 7.0),
        grid.BendStiffnessCandidate(8.0, 1.0, 9.0, 10.0),
    ]
    recorded_by_env = []
    for cand_idx in range(len(candidates)):
        for direction_idx in range(2):
            recorded = _synthetic_replay_arrays(n_frames=2, direction_idx=direction_idx)
            recorded_by_env.append(recorded)

    collectors = grid.BatchedSysIdReplayCollectors(num_envs=4, recorded_by_env=recorded_by_env)
    for env_idx in range(4):
        for frame_idx in range(2):
            obs = {
                "ft_wrist": np.full(6, float(env_idx + 1), dtype=np.float32),
                "tcp_velocity": np.zeros(6, dtype=np.float32),
                "tcp_pos": np.zeros(3, dtype=np.float32),
                "apple_pos": np.zeros(3, dtype=np.float32),
                "woody_part_start_pos": {
                    name: np.zeros(3, dtype=np.float32)
                    for name in recorded["junction_names"]
                },
                "woody_part_end_pos": {
                    name: np.ones(3, dtype=np.float32)
                    for name in recorded["junction_names"]
                },
            }
            recorded = recorded_by_env[env_idx]
            adapted = grid.replay_obs_dict_from_sysid_numpy(
                obs,
                junction_names=list(recorded["junction_names"]),
            )
            collectors._collectors[env_idx].record(adapted, frame_idx=frame_idx)

    n_written = export_replay_candidates_for_structure(
        tmp_path / "replay",
        source_dataset=source,
        source_structure_idx=0,
        specs_and_replays=[
            (
                ReplayCandidateSpec(
                    candidate_index=i,
                    params=candidates[i].apply_to(base),
                    stiffnesses={
                        "primary": candidates[i].primary,
                        "secondary": candidates[i].secondary,
                        "spur": candidates[i].spur,
                        "stem": candidates[i].stem,
                    },
                ),
                grid.direction_episodes_from_collectors(
                    collectors,
                    candidate_index=i,
                    num_directions=2,
                ),
            )
            for i in range(len(candidates))
        ],
        command_argv=["test-batch-export"],
    )
    assert n_written == 2
    ds = BatchedSysIdDataset(
        candidate_replay_dir(tmp_path / "replay", structure_idx=0, candidate_index=1)
    )
    assert ds.manifest["replay"]["candidate_stiffnesses"]["primary"] == pytest.approx(8.0)
