"""Tests for offline exclude_unstable_episodes filter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from apple_pick_gym.batched_envs.exclude_unstable_episodes import (
    FILTERED_MANIFEST_NAME,
    PRE_EXCLUDE_BACKUP_NAME,
    exclude_unstable_episodes,
)
from apple_pick_sim.system_id import BatchedEpisodeWriter, BatchedSysIdDataset, write_manifest
from apple_pick_sim.system_id.batched_trajectory_store import SCHEMA_VERSION, episode_filename


def _synthetic_obs(*, n_woody: int = 2) -> dict:
    names = [f"joint_{i}" for i in range(n_woody)]
    return {
        "excitation_type": 0,
        "excitation_direction": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "tcp_velocity": np.arange(6, dtype=np.float32),
        "woody_part_start_pos": {
            name: np.arange(3, dtype=np.float32) + float(i * 3)
            for i, name in enumerate(names)
        },
        "woody_part_end_pos": {
            name: np.arange(3, dtype=np.float32) + float(i * 3) + 0.5
            for i, name in enumerate(names)
        },
        "ft_wrist": np.arange(6, dtype=np.float32) + 1.0,
        "raw_ft_wrist": np.arange(6, dtype=np.float32) + 101.0,
        "tcp_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "apple_pos": np.array([0.4, 0.5, 0.6], dtype=np.float32),
        "tcp_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "apple_quat": np.array([0.0, 0.1, 0.0, 0.995], dtype=np.float32),
        "robot_joint_q": np.linspace(0.0, 0.6, 7, dtype=np.float32),
        "woody_part_force": np.arange(n_woody * 6, dtype=np.float32),
    }


def _meta(episode_id: str, *, direction_idx: int) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "structure_idx": 0,
        "direction_idx": direction_idx,
        "env_idx": direction_idx,
        "pull_direction": [0.0, 1.0, 0.0],
        "params_fingerprint": "{}",
        "fruiting_system_params": "{}",
        "excitation_type": "quasi_static",
        "control_hz": 30.0,
        "seed": 0,
        "n_woody_parts": 2,
        "junction_names": ["joint_0", "joint_1"],
    }


def _write_two_episode_dataset(
    tmp_path: Path,
    *,
    stability_by_direction: tuple[tuple[bool, ...], ...] = ((True,), (False,)),
) -> Path:
    for direction_idx, stability in enumerate(stability_by_direction):
        writer = BatchedEpisodeWriter(episode_id=f"ep-{direction_idx}")
        for step_idx, stable in enumerate(stability):
            writer.record_step(
                step_idx=step_idx,
                sim_time=float(step_idx) / 30.0,
                phase="hold",
                amplitude_m=0.0,
                action=np.zeros(6, dtype=np.float32),
                obs=_synthetic_obs(),
                stable=stable,
            )
        writer.save(
            tmp_path / episode_filename(0, direction_idx),
            _meta(f"ep-{direction_idx}", direction_idx=direction_idx),
        )
    write_manifest(
        tmp_path,
        command_argv=["collect.py"],
        collection={
            "seed": 0,
            "num_structures": 1,
            "num_directions": len(stability_by_direction),
        },
        structures=[
            {
                "structure_idx": 0,
                "junction_names": ["joint_0", "joint_1"],
                "n_woody_parts": 2,
            }
        ],
        episodes=[
            {
                "structure_idx": 0,
                "direction_idx": 0,
                "env_idx": 0,
                "filename": episode_filename(0, 0),
                "episode_id": "ep-0",
                "pull_direction": [0.0, 1.0, 0.0],
                "n_frames": len(stability_by_direction[0]),
            },
            {
                "structure_idx": 0,
                "direction_idx": 1,
                "env_idx": 1,
                "filename": episode_filename(0, 1),
                "episode_id": "ep-1",
                "pull_direction": [0.0, 1.0, 0.0],
                "n_frames": len(stability_by_direction[1]),
            },
        ],
        overwrite=True,
    )
    return tmp_path


def test_exclude_ignores_single_unstable_frame(tmp_path: Path):
    root = _write_two_episode_dataset(tmp_path)
    out = exclude_unstable_episodes(root, inplace=False)
    assert out.name == FILTERED_MANIFEST_NAME
    filtered = BatchedSysIdDataset(root, manifest_name=FILTERED_MANIFEST_NAME)
    eps = filtered.episode_entries()
    # one-of-one unstable is frac=1.0 > 0.25 → excluded; all-stable stays
    assert eps[0]["excluded"] is False
    assert eps[1]["excluded"] is True
    assert eps[1]["excluded_reason"] == "stability_blowup"


def test_exclude_uses_strict_25_percent_instability_boundary(tmp_path: Path):
    root = _write_two_episode_dataset(
        tmp_path,
        stability_by_direction=(
            (False, True, True, True),
            (False, False, True, True),
        ),
    )

    exclude_unstable_episodes(root, inplace=False)

    episodes = BatchedSysIdDataset(
        root,
        manifest_name=FILTERED_MANIFEST_NAME,
    ).episode_entries()
    assert episodes[0]["excluded"] is False
    assert episodes[1]["excluded"] is True
    assert episodes[1]["excluded_reason"] == "stability_blowup"


def test_exclude_inplace_backs_up_then_rewrites(tmp_path: Path):
    root = _write_two_episode_dataset(tmp_path)
    out = exclude_unstable_episodes(root, inplace=True)
    assert out.name == "manifest.json"
    assert (root / PRE_EXCLUDE_BACKUP_NAME).exists()
    dataset = BatchedSysIdDataset(root)
    eps = dataset.episode_entries()
    assert eps[0]["excluded"] is False
    assert eps[1]["excluded"] is True
