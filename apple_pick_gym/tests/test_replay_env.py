"""Tests for ApplePickReplayEnv dataset-backed trajectory replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apple_pick_sim.system_id import EpisodeMeta, TrajectoryWriter
from apple_pick_sim.tests.conftest import fr3_assets_available


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


def _write_minimal_dataset(tmp_path: Path, *, n_frames: int = 3) -> str:
    episode_id = "replay-ep-1"
    junction_names = ["joint_0", "joint_1"]
    writer = TrajectoryWriter(episode_id=episode_id)
    for i in range(n_frames):
        writer.record_step(
            step_idx=i,
            sim_time=i / 60.0,
            phase="hold",
            dir_idx=0,
            amplitude_m=0.05,
            action=np.array([float(i + 1), 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            obs={
                "excitation_type": 0,
                "excitation_direction": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "tcp_velocity": np.zeros(6, dtype=np.float32),
                "woody_part_start_pos": {
                    name: np.zeros(3, dtype=np.float32) for name in junction_names
                },
                "woody_part_end_pos": {
                    name: np.zeros(3, dtype=np.float32) for name in junction_names
                },
                "ft_wrist": np.zeros(6, dtype=np.float32),
                "tcp_pos": np.zeros(3, dtype=np.float32),
                "apple_pos": np.zeros(3, dtype=np.float32),
                "woody_part_force": np.zeros(12, dtype=np.float32),
            },
        )
    writer.save(
        tmp_path,
        EpisodeMeta(
            episode_id=episode_id,
            weld_direction=(0.0, 0.0, 1.0),
            excitation_type="quasi_static",
            n_woody_parts=2,
            junction_names=junction_names,
            params_fingerprint=json.dumps({"stem_bend_stiffness": 30.0}),
            control_hz=60.0,
            n_directions=1,
            skip_return=True,
        ),
    )
    return episode_id


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


torch_available = pytest.mark.skipif(
    not _torch_available(),
    reason="PyTorch required for VIC joint torques (uv sync --extra vic)",
)


@gymnasium_available
@torch_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_replay_env_loads_episode(tmp_path: Path):
    from apple_pick_gym.envs import ApplePickReplayEnv

    episode_id = _write_minimal_dataset(tmp_path)
    env = ApplePickReplayEnv(
        max_episode_steps=64,
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=0,
        robot_facing_weld=False,
    )
    env.load_dataset(tmp_path, episode_id=episode_id)
    obs, info = env.reset(seed=3)
    assert obs["ft_wrist"].shape == (6,)
    assert info["replay_episode_id"] == episode_id
    env.close()


@gymnasium_available
@torch_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_replay_env_ignores_external_action(tmp_path: Path):
    from apple_pick_gym.envs import ApplePickReplayEnv

    episode_id = _write_minimal_dataset(tmp_path, n_frames=2)
    env = ApplePickReplayEnv(
        max_episode_steps=64,
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=0,
        robot_facing_weld=False,
    )
    env.load_dataset(tmp_path, episode_id=episode_id)
    env.reset(seed=3)
    _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
    replay_action = np.asarray(info["replay_action"], dtype=np.float32)
    np.testing.assert_allclose(replay_action, np.array([1.0, 0, 0, 0, 0, 0], dtype=np.float32))
    env.close()


@gymnasium_available
@torch_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_replay_env_episode_terminates_at_end(tmp_path: Path):
    from apple_pick_gym.envs import ApplePickReplayEnv

    episode_id = _write_minimal_dataset(tmp_path, n_frames=2)
    env = ApplePickReplayEnv(
        max_episode_steps=64,
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=0,
        robot_facing_weld=False,
    )
    env.load_dataset(tmp_path, episode_id=episode_id)
    env.reset(seed=3)
    _, _, _, truncated0, _ = env.step(np.zeros(6, dtype=np.float32))
    assert not truncated0
    _, _, _, truncated1, _ = env.step(np.zeros(6, dtype=np.float32))
    assert truncated1
    env.close()


@gymnasium_available
@torch_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_replay_env_obs_keys(tmp_path: Path):
    from apple_pick_gym.envs import ApplePickReplayEnv

    episode_id = _write_minimal_dataset(tmp_path, n_frames=1)
    env = ApplePickReplayEnv(
        max_episode_steps=64,
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=0,
        robot_facing_weld=False,
    )
    env.load_dataset(tmp_path, episode_id=episode_id)
    obs, _ = env.reset(seed=3)
    for key in (
        "ft_wrist",
        "woody_start",
        "woody_end",
        "tcp_velocity",
        "tcp_pos",
        "apple_pos",
    ):
        assert key in obs
    env.close()
