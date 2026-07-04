"""Capstone: batched collect → observation-only replay → GT error (V.4.2)."""

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
from apple_pick_sim.fruiting_system import fruiting_params_from_json
from apple_pick_sim.system_id import QuasiStaticStepConfig, TrajectoryDataset
from apple_pick_sim.system_id.trajectory_store import PHASE_TO_INT
from apple_pick_sim.tests.conftest import RANGES_FIXTURE, fr3_assets_available

_SEED = 42
_HOLD_PHASE = PHASE_TO_INT["hold"]

# Measured observation-only replay floor (test_minimal, 2026-07-04):
# mean |dtcp| ~32 mm, mean ft_wrist RMSE ~58 N for S=2,D=2 grid.
_MAX_MEAN_TCP_POS_MM = 45.0
_MAX_MEAN_FT_WRIST_RMSE = 90.0


def _maybe_import_gymnasium():
    try:
        import gymnasium as gym  # noqa: F401

        return True
    except Exception:
        return False


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


gymnasium_available = pytest.mark.skipif(
    not _maybe_import_gymnasium(),
    reason="gymnasium not installed",
)

torch_available = pytest.mark.skipif(
    not _torch_available(),
    reason="PyTorch required for VIC replay",
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


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64).reshape(-1) - np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(d * d)))


def _norm_diff(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64).reshape(-1) - np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(d))


def _replay_episode_hold_errors(dataset_dir: Path, episode_id: str) -> tuple[list[float], list[float]]:
    """Return (hold tcp_pos_mm, hold ft_wrist_rmse) lists for one episode."""
    from apple_pick_gym.envs import ApplePickReplayEnv

    import pyarrow.parquet as pq

    dataset = TrajectoryDataset(dataset_dir)
    meta = dataset.load_episode_meta(episode_id)
    recorded = dataset.load_episode_obs_arrays(episode_id)
    phases = pq.read_table(dataset_dir / "frames" / f"{episode_id}.parquet").column("phase").to_pylist()

    params = fruiting_params_from_json(str(meta["fruiting_system_params"]))
    env = ApplePickReplayEnv(
        max_episode_steps=int(recorded["action"].shape[0]),
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=0,
        robot_facing_weld=False,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    tcp_mm: list[float] = []
    ft_rmse: list[float] = []
    try:
        env.load_dataset(dataset_dir, episode_id=episode_id)
        env.reset(seed=int(meta["seed"]), options={"params": params})
        n_frames = int(recorded["action"].shape[0])
        for _ in range(n_frames):
            obs, _reward, _terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
            frame_idx = int(info.get("replay_frame_idx", 0))
            if frame_idx < len(phases) and int(phases[frame_idx]) == _HOLD_PHASE:
                live_ft = np.asarray(obs["ft_wrist"], dtype=np.float32)
                rec_ft = recorded["ft_wrist"][frame_idx]
                live_tcp = np.asarray(obs["tcp_pos"], dtype=np.float32)
                rec_tcp = recorded["tcp_pos"][frame_idx]
                tcp_mm.append(1000.0 * _norm_diff(live_tcp, rec_tcp))
                ft_rmse.append(_rmse(live_ft, rec_ft))
            if truncated:
                break
    finally:
        env.close()
    return tcp_mm, ft_rmse


@gymnasium_available
@torch_available
@requires_fr3
def test_batched_collect_replay_hold_phase_fidelity(tmp_path: Path):
    num_structures = 2
    num_directions = 2
    num_envs = num_structures * num_directions
    config = QuasiStaticStepConfig(
        movement_per_step_m=0.02,
        total_movement_m=0.04,
        move_speed_mps=0.2,
        hold_duration_s=0.15,
        skip_return=True,
    )
    per_env_params = sample_and_broadcast_structure_params(
        RANGES_FIXTURE,
        topology_seed=_SEED,
        num_structures=num_structures,
        num_directions=num_directions,
    )
    collect_env = ApplePickBatchedSysIdEnv(
        num_envs=num_envs,
        max_episode_steps=200,
        ranges_path=RANGES_FIXTURE,
        topology_seed=_SEED,
        use_settle_cache=False,
        sim_config=_test_sim_config(num_envs=num_envs),
        per_env_params=per_env_params,
        control_hz=float(config.control_hz),
    )
    dataset_dir = tmp_path / "batched_gt"
    try:
        collect_batched_quasi_static_dataset(
            collect_env,
            num_structures=num_structures,
            num_directions=num_directions,
            config=config,
            output_dir=dataset_dir,
            seed=_SEED,
            ranges_path=RANGES_FIXTURE,
            max_steps=90,
        )
    finally:
        collect_env.close()

    dataset = TrajectoryDataset(dataset_dir)
    episode_ids = dataset.episode_ids()
    assert len(episode_ids) == num_envs

    all_tcp_mm: list[float] = []
    all_ft_rmse: list[float] = []
    for episode_id in episode_ids:
        tcp_mm, ft_rmse = _replay_episode_hold_errors(dataset_dir, episode_id)
        assert tcp_mm, f"episode {episode_id} produced no hold-frame comparisons"
        all_tcp_mm.extend(tcp_mm)
        all_ft_rmse.extend(ft_rmse)

    mean_tcp_mm = float(np.mean(all_tcp_mm))
    mean_ft_rmse = float(np.mean(all_ft_rmse))
    assert mean_tcp_mm < _MAX_MEAN_TCP_POS_MM, (
        f"mean hold |dtcp|={mean_tcp_mm:.2f} mm exceeds {_MAX_MEAN_TCP_POS_MM}"
    )
    assert mean_ft_rmse < _MAX_MEAN_FT_WRIST_RMSE, (
        f"mean hold ft_wrist RMSE={mean_ft_rmse:.2f} N exceeds {_MAX_MEAN_FT_WRIST_RMSE}"
    )
