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
from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
    BendStiffnessCandidate,
    gt_bend_stiffness_candidate_from_structure,
    replay_batched_sysid_structure,
    trajectory_hold_aggregated_mse,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ControllerConfig,
    ObsConfig,
    RobotConfig,
    SceneSettleCollisionConfig,
)
from apple_pick_sim.fruiting_system import fruiting_params_from_json
from apple_pick_sim.system_id import BatchedSysIdDataset, QuasiStaticStepConfig
from apple_pick_sim.system_id.batched_trajectory_store import materialize_legacy_episode_dir
from apple_pick_sim.system_id.trajectory_store import PHASE_TO_INT, TrajectoryDataset
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
        runtime=dataclasses.replace(
            BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=num_envs).runtime,
            device="cpu",
        ),
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


@gymnasium_available
@torch_available
@requires_fr3
def test_batched_cpu_snapshot_gt_self_replay_is_near_deterministic(tmp_path: Path):
    # Small CPU determinism probe: collect + save snapshot, then GT self-replay with --use-snapshot.
    num_structures = 1
    num_directions = 1
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
    dataset_dir = tmp_path / "batched_cpu_det"
    try:
        collect_batched_quasi_static_dataset(
            collect_env,
            num_structures=num_structures,
            num_directions=num_directions,
            config=config,
            output_dir=dataset_dir,
            seed=_SEED,
            ranges_path=RANGES_FIXTURE,
            max_steps=60,
            save_snapshot=True,
        )
    finally:
        collect_env.close()

    dataset = BatchedSysIdDataset(dataset_dir)
    gt = gt_bend_stiffness_candidate_from_structure(dataset, structure_idx=0)
    candidates = [BendStiffnessCandidate(gt.primary, gt.secondary, gt.spur, gt.stem)]

    def _build_env_fn(*, num_envs: int, per_env_params: list, max_episode_steps: int, gripper=None):
        # replay_batched_sysid_structure passes gripper metadata; ApplePickBatchedSysIdEnv
        # does not currently accept it, but the sysid determinism test does not depend on it.
        del gripper
        return ApplePickBatchedSysIdEnv(
            num_envs=int(num_envs),
            max_episode_steps=int(max_episode_steps),
            ranges_path=RANGES_FIXTURE,
            topology_seed=_SEED,
            use_settle_cache=False,
            sim_config=_test_sim_config(num_envs=int(num_envs)),
            per_env_params=per_env_params,
            control_hz=float(config.control_hz),
        )

    collectors = replay_batched_sysid_structure(
        dataset=dataset,
        structure_idx=0,
        candidates=candidates,
        num_directions=num_directions,
        seed=_SEED,
        build_env_fn=_build_env_fn,
        use_snapshot=True,
        replay_sim_config=_test_sim_config(num_envs=num_directions * len(candidates)),
    )

    # Compare the single GT candidate env to recorded.
    replay = collectors.to_arrays(0)
    recorded = collectors._recorded_by_env[0]
    metrics = trajectory_hold_aggregated_mse(replay=replay, recorded=recorded, aggregation="mean")
    tcp_mm = 1000.0 * float(np.sqrt(float(metrics["tcp_pos_mse"])))
    ft_rmse = float(metrics["ft_force_rmse"])
    print(f"[batched cpu use-snapshot] hold tcp_rmse={tcp_mm:.3f} mm  ft_force_rmse={ft_rmse:.6g} N")

    assert tcp_mm < 2.0
    assert ft_rmse < 1.0


def _replay_episode_hold_errors(dataset_dir: Path, episode_id: str) -> tuple[list[float], list[float]]:
    """Return (hold tcp_pos_mm, hold ft_wrist_rmse) lists for one legacy-layout episode."""
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
        device="cpu",
    )
    tcp_mm: list[float] = []
    ft_rmse: list[float] = []
    try:
        env.load_dataset(dataset_dir, episode_id=episode_id)
        try:
            env.reset(seed=int(meta["seed"]), options={"params": params})
        except Exception as e:
            # IK bootstrap can fail for some random fixtures; treat as "no comparison"
            # for this lightweight fidelity probe.
            from apple_pick_sim.robot.fr3_robot.placement import IKBootstrapConvergenceError

            if isinstance(e, IKBootstrapConvergenceError):
                return [], []
            raise
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


def _replay_episode_hold_errors_with_snapshot(
    *,
    batched_dataset_dir: Path,
    legacy_dataset_dir: Path,
    structure_idx: int,
    direction_idx: int,
    episode_id: str,
) -> tuple[list[float], list[float]]:
    """Same as _replay_episode_hold_errors, but uses saved grasp snapshot init."""
    from apple_pick_gym.envs import ApplePickReplayEnv

    import pyarrow.parquet as pq

    from apple_pick_sim.system_id.batched_episode_snapshot_io import load_npz_for_direction
    from apple_pick_sim.system_id.trajectory_store import load_grasp_snapshot_into_env

    dataset = TrajectoryDataset(legacy_dataset_dir)
    meta = dataset.load_episode_meta(episode_id)
    recorded = dataset.load_episode_obs_arrays(episode_id)
    phases = pq.read_table(legacy_dataset_dir / "frames" / f"{episode_id}.parquet").column("phase").to_pylist()

    params = fruiting_params_from_json(str(meta["fruiting_system_params"]))
    batched_snap = load_npz_for_direction(
        batched_dataset_dir,
        structure_idx=int(structure_idx),
        direction_idx=int(direction_idx),
    )
    # Convert batched EpisodeStateSnapshot arrays into the grasp snapshot schema expected by
    # apple_pick_sim.system_id.trajectory_store.load_grasp_snapshot_into_env.
    snap = {
        "robot_body_q": np.asarray(batched_snap["robot_body_q"], dtype=np.float32),
        "robot_body_qd": np.asarray(batched_snap["robot_body_qd"], dtype=np.float32),
        "robot_joint_q": np.asarray(batched_snap["robot_joint_q"], dtype=np.float32),
        "robot_joint_qd": np.asarray(batched_snap["robot_joint_qd"], dtype=np.float32),
        "cable_body_q": np.asarray(batched_snap["cable_body_q_0"], dtype=np.float32),
        "cable_body_qd": np.asarray(batched_snap["cable_body_qd_0"], dtype=np.float32),
        "cable_state_1_body_q": np.asarray(batched_snap["cable_body_q_1"], dtype=np.float32),
        "cable_state_1_body_qd": np.asarray(batched_snap["cable_body_qd_1"], dtype=np.float32),
        "vic_target_tf": np.concatenate(
            [
                np.asarray(batched_snap["vic_target_pos"], dtype=np.float32).reshape(3),
                np.asarray(batched_snap["vic_target_rot"], dtype=np.float32).reshape(4),
            ],
            axis=0,
        ).astype(np.float32),
    }

    env = ApplePickReplayEnv(
        max_episode_steps=int(recorded["action"].shape[0]),
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=0,
        robot_facing_weld=False,
        mujoco_solver_kwargs={"disable_contacts": True},
        device="cpu",
    )
    tcp_mm: list[float] = []
    ft_rmse: list[float] = []
    try:
        env.load_dataset(legacy_dataset_dir, episode_id=episode_id)
        # Reset builds the sim; snapshot is applied immediately after.
        try:
            env.reset(seed=int(meta["seed"]), options={"params": params})
        except Exception as e:
            from apple_pick_sim.robot.fr3_robot.placement import IKBootstrapConvergenceError

            if isinstance(e, IKBootstrapConvergenceError):
                return [], []
            raise
        load_grasp_snapshot_into_env(env, snap)

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
    # Keep this small: we want a quick, robust CPU fidelity check.
    num_structures = 1
    num_directions = 1
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
            max_steps=60,
            save_snapshot=True,
        )
    finally:
        collect_env.close()

    batched = BatchedSysIdDataset(dataset_dir)
    assert len(batched.episode_entries()) == num_envs

    legacy_dir = tmp_path / "legacy_replay"
    legacy_dir.mkdir()
    all_tcp_mm: list[float] = []
    all_ft_rmse: list[float] = []
    for entry in batched.episode_entries():
        s_idx = int(entry["structure_idx"])
        d_idx = int(entry["direction_idx"])
        materialize_legacy_episode_dir(
            batched,
            structure_idx=s_idx,
            direction_idx=d_idx,
            output_dir=legacy_dir,
        )
        episode_id = str(entry["episode_id"])
        tcp_mm, ft_rmse = _replay_episode_hold_errors(legacy_dir, episode_id)
        if not tcp_mm:
            continue
        all_tcp_mm.extend(tcp_mm)
        all_ft_rmse.extend(ft_rmse)

    assert all_tcp_mm, "no-snapshot replay produced no hold-frame comparisons (IK bootstrap failures?)"
    mean_tcp_mm = float(np.mean(all_tcp_mm))
    mean_ft_rmse = float(np.mean(all_ft_rmse))
    print(f"[cpu no-snapshot replay] mean hold |dtcp| = {mean_tcp_mm:.2f} mm")
    print(f"[cpu no-snapshot replay] mean hold ft_wrist RMSE = {mean_ft_rmse:.2f} N")
    assert mean_tcp_mm < _MAX_MEAN_TCP_POS_MM, (
        f"mean hold |dtcp|={mean_tcp_mm:.2f} mm exceeds {_MAX_MEAN_TCP_POS_MM}"
    )
    assert mean_ft_rmse < _MAX_MEAN_FT_WRIST_RMSE, (
        f"mean hold ft_wrist RMSE={mean_ft_rmse:.2f} N exceeds {_MAX_MEAN_FT_WRIST_RMSE}"
    )

    # NOTE: We intentionally do NOT assert tight determinism bounds for the single-env replay
    # env's grasp snapshot restore. Tight determinism is covered by the batched CPU snapshot
    # self-replay test: test_batched_cpu_snapshot_gt_self_replay_is_near_deterministic.
