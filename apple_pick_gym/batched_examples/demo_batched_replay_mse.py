"""Demo: digital-twin replay vs recorded GT with per-trajectory MSE.

Collects a tiny batched_sysid_v1 dataset, replays with GT stiffness vs a
deliberately wrong candidate, and prints MSE against recorded observations.

Run from repo root::

    uv run python apple_pick_gym/batched_examples/demo_batched_replay_mse.py
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import numpy as np

from apple_pick_gym.batched_envs import ApplePickBatchedSysIdEnv
from apple_pick_gym.batched_envs.batched_sysid_collect import (
    collect_batched_quasi_static_dataset,
    sample_and_broadcast_structure_params,
)
from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
    BendStiffnessCandidate,
    direction_episodes_from_collectors,
    gt_bend_stiffness_candidate_from_structure,
    replay_candidates_for_structure,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ControllerConfig,
    ObsConfig,
    RobotConfig,
    SceneSettleCollisionConfig,
)
from apple_pick_sim.system_id import BatchedSysIdDataset, QuasiStaticStepConfig
from apple_pick_sim.system_id.batched_digital_twin_init import (
    digital_twin_obs_from_batched_episode,
    initialize_batched_env_from_dataset,
)
from apple_pick_sim.system_id.mmd_features import flatten_woody_positions
from apple_pick_sim.system_id.trajectory_store import PHASE_TO_INT
from apple_pick_sim.tests.conftest import RANGES_FIXTURE

_SEED = 42
_MAX_STEPS = 30
_HOLD = int(PHASE_TO_INT["hold"])


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
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(diff * diff))


def _trajectory_errors(replay: dict, recorded: dict) -> dict[str, float]:
    """Frame-aligned errors between replay collector output and recorded GT."""
    n = min(int(replay["ft_wrist"].shape[0]), int(recorded["ft_wrist"].shape[0]))
    junction_names = list(recorded["junction_names"])
    woody_start_rec = np.stack(
        [
            flatten_woody_positions(
                recorded["woody_part_start_pos"],
                frame_idx=i,
                junction_names=junction_names,
            )
            for i in range(n)
        ],
        axis=0,
    )
    woody_start_rep = np.stack(
        [
            flatten_woody_positions(
                replay["woody_part_start_pos"],
                frame_idx=i,
                junction_names=junction_names,
            )
            for i in range(n)
        ],
        axis=0,
    )
    phase = np.asarray(recorded["phase"][:n], dtype=np.int8)
    hold_mask = phase == _HOLD
    out = {
        "n_frames": float(n),
        "n_hold_frames": float(np.count_nonzero(hold_mask)),
        "ft_wrist_mse": _mse(replay["ft_wrist"][:n], recorded["ft_wrist"][:n]),
        "ft_wrist_rmse": _rmse(replay["ft_wrist"][:n], recorded["ft_wrist"][:n]),
        "tcp_pos_mse": _mse(replay["tcp_pos"][:n], recorded["tcp_pos"][:n]),
        "tcp_pos_rmse_mm": _rmse(replay["tcp_pos"][:n], recorded["tcp_pos"][:n]) * 1000.0,
        "woody_start_mse": _mse(woody_start_rep, woody_start_rec),
        "apple_pos_mse": _mse(replay["apple_pos"][:n], recorded["apple_pos"][:n]),
    }
    if np.any(hold_mask):
        out["hold_ft_wrist_mse"] = _mse(
            replay["ft_wrist"][:n][hold_mask], recorded["ft_wrist"][:n][hold_mask]
        )
        out["hold_woody_start_mse"] = _mse(
            woody_start_rep[hold_mask], woody_start_rec[hold_mask]
        )
    return out


def _wrong_stem_only(gt: BendStiffnessCandidate, *, stem_scale: float) -> BendStiffnessCandidate:
    """Perturb only stem stiffness — most sensitive during quasi-static hold."""
    return BendStiffnessCandidate(
        primary=gt.primary,
        secondary=gt.secondary,
        spur=gt.spur,
        stem=gt.stem * stem_scale,
    )


def _collect_tiny_dataset(output_dir: Path) -> BatchedSysIdDataset:
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
    try:
        collect_batched_quasi_static_dataset(
            env,
            num_structures=num_structures,
            num_directions=num_directions,
            config=config,
            output_dir=output_dir,
            seed=_SEED,
            ranges_path=RANGES_FIXTURE,
            max_steps=_MAX_STEPS,
        )
    finally:
        env.close()
    return BatchedSysIdDataset(output_dir)


def _build_env_fn(*, num_envs: int, per_env_params, max_episode_steps: int, gripper=None) -> ApplePickBatchedSysIdEnv:
    sim_config = _test_sim_config(num_envs=num_envs)
    if gripper is not None:
        sim_config = dataclasses.replace(
            sim_config,
            robot=dataclasses.replace(sim_config.robot, gripper=gripper),
        )
    return ApplePickBatchedSysIdEnv(
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
        ranges_path=RANGES_FIXTURE,
        topology_seed=_SEED,
        use_settle_cache=False,
        sim_config=sim_config,
        per_env_params=per_env_params,
    )


def _print_digital_twin_frame0(dataset: BatchedSysIdDataset) -> None:
    twin = digital_twin_obs_from_batched_episode(dataset, structure_idx=0, direction_idx=0)
    recorded = dataset.load_episode_obs_arrays(0, 0)
    junction_names = list(recorded["junction_names"])
    rec_start = flatten_woody_positions(
        recorded["woody_part_start_pos"], frame_idx=0, junction_names=junction_names
    )
    rec_end = flatten_woody_positions(
        recorded["woody_part_end_pos"], frame_idx=0, junction_names=junction_names
    )
    print("\n=== Digital twin frame-0 (woody geometry from recorded obs) ===")
    print(f"  junction_names: {twin.junction_names}")
    print(
        f"  woody_start rmse (mm): {_rmse(twin.woody_part_start_pos, rec_start) * 1000:.4f}"
    )
    print(f"  woody_end rmse (mm): {_rmse(twin.woody_part_end_pos, rec_end) * 1000:.4f}")
    num_directions = 1
    env = ApplePickBatchedSysIdEnv(
        num_envs=1,
        max_episode_steps=int(recorded["action"].shape[0]),
        ranges_path=RANGES_FIXTURE,
        topology_seed=_SEED,
        use_settle_cache=False,
        sim_config=_test_sim_config(num_envs=1),
        per_env_params=sample_and_broadcast_structure_params(
            RANGES_FIXTURE, topology_seed=_SEED, num_structures=1, num_directions=1
        ),
    )
    try:
        env.reset(seed=_SEED)
        initialize_batched_env_from_dataset(
            env, dataset, structure_idx=0, num_directions=num_directions
        )
        live = env.sysid_numpy_obs(0)
        print("\n=== After initialize_batched_env_from_dataset (before replay steps) ===")
        print(f"  tcp_pos rmse vs recorded[0] (mm): {_rmse(live['tcp_pos'], recorded['tcp_pos'][0]) * 1000:.4f}")
        print(
            f"  robot_joint_q rmse: {_rmse(live['robot_joint_q'], recorded['robot_joint_q'][0]):.6f}"
        )
    finally:
        env.close()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="batched_replay_mse_") as tmp:
        out = Path(tmp)
        print(f"Collecting tiny dataset → {out}")
        dataset = _collect_tiny_dataset(out)
        num_directions = int(dataset.manifest["collection"]["num_directions"])
        structure_idx = 0
        recorded = dataset.load_episode_obs_arrays(structure_idx, 0)
        n_frames = int(recorded["action"].shape[0])
        print(f"Collected {n_frames} frames, {num_directions} direction(s)")

        _print_digital_twin_frame0(dataset)

        gt = gt_bend_stiffness_candidate_from_structure(dataset, structure_idx)
        wrong = _wrong_stem_only(gt, stem_scale=0.01)
        candidates = [gt, wrong]
        print("\n=== Stiffness candidates ===")
        print(f"  GT:    {gt}")
        print(f"  Wrong: {wrong}  (stem bend_stiffness ×0.01; other segments unchanged)")

        collectors = replay_candidates_for_structure(
            dataset=dataset,
            structure_idx=structure_idx,
            candidates=candidates,
            num_directions=num_directions,
            seed=None,
            build_env_fn=_build_env_fn,
        )

        print("\n=== Replay vs recorded GT (MSE / RMSE) ===")
        labels = ["GT params", "Wrong stem (×0.01)"]
        gt_hold_mse: list[float] = []
        wrong_hold_mse: list[float] = []
        gt_tcp_mse: list[float] = []
        for cand_idx, label in enumerate(labels):
            replay_eps = direction_episodes_from_collectors(
                collectors,
                candidate_index=cand_idx,
                num_directions=num_directions,
            )
            for d, replay in enumerate(replay_eps):
                err = _trajectory_errors(replay, recorded)
                print(f"\n  [{label}] direction {d}:")
                for key, val in err.items():
                    if key == "n_frames":
                        print(f"    {key}: {int(val)}")
                    elif "mse" in key:
                        print(f"    {key}: {val:.6g}")
                    else:
                        print(f"    {key}: {val:.4f}")
                if cand_idx == 0:
                    gt_hold_mse.append(err.get("hold_ft_wrist_mse", err["ft_wrist_mse"]))
                    gt_tcp_mse.append(err["tcp_pos_mse"])
                else:
                    wrong_hold_mse.append(err.get("hold_ft_wrist_mse", err["ft_wrist_mse"]))

        gt_hold = float(np.mean(gt_hold_mse))
        wrong_hold = float(np.mean(wrong_hold_mse))
        gt_tcp = float(np.mean(gt_tcp_mse))
        ratio = wrong_hold / max(gt_hold, 1e-12)
        print("\n=== Summary ===")
        print(f"  tcp_pos MSE (all frames, GT replay):  {gt_tcp:.6g}  (~sub-mm kinematic tracking)")
        print(f"  hold ft_wrist MSE — GT params:        {gt_hold:.6g}")
        print(f"  hold ft_wrist MSE — Wrong stem:       {wrong_hold:.6g}")
        print(f"  Wrong / GT hold-force ratio:          {ratio:.2f}x")
        if wrong_hold <= gt_hold:
            raise SystemExit(
                "Expected wrong-stem hold ft_wrist MSE > GT; discrimination failed."
            )
        print("\nOK: GT replay tracks TCP; wrong stem stiffness diverges on hold-phase wrench.")


if __name__ == "__main__":
    main()
