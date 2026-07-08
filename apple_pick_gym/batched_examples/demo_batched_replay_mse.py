"""Demo: true-params replay vs recorded GT with per-trajectory MSE.

Collects a tiny batched_sysid_v1 dataset, reconstructs geometry from the
recorded ``fruiting_system_params`` (not observation inference), replays with GT
bend stiffness vs random stiffness draws from the fixture ranges, and prints MSE
against recorded observations.

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
from apple_pick_sim.fruiting_system import load_ranges, sample_params
from apple_pick_sim.system_id import BatchedSysIdDataset, QuasiStaticStepConfig
from apple_pick_sim.system_id.batched_digital_twin_init import (
    infer_base_params_for_structure,
    initialize_batched_env_from_dataset,
    true_params_for_structure,
)
from apple_pick_sim.system_id.mmd_features import flatten_woody_positions
from apple_pick_sim.system_id.trajectory_store import PHASE_TO_INT
from apple_pick_sim.tests.conftest import RANGES_FIXTURE

_SEED = 42
_MAX_STEPS = 30
_N_RANDOM_STIFFNESS = 3
_HOLD = int(PHASE_TO_INT["hold"])
_ROD_SEGMENTS: tuple[str, ...] = ("primary", "secondary", "spur", "stem")
_SETTLE_SUBSTEPS = 5000
_TRIAL_SEEDS: tuple[int, ...] = (_SEED, 7, 123)


def _test_sim_config(*, num_envs: int, topology_seed: int) -> BatchedHeterogeneousCoupledSimConfig:
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
        scene=SceneSettleCollisionConfig(settle_substeps=_SETTLE_SUBSTEPS),
        controller=ControllerConfig(mode="vic", linear_speed=1.0, angular_speed=1.0),
        domain_randomization=dataclasses.replace(
            BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=num_envs).domain_randomization,
            topology_seed=int(topology_seed),
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
    # The batched_sysid_v1 dataset includes a pre-weld reconstruction row at step_idx=-1
    # (phase="pre_weld"). Replay collectors start at the first control step, so we skip
    # the pre-weld row when computing MSE/RMSE diagnostics.
    start = 1 if n > 1 else 0
    junction_names = list(recorded["junction_names"])
    tcp_pos_err = np.asarray(replay["tcp_pos"][start:n], dtype=np.float64) - np.asarray(
        recorded["tcp_pos"][start:n], dtype=np.float64
    )
    tcp_pos_norm_mm = np.linalg.norm(tcp_pos_err, axis=1) * 1000.0
    woody_start_rec = np.stack(
        [
            flatten_woody_positions(
                recorded["woody_part_start_pos"],
                frame_idx=i,
                junction_names=junction_names,
            )
            for i in range(start, n)
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
            for i in range(start, n)
        ],
        axis=0,
    )
    phase = np.asarray(recorded["phase"][start:n], dtype=np.int8)
    hold_mask = phase == _HOLD
    n_eval = max(0, n - start)
    out = {
        "n_frames": float(n_eval),
        "n_hold_frames": float(np.count_nonzero(hold_mask)),
        "ft_wrist_mse": _mse(replay["ft_wrist"][start:n], recorded["ft_wrist"][start:n]),
        "ft_wrist_rmse": _rmse(replay["ft_wrist"][start:n], recorded["ft_wrist"][start:n]),
        "ft_force_rmse_N": _rmse(
            np.asarray(replay["ft_wrist"][start:n], dtype=np.float64)[:, :3],
            np.asarray(recorded["ft_wrist"][start:n], dtype=np.float64)[:, :3],
        ),
        "ft_torque_rmse_Nm": _rmse(
            np.asarray(replay["ft_wrist"][start:n], dtype=np.float64)[:, 3:],
            np.asarray(recorded["ft_wrist"][start:n], dtype=np.float64)[:, 3:],
        ),
        "tcp_pos_mse": _mse(replay["tcp_pos"][start:n], recorded["tcp_pos"][start:n]),
        "tcp_pos_rmse_mm": _rmse(replay["tcp_pos"][start:n], recorded["tcp_pos"][start:n]) * 1000.0,
        "tcp_pos_mean_mm": float(np.mean(tcp_pos_norm_mm)) if n_eval > 0 else 0.0,
        "tcp_pos_max_mm": float(np.max(tcp_pos_norm_mm)) if n_eval > 0 else 0.0,
        "woody_start_mse": _mse(woody_start_rep, woody_start_rec),
        "apple_pos_mse": _mse(replay["apple_pos"][start:n], recorded["apple_pos"][start:n]),
    }
    if np.any(hold_mask):
        out["hold_ft_wrist_mse"] = _mse(
            replay["ft_wrist"][start:n][hold_mask], recorded["ft_wrist"][start:n][hold_mask]
        )
        out["hold_tcp_pos_mse"] = _mse(
            replay["tcp_pos"][start:n][hold_mask], recorded["tcp_pos"][start:n][hold_mask]
        )
        out["hold_tcp_pos_rmse_mm"] = (
            _rmse(replay["tcp_pos"][start:n][hold_mask], recorded["tcp_pos"][start:n][hold_mask])
            * 1000.0
        )
        out["hold_tcp_pos_mean_mm"] = float(np.mean(tcp_pos_norm_mm[hold_mask]))
        out["hold_tcp_pos_max_mm"] = float(np.max(tcp_pos_norm_mm[hold_mask]))
        out["hold_woody_start_mse"] = _mse(
            woody_start_rep[hold_mask], woody_start_rec[hold_mask]
        )
    return out


def _bend_stiffness_candidate_from_params(params) -> BendStiffnessCandidate:
    values: dict[str, float] = {}
    for segment in _ROD_SEGMENTS:
        rod = getattr(params, segment)
        if rod is None:
            raise ValueError(f"segment {segment!r} is missing in sampled params")
        values[segment] = float(rod.bend_stiffness)
    return BendStiffnessCandidate(
        primary=values["primary"],
        secondary=values["secondary"],
        spur=values["spur"],
        stem=values["stem"],
    )


def _stiffness_differs_from_gt(
    candidate: BendStiffnessCandidate,
    gt: BendStiffnessCandidate,
    *,
    min_rel_delta: float = 0.5,
) -> bool:
    """True when at least one segment stiffness differs from GT by ``min_rel_delta``."""
    for segment in _ROD_SEGMENTS:
        cand_val = float(getattr(candidate, segment))
        gt_val = float(getattr(gt, segment))
        if abs(cand_val - gt_val) / max(abs(gt_val), 1e-12) >= float(min_rel_delta):
            return True
    return False


def _random_stiffness_candidates(
    *,
    n: int,
    base_seed: int,
    gt: BendStiffnessCandidate,
) -> list[BendStiffnessCandidate]:
    """Draw bend-stiffness tuples from the fixture sampler, away from GT."""
    ranges = load_ranges(RANGES_FIXTURE)
    out: list[BendStiffnessCandidate] = []
    attempt = 0
    while len(out) < int(n) and attempt < 200:
        params = sample_params(ranges, seed=int(base_seed) + 1000 + attempt * 97)
        candidate = _bend_stiffness_candidate_from_params(params)
        attempt += 1
        if _stiffness_differs_from_gt(candidate, gt):
            out.append(candidate)
    if len(out) < int(n):
        raise RuntimeError(f"could only sample {len(out)} random stiffness candidates away from GT")
    return out


def _collect_tiny_dataset(*, output_dir: Path, seed: int) -> BatchedSysIdDataset:
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
        topology_seed=int(seed),
        num_structures=num_structures,
        num_directions=num_directions,
    )
    env = ApplePickBatchedSysIdEnv(
        num_envs=num_envs,
        max_episode_steps=120,
        ranges_path=RANGES_FIXTURE,
        topology_seed=int(seed),
        use_settle_cache=False,
        sim_config=_test_sim_config(num_envs=num_envs, topology_seed=int(seed)),
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
            seed=int(seed),
            ranges_path=RANGES_FIXTURE,
            max_steps=_MAX_STEPS,
        )
    finally:
        env.close()
    return BatchedSysIdDataset(output_dir)


def _build_env_fn(
    *,
    num_envs: int,
    per_env_params,
    max_episode_steps: int,
    control_hz: float,
    topology_seed: int,
    gripper=None,
) -> ApplePickBatchedSysIdEnv:
    sim_config = _test_sim_config(num_envs=num_envs, topology_seed=int(topology_seed))
    if gripper is not None:
        sim_config = dataclasses.replace(
            sim_config,
            robot=dataclasses.replace(sim_config.robot, gripper=gripper),
        )
    return ApplePickBatchedSysIdEnv(
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
        ranges_path=RANGES_FIXTURE,
        topology_seed=int(topology_seed),
        use_settle_cache=False,
        sim_config=sim_config,
        per_env_params=per_env_params,
        control_hz=float(control_hz),
    )


def _primary_geometry_delta(a, b) -> tuple[float, float]:
    """Return (length delta m, direction cosine error) for primary rods."""
    assert a.primary is not None and b.primary is not None
    length_delta = abs(float(a.primary.length) - float(b.primary.length))
    direction_delta = 1.0 - abs(
        float(
            np.dot(
                np.asarray(a.primary.direction, dtype=np.float64),
                np.asarray(b.primary.direction, dtype=np.float64),
            )
        )
    )
    return length_delta, direction_delta


def _print_true_params_geometry(dataset: BatchedSysIdDataset, *, structure_idx: int) -> None:
    """Show that replay geometry comes from recorded true params, not obs inference."""
    true_params = true_params_for_structure(dataset, structure_idx)
    inferred_params = infer_base_params_for_structure(dataset, structure_idx)
    recorded = dataset.load_episode_obs_arrays(structure_idx, 0)
    n_frames = int(recorded["action"].shape[0])

    inf_len_delta, inf_dir_delta = _primary_geometry_delta(inferred_params, true_params)
    print("\n=== Geometry source: true params vs obs inference (primary segment) ===")
    print(f"  inferred vs true length delta (m): {inf_len_delta:.6g}")
    print(f"  inferred vs true direction error:  {inf_dir_delta:.6g}")

    env = ApplePickBatchedSysIdEnv(
        num_envs=1,
        max_episode_steps=n_frames,
        ranges_path=RANGES_FIXTURE,
        topology_seed=int(dataset.manifest["collection"]["topology_seed"]),
        use_settle_cache=False,
        sim_config=_test_sim_config(
            num_envs=1,
            topology_seed=int(dataset.manifest["collection"]["topology_seed"]),
        ),
        per_env_params=[true_params],
        control_hz=float(dataset.manifest["collection"]["control_hz"]),
    )
    try:
        env.reset(seed=_SEED)
        initialize_batched_env_from_dataset(
            env, dataset, structure_idx=structure_idx, num_directions=1
        )
        built_params = env._sim._per_env_params[0]
        built_len_delta, built_dir_delta = _primary_geometry_delta(built_params, true_params)
        live = env.sysid_numpy_obs(0)
        print("\n=== Built env from true params (frame 0, before replay steps) ===")
        print(f"  built vs true length delta (m):    {built_len_delta:.6g}")
        print(f"  built vs true direction error:     {built_dir_delta:.6g}")
        print(
            f"  tcp_pos rmse vs recorded[0] (mm): "
            f"{_rmse(live['tcp_pos'], recorded['tcp_pos'][0]) * 1000:.4f}"
        )
        print(
            f"  robot_joint_q rmse: "
            f"{_rmse(live['robot_joint_q'], recorded['robot_joint_q'][0]):.6f}"
        )
    finally:
        env.close()


def main() -> None:
    gt_tcp_mean_mm: list[float] = []
    gt_tcp_max_mm: list[float] = []
    gt_hold_ft_mse: list[float] = []

    for trial_seed in _TRIAL_SEEDS:
        with tempfile.TemporaryDirectory(prefix=f"batched_replay_mse_seed_{trial_seed}_") as tmp:
            out = Path(tmp)
            print(f"\n\n=== Trial seed={trial_seed} ===")
            print(f"Collecting tiny dataset → {out}")
            dataset = _collect_tiny_dataset(output_dir=out, seed=int(trial_seed))
            num_directions = int(dataset.manifest["collection"]["num_directions"])
            control_hz = float(dataset.manifest["collection"]["control_hz"])
            topology_seed = int(dataset.manifest["collection"]["topology_seed"])
            structure_idx = 0
            recorded = dataset.load_episode_obs_arrays(structure_idx, 0)
            n_frames = int(recorded["action"].shape[0])
            print(f"Collected {n_frames} frames, {num_directions} direction(s)")

            _print_true_params_geometry(dataset, structure_idx=structure_idx)

            gt = gt_bend_stiffness_candidate_from_structure(dataset, structure_idx)
            random_candidates = _random_stiffness_candidates(
                n=_N_RANDOM_STIFFNESS,
                base_seed=int(trial_seed),
                gt=gt,
            )
            candidates = [gt, *random_candidates]
            print("\n=== Stiffness candidates (geometry fixed from true params) ===")
            print(f"  GT: {gt}")
            for i, candidate in enumerate(random_candidates):
                print(f"  Random[{i}]: {candidate}")

            collectors = replay_candidates_for_structure(
                dataset=dataset,
                structure_idx=structure_idx,
                candidates=candidates,
                num_directions=num_directions,
                seed=None,
                build_env_fn=lambda **kw: _build_env_fn(
                    control_hz=control_hz,
                    topology_seed=topology_seed,
                    **kw,
                ),
            )

            print("\n=== Replay vs recorded GT (MSE / RMSE) ===")
            labels = ["GT stiffness"] + [f"Random[{i}]" for i in range(len(random_candidates))]
            gt_hold_mse: list[float] = []
            random_hold_mse: list[float] = []
            per_candidate_hold: list[tuple[str, float]] = []
            trial_gt_tcp_mean_mm: float | None = None
            trial_gt_tcp_max_mm: float | None = None

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
                    hold_mse = err.get("hold_ft_wrist_mse", err["ft_wrist_mse"])
                    per_candidate_hold.append((label, hold_mse))
                    if cand_idx == 0:
                        gt_hold_mse.append(hold_mse)
                        trial_gt_tcp_mean_mm = float(err["tcp_pos_mean_mm"])
                        trial_gt_tcp_max_mm = float(err["tcp_pos_max_mm"])
                    else:
                        random_hold_mse.append(hold_mse)

            gt_hold = float(np.mean(gt_hold_mse))
            random_hold = float(np.mean(random_hold_mse)) if random_hold_mse else float("nan")
            n_random_worse = sum(1 for _, mse in per_candidate_hold[1:] if mse > gt_hold)

            print("\n=== Trial summary ===")
            if trial_gt_tcp_mean_mm is not None and trial_gt_tcp_max_mm is not None:
                print(
                    f"  GT tcp_pos |Δ| mean/max (mm): "
                    f"{trial_gt_tcp_mean_mm:.4f} / {trial_gt_tcp_max_mm:.4f}"
                )
                gt_tcp_mean_mm.append(trial_gt_tcp_mean_mm)
                gt_tcp_max_mm.append(trial_gt_tcp_max_mm)
            print(f"  GT hold ft_wrist MSE:          {gt_hold:.6g}")
            gt_hold_ft_mse.append(gt_hold)
            print(f"  random hold ft_wrist MSE mean: {random_hold:.6g}")
            print(
                f"  random candidates hold MSE > GT: {n_random_worse}/{len(random_hold_mse)}"
            )

    print("\n\n=== Across-trial summary (GT only) ===")
    if gt_tcp_mean_mm:
        print(
            f"  GT tcp_pos |Δ| mean(mm): mean={float(np.mean(gt_tcp_mean_mm)):.4f} "
            f"max={float(np.max(gt_tcp_mean_mm)):.4f}"
        )
        print(
            f"  GT tcp_pos |Δ| max(mm):  mean={float(np.mean(gt_tcp_max_mm)):.4f} "
            f"max={float(np.max(gt_tcp_max_mm)):.4f}"
        )
    if gt_hold_ft_mse:
        print(
            f"  GT hold ft_wrist MSE:     mean={float(np.mean(gt_hold_ft_mse)):.6g} "
            f"max={float(np.max(gt_hold_ft_mse)):.6g}"
        )


if __name__ == "__main__":
    main()
