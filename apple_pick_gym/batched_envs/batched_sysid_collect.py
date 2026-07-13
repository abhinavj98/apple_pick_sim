"""Orchestration helpers for parallel batched sys-ID Parquet collection."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import numpy as np
import torch

from apple_pick_gym.batched_envs.batched_sysid_world_info import (
    physical_stem_dir_for_world,
    robot_base_pos_for_world,
)
from apple_pick_gym.batched_envs.batched_stability_monitor import (
    BatchedStabilityMonitor,
    StabilityThresholds,
    ik_bootstrap_unstable_mask,
)
from apple_pick_gym.batched_envs.env_disable_controller import EnvDisableController
from apple_pick_sim.fruiting_system import (
    fruiting_params_to_json,
    load_ranges,
    sample_heterogeneous_params_list,
)
from apple_pick_sim.system_id import (
    BatchedEpisodeWriter,
    ExcitationContext,
    QuasiStaticStepConfig,
    QuasiStaticTrajectory,
    estimate_trajectory_frames,
    sample_robot_facing_pull_directions,
    write_manifest,
)
from apple_pick_sim.system_id.batched_trajectory_store import (
    PRE_WELD_STEP_IDX,
    SCHEMA_VERSION,
    episode_filename,
    resolve_batched_dataset_output_dir,
)
from apple_pick_sim.system_id.manifest_sim_config import sim_config_to_manifest_dict
from apple_pick_sim.system_id.pre_weld_obs import complete_pre_weld_sysid_obs

EXCLUDED_REASON_STABILITY_BLOWUP = "stability_blowup"


class OnStepCallback(Protocol):
    """Optional hook after reset (step_idx=-1) and each env step; return False to stop."""

    def __call__(
        self,
        *,
        env: Any,
        step_idx: int,
        phase: str,
        sim_time: float,
        obs: Any,
        amplitude_m: float = 0.0,
    ) -> bool: ...


def structure_and_direction_indices(env_idx: int, num_directions: int) -> tuple[int, int]:
    d = int(num_directions)
    if d < 1:
        raise ValueError("num_directions must be >= 1")
    return int(env_idx) // d, int(env_idx) % d


def broadcast_structure_params(
    structure_params: Sequence[Any],
    num_directions: int,
) -> list[Any]:
    d = int(num_directions)
    if d < 1:
        raise ValueError("num_directions must be >= 1")
    out: list[Any] = []
    for params in structure_params:
        out.extend([params] * d)
    return out


def sample_and_broadcast_structure_params(
    ranges_path: Path | str,
    *,
    topology_seed: int,
    num_structures: int,
    num_directions: int,
) -> list[Any]:
    ranges = load_ranges(ranges_path)
    structure_params = sample_heterogeneous_params_list(
        ranges,
        topology_seed=int(topology_seed),
        num_envs=int(num_structures),
    )
    return broadcast_structure_params(structure_params, int(num_directions))


def assign_pull_directions(
    env: Any,
    *,
    num_structures: int,
    num_directions: int,
    min_world_z: float | None = 0.0,
) -> list[np.ndarray]:
    scene = env._sim.scene
    layout = env._sim.layout
    if layout is None:
        raise RuntimeError("batched scene missing layout")

    per_env: list[np.ndarray] = []
    s_count = int(num_structures)
    d_count = int(num_directions)
    bq = scene.cable.state_0.body_q.numpy().reshape(-1, 7)

    for s in range(s_count):
        rep = s * d_count
        apple_idx = int(layout.apple_body_indices[rep])
        apple_pos = bq[apple_idx, :3]
        robot_base = robot_base_pos_for_world(layout, rep)
        robot_vec = robot_base - apple_pos
        physical_stem = physical_stem_dir_for_world(scene, layout, rep)
        dirs = sample_robot_facing_pull_directions(
            d_count,
            physical_stem,
            robot_vec,
            min_world_z=min_world_z,
        )
        for d in range(d_count):
            per_env.append(np.asarray(dirs[d], dtype=np.float64))
    return per_env


def action_from_velocity_for_direction(vel: Any, direction: np.ndarray) -> np.ndarray:
    lin = np.asarray(vel.linear, dtype=np.float64).reshape(3)
    speed = float(np.linalg.norm(lin))
    if speed < 1e-9:
        lin_out = np.zeros(3, dtype=np.float32)
    else:
        d = np.asarray(direction, dtype=np.float64).reshape(3)
        d_norm = float(np.linalg.norm(d))
        if d_norm < 1e-12:
            raise ValueError("direction must be non-zero")
        lin_out = (d / d_norm * speed).astype(np.float32)
    ang = np.asarray(vel.angular, dtype=np.float32).reshape(3)
    return np.concatenate([lin_out, ang], dtype=np.float32)


def actions_tensor_for_velocity(
    vel: Any,
    per_env_directions: Sequence[np.ndarray],
    *,
    device: torch.device | str,
) -> torch.Tensor:
    rows = [
        action_from_velocity_for_direction(vel, direction)
        for direction in per_env_directions
    ]
    return torch.as_tensor(np.stack(rows, axis=0), dtype=torch.float32, device=device)


class BatchedSysIdCollectors:
    """One BatchedEpisodeWriter per parallel env."""

    def __init__(self, num_envs: int) -> None:
        self._writers = [BatchedEpisodeWriter(episode_id=str(uuid4())) for _ in range(int(num_envs))]

    @property
    def writers(self) -> list[BatchedEpisodeWriter]:
        return list(self._writers)

    def episode_id(self, env_idx: int) -> str:
        return self._writers[int(env_idx)].episode_id

    def record_step(
        self,
        env: Any,
        *,
        env_idx: int,
        step_idx: int,
        sim_time: float,
        phase: str,
        amplitude_m: float,
        action: np.ndarray,
        stable: bool = True,
    ) -> None:
        obs = env.sysid_numpy_obs(int(env_idx))
        self._writers[int(env_idx)].record_step(
            step_idx=int(step_idx),
            sim_time=float(sim_time),
            phase=str(phase),
            amplitude_m=float(amplitude_m),
            action=np.asarray(action, dtype=np.float32),
            obs=obs,
            stable=bool(stable),
        )

    def record_pre_weld_step(
        self,
        *,
        env_idx: int,
        obs: dict[str, Any],
        stable: bool = True,
    ) -> None:
        """Append the post-settle, pre-weld reconstruction frame (``step_idx=-1``)."""
        zero_action = np.zeros(6, dtype=np.float32)
        self._writers[int(env_idx)].record_step(
            step_idx=PRE_WELD_STEP_IDX,
            sim_time=0.0,
            phase="pre_weld",
            amplitude_m=0.0,
            action=zero_action,
            obs=obs,
            stable=bool(stable),
        )

    def save_all(
        self,
        output_dir: Path | str,
        metadata_rows: Sequence[dict[str, Any]],
        *,
        num_directions: int,
    ) -> list[Path]:
        out = Path(output_dir)
        paths: list[Path] = []
        if len(metadata_rows) != len(self._writers):
            raise ValueError("metadata_rows length must match number of writers")
        for writer, meta in zip(self._writers, metadata_rows, strict=True):
            env_idx = int(meta["env_idx"])
            structure_idx, direction_idx = structure_and_direction_indices(
                env_idx, int(num_directions)
            )
            rel = episode_filename(structure_idx, direction_idx)
            paths.append(writer.save(out / rel, meta))
        return paths


def build_episode_metadata(
    *,
    episode_id: str,
    env: Any,
    env_idx: int,
    num_directions: int,
    pull_direction: np.ndarray,
    seed: int,
    config: QuasiStaticStepConfig,
    ranges_path: Path | str,
) -> dict[str, Any]:
    structure_idx, direction_idx = structure_and_direction_indices(
        int(env_idx), int(num_directions)
    )
    per_env = env.per_env_reset_info(int(env_idx))
    params = env._sim._per_env_params[int(env_idx)]
    weld = per_env["weld_direction"]
    weld_list = [float(x) for x in np.asarray(weld, dtype=np.float64).reshape(3)]
    weld_norm = float(np.linalg.norm(weld_list))
    if weld_norm < 1e-12:
        weld_list = [0.0, 0.0, 1.0]
    else:
        weld_list = [float(x / weld_norm) for x in weld_list]

    exported = env.sysid_numpy_obs(int(env_idx))
    fruiting_base = per_env.get("fruiting_base_pos")
    rod_radii = per_env.get("rod_radii")
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": str(episode_id),
        "structure_idx": int(structure_idx),
        "direction_idx": int(direction_idx),
        "env_idx": int(env_idx),
        "pull_direction": [float(x) for x in np.asarray(pull_direction, dtype=np.float64).reshape(3)],
        "params_fingerprint": (
            json.dumps(per_env["params_fingerprint"], sort_keys=True)
            if isinstance(per_env["params_fingerprint"], dict)
            else str(per_env["params_fingerprint"])
        ),
        "fruiting_system_params": fruiting_params_to_json(params),
        "excitation_type": "quasi_static",
        "control_hz": float(config.control_hz),
        "seed": int(seed),
        "n_woody_parts": len(env.junction_names),
        "junction_names": list(env.junction_names),
        "initial_tcp_pos": [float(x) for x in np.asarray(exported["tcp_pos"]).reshape(3)],
        "initial_tcp_quat": [float(x) for x in np.asarray(exported["tcp_quat"]).reshape(4)],
        "initial_apple_pos": [float(x) for x in np.asarray(exported["apple_pos"]).reshape(3)],
        "initial_apple_quat": [float(x) for x in np.asarray(exported["apple_quat"]).reshape(4)],
        "initial_robot_joint_q": [
            float(x) for x in np.asarray(exported["robot_joint_q"]).reshape(-1)
        ],
        "fixture_path": str(Path(ranges_path)),
        "fruiting_base_pos": (
            None
            if fruiting_base is None
            else [float(x) for x in np.asarray(fruiting_base).reshape(3)]
        ),
        "apple_radius": per_env.get("apple_radius"),
        "rod_radii": (
            None
            if rod_radii is None
            else json.dumps({str(k): float(v) for k, v in rod_radii.items()}, sort_keys=True)
        ),
        "weld_direction": weld_list,
        "weld_reference_pos": [
            float(x) for x in np.asarray(per_env["weld_reference_pos"]).reshape(3)
        ],
        "weld_reference_quat": [
            float(x) for x in np.asarray(per_env["weld_reference_quat"]).reshape(4)
        ],
        "movement_per_step_m": float(config.movement_per_step_m),
        "total_movement_m": float(config.total_movement_m),
        "hold_duration_s": float(config.hold_duration_s),
        "move_speed_mps": float(config.move_speed_mps),
        "skip_return": bool(config.skip_return),
    }


def _build_structure_summaries(
    metadata_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_structure: dict[int, dict[str, Any]] = {}
    for meta in metadata_rows:
        s = int(meta["structure_idx"])
        if s in by_structure:
            continue
        by_structure[s] = {
            "structure_idx": s,
            "params_fingerprint": meta["params_fingerprint"],
            "junction_names": meta["junction_names"],
            "n_woody_parts": meta["n_woody_parts"],
        }
    return [by_structure[k] for k in sorted(by_structure)]



def _excluded_env_indices(
    disable_ctrl: EnvDisableController,
    writers: Sequence[BatchedEpisodeWriter],
) -> set[int]:
    """Env indices to mark excluded: sticky-disabled or any unstable recorded frame."""
    out: set[int] = set()
    disabled = disable_ctrl.disabled.detach().cpu().tolist()
    for i, flag in enumerate(disabled):
        if bool(flag):
            out.add(int(i))
    for i, writer in enumerate(writers):
        if writer.has_unstable_frame():
            out.add(int(i))
    return out


def _build_manifest_episodes(
    metadata_rows: Sequence[dict[str, Any]],
    writers: Sequence[BatchedEpisodeWriter],
    *,
    num_directions: int,
    excluded_env_indices: set[int] | frozenset[int] = frozenset(),
    excluded_reason: str = EXCLUDED_REASON_STABILITY_BLOWUP,
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    excluded = {int(i) for i in excluded_env_indices}
    for meta, writer in zip(metadata_rows, writers, strict=True):
        structure_idx = int(meta["structure_idx"])
        direction_idx = int(meta["direction_idx"])
        env_idx = int(meta["env_idx"])
        is_excluded = env_idx in excluded
        episodes.append(
            {
                "structure_idx": structure_idx,
                "direction_idx": direction_idx,
                "env_idx": env_idx,
                "filename": episode_filename(structure_idx, direction_idx),
                "episode_id": str(meta["episode_id"]),
                "pull_direction": meta["pull_direction"],
                "n_frames": int(writer.n_frames),
                "excluded": bool(is_excluded),
                "excluded_reason": str(excluded_reason) if is_excluded else None,
            }
        )
    return episodes


def collect_batched_quasi_static_dataset(
    env: Any,
    *,
    num_structures: int,
    num_directions: int,
    config: QuasiStaticStepConfig,
    output_dir: Path | str,
    seed: int,
    ranges_path: Path | str,
    max_steps: int = 0,
    progress: Callable[[str], None] | None = None,
    on_step: OnStepCallback | None = None,
    command_argv: Sequence[str] | None = None,
    overwrite: bool = False,
    append_timestamp: bool = True,
    pull_direction_min_world_z: float | None = 0.0,
    stability_thresholds: StabilityThresholds | None = None,
    save_snapshot: bool = False,
) -> Path:
    """Run lockstep quasi-static collection and write batched_sysid_v1 dataset."""
    out = resolve_batched_dataset_output_dir(
        output_dir,
        overwrite=bool(overwrite),
        append_timestamp=bool(append_timestamp),
    )
    out.mkdir(parents=True, exist_ok=True)

    num_envs = int(num_structures) * int(num_directions)
    if env.num_envs != num_envs:
        raise ValueError(
            f"env.num_envs ({env.num_envs}) != num_structures*num_directions ({num_envs})"
        )

    per_env_directions = assign_pull_directions(
        env,
        num_structures=int(num_structures),
        num_directions=int(num_directions),
        min_world_z=pull_direction_min_world_z,
    )
    reference_traj = QuasiStaticTrajectory(
        np.asarray(per_env_directions[0], dtype=np.float64).reshape(1, 3),
        config,
    )

    step_cap = int(max_steps)
    if step_cap <= 0:
        step_cap = estimate_trajectory_frames(config, 1) + 64

    obs, _info = env.reset(seed=int(seed))
    if save_snapshot:
        from apple_pick_sim.system_id.batched_episode_snapshot_io import (
            save_per_env_episode_snapshots,
        )

        save_per_env_episode_snapshots(
            env._sim,
            output_dir=out,
            num_directions=int(num_directions),
        )
    initial_unstable = ik_bootstrap_unstable_mask(env, num_envs)
    monitor = BatchedStabilityMonitor(
        num_envs,
        known_obs_keys=set(obs.keys()),
        thresholds=stability_thresholds,
        initial_unstable=initial_unstable,
    )
    disable_ctrl = EnvDisableController(
        num_envs,
        device=env.device,
        initial_disabled=initial_unstable,
    )
    collectors = BatchedSysIdCollectors(num_envs)
    metadata_rows = [
        build_episode_metadata(
            episode_id=collectors.episode_id(i),
            env=env,
            env_idx=i,
            num_directions=int(num_directions),
            pull_direction=per_env_directions[i],
            seed=int(seed),
            config=config,
            ranges_path=ranges_path,
        )
        for i in range(num_envs)
    ]

    def _finalize() -> Path:
        frame_paths = collectors.save_all(
            out,
            metadata_rows,
            num_directions=int(num_directions),
        )
        write_manifest(
            out,
            command_argv=list(command_argv if command_argv is not None else sys.argv),
            collection={
                "seed": int(seed),
                "topology_seed": int(getattr(env, "_topology_seed", 0)),
                "ranges_path": str(Path(ranges_path)),
                "control_hz": float(config.control_hz),
                "num_structures": int(num_structures),
                "num_directions": int(num_directions),
                "max_steps": int(max_steps),
                "trajectory": {
                    "movement_per_step_m": float(config.movement_per_step_m),
                    "total_movement_m": float(config.total_movement_m),
                    "hold_duration_s": float(config.hold_duration_s),
                    "move_speed_mps": float(config.move_speed_mps),
                    "skip_return": bool(config.skip_return),
                },
                "sim_config": sim_config_to_manifest_dict(
                    env._sim.config,
                    applied_joint_kd_overrides=env._sim.build_result.joint_angular_kd_overrides,
                    applied_joint_linear_kd_overrides=env._sim.build_result.joint_linear_kd_overrides,
                    applied_joint_angular_kp_overrides=env._sim.build_result.joint_angular_kp_overrides,
                    applied_joint_linear_kp_overrides=env._sim.build_result.joint_linear_kp_overrides,
                ),
            },
            structures=_build_structure_summaries(metadata_rows),
            episodes=_build_manifest_episodes(
                metadata_rows,
                collectors.writers,
                num_directions=int(num_directions),
                excluded_env_indices=_excluded_env_indices(disable_ctrl, collectors.writers),
            ),
            overwrite=bool(overwrite),
        )
        if progress is not None:
            progress(f"wrote {len(frame_paths)} episodes to {out}")
        return out

    sim_time = 0.0
    pre_weld_report = monitor.check(obs, step_idx=PRE_WELD_STEP_IDX)
    for env_idx in range(num_envs):
        pre_weld_obs = env.pre_weld_sysid_obs(int(env_idx))
        if pre_weld_obs is not None:
            collectors.record_pre_weld_step(
                env_idx=int(env_idx),
                obs=complete_pre_weld_sysid_obs(
                    pre_weld_obs,
                    pull_direction=per_env_directions[int(env_idx)],
                ),
                stable=not bool(pre_weld_report.unstable[int(env_idx)].item()),
            )

    if on_step is not None and not on_step(
        env=env,
        step_idx=PRE_WELD_STEP_IDX,
        phase="pre_weld",
        sim_time=sim_time,
        obs=obs,
        amplitude_m=0.0,
    ):
        if progress is not None:
            progress(f"wrote episodes to {out} (stopped at init)")
        return _finalize()

    for step_idx, (phase, vel) in enumerate(reference_traj.iter_frames()):
        if step_idx >= step_cap:
            break

        contexts = [
            ExcitationContext(
                type="quasi_static",
                f_inst=0.0,
                direction=per_env_directions[i],
            )
            for i in range(num_envs)
        ]
        env.set_excitation_contexts(contexts)
        actions = actions_tensor_for_velocity(
            vel,
            per_env_directions,
            device=env.device,
        )
        actions = disable_ctrl.apply_actions(actions)
        obs, _reward, _terminated, _truncated, _info = env.step(actions)
        sim_time += 1.0 / float(config.control_hz)
        step_report = monitor.check(obs, step_idx=step_idx)
        record_mask = disable_ctrl.should_record_mask().detach().cpu().tolist()

        for i in range(num_envs):
            if not bool(record_mask[i]):
                continue
            action_np = actions[i].detach().cpu().numpy()
            collectors.record_step(
                env,
                env_idx=i,
                step_idx=step_idx,
                sim_time=sim_time,
                phase=phase,
                amplitude_m=reference_traj.current_amplitude_m,
                action=action_np,
                stable=not bool(step_report.unstable[i].item()),
            )
        disable_ctrl.update(step_report.unstable)

        if progress is not None and step_idx % 20 == 0:
            progress(f"step {step_idx} phase={phase}")

        if on_step is not None and not on_step(
            env=env,
            step_idx=step_idx,
            phase=phase,
            sim_time=sim_time,
            obs=obs,
            amplitude_m=float(reference_traj.current_amplitude_m),
        ):
            break

    return _finalize()
