"""Orchestration helpers for parallel batched sys-ID Parquet collection."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from apple_pick_gym.batched_envs.batched_sysid_world_info import (
    physical_stem_dir_for_world,
    robot_base_pos_for_world,
)
from apple_pick_sim.fruiting_system import (
    fruiting_params_to_json,
    load_ranges,
    sample_heterogeneous_params_list,
)
from apple_pick_sim.system_id import (
    EpisodeMeta,
    ExcitationContext,
    QuasiStaticStepConfig,
    QuasiStaticTrajectory,
    TrajectoryWriter,
    estimate_trajectory_frames,
    sample_robot_facing_pull_directions,
)


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
        dirs = sample_robot_facing_pull_directions(d_count, physical_stem, robot_vec)
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
    """One TrajectoryWriter per parallel env."""

    def __init__(self, num_envs: int) -> None:
        self._writers = [TrajectoryWriter(episode_id=str(uuid4())) for _ in range(int(num_envs))]

    @property
    def writers(self) -> list[TrajectoryWriter]:
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
    ) -> None:
        obs = env.sysid_numpy_obs(int(env_idx))
        self._writers[int(env_idx)].record_step(
            step_idx=int(step_idx),
            sim_time=float(sim_time),
            phase=str(phase),
            dir_idx=0,
            amplitude_m=float(amplitude_m),
            action=np.asarray(action, dtype=np.float32),
            obs=obs,
        )

    def save_all(
        self,
        output_dir: Path | str,
        meta_rows: Sequence[EpisodeMeta],
    ) -> list[Path]:
        out = Path(output_dir)
        paths: list[Path] = []
        if len(meta_rows) != len(self._writers):
            raise ValueError("meta_rows length must match number of writers")
        for writer, meta in zip(self._writers, meta_rows, strict=True):
            paths.append(writer.save(out, meta))
        return paths


def build_episode_meta(
    *,
    episode_id: str,
    env: Any,
    env_idx: int,
    seed: int,
    config: QuasiStaticStepConfig,
    ranges_path: Path | str,
    reset_obs: dict[str, Any],
) -> EpisodeMeta:
    per_env = env.per_env_reset_info(int(env_idx))
    params = env._sim._per_env_params[int(env_idx)]
    scene = env._sim.scene
    weld = per_env["weld_direction"]
    weld_tuple = tuple(float(x) for x in np.asarray(weld, dtype=np.float64).reshape(3))
    weld_norm = float(np.linalg.norm(weld_tuple))
    if weld_norm < 1e-12:
        weld_tuple = (0.0, 0.0, 1.0)
    else:
        weld_tuple = (
            float(weld_tuple[0] / weld_norm),
            float(weld_tuple[1] / weld_norm),
            float(weld_tuple[2] / weld_norm),
        )

    exported = env.sysid_numpy_obs(int(env_idx))
    fruiting_base = per_env.get("fruiting_base_pos")
    return EpisodeMeta(
        episode_id=str(episode_id),
        weld_direction=weld_tuple,
        excitation_type="quasi_static",
        n_woody_parts=len(env.junction_names),
        junction_names=list(env.junction_names),
        params_fingerprint=(
            json.dumps(per_env["params_fingerprint"], sort_keys=True)
            if isinstance(per_env["params_fingerprint"], dict)
            else str(per_env["params_fingerprint"])
        ),
        fruiting_system_params=fruiting_params_to_json(params),
        control_hz=float(config.control_hz),
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=int(seed),
        n_directions=1,
        initial_tcp_pos=tuple(float(x) for x in np.asarray(exported["tcp_pos"]).reshape(3)),
        initial_tcp_quat=tuple(float(x) for x in np.asarray(exported["tcp_quat"]).reshape(4)),
        initial_apple_pos=tuple(float(x) for x in np.asarray(exported["apple_pos"]).reshape(3)),
        initial_apple_quat=tuple(float(x) for x in np.asarray(exported["apple_quat"]).reshape(4)),
        initial_robot_joint_q=tuple(
            float(x) for x in np.asarray(exported["robot_joint_q"]).reshape(-1)
        ),
        fixture_path=str(Path(ranges_path)),
        fruiting_base_pos=(
            None
            if fruiting_base is None
            else tuple(float(x) for x in np.asarray(fruiting_base).reshape(3))
        ),
        apple_radius=per_env.get("apple_radius"),
        rod_radii=per_env.get("rod_radii"),
        weld_reference_pos=tuple(
            float(x) for x in np.asarray(per_env["weld_reference_pos"]).reshape(3)
        ),
        weld_reference_quat=tuple(
            float(x) for x in np.asarray(per_env["weld_reference_quat"]).reshape(4)
        ),
        movement_per_step_m=float(config.movement_per_step_m),
        total_movement_m=float(config.total_movement_m),
        hold_duration_s=float(config.hold_duration_s),
        move_speed_mps=float(config.move_speed_mps),
        skip_return=bool(config.skip_return),
    )


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
) -> Path:
    """Run lockstep quasi-static collection and write Parquet episodes for all envs."""
    out = Path(output_dir)
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
    )
    reference_traj = QuasiStaticTrajectory(
        np.asarray(per_env_directions[0], dtype=np.float64).reshape(1, 3),
        config,
    )

    step_cap = int(max_steps)
    if step_cap <= 0:
        step_cap = estimate_trajectory_frames(config, 1) + 64

    obs, _info = env.reset(seed=int(seed))
    collectors = BatchedSysIdCollectors(num_envs)
    meta_rows = [
        build_episode_meta(
            episode_id=collectors.episode_id(i),
            env=env,
            env_idx=i,
            seed=int(seed),
            config=config,
            ranges_path=ranges_path,
            reset_obs=obs,
        )
        for i in range(num_envs)
    ]

    sim_time = 0.0
    dir_idx = 0
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
        obs, _reward, _terminated, _truncated, _info = env.step(actions)
        sim_time += 1.0 / float(config.control_hz)

        for i in range(num_envs):
            action_np = actions[i].detach().cpu().numpy()
            collectors.record_step(
                env,
                env_idx=i,
                step_idx=step_idx,
                sim_time=sim_time,
                phase=phase,
                amplitude_m=reference_traj.current_amplitude_m,
                action=action_np,
            )

        if progress is not None and step_idx % 20 == 0:
            progress(f"step {step_idx} phase={phase}")

    frame_paths = collectors.save_all(out, meta_rows)
    if progress is not None:
        progress(f"wrote {len(frame_paths)} episodes to {out}")
    return out
