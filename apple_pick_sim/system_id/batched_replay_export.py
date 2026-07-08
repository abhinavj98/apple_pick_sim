"""Export batched sys-ID MMD grid replay trajectories to Parquet datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from apple_pick_sim.fruiting_system import fruiting_params_to_json, params_fingerprint
from apple_pick_sim.fruiting_system.params import FruitingSystemParams
from apple_pick_sim.system_id.batched_trajectory_store import (
    BatchedEpisodeWriter,
    BatchedSysIdDataset,
    SCHEMA_VERSION,
    episode_filename,
    write_manifest,
)
from apple_pick_sim.system_id.trajectory_store import PHASE_TO_INT

REPLAY_SCHEMA_VERSION = "batched_sysid_replay_v1"
INT_TO_PHASE: dict[int, str] = {value: name for name, value in PHASE_TO_INT.items()}


@dataclass(frozen=True)
class ReplayCandidateSpec:
    """One stiffness candidate to export as a replay mini-dataset."""

    candidate_index: int
    params: FruitingSystemParams
    stiffnesses: dict[str, float]


def candidate_replay_dir(
    export_dir: Path | str,
    *,
    structure_idx: int,
    candidate_index: int,
) -> Path:
    """Root directory for one candidate's exported replay dataset."""
    return (
        Path(export_dir)
        / f"structure_{int(structure_idx):03d}"
        / "candidates"
        / f"c{int(candidate_index):03d}"
    )


def candidate_export_exists(candidate_dir: Path | str) -> bool:
    """Return True when a candidate export already has a manifest."""
    return (Path(candidate_dir) / "manifest.json").is_file()


def _phase_name(phase_value: int) -> str:
    return INT_TO_PHASE.get(int(phase_value), f"phase_{int(phase_value)}")


def _gt_frame_array(
    gt_recorded: Mapping[str, Any] | None,
    key: str,
    *,
    n_frames: int,
    default: np.ndarray | float,
) -> np.ndarray:
    if gt_recorded is None or key not in gt_recorded:
        if isinstance(default, np.ndarray):
            return np.broadcast_to(default, (n_frames, *default.shape)).copy()
        return np.full(n_frames, float(default), dtype=np.float32)
    values = np.asarray(gt_recorded[key])
    if values.ndim == 1 and values.shape[0] == n_frames:
        return values
    if values.ndim == 2 and values.shape[0] == n_frames:
        return values
    if isinstance(default, np.ndarray):
        return np.broadcast_to(default, (n_frames, *default.shape)).copy()
    return np.full(n_frames, float(default), dtype=np.float32)


def _build_replay_episode_metadata(
    *,
    gt_metadata: Mapping[str, Any],
    candidate_params: FruitingSystemParams,
    direction_idx: int,
    episode_id: str,
    mini_structure_idx: int = 0,
) -> dict[str, Any]:
    meta = dict(gt_metadata)
    meta.update(
        {
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            "structure_idx": int(mini_structure_idx),
            "direction_idx": int(direction_idx),
            "env_idx": int(direction_idx),
            "params_fingerprint": json.dumps(
                params_fingerprint(candidate_params),
                sort_keys=True,
            ),
            "fruiting_system_params": fruiting_params_to_json(candidate_params),
        }
    )
    return meta


def _write_replay_episode_parquet(
    path: Path,
    *,
    replay: Mapping[str, Any],
    episode_metadata: Mapping[str, Any],
    gt_recorded: Mapping[str, Any] | None,
    control_hz: float,
) -> Path:
    writer = BatchedEpisodeWriter(episode_id=str(episode_metadata["episode_id"]))
    n_frames = int(np.asarray(replay["action"]).shape[0])
    junction_names = list(replay["junction_names"])

    sim_time = _gt_frame_array(gt_recorded, "sim_time", n_frames=n_frames, default=0.0)
    amplitude_m = _gt_frame_array(gt_recorded, "amplitude_m", n_frames=n_frames, default=0.0)
    raw_ft = _gt_frame_array(
        gt_recorded,
        "raw_ft_wrist",
        n_frames=n_frames,
        default=np.asarray(replay["ft_wrist"][0], dtype=np.float32),
    )
    tcp_quat = _gt_frame_array(
        gt_recorded,
        "tcp_quat",
        n_frames=n_frames,
        default=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    apple_quat = _gt_frame_array(
        gt_recorded,
        "apple_quat",
        n_frames=n_frames,
        default=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    robot_joint_q = _gt_frame_array(
        gt_recorded,
        "robot_joint_q",
        n_frames=n_frames,
        default=np.zeros(7, dtype=np.float32),
    )
    woody_part_force = _gt_frame_array(
        gt_recorded,
        "woody_part_force",
        n_frames=n_frames,
        default=np.zeros(0, dtype=np.float32),
    )

    for frame_idx in range(n_frames):
        obs = {
            "excitation_type": int(np.asarray(replay["excitation_type"])[frame_idx]),
            "excitation_direction": np.asarray(replay["excitation_direction"])[frame_idx],
            "tcp_velocity": np.asarray(replay["tcp_velocity"])[frame_idx],
            "ft_wrist": np.asarray(replay["ft_wrist"])[frame_idx],
            "raw_ft_wrist": raw_ft[frame_idx] if raw_ft.ndim > 1 else raw_ft,
            "tcp_pos": np.asarray(replay["tcp_pos"])[frame_idx],
            "apple_pos": np.asarray(replay["apple_pos"])[frame_idx],
            "tcp_quat": tcp_quat[frame_idx] if tcp_quat.ndim > 1 else tcp_quat,
            "apple_quat": apple_quat[frame_idx] if apple_quat.ndim > 1 else apple_quat,
            "robot_joint_q": (
                robot_joint_q[frame_idx] if robot_joint_q.ndim > 1 else robot_joint_q
            ),
            "woody_part_force": (
                woody_part_force[frame_idx]
                if woody_part_force.ndim > 1
                else woody_part_force
            ),
            "woody_part_start_pos": {
                name: np.asarray(replay["woody_part_start_pos"][name])[frame_idx]
                for name in junction_names
            },
            "woody_part_end_pos": {
                name: np.asarray(replay["woody_part_end_pos"][name])[frame_idx]
                for name in junction_names
            },
        }
        writer.record_step(
            step_idx=frame_idx,
            sim_time=float(sim_time[frame_idx] if sim_time.ndim else sim_time),
            phase=_phase_name(int(np.asarray(replay["phase"])[frame_idx])),
            amplitude_m=float(
                amplitude_m[frame_idx] if amplitude_m.ndim else amplitude_m
            ),
            action=np.asarray(replay["action"])[frame_idx],
            obs=obs,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    return writer.save(path, dict(episode_metadata))


def write_replay_candidate_dataset(
    candidate_dir: Path | str,
    *,
    source_dataset: BatchedSysIdDataset,
    source_structure_idx: int,
    spec: ReplayCandidateSpec,
    replay_eps_by_direction: Sequence[Mapping[str, Any]],
    command_argv: Sequence[str] | None = None,
    skip_existing: bool = False,
) -> Path | None:
    """Write one candidate replay mini-dataset (manifest + episode parquets)."""
    out = Path(candidate_dir)
    if skip_existing and candidate_export_exists(out):
        return None

    num_directions = len(replay_eps_by_direction)
    if num_directions < 1:
        raise ValueError("replay_eps_by_direction must be non-empty")

    collection = source_dataset.manifest.get("collection", {})
    control_hz = float(collection.get("control_hz", 30.0))
    source_root = str(source_dataset.dataset_dir.resolve())

    episodes: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for direction_idx, replay in enumerate(replay_eps_by_direction):
        gt_meta = source_dataset.load_episode_metadata(
            int(source_structure_idx),
            int(direction_idx),
        )
        gt_recorded = source_dataset.load_episode_obs_arrays(
            int(source_structure_idx),
            int(direction_idx),
        )
        episode_id = str(uuid4())
        ep_meta = _build_replay_episode_metadata(
            gt_metadata=gt_meta,
            candidate_params=spec.params,
            direction_idx=int(direction_idx),
            episode_id=episode_id,
            mini_structure_idx=0,
        )
        rel = episode_filename(0, int(direction_idx))
        writer_path = out / rel
        _write_replay_episode_parquet(
            writer_path,
            replay=replay,
            episode_metadata=ep_meta,
            gt_recorded=gt_recorded,
            control_hz=control_hz,
        )
        metadata_rows.append(ep_meta)
        episodes.append(
            {
                "structure_idx": 0,
                "direction_idx": int(direction_idx),
                "env_idx": int(direction_idx),
                "filename": rel,
                "episode_id": episode_id,
                "pull_direction": ep_meta["pull_direction"],
                "n_frames": int(np.asarray(replay["action"]).shape[0]),
            }
        )

    structures = [
        {
            "structure_idx": 0,
            "params_fingerprint": metadata_rows[0]["params_fingerprint"],
            "junction_names": metadata_rows[0]["junction_names"],
            "n_woody_parts": metadata_rows[0]["n_woody_parts"],
        }
    ]
    replay_manifest = {
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "source_dataset": source_root,
        "source_structure_idx": int(source_structure_idx),
        "candidate_index": int(spec.candidate_index),
        "candidate_stiffnesses": dict(spec.stiffnesses),
    }
    write_manifest(
        out,
        command_argv=list(command_argv or []),
        collection={
            **collection,
            "num_structures": 1,
            "num_directions": num_directions,
        },
        structures=structures,
        episodes=episodes,
        overwrite=True,
    )
    manifest_path = out / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["replay"] = replay_manifest
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def export_replay_candidates_for_structure(
    export_dir: Path | str,
    *,
    source_dataset: BatchedSysIdDataset,
    source_structure_idx: int,
    specs_and_replays: Sequence[tuple[ReplayCandidateSpec, Sequence[Mapping[str, Any]]]],
    command_argv: Sequence[str] | None = None,
    skip_existing: bool = False,
) -> int:
    """Export replay trajectories for all candidates of one structure."""
    n_written = 0
    for spec, replay_eps in specs_and_replays:
        out = candidate_replay_dir(
            export_dir,
            structure_idx=int(source_structure_idx),
            candidate_index=int(spec.candidate_index),
        )
        result = write_replay_candidate_dataset(
            out,
            source_dataset=source_dataset,
            source_structure_idx=int(source_structure_idx),
            spec=spec,
            replay_eps_by_direction=replay_eps,
            command_argv=command_argv,
            skip_existing=bool(skip_existing),
        )
        if result is not None:
            n_written += 1
    return n_written
