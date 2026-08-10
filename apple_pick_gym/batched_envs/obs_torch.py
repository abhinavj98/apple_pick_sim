"""Reshape GPU batched obs buffers into SKRL-ready torch observation dicts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import warp as wp

from apple_pick_sim.batched_obs import BatchedObsBuffers


def _to_torch(arr: wp.array, device: torch.device) -> torch.Tensor:
    return wp.to_torch(arr).to(device=device, dtype=torch.float32)


def obs_dict_from_bufs(
    bufs: BatchedObsBuffers,
    junction_names: list[str],
    device: torch.device | str,
) -> dict[str, Any]:
    """Build batched VIC observation dict with ``woody_part_info`` junction groups."""
    dev = torch.device(device)
    n_envs = int(bufs.num_envs)
    n_j = int(bufs.num_junctions)
    if len(junction_names) != n_j:
        raise ValueError(
            f"junction_names length ({len(junction_names)}) must match num_junctions ({n_j})"
        )

    parent = _to_torch(bufs.woody_parent_pos, dev).reshape(n_envs, n_j, 3)
    child = _to_torch(bufs.woody_child_pos, dev).reshape(n_envs, n_j, 3)
    force = _to_torch(bufs.woody_force, dev).reshape(n_envs, n_j, 3)
    torque = _to_torch(bufs.woody_torque, dev).reshape(n_envs, n_j, 3)

    woody_part_info: dict[str, dict[str, torch.Tensor]] = {}
    for j, name in enumerate(junction_names):
        woody_part_info[name] = {
            "anchors_pos": torch.cat([parent[:, j, :], child[:, j, :]], dim=-1),
            "anchor_force": torch.cat([force[:, j, :], torque[:, j, :]], dim=-1),
        }

    return {
        "woody_part_info": woody_part_info,
        "apple_pos": _to_torch(bufs.apple_pos, dev),
        "tcp_force": _to_torch(bufs.tcp_force, dev),
        "tcp_velocity": _to_torch(bufs.tcp_velocity, dev),
        "ft_wrist": _to_torch(bufs.tcp_coupling_force, dev),
    }


def legacy_v3_numpy_from_batched(obs: dict[str, Any], junction_names: list[str]) -> dict[str, Any]:
    """Map batched torch obs to legacy v3 numpy dict at ``num_envs=1`` (for parity tests)."""
    import numpy as np

    n = 0
    woody_start: dict[str, np.ndarray] = {}
    woody_end: dict[str, np.ndarray] = {}
    woody_force_parts: list[np.ndarray] = []
    for name in junction_names:
        anchors = obs["woody_part_info"][name]["anchors_pos"][n].detach().cpu().numpy()
        woody_start[name] = np.asarray(anchors[:3], dtype=np.float32)
        woody_end[name] = np.asarray(anchors[3:], dtype=np.float32)
        woody_force_parts.append(
            obs["woody_part_info"][name]["anchor_force"][n].detach().cpu().numpy()
        )

    return {
        "woody_part_start_pos": woody_start,
        "woody_part_end_pos": woody_end,
        "woody_part_force": np.concatenate(woody_force_parts, dtype=np.float32),
        "apple_pos": obs["apple_pos"][n].detach().cpu().numpy(),
        "tcp_force": obs["tcp_force"][n].detach().cpu().numpy(),
        "tcp_velocity": obs["tcp_velocity"][n].detach().cpu().numpy(),
        "ft_wrist": obs["ft_wrist"][n].detach().cpu().numpy(),
    }

def sysid_numpy_obs_from_batched(
    obs: dict[str, Any],
    junction_names: list[str],
    env_idx: int,
) -> dict[str, Any]:
    """Map batched sys-ID torch obs to legacy v3 numpy dict for one env index."""
    n = int(env_idx)
    woody_start: dict[str, np.ndarray] = {}
    woody_end: dict[str, np.ndarray] = {}
    woody_force_parts: list[np.ndarray] = []
    for name in junction_names:
        anchors = obs["woody_part_info"][name]["anchors_pos"][n].detach().cpu().numpy()
        woody_start[name] = np.asarray(anchors[:3], dtype=np.float32)
        woody_end[name] = np.asarray(anchors[3:], dtype=np.float32)
        woody_force_parts.append(
            obs["woody_part_info"][name]["anchor_force"][n].detach().cpu().numpy()
        )
    return {
        "woody_part_start_pos": woody_start,
        "woody_part_end_pos": woody_end,
        "woody_part_force": np.concatenate(woody_force_parts, dtype=np.float32),
        "apple_pos": obs["apple_pos"][n].detach().cpu().numpy(),
        "tcp_force": obs["tcp_force"][n].detach().cpu().numpy(),
        "tcp_velocity": obs["tcp_velocity"][n].detach().cpu().numpy(),
        "ft_wrist": obs["ft_wrist"][n].detach().cpu().numpy(),
        "raw_ft_wrist": obs["raw_ft_wrist"][n].detach().cpu().numpy(),
        "tcp_pos": obs["tcp_pos"][n].detach().cpu().numpy(),
        "tcp_quat": obs["tcp_quat"][n].detach().cpu().numpy(),
        "apple_quat": obs["apple_quat"][n].detach().cpu().numpy(),
        "robot_joint_q": obs["robot_joint_q"][n].detach().cpu().numpy(),
        "excitation_type": int(obs["excitation_type"][n].detach().cpu().item()),
        "excitation_f_inst": float(obs["excitation_f_inst"][n].detach().cpu().item()),
        "excitation_direction": obs["excitation_direction"][n].detach().cpu().numpy(),
    }


def _torch_to_numpy_f32_copy(value: Any) -> np.ndarray:
    """Detach to CPU float32 numpy **owned** by NumPy (no torch buffer alias)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=np.float32, copy=True)


def download_batched_replay_obs_numpy(
    obs: Mapping[str, Any],
    junction_names: list[str],
) -> dict[str, np.ndarray]:
    """Download one batched sys-ID obs frame to CPU numpy (one sync per field)."""
    woody_start_parts: list[np.ndarray] = []
    woody_end_parts: list[np.ndarray] = []
    for name in junction_names:
        anchors = _torch_to_numpy_f32_copy(obs["woody_part_info"][name]["anchors_pos"])
        woody_start_parts.append(np.asarray(anchors[:, :3], dtype=np.float32).copy())
        woody_end_parts.append(np.asarray(anchors[:, 3:6], dtype=np.float32).copy())

    return {
        "ft_wrist": _torch_to_numpy_f32_copy(obs["ft_wrist"]),
        "tcp_velocity": _torch_to_numpy_f32_copy(obs["tcp_velocity"]),
        "tcp_pos": _torch_to_numpy_f32_copy(obs["tcp_pos"]),
        "apple_pos": _torch_to_numpy_f32_copy(obs["apple_pos"]),
        "woody_start": np.concatenate(woody_start_parts, axis=1),
        "woody_end": np.concatenate(woody_end_parts, axis=1),
    }


def replay_obs_dict_from_batched_numpy_row(
    batched: Mapping[str, np.ndarray],
    *,
    env_idx: int,
) -> dict[str, Any]:
    """Slice one env row from :func:`download_batched_replay_obs_numpy` output."""
    i = int(env_idx)
    return {
        "ft_wrist": np.asarray(batched["ft_wrist"][i], dtype=np.float32).reshape(6),
        "tcp_velocity": np.asarray(batched["tcp_velocity"][i], dtype=np.float32).reshape(6),
        "tcp_pos": np.asarray(batched["tcp_pos"][i], dtype=np.float32).reshape(3),
        "apple_pos": np.asarray(batched["apple_pos"][i], dtype=np.float32).reshape(3),
        "woody_start": np.asarray(batched["woody_start"][i], dtype=np.float32),
        "woody_end": np.asarray(batched["woody_end"][i], dtype=np.float32),
    }

