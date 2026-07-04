"""Reshape GPU batched obs buffers into SKRL-ready torch observation dicts."""

from __future__ import annotations

from typing import Any

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
    import numpy as np

    n = int(env_idx)
    out = legacy_v3_numpy_from_batched(obs, junction_names)
    # legacy helper always reads env 0; rebuild for arbitrary index.
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
    out["woody_part_start_pos"] = woody_start
    out["woody_part_end_pos"] = woody_end
    out["woody_part_force"] = np.concatenate(woody_force_parts, dtype=np.float32)
    out["apple_pos"] = obs["apple_pos"][n].detach().cpu().numpy()
    out["tcp_force"] = obs["tcp_force"][n].detach().cpu().numpy()
    out["tcp_velocity"] = obs["tcp_velocity"][n].detach().cpu().numpy()
    out["ft_wrist"] = obs["ft_wrist"][n].detach().cpu().numpy()
    out["raw_ft_wrist"] = obs["raw_ft_wrist"][n].detach().cpu().numpy()
    out["tcp_pos"] = obs["tcp_pos"][n].detach().cpu().numpy()
    out["tcp_quat"] = obs["tcp_quat"][n].detach().cpu().numpy()
    out["apple_quat"] = obs["apple_quat"][n].detach().cpu().numpy()
    out["robot_joint_q"] = obs["robot_joint_q"][n].detach().cpu().numpy()
    out["excitation_type"] = int(obs["excitation_type"][n].detach().cpu().item())
    out["excitation_f_inst"] = float(obs["excitation_f_inst"][n].detach().cpu().item())
    out["excitation_direction"] = obs["excitation_direction"][n].detach().cpu().numpy()
    return out

