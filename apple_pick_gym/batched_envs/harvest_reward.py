"""Dense harvest reward shaping: progress, pull-out, and collateral-force penalties.

Ported from ``feature/rl-gym``'s ``harvest_reward.py`` (math unchanged), with
``f_threshold_n`` updated to the maintainer's current placeholder value.
``F_thresh = 5 N`` is explicitly provisional -- flagged by the maintainer for
a later revisit, likely a combined force+torque success criterion.

Reward is privileged (train-time only): it reads uncapped junction forces
and raw ``ft_wrist`` from ``info``, not the sensor-realistic
``obs["ft_wrist"]`` the policy observes. This is legitimate because reward
computation is not part of the deployed policy (see the design spec's
"Reward and termination" section). At the current 5 N threshold this
distinction is largely moot in practice -- 5 N is far below the 40 N stem
harvest cap, so ``obs["ft_wrist"]`` is fully informative there too -- but it
would matter if the threshold is later raised.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch


@dataclasses.dataclass(frozen=True)
class HarvestRewardConfig:
    """Dense reward weights plus the (explicitly provisional) force threshold."""

    f_threshold_n: float = 10.0
    w_progress: float = 1.0
    w_pullout: float = 0.5
    w_collateral: float = 0.1
    success_bonus: float = 10.0
    failure_penalty: float = -20.0


def quat_rotate_vector(quat_xyzw: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` (N,3) by unit quaternion ``quat_xyzw`` (N,4), both world-frame."""
    q_xyz = quat_xyzw[..., :3]
    q_w = quat_xyzw[..., 3:4]
    t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_xyz, t, dim=-1)


def compute_progress_reward(
    target_junction_force: torch.Tensor, cfg: HarvestRewardConfig
) -> torch.Tensor:
    """``clip(||F_target[:3]|| / f_threshold_n, 0, 1)``, shape (N,)."""
    force_norm = torch.linalg.norm(target_junction_force[:, :3], dim=-1)
    return torch.clamp(force_norm / float(cfg.f_threshold_n), min=0.0, max=1.0)


def compute_pullout_penalty(ft_wrist: torch.Tensor, tcp_quat: torch.Tensor) -> torch.Tensor:
    """``relu(F_tcp . ee_z_world)``, shape (N,) -- force along the EE's local +z (grip) axis."""
    ee_z_local = torch.zeros_like(ft_wrist[:, :3])
    ee_z_local[:, 2] = 1.0
    ee_z_world = quat_rotate_vector(tcp_quat, ee_z_local)
    along_z = torch.sum(ft_wrist[:, :3] * ee_z_world, dim=-1)
    return torch.clamp(along_z, min=0.0)


def compute_collateral_penalty(
    woody_part_force: dict[str, torch.Tensor], *, target_junction_name: str
) -> torch.Tensor:
    """Sum of ``||F_j[:3]||`` over every junction except ``target_junction_name``."""
    total = None
    for name, wrench in woody_part_force.items():
        if name == target_junction_name:
            continue
        norm = torch.linalg.norm(wrench[:, :3], dim=-1)
        total = norm if total is None else total + norm
    if total is None:
        raise ValueError("woody_part_force has no non-target junctions to penalize")
    return total


def compute_dense_reward_terms(
    obs: dict[str, Any],
    info: dict[str, Any],
    *,
    target_junction_name: str,
    cfg: HarvestRewardConfig,
) -> dict[str, torch.Tensor]:
    """Raw (unweighted) dense reward terms, each shape (N,)."""
    return {
        "progress": compute_progress_reward(info["target_junction_force"], cfg),
        "pullout": compute_pullout_penalty(info["ft_wrist"], obs["tcp_quat"]),
        "collateral": compute_collateral_penalty(
            info["woody_part_force"], target_junction_name=target_junction_name
        ),
    }


def weight_dense_reward_terms(
    terms: dict[str, torch.Tensor], cfg: HarvestRewardConfig
) -> dict[str, torch.Tensor]:
    """Signed weighted contributions to the dense reward, each shape (N,)."""
    return {
        "progress": cfg.w_progress * terms["progress"],
        "pullout": -cfg.w_pullout * terms["pullout"],
        "collateral": -cfg.w_collateral * terms["collateral"],
    }


def compute_dense_reward(
    obs: dict[str, Any],
    info: dict[str, Any],
    *,
    target_junction_name: str,
    cfg: HarvestRewardConfig,
) -> torch.Tensor:
    """Weighted dense shaping reward (excludes the sparse success/failure bonus), shape (N,1)."""
    terms = compute_dense_reward_terms(
        obs, info, target_junction_name=target_junction_name, cfg=cfg
    )
    weighted = weight_dense_reward_terms(terms, cfg)
    reward = weighted["progress"] + weighted["pullout"] + weighted["collateral"]
    return reward.unsqueeze(-1)
