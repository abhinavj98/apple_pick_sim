"""Dense harvest reward shaping: progress, pull-out, and collateral-force penalties.

Ported from ``feature/rl-gym``'s ``harvest_reward.py``; the progress term now
uses the combined force + torque detach envelope (``harvest_detach.py``,
maintainer decision 2026-09-24) instead of a force-only threshold:

- **progress** = ``clip(sqrt((|F|/F_max)^2 + (|tau|/tau_max)^2), 0, 1)`` of the
  target (spur-stem) junction's anchor-frame wrench -- the radial fraction of
  the detach envelope in use. Success is the same envelope at ``>= 1``.
- **pullout** = ``relu(F_wrist . ee_z)``, force along the gripper axis.
- **collateral** = ``sum_j relu(|F_j| - |F_j|_rest)`` over the non-target
  junctions. The rest baseline (``info["collateral_baseline_norm"]``, recorded
  by the env at reset) removes the constant gravity load the plant carries at
  rest, so the term measures load the *policy* adds to the spur, primary and
  supports -- the objective is enough load at the spur-stem junction and as
  little extra as possible everywhere else.

Reward is privileged (train-time only): it reads uncapped junction wrenches
and raw ``ft_wrist`` from ``info``, not the sensor-realistic
``obs["ft_wrist"]`` the policy observes. This is legitimate because reward
computation is not part of the deployed policy (see the design spec's
"Reward and termination" section). ``F_max = 20 N`` is below the 40 N stem
harvest cap, so ``obs["ft_wrist"]`` stays informative up to detachment.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from apple_pick_gym.batched_envs.harvest_detach import DetachEnvelopeConfig, detach_utilization


@dataclasses.dataclass(frozen=True)
class HarvestRewardConfig:
    """Dense reward weights plus the target junction's detach envelope."""

    detach: DetachEnvelopeConfig = dataclasses.field(default_factory=DetachEnvelopeConfig)
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
    target_junction_wrench: torch.Tensor, cfg: HarvestRewardConfig
) -> torch.Tensor:
    """Detach-envelope utilization of the ``(N, 6)`` target wrench, clipped to ``[0, 1]``, shape (N,)."""
    return torch.clamp(detach_utilization(target_junction_wrench, cfg.detach), min=0.0, max=1.0)


def compute_pullout_penalty(ft_wrist: torch.Tensor, tcp_quat: torch.Tensor) -> torch.Tensor:
    """``relu(F_tcp . ee_z_world)``, shape (N,) -- force along the EE's local +z (grip) axis."""
    ee_z_local = torch.zeros_like(ft_wrist[:, :3])
    ee_z_local[:, 2] = 1.0
    ee_z_world = quat_rotate_vector(tcp_quat, ee_z_local)
    along_z = torch.sum(ft_wrist[:, :3] * ee_z_world, dim=-1)
    return torch.clamp(along_z, min=0.0)


def compute_collateral_penalty(
    woody_part_force: dict[str, torch.Tensor],
    *,
    target_junction_name: str,
    baseline_norm: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Sum over every non-target junction of ``||F_j[:3]||``, shape (N,).

    With ``baseline_norm`` (per-junction ``(N,)`` rest force norms), only load above
    rest counts: ``relu(||F_j|| - baseline_j)``. Unloading below rest is not rewarded.
    """
    total = None
    for name, wrench in woody_part_force.items():
        if name == target_junction_name:
            continue
        norm = torch.linalg.norm(wrench[:, :3], dim=-1)
        if baseline_norm is not None:
            norm = torch.clamp(norm - baseline_norm[name].to(norm), min=0.0)
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
    """Raw (unweighted) dense reward terms, each shape (N,).

    Reads ``info["target_junction_wrench"]`` (anchor-frame target wrench),
    ``info["ft_wrist"]``, ``info["woody_part_force"]`` and, if present,
    ``info["collateral_baseline_norm"]``.
    """
    return {
        "progress": compute_progress_reward(info["target_junction_wrench"], cfg),
        "pullout": compute_pullout_penalty(info["ft_wrist"], obs["tcp_quat"]),
        "collateral": compute_collateral_penalty(
            info["woody_part_force"],
            target_junction_name=target_junction_name,
            baseline_norm=info.get("collateral_baseline_norm"),
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
