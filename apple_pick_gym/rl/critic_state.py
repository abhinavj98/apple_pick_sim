"""The privileged critic state: the harvest critic layout plus RL-side extras.

Layout (``critic_state_layout``), in order:

1. ``harvest_obs.critic_obs_layout`` -- the 40-D actor vector (exact prefix), the
   plant / support / arm DR ground truth (``_PRIVILEGED_FIELDS``) and every junction's
   raw wrench (sorted junction names);
2. ``PLANT_GEOMETRY_FIELDS`` -- rod / apple geometry and the grasp's weld axis;
3. ``raw_ft_wrist`` -- the noise-free wrist wrench (the actor sees the sensor model);
4. ``target_junction_wrench`` -- the anchor-frame spur-stem wrench the envelope reads;
5. ``detach_index``, ``success_streak_frac`` -- progress toward the success condition;
6. ``frozen``, ``invalid`` -- so the critic can learn V ~ 0 on frozen / invalid envs,
   whose samples still enter the PPO batch (the sim resets the whole batch only).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from apple_pick_gym.batched_envs.harvest_obs import (
    ObsLayout,
    ObsLayoutEntry,
    critic_obs_layout,
    flatten_critic_obs,
)
from apple_pick_gym.batched_envs.harvest_privileged import PLANT_GEOMETRY_FIELDS

CRITIC_EXTRA_FIELDS: tuple[tuple[str, int], ...] = PLANT_GEOMETRY_FIELDS + (
    ("raw_ft_wrist", 6),
    ("target_junction_wrench", 6),
    ("detach_index", 1),
    ("success_streak_frac", 1),
    ("frozen", 1),
    ("invalid", 1),
)


def critic_state_layout(junction_names: Sequence[str]) -> ObsLayout:
    base = critic_obs_layout(list(junction_names))
    entries = list(base.entries)
    start = base.total_width
    for name, width in CRITIC_EXTRA_FIELDS:
        entries.append(ObsLayoutEntry(name=name, start=start, width=width))
        start += width
    return ObsLayout(entries=tuple(entries), total_width=start)


def build_critic_state(
    obs: dict[str, Any],
    info: dict[str, Any],
    *,
    privileged: dict[str, torch.Tensor],
    geometry: dict[str, torch.Tensor],
    junction_names: Sequence[str],
    success_streak_steps: int,
    invalid: torch.Tensor,
) -> torch.Tensor:
    """``(N, critic_state_layout(junction_names).total_width)`` float32.

    ``info`` is a reset or step info; reset infos have no ``"episode"`` entry, in which case
    the streak is 0 and ``frozen`` equals ``invalid`` (invalid envs freeze at reset).
    """
    base = flatten_critic_obs(obs, privileged, info["woody_part_force"], junction_names=list(junction_names))
    n = base.shape[0]
    ep = info.get("episode")
    if ep is None:
        streak = torch.zeros(n, device=base.device)
        frozen = invalid
    else:
        streak = ep["success_streak"].to(base.dtype)
        frozen = ep["frozen"]
    extras = [geometry[name] for name, _ in PLANT_GEOMETRY_FIELDS] + [
        info["ft_wrist"],
        info["target_junction_wrench"],
        info["detach_index"].reshape(n, 1),
        (torch.clamp(streak / float(max(1, success_streak_steps)), max=1.0)).reshape(n, 1),
        frozen.to(base.dtype).reshape(n, 1),
        invalid.to(base.dtype).reshape(n, 1),
    ]
    return torch.cat([base, *[e.to(device=base.device, dtype=base.dtype) for e in extras]], dim=-1)
