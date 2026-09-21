"""Flatten the observation dict + privileged extras into fixed-order ``(N, D)``
tensors for the actor (proprioception + F/T) and critic (privileged).

The actor is deliberately scoped to proprioception and F/T only --
``apple_pos``/``apple_quat``/``woody_part_start_pos``/``woody_part_end_pos``
are vision-tracked geometry from the sys-ID v3 observation contract, and the
pick policy does not use vision. Those fields are still produced by the env,
just via ``info`` rather than ``obs`` (see
``apple_pick_vic_harvest_env.py::_make_info``) -- available for logging,
reward shaping, or a future vision-augmented variant, but not flattened here.

The critic vector is always the actor vector as an exact prefix, followed by
privileged ground truth and every junction's wrench (``dict[str, (N,6)]`
keyed by junction name, flattened deterministically **by sorted junction
name** so the layout is stable across runs and topologies sharing the same
junction set). ``flatten_actor_obs`` never receives privileged inputs at
all, so privileged data cannot leak into the actor vector by construction,
not merely by convention.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

# Fixed-width actor fields, in order. Do not reorder without bumping a layout
# version -- this determines checkpoint / policy-weight compatibility.
_ACTOR_FIXED_FIELDS: tuple[tuple[str, int], ...] = (
    ("tcp_pos", 3),
    ("tcp_quat", 4),
    ("tcp_velocity", 6),
    ("ft_wrist", 6),
    ("robot_joint_q", 7),
)
_LAST_ACTION_DIM = 13

# Privileged (critic-only) fields, in order.
_PRIVILEGED_FIELDS: tuple[tuple[str, int], ...] = (
    ("spur_flexural_modulus_pa", 1),
    ("stem_flexural_modulus_pa", 1),
    ("spur_youngs_modulus_pa", 1),
    ("stem_youngs_modulus_pa", 1),
    ("spur_damping_ratio", 1),
    ("stem_damping_ratio", 1),
    ("primary_density", 1),
    ("support_kp", 1),
    ("support_roll_kp", 1),
    ("support_zeta", 1),
    ("arm_armature", 7),
    ("arm_friction", 7),
    ("arm_joint_damping", 7),
    ("arm_link_mass_scale", 1),
    ("arm_link_inertia_scale", 1),
    ("arm_ee_mass_scale", 1),
    ("arm_ee_inertia_scale", 1),
)


@dataclasses.dataclass(frozen=True)
class ObsLayoutEntry:
    name: str
    start: int
    width: int


@dataclasses.dataclass(frozen=True)
class ObsLayout:
    """Field -> ``(start, width)`` slice, recorded for checkpoint compatibility."""

    entries: tuple[ObsLayoutEntry, ...]
    total_width: int

    def slice_for(self, name: str) -> slice:
        for e in self.entries:
            if e.name == name:
                return slice(e.start, e.start + e.width)
        raise KeyError(name)


def _build_layout(field_widths: list[tuple[str, int]]) -> ObsLayout:
    entries = []
    start = 0
    for name, width in field_widths:
        entries.append(ObsLayoutEntry(name=name, start=start, width=width))
        start += width
    return ObsLayout(entries=tuple(entries), total_width=start)


def _actor_field_widths() -> list[tuple[str, int]]:
    widths = list(_ACTOR_FIXED_FIELDS)
    widths.append(("last_action", _LAST_ACTION_DIM))
    widths.append(("step_frac", 1))
    return widths


def actor_obs_layout() -> ObsLayout:
    """The actor's fixed-order layout (proprioception + F/T only; no junction geometry)."""
    return _build_layout(_actor_field_widths())


def critic_obs_layout(junction_names: list[str]) -> ObsLayout:
    """The critic's layout: the actor layout as an exact prefix, then privileged fields
    (including every junction's wrench, keyed by sorted junction name)."""
    jn = sorted(junction_names)
    widths = _actor_field_widths() + list(_PRIVILEGED_FIELDS)
    for j in jn:
        widths.append((f"woody_part_force/{j}", 6))
    return _build_layout(widths)


def flatten_actor_obs(obs: dict[str, Any]) -> torch.Tensor:
    """Flatten the proprioception + F/T subset of ``obs`` into a fixed-order ``(N, D)`` tensor.

    ``obs["ft_wrist"]`` is expected to already be the sensor-realistic value
    (see ``apple_pick_gym.batched_envs.sensor_realism.FtSensorModel``) -- this
    function only flattens, it does not apply the sensor model itself.
    """
    parts = [obs[name] for name, _ in _ACTOR_FIXED_FIELDS]
    parts.append(obs["last_action"])
    parts.append(obs["step_frac"])
    return torch.cat(parts, dim=-1)


def flatten_critic_obs(
    obs: dict[str, Any],
    privileged: dict[str, torch.Tensor],
    woody_part_force: dict[str, torch.Tensor],
    *,
    junction_names: list[str] | None = None,
) -> torch.Tensor:
    """Actor obs (exact prefix) + privileged ground truth + every junction's wrench."""
    jn = sorted(junction_names) if junction_names is not None else sorted(woody_part_force)
    actor_part = flatten_actor_obs(obs)
    priv_parts = [privileged[name] for name, _ in _PRIVILEGED_FIELDS]
    force_parts = [woody_part_force[j] for j in jn]
    return torch.cat([actor_part, *priv_parts, *force_parts], dim=-1)
