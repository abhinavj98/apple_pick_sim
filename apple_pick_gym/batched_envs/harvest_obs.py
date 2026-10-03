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
from typing import Any, Literal

import torch

ObsFrame = Literal["world", "tool"]

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

# [D17] Tool-frame actor fields ("tool" layout), in order: everything relative to the episode
# start and expressed in the current TCP frame, so the actor cannot memorise where the apple sits
# in the world. No joint angles and no clock (step_frac moves to the critic).
_ACTOR_TOOL_FIELDS: tuple[tuple[str, int], ...] = (
    ("d_pos_tool", 3),  # TCP displacement since the episode start
    ("d_rot_tool", 3),  # TCP rotation since the episode start (body-frame rotation vector)
    ("gravity_tool", 3),  # world -z: encodes the grasp direction relative to gravity
    ("twist_tool", 6),  # TCP linear + angular velocity
    ("ft_tool", 6),  # wrist F/T (sensor model applied in this frame)
    ("target_offset_tool", 6),  # VIC setpoint minus TCP: position + orientation error rotvec
)

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
    ("detach_scale", 1),  # [D17] bend-pull detach-force scale, per episode
    ("realized_k_scale", 6),  # [D17] applied / commanded stiffness, per episode
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


def _check_frame(frame: str) -> None:
    if frame not in ("world", "tool"):
        raise ValueError(f"obs frame must be 'world' or 'tool', got {frame!r}")


def _actor_field_widths(frame: ObsFrame = "world") -> list[tuple[str, int]]:
    _check_frame(frame)
    if frame == "tool":
        return list(_ACTOR_TOOL_FIELDS) + [("last_action", _LAST_ACTION_DIM)]
    widths = list(_ACTOR_FIXED_FIELDS)
    widths.append(("last_action", _LAST_ACTION_DIM))
    widths.append(("step_frac", 1))
    return widths


def actor_obs_layout(frame: ObsFrame = "world") -> ObsLayout:
    """The actor's fixed-order layout (proprioception + F/T only; no junction geometry).

    ``"world"``: the D1-D16 absolute world-frame fields. ``"tool"`` ([D17]): start-relative,
    TCP-frame fields (``_ACTOR_TOOL_FIELDS``).
    """
    return _build_layout(_actor_field_widths(frame))


def critic_obs_layout(junction_names: list[str], frame: ObsFrame = "world") -> ObsLayout:
    """The critic's layout: the actor layout as an exact prefix, then privileged fields
    (including every junction's wrench, keyed by sorted junction name)."""
    jn = sorted(junction_names)
    widths = _actor_field_widths(frame) + list(_PRIVILEGED_FIELDS)
    for j in jn:
        widths.append((f"woody_part_force/{j}", 6))
    return _build_layout(widths)


def flatten_actor_obs(obs: dict[str, Any], frame: ObsFrame = "world") -> torch.Tensor:
    """Flatten the proprioception + F/T subset of ``obs`` into a fixed-order ``(N, D)`` tensor.

    ``obs["ft_wrist"]`` / ``obs["ft_tool"]`` are expected to already be the sensor-realistic
    value (see ``apple_pick_gym.batched_envs.sensor_realism.FtSensorModel``) -- this
    function only flattens, it does not apply the sensor model itself.
    """
    return torch.cat([obs[name] for name, _ in _actor_field_widths(frame)], dim=-1)


def flatten_critic_obs(
    obs: dict[str, Any],
    privileged: dict[str, torch.Tensor],
    woody_part_force: dict[str, torch.Tensor],
    *,
    junction_names: list[str] | None = None,
    frame: ObsFrame = "world",
) -> torch.Tensor:
    """Actor obs (exact prefix) + privileged ground truth + every junction's wrench."""
    jn = sorted(junction_names) if junction_names is not None else sorted(woody_part_force)
    actor_part = flatten_actor_obs(obs, frame)
    priv_parts = [privileged[name] for name, _ in _PRIVILEGED_FIELDS]
    force_parts = [woody_part_force[j] for j in jn]
    return torch.cat([actor_part, *priv_parts, *force_parts], dim=-1)


# ---------------------------------------------------------------------------------------------
# [D17] tool-frame observation helpers (pure torch, xyzw quaternions as in obs["tcp_quat"])
# ---------------------------------------------------------------------------------------------


def _quat_xyzw_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)


def _quat_xyzw_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a * b`` of ``(N, 4)`` xyzw quaternions."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def _rotate_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    t = 2.0 * torch.cross(q[:, :3], v, dim=-1)
    return v + q[:, 3:4] * t + torch.cross(q[:, :3], t, dim=-1)


def quat_xyzw_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    """``(N, 4)`` xyzw quaternion -> ``(N, 3)`` rotation vector, shortest arc (``q`` and ``-q`` agree)."""
    q = torch.where(q[:, 3:4] < 0.0, -q, q)
    s = torch.linalg.norm(q[:, :3], dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s, q[:, 3:4])
    # angle / sin(angle/2) -> 2 as angle -> 0
    scale = torch.where(s > 1e-8, angle / s.clamp_min(1e-12), torch.full_like(s, 2.0))
    return q[:, :3] * scale


def tool_frame_obs(
    *,
    tcp_pos: torch.Tensor,
    tcp_quat: torch.Tensor,
    tcp_velocity: torch.Tensor,
    ft: torch.Tensor,
    start_pos: torch.Tensor,
    start_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """The ``_ACTOR_TOOL_FIELDS`` (except ``last_action``) from world-frame inputs.

    Quaternions are xyzw. ``ft`` is a world-frame ``(N, 6)`` wrench (pass the sensor-frame value
    through ``ft_tool`` directly when the sensor model already runs in the TCP frame).
    ``d_rot_tool`` is ``log(q_start^-1 q_now)``: the rotation since the start about the tool's own
    axes. ``target_offset_tool``'s angular half is the VIC orientation error ``log(q_tgt q_now^-1)``
    rotated into the TCP frame.
    """
    q_inv = _quat_xyzw_conj(tcp_quat)
    to_tool = lambda v: _rotate_xyzw(q_inv, v)
    gravity = torch.zeros_like(tcp_pos)
    gravity[:, 2] = -1.0
    rot_err_world = quat_xyzw_to_rotvec(_quat_xyzw_mul(target_quat, q_inv))
    return {
        "d_pos_tool": to_tool(tcp_pos - start_pos),
        "d_rot_tool": quat_xyzw_to_rotvec(_quat_xyzw_mul(_quat_xyzw_conj(start_quat), tcp_quat)),
        "gravity_tool": to_tool(gravity),
        "twist_tool": torch.cat([to_tool(tcp_velocity[:, :3]), to_tool(tcp_velocity[:, 3:6])], dim=-1),
        "ft_tool": torch.cat([to_tool(ft[:, :3]), to_tool(ft[:, 3:6])], dim=-1),
        "target_offset_tool": torch.cat([to_tool(target_pos - tcp_pos), to_tool(rot_err_world)], dim=-1),
    }


def rotate_tool_wrench_to_world(wrench_tool: torch.Tensor, tcp_quat: torch.Tensor) -> torch.Tensor:
    """``(N, 6)`` TCP-frame wrench -> world frame (inverse of ``tool_frame_obs``'s ``ft_tool``)."""
    return torch.cat([_rotate_xyzw(tcp_quat, wrench_tool[:, :3]), _rotate_xyzw(tcp_quat, wrench_tool[:, 3:6])], dim=-1)
