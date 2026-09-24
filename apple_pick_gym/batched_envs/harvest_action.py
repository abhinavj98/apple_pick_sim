"""13-D delta-pose harvest action: bounded pose delta + per-axis Kp + zeta -> 19-D vic_pose.

Layout: ``[dp(3), drot(3), Klin(3), Kang(3), zeta(1)]``. The policy commands a
bounded *delta* around a per-env target pose the caller maintains across
steps, rather than an absolute world pose, for better-conditioned
exploration. Damping is derived (not a free action dim) via
:func:`derive_critical_damping`, so the policy cannot pick a K/D pair that
destabilizes the impedance controller.

The env-level target pose is represented as ``(N, 7)`` ``[pos(3),
quat_wxyz(4)]``, matching the external ``vic_pose`` contract's
``action[0:7]`` directly (H2 sec 3, ``docs/handbook-variable-impedance.md``):
``pack_vic_pose_action`` does no quaternion reordering -- xyzw reordering for
Warp happens downstream in the controller's ``unpack_pose_action``
(``apple_pick_sim/robot/fr3_robot/controllers/ee_impedance_batched.py``).
"""

from __future__ import annotations

import dataclasses

import torch

from apple_pick_sim.robot.fr3_robot.controllers.batched_action_twists import clip_action_tensor

_ACTION_DIM = 13
_VIC_POSE_ACTION_DIM = 19


def derive_critical_damping(
    stiffness: torch.Tensor, zeta: torch.Tensor | float
) -> torch.Tensor:
    """Return ``D = 2*zeta*sqrt(K)`` element-wise; ``K`` is clamped to ``>= 0`` first.

    ``zeta`` may be a scalar float (broadcast to every element) or a per-env
    tensor shaped ``(N, 1)`` (broadcasts across ``stiffness``'s trailing axis).
    """
    if isinstance(zeta, torch.Tensor):
        zeta_b = zeta.reshape(zeta.shape[0], *([1] * (stiffness.dim() - 1)))
    else:
        zeta_b = float(zeta)
    return 2.0 * zeta_b * torch.sqrt(torch.clamp(stiffness, min=0.0))


@dataclasses.dataclass(frozen=True)
class HarvestActionBounds:
    """Bounds for the 13-D delta-pose harvest action.

    ``linear_delta_m``/``angular_delta_rad`` bound the PER-STEP pose delta
    directly (there is no further ``dt`` scaling in :func:`integrate_delta_pose`).
    Defaults are a starting design choice pending RL tuning, not derived from
    data; they are picked in the same order of magnitude as the twist-mode
    ``vic`` controller's default per-step displacement at 60 Hz control
    (``linear_speed=1.0`` m/s / 60 Hz ~= 0.0167 m/step).
    """

    linear_delta_m: float = 0.02
    angular_delta_rad: float = 0.1
    k_lin_min: float = 20.0
    k_lin_max: float = 200.0
    k_ang_min: float = 2.0
    k_ang_max: float = 40.0
    zeta_min: float = 0.3
    zeta_max: float = 2.0


@dataclasses.dataclass(frozen=True)
class SplitHarvestAction:
    """Bounded pose delta + per-axis Kp + zeta decoded from a raw 13-D action."""

    delta: torch.Tensor  # (N, 6): [dp(3), drot(3)], norm-clamped per bounds
    linear_k: torch.Tensor  # (N, 3)
    angular_k: torch.Tensor  # (N, 3)
    zeta: torch.Tensor  # (N, 1)


def split_harvest_action(
    actions: torch.Tensor, bounds: HarvestActionBounds
) -> SplitHarvestAction:
    """Split a raw ``(N, 13)`` action into a bounded pose delta, per-axis Kp, and zeta.

    Layout: ``[dp(3), drot(3), Klin(3), Kang(3), zeta(1)]``.
    """
    if actions.shape[-1] != _ACTION_DIM:
        raise ValueError(f"expected action last dim {_ACTION_DIM}, got {actions.shape[-1]}")

    # Reuse the existing (N,6) norm-clip rather than reimplementing it -- see
    # apple_pick_sim/robot/fr3_robot/controllers/batched_action_twists.py::clip_action_tensor.
    delta = clip_action_tensor(
        actions[:, :6],
        linear_speed=bounds.linear_delta_m,
        angular_speed=bounds.angular_delta_rad,
    )
    linear_k = torch.clamp(actions[:, 6:9], min=bounds.k_lin_min, max=bounds.k_lin_max)
    angular_k = torch.clamp(actions[:, 9:12], min=bounds.k_ang_min, max=bounds.k_ang_max)
    zeta = torch.clamp(actions[:, 12:13], min=bounds.zeta_min, max=bounds.zeta_max)
    return SplitHarvestAction(delta=delta, linear_k=linear_k, angular_k=angular_k, zeta=zeta)


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a * b``, both ``(N, 4)`` in ``[w, x, y, z]``."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _axis_angle_to_quat_wxyz(rotvec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """``(N, 3)`` axis-angle rotation vector -> ``(N, 4)`` quaternion ``[w, x, y, z]``.

    A zero rotation vector maps exactly to the identity quaternion.
    """
    angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    small = angle < eps
    axis = torch.where(small, torch.zeros_like(rotvec), rotvec / angle.clamp_min(eps))
    half = angle * 0.5
    quat = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)
    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    return torch.where(small, identity, quat)


def integrate_delta_pose(target: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Integrate a bounded ``(dp, drot)`` delta onto a ``(N, 7)`` ``[pos, quat_wxyz]`` target.

    World-frame incremental rotation: ``quat_new = normalize(delta_q (x) quat)``
    (left-multiply by the incremental rotation), matching
    ``apple_pick_sim.robot.fr3_robot.controllers.keyboard.integrate_tcp_target``'s
    convention, adapted to batched torch and the ``wxyz`` quaternion layout.
    ``delta`` is expected already-bounded (see :func:`split_harvest_action`);
    this function applies no clamping of its own.
    """
    pos, quat = target[:, :3], target[:, 3:7]
    dp, drot = delta[:, :3], delta[:, 3:6]
    pos_new = pos + dp
    delta_q = _axis_angle_to_quat_wxyz(drot)
    quat_new = _quat_mul_wxyz(delta_q, quat)
    quat_new = quat_new / torch.linalg.norm(quat_new, dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.cat([pos_new, quat_new], dim=-1)


def _quat_conj_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[:, :1], -q[:, 1:]], dim=-1)


def leash_target_pose(
    target: torch.Tensor,
    tcp: torch.Tensor,
    *,
    max_pos_offset_m: float | None,
    max_rot_offset_rad: float | None,
) -> torch.Tensor:
    """Keep the integrated ``(N, 7)`` ``[pos, quat_wxyz]`` target within a leash of the TCP.

    The VIC wrench is ``K * (target - tcp)``, so an unbounded integrated target lets a
    random-walk policy wind up forces far past any safety cap (``+-0.02 m/step`` for
    500 steps is ~0.45 m, 90 N at ``K = 200 N/m``). The leash bounds the commanded
    wrench to ``K_max * max_pos_offset_m`` / ``K_ang_max * max_rot_offset_rad``, the
    same way a real-rig impedance controller saturates its setpoint error. Position
    is projected onto the sphere of radius ``max_pos_offset_m`` around the TCP;
    rotation is shortened along the same relative axis to ``max_rot_offset_rad``.
    ``None`` disables that half.
    """
    out = target.clone()
    if max_pos_offset_m is not None:
        offset = target[:, :3] - tcp[:, :3]
        dist = torch.linalg.norm(offset, dim=-1, keepdim=True)
        scale = torch.clamp(float(max_pos_offset_m) / dist.clamp_min(1e-12), max=1.0)
        out[:, :3] = tcp[:, :3] + offset * scale
    if max_rot_offset_rad is not None:
        q_tcp = tcp[:, 3:7]
        rel = _quat_mul_wxyz(target[:, 3:7], _quat_conj_wxyz(q_tcp))  # world-frame: target = rel * tcp
        rel = torch.where(rel[:, :1] < 0.0, -rel, rel)  # shortest arc (double cover)
        sin_half = torch.linalg.norm(rel[:, 1:], dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(sin_half, rel[:, :1])
        axis = rel[:, 1:] / sin_half.clamp_min(1e-12)
        clamped = torch.clamp(angle, max=float(max_rot_offset_rad))
        rel_new = _axis_angle_to_quat_wxyz(axis * clamped)
        q_new = _quat_mul_wxyz(rel_new, q_tcp)
        q_new = q_new / torch.linalg.norm(q_new, dim=-1, keepdim=True).clamp_min(1e-9)
        over = angle > float(max_rot_offset_rad)
        out[:, 3:7] = torch.where(over, q_new, target[:, 3:7])
    return out


def pack_vic_pose_action(
    target: torch.Tensor,
    linear_k: torch.Tensor,
    angular_k: torch.Tensor,
    zeta: torch.Tensor,
) -> torch.Tensor:
    """Pack ``[target(7), Kp(6), Kd(6)]`` into the 19-D ``vic_pose`` action layout.

    ``target`` is ``(N, 7)`` ``[pos(3), quat_wxyz(4)]`` and is copied through
    unchanged into ``action[0:7]`` -- no reordering here (see module docstring).
    ``Kd`` is derived from a single ``zeta`` shared across both linear and
    angular stiffness, per the harvest action design (13-D layout has one
    ``zeta`` for all six ``Kp`` axes, not a per-axis damping ratio).
    """
    kp = torch.cat([linear_k, angular_k], dim=-1)
    kd = derive_critical_damping(kp, zeta)
    out = torch.cat([target, kp, kd], dim=-1)
    assert out.shape[-1] == _VIC_POSE_ACTION_DIM
    return out
