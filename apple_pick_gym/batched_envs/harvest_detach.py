"""Elliptical force + torque fruit-detachment envelope at the target (spur-stem) junction.

Fruit detaches when the junction's combined load crosses an elliptical failure
envelope (the biomechanical analogue of a von Mises / Tsai-Wu combined-loading
criterion)::

    (|F| / F_max)^2 + (|tau| / tau_max)^2 >= 1

The quadratic interaction captures the "twist-and-pull" synergy pickers use:
torque weakens the junction's resistance to pulling, and vice versa. A pull
alone needs ``F_max``; a pull at 70 % of ``F_max`` plus a twist at 75 % of
``tau_max`` already fails. Defaults are the maintainer's values
(2026-09-24): ``F_max = 20 N``, ``tau_max = 0.05 N*m``.

The index is dimensionless: ``detach_index`` is the left-hand side above
(success at ``>= 1``) and ``detach_utilization = sqrt(index)`` is the radial
fraction of the envelope used, which is linear in load and so a better-scaled
dense reward term.

**Which torque.** ``SolverVBD``'s fixed-joint readout returns the wrench on the
child body with the torque taken about the child's *COM*
(``gather_joint_wrench_child_com``). That torque is ``couple + (x_anchor -
x_com) x F``; the lever term is not load carried by the junction. With
``tau_max = 0.05 N*m`` it is not negligible either: a 2.5 mm COM offset at
``F_max`` alone reaches ``tau_max``. :func:`junction_wrench_at_anchor` removes
it, leaving the couple the junction itself carries (the AVBD angular-constraint
torque for a fixed joint).
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import torch


@dataclasses.dataclass(frozen=True)
class DetachEnvelopeConfig:
    """Elliptical detachment envelope for the target junction.

    ``torque_mode="total"`` (default): ``(|F|/f_max)^2 + (|tau|/tau_max)^2`` with the *total*
    junction moment. That moment is dominated by **bending**: the junction sits ~46 mm from the
    apple centre, so ~1.1 N of sideways force at the apple already reaches 0.05 N*m (measured on
    GPU: a random policy detaches 100 % this way).

    ``torque_mode="split"``: ``(|F|/f_max)^2 + (tau_t/torsion_max)^2 + (M_b/bending_max)^2``,
    with torsion ``tau_t`` about the stem axis and bending ``M_b`` perpendicular to it (see
    :func:`split_torque`). ``bending_max_nm = 0.9`` ~ ``f_max * 46 mm`` is a placeholder for the
    maintainer to set; the split needs the stem axis.
    """

    f_max_n: float = 20.0
    tau_max_nm: float = 0.05
    torque_mode: Literal["total", "split"] = "total"
    torsion_max_nm: float = 0.05
    bending_max_nm: float = 0.9
    # [D1] Which spur-stem wrench the envelope reads.
    # "stem_elastic" (default): the wrench of the first soft stem cable joint, shifted to the
    #   junction by statics. It has the same mean as the readout and ~65x less noise.
    # "junction_readout": the rigid fixed joint's AVBD constraint wrench, which carries a
    #   +-0.03 N*m step-to-step solver-noise floor.
    wrench_source: Literal["stem_elastic", "junction_readout"] = "stem_elastic"

    def __post_init__(self) -> None:
        for name in ("f_max_n", "tau_max_nm", "torsion_max_nm", "bending_max_nm"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.torque_mode not in ("total", "split"):
            raise ValueError(f"unknown torque_mode {self.torque_mode!r}")
        if self.wrench_source not in ("stem_elastic", "junction_readout"):
            raise ValueError(f"unknown wrench_source {self.wrench_source!r}")


def split_torque(torque: torch.Tensor, stem_axis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(N, 3)`` torque -> (``|torsion|`` about ``stem_axis``, ``|bending|`` perpendicular), each ``(N,)``."""
    axis = torch.nn.functional.normalize(stem_axis, dim=-1)
    along = (torque * axis).sum(-1, keepdim=True)
    return along.squeeze(-1).abs(), torch.linalg.norm(torque - along * axis, dim=-1)


def detach_index(
    wrench: torch.Tensor, cfg: DetachEnvelopeConfig, *, stem_axis: torch.Tensor | None = None
) -> torch.Tensor:
    """Envelope index of ``(N, 6)`` ``[F, tau]`` wrenches, shape ``(N,)`` (detach at ``>= 1``)."""
    f = torch.linalg.norm(wrench[:, :3], dim=-1) / float(cfg.f_max_n)
    if cfg.torque_mode == "total":
        t = torch.linalg.norm(wrench[:, 3:6], dim=-1) / float(cfg.tau_max_nm)
        return f * f + t * t
    if stem_axis is None:
        raise ValueError("torque_mode='split' needs stem_axis (N, 3)")
    tors, bend = split_torque(wrench[:, 3:6], stem_axis)
    tt = tors / float(cfg.torsion_max_nm)
    bb = bend / float(cfg.bending_max_nm)
    return f * f + tt * tt + bb * bb


def detach_utilization(
    wrench: torch.Tensor, cfg: DetachEnvelopeConfig, *, stem_axis: torch.Tensor | None = None
) -> torch.Tensor:
    """``sqrt(detach_index)``: radial fraction of the envelope in use (1 = on the envelope)."""
    return torch.sqrt(detach_index(wrench, cfg, stem_axis=stem_axis))


def shift_moment(
    moment: torch.Tensor, force: torch.Tensor, *, from_point: torch.Tensor, to_point: torch.Tensor
) -> torch.Tensor:
    """Moment about ``to_point`` of a wrench ``(force, moment about from_point)``: ``M + (from - to) x F``."""
    return moment + torch.cross(from_point - to_point, force, dim=-1)


def junction_wrench_at_anchor(
    wrench_at_child_com: torch.Tensor,
    *,
    child_anchor: torch.Tensor,
    child_com: torch.Tensor,
) -> torch.Tensor:
    """Shift a ``(N, 6)`` child-COM wrench to the joint anchor: ``tau -= (anchor - com) x F``.

    ``child_anchor`` / ``child_com`` are ``(N, 3)`` world positions. The force is unchanged.
    """
    force = wrench_at_child_com[:, :3]
    torque_com = wrench_at_child_com[:, 3:6]
    torque_anchor = torque_com - torch.cross(child_anchor - child_com, force, dim=-1)
    return torch.cat([force, torque_anchor], dim=-1)
