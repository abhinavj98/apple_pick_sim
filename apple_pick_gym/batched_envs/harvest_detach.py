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

import torch


@dataclasses.dataclass(frozen=True)
class DetachEnvelopeConfig:
    """Elliptical detachment envelope for the target junction."""

    f_max_n: float = 20.0
    tau_max_nm: float = 0.05

    def __post_init__(self) -> None:
        if not self.f_max_n > 0.0:
            raise ValueError(f"f_max_n must be > 0, got {self.f_max_n}")
        if not self.tau_max_nm > 0.0:
            raise ValueError(f"tau_max_nm must be > 0, got {self.tau_max_nm}")


def detach_index(wrench: torch.Tensor, cfg: DetachEnvelopeConfig) -> torch.Tensor:
    """``(|F|/F_max)^2 + (|tau|/tau_max)^2`` for ``(N, 6)`` ``[F, tau]`` wrenches, shape ``(N,)``."""
    f = torch.linalg.norm(wrench[:, :3], dim=-1) / float(cfg.f_max_n)
    t = torch.linalg.norm(wrench[:, 3:6], dim=-1) / float(cfg.tau_max_nm)
    return f * f + t * t


def detach_utilization(wrench: torch.Tensor, cfg: DetachEnvelopeConfig) -> torch.Tensor:
    """``sqrt(detach_index)``: radial fraction of the envelope in use (1 = on the envelope)."""
    return torch.sqrt(detach_index(wrench, cfg))


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
