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
import math
from typing import Literal

import numpy as np
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
    # [D7] F_max / tau_max are rough estimates. With a range set, every env draws its own limit
    # at each reset (force uniform, torque log-uniform; see :func:`envelope_thresholds`), so the
    # policy learns to keep loading until the apple comes off instead of fitting one guessed
    # threshold. ``None`` keeps the fixed ``f_max_n`` / ``tau_max_nm``. The actor never sees
    # the draw; the privileged critic does. Total-moment envelope only.
    f_max_range_n: tuple[float, float] | None = None
    tau_max_range_nm: tuple[float, float] | None = None
    # [D17] Which criterion the env evaluates. "envelope": the elliptical force + torque envelope
    # above. "bend_pull": Li et al. 2016 (Comput. Electron. Agric. 127:633, Eq 3/4): detach when
    # the stem tension reaches F_d(theta_b) = s * (a theta^2 + b theta + c) N, theta_b the stem
    # chord's bending angle (deg) from its rest direction, flat past the quadratic's minimum. The
    # coefficients average the paper's against- and along-orientation fits; s is drawn per env at
    # each reset from detach_scale_range (None: s = 1).
    criterion: Literal["envelope", "bend_pull"] = "envelope"
    bend_pull_coeffs: tuple[float, float, float] = (0.015, -1.155, 26.09)
    detach_scale_range: tuple[float, float] | None = (0.85, 1.15)

    def __post_init__(self) -> None:
        if self.criterion not in ("envelope", "bend_pull"):
            raise ValueError(f"unknown criterion {self.criterion!r}")
        a, b, _c = (float(v) for v in self.bend_pull_coeffs)
        if not (a > 0.0 and b < 0.0):
            raise ValueError(
                f"bend_pull_coeffs must be a convex quadratic with its minimum above 0 deg "
                f"(a > 0, b < 0), got {self.bend_pull_coeffs}"
            )
        rng = self.detach_scale_range
        if rng is not None and (len(rng) != 2 or not 0.0 < float(rng[0]) <= float(rng[1])):
            raise ValueError(f"detach_scale_range must be (lo, hi) with 0 < lo <= hi, got {rng}")
        for name in ("f_max_n", "tau_max_nm", "torsion_max_nm", "bending_max_nm"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.torque_mode not in ("total", "split"):
            raise ValueError(f"unknown torque_mode {self.torque_mode!r}")
        if self.wrench_source not in ("stem_elastic", "junction_readout"):
            raise ValueError(f"unknown wrench_source {self.wrench_source!r}")
        for name in ("f_max_range_n", "tau_max_range_nm"):
            rng = getattr(self, name)
            if rng is None:
                continue
            if len(rng) != 2 or not 0.0 < float(rng[0]) <= float(rng[1]):
                raise ValueError(f"{name} must be (lo, hi) with 0 < lo <= hi, got {rng}")
            if self.torque_mode != "total":
                raise ValueError(f"{name} is only supported with torque_mode='total'")


def envelope_thresholds(
    cfg: DetachEnvelopeConfig, n: int, rng: np.random.Generator, *, device: torch.device | str
) -> torch.Tensor:
    """[D7] Per-env ``(n, 2)`` ``[f_max, tau_max]`` for one episode.

    Without a range the nominal value is used (and ``rng`` is not consumed for it). ``f_max``
    is uniform over ``f_max_range_n``; ``tau_max`` is log-uniform over ``tau_max_range_nm``
    (its guess is uncertain by a factor, not by an offset).
    """
    if cfg.f_max_range_n is None:
        f = np.full(n, float(cfg.f_max_n))
    else:
        f = rng.uniform(float(cfg.f_max_range_n[0]), float(cfg.f_max_range_n[1]), size=n)
    if cfg.tau_max_range_nm is None:
        t = np.full(n, float(cfg.tau_max_nm))
    else:
        lo, hi = (math.log(float(v)) for v in cfg.tau_max_range_nm)
        t = np.exp(rng.uniform(lo, hi, size=n))
    return torch.as_tensor(np.stack([f, t], axis=-1), dtype=torch.float32, device=device)


def split_torque(torque: torch.Tensor, stem_axis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(N, 3)`` torque -> (``|torsion|`` about ``stem_axis``, ``|bending|`` perpendicular), each ``(N,)``."""
    axis = torch.nn.functional.normalize(stem_axis, dim=-1)
    along = (torque * axis).sum(-1, keepdim=True)
    return along.squeeze(-1).abs(), torch.linalg.norm(torque - along * axis, dim=-1)


def detach_index(
    wrench: torch.Tensor,
    cfg: DetachEnvelopeConfig,
    *,
    stem_axis: torch.Tensor | None = None,
    thresholds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Envelope index of ``(N, 6)`` ``[F, tau]`` wrenches, shape ``(N,)`` (detach at ``>= 1``).

    ``thresholds`` (``(N, 2)`` ``[f_max, tau_max]``, [D7]) overrides the config's limits per env;
    in ``split`` mode only its force column is used.
    """
    f_max = float(cfg.f_max_n) if thresholds is None else thresholds[:, 0].to(wrench)
    f = torch.linalg.norm(wrench[:, :3], dim=-1) / f_max
    if cfg.torque_mode == "total":
        tau_max = float(cfg.tau_max_nm) if thresholds is None else thresholds[:, 1].to(wrench)
        t = torch.linalg.norm(wrench[:, 3:6], dim=-1) / tau_max
        return f * f + t * t
    if stem_axis is None:
        raise ValueError("torque_mode='split' needs stem_axis (N, 3)")
    tors, bend = split_torque(wrench[:, 3:6], stem_axis)
    tt = tors / float(cfg.torsion_max_nm)
    bb = bend / float(cfg.bending_max_nm)
    return f * f + tt * tt + bb * bb


def detach_utilization(
    wrench: torch.Tensor,
    cfg: DetachEnvelopeConfig,
    *,
    stem_axis: torch.Tensor | None = None,
    thresholds: torch.Tensor | None = None,
) -> torch.Tensor:
    """``sqrt(detach_index)``: radial fraction of the envelope in use (1 = on the envelope)."""
    return torch.sqrt(detach_index(wrench, cfg, stem_axis=stem_axis, thresholds=thresholds))


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


# ---------------------------------------------------------------------------------------------
# [D17] Bend-and-pull criterion (``DetachEnvelopeConfig.criterion == "bend_pull"``)
# ---------------------------------------------------------------------------------------------


def bend_pull_min_angle_deg(cfg: DetachEnvelopeConfig) -> float:
    """Bending angle (deg) where the detach-force quadratic bottoms out: ``-b / 2a``."""
    a, b, _c = (float(v) for v in cfg.bend_pull_coeffs)
    return -b / (2.0 * a)


def bend_pull_detach_force(theta_deg: torch.Tensor, scale: torch.Tensor, cfg: DetachEnvelopeConfig) -> torch.Tensor:
    """Stem tension (N) that detaches the fruit at bending angle ``theta_deg``, shape ``(N,)``.

    ``scale * (a theta^2 + b theta + c)`` with ``theta`` clamped to ``[0, theta_min]``: the
    curve is held flat past its minimum. ``scale`` is the per-env draw (:func:`detach_scales`).
    """
    a, b, c = (float(v) for v in cfg.bend_pull_coeffs)
    th = torch.clamp(theta_deg, min=0.0, max=bend_pull_min_angle_deg(cfg))
    return scale.to(th) * (a * th * th + b * th + c)


def chord_angle_deg(chord: torch.Tensor, rest_chord: torch.Tensor) -> torch.Tensor:
    """Angle (deg) between ``(N, 3)`` stem chords (any length) now and at rest, shape ``(N,)``.

    ``atan2(|a x b|, a . b)`` stays accurate near 0 deg, where ``acos`` of a dot product does not.
    """
    cross = torch.linalg.norm(torch.cross(chord, rest_chord.to(chord), dim=-1), dim=-1)
    dot = (chord * rest_chord.to(chord)).sum(-1)
    return torch.rad2deg(torch.atan2(cross, dot))


def stem_tension(force_on_child: torch.Tensor, chord: torch.Tensor) -> torch.Tensor:
    """Axial stem tension (N, positive = pulled) from the stem-root joint force, shape ``(N,)``.

    ``force_on_child`` is the joint's force on its child segment (the apple side); ``chord`` points
    junction -> apple. A pulled or hanging apple drags the child away from the junction and the
    joint holds it back, so tension is the force component along ``-chord``.
    """
    axis = torch.nn.functional.normalize(chord.to(force_on_child), dim=-1)
    return -(force_on_child * axis).sum(-1)


def bend_pull_utilization(
    tension: torch.Tensor, theta_deg: torch.Tensor, scale: torch.Tensor, cfg: DetachEnvelopeConfig
) -> torch.Tensor:
    """``tension / F_d(theta)``, shape ``(N,)``: detach at ``>= 1``. Not clipped (compression < 0)."""
    return tension / bend_pull_detach_force(theta_deg, scale, cfg)


def detach_scales(
    cfg: DetachEnvelopeConfig, n: int, rng: np.random.Generator, *, device: torch.device | str
) -> torch.Tensor:
    """Per-env ``(n,)`` scale on the bend-pull detach force for one episode (uniform; ones if off)."""
    if cfg.detach_scale_range is None:
        s = np.ones(n)
    else:
        s = rng.uniform(float(cfg.detach_scale_range[0]), float(cfg.detach_scale_range[1]), size=n)
    return torch.as_tensor(s, dtype=torch.float32, device=device)
