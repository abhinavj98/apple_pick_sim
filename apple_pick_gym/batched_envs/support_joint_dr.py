"""Per-env support-joint domain randomization: k_p, roll k_p, and zeta.

Fills CMA search-box knobs 8-10 (support k_p, support roll k_p, support
joint zeta) that the shared ranges fixture does not vary -- they live as
scalars in a fixture's ``sim_build`` block, not as ranges. Sampling reads a
``SupportJointDRRanges`` (``sim_build.support_dr``, an additive schema
extension -- see ``apple_pick_sim.fruiting_system.params``). Application
reuses the existing ``apply_per_env_support_joint_penalties`` /
``apply_per_env_support_roll_penalties`` helpers built for CMA in
``apple_pick_gym.batched_envs.support_joint_penalties``; this module does not
reimplement them, only samples per-env values and dispatches to them.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from apple_pick_sim.fruiting_system.joint_kd_scaling import support_dowel_length_m
from apple_pick_sim.fruiting_system.params import RangeF, SupportJointDRRanges
from apple_pick_gym.batched_envs.support_joint_penalties import (
    apply_per_env_support_joint_penalties,
    apply_per_env_support_roll_penalties,
)


@dataclasses.dataclass(frozen=True)
class SupportJointDRSample:
    """Per-env sampled support-joint values."""

    kp: np.ndarray  # (N,) linear support k_p [N/m]
    roll_kp: np.ndarray  # (N,) revolute T-roll k_p [N*m/rad]
    zeta: np.ndarray  # (N,) support joint damping ratio


def _log_uniform(bounds: RangeF, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample uniformly in log10-space over ``[bounds.min, bounds.max]``.

    ``kp``/``roll_kp`` span multiple decades and are searched on a log10 scale
    by CMA (``candidates_from_log10_vector``); a linear-uniform draw over a
    wide range like ``[100, 1e6]`` would waste most samples in the top decade,
    unlike what CMA actually explored.
    """
    lo, hi = float(bounds.min), float(bounds.max)
    if lo <= 0.0:
        raise ValueError(f"log-uniform range requires min > 0, got {lo!r}")
    log_lo, log_hi = math.log10(lo), math.log10(hi)
    return (10.0 ** rng.uniform(log_lo, log_hi, size=size)).astype(np.float32)


def sample_support_joint_dr(
    ranges: SupportJointDRRanges, *, num_envs: int, rng: np.random.Generator
) -> SupportJointDRSample:
    """Draw one sample per env for ``kp``, ``roll_kp`` (log-uniform, matching
    their CMA log10 encoding) and ``zeta`` (linear-uniform, matching CMA's own
    linear ``[0, 1]``-style dimension for it).
    """
    n = int(num_envs)
    return SupportJointDRSample(
        kp=_log_uniform(ranges.kp, n, rng),
        roll_kp=_log_uniform(ranges.roll_kp, n, rng),
        zeta=rng.uniform(ranges.zeta.min, ranges.zeta.max, size=n).astype(np.float32),
    )


def apply_support_joint_dr(
    scene: Any,
    sample: SupportJointDRSample,
    *,
    num_envs: int,
    joints_per_world: int,
    per_env_params: Sequence[Any],
) -> None:
    """Write per-env support ``kp``/``roll_kp``/``zeta`` into the built scene.

    Dispatches to the existing CMA-built applicators
    (``apply_per_env_support_joint_penalties`` for ``kp``/``zeta``,
    ``apply_per_env_support_roll_penalties`` for ``roll_kp``/``zeta``); see
    ``apple_pick_sim/examples/stress_plant_rebuild_loop.py`` for the reference
    call pattern this mirrors.
    """
    dowel_lengths = [support_dowel_length_m(p) for p in per_env_params]
    apply_per_env_support_joint_penalties(
        scene,
        sample.kp.tolist(),
        num_envs=num_envs,
        joints_per_world=joints_per_world,
        dowel_length_m_per_env=dowel_lengths,
        zeta_per_env=sample.zeta.tolist(),
    )
    apply_per_env_support_roll_penalties(
        scene,
        sample.roll_kp.tolist(),
        num_envs=num_envs,
        joints_per_world=joints_per_world,
        zeta_per_env=sample.zeta.tolist(),
    )
