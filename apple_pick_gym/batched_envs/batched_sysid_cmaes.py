"""CMA-ES / Young's-modulus candidate helpers for batched sys-ID (V.5.2).

This module currently owns the material candidate type and log10 maps used by
the interactive E-grid demos. The full pycma loop lands in a later slice.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from itertools import product
from typing import TYPE_CHECKING, NamedTuple

from apple_pick_sim.fruiting_system import params as fs
from apple_pick_sim.fruiting_system.params import FruitingSystemParams

if TYPE_CHECKING:
    from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset


class YoungsModulusCandidate(NamedTuple):
    """One material candidate: Young's modulus (Pa) for primary, spur, stem."""

    primary: float
    spur: float
    stem: float

    def apply_to(self, base: FruitingSystemParams) -> FruitingSystemParams:
        """Return a copy with candidate ``E`` re-derived into VBD knobs.

        Only ``primary``, ``spur``, and ``stem`` are updated when present on
        ``base``. ``secondary`` (and any other fields) are left unchanged.
        Geometry and ``damping_ratio`` are frozen; axial stretch overrides on
        the base rod are preserved when they differ from beam theory.
        """
        out = base
        for segment, value in (
            ("primary", self.primary),
            ("spur", self.spur),
            ("stem", self.stem),
        ):
            if getattr(base, segment) is not None:
                out = fs.set_rod_youngs_modulus(out, segment, float(value))
        return out

    def short_label(self) -> str:
        """Compact legend label."""
        return (
            f"log10=({math.log10(self.primary):.2f},"
            f"{math.log10(self.spur):.2f},"
            f"{math.log10(self.stem):.2f})"
        )


def iter_youngs_modulus_candidates(
    *,
    primary_values: Sequence[float],
    spur_values: Sequence[float],
    stem_values: Sequence[float],
) -> Iterable[YoungsModulusCandidate]:
    """Yield Young's-modulus grid candidates in Cartesian product order."""
    for primary, spur, stem in product(primary_values, spur_values, stem_values):
        yield YoungsModulusCandidate(
            primary=float(primary),
            spur=float(spur),
            stem=float(stem),
        )


def candidates_from_log10_e(
    log10_e: Sequence[float],
) -> YoungsModulusCandidate:
    """Map ``log10([E_primary, E_spur, E_stem])`` to a physical candidate."""
    if len(log10_e) != 3:
        raise ValueError(f"log10_e must have length 3, got {len(log10_e)}")
    return YoungsModulusCandidate(
        primary=10.0 ** float(log10_e[0]),
        spur=10.0 ** float(log10_e[1]),
        stem=10.0 ** float(log10_e[2]),
    )


def log10_e_from_params(params: FruitingSystemParams) -> tuple[float, float, float]:
    """Extract ``log10(E)`` for primary, spur, stem (hard error if missing)."""
    if params.primary is None or params.spur is None or params.stem is None:
        raise ValueError(
            "params must include primary, spur, and stem rods for log10_e_from_params"
        )
    return (
        math.log10(float(params.primary.youngs_modulus_pa)),
        math.log10(float(params.spur.youngs_modulus_pa)),
        math.log10(float(params.stem.youngs_modulus_pa)),
    )


def youngs_modulus_candidate_from_params(
    params: FruitingSystemParams,
) -> YoungsModulusCandidate:
    """Build a candidate from absolute ``E`` on primary/spur/stem."""
    if params.primary is None or params.spur is None or params.stem is None:
        raise ValueError(
            "params must include primary, spur, and stem rods"
        )
    return YoungsModulusCandidate(
        primary=float(params.primary.youngs_modulus_pa),
        spur=float(params.spur.youngs_modulus_pa),
        stem=float(params.stem.youngs_modulus_pa),
    )


def gt_youngs_modulus_candidate_from_structure(
    dataset: BatchedSysIdDataset,
    structure_idx: int,
) -> YoungsModulusCandidate:
    import apple_pick_gym.batched_envs.batched_sysid_cmaes as _mod

    return youngs_modulus_candidate_from_params(
        _mod.true_params_for_structure(dataset, int(structure_idx))
    )


def youngs_modulus_values_match(
    left: YoungsModulusCandidate,
    right: YoungsModulusCandidate,
    *,
    log10_atol: float = 1e-9,
) -> bool:
    return all(
        math.isclose(math.log10(a), math.log10(b), rel_tol=0.0, abs_tol=log10_atol)
        for a, b in zip(left, right, strict=True)
    )


def maybe_include_gt_candidate(
    candidates: Sequence[YoungsModulusCandidate],
    gt: YoungsModulusCandidate,
    *,
    include_gt: bool,
) -> list[YoungsModulusCandidate]:
    items = list(candidates)
    if not include_gt or any(youngs_modulus_values_match(item, gt) for item in items):
        return items
    return [*items, gt]


from apple_pick_sim.system_id.batched_digital_twin_init import (  # noqa: E402
    true_params_for_structure,
)
