"""Parse shared CLI helpers for Young's-modulus E-grid examples."""

from __future__ import annotations

import math
from collections.abc import Sequence

from apple_pick_gym.batched_envs.batched_sysid_cmaes import (
    YoungsModulusCandidate,
    iter_youngs_modulus_candidates,
)


def parse_float_list(text: str) -> tuple[float, ...]:
    """Parse a comma-separated float list (empty → error)."""
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        raise ValueError("expected a non-empty comma-separated float list")
    return tuple(float(p) for p in parts)


def candidates_from_log10_cli(
    *,
    log10_e_primary: str,
    log10_e_spur: str,
    log10_e_stem: str,
) -> list[YoungsModulusCandidate]:
    """Cartesian product of per-segment log10(E) CLI lists."""
    primary = tuple(10.0 ** v for v in parse_float_list(log10_e_primary))
    spur = tuple(10.0 ** v for v in parse_float_list(log10_e_spur))
    stem = tuple(10.0 ** v for v in parse_float_list(log10_e_stem))
    return list(
        iter_youngs_modulus_candidates(
            primary_values=primary,
            spur_values=spur,
            stem_values=stem,
        )
    )


def candidate_log10_triples(
    candidates: Sequence[YoungsModulusCandidate],
) -> list[tuple[float, float, float]]:
    """Physical candidates → log10 triples for overlay metadata."""
    return [
        (math.log10(c.primary), math.log10(c.spur), math.log10(c.stem))
        for c in candidates
    ]
