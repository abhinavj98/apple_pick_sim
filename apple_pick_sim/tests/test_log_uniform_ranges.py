"""[D17p] Log-uniform range sampling and the very-wide bending fixture.

A range may carry ``"scale": "log"`` so ``sample_params`` draws it log-uniformly.
A plain linear draw over 1e5..1e11 Pa would put ~99% of worlds above 1 GPa.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from apple_pick_sim.fruiting_system import load_ranges, sample_heterogeneous_params_list
from apple_pick_sim.fruiting_system.params import _validate_ranges, parse_sim_build

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
_BASE = _FIXTURES_DIR / "fruiting_system_ranges_rl_harvest_real_g05_m1_wide_support.json"
_VVWIDE = _FIXTURES_DIR / "fruiting_system_ranges_rl_harvest_vvwide_bend.json"


def _log_ranges() -> dict:
    ranges = copy.deepcopy(load_ranges(_BASE))
    for seg in ("spur", "stem"):
        ranges[seg]["flexural_modulus_pa"] = {"min": 1e5, "max": 1e11, "scale": "log"}
    return ranges


def test_log_scale_range_draws_uniform_decades():
    params = sample_heterogeneous_params_list(_log_ranges(), topology_seed=5, num_envs=600)
    for seg in ("spur", "stem"):
        x = np.log10([getattr(p, seg).flexural_modulus_pa for p in params])
        assert x.min() >= 5.0 - 1e-9 and x.max() <= 11.0 + 1e-9
        counts = np.histogram(x, bins=6, range=(5.0, 11.0))[0]
        assert counts.min() > 60, counts  # ~100 per decade; linear sampling gives ~0 below 1e10


def test_linear_ranges_are_unchanged():
    """Ranges without ``scale`` draw exactly as before (same seed -> same values)."""
    base = load_ranges(_BASE)
    a = sample_heterogeneous_params_list(base, topology_seed=3, num_envs=4)
    tagged = copy.deepcopy(base)
    tagged["spur"]["flexural_modulus_pa"]["scale"] = "linear"
    b = sample_heterogeneous_params_list(tagged, topology_seed=3, num_envs=4)
    assert [p.spur.flexural_modulus_pa for p in a] == [p.spur.flexural_modulus_pa for p in b]


@pytest.mark.parametrize(
    "rng",
    [{"min": 0.0, "max": 1e11, "scale": "log"}, {"min": 1e5, "max": 1e11, "scale": "cubic"}],
)
def test_bad_scale_is_rejected(rng):
    ranges = _log_ranges()
    ranges["stem"]["flexural_modulus_pa"] = rng
    with pytest.raises(ValueError, match="flexural_modulus_pa"):
        _validate_ranges(ranges)


def test_vvwide_fixture():
    ranges = load_ranges(_VVWIDE)
    for seg in ("spur", "stem"):
        r = ranges[seg]["flexural_modulus_pa"]
        assert (r["min"], r["max"], r["scale"]) == (1e5, 1e11, "log")
    kp = parse_sim_build(ranges).support_dr.kp
    assert (kp.min, kp.max) == (100.0, 10000.0)
    # everything else matches the D17o fixture
    base = load_ranges(_BASE)
    for seg in ("primary", "spur", "stem", "apple"):
        for k, v in base[seg].items():
            if k != "flexural_modulus_pa":
                assert ranges[seg][k] == v, (seg, k)
    assert math.isclose(parse_sim_build(ranges).support_dr.zeta.max, parse_sim_build(base).support_dr.zeta.max)
