"""RL harvest DR fixture: extends plant DR to all ten CMA search-box knobs.

Verifies the new fixture (fruiting_system_ranges_rl_harvest_variance.json)
covers every knob the CMA search varies, without rebuilding
sample_heterogeneous_params_list's already-correct per-env sampling -- see
docs/superpowers/plans/2026-09-17-rl-vic-harvest-policy.md Task 3.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from apple_pick_sim.fruiting_system import load_ranges, sample_heterogeneous_params_list
from apple_pick_sim.fruiting_system.params import parse_sim_build

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
_RL_HARVEST_FIXTURE = _FIXTURES_DIR / "fruiting_system_ranges_rl_harvest_variance.json"
_SHARED_FIXTURE = _FIXTURES_DIR / "fruiting_system_ranges_real_world_proxy_variance.json"

_NUM_ENVS = 30


def test_fixture_loads_and_parses():
    ranges = load_ranges(_RL_HARVEST_FIXTURE)
    sim_build = parse_sim_build(ranges)
    assert sim_build is not None
    assert sim_build.support_dr is not None


def test_support_dr_ranges_match_declared_values():
    ranges = load_ranges(_RL_HARVEST_FIXTURE)
    sim_build = parse_sim_build(ranges)
    assert sim_build.support_dr.kp.min == pytest.approx(100.0)
    assert sim_build.support_dr.kp.max == pytest.approx(1_000_000.0)
    assert sim_build.support_dr.roll_kp.min == pytest.approx(0.075)
    assert sim_build.support_dr.roll_kp.max == pytest.approx(7.5)
    assert sim_build.support_dr.zeta.min == pytest.approx(0.05)
    assert sim_build.support_dr.zeta.max == pytest.approx(0.95)


def test_shared_fixture_is_byte_unaffected():
    """The new fixture must not touch the shared fixture other consumers rely on."""
    shared = load_ranges(_SHARED_FIXTURE)
    assert parse_sim_build(shared).support_dr is None
    assert shared["spur"]["damping_ratio"] == {"min": 0.3, "max": 0.3}
    assert shared["stem"]["damping_ratio"] == {"min": 0.3, "max": 0.3}


def test_spur_stem_damping_ratio_is_non_degenerate_across_envs():
    """Knobs 6-7: spur/stem damping_ratio must vary, not be pinned to a constant."""
    ranges = load_ranges(_RL_HARVEST_FIXTURE)
    params = sample_heterogeneous_params_list(ranges, topology_seed=11, num_envs=_NUM_ENVS)
    spur_zetas = {round(p.spur.damping_ratio, 6) for p in params}
    stem_zetas = {round(p.stem.damping_ratio, 6) for p in params}
    assert len(spur_zetas) > 1, "spur damping_ratio must not be degenerate across envs"
    assert len(stem_zetas) > 1, "stem damping_ratio must not be degenerate across envs"
    for p in params:
        assert 0.05 - 1e-6 <= p.spur.damping_ratio <= 0.95 + 1e-6
        assert 0.05 - 1e-6 <= p.stem.damping_ratio <= 0.95 + 1e-6


def test_flexural_and_axial_moduli_vary_across_envs():
    """Knobs 1-4: spur/stem flexural and axial modulus already vary (no regression)."""
    ranges = load_ranges(_RL_HARVEST_FIXTURE)
    params = sample_heterogeneous_params_list(ranges, topology_seed=11, num_envs=_NUM_ENVS)
    assert len({round(p.spur.flexural_modulus_pa, -3) for p in params}) > 1
    assert len({round(p.stem.flexural_modulus_pa, -3) for p in params}) > 1
    assert len({round(p.spur.youngs_modulus_pa, -3) for p in params}) > 1
    assert len({round(p.stem.youngs_modulus_pa, -3) for p in params}) > 1


def test_primary_density_varies_across_envs():
    """Knob 5: primary rod density already varies (no regression)."""
    ranges = load_ranges(_RL_HARVEST_FIXTURE)
    params = sample_heterogeneous_params_list(ranges, topology_seed=11, num_envs=_NUM_ENVS)
    assert len({round(p.primary.density, 1) for p in params}) > 1


def test_sampling_is_reproducible_for_fixed_seed():
    ranges = load_ranges(_RL_HARVEST_FIXTURE)
    a = sample_heterogeneous_params_list(ranges, topology_seed=5, num_envs=8)
    b = sample_heterogeneous_params_list(ranges, topology_seed=5, num_envs=8)
    for pa, pb in zip(a, b, strict=True):
        assert pa.spur.damping_ratio == pytest.approx(pb.spur.damping_ratio)
        assert pa.stem.damping_ratio == pytest.approx(pb.stem.damping_ratio)
        assert pa.primary.density == pytest.approx(pb.primary.density)


def test_segment_topology_stays_identical_across_envs():
    ranges = load_ranges(_RL_HARVEST_FIXTURE)
    params = sample_heterogeneous_params_list(ranges, topology_seed=11, num_envs=_NUM_ENVS)
    topo0 = params[0]
    for p in params[1:]:
        assert p.primary.num_segments == topo0.primary.num_segments
        assert p.spur.num_segments == topo0.spur.num_segments
        assert p.stem.num_segments == topo0.stem.num_segments
        assert (p.apple_radius is None) == (topo0.apple_radius is None)


def test_apple_density_within_declared_fixture_range():
    """Correctness check: apple density stays within THIS fixture's own declared
    range. (Not asserted against ~800 kg/m^3: that plausibility band is from a
    different pipeline -- real-data chord-closure back-solving in
    apple_pick_sim/system_id/real_pre_grasp_params.py -- and does not apply to
    this synthetic uniform-range sampler. Note for the record: this fixture's
    own declared apple density range, [400, 600], sits below that plausibility
    band; left as-is since apple geometry/density is not one of the ten CMA
    search-box knobs this task extends, and changing it is out of scope.)
    """
    ranges = load_ranges(_RL_HARVEST_FIXTURE)
    params = sample_heterogeneous_params_list(ranges, topology_seed=11, num_envs=_NUM_ENVS)
    lo, hi = ranges["apple"]["density"]["min"], ranges["apple"]["density"]["max"]
    for p in params:
        assert lo - 1e-6 <= p.apple_density <= hi + 1e-6
