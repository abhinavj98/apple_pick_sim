"""Unit tests for YoungsModulusCandidate and log10 helpers."""

from __future__ import annotations

import math

import pytest

from apple_pick_sim.fruiting_system import params as fs
from apple_pick_sim.tests.conftest import RANGES_FIXTURE


def _base_primary_spur_stem(seed: int = 0) -> fs.FruitingSystemParams:
    ranges = fs.load_ranges(RANGES_FIXTURE)
    return fs.sample_params(ranges, seed=seed, omit=("secondary",))


def _base_primary_spur_stem_with_secondary(seed: int = 0) -> fs.FruitingSystemParams:
    ranges = fs.load_ranges(RANGES_FIXTURE)
    params = fs.sample_params(ranges, seed=seed)
    assert params.primary is not None and params.spur is not None and params.stem is not None
    assert params.secondary is not None
    return params


def test_iter_youngs_modulus_candidates_cartesian_product():
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    candidates = list(
        cmaes.iter_youngs_modulus_candidates(
            primary_values=(1.0e8, 2.0e8),
            spur_values=(3.0e7,),
            stem_values=(1.0e7, 2.0e7),
        )
    )
    assert candidates == [
        cmaes.YoungsModulusCandidate(primary=1.0e8, spur=3.0e7, stem=1.0e7),
        cmaes.YoungsModulusCandidate(primary=1.0e8, spur=3.0e7, stem=2.0e7),
        cmaes.YoungsModulusCandidate(primary=2.0e8, spur=3.0e7, stem=1.0e7),
        cmaes.YoungsModulusCandidate(primary=2.0e8, spur=3.0e7, stem=2.0e7),
    ]


def test_candidates_from_log10_e_and_round_trip():
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    log10_e = (8.0, 7.5, 7.0)
    cand = cmaes.candidates_from_log10_e(log10_e)
    assert cand.primary == pytest.approx(1.0e8)
    assert cand.spur == pytest.approx(10**7.5)
    assert cand.stem == pytest.approx(1.0e7)

    base = _base_primary_spur_stem()
    applied = cand.apply_to(base)
    got = cmaes.log10_e_from_params(applied)
    assert got == pytest.approx(log10_e)


def test_apply_to_sets_e_rederives_bend_freezes_geometry_and_zeta():
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    base = _base_primary_spur_stem_with_secondary()
    assert base.secondary is not None
    assert base.primary is not None and base.spur is not None and base.stem is not None
    base_secondary = base.secondary

    e_p, e_sp, e_st = 5.0e8, 4.0e7, 2.0e7
    out = cmaes.YoungsModulusCandidate(primary=e_p, spur=e_sp, stem=e_st).apply_to(base)

    assert out.secondary is not None
    assert out.primary is not None and out.spur is not None and out.stem is not None
    assert out.secondary.youngs_modulus_pa == pytest.approx(base_secondary.youngs_modulus_pa)
    assert out.secondary.damping_ratio == pytest.approx(base_secondary.damping_ratio)
    assert out.secondary.length == pytest.approx(base_secondary.length)
    assert out.secondary.radius == pytest.approx(base_secondary.radius)
    assert out.secondary.density == pytest.approx(base_secondary.density)
    assert out.secondary.num_segments == base_secondary.num_segments
    assert out.secondary.direction == base_secondary.direction
    assert out.secondary.bend_stiffness == pytest.approx(base_secondary.bend_stiffness)
    assert out.secondary.bend_damping == pytest.approx(base_secondary.bend_damping)

    for rod, e_new, base_rod in (
        (out.primary, e_p, base.primary),
        (out.spur, e_sp, base.spur),
        (out.stem, e_st, base.stem),
    ):
        assert rod.youngs_modulus_pa == pytest.approx(e_new)
        assert rod.damping_ratio == pytest.approx(base_rod.damping_ratio)
        assert rod.length == pytest.approx(base_rod.length)
        assert rod.radius == pytest.approx(base_rod.radius)
        assert rod.density == pytest.approx(base_rod.density)
        assert rod.num_segments == base_rod.num_segments
        assert rod.direction == base_rod.direction
        expected = fs.rod_params_from_material(
            e_new,
            base_rod.damping_ratio,
            base_rod.length,
            base_rod.radius,
            base_rod.density,
            base_rod.num_segments,
            base_rod.direction,
        )
        assert rod.bend_stiffness == pytest.approx(expected.bend_stiffness)
        assert rod.bend_damping == pytest.approx(expected.bend_damping)


def test_apply_to_preserves_fixed_axial_stretch_overrides():
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    fixed_k, fixed_c = 5.0e5, 30.0
    primary = fs.rod_params_from_material(
        youngs_modulus_pa=1.0e7,
        damping_ratio=0.05,
        length=0.20,
        radius=0.01,
        density=300.0,
        num_segments=4,
        direction=(1.0, 0.0, 0.0),
        stretch_stiffness=fixed_k,
        stretch_damping=fixed_c,
    )
    spur = fs.rod_params_from_material(
        youngs_modulus_pa=5.0e6,
        damping_ratio=0.04,
        length=0.08,
        radius=0.006,
        density=280.0,
        num_segments=3,
        direction=(0.0, 0.0, -1.0),
        stretch_stiffness=fixed_k,
        stretch_damping=fixed_c,
    )
    stem = fs.rod_params_from_material(
        youngs_modulus_pa=2.0e6,
        damping_ratio=0.03,
        length=0.04,
        radius=0.004,
        density=250.0,
        num_segments=2,
        direction=(0.0, 0.0, -1.0),
        stretch_stiffness=fixed_k,
        stretch_damping=fixed_c,
    )
    base = fs.FruitingSystemParams(
        primary=primary,
        secondary=None,
        spur=spur,
        stem=stem,
        apple_radius=0.04,
        apple_density=800.0,
    )
    out = cmaes.YoungsModulusCandidate(
        primary=9.0e7, spur=6.0e6, stem=3.0e6
    ).apply_to(base)
    assert out.primary is not None and out.spur is not None and out.stem is not None
    for rod in (out.primary, out.spur, out.stem):
        assert rod.stretch_stiffness == pytest.approx(fixed_k)
        assert rod.stretch_damping == pytest.approx(fixed_c)


def test_log10_e_from_params_requires_primary_spur_stem():
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    rod = fs.rod_params_from_material(
        youngs_modulus_pa=1.0e7,
        damping_ratio=0.05,
        length=0.10,
        radius=0.01,
        density=300.0,
        num_segments=4,
        direction=(1.0, 0.0, 0.0),
    )
    base = fs.FruitingSystemParams(
        primary=rod,
        secondary=None,
        spur=None,
        stem=None,
        apple_radius=0.04,
        apple_density=800.0,
    )
    with pytest.raises(ValueError, match="primary.*spur.*stem"):
        cmaes.log10_e_from_params(base)


def test_set_rod_youngs_modulus_rederives_and_preserves_zeta():
    base = _base_primary_spur_stem()
    assert base.primary is not None
    old_zeta = base.primary.damping_ratio
    e_new = 3.0e8
    out = fs.set_rod_youngs_modulus(base, "primary", e_new)
    assert out.primary is not None
    assert out.primary.youngs_modulus_pa == pytest.approx(e_new)
    assert out.primary.damping_ratio == pytest.approx(old_zeta)
    expected = fs.rod_params_from_material(
        e_new,
        old_zeta,
        base.primary.length,
        base.primary.radius,
        base.primary.density,
        base.primary.num_segments,
        base.primary.direction,
    )
    # Without fixed stretch on fixture sample, bend (and beam stretch) follow E.
    assert out.primary.bend_stiffness == pytest.approx(expected.bend_stiffness)


def test_candidates_from_log10_e_rejects_wrong_length():
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    with pytest.raises(ValueError, match="3"):
        cmaes.candidates_from_log10_e((8.0, 7.0))


def test_gt_candidate_reads_lossless_structure_params(monkeypatch):
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    gt_params = _base_primary_spur_stem_with_secondary()
    monkeypatch.setattr(cmaes, "true_params_for_structure", lambda _dataset, _idx: gt_params)

    candidate = cmaes.gt_youngs_modulus_candidate_from_structure(object(), 3)

    assert candidate == cmaes.YoungsModulusCandidate(
        primary=gt_params.primary.youngs_modulus_pa,
        spur=gt_params.spur.youngs_modulus_pa,
        stem=gt_params.stem.youngs_modulus_pa,
    )
    assert gt_params.secondary is not None


def test_maybe_include_gt_candidate_is_configurable_and_deduplicates_in_log_space():
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    gt = cmaes.YoungsModulusCandidate(1e8, 10**7.5, 1e7)
    near = cmaes.YoungsModulusCandidate(
        10 ** (8.0 + 5e-10), 10**7.5, 1e7
    )

    assert cmaes.maybe_include_gt_candidate([near], gt, include_gt=True) == [near]
    assert cmaes.maybe_include_gt_candidate([], gt, include_gt=False) == []
    assert cmaes.maybe_include_gt_candidate([], gt, include_gt=True) == [gt]
