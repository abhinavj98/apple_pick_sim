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


# --- Task 1: bounds, sigma, RNG, pycma options ---


def _valid_youngs_ranges() -> dict:
    return {
        "primary": {"youngs_modulus_pa": {"min": 1.0e7, "max": 1.0e9}},
        "spur": {"youngs_modulus_pa": {"min": 1.0e6, "max": 1.0e8}},
        "stem": {"youngs_modulus_pa": {"min": 1.0e5, "max": 1.0e7}},
        "secondary": {"youngs_modulus_pa": {"min": 1.0e7, "max": 1.0e8}},
    }


def test_default_initial_sigma_log10_is_one_decade():
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    assert cmaes.DEFAULT_INITIAL_SIGMA_LOG10 == 1.0
    # Before bound effects, mean +/- 2 sigma spans +/- 2 decades.
    assert 10.0 ** (2.0 * cmaes.DEFAULT_INITIAL_SIGMA_LOG10) == pytest.approx(100.0)


def test_extract_youngs_modulus_cma_bounds_from_fixture_paths():
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    bounds = cmaes.extract_youngs_modulus_cma_bounds(_valid_youngs_ranges())
    assert bounds.primary.physical_min_pa == 1.0e7
    assert bounds.primary.physical_max_pa == 1.0e9
    assert bounds.spur.physical_min_pa == 1.0e6
    assert bounds.stem.physical_max_pa == 1.0e7
    assert bounds.primary.log10_min == pytest.approx(math.log10(1.0e7))
    assert bounds.primary.log10_max == pytest.approx(math.log10(1.0e9))
    assert bounds.primary.log10_midpoint == pytest.approx(
        0.5 * (math.log10(1.0e7) + math.log10(1.0e9))
    )
    assert bounds.log10_lower == pytest.approx(
        (math.log10(1.0e7), math.log10(1.0e6), math.log10(1.0e5))
    )
    assert bounds.log10_upper == pytest.approx(
        (math.log10(1.0e9), math.log10(1.0e8), math.log10(1.0e7))
    )
    assert bounds.log10_midpoint == pytest.approx(
        (
            0.5 * (math.log10(1.0e7) + math.log10(1.0e9)),
            0.5 * (math.log10(1.0e6) + math.log10(1.0e8)),
            0.5 * (math.log10(1.0e5) + math.log10(1.0e7)),
        )
    )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r: r["primary"].pop("youngs_modulus_pa"), "primary"),
        (lambda r: r["spur"]["youngs_modulus_pa"].pop("min"), "spur"),
        (lambda r: r["stem"]["youngs_modulus_pa"].__setitem__("max", None), "stem"),
        (lambda r: r["primary"]["youngs_modulus_pa"].__setitem__("min", "bad"), "primary"),
        (lambda r: r["spur"]["youngs_modulus_pa"].__setitem__("min", float("nan")), "spur"),
        (lambda r: r["stem"]["youngs_modulus_pa"].__setitem__("max", float("inf")), "stem"),
        (lambda r: r["primary"]["youngs_modulus_pa"].__setitem__("min", 0.0), "primary"),
        (lambda r: r["spur"]["youngs_modulus_pa"].__setitem__("min", -1.0), "spur"),
        (
            lambda r: r["stem"]["youngs_modulus_pa"].__setitem__("min", 1.0e7)
            or r["stem"]["youngs_modulus_pa"].__setitem__("max", 1.0e7),
            "stem",
        ),
        (
            lambda r: r["primary"]["youngs_modulus_pa"].__setitem__("min", 1.0e9)
            or r["primary"]["youngs_modulus_pa"].__setitem__("max", 1.0e7),
            "primary",
        ),
    ],
)
def test_extract_youngs_modulus_cma_bounds_rejects_invalid(mutate, match):
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    ranges = _valid_youngs_ranges()
    mutate(ranges)
    with pytest.raises(ValueError, match=match):
        cmaes.extract_youngs_modulus_cma_bounds(ranges)


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_validate_initial_sigma_log10_rejects_non_positive_or_non_finite(sigma):
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    with pytest.raises(ValueError, match="initial.sigma|sigma"):
        cmaes.validate_initial_sigma_log10(sigma)


def test_validate_initial_sigma_log10_accepts_positive_finite():
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    assert cmaes.validate_initial_sigma_log10(1.0) == 1.0
    assert cmaes.validate_initial_sigma_log10(0.25) == 0.25


def test_derive_structure_cma_seed_is_stable_positive_and_remaps_zero():
    import numpy as np
    from numpy.random import SeedSequence

    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    seed = cmaes.derive_structure_cma_seed(base_seed=0, structure_idx=3)
    expected = int(
        SeedSequence([0 % 2**32, 3 % 2**32]).generate_state(1, dtype=np.uint32)[0]
    )
    if expected == 0:
        expected = 1
    assert seed == expected
    assert 1 <= seed <= 2**32 - 1
    assert cmaes.derive_structure_cma_seed(0, 3) == seed


def test_derive_structure_cma_seeds_rejects_selected_structure_collisions(monkeypatch):
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    monkeypatch.setattr(
        cmaes,
        "derive_structure_cma_seed",
        lambda base_seed, structure_idx: 7,
    )
    with pytest.raises(ValueError, match="collision|collide|duplicate"):
        cmaes.derive_structure_cma_seeds(base_seed=0, structure_indices=(0, 1))


def test_build_pycma_options_omit_bounds_include_randn_nan_seed_and_optional_popsize():
    import numpy as np

    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    rng = np.random.default_rng(1)
    randn = cmaes.make_pycma_randn(rng)
    options = cmaes.build_pycma_options(randn=randn)
    assert "bounds" not in options
    assert options["randn"] is randn
    assert np.isnan(options["seed"])
    assert options["verbose"] == -9
    assert "popsize" not in options

    options_pop = cmaes.build_pycma_options(randn=randn, population_size=8)
    assert options_pop["popsize"] == 8


def test_resolve_initial_mean_log10_bounds_midpoint_and_explicit():
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    bounds = cmaes.extract_youngs_modulus_cma_bounds(_valid_youngs_ranges())
    assert cmaes.resolve_initial_mean_log10("bounds_midpoint", bounds) == pytest.approx(
        list(bounds.log10_midpoint)
    )
    explicit = (
        bounds.log10_lower[0] + 0.1,
        bounds.log10_lower[1] + 0.1,
        bounds.log10_lower[2] + 0.1,
    )
    assert cmaes.resolve_initial_mean_log10(explicit, bounds) == pytest.approx(list(explicit))


def test_resolve_initial_mean_log10_allows_outside_fixture_box():
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    bounds = cmaes.extract_youngs_modulus_cma_bounds(_valid_youngs_ranges())
    outside = (
        bounds.log10_upper[0] + 1.0,
        bounds.log10_midpoint[1],
        bounds.log10_midpoint[2],
    )
    assert cmaes.resolve_initial_mean_log10(outside, bounds) == pytest.approx(list(outside))


def test_create_structure_cma_optimizer_uses_midpoint_sigma_and_dedicated_rng():
    import cma
    import numpy as np

    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    bounds = cmaes.extract_youngs_modulus_cma_bounds(_valid_youngs_ranges())
    es, effective_seed, rng = cmaes.create_structure_cma_optimizer(
        bounds,
        initial_sigma_log10=1.0,
        base_seed=0,
        structure_idx=2,
        population_size=6,
    )
    assert isinstance(es, cma.CMAEvolutionStrategy)
    assert effective_seed == cmaes.derive_structure_cma_seed(0, 2)
    assert isinstance(rng, np.random.Generator)
    assert list(es.mean) == pytest.approx(list(bounds.log10_midpoint))
    assert es.sigma == pytest.approx(1.0)
    assert es.opts["popsize"] == 6
    assert np.isnan(es.opts["seed"])
    assert es.opts.get("bounds") in (None, [None, None])


def test_create_structure_cma_optimizer_uses_explicit_initial_mean():
    import cma

    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    bounds = cmaes.extract_youngs_modulus_cma_bounds(_valid_youngs_ranges())
    mean = (
        bounds.log10_lower[0] + 0.25,
        bounds.log10_lower[1] + 0.25,
        bounds.log10_lower[2] + 0.25,
    )
    es, _, _ = cmaes.create_structure_cma_optimizer(
        bounds,
        initial_mean_log10=mean,
        initial_sigma_log10=0.5,
        base_seed=1,
        structure_idx=0,
        population_size=4,
    )
    assert isinstance(es, cma.CMAEvolutionStrategy)
    assert list(es.result.xfavorite) == pytest.approx(list(mean))
    assert es.sigma == pytest.approx(0.5)


def test_interleaved_structure_optimizers_are_deterministic_and_distinct():
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    bounds = cmaes.extract_youngs_modulus_cma_bounds(_valid_youngs_ranges())

    def _run_once():
        es0, _, _ = cmaes.create_structure_cma_optimizer(
            bounds, base_seed=0, structure_idx=0, population_size=4
        )
        es1, _, _ = cmaes.create_structure_cma_optimizer(
            bounds, base_seed=0, structure_idx=1, population_size=4
        )
        out0 = []
        out1 = []
        for _ in range(3):
            s0 = es0.ask()
            s1 = es1.ask()
            out0.append([list(map(float, row)) for row in s0])
            out1.append([list(map(float, row)) for row in s1])
            fitness0 = [
                float(sum((x - m) ** 2 for x, m in zip(row, (7.0, 6.0, 5.0))))
                for row in s0
            ]
            fitness1 = [
                float(sum((x - m) ** 2 for x, m in zip(row, (8.0, 7.0, 6.0))))
                for row in s1
            ]
            es0.tell(s0, fitness0)
            es1.tell(s1, fitness1)
        return out0, out1

    first0, first1 = _run_once()
    second0, second1 = _run_once()
    assert first0 == second0
    assert first1 == second1
    assert first0 != first1


def test_sigma_exploration_is_not_clipped_to_fixture_bounds():
    """With sigma larger than the fixture box, ask samples may leave the box."""
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    narrow = {
        "primary": {"youngs_modulus_pa": {"min": 1.0e8, "max": 2.0e8}},
        "spur": {"youngs_modulus_pa": {"min": 1.0e7, "max": 2.0e7}},
        "stem": {"youngs_modulus_pa": {"min": 1.0e6, "max": 2.0e6}},
    }
    bounds = cmaes.extract_youngs_modulus_cma_bounds(narrow)
    for seg in (bounds.primary, bounds.spur, bounds.stem):
        half_width = 0.5 * (seg.log10_max - seg.log10_min)
        assert half_width < 2.0 * cmaes.DEFAULT_INITIAL_SIGMA_LOG10
    es, _, _ = cmaes.create_structure_cma_optimizer(
        bounds, base_seed=0, structure_idx=0, population_size=12
    )
    samples = es.ask()
    outside = False
    for row in samples:
        for value, lo, hi in zip(row, bounds.log10_lower, bounds.log10_upper, strict=True):
            if float(value) < lo - 1e-9 or float(value) > hi + 1e-9:
                outside = True
    assert outside, "expected at least one ask sample outside the narrow fixture box"


def test_validate_ask_population_allows_outside_fixture_box():
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    bounds = cmaes.extract_youngs_modulus_cma_bounds(_valid_youngs_ranges())
    outside = (
        (
            bounds.log10_upper[0] + 2.0,
            bounds.log10_lower[1] - 2.0,
            bounds.log10_midpoint[2],
        ),
    )
    parsed = cmaes._validate_ask_population(
        outside, population_size=1, bounds=bounds
    )
    assert parsed == outside


# Shipped CMA absolute safety box: all roles 0.1–100 GPa (log10 8–11).
# Former asymmetric box was primary [9,11], spur/stem [8,10].
_SHIPPED_SEARCH_BOUNDS_LOG10 = ((8.0, 8.0, 8.0), (11.0, 11.0, 11.0))


def test_validate_ask_population_rejects_outside_search_bounds():
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    bounds = cmaes.extract_youngs_modulus_cma_bounds(_valid_youngs_ranges())
    outside = ((7.9, 9.5, 9.5),)
    with pytest.raises(cmaes.CmaGenerationFailure, match="outside search bounds"):
        cmaes._validate_ask_population(
            outside,
            population_size=1,
            bounds=bounds,
            search_bounds_log10=_SHIPPED_SEARCH_BOUNDS_LOG10,
        )
    outside_high = ((9.5, 11.1, 9.5),)
    with pytest.raises(cmaes.CmaGenerationFailure, match="outside search bounds"):
        cmaes._validate_ask_population(
            outside_high,
            population_size=1,
            bounds=bounds,
            search_bounds_log10=_SHIPPED_SEARCH_BOUNDS_LOG10,
        )


def test_validate_ask_population_accepts_expanded_box_former_asymmetric_exterior():
    """Points excluded by old [9,11]/[8,10]/[8,10] box must pass under [8,11]^3."""
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    bounds = cmaes.extract_youngs_modulus_cma_bounds(_valid_youngs_ranges())
    # primary lower 9→8, spur/stem upper 10→11
    interior = ((8.5, 10.5, 10.5),)
    parsed = cmaes._validate_ask_population(
        interior,
        population_size=1,
        bounds=bounds,
        search_bounds_log10=_SHIPPED_SEARCH_BOUNDS_LOG10,
    )
    assert parsed == interior


def test_create_structure_cma_optimizer_passes_search_bounds_to_pycma():
    import cma

    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    bounds = cmaes.extract_youngs_modulus_cma_bounds(_valid_youngs_ranges())
    es, _, _ = cmaes.create_structure_cma_optimizer(
        bounds,
        initial_mean_log10=(9.5, 9.5, 9.5),
        initial_sigma_log10=0.5,
        base_seed=0,
        structure_idx=0,
        population_size=4,
        search_bounds_log10=_SHIPPED_SEARCH_BOUNDS_LOG10,
    )
    assert isinstance(es, cma.CMAEvolutionStrategy)
    assert es.opts["bounds"] == [[8.0, 8.0, 8.0], [11.0, 11.0, 11.0]]


def test_pycma_dependency_is_importable():
    import cma

    assert hasattr(cma, "CMAEvolutionStrategy")
