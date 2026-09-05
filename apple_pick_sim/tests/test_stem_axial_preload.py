"""Optional axial-preload helpers and real-replay no-preload wiring.

Real-replay mapping keeps catalog stem rest length and does **not** set
``preload_chord_m`` / ``axial_preload_n``. Apple radius absorbs spur→CoM slop;
natural stretch develops during settle. The pure-math helpers below remain for
optional/explicit preload tests only — they are not used on the real path.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from apple_pick_sim.fruiting_system import params as fs
from apple_pick_sim.system_id.real_pre_grasp_params import (
    fruiting_params_from_pre_grasp_meta,
    map_pre_grasp_geometry,
    primary_direction_from_fixture,
)
from apple_pick_sim.tests.test_real_pre_grasp_params import _synthetic_pre_grasp_meta

VARIANCE = Path(
    "apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json"
)

RADIUS = 0.0009
E_NOMINAL = 62.5e6
CHORD = 0.01155
MG = 2.158


def _area(radius: float) -> float:
    return math.pi * radius * radius


# ---------------------------------------------------------------------------
# Pure rest-length solve
# ---------------------------------------------------------------------------


def test_rest_length_is_shorter_than_chord_by_the_elastic_stretch():
    rest = fs.rest_length_for_axial_preload(
        chord_m=CHORD,
        preload_n=MG,
        youngs_modulus_pa=E_NOMINAL,
        radius=RADIUS,
    )
    ea = E_NOMINAL * _area(RADIUS)
    assert rest == pytest.approx(CHORD / (1.0 + MG / ea))
    assert rest < CHORD
    # ~150 um of stretch on an ~11.5 mm stem at the fixture-midpoint modulus.
    assert 1e-4 < CHORD - rest < 2e-4


def test_zero_preload_leaves_the_rest_length_at_the_chord():
    rest = fs.rest_length_for_axial_preload(
        chord_m=CHORD,
        preload_n=0.0,
        youngs_modulus_pa=E_NOMINAL,
        radius=RADIUS,
    )
    assert rest == pytest.approx(CHORD)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"chord_m": 0.0}, "chord_m"),
        ({"chord_m": -0.01}, "chord_m"),
        ({"preload_n": -1.0}, "preload_n"),
        ({"youngs_modulus_pa": 0.0}, "youngs_modulus_pa"),
        ({"radius": 0.0}, "radius"),
    ],
)
def test_rest_length_rejects_invalid_inputs(kwargs, match):
    base = {
        "chord_m": CHORD,
        "preload_n": MG,
        "youngs_modulus_pa": E_NOMINAL,
        "radius": RADIUS,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        fs.rest_length_for_axial_preload(**base)


# ---------------------------------------------------------------------------
# The rest length must actually reproduce the requested tension
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_segments", [2, 3, 5, 8])
def test_rod_built_at_preload_carries_mg_when_stretched_to_the_chord(num_segments):
    """Segment count must not change the tension at a given total length."""
    rod = fs.rod_params_from_material(
        1.0e7,
        E_NOMINAL,
        0.05,
        CHORD,  # ignored when a preload spec is supplied
        RADIUS,
        300.0,
        num_segments,
        (0.0, 0.0, 1.0),
        preload_chord_m=CHORD,
        axial_preload_n=MG,
    )
    assert rod.length < CHORD
    assert fs.axial_tension_at_length(rod, CHORD) == pytest.approx(MG, rel=1e-9)


def test_rod_without_preload_spec_keeps_the_supplied_length():
    rod = fs.rod_params_from_material(
        1.0e7, E_NOMINAL, 0.05, CHORD, RADIUS, 300.0, 4, (0.0, 0.0, 1.0)
    )
    assert rod.length == pytest.approx(CHORD)
    assert rod.preload_chord_m is None
    assert rod.axial_preload_n is None
    assert fs.axial_tension_at_length(rod, CHORD) == pytest.approx(0.0)


def test_axial_tension_is_negative_in_compression():
    rod = fs.rod_params_from_material(
        1.0e7, E_NOMINAL, 0.05, CHORD, RADIUS, 300.0, 4, (0.0, 0.0, 1.0)
    )
    assert fs.axial_tension_at_length(rod, 0.9 * CHORD) < 0.0


# ---------------------------------------------------------------------------
# The whole point: preload must not ride along with the fitted modulus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("e_youngs", [1.0e7, 3.0e7, 6.25e7, 1.5e8, 5.0e8])
def test_preload_survives_a_youngs_modulus_override(e_youngs):
    """CMA sweeps E_youngs_stem; frame-0 tension must stay at mg regardless."""
    rod = fs.rod_params_from_material(
        1.0e7,
        E_NOMINAL,
        0.05,
        CHORD,
        RADIUS,
        300.0,
        4,
        (0.0, 0.0, 1.0),
        preload_chord_m=CHORD,
        axial_preload_n=MG,
    )
    params = fs.FruitingSystemParams(
        primary=rod,
        secondary=None,
        spur=None,
        stem=rod,
        apple_radius=0.04,
        apple_density=800.0,
    )
    out = fs.set_rod_youngs_modulus(params, "stem", e_youngs)

    assert out.stem.youngs_modulus_pa == pytest.approx(e_youngs)
    # Rest length moves so that the tension at the measured chord is unchanged.
    assert fs.axial_tension_at_length(out.stem, CHORD) == pytest.approx(MG, rel=1e-9)
    assert out.stem.preload_chord_m == pytest.approx(CHORD)
    assert out.stem.axial_preload_n == pytest.approx(MG)


def test_without_preload_spec_tension_scales_with_modulus():
    """Guards the regression this change exists to prevent."""
    rest = fs.rest_length_for_axial_preload(
        chord_m=CHORD, preload_n=MG, youngs_modulus_pa=E_NOMINAL, radius=RADIUS
    )
    rod = fs.rod_params_from_material(
        1.0e7, E_NOMINAL, 0.05, rest, RADIUS, 300.0, 4, (0.0, 0.0, 1.0)
    )
    params = fs.FruitingSystemParams(
        primary=rod,
        secondary=None,
        spur=None,
        stem=rod,
        apple_radius=0.04,
        apple_density=800.0,
    )
    out = fs.set_rod_youngs_modulus(params, "stem", 4.0 * E_NOMINAL)
    assert fs.axial_tension_at_length(out.stem, CHORD) == pytest.approx(4.0 * MG, rel=1e-6)


def test_bend_modulus_override_preserves_the_preload():
    rod = fs.rod_params_from_material(
        1.0e7,
        E_NOMINAL,
        0.05,
        CHORD,
        RADIUS,
        300.0,
        4,
        (0.0, 0.0, 1.0),
        preload_chord_m=CHORD,
        axial_preload_n=MG,
    )
    params = fs.FruitingSystemParams(
        primary=rod,
        secondary=None,
        spur=None,
        stem=rod,
        apple_radius=0.04,
        apple_density=800.0,
    )
    out = fs.set_rod_flexural_modulus(params, "stem", 5.0e7)
    assert fs.axial_tension_at_length(out.stem, CHORD) == pytest.approx(MG, rel=1e-9)


def test_round_trip_through_params_dict_preserves_preload_spec():
    rod = fs.rod_params_from_material(
        1.0e7,
        E_NOMINAL,
        0.05,
        CHORD,
        RADIUS,
        300.0,
        4,
        (0.0, 0.0, 1.0),
        preload_chord_m=CHORD,
        axial_preload_n=MG,
    )
    params = fs.FruitingSystemParams(
        primary=rod,
        secondary=None,
        spur=None,
        stem=rod,
        apple_radius=0.04,
        apple_density=800.0,
    )
    restored = fs.fruiting_params_from_dict(fs.fruiting_params_to_dict(params))
    assert restored.stem.length == pytest.approx(rod.length)
    assert fs.axial_tension_at_length(restored.stem, CHORD) == pytest.approx(MG, rel=1e-6)
    # The spec itself must survive, or a modulus override after a round trip would
    # silently fall back to scaling the preload with E.
    assert restored.stem.preload_chord_m == pytest.approx(CHORD)
    assert restored.stem.axial_preload_n == pytest.approx(MG)
    bumped = fs.set_rod_youngs_modulus(restored, "stem", 4.0 * E_NOMINAL)
    assert fs.axial_tension_at_length(bumped.stem, CHORD) == pytest.approx(MG, rel=1e-6)


# ---------------------------------------------------------------------------
# Wiring: real-replay mapping keeps catalog rest length (no axial preload)
# ---------------------------------------------------------------------------

_CATALOG_STEM_L = 0.025


def test_mapped_stem_geometry_has_no_preload():
    mapped = map_pre_grasp_geometry(
        _synthetic_pre_grasp_meta(),
        primary_dir=primary_direction_from_fixture(VARIANCE),
    )
    stem = mapped.rod_geometry["stem"]
    assert stem["length_m"] == pytest.approx(_CATALOG_STEM_L)
    assert "preload_chord_m" not in stem
    assert "axial_preload_n" not in stem
    assert "preload_chord_m" not in mapped.rod_geometry["spur"]
    assert "preload_chord_m" not in mapped.rod_geometry["primary"]


def test_built_stem_uses_catalog_rest_length_without_preload():
    params, _base, diagnostics = fruiting_params_from_pre_grasp_meta(
        _synthetic_pre_grasp_meta(), fixture_path=VARIANCE
    )
    assert params.stem.length == pytest.approx(_CATALOG_STEM_L)
    assert params.stem.axial_preload_n is None
    assert params.stem.preload_chord_m is None
    assert diagnostics["stem_axial_preload_n"] is None
    assert diagnostics["stem_preload_chord_m"] is None
    assert diagnostics["stem_catalog_length_m"] == pytest.approx(_CATALOG_STEM_L)


def test_woody_rods_are_not_preloaded():
    params, _base, _diag = fruiting_params_from_pre_grasp_meta(
        _synthetic_pre_grasp_meta(), fixture_path=VARIANCE
    )
    assert params.spur.axial_preload_n is None
    assert params.primary.axial_preload_n is None
    assert params.spur.length == pytest.approx(0.1)
    assert params.primary.length == pytest.approx(0.2)


@pytest.mark.parametrize("e_youngs", [1.0e7, 6.25e7, 5.0e8])
def test_mapped_stem_rest_length_invariant_to_fitted_modulus(e_youngs):
    params, _base, _diag = fruiting_params_from_pre_grasp_meta(
        _synthetic_pre_grasp_meta(), fixture_path=VARIANCE
    )
    out = fs.set_rod_youngs_modulus(params, "stem", e_youngs)
    assert out.stem.length == pytest.approx(_CATALOG_STEM_L)
    assert out.stem.axial_preload_n is None
