"""make_rl_harvest_real_fixture.build_fixture: sys-ID materials (log / linear mu +- k sigma) and real geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from apple_pick_gym.batched_examples.make_rl_harvest_real_fixture import PHENO, build_fixture

_MODULI = ("youngs_modulus_pa", "flexural_modulus_pa")


def _base_fixture() -> dict:
    seg = {"min": 1.0, "max": 1.0}
    return {
        "args": {"robot_base_pos": [0.0, 0.0, 0.0]},
        "sim_build": {"support_dr": {"kp": dict(seg), "roll_kp": dict(seg), "zeta": dict(seg)}},
        "primary": {"length": dict(seg), "radius": dict(seg), "density": dict(seg)},
        "spur": {},
        "stem": {},
        "apple": {"radius": dict(seg), "density": dict(seg)},
    }


def _row(spur_len: float, stem_len: float, apple_r: float, stem_dir: list[float]) -> dict:
    return {
        "primary": {"length": 0.827, "radius": 0.0125},
        "spur": {"length": spur_len, "radius": 0.005, "density": 5000.0},
        "stem": {"length": stem_len, "radius": 0.0009, "density": 1000.0, "direction": stem_dir},
        "apple_radius": apple_r,
        "apple_density": 1900.0,
    }


def _inputs():
    rng = np.random.default_rng(0)
    # 10 phenotype columns: six log knobs (positive), three zetas in (0, 1), primary density
    final = np.column_stack(
        [10.0 ** rng.uniform(2.2, 2.6, 6)]
        + [10.0 ** rng.uniform(6, 9, 6) for _ in range(4)]
        + [10.0 ** rng.uniform(-0.3, 0.0, 6)]
        + [rng.uniform(0.2, 0.8, 6) for _ in range(3)]
        + [rng.uniform(400, 1500, 6)]
    )
    rows = [
        _row(0.09, 0.0137, 0.030, [0.0, 0.0, -1.0]),
        _row(0.10, 0.0145, 0.032, [0.3, 0.1, -0.95]),
        _row(0.106, 0.0148, 0.0335, [-0.2, 0.4, -0.89]),
    ]
    bases = [[0.0, 0.7, 0.6]] * 3
    return final, rows, bases


def _build(**kw):
    final, rows, bases = _inputs()
    args = dict(k_mat=1.96, k_geom=0.5, stem_angles="uniform", geom_mode="sigma", robot_base_pos=None)
    args.update(kw)
    return final, rows, build_fixture(final, ["02", "03"], rows, bases, _base_fixture(), **args)


def test_moduli_are_log_uniform_mu_k_sigma_in_log10() -> None:
    final, _, out = _build()
    for seg, cols in (("spur", (3, 1)), ("stem", (4, 2))):  # (youngs, flex) columns of PHENO
        for key, i in zip(_MODULI, cols):
            lx = np.log10(final[:, i])
            r = out[seg][key]
            assert r["scale"] == "log"
            assert math.log10(r["min"]) == pytest.approx(lx.mean() - 1.96 * lx.std(ddof=1))
            assert math.log10(r["max"]) == pytest.approx(lx.mean() + 1.96 * lx.std(ddof=1))
    assert PHENO[3] == "E_youngs_spur" and PHENO[2] == "E_flex_stem"


def test_zetas_and_primary_density_are_linear_mu_k_sigma() -> None:
    final, _, out = _build()
    z = final[:, 6]
    r = out["spur"]["damping_ratio"]
    assert "scale" not in r
    assert r["min"] == pytest.approx(max(0.0, z.mean() - 1.96 * z.std(ddof=1)))
    assert r["max"] == pytest.approx(z.mean() + 1.96 * z.std(ddof=1))
    assert "scale" not in out["primary"]["density"]


def test_support_dr_bounds_carry_no_scale_key() -> None:
    # support kp / roll_kp are log-sampled by support_joint_dr itself; their schema takes min/max only
    _, _, out = _build()
    for key in ("kp", "roll_kp", "zeta"):
        assert set(out["sim_build"]["support_dr"][key]) == {"min", "max"}


def test_geom_minmax_spans_the_real_structures() -> None:
    _, rows, out = _build(geom_mode="minmax")
    assert out["spur"]["length"] == {"min": 0.09, "max": 0.106}
    assert out["stem"]["length"] == {"min": 0.0137, "max": 0.0148}
    assert out["apple"]["radius"] == {"min": 0.030, "max": 0.0335}
    assert out["primary"]["length"] == {"min": 0.827, "max": 0.827}
    for r in (out["spur"]["length"], out["stem"]["length"], out["apple"]["radius"]):
        assert "scale" not in r


def test_geom_sigma_default_is_unchanged() -> None:
    _, rows, out = _build(geom_mode="sigma", k_geom=0.5)
    x = np.array([r["spur"]["length"] for r in rows])
    assert out["spur"]["length"]["min"] == pytest.approx(x.mean() - 0.5 * x.std(ddof=1))
    assert out["spur"]["length"]["max"] == pytest.approx(x.mean() + 0.5 * x.std(ddof=1))


def test_fixture_validates_and_samples_moduli_log_uniform() -> None:
    from apple_pick_sim.fruiting_system.params import _validate_ranges, sample_params

    import copy
    import json
    from pathlib import Path

    real = json.loads(
        (Path(__file__).resolve().parents[2] / "apple_pick_sim/fixtures/fruiting_system_ranges_rl_harvest_real_g05_m1.json").read_text()
    )
    out = copy.deepcopy(real)
    out["spur"]["flexural_modulus_pa"] = {"min": 1e5, "max": 1e11, "scale": "log"}
    _validate_ranges(out)
    e = np.log10([sample_params(out, seed).spur.flexural_modulus_pa for seed in range(400)])
    # log-uniform over 6 decades: about a sixth below 1e6; linear would put ~0 there
    assert 0.08 < np.mean(e < 6.0) < 0.26
