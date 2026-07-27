"""Tests for P0 variational fruiting-system generation.

Validates:
  - JSON range schema (fixture loads and has required keys)
  - Generator determinism: same (ranges, seed) → identical geometry fingerprint
  - Generator variance: different seeds → distinct geometry
  - Stiffness ordering: primary bend stiffness ≥ secondary bend stiffness (always)
  - Topology: optional chain (primary → secondary → spur → stem → apple); JSON null skips a piece
  - Headless Newton rollout: short SolverVBD simulation runs without error
  - Rollout determinism: same seed → same body positions after N steps
"""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from apple_pick_sim.tests.conftest import NO_SELF_COLLISION_KW

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
RANGES_FIXTURE = FIXTURES_DIR / "fruiting_system_ranges_straight_rod_test.json"
VARIANCE_FIXTURE = FIXTURES_DIR / "fruiting_system_ranges_example_variance.json"
SOFT_VARIANCE_FIXTURE = FIXTURES_DIR / "fruiting_system_ranges_example_variance_soft.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_module():
    """Lazy import so the test file can be collected before the module exists."""
    import apple_pick_sim.fruiting_system as fs

    return fs


# ---------------------------------------------------------------------------
# Fixture schema
# ---------------------------------------------------------------------------


def test_ranges_fixture_exists():
    assert RANGES_FIXTURE.exists(), f"Range fixture not found at {RANGES_FIXTURE}"


def test_load_ranges_returns_dict():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    assert isinstance(ranges, dict)


def test_load_ranges_has_required_segments():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    for segment in ("primary", "secondary", "spur", "stem", "apple"):
        assert segment in ranges, f"Missing segment '{segment}' in ranges"


def test_load_ranges_segment_keys():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    required_rod_keys = {
        "num_segments",
        "length",
        "radius",
        "youngs_modulus_pa",
        "damping_ratio",
        "density",
    }
    for seg in ("primary", "secondary", "spur", "stem"):
        if ranges.get(seg) is None:
            continue
        for key in required_rod_keys:
            assert key in ranges[seg], f"Missing key '{key}' in segment '{seg}'"
    if ranges.get("apple") is not None:
        for key in ("radius", "density"):
            assert key in ranges["apple"], f"Missing key '{key}' in apple"


def test_load_ranges_min_max_ordering():
    """Every range must have min <= max."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    for seg_name, seg in ranges.items():
        if seg_name == "args" or seg is None or not isinstance(seg, dict):
            continue
        for key, val in seg.items():
            if isinstance(val, dict) and "min" in val and "max" in val:
                assert val["min"] <= val["max"], (
                    f"Range {seg_name}.{key}: min ({val['min']}) > max ({val['max']})"
                )


def test_fixture_args_in_straight_rod_fixture():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    args = fs.parse_fixture_args(ranges)
    assert args.fruiting_base_pos == (0.0, 0.2, 1.3)
    assert args.robot_base_pos == (0.0, 0.0, 1.0)


def test_fixture_args_null_robot_base_in_variance_fixture():
    fs = _import_module()
    ranges = fs.load_ranges(VARIANCE_FIXTURE)
    args = fs.parse_fixture_args(ranges)
    assert args.fruiting_base_pos == (0.0, 0.2, 1.5)
    assert args.robot_base_pos == (0.0, 0.0, 1.0)


def test_soft_variance_fixture_uses_tenth_stiffness_ranges():
    fs = _import_module()
    baseline = fs.load_ranges(VARIANCE_FIXTURE)
    soft = fs.load_ranges(SOFT_VARIANCE_FIXTURE)

    for segment in ("primary", "secondary", "spur", "stem"):
        for key in ("youngs_modulus_pa",):
            assert soft[segment][key]["min"] == pytest.approx(
                baseline[segment][key]["min"] * 0.1
            )
            assert soft[segment][key]["max"] == pytest.approx(
                baseline[segment][key]["max"] * 0.1
            )


def test_resolve_fruiting_base_pos_prefers_override():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    assert fs.resolve_fruiting_base_pos(ranges, (9.0, 9.0, 9.0)) == (0.0, 0.2, 1.3)
    assert fs.resolve_fruiting_base_pos(
        ranges, (9.0, 9.0, 9.0), override=(1.0, 2.0, 3.0)
    ) == (1.0, 2.0, 3.0)


def test_resolve_fruiting_base_pos_falls_back_to_default():
    fs = _import_module()
    ranges = {"primary": fs.load_ranges(RANGES_FIXTURE)["primary"]}
    assert fs.resolve_fruiting_base_pos(ranges, (0.5, 0.5, 0.5)) == (0.5, 0.5, 0.5)


def test_invalid_fixture_args_rejected():
    fs = _import_module()
    import copy
    import json
    import tempfile

    ranges = fs.load_ranges(RANGES_FIXTURE)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        bad_ranges = copy.deepcopy(ranges)
        bad_ranges["args"] = {"fruiting_base_pos": [0.0, 0.0]}
        json.dump(bad_ranges, f)
        path = f.name
    with pytest.raises(ValueError, match="fruiting_base_pos"):
        fs.load_ranges(path)


# ---------------------------------------------------------------------------
# Optional sim_build (VIC + joint overrides)
# ---------------------------------------------------------------------------


_GOOD_SIM_BUILD = {
    "vic_gains": {
        "linear_k": 200.0,
        "linear_d": 10.0,
        "angular_k": 10.0,
        "angular_d": 1.0,
    },
    "joint_angular_kd_overrides": {
        "support": 0.3,
        "primary_spur": 0.3,
        "spur_stem": 0.3,
        "stem_apple": 0.3,
    },
    "joint_linear_kd_overrides": {
        "support": 0.3,
        "primary_spur": 0.3,
        "spur_stem": 0.3,
        "stem_apple": 0.3,
    },
    "joint_angular_kp_overrides": {"support": 2000.0},
    "joint_linear_kp_overrides": {"support": 2000.0},
}


def _write_ranges_with_sim_build(base_ranges: dict, sim_build: dict | None, *, omit: bool = False):
    import copy
    import json
    import tempfile

    payload = copy.deepcopy(base_ranges)
    if omit:
        payload.pop("sim_build", None)
    elif sim_build is None:
        payload["sim_build"] = None
    else:
        payload["sim_build"] = sim_build
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        return f.name


def test_parse_sim_build_absent_returns_none():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    assert fs.parse_sim_build(ranges) is None


def test_load_ranges_accepts_good_sim_build():
    fs = _import_module()
    path = _write_ranges_with_sim_build(fs.load_ranges(RANGES_FIXTURE), _GOOD_SIM_BUILD)
    ranges = fs.load_ranges(path)
    sb = fs.parse_sim_build(ranges)
    assert sb is not None
    assert sb.vic_gains.linear_k == pytest.approx(200.0)
    assert sb.vic_gains.linear_d == pytest.approx(10.0)
    assert sb.vic_gains.angular_k == pytest.approx(10.0)
    assert sb.vic_gains.angular_d == pytest.approx(1.0)
    assert sb.joint_angular_kd_overrides == _GOOD_SIM_BUILD["joint_angular_kd_overrides"]
    assert sb.joint_linear_kd_overrides == _GOOD_SIM_BUILD["joint_linear_kd_overrides"]
    assert sb.joint_angular_kp_overrides == {"support": 2000.0}
    assert sb.joint_linear_kp_overrides == {"support": 2000.0}


def test_load_ranges_rejects_sim_build_negative_vic_gain():
    fs = _import_module()
    bad = dict(_GOOD_SIM_BUILD)
    bad["vic_gains"] = dict(_GOOD_SIM_BUILD["vic_gains"], linear_k=-1.0)
    path = _write_ranges_with_sim_build(fs.load_ranges(RANGES_FIXTURE), bad)
    with pytest.raises(ValueError, match="linear_k"):
        fs.load_ranges(path)


def test_load_ranges_rejects_sim_build_missing_vic_gains():
    fs = _import_module()
    bad = {k: v for k, v in _GOOD_SIM_BUILD.items() if k != "vic_gains"}
    path = _write_ranges_with_sim_build(fs.load_ranges(RANGES_FIXTURE), bad)
    with pytest.raises(ValueError, match="vic_gains"):
        fs.load_ranges(path)


def test_load_ranges_rejects_sim_build_unknown_joint_role():
    fs = _import_module()
    bad = dict(_GOOD_SIM_BUILD)
    bad["joint_angular_kd_overrides"] = {"not_a_role": 0.1}
    path = _write_ranges_with_sim_build(fs.load_ranges(RANGES_FIXTURE), bad)
    with pytest.raises(ValueError, match="not_a_role"):
        fs.load_ranges(path)


def test_load_ranges_rejects_unknown_sim_build_key():
    fs = _import_module()
    bad = dict(_GOOD_SIM_BUILD)
    bad["control_hz"] = 30.0
    path = _write_ranges_with_sim_build(fs.load_ranges(RANGES_FIXTURE), bad)
    with pytest.raises(ValueError, match="control_hz"):
        fs.load_ranges(path)


def test_load_ranges_sim_build_optional_joint_overrides():
    fs = _import_module()
    minimal = {"vic_gains": dict(_GOOD_SIM_BUILD["vic_gains"])}
    path = _write_ranges_with_sim_build(fs.load_ranges(RANGES_FIXTURE), minimal)
    sb = fs.parse_sim_build(fs.load_ranges(path))
    assert sb is not None
    assert sb.joint_angular_kd_overrides == {}
    assert sb.joint_linear_kp_overrides == {}
    assert sb.joint_damping_ratio is None


def test_parse_sim_build_accepts_joint_damping_ratio():
    fs = _import_module()
    block = {
        "vic_gains": dict(_GOOD_SIM_BUILD["vic_gains"]),
        "joint_damping_ratio": 0.2,
        "joint_angular_kp_overrides": {"support": 10000.0},
        "joint_linear_kp_overrides": {"support": 10000.0},
    }
    path = _write_ranges_with_sim_build(fs.load_ranges(RANGES_FIXTURE), block)
    sb = fs.parse_sim_build(fs.load_ranges(path))
    assert sb is not None
    assert sb.joint_damping_ratio == pytest.approx(0.2)
    assert sb.joint_angular_kd_overrides == {}
    assert sb.joint_linear_kd_overrides == {}


def test_parse_sim_build_rejects_joint_damping_ratio_with_absolute_kd():
    fs = _import_module()
    bad = {
        "vic_gains": dict(_GOOD_SIM_BUILD["vic_gains"]),
        "joint_damping_ratio": 0.2,
        "joint_angular_kd_overrides": {"support": 1.0},
    }
    path = _write_ranges_with_sim_build(fs.load_ranges(RANGES_FIXTURE), bad)
    with pytest.raises(ValueError, match="joint_damping_ratio.*mutually exclusive|mutually exclusive"):
        fs.load_ranges(path)


def test_parse_sim_build_accepts_joint_damping_ratio_above_one():
    """Joint ζ is nonnegative (may be > 1); not limited to the unit interval."""
    fs = _import_module()
    block = {
        "vic_gains": dict(_GOOD_SIM_BUILD["vic_gains"]),
        "joint_damping_ratio": 10.0,
    }
    path = _write_ranges_with_sim_build(fs.load_ranges(RANGES_FIXTURE), block)
    sb = fs.parse_sim_build(fs.load_ranges(path))
    assert sb is not None
    assert sb.joint_damping_ratio == pytest.approx(10.0)


def test_parse_sim_build_rejects_negative_joint_damping_ratio():
    fs = _import_module()
    bad = {
        "vic_gains": dict(_GOOD_SIM_BUILD["vic_gains"]),
        "joint_damping_ratio": -0.1,
    }
    path = _write_ranges_with_sim_build(fs.load_ranges(RANGES_FIXTURE), bad)
    with pytest.raises(ValueError, match="joint_damping_ratio"):
        fs.load_ranges(path)


# ---------------------------------------------------------------------------
# Material-parameter sampling (E, ζ)
# ---------------------------------------------------------------------------


def test_rod_params_from_material_derivation():
    fs = _import_module()
    rod = fs.rod_params_from_material(
        youngs_modulus_pa=1.0e7,
        damping_ratio=0.05,
        length=0.10,
        radius=0.01,
        density=300.0,
        num_segments=4,
        direction=(1.0, 0.0, 0.0),
    )
    area = math.pi * 0.01**2
    inertia = math.pi * 0.01**4 / 4.0
    l_seg = 0.10 / 4.0
    m_seg = 300.0 * area * l_seg
    j_seg = m_seg * (3.0 * 0.01**2 + l_seg**2) / 12.0
    assert rod.stretch_stiffness == pytest.approx(1.0e7 * area / l_seg)
    assert rod.bend_stiffness == pytest.approx(1.0e7 * inertia / l_seg)
    assert rod.stretch_damping == pytest.approx(
        2.0 * 0.05 * math.sqrt(rod.stretch_stiffness * m_seg)
    )
    assert rod.bend_damping == pytest.approx(
        2.0 * 0.05 * math.sqrt(rod.bend_stiffness * j_seg)
    )


def test_sample_params_stores_E_and_zeta_on_rod():
    fs = _import_module()
    params = fs.sample_params(fs.load_ranges(RANGES_FIXTURE), seed=0)
    assert params.primary is not None
    assert params.primary.youngs_modulus_pa > 0.0
    assert params.primary.damping_ratio >= 0.0


def test_sample_params_primary_E_ge_secondary():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    for seed in range(20):
        params = fs.sample_params(ranges, seed=seed)
        if params.primary is None or params.secondary is None:
            continue
        assert params.primary.youngs_modulus_pa >= params.secondary.youngs_modulus_pa


def test_load_ranges_rejects_bend_stiffness_key():
    fs = _import_module()
    import copy
    import json
    import tempfile

    ranges = fs.load_ranges(RANGES_FIXTURE)
    bad = copy.deepcopy(ranges)
    bad["primary"]["bend_stiffness"] = {"min": 1.0, "max": 2.0}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(bad, f)
        path = f.name
    with pytest.raises(ValueError, match="deprecated keys"):
        fs.load_ranges(path)


def test_load_ranges_rejects_min_gt_max_on_rod_keys():
    """Required rod ranges must validate min <= max (not only missing-key errors)."""
    fs = _import_module()
    import copy
    import json
    import tempfile

    ranges = fs.load_ranges(RANGES_FIXTURE)
    bad = copy.deepcopy(ranges)
    bad["primary"]["youngs_modulus_pa"] = {"min": 2.0e9, "max": 1.0e9}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(bad, f)
        path = f.name
    with pytest.raises(ValueError, match=r"min \(.*\) > max"):
        fs.load_ranges(path)


def test_load_ranges_requires_youngs_modulus_pa():
    fs = _import_module()
    import copy
    import json
    import tempfile

    ranges = fs.load_ranges(RANGES_FIXTURE)
    bad = copy.deepcopy(ranges)
    del bad["primary"]["youngs_modulus_pa"]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(bad, f)
        path = f.name
    with pytest.raises(ValueError, match="youngs_modulus_pa"):
        fs.load_ranges(path)


def test_fruiting_params_v2_roundtrip():
    fs = _import_module()
    params = fs.sample_params(fs.load_ranges(RANGES_FIXTURE), seed=42)
    encoded = fs.fruiting_params_to_dict(params)
    assert encoded["schema"] == fs.FRUITING_SYSTEM_PARAMS_SCHEMA
    assert encoded["primary"]["youngs_modulus_pa"] == pytest.approx(
        params.primary.youngs_modulus_pa
    )
    decoded = fs.fruiting_params_from_dict(encoded)
    assert decoded.primary.youngs_modulus_pa == pytest.approx(params.primary.youngs_modulus_pa)
    assert decoded.primary.damping_ratio == pytest.approx(params.primary.damping_ratio)


def test_fruiting_params_v1_deserialization():
    fs = _import_module()
    params = fs.sample_params(fs.load_ranges(RANGES_FIXTURE), seed=7)
    v1 = fs.fruiting_params_to_dict(params)
    v1["schema"] = fs.FRUITING_SYSTEM_PARAMS_SCHEMA_V1
    for seg in ("primary", "secondary", "spur", "stem"):
        if v1.get(seg) is None:
            continue
        v1[seg].pop("youngs_modulus_pa", None)
        v1[seg].pop("damping_ratio", None)
        v1[seg].pop("stretch_damping", None)
    decoded = fs.fruiting_params_from_dict(v1)
    assert decoded.primary.bend_stiffness == pytest.approx(params.primary.bend_stiffness)


# ---------------------------------------------------------------------------
# vbd_stretch_force override (F_max + ζ_stretch → k/c from geometry)
# ---------------------------------------------------------------------------


def test_stretch_knobs_from_max_force_matches_delta_fraction():
    fs = _import_module()
    length, radius, density, n = 0.10, 0.01, 300.0, 4
    f_max, zeta = 35.0, 1.5
    k, c = fs.stretch_knobs_from_max_force(
        f_max,
        zeta,
        length,
        radius,
        density,
        n,
    )
    l_seg = length / n
    m_seg = density * math.pi * radius**2 * l_seg
    assert k == pytest.approx(f_max / (0.05 * l_seg))
    assert c == pytest.approx(2.0 * zeta * math.sqrt(k * m_seg))


def test_rod_params_from_material_stretch_override():
    fs = _import_module()
    rod = fs.rod_params_from_material(
        youngs_modulus_pa=1.0e7,
        damping_ratio=0.05,
        length=0.10,
        radius=0.01,
        density=300.0,
        num_segments=4,
        direction=(1.0, 0.0, 0.0),
        stretch_stiffness=500000.0,
        stretch_damping=30.0,
    )
    assert rod.stretch_stiffness == pytest.approx(500000.0)
    assert rod.stretch_damping == pytest.approx(30.0)
    area = math.pi * 0.01**2
    inertia = math.pi * 0.01**4 / 4.0
    l_seg = 0.10 / 4.0
    j_seg = 300.0 * area * l_seg * (3.0 * 0.01**2 + l_seg**2) / 12.0
    assert rod.bend_stiffness == pytest.approx(1.0e7 * inertia / l_seg)
    assert rod.bend_damping == pytest.approx(
        2.0 * 0.05 * math.sqrt(rod.bend_stiffness * j_seg)
    )


def test_load_ranges_vbd_stretch_force_validates():
    fs = _import_module()
    import copy
    import json
    import tempfile

    ranges = copy.deepcopy(fs.load_ranges(RANGES_FIXTURE))
    ranges["primary"]["vbd_stretch_force"] = {
        "max_force_n": 35.0,
        "damping_ratio": 1.0,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(ranges, f)
        path = f.name
    loaded = fs.load_ranges(path)
    assert loaded["primary"]["vbd_stretch_force"]["max_force_n"] == 35.0
    assert loaded["primary"]["vbd_stretch_force"]["damping_ratio"] == 1.0


def test_load_ranges_vbd_stretch_force_rejects_partial_nonpositive_legacy():
    fs = _import_module()
    import copy
    import json
    import tempfile

    ranges = copy.deepcopy(fs.load_ranges(RANGES_FIXTURE))

    bad_partial = copy.deepcopy(ranges)
    bad_partial["primary"]["vbd_stretch_force"] = {"max_force_n": 35.0}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(bad_partial, f)
        path = f.name
    with pytest.raises(ValueError, match="damping_ratio"):
        fs.load_ranges(path)

    bad_nonpositive = copy.deepcopy(ranges)
    bad_nonpositive["primary"]["vbd_stretch_force"] = {
        "max_force_n": 35.0,
        "damping_ratio": 0.0,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(bad_nonpositive, f)
        path = f.name
    with pytest.raises(ValueError, match="damping_ratio"):
        fs.load_ranges(path)

    legacy = copy.deepcopy(ranges)
    legacy["primary"]["vbd_stretch_fixed"] = {
        "stretch_stiffness": 500000.0,
        "stretch_damping": 30.0,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(legacy, f)
        path = f.name
    with pytest.raises(ValueError, match="vbd_stretch_force"):
        fs.load_ranges(path)


def test_sample_params_stretch_force_derives_from_geometry():
    fs = _import_module()
    import copy
    import json
    import tempfile

    ranges = copy.deepcopy(fs.load_ranges(RANGES_FIXTURE))
    ranges["primary"]["vbd_stretch_force"] = {
        "max_force_n": 35.0,
        "damping_ratio": 1.5,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(ranges, f)
        path = f.name
    loaded = fs.load_ranges(path)
    p0 = fs.sample_params(loaded, seed=0)
    p1 = fs.sample_params(loaded, seed=99)
    assert p0.primary is not None and p1.primary is not None
    for rod in (p0.primary, p1.primary):
        k_exp, c_exp = fs.stretch_knobs_from_max_force(
            35.0,
            1.5,
            rod.length,
            rod.radius,
            rod.density,
            rod.num_segments,
        )
        assert rod.stretch_stiffness == pytest.approx(k_exp)
        assert rod.stretch_damping == pytest.approx(c_exp)
    assert p0.primary.bend_stiffness != pytest.approx(p1.primary.bend_stiffness)


def test_params_from_ranges_median_honors_vbd_stretch_force():
    from apple_pick_sim.digital_twin.from_obs import params_from_ranges_median

    fs = _import_module()
    import copy
    import json
    import tempfile

    ranges = copy.deepcopy(fs.load_ranges(RANGES_FIXTURE))
    ranges["primary"]["vbd_stretch_force"] = {
        "max_force_n": 35.0,
        "damping_ratio": 1.0,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(ranges, f)
        path = f.name
    loaded = fs.load_ranges(path)
    params = params_from_ranges_median(loaded)
    assert params.primary is not None
    k_exp, c_exp = fs.stretch_knobs_from_max_force(
        35.0,
        1.0,
        params.primary.length,
        params.primary.radius,
        params.primary.density,
        params.primary.num_segments,
    )
    assert params.primary.stretch_stiffness == pytest.approx(k_exp)
    assert params.primary.stretch_damping == pytest.approx(c_exp)


# ---------------------------------------------------------------------------
# Parameter sampling determinism
# ---------------------------------------------------------------------------


def test_sample_params_deterministic():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    p1 = fs.sample_params(ranges, seed=42)
    p2 = fs.sample_params(ranges, seed=42)
    fp1 = fs.params_fingerprint(p1)
    fp2 = fs.params_fingerprint(p2)
    assert fp1 == fp2, "Same seed must produce identical params"


def test_fruiting_params_json_roundtrip_preserves_sampled_values():
    """Exact sampled params must survive metadata JSON without fingerprint rounding."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    params = fs.sample_params(ranges, seed=42)

    encoded = fs.fruiting_params_to_json(params)
    decoded = fs.fruiting_params_from_json(encoded)

    assert dataclasses.asdict(decoded) == dataclasses.asdict(params)


def test_sample_params_populates_all_segments_when_ranges_non_null():
    """Regression: sample_params must return every segment present in the range dict."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    p = fs.sample_params(ranges, seed=0)
    assert p.primary is not None
    assert p.secondary is not None
    assert p.spur is not None
    assert p.stem is not None
    assert p.apple_radius is not None and p.apple_density is not None


def test_sample_params_omit_unknown_key_raises():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    with pytest.raises(ValueError, match="unknown keys"):
        fs.sample_params(ranges, seed=0, omit=("secondary", "not_a_segment"))


def test_sample_params_omit_all_but_primary():
    """Programmatically skip segments (same effect as JSON null) without editing ranges."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    p = fs.sample_params(
        ranges,
        seed=7,
        omit=frozenset({"secondary", "spur", "stem", "apple"}),
    )
    assert p.primary is not None
    assert p.secondary is None and p.spur is None and p.stem is None
    assert p.apple_radius is None and p.apple_density is None


def test_sample_params_omit_secondary_matches_json_null_secondary():
    """omit={'secondary'} must match sampling when secondary range is JSON null (parent_dir for spur)."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    p_omit = fs.sample_params(ranges, seed=11, omit=frozenset({"secondary"}))
    ranges_no_sec = dict(ranges)
    ranges_no_sec["secondary"] = None
    p_json = fs.sample_params(ranges_no_sec, seed=11)
    assert p_omit.secondary is None and p_json.secondary is None
    assert p_omit.spur is not None and p_json.spur is not None
    assert p_omit.spur.direction == p_json.spur.direction
    assert p_omit.stem.direction == p_json.stem.direction


def test_generate_scene_forwards_omit():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_scene(
        ranges, seed=3, omit=frozenset({"apple"}), **NO_SELF_COLLISION_KW
    )
    assert scene.apple_body is None
    assert scene.params.apple_radius is None


def test_sample_params_omit_all_rods_raises():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    with pytest.raises(ValueError, match="At least one rod segment"):
        fs.sample_params(
            ranges,
            seed=0,
            omit=frozenset({"primary", "secondary", "spur", "stem"}),
        )


def test_sample_params_varies_with_seed():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    p1 = fs.sample_params(ranges, seed=1)
    p2 = fs.sample_params(ranges, seed=2)
    fp1 = fs.params_fingerprint(p1)
    fp2 = fs.params_fingerprint(p2)
    assert fp1 != fp2, "Different seeds must produce distinct params"


def test_primary_stiffer_than_secondary():
    """Primary Young's modulus must be >= secondary when both segments are enabled."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    for seed in range(20):
        params = fs.sample_params(ranges, seed=seed)
        if params.primary is None or params.secondary is None:
            continue
        assert params.primary.youngs_modulus_pa >= params.secondary.youngs_modulus_pa, (
            f"seed={seed}: primary.youngs_modulus_pa ({params.primary.youngs_modulus_pa}) "
            f"< secondary.youngs_modulus_pa ({params.secondary.youngs_modulus_pa})"
        )


def test_params_within_bounds():
    """Sampled scalar parameters must lie within the declared min/max bounds."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    for seed in range(10):
        params = fs.sample_params(ranges, seed=seed)
        for seg_name in ("primary", "secondary", "spur", "stem"):
            seg_params = getattr(params, seg_name)
            seg_ranges = ranges.get(seg_name)
            if seg_params is None or seg_ranges is None:
                continue
            for attr in ("length", "radius", "youngs_modulus_pa", "damping_ratio", "density"):
                v = getattr(seg_params, attr)
                lo = seg_ranges[attr]["min"]
                hi = seg_ranges[attr]["max"]
                if (
                    seg_name == "secondary"
                    and attr == "youngs_modulus_pa"
                    and params.primary is not None
                ):
                    hi = min(hi, params.primary.youngs_modulus_pa)
                    lo = min(lo, hi)
                assert lo <= v <= hi, (
                    f"seed={seed}: {seg_name}.{attr}={v} out of [{lo}, {hi}]"
                )
        # Apple
        if params.apple_radius is not None and ranges.get("apple") is not None:
            assert (
                ranges["apple"]["radius"]["min"]
                <= params.apple_radius
                <= ranges["apple"]["radius"]["max"]
            )
            assert params.apple_density is not None
            assert (
                ranges["apple"]["density"]["min"]
                <= params.apple_density
                <= ranges["apple"]["density"]["max"]
            )


# ---------------------------------------------------------------------------
# Scene generation + body count
# ---------------------------------------------------------------------------


def test_generate_scene_returns_scene():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_scene(ranges, seed=0, **NO_SELF_COLLISION_KW)
    assert scene is not None
    assert scene.model is not None


REAL_WORLD_PROXY_FIXTURE = FIXTURES_DIR / "fruiting_system_ranges_real_world_proxy.json"


def _shape_pairs_filtered(model, body_a: int, body_b: int) -> bool:
    """Return True if every shape pair between two bodies is collision-filtered."""
    shapes_a = model.body_shapes.get(body_a, [])
    shapes_b = model.body_shapes.get(body_b, [])
    if not shapes_a or not shapes_b:
        return False
    pairs = set(model.shape_collision_filter_pairs)
    for s1 in shapes_a:
        for s2 in shapes_b:
            if (s1, s2) in pairs or (s2, s1) in pairs:
                return True
    return False


def test_default_collision_filters_self_and_within_chain_only():
    """Self and within-chain cable pairs filtered; apple/proxy can contact the chain."""
    fs = _import_module()
    ranges = fs.load_ranges(REAL_WORLD_PROXY_FIXTURE)
    scene = fs.generate_scene(ranges, seed=42, device="cpu", enable_self_collisions=False)
    assert scene.apple_body is not None
    assert scene.primary_bodies
    assert scene.spur_bodies
    assert scene.stem_bodies

    model = scene.model
    apple = scene.apple_body
    primary = scene.primary_bodies[0]
    spur = scene.spur_bodies[0]
    stem = scene.stem_bodies[0]

    # Case 1: self-collision within segment groups.
    assert _shape_pairs_filtered(model, primary, spur)

    # Case 2: within-chain cross-segment (stem ↔ woody).
    assert _shape_pairs_filtered(model, stem, primary)
    assert _shape_pairs_filtered(model, stem, spur)

    # Case 3: apple ↔ chain — collidable.
    assert not _shape_pairs_filtered(model, apple, primary)
    assert not _shape_pairs_filtered(model, apple, spur)
    assert not _shape_pairs_filtered(model, apple, stem)


def test_generate_scene_self_collision_off_by_default():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene_default = fs.generate_scene(ranges, seed=0)
    scene_off = fs.generate_scene(ranges, seed=0, enable_self_collisions=False)
    assert (
        scene_default.model.shape_collision_filter_pairs
        == scene_off.model.shape_collision_filter_pairs
    )


def test_generate_scene_disable_self_collision_superset_of_joint_default_filters():
    """enable_self_collisions=False adds intra-chain filter pairs (see _apply_default_fruiting_collision_filters)."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    seed = 0
    scene_on = fs.generate_scene(ranges, seed=seed, enable_self_collisions=True)
    scene_off = fs.generate_scene(ranges, seed=seed)
    pairs_on = scene_on.model.shape_collision_filter_pairs
    pairs_off = scene_off.model.shape_collision_filter_pairs
    assert pairs_on <= pairs_off
    assert len(pairs_off) > len(pairs_on)


def test_short_rollout_disable_self_collision_no_crash():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_scene(ranges, seed=3, enable_self_collisions=False)
    fs.run_rollout(scene, num_steps=5, sim_substeps=4)
    body_q = scene.state_0.body_q.to("cpu").numpy()
    assert np.isfinite(body_q).all()


def test_body_count_matches_params():
    """Total body count should be sum of all segment bodies plus the apple body when present."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_scene(ranges, seed=7, **NO_SELF_COLLISION_KW)
    expected_body_count = (
        len(scene.primary_bodies)
        + len(scene.secondary_bodies)
        + len(scene.spur_bodies)
        + len(scene.stem_bodies)
        + (1 if scene.apple_body is not None else 0)
    )
    # Model may have additional kinematic ground body; check at-least equality
    assert scene.model.body_count >= expected_body_count


def test_body_counts_in_range():
    """Each segment's body list must have the sampled num_segments count."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    for seed in range(5):
        scene = fs.generate_scene(ranges, seed=seed, **NO_SELF_COLLISION_KW)
        params = scene.params
        if params.primary is not None:
            assert len(scene.primary_bodies) == params.primary.num_segments
        else:
            assert scene.primary_bodies == []
        if params.secondary is not None:
            assert len(scene.secondary_bodies) == params.secondary.num_segments
        else:
            assert scene.secondary_bodies == []
        if params.spur is not None:
            assert len(scene.spur_bodies) == params.spur.num_segments
        else:
            assert scene.spur_bodies == []
        if params.stem is not None:
            assert len(scene.stem_bodies) == params.stem.num_segments
        else:
            assert scene.stem_bodies == []


def test_apple_joint_anchor_offset_from_com_by_radius():
    """Stem–apple fixed joint attaches at the sphere pole (one radius from COM), not at COM."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)

    def _row_transform(row: np.ndarray) -> wp.transform:
        return wp.transform(
            wp.vec3(float(row[0]), float(row[1]), float(row[2])),
            wp.quat(float(row[3]), float(row[4]), float(row[5]), float(row[6])),
        )

    for seed in (0, 3, 7):
        scene = fs.generate_scene(ranges, seed=seed, **NO_SELF_COLLISION_KW)
        if scene.apple_body is None:
            continue
        labels = scene.model.joint_label
        ji = next(i for i, lab in enumerate(labels) if lab.endswith("_apple"))
        jc = int(scene.model.joint_child.numpy()[ji])
        assert jc == scene.apple_body
        Xc = _row_transform(scene.model.joint_X_c.numpy()[ji])
        T_apple = _row_transform(scene.state_0.body_q.numpy()[jc])
        joint_world = wp.mul(T_apple, Xc)
        anchor = wp.transform_get_translation(joint_world)
        com = wp.vec3(*scene.state_0.body_q.numpy()[jc, :3].tolist())
        dist = float(wp.length(anchor - com))
        R = float(scene.params.apple_radius)
        assert abs(dist - R) < 5e-5, f"seed={seed}: |anchor−COM|={dist} expected {R}"


def test_fixed_joint_anchors_world_rod_rod():
    """Inter-rod fixed joints report tip/base anchors, not body COM."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_scene(ranges, seed=0, **NO_SELF_COLLISION_KW)

    rod_rod = next(
        (pair for pair in scene.fruiting_fixed_joints if pair[1].endswith("_apple") is False),
        None,
    )
    assert rod_rod is not None
    ji, _label = rod_rod
    jparent = scene.model.joint_parent.numpy()
    jchild = scene.model.joint_child.numpy()
    parent = int(jparent[ji])
    child = int(jchild[ji])
    assert parent >= 0

    parent_anchors, child_anchors = fs.fixed_joint_anchors_world(
        scene.model,
        scene.state_0.body_q,
        scene.fruiting_fixed_joints,
    )
    idx = next(i for i, (j, _) in enumerate(scene.fruiting_fixed_joints) if j == ji)
    anchor_p = parent_anchors[idx * 3 : (idx + 1) * 3]
    anchor_c = child_anchors[idx * 3 : (idx + 1) * 3]
    bq = scene.state_0.body_q.numpy().reshape(-1, 7)
    com_p = bq[parent, :3]
    com_c = bq[child, :3]

    np.testing.assert_allclose(anchor_p, anchor_c, rtol=0.0, atol=1e-5)
    assert float(np.linalg.norm(anchor_p - com_p)) > 1e-4
    # Downstream rod base anchor is local (0,0,0), so child COM may coincide at the junction.


def test_fixed_joint_anchors_world_apple_pole():
    """Stem–apple child anchor is one apple radius from COM."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_scene(ranges, seed=3, **NO_SELF_COLLISION_KW)
    if scene.apple_body is None:
        pytest.skip("fixture produced no apple")

    _, child_anchors = fs.fixed_joint_anchors_world(
        scene.model,
        scene.state_0.body_q,
        scene.fruiting_fixed_joints,
    )
    apple_idx = next(
        i for i, (_, lab) in enumerate(scene.fruiting_fixed_joints) if lab.endswith("_apple")
    )
    anchor_c = child_anchors[apple_idx * 3 : (apple_idx + 1) * 3]
    com = scene.state_0.body_q.numpy()[scene.apple_body, :3]
    dist = float(np.linalg.norm(anchor_c - com))
    assert abs(dist - float(scene.params.apple_radius)) < 5e-5


def test_fixed_joint_anchors_world_order():
    """Anchor arrays follow ``fruiting_fixed_joints`` order."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_scene(ranges, seed=7, **NO_SELF_COLLISION_KW)
    n = len(scene.fruiting_fixed_joints)
    parent_anchors, child_anchors = fs.fixed_joint_anchors_world(
        scene.model,
        scene.state_0.body_q,
        scene.fruiting_fixed_joints,
    )
    assert parent_anchors.shape == (n * 3,)
    assert child_anchors.shape == (n * 3,)
    assert parent_anchors.dtype == np.float32
    assert child_anchors.dtype == np.float32


# ---------------------------------------------------------------------------
# Geometry fingerprint stability and variance
# ---------------------------------------------------------------------------


def test_geometry_fingerprint_stable_same_seed():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    fp1 = fs.geometry_fingerprint(
        fs.generate_scene(ranges, seed=99, **NO_SELF_COLLISION_KW)
    )
    fp2 = fs.geometry_fingerprint(
        fs.generate_scene(ranges, seed=99, **NO_SELF_COLLISION_KW)
    )
    assert fp1 == fp2, "Geometry fingerprint must be identical for the same seed"


def test_geometry_fingerprint_varies_across_seeds():
    """At least one fingerprint value must differ across three distinct seeds."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    fps = [
        fs.geometry_fingerprint(
            fs.generate_scene(ranges, seed=s, **NO_SELF_COLLISION_KW)
        )
        for s in (0, 1, 2)
    ]
    # Not all fingerprints should be equal to each other
    assert not (fps[0] == fps[1] and fps[1] == fps[2]), (
        "All three seeds produced identical geometry fingerprints — variance not working"
    )


def test_fingerprint_primary_stiffer_than_secondary():
    """Fingerprint must reflect the stiffness ordering invariant when both exist."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    for seed in range(10):
        fp = fs.geometry_fingerprint(
            fs.generate_scene(ranges, seed=seed, **NO_SELF_COLLISION_KW)
        )
        pe, se = fp.get("primary_youngs_modulus_pa"), fp.get("secondary_youngs_modulus_pa")
        if pe is None or se is None:
            continue
        assert pe >= se, f"seed={seed}: fingerprint E ordering violated"


# ---------------------------------------------------------------------------
# Rollout (headless Newton simulation)
# ---------------------------------------------------------------------------


def test_short_rollout_no_crash():
    """A short SolverVBD rollout must complete without raising and produce finite transforms."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_scene(ranges, seed=3, **NO_SELF_COLLISION_KW)
    fs.run_rollout(scene, num_steps=5, sim_substeps=4)
    body_q = scene.state_0.body_q.to("cpu").numpy()
    assert np.isfinite(body_q).all(), "Non-finite body transforms after rollout"


def test_rollout_finite_primary_secondary_spur_stem_apple():
    """Full topology including stem + apple must stay numerically stable across seeds."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    for seed in range(12):
        scene = fs.generate_scene(ranges, seed=seed, **NO_SELF_COLLISION_KW)
        assert scene.stem_bodies, "fixture should include stem"
        assert scene.apple_body is not None, "fixture should include apple"
        fs.run_rollout(scene, num_steps=12, sim_substeps=6)
        body_q = scene.state_0.body_q.to("cpu").numpy()
        assert np.isfinite(body_q).all(), f"non-finite body_q after rollout seed={seed}"


def test_rollout_deterministic():
    """Two scenes with the same seed must reach the same state after N rollout steps."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene_a = fs.generate_scene(ranges, seed=5, **NO_SELF_COLLISION_KW)
    scene_b = fs.generate_scene(ranges, seed=5, **NO_SELF_COLLISION_KW)
    fs.run_rollout(scene_a, num_steps=10, sim_substeps=4)
    fs.run_rollout(scene_b, num_steps=10, sim_substeps=4)
    q_a = scene_a.state_0.body_q.to("cpu").numpy()
    q_b = scene_b.state_0.body_q.to("cpu").numpy()
    np.testing.assert_allclose(
        q_a, q_b, atol=1e-5, err_msg="Rollout not deterministic for the same seed"
    )


def test_run_rollout_with_example_collision_pipeline_matches_default():
    """Explicit example collision pipeline must match bare collide for identical rollouts."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    seed = 5
    scene_a = fs.generate_scene(
        ranges, seed=seed, device="cpu", **NO_SELF_COLLISION_KW
    )
    scene_b = fs.generate_scene(
        ranges, seed=seed, device="cpu", **NO_SELF_COLLISION_KW
    )
    pipe = fs.example_collision_pipeline(scene_b.model, args=None)
    fs.run_rollout(scene_a, num_steps=4, sim_substeps=6, fps=60.0)
    fs.run_rollout(
        scene_b,
        num_steps=4,
        sim_substeps=6,
        fps=60.0,
        collision_pipeline=pipe,
    )
    q_a = scene_a.state_0.body_q.to("cpu").numpy()
    q_b = scene_b.state_0.body_q.to("cpu").numpy()
    np.testing.assert_allclose(q_a, q_b, atol=1e-5)


def test_fruiting_fixed_joints_matches_label_heuristic():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_scene(ranges, seed=3, device="cpu", **NO_SELF_COLLISION_KW)
    assert list(scene.fruiting_fixed_joints) == fs.iter_fixed_joint_indices(scene.model)


def _scene_for_joint_kd_tests():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    return fs.generate_scene(ranges, seed=3, device="cpu", **NO_SELF_COLLISION_KW)


def _angular_kd_for_joint(solver, joint_index: int) -> float:
    import newton

    jc_start = solver.joint_constraint_start.numpy()
    kd = solver.joint_penalty_kd.numpy()
    c0 = int(jc_start[joint_index])
    return float(kd[c0 + newton.solvers.SolverVBD.JointSlot.ANGULAR])


def _joint_index_by_label(fruiting_fixed_joints, label_substr: str) -> int:
    matches = [j for j, lab in fruiting_fixed_joints if label_substr in lab]
    assert len(matches) == 1, f"expected one joint for {label_substr!r}, got {matches}"
    return matches[0]


def _run_fruiting_vbd_substeps(scene, *, num_substeps: int, sim_dt: float) -> None:
    fs = _import_module()
    pipe = fs.example_collision_pipeline(scene.model, args=None)
    for _ in range(num_substeps):
        scene.state_0.clear_forces()
        contacts = scene.model.collide(scene.state_0, collision_pipeline=pipe)
        scene.solver.step(
            scene.state_0, scene.state_1, scene.control, contacts, sim_dt
        )
        scene.state_0, scene.state_1 = scene.state_1, scene.state_0


def test_set_fruiting_joint_angular_kd_persists_through_solver_step():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()
    j_stem_apple = _joint_index_by_label(scene.fruiting_fixed_joints, "stem_apple")
    fs.set_fruiting_joint_angular_kd(
        scene.solver,
        scene.fruiting_fixed_joints,
        {"stem_apple": 2.5},
    )
    _run_fruiting_vbd_substeps(scene, num_substeps=8, sim_dt=(1.0 / 60.0) / 10.0)
    assert _angular_kd_for_joint(scene.solver, j_stem_apple) == pytest.approx(2.5)


def test_set_fruiting_joint_angular_kd_changes_trajectory_after_steps():
    fs = _import_module()
    sim_dt = (1.0 / 60.0) / 10.0
    substeps = 120

    scene_default = _scene_for_joint_kd_tests()
    _run_fruiting_vbd_substeps(scene_default, num_substeps=substeps, sim_dt=sim_dt)
    q_default = scene_default.state_0.body_q.numpy().copy()

    scene_patched = _scene_for_joint_kd_tests()
    fs.set_fruiting_joint_angular_kd(
        scene_patched.solver,
        scene_patched.fruiting_fixed_joints,
        {"stem_apple": 50.0},
    )
    _run_fruiting_vbd_substeps(scene_patched, num_substeps=substeps, sim_dt=sim_dt)
    q_patched = scene_patched.state_0.body_q.numpy().copy()

    assert not np.allclose(q_default, q_patched, rtol=0.0, atol=1.0e-4), (
        "patched stem_apple angular kd should change integrated trajectory"
    )


def test_set_fruiting_joint_angular_kd_patches_matching_slots():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()
    j_primary = _joint_index_by_label(scene.fruiting_fixed_joints, "primary_secondary")
    j_stem_apple = _joint_index_by_label(scene.fruiting_fixed_joints, "stem_apple")

    fs.set_fruiting_joint_angular_kd(
        scene.solver,
        scene.fruiting_fixed_joints,
        {"primary_secondary": 2.5, "stem_apple": 0.25},
    )

    assert _angular_kd_for_joint(scene.solver, j_primary) == pytest.approx(2.5)
    assert _angular_kd_for_joint(scene.solver, j_stem_apple) == pytest.approx(0.25)


def test_set_fruiting_joint_angular_kd_leaves_unmatched_joints_at_default():
    from apple_pick_sim.fruiting_system.build import FRUITING_VBD_RIGID_JOINT_ANGULAR_KD

    fs = _import_module()
    scene = _scene_for_joint_kd_tests()
    j_spur_stem = _joint_index_by_label(scene.fruiting_fixed_joints, "spur_stem")
    default_kd = _angular_kd_for_joint(scene.solver, j_spur_stem)
    assert default_kd == pytest.approx(FRUITING_VBD_RIGID_JOINT_ANGULAR_KD)

    fs.set_fruiting_joint_angular_kd(
        scene.solver,
        scene.fruiting_fixed_joints,
        {"primary_secondary": 3.0},
    )

    assert _angular_kd_for_joint(scene.solver, j_spur_stem) == pytest.approx(
        FRUITING_VBD_RIGID_JOINT_ANGULAR_KD
    )


def test_set_fruiting_joint_angular_kd_raises_on_unmatched_key():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()

    with pytest.raises(ValueError, match="nonexistent_key_xyz"):
        fs.set_fruiting_joint_angular_kd(
            scene.solver,
            scene.fruiting_fixed_joints,
            {"nonexistent_key_xyz": 1.0},
        )


def test_set_fruiting_joint_angular_kd_raises_on_ambiguous_multi_key_match():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()

    with pytest.raises(ValueError, match="ambiguous"):
        fs.set_fruiting_joint_angular_kd(
            scene.solver,
            scene.fruiting_fixed_joints,
            {"apple": 0.5, "stem_apple": 0.25},
        )


def test_set_fruiting_joint_angular_kd_raises_on_negative_kd():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()

    with pytest.raises(ValueError, match="negative"):
        fs.set_fruiting_joint_angular_kd(
            scene.solver,
            scene.fruiting_fixed_joints,
            {"stem_apple": -0.1},
        )


def test_set_fruiting_joint_angular_kd_returns_matched_indices_per_key():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()
    j_primary = _joint_index_by_label(scene.fruiting_fixed_joints, "primary_secondary")
    j_stem_apple = _joint_index_by_label(scene.fruiting_fixed_joints, "stem_apple")

    matched = fs.set_fruiting_joint_angular_kd(
        scene.solver,
        scene.fruiting_fixed_joints,
        {"primary_secondary": 2.0, "stem_apple": 0.2},
    )

    assert matched == {"primary_secondary": [j_primary], "stem_apple": [j_stem_apple]}


def _linear_kd_for_joint(solver, joint_index: int) -> float:
    import newton

    jc_start = solver.joint_constraint_start.numpy()
    kd = solver.joint_penalty_kd.numpy()
    c0 = int(jc_start[joint_index])
    return float(kd[c0 + newton.solvers.SolverVBD.JointSlot.LINEAR])


def test_set_fruiting_joint_linear_kd_persists_through_solver_step():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()
    j_stem_apple = _joint_index_by_label(scene.fruiting_fixed_joints, "stem_apple")
    fs.set_fruiting_joint_linear_kd(
        scene.solver,
        scene.fruiting_fixed_joints,
        {"stem_apple": 2.5},
    )
    _run_fruiting_vbd_substeps(scene, num_substeps=8, sim_dt=(1.0 / 60.0) / 10.0)
    assert _linear_kd_for_joint(scene.solver, j_stem_apple) == pytest.approx(2.5)


def test_set_fruiting_joint_linear_kd_changes_trajectory_after_steps():
    fs = _import_module()
    sim_dt = (1.0 / 60.0) / 10.0
    substeps = 120

    scene_default = _scene_for_joint_kd_tests()
    _run_fruiting_vbd_substeps(scene_default, num_substeps=substeps, sim_dt=sim_dt)
    q_default = scene_default.state_0.body_q.numpy().copy()

    scene_patched = _scene_for_joint_kd_tests()
    fs.set_fruiting_joint_linear_kd(
        scene_patched.solver,
        scene_patched.fruiting_fixed_joints,
        {"stem_apple": 50.0},
    )
    _run_fruiting_vbd_substeps(scene_patched, num_substeps=substeps, sim_dt=sim_dt)
    q_patched = scene_patched.state_0.body_q.numpy().copy()

    assert not np.allclose(q_default, q_patched, rtol=0.0, atol=1.0e-4), (
        "patched stem_apple linear kd should change integrated trajectory"
    )


def test_set_fruiting_joint_linear_kd_patches_matching_slots():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()
    j_primary = _joint_index_by_label(scene.fruiting_fixed_joints, "primary_secondary")
    j_stem_apple = _joint_index_by_label(scene.fruiting_fixed_joints, "stem_apple")

    fs.set_fruiting_joint_linear_kd(
        scene.solver,
        scene.fruiting_fixed_joints,
        {"primary_secondary": 2.5, "stem_apple": 0.25},
    )

    assert _linear_kd_for_joint(scene.solver, j_primary) == pytest.approx(2.5)
    assert _linear_kd_for_joint(scene.solver, j_stem_apple) == pytest.approx(0.25)


def test_set_fruiting_joint_linear_kd_leaves_unmatched_joints_at_default():
    from apple_pick_sim.fruiting_system.build import FRUITING_VBD_RIGID_JOINT_LINEAR_KD

    fs = _import_module()
    scene = _scene_for_joint_kd_tests()
    j_spur_stem = _joint_index_by_label(scene.fruiting_fixed_joints, "spur_stem")
    default_kd = _linear_kd_for_joint(scene.solver, j_spur_stem)
    assert default_kd == pytest.approx(FRUITING_VBD_RIGID_JOINT_LINEAR_KD)

    fs.set_fruiting_joint_linear_kd(
        scene.solver,
        scene.fruiting_fixed_joints,
        {"primary_secondary": 3.0},
    )

    assert _linear_kd_for_joint(scene.solver, j_spur_stem) == pytest.approx(
        FRUITING_VBD_RIGID_JOINT_LINEAR_KD
    )


def test_set_fruiting_joint_linear_kd_raises_on_unmatched_key():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()

    with pytest.raises(ValueError, match="nonexistent_key_xyz"):
        fs.set_fruiting_joint_linear_kd(
            scene.solver,
            scene.fruiting_fixed_joints,
            {"nonexistent_key_xyz": 1.0},
        )


def test_set_fruiting_joint_linear_kd_raises_on_ambiguous_multi_key_match():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()

    with pytest.raises(ValueError, match="ambiguous"):
        fs.set_fruiting_joint_linear_kd(
            scene.solver,
            scene.fruiting_fixed_joints,
            {"apple": 0.5, "stem_apple": 0.25},
        )


def test_set_fruiting_joint_linear_kd_raises_on_negative_kd():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()

    with pytest.raises(ValueError, match="negative"):
        fs.set_fruiting_joint_linear_kd(
            scene.solver,
            scene.fruiting_fixed_joints,
            {"stem_apple": -0.1},
        )


def test_set_fruiting_joint_linear_kd_returns_matched_indices_per_key():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()
    j_primary = _joint_index_by_label(scene.fruiting_fixed_joints, "primary_secondary")
    j_stem_apple = _joint_index_by_label(scene.fruiting_fixed_joints, "stem_apple")

    matched = fs.set_fruiting_joint_linear_kd(
        scene.solver,
        scene.fruiting_fixed_joints,
        {"primary_secondary": 2.0, "stem_apple": 0.2},
    )

    assert matched == {"primary_secondary": [j_primary], "stem_apple": [j_stem_apple]}


def _angular_kp_triple_for_joint(solver, joint_index: int) -> tuple[float, float, float]:
    import newton

    jc_start = solver.joint_constraint_start.numpy()
    k = solver.joint_penalty_k.numpy()
    k_min = solver.joint_penalty_k_min.numpy()
    k_max = solver.joint_penalty_k_max.numpy()
    c0 = int(jc_start[joint_index])
    slot = c0 + newton.solvers.SolverVBD.JointSlot.ANGULAR
    return float(k[slot]), float(k_min[slot]), float(k_max[slot])


def test_set_fruiting_joint_angular_kp_persists_through_solver_step():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()
    j_stem_apple = _joint_index_by_label(scene.fruiting_fixed_joints, "stem_apple")
    kp_set = 2.5e5
    substeps = 8
    fs.set_fruiting_joint_angular_kp(
        scene.solver,
        scene.fruiting_fixed_joints,
        {"stem_apple": kp_set},
    )
    _run_fruiting_vbd_substeps(scene, num_substeps=substeps, sim_dt=(1.0 / 60.0) / 10.0)
    k, _k_min, k_max = _angular_kp_triple_for_joint(scene.solver, j_stem_apple)
    expected_k = kp_set * (scene.solver.rigid_avbd_gamma**substeps)
    assert k == pytest.approx(expected_k)
    assert k_max >= kp_set


def test_set_fruiting_joint_angular_kp_changes_trajectory_after_steps():
    fs = _import_module()
    sim_dt = (1.0 / 60.0) / 10.0
    substeps = 120

    scene_default = _scene_for_joint_kd_tests()
    _run_fruiting_vbd_substeps(scene_default, num_substeps=substeps, sim_dt=sim_dt)
    q_default = scene_default.state_0.body_q.numpy().copy()

    scene_patched = _scene_for_joint_kd_tests()
    fs.set_fruiting_joint_angular_kp(
        scene_patched.solver,
        scene_patched.fruiting_fixed_joints,
        {"stem_apple": 1.0e4},
    )
    _run_fruiting_vbd_substeps(scene_patched, num_substeps=substeps, sim_dt=sim_dt)
    q_patched = scene_patched.state_0.body_q.numpy().copy()

    assert not np.allclose(q_default, q_patched, rtol=0.0, atol=1.0e-4), (
        "patched stem_apple angular kp should change integrated trajectory"
    )


def test_set_fruiting_joint_angular_kp_patches_matching_slots():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()
    j_primary = _joint_index_by_label(scene.fruiting_fixed_joints, "primary_secondary")
    j_stem_apple = _joint_index_by_label(scene.fruiting_fixed_joints, "stem_apple")

    fs.set_fruiting_joint_angular_kp(
        scene.solver,
        scene.fruiting_fixed_joints,
        {"primary_secondary": 2.0e5, "stem_apple": 5.0e4},
    )

    k_primary, _, kmax_primary = _angular_kp_triple_for_joint(scene.solver, j_primary)
    k_stem, _, kmax_stem = _angular_kp_triple_for_joint(scene.solver, j_stem_apple)
    assert k_primary == pytest.approx(2.0e5)
    assert k_stem == pytest.approx(5.0e4)
    assert kmax_primary >= 2.0e5
    assert kmax_stem >= 5.0e4


def test_set_fruiting_joint_angular_kp_leaves_unmatched_joints_at_default():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()
    j_spur_stem = _joint_index_by_label(scene.fruiting_fixed_joints, "spur_stem")
    default_k, _, _ = _angular_kp_triple_for_joint(scene.solver, j_spur_stem)

    fs.set_fruiting_joint_angular_kp(
        scene.solver,
        scene.fruiting_fixed_joints,
        {"primary_secondary": 2.0e5},
    )

    k_spur, _, _ = _angular_kp_triple_for_joint(scene.solver, j_spur_stem)
    assert k_spur == pytest.approx(default_k)


def test_set_fruiting_joint_angular_kp_raises_on_unmatched_key():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()

    with pytest.raises(ValueError, match="nonexistent_key_xyz"):
        fs.set_fruiting_joint_angular_kp(
            scene.solver,
            scene.fruiting_fixed_joints,
            {"nonexistent_key_xyz": 1.0e5},
        )


def test_set_fruiting_joint_angular_kp_raises_on_ambiguous_multi_key_match():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()

    with pytest.raises(ValueError, match="ambiguous"):
        fs.set_fruiting_joint_angular_kp(
            scene.solver,
            scene.fruiting_fixed_joints,
            {"apple": 1.0e5, "stem_apple": 5.0e4},
        )


def test_set_fruiting_joint_angular_kp_raises_on_negative_kp():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()

    with pytest.raises(ValueError, match="negative"):
        fs.set_fruiting_joint_angular_kp(
            scene.solver,
            scene.fruiting_fixed_joints,
            {"stem_apple": -1.0},
        )


def test_set_fruiting_joint_angular_kp_returns_matched_indices_per_key():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()
    j_primary = _joint_index_by_label(scene.fruiting_fixed_joints, "primary_secondary")
    j_stem_apple = _joint_index_by_label(scene.fruiting_fixed_joints, "stem_apple")

    matched = fs.set_fruiting_joint_angular_kp(
        scene.solver,
        scene.fruiting_fixed_joints,
        {"primary_secondary": 2.0e5, "stem_apple": 5.0e4},
    )

    assert matched == {"primary_secondary": [j_primary], "stem_apple": [j_stem_apple]}


def _linear_kp_triple_for_joint(solver, joint_index: int) -> tuple[float, float, float]:
    import newton

    jc_start = solver.joint_constraint_start.numpy()
    k = solver.joint_penalty_k.numpy()
    k_min = solver.joint_penalty_k_min.numpy()
    k_max = solver.joint_penalty_k_max.numpy()
    c0 = int(jc_start[joint_index])
    slot = c0 + newton.solvers.SolverVBD.JointSlot.LINEAR
    return float(k[slot]), float(k_min[slot]), float(k_max[slot])


def test_set_fruiting_joint_linear_kp_patches_matching_slots():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()
    j_primary = _joint_index_by_label(scene.fruiting_fixed_joints, "primary_secondary")
    j_stem_apple = _joint_index_by_label(scene.fruiting_fixed_joints, "stem_apple")

    fs.set_fruiting_joint_linear_kp(
        scene.solver,
        scene.fruiting_fixed_joints,
        {"primary_secondary": 3.0e5, "stem_apple": 6.0e4},
    )

    k_primary, _, kmax_primary = _linear_kp_triple_for_joint(scene.solver, j_primary)
    k_stem, _, kmax_stem = _linear_kp_triple_for_joint(scene.solver, j_stem_apple)
    assert k_primary == pytest.approx(3.0e5)
    assert k_stem == pytest.approx(6.0e4)
    assert kmax_primary >= 3.0e5
    assert kmax_stem >= 6.0e4


def test_set_fruiting_joint_linear_kp_leaves_unmatched_joints_at_default():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()
    j_spur_stem = _joint_index_by_label(scene.fruiting_fixed_joints, "spur_stem")
    default_k, _, _ = _linear_kp_triple_for_joint(scene.solver, j_spur_stem)

    fs.set_fruiting_joint_linear_kp(
        scene.solver,
        scene.fruiting_fixed_joints,
        {"primary_secondary": 3.0e5},
    )

    k_spur, _, _ = _linear_kp_triple_for_joint(scene.solver, j_spur_stem)
    assert k_spur == pytest.approx(default_k)


def test_set_fruiting_joint_linear_kp_raises_on_unmatched_key():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()

    with pytest.raises(ValueError, match="nonexistent_key_xyz"):
        fs.set_fruiting_joint_linear_kp(
            scene.solver,
            scene.fruiting_fixed_joints,
            {"nonexistent_key_xyz": 1.0e5},
        )


def test_set_fruiting_joint_linear_kp_raises_on_ambiguous_multi_key_match():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()

    with pytest.raises(ValueError, match="ambiguous"):
        fs.set_fruiting_joint_linear_kp(
            scene.solver,
            scene.fruiting_fixed_joints,
            {"apple": 1.0e5, "stem_apple": 5.0e4},
        )


def test_set_fruiting_joint_linear_kp_raises_on_negative_kp():
    fs = _import_module()
    scene = _scene_for_joint_kd_tests()

    with pytest.raises(ValueError, match="negative"):
        fs.set_fruiting_joint_linear_kp(
            scene.solver,
            scene.fruiting_fixed_joints,
            {"stem_apple": -1.0},
        )


def test_measure_fruiting_forces_returns_fixed_and_cable_indices():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_scene(ranges, seed=3, device="cpu", **NO_SELF_COLLISION_KW)
    sim_dt = (1.0 / 60.0) / 10.0
    scene.state_0.clear_forces()
    pipe = fs.example_collision_pipeline(scene.model, args=None)
    contacts = scene.model.collide(scene.state_0, collision_pipeline=pipe)
    q_prev = scene.state_0.body_q.numpy().copy()
    scene.solver.step(scene.state_0, scene.state_1, scene.control, contacts, sim_dt)
    scene.state_0, scene.state_1 = scene.state_1, scene.state_0
    out = fs.measure_fruiting_forces(
        scene,
        scene.state_0.body_q.numpy(),
        body_q_prev=q_prev,
        dt=sim_dt,
    )
    assert "fixed_joints" in out and "cable_joint_indices" in out
    assert len(out["fixed_joints"]) == len(scene.fruiting_fixed_joints)
    assert len(out["cable_joint_indices"]) > 0


def test_ranges_null_secondary_loads_and_scene_skips_secondary():
    """JSON null for a rod segment omits that piece from params and the built scene."""
    fs = _import_module()
    ranges = dict(fs.load_ranges(RANGES_FIXTURE))
    ranges["secondary"] = None
    params = fs.sample_params(ranges, seed=11)
    assert params.secondary is None
    assert params.primary is not None
    scene = fs.generate_scene(ranges, seed=11, **NO_SELF_COLLISION_KW)
    assert scene.secondary_bodies == []
    assert len(scene.primary_bodies) == params.primary.num_segments
    assert len(scene.spur_bodies) == params.spur.num_segments


def test_ranges_null_apple_skips_apple_body():
    fs = _import_module()
    ranges = dict(fs.load_ranges(RANGES_FIXTURE))
    ranges["apple"] = None
    params = fs.sample_params(ranges, seed=2)
    assert params.apple_radius is None and params.apple_density is None
    scene = fs.generate_scene(ranges, seed=2, **NO_SELF_COLLISION_KW)
    assert scene.apple_body is None


def test_manual_params_skips_primary():
    fs = _import_module()
    full = fs.load_ranges(RANGES_FIXTURE)
    p = fs.sample_params(full, seed=0)
    p = dataclasses.replace(p, primary=None)
    scene = fs._build_scene(p, base_pos=(0.0, 0.0, 3.0), device="cpu")
    assert scene.primary_bodies == []
    assert scene.secondary_bodies
    fp = fs.geometry_fingerprint(scene)
    assert fp["primary_base_pos"] is None
    assert fp["apple_pos"] is not None


# ---------------------------------------------------------------------------
# Fixed-joint wrenches (SolverVBD)
# ---------------------------------------------------------------------------


def test_iter_fixed_joint_indices_full_fixture():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_scene(ranges, seed=3, device="cpu", **NO_SELF_COLLISION_KW)
    pairs = fs.iter_fixed_joint_indices(scene.model)
    labels = [lab for _, lab in pairs]
    assert len(pairs) == 4
    assert "joint_primary_secondary" in labels
    assert "joint_secondary_spur" in labels
    assert "joint_spur_stem" in labels
    assert "joint_stem_apple" in labels


def test_iter_fixed_joint_indices_omit_apple():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_scene(
        ranges,
        seed=3,
        device="cpu",
        omit=frozenset({"apple"}),
        **NO_SELF_COLLISION_KW,
    )
    labels = [lab for _, lab in fs.iter_fixed_joint_indices(scene.model)]
    assert len(labels) == 3
    assert "joint_stem_apple" not in labels


def test_fixed_joint_wrenches_finite_after_substep():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_scene(ranges, seed=3, device="cpu", **NO_SELF_COLLISION_KW)
    sim_dt = (1.0 / 60.0) / 10.0
    scene.state_0.clear_forces()
    pipe = fs.example_collision_pipeline(scene.model, args=None)
    contacts = scene.model.collide(scene.state_0, collision_pipeline=pipe)
    q_prev = scene.state_0.body_q.numpy().copy()
    scene.solver.step(scene.state_0, scene.state_1, scene.control, contacts, sim_dt)
    scene.state_0, scene.state_1 = scene.state_1, scene.state_0
    wrenches = fs.fixed_joint_wrenches_child_com_vbd(
        scene.model,
        scene.solver,
        body_q=scene.state_0.body_q.numpy(),
        body_q_prev=q_prev,
        dt=sim_dt,
        joint_pairs=list(scene.fruiting_fixed_joints),
    )
    assert len(wrenches) == 4
    for w in wrenches:
        assert np.isfinite(w.force_world).all()
        assert np.isfinite(w.torque_at_child_com_world).all()


def test_perturb_rod_stiffness_rejects_disabled_segment():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    p = fs.sample_params(ranges, seed=0, omit=frozenset({"secondary"}))
    assert p.secondary is None
    with pytest.raises(ValueError, match="disabled"):
        fs.perturb_rod_stiffness(p, "secondary", bend_delta=1.0)


def test_perturb_rod_stiffness_rejects_nonpositive_result():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    p = fs.sample_params(ranges, seed=0)
    with pytest.raises(ValueError, match="positive"):
        fs.perturb_rod_stiffness(p, "stem", bend_delta=-1.0e9)


def test_set_rod_bend_stiffness_sets_absolute_value():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    p = fs.sample_params(ranges, seed=0)
    target = 123.45
    out = fs.set_rod_bend_stiffness(p, "primary", target)
    assert out.primary is not None
    assert out.primary.bend_stiffness == pytest.approx(target)
    assert p.primary.bend_stiffness != pytest.approx(target)
    assert out.secondary == p.secondary


def test_set_rod_bend_stiffness_rejects_nonpositive():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    p = fs.sample_params(ranges, seed=0)
    with pytest.raises(ValueError, match="positive"):
        fs.set_rod_bend_stiffness(p, "primary", 0.0)


def test_set_rod_bend_stiffness_preserves_damping_ratio():
    fs = _import_module()
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
    old_k = rod.bend_stiffness
    new_k = old_k * 100.0
    out = fs.set_rod_bend_stiffness(base, "primary", new_k)
    assert out.primary is not None
    assert out.primary.bend_stiffness == pytest.approx(new_k)
    assert out.primary.damping_ratio == pytest.approx(rod.damping_ratio)
    assert out.primary.bend_damping == pytest.approx(
        rod.bend_damping * math.sqrt(new_k / old_k)
    )


def test_variance_fixture_loads_and_samples_in_bounds():
    """Wide example-variance JSON is valid and produces in-range params."""
    fs = _import_module()
    assert VARIANCE_FIXTURE.exists()
    ranges = fs.load_ranges(VARIANCE_FIXTURE)
    for seg in ("primary", "secondary", "spur", "stem", "apple"):
        assert seg in ranges
    params = fs.sample_params(ranges, seed=17)
    assert params.primary is not None and params.secondary is not None
    for seg_name in ("primary", "secondary", "spur", "stem"):
        seg_params = getattr(params, seg_name)
        seg_ranges = ranges[seg_name]
        for attr in ("length", "radius", "youngs_modulus_pa", "damping_ratio", "density"):
            v = getattr(seg_params, attr)
            lo = seg_ranges[attr]["min"]
            hi = seg_ranges[attr]["max"]
            assert lo <= v <= hi, f"{seg_name}.{attr}={v} not in [{lo}, {hi}]"
    assert params.apple_radius is not None
    assert (
        ranges["apple"]["radius"]["min"]
        <= params.apple_radius
        <= ranges["apple"]["radius"]["max"]
    )


def test_fix_to_apple_requires_apple_body():
    """Welded proxy build must fail when the fruiting tree has no apple."""
    fs = _import_module()
    ranges = dict(fs.load_ranges(RANGES_FIXTURE))
    ranges["apple"] = None
    with pytest.raises(ValueError, match="fix_to_apple requires an apple"):
        fs.generate_coupled_cable_scene(
            ranges,
            seed=0,
            gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True),
            **NO_SELF_COLLISION_KW,
        )


def test_stem_apple_fixed_joint_child_is_apple_proxy_joint_is_not():
    """Stem→apple and proxy→apple joints are distinct; only stem attaches to the apple body."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_coupled_cable_scene(
        ranges,
        seed=4,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True),
        **NO_SELF_COLLISION_KW,
    )
    assert scene.apple_body is not None
    assert scene.gripper_proxy_apple_joint is not None
    jchild = scene.model.joint_child.numpy()
    assert int(jchild[scene.gripper_proxy_apple_joint]) == scene.gripper_proxy_body
    from apple_pick_sim.coupled_fruiting.stem import _find_stem_apple_joint

    stem_j = _find_stem_apple_joint(scene)
    assert stem_j is not None
    assert stem_j != scene.gripper_proxy_apple_joint
    assert int(jchild[stem_j]) == scene.apple_body


def test_measure_fruiting_forces_state1_matches_solver_body_q_prev():
    """After swap, VBD ``solver.body_q_prev`` is end-of-step (``state_0.body_q``); wrenches use ``state_1.body_q`` as pre-step."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_coupled_cable_scene(ranges, seed=3, device="cpu", **NO_SELF_COLLISION_KW)
    sim_dt = (1.0 / 60.0) / 10.0
    scene.state_0.clear_forces()
    pipe = fs.example_collision_pipeline(scene.model, args=None)
    contacts = scene.model.collide(scene.state_0, collision_pipeline=pipe)
    scene.solver.step(scene.state_0, scene.state_1, scene.control, contacts, sim_dt)
    scene.state_0, scene.state_1 = scene.state_1, scene.state_0

    bq0 = scene.state_0.body_q.numpy().reshape(-1, 7)
    bq1 = scene.state_1.body_q.numpy().reshape(-1, 7)
    bqp = scene.solver.body_q_prev.numpy().reshape(-1, 7)
    np.testing.assert_allclose(
        bq0,
        bqp,
        rtol=0.0,
        atol=0.0,
        err_msg="state_0.body_q must match solver.body_q_prev after step (end-of-step pose)",
    )
    assert not np.allclose(bq1, bqp, rtol=0.0, atol=0.0), (
        "pre-step state_1.body_q must not be reused as solver.body_q_prev after step"
    )
    out = fs.measure_fruiting_forces(
        scene, scene.state_0.body_q, scene.state_1.body_q, dt=sim_dt
    )
    out_bqp = fs.measure_fruiting_forces(
        scene, scene.state_0.body_q, scene.solver.body_q_prev, dt=sim_dt
    )
    assert len(out["fixed_joints"]) == len(scene.fruiting_fixed_joints)
    f0 = out["fixed_joints"][0].force_world
    f1 = out_bqp["fixed_joints"][0].force_world
    np.testing.assert_allclose(f0, f1, rtol=0.0, atol=0.0)


def test_example_fruiting_system_regenerate_uses_self_collision_parser_default():
    from apple_pick_sim.examples import example_fruiting_system as ex

    class _ViewerStub:
        def set_model(self, model):
            del model

    args = ex._make_parser().parse_args(["--seed", "0", "--viewer", "null"])
    example = ex.ExampleFruitingSystem(_ViewerStub(), args)
    pairs = len(example.model.shape_collision_filter_pairs)
    fs = _import_module()
    scene_off = fs.generate_scene(
        example.ranges, 0, device=example._device_str(), enable_self_collisions=False
    )
    assert pairs == len(scene_off.model.shape_collision_filter_pairs)


def test_example_fruiting_system_default_ranges_path_is_real_world_proxy_variance():
    from apple_pick_sim.examples import example_fruiting_system as ex
    from apple_pick_sim.fruiting_system import default_ranges_fixture_path

    fs = _import_module()
    path = ex._default_ranges_path()
    assert path == default_ranges_fixture_path()
    assert path.name == "fruiting_system_ranges_real_world_proxy_variance.json"
    ranges = fs.load_ranges(path)
    assert ranges["secondary"] is None
    assert fs.parse_fixture_args(ranges).fruiting_base_pos == pytest.approx((0.0, 0.4, 0.75))


def test_example_coupled_fruiting_default_ranges_path_is_real_world_proxy_variance():
    from apple_pick_sim.examples import example_coupled_fruiting as ex
    from apple_pick_sim.fruiting_system import default_ranges_fixture_path

    assert ex._default_ranges_path() == default_ranges_fixture_path()


def test_example_batched_heterogeneous_coupled_sim_default_ranges_path_is_real_world_proxy_variance():
    from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
        BatchedHeterogeneousCoupledSimConfig,
    )
    from apple_pick_sim.fruiting_system import default_ranges_fixture_path

    cfg = BatchedHeterogeneousCoupledSimConfig.defaults()
    assert cfg.domain_randomization.resolved_ranges_path() == default_ranges_fixture_path()


def test_example_fruiting_system_enable_self_collision_parser_default():
    from apple_pick_sim.examples import example_fruiting_system as ex

    args = ex._make_parser().parse_args([])
    assert ex._enable_self_collisions_from_args(args) is False


def test_example_fruiting_system_enable_self_collision_parser_enabled():
    from apple_pick_sim.examples import example_fruiting_system as ex

    args = ex._make_parser().parse_args(["--enable-self-collision"])
    assert ex._enable_self_collisions_from_args(args) is True


# ---------------------------------------------------------------------------
# Directional overlap check
# ---------------------------------------------------------------------------


def _rod(fs, direction: tuple[float, float, float]):
    return fs.rod_params_from_material(
        youngs_modulus_pa=1.0e7,
        damping_ratio=0.05,
        length=0.10,
        radius=0.01,
        density=300.0,
        num_segments=3,
        direction=direction,
    )


def _t_junction_params(fs, *, primary_dir, spur_dir, stem_dir):
    return fs.FruitingSystemParams(
        primary=_rod(fs, primary_dir),
        secondary=None,
        spur=_rod(fs, spur_dir),
        stem=_rod(fs, stem_dir),
        apple_radius=0.04,
        apple_density=400.0,
        topology=fs.TOPOLOGY_T_JUNCTION,
    )


def _linear_chain_params(fs, *, primary_dir, spur_dir, stem_dir):
    return fs.FruitingSystemParams(
        primary=_rod(fs, primary_dir),
        secondary=None,
        spur=_rod(fs, spur_dir),
        stem=_rod(fs, stem_dir),
        apple_radius=0.04,
        apple_density=400.0,
        topology=fs.TOPOLOGY_LINEAR_CHAIN,
    )


def test_overlap_parallel_spur_primary_t_junction():
    fs = _import_module()
    params = _t_junction_params(
        fs,
        primary_dir=(1.0, 0.0, 0.0),
        spur_dir=(1.0, 0.0, 0.0),
        stem_dir=(0.0, 0.0, 1.0),
    )
    assert fs.branches_overlap_by_direction(params) is True


def test_overlap_antiparallel_spur_primary_t_junction():
    fs = _import_module()
    params = _t_junction_params(
        fs,
        primary_dir=(1.0, 0.0, 0.0),
        spur_dir=(-1.0, 0.0, 0.0),
        stem_dir=(0.0, 0.0, 1.0),
    )
    assert fs.branches_overlap_by_direction(params) is True


def test_no_overlap_perpendicular_spur_primary():
    fs = _import_module()
    params = _t_junction_params(
        fs,
        primary_dir=(1.0, 0.0, 0.0),
        spur_dir=(0.0, 0.0, 1.0),
        stem_dir=(0.0, 1.0, 0.0),
    )
    assert fs.branches_overlap_by_direction(params) is False


def test_overlap_antiparallel_stem_spur():
    fs = _import_module()
    params = _t_junction_params(
        fs,
        primary_dir=(1.0, 0.0, 0.0),
        spur_dir=(0.0, 0.0, -1.0),
        stem_dir=(0.0, 0.0, 1.0),
    )
    assert fs.branches_overlap_by_direction(params) is True


def test_no_overlap_parallel_stem_spur_linear():
    fs = _import_module()
    params = _linear_chain_params(
        fs,
        primary_dir=(1.0, 0.0, 0.0),
        spur_dir=(0.0, 0.0, -1.0),
        stem_dir=(0.0, 0.0, -1.0),
    )
    assert fs.branches_overlap_by_direction(params) is False


def test_overlap_threshold_respected():
    fs = _import_module()
    params = _t_junction_params(
        fs,
        primary_dir=(1.0, 0.0, 0.0),
        spur_dir=(0.707, 0.707, 0.0),
        stem_dir=(0.0, 0.0, 1.0),
    )
    assert fs.branches_overlap_by_direction(params, threshold=0.75) is False
    assert fs.branches_overlap_by_direction(params, threshold=0.5) is True


def _capsule_display_color_for_body(builder, body_id: int) -> tuple[float, float, float]:
    import newton

    for shape_idx, bid in enumerate(builder.shape_body):
        if bid == body_id and builder.shape_flags[shape_idx] & newton.ShapeFlags.VISIBLE:
            c = builder.shape_color[shape_idx]
            return (float(c[0]), float(c[1]), float(c[2]))
    raise AssertionError(f"no visible capsule shape for body {body_id}")


def test_rod_segments_have_distinct_display_colors():
    """Each woody rod type (primary/spur/stem) gets a unique viewer color."""
    from apple_pick_sim.fruiting_system import build as fruiting_build

    fs = _import_module()
    builder = fruiting_build._new_fruiting_builder()
    params = _t_junction_params(
        fs,
        primary_dir=(1.0, 0.0, 0.0),
        spur_dir=(0.0, 1.0, 0.0),
        stem_dir=(0.0, 0.0, 1.0),
    )
    artifacts = fruiting_build._build_fruiting_chain_into_builder(
        builder, params, (0.5, 0.5, 0.5)
    )

    colors: dict[str, tuple[float, float, float]] = {}
    for name, bodies in artifacts.seg_bodies.items():
        if bodies:
            colors[name] = _capsule_display_color_for_body(builder, bodies[0])

    assert set(colors) == {"primary", "spur", "stem"}
    assert len(set(colors.values())) == len(colors)
    np.testing.assert_allclose(
        colors["primary"],
        fruiting_build._rod_display_color("primary"),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        colors["spur"],
        fruiting_build._rod_display_color("spur"),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        colors["stem"],
        fruiting_build._rod_display_color("stem"),
        atol=1e-6,
    )


def test_sample_params_no_overlap_result_passes_check():
    fs = _import_module()
    ranges = fs.load_ranges(VARIANCE_FIXTURE)
    params = fs.sample_params_no_overlap(ranges, seed=42)
    assert fs.branches_overlap_by_direction(params) is False


def test_sample_params_no_overlap_deterministic():
    fs = _import_module()
    ranges = fs.load_ranges(VARIANCE_FIXTURE)
    p1 = fs.sample_params_no_overlap(ranges, seed=99)
    p2 = fs.sample_params_no_overlap(ranges, seed=99)
    assert fs.params_fingerprint(p1) == fs.params_fingerprint(p2)


def test_sample_params_no_overlap_exhaustion_raises(monkeypatch):
    fs = _import_module()
    ranges = fs.load_ranges(VARIANCE_FIXTURE)

    def _always_overlap(params, threshold=0.75):
        del params, threshold
        return True

    monkeypatch.setattr(
        "apple_pick_sim.fruiting_system.params.branches_overlap_by_direction",
        _always_overlap,
    )
    with pytest.raises(RuntimeError, match="non-overlapping"):
        fs.sample_params_no_overlap(ranges, seed=0, max_retries=3)


def test_hetero_list_with_overlap_threshold_all_clean():
    fs = _import_module()
    ranges = fs.load_ranges(VARIANCE_FIXTURE)
    params_list = fs.sample_heterogeneous_params_list(
        ranges, topology_seed=42, num_envs=8, overlap_threshold=0.75
    )
    assert len(params_list) == 8
    for params in params_list:
        assert fs.branches_overlap_by_direction(params, threshold=0.75) is False
