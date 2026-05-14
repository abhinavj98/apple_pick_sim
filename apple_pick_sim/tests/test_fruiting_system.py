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
from pathlib import Path

import numpy as np
import pytest
import warp as wp

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
RANGES_FIXTURE = FIXTURES_DIR / "fruiting_system_ranges_straight_rod_test.json"


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
        "bend_stiffness",
        "bend_damping",
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
        if seg is None or not isinstance(seg, dict):
            continue
        for key, val in seg.items():
            if isinstance(val, dict) and "min" in val and "max" in val:
                assert val["min"] <= val["max"], (
                    f"Range {seg_name}.{key}: min ({val['min']}) > max ({val['max']})"
                )


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
    scene = fs.generate_scene(ranges, seed=3, omit=frozenset({"apple"}))
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
    """Primary bend stiffness must be >= secondary when both segments are enabled."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    for seed in range(20):
        params = fs.sample_params(ranges, seed=seed)
        if params.primary is None or params.secondary is None:
            continue
        assert params.primary.bend_stiffness >= params.secondary.bend_stiffness, (
            f"seed={seed}: primary.bend_stiffness ({params.primary.bend_stiffness}) "
            f"< secondary.bend_stiffness ({params.secondary.bend_stiffness})"
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
            for attr in ("length", "radius", "bend_stiffness", "bend_damping", "density"):
                v = getattr(seg_params, attr)
                lo = seg_ranges[attr]["min"]
                hi = seg_ranges[attr]["max"]
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
    scene = fs.generate_scene(ranges, seed=0)
    assert scene is not None
    assert scene.model is not None


def test_generate_scene_disable_self_collision_superset_of_joint_default_filters():
    """enable_self_collisions=False adds intra-chain filter pairs (see _apply_all_chain_collision_filters)."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    seed = 0
    scene_on = fs.generate_scene(ranges, seed=seed, enable_self_collisions=True)
    scene_off = fs.generate_scene(ranges, seed=seed, enable_self_collisions=False)
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
    scene = fs.generate_scene(ranges, seed=7)
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
        scene = fs.generate_scene(ranges, seed=seed)
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
        scene = fs.generate_scene(ranges, seed=seed)
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


# ---------------------------------------------------------------------------
# Geometry fingerprint stability and variance
# ---------------------------------------------------------------------------


def test_geometry_fingerprint_stable_same_seed():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    fp1 = fs.geometry_fingerprint(fs.generate_scene(ranges, seed=99))
    fp2 = fs.geometry_fingerprint(fs.generate_scene(ranges, seed=99))
    assert fp1 == fp2, "Geometry fingerprint must be identical for the same seed"


def test_geometry_fingerprint_varies_across_seeds():
    """At least one fingerprint value must differ across three distinct seeds."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    fps = [fs.geometry_fingerprint(fs.generate_scene(ranges, seed=s)) for s in (0, 1, 2)]
    # Not all fingerprints should be equal to each other
    assert not (fps[0] == fps[1] and fps[1] == fps[2]), (
        "All three seeds produced identical geometry fingerprints — variance not working"
    )


def test_fingerprint_primary_stiffer_than_secondary():
    """Fingerprint must reflect the stiffness ordering invariant when both exist."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    for seed in range(10):
        fp = fs.geometry_fingerprint(fs.generate_scene(ranges, seed=seed))
        pb, sb = fp.get("primary_bend_stiffness"), fp.get("secondary_bend_stiffness")
        if pb is None or sb is None:
            continue
        assert pb >= sb, f"seed={seed}: fingerprint stiffness ordering violated"


# ---------------------------------------------------------------------------
# Rollout (headless Newton simulation)
# ---------------------------------------------------------------------------


def test_short_rollout_no_crash():
    """A short SolverVBD rollout must complete without raising and produce finite transforms."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_scene(ranges, seed=3)
    fs.run_rollout(scene, num_steps=5, sim_substeps=4)
    body_q = scene.state_0.body_q.to("cpu").numpy()
    assert np.isfinite(body_q).all(), "Non-finite body transforms after rollout"


def test_rollout_finite_primary_secondary_spur_stem_apple():
    """Full topology including stem + apple must stay numerically stable across seeds."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    for seed in range(12):
        scene = fs.generate_scene(ranges, seed=seed)
        assert scene.stem_bodies, "fixture should include stem"
        assert scene.apple_body is not None, "fixture should include apple"
        fs.run_rollout(scene, num_steps=12, sim_substeps=6)
        body_q = scene.state_0.body_q.to("cpu").numpy()
        assert np.isfinite(body_q).all(), f"non-finite body_q after rollout seed={seed}"


def test_rollout_deterministic():
    """Two scenes with the same seed must reach the same state after N rollout steps."""
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene_a = fs.generate_scene(ranges, seed=5)
    scene_b = fs.generate_scene(ranges, seed=5)
    fs.run_rollout(scene_a, num_steps=10, sim_substeps=4)
    fs.run_rollout(scene_b, num_steps=10, sim_substeps=4)
    q_a = scene_a.state_0.body_q.to("cpu").numpy()
    q_b = scene_b.state_0.body_q.to("cpu").numpy()
    np.testing.assert_allclose(
        q_a, q_b, atol=1e-5, err_msg="Rollout not deterministic for the same seed"
    )


def test_ranges_null_secondary_loads_and_scene_skips_secondary():
    """JSON null for a rod segment omits that piece from params and the built scene."""
    fs = _import_module()
    ranges = dict(fs.load_ranges(RANGES_FIXTURE))
    ranges["secondary"] = None
    params = fs.sample_params(ranges, seed=11)
    assert params.secondary is None
    assert params.primary is not None
    scene = fs.generate_scene(ranges, seed=11)
    assert scene.secondary_bodies == []
    assert len(scene.primary_bodies) == params.primary.num_segments
    assert len(scene.spur_bodies) == params.spur.num_segments


def test_ranges_null_apple_skips_apple_body():
    fs = _import_module()
    ranges = dict(fs.load_ranges(RANGES_FIXTURE))
    ranges["apple"] = None
    params = fs.sample_params(ranges, seed=2)
    assert params.apple_radius is None and params.apple_density is None
    scene = fs.generate_scene(ranges, seed=2)
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
    scene = fs.generate_scene(ranges, seed=3, device="cpu")
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
    scene = fs.generate_scene(ranges, seed=3, device="cpu", omit=frozenset({"apple"}))
    labels = [lab for _, lab in fs.iter_fixed_joint_indices(scene.model)]
    assert len(labels) == 3
    assert "joint_stem_apple" not in labels


def test_fixed_joint_wrenches_finite_after_substep():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = fs.generate_scene(ranges, seed=3, device="cpu")
    sim_dt = (1.0 / 60.0) / 10.0
    scene.state_0.clear_forces()
    contacts = scene.model.collide(scene.state_0)
    q_prev = scene.state_0.body_q.numpy().copy()
    scene.solver.step(scene.state_0, scene.state_1, scene.control, contacts, sim_dt)
    scene.state_0, scene.state_1 = scene.state_1, scene.state_0
    wrenches = fs.fixed_joint_wrenches_child_com_vbd(
        scene.model,
        scene.solver,
        body_q=scene.state_0.body_q.numpy(),
        body_q_prev=q_prev,
        dt=sim_dt,
    )
    assert len(wrenches) == 4
    for w in wrenches:
        assert np.isfinite(w.force_world).all()
        assert np.isfinite(w.torque_at_child_com_world).all()
