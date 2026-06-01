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

from apple_pick_sim.tests.conftest import NO_SELF_COLLISION_KW

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
RANGES_FIXTURE = FIXTURES_DIR / "fruiting_system_ranges_straight_rod_test.json"
VARIANCE_FIXTURE = FIXTURES_DIR / "fruiting_system_ranges_example_variance.json"


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
    assert args.fruiting_base_pos == (0.2, 0.2, 0.5)
    assert args.robot_base_pos == (0.0, 0.0, 0.0)


def test_fixture_args_null_robot_base_in_variance_fixture():
    fs = _import_module()
    ranges = fs.load_ranges(VARIANCE_FIXTURE)
    args = fs.parse_fixture_args(ranges)
    assert args.fruiting_base_pos == (0.5, 0.5, 0.5)
    assert args.robot_base_pos is None


def test_resolve_fruiting_base_pos_prefers_override():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    assert fs.resolve_fruiting_base_pos(ranges, (9.0, 9.0, 9.0)) == (0.2, 0.2, 0.5)
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
    scene = fs.generate_scene(ranges, seed=0, **NO_SELF_COLLISION_KW)
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


def test_fd_stiffness_param_columns_nominal_first_and_epsilon_guard():
    fs = _import_module()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=5)
    with pytest.raises(ValueError, match="epsilon"):
        fs.fd_stiffness_param_columns(nominal, 0.0)
    cols = fs.fd_stiffness_param_columns(nominal, 0.02)
    segs = fs.enabled_rod_segments(nominal)
    assert len(cols) == 1 + len(segs)
    assert cols[0].primary.bend_stiffness == nominal.primary.bend_stiffness
    for col, seg in zip(cols[1:], segs, strict=True):
        rod_nom = getattr(nominal, seg)
        rod_col = getattr(col, seg)
        assert rod_col.bend_stiffness == pytest.approx(rod_nom.bend_stiffness + 0.02)
        assert rod_col.stretch_stiffness == rod_nom.stretch_stiffness


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
        for attr in ("length", "radius", "bend_stiffness", "bend_damping", "density"):
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
