"""WorldSpec: one screened harvest world (plant + grasp + build-time DR), JSONL round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apple_pick_gym.batched_envs.world_set import (
    WorldSpec,
    load_world_set,
    save_world_set,
    world_specs_to_env_kwargs,
)


def _params():
    from apple_pick_sim.fruiting_system import load_ranges, sample_params

    fixture = (
        Path(__file__).resolve().parents[2]
        / "apple_pick_sim"
        / "fixtures"
        / "fruiting_system_ranges_rl_harvest_real_g05_m1.json"
    )
    return sample_params(load_ranges(fixture), 0)


def _spec(i: int = 0, **kw) -> WorldSpec:
    from apple_pick_sim.fruiting_system.params import fruiting_params_to_json

    base = dict(
        world_id=f"s7_b0_e{i}",
        params_json=fruiting_params_to_json(_params()),
        weld_direction=(0.0, 0.0, -1.0),
        support_kp=250.0 + i,
        support_roll_kp=0.7,
        support_zeta=0.5,
        arm_link_mass_scale=1.01,
        arm_link_inertia_scale=0.99,
        arm_ee_mass_scale=1.0,
        arm_ee_inertia_scale=1.02,
        screening={"passed": True, "peak_wrist_n": 1.2},
    )
    base.update(kw)
    return WorldSpec(**base)


def test_jsonl_round_trip_preserves_every_field(tmp_path):
    specs = [_spec(0), _spec(1, weld_direction=(-0.6, -0.8, 0.0))]
    path = tmp_path / "worlds.jsonl"
    save_world_set(path, specs)
    assert len(path.read_text().strip().splitlines()) == 2
    loaded = load_world_set(path)
    assert loaded == specs


def test_save_appends_so_screening_batches_accumulate(tmp_path):
    path = tmp_path / "worlds.jsonl"
    save_world_set(path, [_spec(0)])
    save_world_set(path, [_spec(1)], append=True)
    assert [s.world_id for s in load_world_set(path)] == ["s7_b0_e0", "s7_b0_e1"]


def test_load_can_filter_to_passed_worlds(tmp_path):
    path = tmp_path / "worlds.jsonl"
    save_world_set(path, [_spec(0), _spec(1, screening={"passed": False})])
    assert [s.world_id for s in load_world_set(path, passed_only=True)] == ["s7_b0_e0"]


def test_env_kwargs_rebuild_the_exact_plant_grasp_and_build_time_dr():
    from apple_pick_sim.fruiting_system.params import fruiting_params_to_json

    specs = [_spec(0), _spec(1)]
    kw = world_specs_to_env_kwargs(specs)
    assert kw["num_envs"] == 2
    assert [fruiting_params_to_json(p) for p in kw["per_env_params"]] == [s.params_json for s in specs]
    assert [g.weld_direction for g in kw["per_env_grippers"]] == [s.weld_direction for s in specs]
    assert all(g.dynamic_apple and g.fix_to_apple for g in kw["per_env_grippers"])
    np.testing.assert_allclose(kw["support_dr_sample"].kp, [250.0, 251.0])
    np.testing.assert_allclose(kw["arm_build_dr_scales"]["link_mass_scale"], [1.01, 1.01])


def test_world_spec_rejects_non_unit_weld_direction():
    with pytest.raises(ValueError, match="weld_direction"):
        _spec(0, weld_direction=(0.0, 0.0, -2.0))


def test_jsonl_lines_are_plain_json(tmp_path):
    path = tmp_path / "worlds.jsonl"
    save_world_set(path, [_spec(0)])
    row = json.loads(path.read_text().splitlines()[0])
    assert row["world_id"] == "s7_b0_e0"
    assert isinstance(row["params_json"], str)


def test_coverage_report_flags_knobs_where_rejections_concentrate():
    from apple_pick_gym.batched_envs.world_set import world_set_coverage, world_spec_knobs

    specs = [
        _spec(i, support_kp=100.0 + 10 * i, screening={"passed": i < 7}) for i in range(10)
    ]
    knobs = world_spec_knobs(specs[0])
    assert {"support_kp", "spur.youngs_modulus_pa", "stem.elevation_deg", "apple_radius", "weld_polar_deg"} <= set(knobs)

    rows = {r["knob"]: r for r in world_set_coverage(specs)}
    kp = rows["support_kp"]
    assert kp["n_candidates"] == 10 and kp["n_accepted"] == 7
    assert kp["candidate_min"] == 100.0 and kp["candidate_max"] == 190.0
    assert kp["accepted_max"] == 160.0
    # all three rejections sit in the top tercile of support_kp
    assert kp["reject_rate_by_tercile"][2] > 0.5
    assert kp["reject_rate_by_tercile"][0] == 0.0
