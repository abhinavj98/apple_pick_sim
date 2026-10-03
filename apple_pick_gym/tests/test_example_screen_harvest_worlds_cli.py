"""CLI: screen candidate harvest worlds (sample) and re-screen the survivors (rescreen)."""

from __future__ import annotations

import pytest

from apple_pick_gym.tests.test_apple_pick_vic_harvest_env import requires_fr3

_SHORT = [
    "--hold-steps", "3",
    "--pull-rest-steps", "2",
    "--pull-ramp-steps", "2",
    "--pull-hold-steps", "1",
    "--pull-settle-steps", "1",
    "--num-pull-episodes", "1",
]


def test_parser_defaults_and_modes():
    from apple_pick_gym.batched_examples.example_screen_harvest_worlds import make_parser

    a = make_parser().parse_args(["sample", "--num-envs", "8", "--seed", "3", "--out", "x.jsonl"])
    assert (a.mode, a.num_envs, a.seed, a.out) == ("sample", 8, 3, "x.jsonl")
    b = make_parser().parse_args(
        ["rescreen", "--from-set", "x.jsonl", "--shuffle-seed", "1", "--chunk-size", "4", "--chunk-index", "2", "--out", "y.jsonl"]
    )
    assert (b.mode, b.from_set, b.chunk_size, b.chunk_index) == ("rescreen", "x.jsonl", 4, 2)


def _run_cli(*args: str) -> None:
    """One build per OS process (docs/in-process-rebuild-heap-corruption.md)."""
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "apple_pick_gym.batched_examples.example_screen_harvest_worlds", *args]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


@requires_fr3
def test_sample_then_rescreen_end_to_end_on_cpu(tmp_path):
    import dataclasses

    from apple_pick_gym.batched_envs.world_set import load_world_set, save_world_set

    p1 = tmp_path / "pass1.jsonl"
    _run_cli("sample", "--num-envs", "2", "--seed", "5", "--device", "cpu", "--out", str(p1), *_SHORT)
    specs = load_world_set(p1)
    assert [s.world_id for s in specs] == ["s5_e0", "s5_e1"]
    for s in specs:
        rec = s.screening["pass1"]
        assert set(rec) >= {"passed", "reasons", "pull_max_wrist_n", "hold_tcp_drift_m"}
        assert s.screening["passed"] == rec["passed"]

    # Force both through to pass 2 regardless of the (tiny-step) pass-1 outcome.
    forced = [dataclasses.replace(s, screening={**s.screening, "passed": True}) for s in specs]
    save_world_set(p1, forced)
    p2 = tmp_path / "pass2.jsonl"
    _run_cli("rescreen", "--from-set", str(p1), "--shuffle-seed", "0", "--chunk-size", "2",
             "--chunk-index", "0", "--device", "cpu", "--out", str(p2), *_SHORT)
    again = load_world_set(p2)
    assert sorted(s.world_id for s in again) == ["s5_e0", "s5_e1"]
    for s in again:
        assert "pass1" in s.screening and "pass2" in s.screening
        assert s.screening["passed"] == (s.screening["pass1"]["passed"] and s.screening["pass2"]["passed"])


def test_report_mode_parses_multiple_sets():
    from apple_pick_gym.batched_examples.example_screen_harvest_worlds import make_parser

    a = make_parser().parse_args(["report", "a.jsonl", "b.jsonl", "--out-md", "r.md"])
    assert (a.mode, a.sets, a.out_md) == ("report", ["a.jsonl", "b.jsonl"], "r.md")


def test_d17_sample_takes_the_grasp_cone_and_roll():
    import math

    from apple_pick_gym.batched_examples.example_screen_harvest_worlds import grasp_env_kwargs, make_parser

    a = make_parser().parse_args(["sample", "--num-envs", "8", "--seed", "3", "--out", "x.jsonl"])
    assert grasp_env_kwargs(a) == {}  # defaults: the env's own cone, no roll (v1/v2 recipe)
    b = make_parser().parse_args(
        ["sample", "--num-envs", "8", "--seed", "3", "--out", "x.jsonl",
         "--grasp-pole", "0", "-1", "0", "--grasp-max-polar-deg", "60", "--grasp-roll-max-deg", "90"]
    )
    kw = grasp_env_kwargs(b)
    assert kw["grasp_hemisphere_pole"] == (0.0, -1.0, 0.0)
    assert kw["grasp_max_polar_angle_rad"] == pytest.approx(math.pi / 3)
    assert kw["grasp_roll_max_rad"] == pytest.approx(math.pi / 2)
