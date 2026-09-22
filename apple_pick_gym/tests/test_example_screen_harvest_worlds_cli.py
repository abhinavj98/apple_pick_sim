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


@requires_fr3
def test_sample_then_rescreen_end_to_end_on_cpu(tmp_path):
    from apple_pick_gym.batched_examples.example_screen_harvest_worlds import main
    from apple_pick_gym.batched_envs.world_set import load_world_set

    p1 = tmp_path / "pass1.jsonl"
    main(["sample", "--num-envs", "2", "--seed", "5", "--device", "cpu", "--out", str(p1), *_SHORT])
    specs = load_world_set(p1)
    assert [s.world_id for s in specs] == ["s5_e0", "s5_e1"]
    for s in specs:
        rec = s.screening["pass1"]
        assert set(rec) >= {"passed", "reasons", "pull_max_wrist_n", "hold_tcp_drift_m"}
        assert s.screening["passed"] == rec["passed"]

    # Force both through to pass 2 regardless of the (tiny-step) pass-1 outcome.
    from apple_pick_gym.batched_envs.world_set import save_world_set
    import dataclasses

    forced = [dataclasses.replace(s, screening={**s.screening, "passed": True}) for s in specs]
    save_world_set(p1, forced)
    p2 = tmp_path / "pass2.jsonl"
    main(["rescreen", "--from-set", str(p1), "--shuffle-seed", "0", "--chunk-size", "2", "--chunk-index", "0",
          "--device", "cpu", "--out", str(p2), *_SHORT])
    again = load_world_set(p2)
    assert sorted(s.world_id for s in again) == ["s5_e0", "s5_e1"]
    for s in again:
        assert "pass1" in s.screening and "pass2" in s.screening
        assert s.screening["passed"] == (s.screening["pass1"]["passed"] and s.screening["pass2"]["passed"])
