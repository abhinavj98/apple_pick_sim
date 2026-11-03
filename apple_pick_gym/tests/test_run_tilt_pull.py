"""run_tilt_pull: CLI options -> ScriptedStaticRotatePullPolicy (no sim needed)."""

from __future__ import annotations

import math

import pytest

from apple_pick_gym.rl.run_tilt_pull import build_parser, build_policy, snapshot_for
from apple_pick_gym.rl.baselines import ScriptedStaticRotatePullPolicy

_BASE = ["--config", "cfg.json", "--world-set", "ws/shard_00.jsonl", "--out", "o.json"]


def _policy(*extra):
    return build_policy(build_parser().parse_args([*_BASE, *extra]))


def test_defaults_are_tip_up_30_then_pull_back_along_the_tool_axis_with_k_ang_40():
    p = _policy()
    assert isinstance(p, ScriptedStaticRotatePullPolicy)
    assert p.rotate == pytest.approx(math.radians(30.0))
    assert p.axis == "sideways" and p.sign == -1.0  # tip up
    assert p.pull_dir == "tool"  # pull back in line with the TCP
    assert (p.k_lin, p.k_ang) == (400.0, 40.0)
    assert p.rate == pytest.approx(0.00105)


def test_options_change_the_tilt_direction_pull_and_gains():
    p = _policy("--tip", "down", "--pull", "down", "--tilt-deg", "20", "--k-lin", "150", "--k-ang", "20", "--rate-mm", "2")
    assert p.sign == 1.0 and p.pull_dir == "down"
    assert p.rotate == pytest.approx(math.radians(20.0))
    assert (p.k_lin, p.k_ang) == (150.0, 20.0)
    assert p.rate == pytest.approx(0.002)


def test_k_ang_above_the_env_cap_is_refused_instead_of_silently_clipped():
    with pytest.raises(SystemExit):
        build_parser().parse_args([*_BASE, "--k-ang", "60"])


def test_snapshot_defaults_to_the_sibling_of_the_world_set():
    assert snapshot_for("a/b/shard_00.jsonl", None) == "a/b/shard_00_snapshot.npz"
    assert snapshot_for("a/b/shard_00.jsonl", "x.npz") == "x.npz"


def test_world_back_pull_option():
    assert _policy("--pull", "world-back").pull_dir == "world_back"
