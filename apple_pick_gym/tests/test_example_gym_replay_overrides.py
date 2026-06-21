"""CLI and helper tests for replay with stiffness/damping overrides."""

from __future__ import annotations

import argparse
import importlib.util
import warnings
from pathlib import Path

import pytest


def _load_example_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "example_gym_replay_overrides.py"
    )
    spec = importlib.util.spec_from_file_location("example_gym_replay_overrides_under_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parser_defaults_to_observable_data_and_warns_for_snapshot(monkeypatch):
    module = _load_example_module()

    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)

    parser = module._make_parser()

    default_args = parser.parse_args(["--dataset", "dataset-dir"])
    snapshot_args = parser.parse_args(["--dataset", "dataset-dir", "--use-snapshot"])

    assert default_args.use_snapshot is False
    assert snapshot_args.use_snapshot is True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module._warn_if_snapshot_mode(snapshot_args.use_snapshot)
    assert any("privileged simulator state" in str(w.message) for w in caught)


def test_apply_param_overrides_changes_only_requested_fields():
    module = _load_example_module()
    import apple_pick_sim.fruiting_system as fs

    ranges = fs.load_ranges(
        Path(__file__).resolve().parents[2]
        / "apple_pick_sim"
        / "fixtures"
        / "fruiting_system_ranges_example_variance.json"
    )
    params = fs.sample_params(ranges, seed=0)

    out = module.apply_param_overrides(
        params,
        {
            "stem": {
                "bend_stiffness": 123.0,
                "bend_damping": 4.5,
                "stretch_stiffness": 900.0,
            }
        },
    )

    assert out.stem is not None
    assert params.stem is not None
    assert out.stem.bend_stiffness == pytest.approx(123.0)
    assert out.stem.bend_damping == pytest.approx(4.5)
    assert out.stem.stretch_stiffness == pytest.approx(900.0)
    assert out.stem.length == pytest.approx(params.stem.length)
    assert out.primary == params.primary


def test_apply_param_overrides_rejects_disabled_segment():
    module = _load_example_module()
    import apple_pick_sim.fruiting_system as fs

    ranges = fs.load_ranges(
        Path(__file__).resolve().parents[2]
        / "apple_pick_sim"
        / "fixtures"
        / "fruiting_system_ranges_example_variance.json"
    )
    params = fs.sample_params(ranges, seed=0, omit=frozenset({"secondary"}))

    with pytest.raises(ValueError, match="disabled"):
        module.apply_param_overrides(
            params,
            {"secondary": {"bend_stiffness": 100.0}},
        )
