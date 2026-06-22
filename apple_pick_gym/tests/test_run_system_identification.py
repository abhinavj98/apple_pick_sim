"""CLI and helper tests for the diagnostic sys-ID grid runner."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


def _load_example_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "run_system_identification.py"
    )
    spec = importlib.util.spec_from_file_location("run_system_identification_under_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_positive_float_grid_accepts_comma_separated_values():
    module = _load_example_module()

    assert module.parse_positive_float_grid("10, 25.5,50") == (10.0, 25.5, 50.0)


@pytest.mark.parametrize("value", ["", "0", "-1,2", "1,,2", "nan", "inf"])
def test_parse_positive_float_grid_rejects_empty_or_non_positive_values(value):
    module = _load_example_module()

    with pytest.raises(argparse.ArgumentTypeError):
        module.parse_positive_float_grid(value)


def test_bend_stiffness_candidates_are_cartesian_product_in_grid_order():
    module = _load_example_module()

    candidates = list(
        module.iter_bend_stiffness_candidates(
            primary_values=(1.0, 2.0),
            secondary_values=(10.0,),
            spur_values=(100.0,),
            stem_values=(1000.0, 2000.0),
        )
    )

    assert candidates == [
        module.BendStiffnessCandidate(
            primary=1.0,
            secondary=10.0,
            spur=100.0,
            stem=1000.0,
        ),
        module.BendStiffnessCandidate(
            primary=1.0,
            secondary=10.0,
            spur=100.0,
            stem=2000.0,
        ),
        module.BendStiffnessCandidate(
            primary=2.0,
            secondary=10.0,
            spur=100.0,
            stem=1000.0,
        ),
        module.BendStiffnessCandidate(
            primary=2.0,
            secondary=10.0,
            spur=100.0,
            stem=2000.0,
        ),
    ]


def test_bend_stiffness_candidate_maps_to_segment_override_dict():
    module = _load_example_module()

    candidate = module.BendStiffnessCandidate(
        primary=1.0,
        secondary=2.0,
        spur=3.0,
        stem=4.0,
    )

    assert candidate.to_overrides() == {
        "primary": {"bend_stiffness": 1.0},
        "secondary": {"bend_stiffness": 2.0},
        "spur": {"bend_stiffness": 3.0},
        "stem": {"bend_stiffness": 4.0},
    }


def test_parser_defaults_to_observation_only_replay(monkeypatch):
    module = _load_example_module()

    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)

    args = module._make_parser().parse_args(
        [
            "--dataset",
            "dataset-dir",
            "--primary-bend-stiffness-values",
            "1",
            "--secondary-bend-stiffness-values",
            "2",
            "--spur-bend-stiffness-values",
            "3",
            "--stem-bend-stiffness-values",
            "4",
        ]
    )

    assert args.use_snapshot is False
    assert args.primary_bend_stiffness_values == (1.0,)
    assert args.secondary_bend_stiffness_values == (2.0,)
    assert args.spur_bend_stiffness_values == (3.0,)
    assert args.stem_bend_stiffness_values == (4.0,)


def test_parser_allows_listing_episodes_without_grid_values(monkeypatch):
    module = _load_example_module()

    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)

    args = module._make_parser().parse_args(
        [
            "--dataset",
            "dataset-dir",
            "--list-episodes",
        ]
    )

    assert args.list_episodes is True
