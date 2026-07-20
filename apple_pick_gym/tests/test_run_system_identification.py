"""CLI and helper tests for the diagnostic sys-ID grid runner."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
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
    assert args.mmd_output is None
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


def test_parser_accepts_mmd_output_directory(monkeypatch):
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
            "--mmd-output",
            "mmd-out",
        ]
    )

    assert args.mmd_output == "mmd-out"


def _mmd_arrays(*, shift: float = 0.0) -> dict:
    steps = 6
    base = np.arange(steps, dtype=np.float32).reshape(steps, 1)
    return {
        "ft_wrist": np.hstack([base + shift + i for i in range(6)]).astype(np.float32),
        "tcp_velocity": np.hstack([base + 10.0 + i for i in range(6)]).astype(
            np.float32
        ),
        "action": np.hstack([base + 20.0 + i for i in range(6)]).astype(np.float32),
        "tcp_pos": np.hstack([base + 30.0 + i for i in range(3)]).astype(np.float32),
        "apple_pos": np.hstack([base + 40.0 + i for i in range(3)]).astype(np.float32),
        "woody_part_start_pos": {
            "stem_apple": np.hstack([base + 50.0 + i for i in range(3)]).astype(
                np.float32
            )
        },
        "woody_part_end_pos": {
            "stem_apple": np.hstack([base + 60.0 + i for i in range(3)]).astype(
                np.float32
            )
        },
        "excitation_direction": np.tile(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (steps, 1)
        ),
        "phase": np.ones(steps, dtype=np.int8),
        "dir_idx": np.zeros(steps, dtype=np.int32),
        "excitation_type": np.zeros(steps, dtype=np.int8),
        "junction_names": ["stem_apple"],
    }


def test_mmd_result_helpers_rank_identical_candidate_below_shifted_candidate():
    module = _load_example_module()
    gt_context = module._prepare_gt_mmd_context([_mmd_arrays()])
    candidate = module.BendStiffnessCandidate(
        primary=1.0,
        secondary=2.0,
        spur=3.0,
        stem=4.0,
    )

    identical = module._compute_candidate_mmd_result(
        candidate_index=1,
        candidate=candidate,
        gt_context=gt_context,
        replay_observations=[_mmd_arrays()],
    )
    shifted = module._compute_candidate_mmd_result(
        candidate_index=2,
        candidate=candidate,
        gt_context=gt_context,
        replay_observations=[_mmd_arrays(shift=10.0)],
    )

    assert identical.aggregate_mmd2 < shifted.aggregate_mmd2
    assert identical.per_direction_mmd2.keys() == {0}
    assert identical.stiffnesses == {
        "primary": 1.0,
        "secondary": 2.0,
        "spur": 3.0,
        "stem": 4.0,
    }


def test_write_mmd_outputs_creates_csv_and_diagnostic_plot_bundle(tmp_path: Path, capsys):
    module = _load_example_module()
    from apple_pick_sim.system_id.mmd_results import MmdCandidateResult

    result = MmdCandidateResult(
        candidate_index=1,
        stiffnesses={
            "primary": 1.0,
            "secondary": 2.0,
            "spur": 3.0,
            "stem": 4.0,
        },
        aggregate_mmd2=0.25,
        per_direction_mmd2={0: 0.1, 1: 0.4},
    )

    module._write_mmd_outputs([result], tmp_path)

    output = capsys.readouterr().out
    assert (tmp_path / "mmd_results.csv").is_file()
    assert (tmp_path / "mmd_ranked_loss.png").is_file()
    assert (tmp_path / "mmd_direction_heatmap.png").is_file()
    assert (tmp_path / "mmd_stiffness_sensitivity.png").is_file()
    assert "MMD results CSV:" in output
    assert "mmd_ranked_loss.png" in output
    assert "mmd_direction_heatmap.png" in output
    assert "mmd_stiffness_sensitivity.png" in output
