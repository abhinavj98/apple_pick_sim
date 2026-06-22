"""Tests for MMD grid result output helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from apple_pick_sim.system_id.mmd_results import (
    MmdCandidateResult,
    rank_results,
    write_diagnostic_plots,
    write_direction_heatmap_plot,
    write_ranked_loss_plot,
    write_results_csv,
    write_stiffness_sensitivity_plot,
)


def _result(
    idx: int,
    *,
    aggregate: float,
    directions: dict[int, float] | None = None,
) -> MmdCandidateResult:
    return MmdCandidateResult(
        candidate_index=idx,
        stiffnesses={
            "primary": float(idx),
            "secondary": 10.0,
            "spur": 20.0,
            "stem": 30.0,
        },
        aggregate_mmd2=aggregate,
        per_direction_mmd2={0: aggregate} if directions is None else directions,
    )


def test_rank_results_orders_by_aggregate_mmd2():
    ranked = rank_results([_result(1, aggregate=3.0), _result(2, aggregate=1.0)])

    assert [result.candidate_index for result in ranked] == [2, 1]


def test_write_results_csv_includes_stiffnesses_and_direction_columns(tmp_path: Path):
    path = write_results_csv(
        [
            _result(0, aggregate=0.25, directions={0: 0.1, 2: 0.4}),
            _result(1, aggregate=0.5, directions={0: 0.5}),
        ],
        tmp_path,
    )

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert path.name == "mmd_results.csv"
    assert rows[0]["candidate_index"] == "0"
    assert rows[0]["rank"] == "1"
    assert rows[0]["primary_bend_stiffness"] == "0.0"
    assert rows[0]["aggregate_mmd2"] == "0.25"
    assert rows[0]["n_directions"] == "2"
    assert rows[0]["direction_0_mmd2"] == "0.1"
    assert rows[0]["direction_2_mmd2"] == "0.4"
    assert rows[1]["direction_2_mmd2"] == ""


def test_write_ranked_loss_plot_creates_png(tmp_path: Path):
    path = write_ranked_loss_plot(
        [_result(0, aggregate=0.25), _result(1, aggregate=0.5)],
        tmp_path,
    )

    assert path.name == "mmd_ranked_loss.png"
    assert path.read_bytes().startswith(b"\x89PNG")


def test_write_direction_heatmap_plot_handles_sparse_direction_results(tmp_path: Path):
    path = write_direction_heatmap_plot(
        [
            _result(0, aggregate=0.25, directions={0: 0.1, 2: 0.4}),
            _result(1, aggregate=0.5, directions={0: 0.5}),
        ],
        tmp_path,
    )

    assert path.name == "mmd_direction_heatmap.png"
    assert path.read_bytes().startswith(b"\x89PNG")


def test_write_stiffness_sensitivity_plot_creates_png(tmp_path: Path):
    path = write_stiffness_sensitivity_plot(
        [
            _result(0, aggregate=0.25),
            _result(1, aggregate=0.5),
            _result(2, aggregate=0.1),
        ],
        tmp_path,
    )

    assert path.name == "mmd_stiffness_sensitivity.png"
    assert path.read_bytes().startswith(b"\x89PNG")


def test_write_diagnostic_plots_returns_all_png_paths(tmp_path: Path):
    paths = write_diagnostic_plots(
        [
            _result(0, aggregate=0.25, directions={0: 0.1, 2: 0.4}),
            _result(1, aggregate=0.5, directions={0: 0.5}),
        ],
        tmp_path,
    )

    assert [path.name for path in paths] == [
        "mmd_ranked_loss.png",
        "mmd_direction_heatmap.png",
        "mmd_stiffness_sensitivity.png",
    ]
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)


def test_result_requires_at_least_one_direction():
    with pytest.raises(ValueError, match="at least one direction"):
        _result(0, aggregate=0.0, directions={})
