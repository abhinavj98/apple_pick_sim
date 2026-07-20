"""Tests for CMA-ES generation score and spur/stem vs Sinkhorn visualizations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apple_pick_gym.youngs_modulus_cmaes_viz import (
    build_cma_evaluated_points,
    generation_distribution_series_from_structure,
    generation_score_series_from_structure,
    write_cmaes_visualization_bundle,
    write_generation_score_figures,
    write_structure_optimizer_diagnostics_figure,
    write_spur_stem_sinkhorn_scatter_3d,
)


def _cov(*, sigma: float, phenotype_std: list[float], diag: list[float]) -> dict:
    return {
        "C": [[diag[0], 0.0, 0.0], [0.0, diag[1], 0.0], [0.0, 0.0, diag[2]]],
        "sigma": sigma,
        "sigma_vec_scaling": [1.0, 1.0, 1.0],
        "phenotype_std": phenotype_std,
        "effective_unbounded_covariance": [
            [diag[0] * sigma**2, 0.0, 0.0],
            [0.0, diag[1] * sigma**2, 0.0],
            [0.0, 0.0, diag[2] * sigma**2],
        ],
    }


def _structure_payload() -> dict:
    return {
        "structure_idx": 0,
        "status": "fitted",
        "gt": {"log10_e": [8.0, 7.0, 6.0], "e_pa": [1e8, 1e7, 1e6]},
        "final_mean": {
            "log10_e": [8.1, 7.1, 6.1],
            "e_pa": [1.2589254117941673e8, 1.2589254117941673e7, 1.2589254117941673e6],
            "aggregate_sinkhorn": 12.5,
        },
        "best_sample": {
            "log10_e": [8.05, 7.05, 6.05],
            "fitness": 10.0,
        },
        "generations": [
            {
                "generation_index": 0,
                "ask_samples_log10": [[8.0, 7.2, 6.1], [8.2, 6.8, 5.9]],
                "ask_distribution": {
                    "mean_log10": [8.0, 7.5, 6.5],
                    "sigma": 1.0,
                    "covariance": _cov(
                        sigma=1.0,
                        phenotype_std=[1.0, 1.0, 1.0],
                        diag=[1.0, 1.0, 1.0],
                    ),
                },
                "post_tell_distribution": {
                    "mean_log10": [8.0, 7.4, 6.4],
                    "sigma": 0.9,
                    "covariance": _cov(
                        sigma=0.9,
                        phenotype_std=[0.9, 0.9, 0.9],
                        diag=[1.0, 1.0, 1.0],
                    ),
                },
                "penalized_fitness": [10.0, 30.0],
                "penalty_metadata": [
                    {
                        "candidate_index": 0,
                        "penalized": False,
                        "raw_aggregate_sinkhorn": 10.0,
                        "fitness": 10.0,
                        "disqualification_reason": None,
                    },
                    {
                        "candidate_index": 1,
                        "penalized": True,
                        "raw_aggregate_sinkhorn": 20.0,
                        "fitness": 30.0,
                        "disqualification_reason": "replay_instability",
                    },
                ],
                "score_summary": {
                    "n_eligible": 1,
                    "eligible_mean": 10.0,
                    "eligible_variance": None,
                    "eligible_std": None,
                    "best_eligible": 10.0,
                    "penalized_mean": 20.0,
                    "penalized_variance": 200.0,
                    "penalized_std": 14.142135623730951,
                    "n_penalized": 1,
                },
            },
            {
                "generation_index": 1,
                "ask_samples_log10": [[8.1, 7.0, 6.0], [8.0, 7.1, 6.2]],
                "ask_distribution": {
                    "mean_log10": [8.0, 7.4, 6.4],
                    "sigma": 0.9,
                    "covariance": _cov(
                        sigma=0.9,
                        phenotype_std=[0.9, 0.9, 0.9],
                        diag=[1.0, 1.0, 1.0],
                    ),
                },
                "post_tell_distribution": {
                    "mean_log10": [8.0, 7.1, 6.1],
                    "sigma": 0.5,
                    "covariance": _cov(
                        sigma=0.5,
                        phenotype_std=[0.5, 0.4, 0.3],
                        diag=[1.0, 0.64, 0.36],
                    ),
                },
                "penalized_fitness": [8.0, 12.0],
                "penalty_metadata": [
                    {
                        "candidate_index": 0,
                        "penalized": False,
                        "raw_aggregate_sinkhorn": 8.0,
                        "fitness": 8.0,
                        "disqualification_reason": None,
                    },
                    {
                        "candidate_index": 1,
                        "penalized": False,
                        "raw_aggregate_sinkhorn": 12.0,
                        "fitness": 12.0,
                        "disqualification_reason": None,
                    },
                ],
                "score_summary": {
                    "n_eligible": 2,
                    "eligible_mean": 10.0,
                    "eligible_variance": 8.0,
                    "eligible_std": 2.8284271247461903,
                    "best_eligible": 8.0,
                    "penalized_mean": 10.0,
                    "penalized_variance": 8.0,
                    "penalized_std": 2.8284271247461903,
                    "n_penalized": 0,
                },
            },
        ],
    }


def test_generation_score_series_from_structure():
    series = generation_score_series_from_structure(_structure_payload())
    assert series["generation_index"] == [0, 1]
    assert series["eligible_mean"] == [10.0, 10.0]
    assert series["eligible_variance"] == [None, 8.0]
    assert series["eligible_std"] == [None, pytest.approx(2.8284271247461903)]


def test_build_cma_evaluated_points_marks_gt_final_and_disqualified():
    points = build_cma_evaluated_points(_structure_payload())
    kinds = {p["kind"] for p in points}
    assert "candidate" in kinds
    assert "disqualified" in kinds
    assert "gt" in kinds
    assert "final_mean" in kinds
    gt = next(p for p in points if p["kind"] == "gt")
    assert gt["log10_e"] == [8.0, 7.0, 6.0]
    final = next(p for p in points if p["kind"] == "final_mean")
    assert final["score"] == pytest.approx(12.5)


def test_generation_distribution_series_tracks_mean_and_covariance():
    series = generation_distribution_series_from_structure(_structure_payload())
    assert series["generation_index"] == [0, 1]
    assert series["mean_spur"] == [pytest.approx(7.4), pytest.approx(7.1)]
    assert series["mean_stem"] == [pytest.approx(6.4), pytest.approx(6.1)]
    assert series["sigma"] == [pytest.approx(0.9), pytest.approx(0.5)]
    assert series["phenotype_std_spur"] == [pytest.approx(0.9), pytest.approx(0.4)]
    assert series["distance_to_gt_spur_stem"][0] > series["distance_to_gt_spur_stem"][1]
    assert series["effective_cov_trace"][0] > series["effective_cov_trace"][1]


def test_write_generation_score_and_spur_stem_sinkhorn_figures(tmp_path: Path):
    report = {
        "structures": {"0": _structure_payload(), "1": {**_structure_payload(), "structure_idx": 1}},
        "timing": {"fit_seconds": 1.5, "command_seconds": 2.0, "waves": []},
    }
    report_path = tmp_path / "cmaes_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    out_dir = tmp_path / "viz"
    paths = write_cmaes_visualization_bundle(report_path, out_dir)
    assert (out_dir / "generation_score_mean_variance.html").is_file()
    assert (out_dir / "structure_000_generation_scores.html").is_file()
    assert (out_dir / "structure_001_generation_scores.html").is_file()
    assert (out_dir / "structure_000_spur_stem_sinkhorn_3d.html").is_file()
    assert (out_dir / "structure_000_optimizer_diagnostics.html").is_file()
    assert any(p.name.endswith("generation_score_mean_variance.html") for p in paths)

    mean_path = write_generation_score_figures(report, out_dir / "scores.html")
    assert mean_path.is_file()
    mean_html = mean_path.read_text(encoding="utf-8")
    assert "Eligible Sinkhorn mean" in mean_html
    assert "Eligible Sinkhorn variance" in mean_html

    per_structure = (out_dir / "structure_000_generation_scores.html").read_text(
        encoding="utf-8"
    )
    assert "structure 0" in per_structure
    assert "Eligible Sinkhorn variance" in per_structure

    diag = write_structure_optimizer_diagnostics_figure(
        _structure_payload(),
        out_dir / "s0_diag.html",
    )
    assert diag.is_file()
    diag_html = diag.read_text(encoding="utf-8")
    assert "mean log10" in diag_html.lower() or "Mean log10" in diag_html
    assert "sigma" in diag_html.lower()
    assert "GT" in diag_html
    assert "phenotype std" in diag_html.lower() or "Phenotype std" in diag_html

    scatter = write_spur_stem_sinkhorn_scatter_3d(
        _structure_payload(),
        out_dir / "s0_3d.html",
    )
    assert scatter.is_file()
    html = scatter.read_text(encoding="utf-8")
    assert "log10(spur)" in html
    assert "log10(stem)" in html
    assert "Sinkhorn" in html
    assert "log10(primary)" not in html
    assert "GT" in html
    assert "best sample" in html
