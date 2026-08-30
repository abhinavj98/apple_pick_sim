"""Tests for the Young's-modulus CMA-ES fit-integrity gate reporter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from apple_pick_gym.batched_envs.youngs_modulus_cmaes_gate_report import (
    _parse_seeds,
    extract_structure_evidence,
    finalize_gate,
    main,
    write_seed_summary,
)


def _bounds(
    *,
    log10_lower=(6.0, 5.0, 4.0),
    log10_upper=(10.0, 9.0, 8.0),
) -> dict:
    physical_min = [10.0**v for v in log10_lower]
    physical_max = [10.0**v for v in log10_upper]
    midpoint = [0.5 * (lo + hi) for lo, hi in zip(log10_lower, log10_upper)]
    return {
        "physical_min_pa": physical_min,
        "physical_max_pa": physical_max,
        "log10_lower": list(log10_lower),
        "log10_upper": list(log10_upper),
        "log10_midpoint": midpoint,
    }


def _covariance() -> dict:
    return {
        "C": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "sigma": 0.5,
        "sigma_vec_scaling": [1.0, 1.0, 1.0],
        "phenotype_std": [0.5, 0.5, 0.5],
        "effective_unbounded_covariance": [
            [0.25, 0.0, 0.0],
            [0.0, 0.25, 0.0],
            [0.0, 0.0, 0.25],
        ],
    }


def _extrema(
    *,
    min_log10=(7.0, 6.0, 5.0),
    max_log10=(8.0, 7.0, 6.0),
) -> dict:
    return {
        "min_log10_e": list(min_log10),
        "max_log10_e": list(max_log10),
        "min_e_pa": [10.0**v for v in min_log10],
        "max_e_pa": [10.0**v for v in max_log10],
    }


def _fitted_structure(
    structure_idx: int,
    *,
    log10_e=(7.5, 6.5, 5.5),
    aggregate_sinkhorn: float = 0.12,
    per_direction: dict | None = None,
    completed_generations: int = 2,
    population_size: int = 4,
    stop_kind: str = "generation_cap",
    stop_conditions: dict | None = None,
    covariance: dict | None = None,
    extrema: dict | None = None,
    include_covariance: bool = True,
    include_extrema: bool = True,
) -> dict:
    if per_direction is None:
        per_direction = {"0": 0.1, "1": 0.14}
    if stop_conditions is None:
        stop_conditions = {} if stop_kind == "generation_cap" else {"tolx": 1e-12}
    if covariance is None and include_covariance:
        covariance = _covariance()
    if extrema is None and include_extrema:
        extrema = _extrema()
    e_pa = [10.0**v for v in log10_e]
    optimizer_samples_told = completed_generations * population_size
    return {
        "structure_idx": structure_idx,
        "status": "fitted",
        "base_seed": 0,
        "effective_seed": 1 + structure_idx,
        "initial_sigma_log10": 1.0,
        "population_size": population_size,
        "bounds": _bounds(),
        "completed_generations": completed_generations,
        "optimizer_samples_told": optimizer_samples_told,
        "replay_candidate_evaluations": optimizer_samples_told + 1,
        "final_mean_evaluations": 1,
        "physical_env_slots": 12,
        "stop_kind": stop_kind,
        "stop_conditions": stop_conditions,
        "generations": [],
        "final_mean": {
            "log10_e": list(log10_e),
            "e_pa": e_pa,
            "aggregate_sinkhorn": aggregate_sinkhorn,
            "per_direction_sinkhorn": per_direction,
        },
        "best_sample": {
            "log10_e": list(log10_e),
            "e_pa": e_pa,
            "fitness": aggregate_sinkhorn,
        },
        "gt": {
            "e_pa": [1e8, 1e7, 1e6],
            "log10_e": [8.0, 7.0, 6.0],
        },
        "evaluated_history_extrema": extrema,
        "covariance": covariance,
        "failure": None,
        "artifact_errors": [],
    }


def _cmaes_report(structure_indices: list[int] | None = None) -> dict:
    if structure_indices is None:
        structure_indices = [0, 1, 2, 3, 4]
    structures = {
        str(idx): _fitted_structure(idx) for idx in structure_indices
    }
    return {
        "structures": structures,
        "aggregate": {
            "requested_structures": len(structures),
            "fitted_structures": len(structures),
            "failed_structures": 0,
            "mean_log10_e": [7.5, 6.5, 5.5],
            "geometric_mean_e_pa": [10**7.5, 10**6.5, 10**5.5],
            "mean_e_pa": [10**7.5, 10**6.5, 10**5.5],
            "min_e_pa": [10**7.5, 10**6.5, 10**5.5],
            "max_e_pa": [10**7.5, 10**6.5, 10**5.5],
            "mean_gt_e_pa": [1e8, 1e7, 1e6],
            "sample_cov_log10_e": None,
            "sample_std_log10_e": None,
        },
    }


def _write_report(tmp_path: Path, payload: dict, name: str = "cmaes_report.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_seed(
    tmp_path: Path,
    *,
    seed: int,
    n_structures: int = 5,
    expected_structures: int | None = None,
) -> tuple[dict, Path]:
    report_path = _write_report(
        tmp_path,
        _cmaes_report(list(range(n_structures))),
        f"cmaes_report_seed{seed}.json",
    )
    summary_path = tmp_path / f"seed{seed}_summary.json"
    summary = write_seed_summary(
        seed=seed,
        cmaes_json=report_path,
        out_summary=summary_path,
        expected_structures=expected_structures or n_structures,
    )
    return summary, summary_path


def test_extracts_fitted_structure_integrity_evidence():
    evidence = extract_structure_evidence(_fitted_structure(2))

    assert evidence["structure_idx"] == 2
    assert evidence["status"] == "fitted"
    assert evidence["final_mean_log10_e"] == [7.5, 6.5, 5.5]
    assert evidence["final_mean_aggregate_sinkhorn"] == 0.12
    assert evidence["final_mean_per_direction_sinkhorn"] == {"0": 0.1, "1": 0.14}
    assert evidence["stop_kind"] == "generation_cap"
    assert evidence["completed_generations"] == 2
    assert evidence["final_mean_evaluations"] == 1
    assert evidence["covariance_present"] is True
    assert evidence["evaluated_history_extrema_present"] is True


def test_seed_passes_when_every_expected_structure_fitted(tmp_path: Path):
    summary, output = _write_seed(tmp_path, seed=7, n_structures=5)

    assert summary["seed"] == 7
    assert summary["expected_structures"] == 5
    assert summary["n_fitted"] == 5
    assert summary["passed"] is True
    assert len(summary["structures"]) == 5
    assert "pass_threshold" not in summary
    assert "n_gt_rank_1" not in summary
    assert json.loads(output.read_text(encoding="utf-8")) == summary


def test_seed_fails_when_any_structure_not_fitted(tmp_path: Path):
    report = _cmaes_report([0, 1, 2, 3, 4])
    report["structures"]["2"]["status"] = "failed"
    report["structures"]["2"]["failure"] = {
        "stage": "all_invalid",
        "message": "all samples invalid",
    }
    report["structures"]["2"]["final_mean"] = None
    report["aggregate"]["fitted_structures"] = 4
    report["aggregate"]["failed_structures"] = 1
    ranking_path = _write_report(tmp_path, report)

    summary = write_seed_summary(
        seed=0,
        cmaes_json=ranking_path,
        out_summary=tmp_path / "seed0_summary.json",
        expected_structures=5,
    )

    assert summary["passed"] is False
    assert summary["n_fitted"] == 4


def test_does_not_impose_gt_error_threshold(tmp_path: Path):
    """Large GT relative error must not fail an otherwise valid integrity seed."""
    report = _cmaes_report([0])
    # Far from stored GT but still inside fixture bounds.
    report["structures"]["0"] = _fitted_structure(0, log10_e=(9.5, 8.5, 7.5))
    report_path = _write_report(tmp_path, report)

    summary = write_seed_summary(
        seed=3,
        cmaes_json=report_path,
        out_summary=tmp_path / "seed3_summary.json",
        expected_structures=1,
    )

    assert summary["passed"] is True
    assert summary["structures"][0]["final_mean_log10_e"] == [9.5, 8.5, 7.5]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda structure: structure["final_mean"].update(
                {"log10_e": [11.0, 6.5, 5.5]}
            ),
            "bounds",
        ),
        (
            lambda structure: structure["final_mean"].update(
                {"aggregate_sinkhorn": None}
            ),
            "aggregate",
        ),
        (
            lambda structure: structure["final_mean"].update(
                {"aggregate_sinkhorn": float("inf")}
            ),
            "finite",
        ),
        (
            lambda structure: structure["final_mean"].update(
                {"per_direction_sinkhorn": {}}
            ),
            "per-direction",
        ),
        (
            lambda structure: structure.update({"stop_kind": None}),
            "stop",
        ),
        (
            lambda structure: structure.update(
                {"stop_kind": "pycma", "stop_conditions": {}}
            ),
            "stop",
        ),
        (
            lambda structure: structure.update(
                {"optimizer_samples_told": 3, "completed_generations": 2}
            ),
            "coherent|samples|population",
        ),
        (
            lambda structure: structure.update({"final_mean_evaluations": 0}),
            "final_mean_evaluations",
        ),
    ],
)
def test_rejects_invalid_fitted_structure_fields(mutate, match: str):
    structure = _fitted_structure(0)
    mutate(structure)

    with pytest.raises(ValueError, match=match):
        extract_structure_evidence(structure)


@pytest.mark.parametrize(
    "bad_covariance",
    [
        {"sigma": float("nan"), "C": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        {
            "C": [[1.0, 0.0], [0.0, 1.0]],
            "sigma": 0.5,
            "sigma_vec_scaling": [1.0, 1.0, 1.0],
            "phenotype_std": [0.5, 0.5, 0.5],
            "effective_unbounded_covariance": [
                [0.25, 0.0, 0.0],
                [0.0, 0.25, 0.0],
                [0.0, 0.0, 0.25],
            ],
        },
        {
            "C": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "sigma": float("inf"),
            "sigma_vec_scaling": [1.0, 1.0, 1.0],
            "phenotype_std": [0.5, 0.5, 0.5],
            "effective_unbounded_covariance": [
                [0.25, 0.0, 0.0],
                [0.0, 0.25, 0.0],
                [0.0, 0.0, 0.25],
            ],
        },
    ],
)
def test_rejects_non_finite_or_incoherent_covariance_when_present(bad_covariance):
    structure = _fitted_structure(0, covariance=bad_covariance)

    with pytest.raises(ValueError, match="covariance"):
        extract_structure_evidence(structure)


@pytest.mark.parametrize(
    "bad_extrema",
    [
        {
            "min_log10_e": [8.0, 6.0, 5.0],
            "max_log10_e": [7.0, 7.0, 6.0],
            "min_e_pa": [1e8, 1e6, 1e5],
            "max_e_pa": [1e7, 1e7, 1e6],
        },
        {
            "min_log10_e": [7.0, 6.0, float("nan")],
            "max_log10_e": [8.0, 7.0, 6.0],
            "min_e_pa": [1e7, 1e6, 1e5],
            "max_e_pa": [1e8, 1e7, 1e6],
        },
        {
            "min_log10_e": [7.0, 6.0, 5.0],
            "max_log10_e": [8.0, 7.0, 6.0],
            "min_e_pa": [1e9, 1e6, 1e5],
            "max_e_pa": [1e8, 1e7, 1e6],
        },
    ],
)
def test_rejects_non_finite_or_incoherent_evaluated_history_extrema_when_present(
    bad_extrema,
):
    structure = _fitted_structure(0, extrema=bad_extrema)

    with pytest.raises(ValueError, match="extrema|history|min|max"):
        extract_structure_evidence(structure)


def test_allows_null_covariance_and_null_extrema():
    structure = _fitted_structure(
        0,
        include_covariance=False,
        include_extrema=False,
    )
    structure["covariance"] = None
    structure["evaluated_history_extrema"] = {
        "min_log10_e": None,
        "max_log10_e": None,
        "min_e_pa": None,
        "max_e_pa": None,
    }

    evidence = extract_structure_evidence(structure)

    assert evidence["covariance_present"] is False
    assert evidence["evaluated_history_extrema_present"] is False


def test_expected_structure_count_is_configurable(tmp_path: Path):
    summary, _ = _write_seed(
        tmp_path, seed=0, n_structures=3, expected_structures=3
    )

    assert summary["expected_structures"] == 3
    assert summary["n_fitted"] == 3
    assert summary["passed"] is True


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda report: report["structures"].pop("4"), "expected 5"),
        (
            lambda report: report["aggregate"].update({"fitted_structures": 4}),
            "fitted",
        ),
        (
            lambda report: report["aggregate"].update({"failed_structures": 1}),
            "failed",
        ),
        (
            lambda report: report["structures"]["0"].update({"structure_idx": 1}),
            "unique|structure_idx|key",
        ),
    ],
)
def test_rejects_invalid_cmaes_reports(tmp_path: Path, mutate, match: str):
    report = _cmaes_report([0, 1, 2, 3, 4])
    mutate(report)
    report_path = _write_report(tmp_path, report)

    with pytest.raises(ValueError, match=match):
        write_seed_summary(
            seed=0,
            cmaes_json=report_path,
            out_summary=tmp_path / "summary.json",
            expected_structures=5,
        )


@pytest.mark.parametrize("content", ["{not JSON", "[]"])
def test_rejects_malformed_json_or_report_shape(tmp_path: Path, content: str):
    report_path = tmp_path / "cmaes_report.json"
    report_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="cmaes report"):
        write_seed_summary(
            seed=0,
            cmaes_json=report_path,
            out_summary=tmp_path / "summary.json",
            expected_structures=5,
        )


def test_seed_cli_writes_failure_evidence_for_invalid_report(tmp_path: Path, capsys):
    report_path = tmp_path / "cmaes_report.json"
    report_path.write_text("{not JSON", encoding="utf-8")
    output = tmp_path / "seed4_summary.json"

    result = main(
        [
            "--seed",
            "4",
            "--cmaes-json",
            str(report_path),
            "--out-summary",
            str(output),
        ]
    )

    assert result == 1
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary == {
        "seed": 4,
        "cmaes_json": str(report_path),
        "passed": False,
        "error": summary["error"],
        "structures": [],
    }
    assert "invalid cmaes report JSON" in summary["error"]
    assert capsys.readouterr().out.strip().startswith("seed=4 passed=False error=")


def test_seed_summary_is_strict_json(tmp_path: Path):
    _, output = _write_seed(tmp_path, seed=0, n_structures=2)

    text = output.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    json.loads(text, parse_constant=lambda value: pytest.fail(f"non-finite {value}"))


def test_finalize_all_pass_preserves_seed_and_structure_evidence(tmp_path: Path):
    seed0, _ = _write_seed(tmp_path, seed=0, n_structures=2)
    seed1, _ = _write_seed(tmp_path, seed=1, n_structures=2)
    output = tmp_path / "summary.json"

    summary = finalize_gate(report_dir=tmp_path, seeds=[0, 1], out_summary=output)

    assert summary["passed"] is True
    assert summary["seeds"] == [seed0, seed1]
    assert json.loads(output.read_text(encoding="utf-8")) == summary


def test_finalize_rejects_duplicate_seeds(tmp_path: Path):
    with pytest.raises(ValueError, match="duplicate seed"):
        finalize_gate(report_dir=tmp_path, seeds=[0, 0])


def test_finalize_partial_failure_fails_and_preserves_evidence(tmp_path: Path):
    _write_seed(tmp_path, seed=0, n_structures=2)
    report = _cmaes_report([0, 1])
    report["structures"]["1"]["status"] = "failed"
    report["structures"]["1"]["final_mean"] = None
    report["aggregate"]["fitted_structures"] = 1
    report["aggregate"]["failed_structures"] = 1
    failed = write_seed_summary(
        seed=1,
        cmaes_json=_write_report(tmp_path, report, "cmaes_report_seed1.json"),
        out_summary=tmp_path / "seed1_summary.json",
        expected_structures=2,
    )

    summary = finalize_gate(
        report_dir=tmp_path,
        seeds=[0, 1],
        out_summary=tmp_path / "summary.json",
    )

    assert summary["passed"] is False
    assert summary["seeds"][1] == failed
    assert len(summary["seeds"][1]["structures"]) == 2


def test_finalize_retains_valid_missing_and_malformed_seed_evidence(tmp_path: Path):
    valid, _ = _write_seed(tmp_path, seed=0, n_structures=2)
    (tmp_path / "seed2_summary.json").write_text("{bad JSON", encoding="utf-8")
    output = tmp_path / "summary.json"

    summary = finalize_gate(
        report_dir=tmp_path,
        seeds=[0, 1, 2],
        out_summary=output,
    )

    assert summary["passed"] is False
    assert summary["n_passed_seeds"] == 1
    assert summary["seeds"][0] == valid
    assert summary["seeds"][1]["seed"] == 1
    assert summary["seeds"][1]["passed"] is False
    assert summary["seeds"][1]["structures"] == []
    assert "invalid seed 1 summary JSON" in summary["seeds"][1]["error"]
    assert summary["seeds"][2]["seed"] == 2
    assert summary["seeds"][2]["passed"] is False
    assert "invalid seed 2 summary JSON" in summary["seeds"][2]["error"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda summary: summary.update({"structures": []}),
        lambda summary: summary.update({"n_fitted": 1}),
        lambda summary: summary.update({"passed": False}),
        lambda summary: summary["structures"][1].update(
            {"structure_idx": summary["structures"][0]["structure_idx"]}
        ),
        lambda summary: summary.update({"error": "stale failure"}),
    ],
)
def test_finalize_rejects_fake_or_inconsistent_passed_seed_summaries(
    tmp_path: Path, mutate
):
    seed, path = _write_seed(tmp_path, seed=0, n_structures=2)
    mutate(seed)
    path.write_text(json.dumps(seed), encoding="utf-8")

    summary = finalize_gate(report_dir=tmp_path, seeds=[0])

    assert summary["passed"] is False
    assert summary["n_passed_seeds"] == 0
    assert summary["seeds"][0]["passed"] is False
    assert summary["seeds"][0]["structures"] == []
    assert "invalid" in summary["seeds"][0]["error"]


def test_finalize_preserves_explicit_failed_validation_summary(tmp_path: Path):
    failed = {
        "seed": 0,
        "cmaes_json": "cmaes_report.json",
        "passed": False,
        "error": "cmaes report malformed",
        "structures": [],
    }
    (tmp_path / "seed0_summary.json").write_text(
        json.dumps(failed), encoding="utf-8"
    )

    summary = finalize_gate(report_dir=tmp_path, seeds=[0])

    assert summary["passed"] is False
    assert summary["seeds"] == [failed]


def test_finalize_cli_reports_partial_evidence_without_traceback(tmp_path: Path, capsys):
    _write_seed(tmp_path, seed=0, n_structures=2)
    output = tmp_path / "summary.json"

    result = main(
        [
            "--finalize",
            "--report-dir",
            str(tmp_path),
            "--seeds",
            "0,1",
            "--out-summary",
            str(output),
        ]
    )

    assert result == 1
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["passed"] is False
    assert [seed["passed"] for seed in summary["seeds"]] == [True, False]
    assert capsys.readouterr().out.strip() == (
        "seeds=2 passed=False passed_seeds=1/2"
    )


@pytest.mark.parametrize("value", ["1,1", "1,01", "+1,1"])
def test_parse_seeds_rejects_duplicate_integer_values(value: str):
    with pytest.raises(argparse.ArgumentTypeError, match="duplicate seed"):
        _parse_seeds(value)


def test_cli_exit_codes_and_concise_output(tmp_path: Path, capsys):
    report_path = _write_report(tmp_path, _cmaes_report([0, 1]))
    seed_output = tmp_path / "seed0_summary.json"

    assert (
        main(
            [
                "--seed",
                "0",
                "--cmaes-json",
                str(report_path),
                "--out-summary",
                str(seed_output),
                "--expected-structures",
                "2",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "seed=0 passed=True fitted=2/2"

    failed_report = _cmaes_report([0, 1])
    failed_report["structures"]["1"]["status"] = "failed"
    failed_report["structures"]["1"]["final_mean"] = None
    failed_report["aggregate"]["fitted_structures"] = 1
    failed_report["aggregate"]["failed_structures"] = 1
    assert (
        main(
            [
                "--seed",
                "1",
                "--cmaes-json",
                str(_write_report(tmp_path, failed_report, "failed.json")),
                "--out-summary",
                str(tmp_path / "seed1_summary.json"),
                "--expected-structures",
                "2",
            ]
        )
        == 1
    )

    assert (
        main(
            [
                "--finalize",
                "--report-dir",
                str(tmp_path),
                "--seeds",
                "0,1",
                "--out-summary",
                str(tmp_path / "summary.json"),
            ]
        )
        == 1
    )
    assert capsys.readouterr().out.splitlines()[-1] == (
        "seeds=2 passed=False passed_seeds=1/2"
    )
