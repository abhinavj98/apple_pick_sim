"""Tests for the Young's-modulus ranking gate reporter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from apple_pick_gym.batched_envs.youngs_modulus_gate_report import (
    _parse_seeds,
    extract_structure_evidence,
    finalize_gate,
    main,
    write_seed_summary,
)


def _structure(
    structure_idx: int,
    *,
    gt_rank: int | None,
    gt_score: float | None = 0.1,
    gt_per_direction: dict | None = None,
    gt_count: int = 1,
    disqualified: bool = False,
    disqualification_reason: str | None = None,
) -> dict:
    if gt_per_direction is None and gt_score is not None:
        gt_per_direction = {"0": 0.08, "1": 0.12}
    elif gt_per_direction is None:
        gt_per_direction = {}
    candidates = [
        {
            "candidate_index": 10 + structure_idx,
            "aggregate_sinkhorn": gt_score,
            "per_direction_sinkhorn": gt_per_direction,
            "rank": gt_rank,
            "is_gt": True,
            "disqualified": disqualified,
            "disqualification_reason": disqualification_reason,
        }
        for _ in range(gt_count)
    ]
    if gt_rank != 1:
        candidates.append(
            {
                "candidate_index": 100 + structure_idx,
                "aggregate_sinkhorn": 0.05,
                "per_direction_sinkhorn": {"0": 0.04, "1": 0.06},
                "rank": 1,
                "is_gt": False,
                "disqualified": False,
                "disqualification_reason": None,
            }
        )
    winner = next((candidate for candidate in candidates if candidate["rank"] == 1), None)
    return {
        "structure_idx": structure_idx,
        "candidates": candidates,
        "winner": (
            {"candidate_index": winner["candidate_index"]} if winner is not None else None
        ),
        "gt_rank": gt_rank,
    }


def _ranking(ranks: list[int | None]) -> dict:
    structures = [
        _structure(structure_idx, gt_rank=rank)
        for structure_idx, rank in enumerate(ranks)
    ]
    return {
        "structures": structures,
        "skipped_structures": [],
        "aggregate": {
            "n_structures": len(structures),
            "n_evaluated": len(structures),
            "n_skipped": 0,
        },
    }


def _write_ranking(tmp_path: Path, payload: dict, name: str = "ranking.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_seed(
    tmp_path: Path,
    *,
    seed: int,
    ranks: list[int | None],
    expected_structures: int | None = None,
    threshold: int | None = None,
) -> tuple[dict, Path]:
    ranking_path = _write_ranking(tmp_path, _ranking(ranks), f"ranking_seed{seed}.json")
    summary_path = tmp_path / f"seed{seed}_summary.json"
    summary = write_seed_summary(
        seed=seed,
        ranking_json=ranking_path,
        out_summary=summary_path,
        expected_structures=expected_structures or len(ranks),
        pass_threshold=threshold,
    )
    return summary, summary_path


def test_extracts_complete_gt_and_winner_evidence():
    evidence = extract_structure_evidence(
        _structure(
            2,
            gt_rank=2,
            disqualified=True,
            disqualification_reason="replay_instability",
        )
    )

    assert evidence == {
        "structure_idx": 2,
        "gt_candidate_index": 12,
        "gt_aggregate_sinkhorn": 0.1,
        "gt_per_direction_sinkhorn": {"0": 0.08, "1": 0.12},
        "gt_rank": 2,
        "gt_disqualified": True,
        "gt_disqualification_reason": "replay_instability",
        "winner_candidate_index": 102,
        "winner_is_gt": False,
    }


def test_disqualified_gt_with_null_rank_preserves_evidence_and_fails_seed(
    tmp_path: Path,
):
    report = _ranking([1, 1, None, 2, 3])
    report["structures"][2] = _structure(
        2,
        gt_rank=None,
        disqualified=True,
        disqualification_reason="replay_instability",
    )
    ranking_path = _write_ranking(tmp_path, report)

    summary = write_seed_summary(
        seed=0,
        ranking_json=ranking_path,
        out_summary=tmp_path / "seed0_summary.json",
        expected_structures=5,
    )

    assert summary["passed"] is False
    assert summary["n_gt_rank_1"] == 2
    assert summary["structures"][2] == {
        "structure_idx": 2,
        "gt_candidate_index": 12,
        "gt_aggregate_sinkhorn": 0.1,
        "gt_per_direction_sinkhorn": {"0": 0.08, "1": 0.12},
        "gt_rank": None,
        "gt_disqualified": True,
        "gt_disqualification_reason": "replay_instability",
        "winner_candidate_index": 102,
        "winner_is_gt": False,
    }


def test_four_of_five_rank_one_passes_with_one_disqualified_gt(tmp_path: Path):
    report = _ranking([1, 1, 1, 1, None])
    report["structures"][4] = _structure(
        4,
        gt_rank=None,
        disqualified=True,
        disqualification_reason="replay_instability",
    )
    ranking_path = _write_ranking(tmp_path, report)
    summary_path = tmp_path / "seed7_summary.json"

    summary = write_seed_summary(
        seed=7,
        ranking_json=ranking_path,
        out_summary=summary_path,
        expected_structures=5,
        pass_threshold=3,
    )

    assert summary["n_gt_rank_1"] == 4
    assert summary["structures"][4]["gt_disqualified"] is True
    assert summary["passed"] is True


def test_four_of_five_passes_with_disqualified_gt_null_empty_scores(tmp_path: Path):
    """DQ empty_transition_bag serializes null/{} and must not veto a 4/5 seed."""
    report = _ranking([1, 1, 1, 1, None])
    report["structures"][4] = _structure(
        4,
        gt_rank=None,
        gt_score=None,
        gt_per_direction={},
        disqualified=True,
        disqualification_reason="empty_transition_bag",
    )
    ranking_path = _write_ranking(tmp_path, report)

    summary = write_seed_summary(
        seed=11,
        ranking_json=ranking_path,
        out_summary=tmp_path / "seed11_summary.json",
        expected_structures=5,
        pass_threshold=4,
    )

    assert summary["n_gt_rank_1"] == 4
    assert summary["passed"] is True
    assert summary["structures"][4] == {
        "structure_idx": 4,
        "gt_candidate_index": 14,
        "gt_aggregate_sinkhorn": None,
        "gt_per_direction_sinkhorn": {},
        "gt_rank": None,
        "gt_disqualified": True,
        "gt_disqualification_reason": "empty_transition_bag",
        "winner_candidate_index": 104,
        "winner_is_gt": False,
    }


@pytest.mark.parametrize("malformed", ["omitted", "null"])
def test_write_rejects_disqualified_unranked_gt_without_empty_diagnostics(
    tmp_path: Path, malformed: str
):
    report = _ranking([1, 1, 1, 1, None])
    report["structures"][4] = _structure(
        4,
        gt_rank=None,
        gt_score=None,
        gt_per_direction={},
        disqualified=True,
        disqualification_reason="empty_transition_bag",
    )
    gt = report["structures"][4]["candidates"][0]
    if malformed == "omitted":
        gt.pop("per_direction_sinkhorn")
    else:
        gt["per_direction_sinkhorn"] = None

    with pytest.raises(ValueError, match="Sinkhorn"):
        write_seed_summary(
            seed=11,
            ranking_json=_write_ranking(tmp_path, report),
            out_summary=tmp_path / "seed11_summary.json",
            expected_structures=5,
            pass_threshold=4,
        )


def test_write_rejects_disqualified_unranked_gt_with_omitted_aggregate(
    tmp_path: Path,
):
    report = _ranking([1, 1, 1, 1, None])
    report["structures"][4] = _structure(
        4,
        gt_rank=None,
        gt_score=None,
        gt_per_direction={},
        disqualified=True,
        disqualification_reason="empty_transition_bag",
    )
    report["structures"][4]["candidates"][0].pop("aggregate_sinkhorn")

    with pytest.raises(ValueError, match="aggregate Sinkhorn"):
        write_seed_summary(
            seed=11,
            ranking_json=_write_ranking(tmp_path, report),
            out_summary=tmp_path / "seed11_summary.json",
            expected_structures=5,
            pass_threshold=4,
        )


@pytest.mark.parametrize("malformed", ["omitted", "null"])
def test_finalize_rejects_disqualified_unranked_gt_without_empty_diagnostics(
    tmp_path: Path, malformed: str
):
    report = _ranking([1, 1, 1, 1, None])
    report["structures"][4] = _structure(
        4,
        gt_rank=None,
        gt_score=None,
        gt_per_direction={},
        disqualified=True,
        disqualification_reason="empty_transition_bag",
    )
    seed_summary = write_seed_summary(
        seed=0,
        ranking_json=_write_ranking(tmp_path, report),
        out_summary=tmp_path / "seed0_summary.json",
        expected_structures=5,
        pass_threshold=4,
    )
    evidence = seed_summary["structures"][4]
    if malformed == "omitted":
        evidence.pop("gt_per_direction_sinkhorn")
    else:
        evidence["gt_per_direction_sinkhorn"] = None
    (tmp_path / "seed0_summary.json").write_text(
        json.dumps(seed_summary), encoding="utf-8"
    )

    final = finalize_gate(report_dir=tmp_path, seeds=[0])

    assert final["passed"] is False
    assert final["seeds"][0]["structures"] == []
    assert "Sinkhorn" in final["seeds"][0]["error"]


def test_finalize_rejects_disqualified_unranked_gt_with_omitted_aggregate(
    tmp_path: Path,
):
    report = _ranking([1, 1, 1, 1, None])
    report["structures"][4] = _structure(
        4,
        gt_rank=None,
        gt_score=None,
        gt_per_direction={},
        disqualified=True,
        disqualification_reason="empty_transition_bag",
    )
    seed_summary = write_seed_summary(
        seed=0,
        ranking_json=_write_ranking(tmp_path, report),
        out_summary=tmp_path / "seed0_summary.json",
        expected_structures=5,
        pass_threshold=4,
    )
    seed_summary["structures"][4].pop("gt_aggregate_sinkhorn")
    (tmp_path / "seed0_summary.json").write_text(
        json.dumps(seed_summary), encoding="utf-8"
    )

    final = finalize_gate(report_dir=tmp_path, seeds=[0])

    assert final["passed"] is False
    assert final["seeds"][0]["structures"] == []
    assert "aggregate Sinkhorn" in final["seeds"][0]["error"]


@pytest.mark.parametrize(
    ("gt_rank", "disqualified", "match"),
    [
        (1, False, "aggregate"),
        (2, True, "aggregate"),
        (None, False, "ranked"),
    ],
)
def test_rejects_null_empty_scores_unless_disqualified_unranked(
    gt_rank: int | None, disqualified: bool, match: str
):
    structure = _structure(
        0,
        gt_rank=gt_rank,
        gt_score=None,
        gt_per_direction={},
        disqualified=disqualified,
        disqualification_reason="empty_transition_bag" if disqualified else None,
    )

    with pytest.raises(ValueError, match=match):
        extract_structure_evidence(structure)


def test_finalize_accepts_seed_summary_with_disqualified_null_gt(tmp_path: Path):
    report = _ranking([1, 1, 1, 1, None])
    report["structures"][4] = _structure(
        4,
        gt_rank=None,
        gt_score=None,
        gt_per_direction={},
        disqualified=True,
        disqualification_reason="empty_transition_bag",
    )
    ranking_path = _write_ranking(tmp_path, report)
    seed_summary = write_seed_summary(
        seed=0,
        ranking_json=ranking_path,
        out_summary=tmp_path / "seed0_summary.json",
        expected_structures=5,
        pass_threshold=4,
    )

    final = finalize_gate(
        report_dir=tmp_path,
        seeds=[0],
        out_summary=tmp_path / "summary.json",
    )

    assert seed_summary["passed"] is True
    assert final["passed"] is True
    assert final["seeds"][0]["structures"][4]["gt_aggregate_sinkhorn"] is None
    assert final["seeds"][0]["structures"][4]["gt_per_direction_sinkhorn"] == {}


def test_all_candidates_disqualified_allows_null_winner():
    structure = _structure(
        3,
        gt_rank=None,
        disqualified=True,
        disqualification_reason="replay_instability",
    )
    for candidate in structure["candidates"]:
        if not candidate["is_gt"]:
            candidate.update(
                rank=None,
                disqualified=True,
                disqualification_reason="replay_instability",
            )
    structure["winner"] = None

    evidence = extract_structure_evidence(structure)

    assert evidence["gt_rank"] is None
    assert evidence["gt_disqualified"] is True
    assert evidence["winner_candidate_index"] is None
    assert evidence["winner_is_gt"] is None


def test_seed_passes_with_default_strict_majority_three_of_five(tmp_path: Path):
    summary, output = _write_seed(tmp_path, seed=7, ranks=[1, 2, 1, 3, 1])

    assert summary["seed"] == 7
    assert summary["expected_structures"] == 5
    assert summary["pass_threshold"] == 3
    assert summary["n_gt_rank_1"] == 3
    assert summary["passed"] is True
    assert len(summary["structures"]) == 5
    assert json.loads(output.read_text(encoding="utf-8")) == summary


def test_seed_fails_with_two_of_five(tmp_path: Path):
    summary, _ = _write_seed(tmp_path, seed=0, ranks=[1, 2, 1, 3, 4])

    assert summary["n_gt_rank_1"] == 2
    assert summary["passed"] is False


def test_expected_count_and_threshold_are_configurable(tmp_path: Path):
    summary, _ = _write_seed(
        tmp_path,
        seed=0,
        ranks=[1, 2, 1, 3],
        expected_structures=4,
        threshold=2,
    )

    assert summary["pass_threshold"] == 2
    assert summary["n_gt_rank_1"] == 2
    assert summary["passed"] is True


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda report: report["skipped_structures"].append(
                {"structure_idx": 4, "error": "failed"}
            ),
            "skipped",
        ),
        (lambda report: report["structures"].pop(), "expected 5"),
        (
            lambda report: report["structures"][0].update({"candidates": []}),
            "exactly one GT",
        ),
        (
            lambda report: report["structures"][0].update(
                {"candidates": _structure(0, gt_rank=1, gt_count=2)["candidates"]}
            ),
            "exactly one GT",
        ),
        (
            lambda report: report["structures"][0]["candidates"][0].update({"rank": None}),
            "ranked",
        ),
        (
            lambda report: report["structures"][0]["candidates"][0].update(
                {"aggregate_sinkhorn": None}
            ),
            "aggregate",
        ),
        (
            lambda report: report["structures"][0]["candidates"][0].update(
                {"aggregate_sinkhorn": float("inf")}
            ),
            "finite",
        ),
    ],
)
def test_rejects_invalid_ranking_reports(
    tmp_path: Path, mutate, match: str
):
    report = _ranking([1, 2, 1, 3, 1])
    mutate(report)
    ranking_path = _write_ranking(tmp_path, report)

    with pytest.raises(ValueError, match=match):
        write_seed_summary(
            seed=0,
            ranking_json=ranking_path,
            out_summary=tmp_path / "summary.json",
            expected_structures=5,
        )


@pytest.mark.parametrize("bad_value", [None, 0, 1, 0.0, 1.0, "true"])
def test_rejects_non_boolean_disqualified(bad_value):
    structure = _structure(0, gt_rank=1)
    structure["candidates"][0]["disqualified"] = bad_value

    with pytest.raises(ValueError, match="disqualified must be boolean"):
        extract_structure_evidence(structure)


@pytest.mark.parametrize("bad_value", [False, 1, 1.0, [], {}])
def test_rejects_non_string_disqualification_reason(bad_value):
    structure = _structure(0, gt_rank=1)
    structure["candidates"][0]["disqualification_reason"] = bad_value

    with pytest.raises(ValueError, match="disqualification_reason"):
        extract_structure_evidence(structure)


@pytest.mark.parametrize("field", ["n_structures", "n_evaluated", "n_skipped"])
@pytest.mark.parametrize("bad_value", [True, 5.0, "5"])
def test_rejects_non_integer_aggregate_counters(
    tmp_path: Path, field: str, bad_value
):
    report = _ranking([1, 2, 1, 3, 1])
    report["aggregate"][field] = bad_value
    ranking_path = _write_ranking(tmp_path, report)

    with pytest.raises(ValueError, match="JSON integer"):
        write_seed_summary(
            seed=0,
            ranking_json=ranking_path,
            out_summary=tmp_path / "summary.json",
            expected_structures=5,
        )


@pytest.mark.parametrize("content", ["{not JSON", "[]"])
def test_rejects_malformed_json_or_report_shape(tmp_path: Path, content: str):
    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="ranking report"):
        write_seed_summary(
            seed=0,
            ranking_json=ranking_path,
            out_summary=tmp_path / "summary.json",
            expected_structures=5,
        )


def test_seed_cli_writes_failure_evidence_for_invalid_ranking(
    tmp_path: Path, capsys
):
    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text("{not JSON", encoding="utf-8")
    output = tmp_path / "seed4_summary.json"

    result = main(
        [
            "--seed",
            "4",
            "--ranking-json",
            str(ranking_path),
            "--out-summary",
            str(output),
        ]
    )

    assert result == 1
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary == {
        "seed": 4,
        "ranking_json": str(ranking_path),
        "passed": False,
        "error": summary["error"],
        "structures": [],
    }
    assert "invalid ranking report JSON" in summary["error"]
    assert capsys.readouterr().out.strip().startswith(
        "seed=4 passed=False error="
    )


def test_seed_cli_writes_failure_evidence_for_ranking_validation_error(
    tmp_path: Path, capsys
):
    report = _ranking([1, 2, 1, 3, 1])
    report["skipped_structures"] = [{"structure_idx": 4}]
    ranking_path = _write_ranking(tmp_path, report)
    output = tmp_path / "seed5_summary.json"

    result = main(
        [
            "--seed",
            "5",
            "--ranking-json",
            str(ranking_path),
            "--out-summary",
            str(output),
        ]
    )

    assert result == 1
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["seed"] == 5
    assert summary["passed"] is False
    assert summary["structures"] == []
    assert "skipped" in summary["error"]
    assert capsys.readouterr().out.strip().startswith(
        "seed=5 passed=False error="
    )


@pytest.mark.parametrize(
    "per_direction",
    [
        {},
        {"not-an-int": 0.1},
        {"0": True},
        {"0": "0.1"},
        {"0": None},
        {"0": float("nan")},
        {"0": float("inf")},
    ],
)
def test_rejects_invalid_gt_per_direction_sinkhorn(per_direction: dict):
    structure = _structure(0, gt_rank=1)
    structure["candidates"][0]["per_direction_sinkhorn"] = per_direction

    with pytest.raises(ValueError, match="per-direction Sinkhorn"):
        extract_structure_evidence(structure)


def test_normalizes_gt_per_direction_sinkhorn():
    structure = _structure(0, gt_rank=1)
    structure["candidates"][0]["per_direction_sinkhorn"] = {
        "01": 1,
        "-2": 2.5,
    }

    evidence = extract_structure_evidence(structure)

    assert evidence["gt_per_direction_sinkhorn"] == {"1": 1.0, "-2": 2.5}


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("structure_idx", True),
        ("structure_idx", 1.0),
        ("structure_idx", "1"),
        ("gt_candidate_index", False),
        ("gt_candidate_index", 10.0),
        ("gt_candidate_index", "10"),
        ("winner_candidate_index", True),
        ("winner_candidate_index", 10.0),
        ("winner_candidate_index", "10"),
        ("rank", False),
        ("rank", 1.0),
        ("rank", "1"),
        ("rank", 0),
    ],
)
def test_rejects_coercive_integer_fields(field: str, bad_value):
    structure = _structure(0, gt_rank=1)
    if field == "structure_idx":
        structure["structure_idx"] = bad_value
    elif field == "gt_candidate_index":
        structure["candidates"][0]["candidate_index"] = bad_value
    elif field == "winner_candidate_index":
        structure["winner"]["candidate_index"] = bad_value
    else:
        structure["candidates"][0]["rank"] = bad_value

    with pytest.raises(ValueError):
        extract_structure_evidence(structure)


def test_winner_must_reference_exactly_one_rank_one_candidate():
    structure = _structure(0, gt_rank=2)
    structure["winner"]["candidate_index"] = structure["candidates"][0]["candidate_index"]

    with pytest.raises(ValueError, match="exactly one candidate with rank 1"):
        extract_structure_evidence(structure)

    structure = _structure(0, gt_rank=1)
    duplicate = dict(structure["candidates"][0])
    duplicate["is_gt"] = False
    structure["candidates"].append(duplicate)
    with pytest.raises(ValueError, match="exactly one candidate with rank 1"):
        extract_structure_evidence(structure)


def test_winner_gt_flag_must_agree_with_unique_gt_index():
    structure = _structure(0, gt_rank=2)
    structure["candidates"][1]["candidate_index"] = structure["candidates"][0][
        "candidate_index"
    ]
    structure["winner"]["candidate_index"] = structure["candidates"][1][
        "candidate_index"
    ]

    with pytest.raises(ValueError, match="GT identity"):
        extract_structure_evidence(structure)


def test_seed_summary_is_strict_json(tmp_path: Path):
    _, output = _write_seed(tmp_path, seed=0, ranks=[1, 2, 1, 3, 1])

    text = output.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    json.loads(text, parse_constant=lambda value: pytest.fail(f"non-finite {value}"))


def test_finalize_all_pass_preserves_seed_and_structure_evidence(tmp_path: Path):
    seed0, _ = _write_seed(tmp_path, seed=0, ranks=[1, 2, 1, 3, 1])
    seed1, _ = _write_seed(tmp_path, seed=1, ranks=[1, 1, 2, 1, 4])
    output = tmp_path / "summary.json"

    summary = finalize_gate(report_dir=tmp_path, seeds=[0, 1], out_summary=output)

    assert summary["passed"] is True
    assert summary["seeds"] == [seed0, seed1]
    assert json.loads(output.read_text(encoding="utf-8")) == summary


def test_finalize_rejects_duplicate_seeds(tmp_path: Path):
    with pytest.raises(ValueError, match="duplicate seed"):
        finalize_gate(report_dir=tmp_path, seeds=[0, 0])


def test_finalize_partial_failure_fails_and_preserves_evidence(tmp_path: Path):
    _write_seed(tmp_path, seed=0, ranks=[1, 2, 1, 3, 1])
    failed, _ = _write_seed(tmp_path, seed=1, ranks=[1, 2, 1, 3, 4])

    summary = finalize_gate(
        report_dir=tmp_path,
        seeds=[0, 1],
        out_summary=tmp_path / "summary.json",
    )

    assert summary["passed"] is False
    assert summary["seeds"][1] == failed
    assert len(summary["seeds"][1]["structures"]) == 5


def test_finalize_retains_valid_missing_and_malformed_seed_evidence(tmp_path: Path):
    valid, _ = _write_seed(tmp_path, seed=0, ranks=[1, 2, 1, 3, 1])
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
    assert json.loads(output.read_text(encoding="utf-8")) == summary


@pytest.mark.parametrize(
    "mutate",
    [
        lambda summary: summary.update({"structures": []}),
        lambda summary: summary.update({"n_gt_rank_1": 2}),
        lambda summary: summary.update({"pass_threshold": 4}),
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
    seed, path = _write_seed(tmp_path, seed=0, ranks=[1, 2, 1, 3, 1])
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
        "ranking_json": "ranking.json",
        "passed": False,
        "error": "ranking report malformed",
        "structures": [],
    }
    (tmp_path / "seed0_summary.json").write_text(
        json.dumps(failed), encoding="utf-8"
    )

    summary = finalize_gate(report_dir=tmp_path, seeds=[0])

    assert summary["passed"] is False
    assert summary["seeds"] == [failed]


def test_finalize_cli_reports_partial_evidence_without_traceback(tmp_path: Path, capsys):
    _write_seed(tmp_path, seed=0, ranks=[1, 2, 1, 3, 1])
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
    ranking_path = _write_ranking(tmp_path, _ranking([1, 2, 1, 3, 1]))
    seed_output = tmp_path / "seed0_summary.json"

    assert main(
        [
            "--seed",
            "0",
            "--ranking-json",
            str(ranking_path),
            "--out-summary",
            str(seed_output),
            "--expected-structures",
            "5",
        ]
    ) == 0
    assert capsys.readouterr().out.strip() == "seed=0 passed=True rank1=3/5 threshold=3"

    failed_ranking = _write_ranking(
        tmp_path, _ranking([1, 2, 1, 3, 4]), "failed_ranking.json"
    )
    assert main(
        [
            "--seed",
            "1",
            "--ranking-json",
            str(failed_ranking),
            "--out-summary",
            str(tmp_path / "seed1_summary.json"),
            "--expected-structures",
            "5",
        ]
    ) == 1

    assert main(
        [
            "--finalize",
            "--report-dir",
            str(tmp_path),
            "--seeds",
            "0,1",
            "--out-summary",
            str(tmp_path / "summary.json"),
        ]
    ) == 1
    assert capsys.readouterr().out.splitlines()[-1] == "seeds=2 passed=False passed_seeds=1/2"
