"""Tests for sys-ID Sinkhorn gate reporting."""

from __future__ import annotations

import json
from pathlib import Path

from apple_pick_gym.batched_envs.sysid_gate_report import (
    compare_gates,
    evaluate_structure_cell,
    finalize_gate,
    write_seed_summary,
)


def test_evaluate_structure_cell_rank_le2_passes():
    assert evaluate_structure_cell({"gt_rank": 1, "gt_disqualified": False})["passed"]
    assert evaluate_structure_cell({"gt_rank": 2, "gt_disqualified": False})["passed"]
    assert not evaluate_structure_cell({"gt_rank": 3, "gt_disqualified": False})["passed"]
    assert not evaluate_structure_cell({"gt_rank": 1, "gt_disqualified": True})["passed"]


def test_write_seed_and_finalize_and_compare(tmp_path: Path):
    score = {
        "structures": [
            {
                "structure_idx": 0,
                "gt_candidate_index": 0,
                "gt_rank": 1,
                "best_is_gt": True,
                "best_candidate_index": 0,
                "gt_disqualified": False,
                "gt_disqualify_reasons": [],
                "n_disqualified": 0,
                "n_candidates": 2,
                "gt_aggregate_sinkhorn": 0.1,
                "best_aggregate_sinkhorn": 0.1,
                "per_candidate": [
                    {
                        "candidate_index": 0,
                        "aggregate_sinkhorn": 0.1,
                        "disqualified": False,
                        "disqualify_reasons": [],
                        "stiffnesses": {},
                        "n_transitions": {0: 7},
                        "low_sample_directions": [],
                        "err_pos_hold": 0.01,
                        "err_force_hold": 0.02,
                        "err_torque_hold": 0.03,
                    }
                ],
            }
        ]
    }
    report = tmp_path / "gate_median_hold"
    report.mkdir()
    score_path = report / "seed0_scores.json"
    score_path.write_text(json.dumps(score), encoding="utf-8")
    summary = write_seed_summary(
        gate="gate_median_hold",
        seed=0,
        score_json=score_path,
        dataset="/tmp/ds",
        plot_output="/tmp/plots",
        report_dir=report,
        out_summary=report / "seed0_summary.json",
    )
    assert summary["passed"]
    finalized = finalize_gate(gate="gate_median_hold", report_dir=report, seeds=[0])
    assert finalized["passed"]

    # Clone for other gates with worse ranks to exercise compare.
    reports = {}
    for gate, rank in (
        ("gate_median_hold", 1),
        ("gate_hold_id", 2),
        ("gate_pooled_dirs", 1),
    ):
        d = tmp_path / gate
        d.mkdir(exist_ok=True)
        score2 = json.loads(json.dumps(score))
        score2["structures"][0]["gt_rank"] = rank
        score2["structures"][0]["best_is_gt"] = rank == 1
        (d / "seed0_scores.json").write_text(json.dumps(score2), encoding="utf-8")
        write_seed_summary(
            gate=gate,
            seed=0,
            score_json=d / "seed0_scores.json",
            dataset="/tmp/ds",
            plot_output="/tmp/plots",
            report_dir=d,
            out_summary=d / "seed0_summary.json",
        )
        finalize_gate(gate=gate, report_dir=d, seeds=[0])
        reports[gate] = d

    out = tmp_path / "compare"
    compare = compare_gates(report_dirs=reports, out_dir=out)
    assert (out / "COMPARE.md").exists()
    assert compare["best_gate"] in ("gate_median_hold", "gate_pooled_dirs")
