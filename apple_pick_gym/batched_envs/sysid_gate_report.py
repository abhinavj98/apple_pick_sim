"""Gate summary / failure / cross-gate comparison for sys-ID Sinkhorn hardening."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GATE_PASS_MAX_RANK = 2
GATE_ORDER = ("gate_median_hold", "gate_hold_id", "gate_pooled_dirs")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gt_mse_field(structure: dict[str, Any], key: str) -> float | None:
    gt_idx = structure.get("gt_candidate_index")
    if gt_idx is None:
        return None
    for cand in structure.get("per_candidate") or []:
        if int(cand.get("candidate_index", -1)) == int(gt_idx):
            val = cand.get(key)
            return None if val is None else float(val)
    return None


def evaluate_structure_cell(structure: dict[str, Any]) -> dict[str, Any]:
    gt_rank = structure.get("gt_rank")
    gt_disqualified = bool(structure.get("gt_disqualified", False))
    passed = (
        not gt_disqualified
        and gt_rank is not None
        and int(gt_rank) <= GATE_PASS_MAX_RANK
    )
    return {
        "passed": bool(passed),
        "gt_rank": gt_rank,
        "gt_disqualified": gt_disqualified,
        "gt_disqualify_reasons": list(structure.get("gt_disqualify_reasons") or []),
        "best_is_gt": structure.get("best_is_gt"),
        "best_candidate_index": structure.get("best_candidate_index"),
        "gt_aggregate_sinkhorn": structure.get("gt_aggregate_sinkhorn"),
        "best_aggregate_sinkhorn": structure.get("best_aggregate_sinkhorn"),
        "err_pos_hold_gt": _gt_mse_field(structure, "err_pos_hold"),
        "err_force_hold_gt": _gt_mse_field(structure, "err_force_hold"),
        "err_torque_hold_gt": _gt_mse_field(structure, "err_torque_hold"),
    }

def write_seed_summary(
    *,
    gate: str,
    seed: int,
    score_json: Path,
    dataset: str,
    plot_output: str,
    report_dir: Path,
    out_summary: Path,
) -> dict[str, Any]:
    payload = _load_json(score_json)
    cells = []
    failed = []
    for structure in payload.get("structures") or []:
        cell = evaluate_structure_cell(structure)
        cell["structure_idx"] = int(structure["structure_idx"])
        cells.append(cell)
        if not cell["passed"]:
            failed.append(cell)

    summary = {
        "gate": gate,
        "seed": int(seed),
        "dataset": str(dataset),
        "plot_output": str(plot_output),
        "score_json": str(score_json),
        "n_structures": len(cells),
        "n_passed": sum(1 for c in cells if c["passed"]),
        "n_failed": len(failed),
        "passed": len(failed) == 0 and len(cells) > 0,
        "cells": cells,
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if failed:
        lines = [
            f"# Failure: {gate} seed={seed}",
            "",
            f"- dataset: `{dataset}`",
            f"- plots: `{plot_output}`",
            f"- score_json: `{score_json}`",
            "",
            "## Failed structures",
            "",
        ]
        for cell in failed:
            lines.append(
                f"### structure {cell['structure_idx']}: "
                f"gt_rank={cell['gt_rank']} gt_disqualified={cell['gt_disqualified']}"
            )
            if cell["gt_disqualify_reasons"]:
                lines.append(
                    "- reasons: " + ", ".join(str(r) for r in cell["gt_disqualify_reasons"])
                )
            lines.append(
                f"- sinkhorn GT={cell['gt_aggregate_sinkhorn']} "
                f"best={cell['best_aggregate_sinkhorn']} "
                f"best_candidate={cell['best_candidate_index']}"
            )
            lines.append("")
        (report_dir / f"failure_seed{seed}.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    return summary


def finalize_gate(*, gate: str, report_dir: Path, seeds: list[int]) -> dict[str, Any]:
    seed_summaries = []
    for seed in seeds:
        path = report_dir / f"seed{seed}_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"missing seed summary: {path}")
        seed_summaries.append(_load_json(path))

    all_cells = []
    for ss in seed_summaries:
        for cell in ss.get("cells") or []:
            row = dict(cell)
            row["seed"] = int(ss["seed"])
            all_cells.append(row)

    n_rank1 = sum(1 for c in all_cells if c.get("gt_rank") == 1)
    n_rank2 = sum(1 for c in all_cells if c.get("gt_rank") == 2)
    n_pass = sum(1 for c in all_cells if c.get("passed"))
    n_disq = sum(1 for c in all_cells if c.get("gt_disqualified"))
    summary = {
        "gate": gate,
        "report_dir": str(report_dir),
        "seeds": [int(s) for s in seeds],
        "n_cells": len(all_cells),
        "n_passed": n_pass,
        "n_failed": len(all_cells) - n_pass,
        "n_gt_rank_1": n_rank1,
        "n_gt_rank_2": n_rank2,
        "n_gt_disqualified": n_disq,
        "passed": n_pass == len(all_cells) and len(all_cells) > 0,
        "cells": all_cells,
    }
    out = report_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not summary["passed"]:
        (report_dir / "failure.md").write_text(
            f"# Gate failed: {gate}\n\n"
            f"passed {n_pass}/{len(all_cells)}; "
            f"rank1={n_rank1} rank2={n_rank2} disqualified={n_disq}\n"
            f"See per-seed `failure_seed*.md` and `{out}`.\n",
            encoding="utf-8",
        )
    return summary


def compare_gates(*, report_dirs: dict[str, Path], out_dir: Path) -> dict[str, Any]:
    """Build COMPARE.md / compare.json across named gates."""
    gate_summaries: dict[str, dict[str, Any]] = {}
    for gate, path in report_dirs.items():
        summary_path = path / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing {summary_path}")
        gate_summaries[gate] = _load_json(summary_path)

    # Index cells by (seed, structure)
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for gate, summary in gate_summaries.items():
        for cell in summary.get("cells") or []:
            key = (int(cell["seed"]), int(cell["structure_idx"]))
            by_key.setdefault(key, {})[gate] = cell

    rank_rows = []
    mse_rows = []
    for key in sorted(by_key.keys()):
        seed, structure_idx = key
        rank_row: dict[str, Any] = {"seed": seed, "structure_idx": structure_idx}
        mse_row: dict[str, Any] = {"seed": seed, "structure_idx": structure_idx}
        for gate in GATE_ORDER:
            cell = by_key[key].get(gate)
            if cell is None:
                continue
            rank_row[f"{gate}_gt_rank"] = cell.get("gt_rank")
            rank_row[f"{gate}_best_is_gt"] = cell.get("best_is_gt")
            rank_row[f"{gate}_disqualified"] = cell.get("gt_disqualified")
            mse_row[f"{gate}_err_pos"] = cell.get("err_pos_hold_gt")
            mse_row[f"{gate}_err_force"] = cell.get("err_force_hold_gt")
            mse_row[f"{gate}_err_torque"] = cell.get("err_torque_hold_gt")
        rank_rows.append(rank_row)
        mse_rows.append(mse_row)

    scores: dict[str, dict[str, Any]] = {}
    for gate in GATE_ORDER:
        summary = gate_summaries.get(gate)
        if summary is None:
            continue
        scores[gate] = {
            "frac_rank1": (
                float(summary["n_gt_rank_1"]) / float(summary["n_cells"])
                if summary["n_cells"]
                else 0.0
            ),
            "frac_pass_rank_le2": (
                float(summary["n_passed"]) / float(summary["n_cells"])
                if summary["n_cells"]
                else 0.0
            ),
            "n_gt_disqualified": int(summary["n_gt_disqualified"]),
            "n_cells": int(summary["n_cells"]),
        }

    # Primary: frac_rank1, then frac_pass, then fewer disqualifications
    ranked = sorted(
        scores.items(),
        key=lambda item: (
            -item[1]["frac_rank1"],
            -item[1]["frac_pass_rank_le2"],
            item[1]["n_gt_disqualified"],
        ),
    )
    best_gate = ranked[0][0] if ranked else None

    changes = {
        "gate_median_hold": (
            "Full-hold median hold→hold transition bags; per-direction Sinkhorn; "
            "paired per-hold median MSE."
        ),
        "gate_hold_id": (
            "Same as gate_median_hold plus hold_number one-hot on bag rows "
            "(still per-direction Sinkhorn)."
        ),
        "gate_pooled_dirs": (
            "Same as gate_hold_id but pools bags across directions for Sinkhorn "
            "(more samples, diluted per-dir OT)."
        ),
    }

    compare = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gate_report_dirs": {g: str(p) for g, p in report_dirs.items()},
        "scores": scores,
        "best_gate": best_gate,
        "rank_rows": rank_rows,
        "mse_rows": mse_rows,
        "what_changed": changes,
        "recommendation": (
            f"Prefer `{best_gate}` for further V.5 / CEM work based on primary "
            "ranking criteria (rank1 fraction, then rank≤2 pass rate, then fewer GT disqualifications)."
            if best_gate
            else "No gate summaries available."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare.json").write_text(
        json.dumps(compare, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    md = [
        "# Sys-ID gate comparison",
        "",
        f"Best gate (by rank-1 fraction, then pass≤2, then fewer GT DQ): **{best_gate}**",
        "",
        "## What changed",
        "",
    ]
    for gate in GATE_ORDER:
        md.append(f"- **{gate}**: {changes[gate]}")
    md.extend(["", "## Aggregate scores", ""])
    for gate, sc in scores.items():
        md.append(
            f"- `{gate}`: frac_rank1={sc['frac_rank1']:.3f} "
            f"frac_pass_le2={sc['frac_pass_rank_le2']:.3f} "
            f"gt_disqualified={sc['n_gt_disqualified']}/{sc['n_cells']}"
        )
    md.extend(["", "## Sinkhorn ranks by seed×structure", ""])
    header = "| seed | structure | " + " | ".join(GATE_ORDER) + " |"
    sep = "| --- | --- | " + " | ".join(["---"] * len(GATE_ORDER)) + " |"
    md.extend([header, sep])
    for row in rank_rows:
        cols = [
            str(row.get(f"{g}_gt_rank", ""))
            + (" DQ" if row.get(f"{g}_disqualified") else "")
            for g in GATE_ORDER
        ]
        md.append(
            f"| {row['seed']} | {row['structure_idx']} | " + " | ".join(cols) + " |"
        )
    md.extend(["", "## Hold MSE (GT candidate) pos / force / torque", ""])
    md.append(
        "| seed | structure | "
        + " | ".join(f"{g} pos/force/torque" for g in GATE_ORDER)
        + " |"
    )
    md.append("| --- | --- | " + " | ".join(["---"] * len(GATE_ORDER)) + " |")
    for row in mse_rows:
        cols = []
        for g in GATE_ORDER:
            cols.append(
                f"{row.get(f'{g}_err_pos')}/{row.get(f'{g}_err_force')}/{row.get(f'{g}_err_torque')}"
            )
        md.append(
            f"| {row['seed']} | {row['structure_idx']} | " + " | ".join(cols) + " |"
        )
    md.extend(["", "## Recommendation", "", compare["recommendation"], ""])
    (out_dir / "COMPARE.md").write_text("\n".join(md), encoding="utf-8")
    return compare


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--score-json", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--plot-output", type=str, default=None)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, default=None)
    parser.add_argument("--finalize-gate", action="store_true")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument(
        "--gate-report-dirs",
        type=str,
        default=None,
        help="JSON map of gate name -> report dir for --compare",
    )
    parser.add_argument("--compare-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.compare:
        if not args.gate_report_dirs or not args.compare_out:
            raise SystemExit("--compare requires --gate-report-dirs and --compare-out")
        mapping = {
            str(k): Path(v) for k, v in json.loads(args.gate_report_dirs).items()
        }
        compare_gates(report_dirs=mapping, out_dir=Path(args.compare_out))
        print(f"wrote compare under {args.compare_out}")
        return 0

    if args.finalize_gate:
        if not args.gate:
            raise SystemExit("--finalize-gate requires --gate")
        seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
        summary = finalize_gate(gate=str(args.gate), report_dir=Path(args.report_dir), seeds=seeds)
        print(
            f"gate={args.gate} passed={summary['passed']} "
            f"{summary['n_passed']}/{summary['n_cells']}"
        )
        return 0 if summary["passed"] else 1

    if (
        args.gate is None
        or args.seed is None
        or args.score_json is None
        or args.out_summary is None
    ):
        raise SystemExit("seed mode requires --gate --seed --score-json --out-summary")
    summary = write_seed_summary(
        gate=str(args.gate),
        seed=int(args.seed),
        score_json=Path(args.score_json),
        dataset=str(args.dataset or ""),
        plot_output=str(args.plot_output or ""),
        report_dir=Path(args.report_dir),
        out_summary=Path(args.out_summary),
    )
    print(
        f"seed={args.seed} passed={summary['passed']} "
        f"{summary['n_passed']}/{summary['n_structures']}"
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
