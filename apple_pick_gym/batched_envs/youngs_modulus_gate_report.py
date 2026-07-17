"""Validate Young's-modulus rankings and report the multi-seed gate result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value}")


def _load_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {description} JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def _write_strict_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"summary is not strict JSON: {exc}") from exc
    path.write_text(text, encoding="utf-8")


def _required_list(payload: dict[str, Any], key: str, *, context: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{context}.{key} must be a list")
    return value


def _json_integer(value: Any, *, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be a JSON integer")
    return value


def _finite_json_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite JSON number")
    return result


def _per_direction_sinkhorn(
    value: Any, *, structure_idx: int
) -> dict[str, float]:
    field = f"structure {structure_idx} GT per-direction Sinkhorn"
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{field} must be a non-empty object")

    normalized: dict[str, float] = {}
    for direction, score in value.items():
        if isinstance(direction, bool) or not isinstance(direction, (str, int)):
            raise ValueError(f"{field} direction IDs must be integer-like")
        try:
            direction_id = int(direction)
        except ValueError as exc:
            raise ValueError(f"{field} direction IDs must be integer-like") from exc
        if isinstance(direction, str) and str(direction_id) != direction.lstrip("+"):
            if not (
                direction.startswith("-")
                and str(direction_id) == direction
            ) and not (
                direction.isdigit() and int(direction) == direction_id
            ):
                raise ValueError(f"{field} direction IDs must be integer-like")
        key = str(direction_id)
        if key in normalized:
            raise ValueError(f"{field} contains duplicate direction ID {key}")
        normalized[key] = _finite_json_number(score, field=field)
    return normalized


def _allows_missing_gt_sinkhorn(
    *,
    disqualified: bool,
    rank: int | None,
    aggregate: Any,
    per_direction: Any,
) -> bool:
    """Allow only the exact serialized empty-bag DQ shape."""
    if not disqualified or rank is not None:
        return False
    if aggregate is not None:
        return False
    return type(per_direction) is dict and not per_direction


def extract_structure_evidence(structure: dict[str, Any]) -> dict[str, Any]:
    """Extract and validate the GT row and winner from one ranking structure."""
    if not isinstance(structure, dict):
        raise ValueError("ranking report structure must be an object")
    try:
        structure_idx = _json_integer(
            structure["structure_idx"], field="structure_idx"
        )
    except (KeyError, ValueError) as exc:
        raise ValueError("ranking report structure has invalid structure_idx") from exc

    candidates = _required_list(
        structure, "candidates", context=f"structure {structure_idx}"
    )
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError(f"structure {structure_idx} candidate must be an object")
        _json_integer(
            candidate.get("candidate_index"),
            field=f"structure {structure_idx} candidate_index",
        )
        rank = candidate.get("rank")
        if rank is not None:
            rank = _json_integer(
                rank, field=f"structure {structure_idx} candidate rank"
            )
            if rank < 1:
                raise ValueError(
                    f"structure {structure_idx} candidate rank must be at least 1"
                )
        if not isinstance(candidate.get("is_gt"), bool):
            raise ValueError(f"structure {structure_idx} candidate is_gt must be boolean")
        if type(candidate.get("disqualified")) is not bool:
            raise ValueError(
                f"structure {structure_idx} candidate disqualified must be boolean"
            )
        reason = candidate.get("disqualification_reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError(
                f"structure {structure_idx} candidate "
                "disqualification_reason must be a string or null"
            )

    gt_candidates = [
        candidate
        for candidate in candidates
        if candidate["is_gt"] is True
    ]
    if len(gt_candidates) != 1:
        raise ValueError(
            f"structure {structure_idx} must contain exactly one GT candidate; "
            f"found {len(gt_candidates)}"
        )
    gt = gt_candidates[0]

    gt_candidate_index = gt["candidate_index"]
    gt_rank = gt.get("rank")
    gt_disqualified = gt["disqualified"]
    if gt_rank is None and not gt_disqualified:
        raise ValueError(f"structure {structure_idx} GT candidate must be ranked")

    if "aggregate_sinkhorn" not in gt:
        raise ValueError(
            f"structure {structure_idx} GT candidate is missing aggregate Sinkhorn"
        )
    aggregate = gt["aggregate_sinkhorn"]
    per_direction_raw = gt.get("per_direction_sinkhorn")
    if _allows_missing_gt_sinkhorn(
        disqualified=gt_disqualified,
        rank=gt_rank,
        aggregate=aggregate,
        per_direction=per_direction_raw,
    ):
        aggregate_out: float | None = None
        per_direction: dict[str, float] = {}
    else:
        if aggregate is None:
            raise ValueError(
                f"structure {structure_idx} GT candidate is missing aggregate Sinkhorn"
            )
        aggregate_out = _finite_json_number(
            aggregate, field=f"structure {structure_idx} GT aggregate Sinkhorn"
        )
        per_direction = _per_direction_sinkhorn(
            per_direction_raw, structure_idx=structure_idx
        )

    rank_one_candidates = [
        candidate for candidate in candidates if candidate.get("rank") == 1
    ]
    if len(rank_one_candidates) > 1:
        raise ValueError(
            f"structure {structure_idx} must contain exactly one candidate with rank 1"
        )

    winner = structure.get("winner")
    if not rank_one_candidates:
        if winner is not None:
            raise ValueError(
                f"structure {structure_idx} winner requires a candidate with rank 1"
            )
        winner_candidate_index = None
        winner_is_gt = None
    else:
        if not isinstance(winner, dict):
            raise ValueError(f"structure {structure_idx} must contain a ranked winner")
        try:
            winner_candidate_index = _json_integer(
                winner["candidate_index"],
                field=f"structure {structure_idx} winner candidate_index",
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"structure {structure_idx} winner has invalid candidate_index"
            ) from exc
        winner_candidates = [
            candidate
            for candidate in rank_one_candidates
            if candidate["candidate_index"] == winner_candidate_index
        ]
        if len(winner_candidates) != 1:
            raise ValueError(
                f"structure {structure_idx} winner must reference exactly one "
                "candidate with rank 1"
            )
        winner_is_gt = winner_candidates[0]["is_gt"]
        if winner_is_gt != (winner_candidate_index == gt_candidate_index):
            raise ValueError(
                f"structure {structure_idx} winner GT identity disagrees with GT index"
            )

    return {
        "structure_idx": structure_idx,
        "gt_candidate_index": gt_candidate_index,
        "gt_aggregate_sinkhorn": aggregate_out,
        "gt_per_direction_sinkhorn": per_direction,
        "gt_rank": gt_rank,
        "gt_disqualified": gt_disqualified,
        "gt_disqualification_reason": gt.get("disqualification_reason"),
        "winner_candidate_index": winner_candidate_index,
        "winner_is_gt": winner_is_gt,
    }


def write_seed_summary(
    *,
    seed: int,
    ranking_json: Path,
    out_summary: Path,
    expected_structures: int = 5,
    pass_threshold: int | None = None,
) -> dict[str, Any]:
    """Validate one ranking report, write its evidence, and apply the seed gate."""
    expected_structures = int(expected_structures)
    if expected_structures < 1:
        raise ValueError("expected_structures must be positive")
    threshold = (
        expected_structures // 2 + 1
        if pass_threshold is None
        else int(pass_threshold)
    )
    if threshold < 1 or threshold > expected_structures:
        raise ValueError("pass_threshold must be between 1 and expected_structures")

    report = _load_object(Path(ranking_json), description="ranking report")
    structures = _required_list(report, "structures", context="ranking report")
    skipped = _required_list(
        report, "skipped_structures", context="ranking report"
    )
    aggregate = report.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError("ranking report.aggregate must be an object")
    if skipped:
        raise ValueError(
            f"ranking report contains {len(skipped)} skipped structure(s)"
        )
    if len(structures) != expected_structures:
        raise ValueError(
            f"ranking report evaluated {len(structures)} structures; "
            f"expected {expected_structures}"
        )
    n_structures = _json_integer(
        aggregate.get("n_structures"),
        field="ranking report aggregate n_structures",
    )
    n_evaluated = _json_integer(
        aggregate.get("n_evaluated"),
        field="ranking report aggregate n_evaluated",
    )
    n_skipped = _json_integer(
        aggregate.get("n_skipped"),
        field="ranking report aggregate n_skipped",
    )
    if n_evaluated != len(structures):
        raise ValueError("ranking report aggregate n_evaluated is inconsistent")
    if n_skipped != 0:
        raise ValueError("ranking report aggregate reports skipped structures")
    if n_structures != expected_structures:
        raise ValueError("ranking report aggregate n_structures is inconsistent")

    evidence = [extract_structure_evidence(structure) for structure in structures]
    structure_indices = [item["structure_idx"] for item in evidence]
    if len(set(structure_indices)) != expected_structures:
        raise ValueError("ranking report must contain unique evaluated structures")

    n_rank_one = sum(item["gt_rank"] == 1 for item in evidence)
    summary = {
        "seed": int(seed),
        "ranking_json": str(ranking_json),
        "expected_structures": expected_structures,
        "pass_threshold": threshold,
        "n_gt_rank_1": n_rank_one,
        "passed": n_rank_one >= threshold,
        "structures": evidence,
    }
    _write_strict_json(Path(out_summary), summary)
    return summary


def _validate_seed_evidence(
    evidence: Any, *, seed: int, expected_structures: int
) -> int:
    if not isinstance(evidence, list) or len(evidence) != expected_structures:
        raise ValueError(f"invalid seed {seed} structure evidence")

    structure_indices: list[int] = []
    n_rank_one = 0
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError(f"invalid seed {seed} structure evidence")
        structure_idx = _json_integer(
            item.get("structure_idx"), field=f"seed {seed} structure_idx"
        )
        structure_indices.append(structure_idx)
        gt_candidate_index = _json_integer(
            item.get("gt_candidate_index"),
            field=f"seed {seed} structure {structure_idx} gt_candidate_index",
        )

        disqualified = item.get("gt_disqualified")
        if type(disqualified) is not bool:
            raise ValueError(
                f"invalid seed {seed} structure {structure_idx} "
                "gt_disqualified must be boolean"
            )
        reason = item.get("gt_disqualification_reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError(
                f"invalid seed {seed} structure {structure_idx} "
                "gt_disqualification_reason"
            )
        rank = item.get("gt_rank")
        if rank is None:
            if not disqualified:
                raise ValueError(
                    f"invalid seed {seed} structure {structure_idx} GT rank"
                )
        else:
            rank = _json_integer(
                rank, field=f"seed {seed} structure {structure_idx} GT rank"
            )
            if rank < 1:
                raise ValueError(
                    f"invalid seed {seed} structure {structure_idx} GT rank"
                )

        if "gt_aggregate_sinkhorn" not in item:
            raise ValueError(
                f"seed {seed} structure {structure_idx} "
                "GT aggregate Sinkhorn is missing"
            )
        aggregate = item["gt_aggregate_sinkhorn"]
        per_direction = item.get("gt_per_direction_sinkhorn")
        if not _allows_missing_gt_sinkhorn(
            disqualified=disqualified,
            rank=rank,
            aggregate=aggregate,
            per_direction=per_direction,
        ):
            _finite_json_number(
                aggregate,
                field=f"seed {seed} structure {structure_idx} GT aggregate Sinkhorn",
            )
            _per_direction_sinkhorn(per_direction, structure_idx=structure_idx)

        n_rank_one += rank == 1

        winner_index = item.get("winner_candidate_index")
        winner_is_gt = item.get("winner_is_gt")
        if winner_index is None:
            if winner_is_gt is not None:
                raise ValueError(
                    f"invalid seed {seed} structure {structure_idx} winner evidence"
                )
        else:
            winner_index = _json_integer(
                winner_index,
                field=(
                    f"seed {seed} structure {structure_idx} "
                    "winner_candidate_index"
                ),
            )
            if type(winner_is_gt) is not bool:
                raise ValueError(
                    f"invalid seed {seed} structure {structure_idx} winner evidence"
                )
            if winner_is_gt != (winner_index == gt_candidate_index):
                raise ValueError(
                    f"invalid seed {seed} structure {structure_idx} winner GT identity"
                )
        if rank == 1 and (
            winner_index != gt_candidate_index or winner_is_gt is not True
        ):
            raise ValueError(
                f"invalid seed {seed} structure {structure_idx} rank-one winner"
            )

    if len(set(structure_indices)) != expected_structures:
        raise ValueError(f"invalid seed {seed} duplicate structure evidence")
    return n_rank_one


def _validate_seed_summary(
    summary: dict[str, Any], *, seed: int, path: Path
) -> None:
    if type(summary.get("seed")) is not int or summary["seed"] != seed:
        raise ValueError(f"invalid seed {seed} summary identity at {path}")
    if type(summary.get("passed")) is not bool:
        raise ValueError(f"invalid seed {seed} summary passed status at {path}")
    structures = summary.get("structures")
    if not isinstance(structures, list):
        raise ValueError(f"invalid seed {seed} summary structure evidence at {path}")
    if not isinstance(summary.get("ranking_json"), str):
        raise ValueError(f"invalid seed {seed} ranking_json at {path}")

    if "error" in summary:
        error = summary["error"]
        if (
            summary["passed"] is not False
            or not isinstance(error, str)
            or not error
            or structures
        ):
            raise ValueError(f"invalid seed {seed} failure summary at {path}")
        return

    expected = _json_integer(
        summary.get("expected_structures"),
        field=f"invalid seed {seed} expected_structures",
    )
    threshold = _json_integer(
        summary.get("pass_threshold"),
        field=f"invalid seed {seed} pass_threshold",
    )
    reported_rank_one = _json_integer(
        summary.get("n_gt_rank_1"),
        field=f"invalid seed {seed} n_gt_rank_1",
    )
    if expected < 1 or threshold < 1 or threshold > expected:
        raise ValueError(f"invalid seed {seed} gate dimensions at {path}")
    n_rank_one = _validate_seed_evidence(
        structures, seed=seed, expected_structures=expected
    )
    expected_passed = n_rank_one >= threshold
    if reported_rank_one != n_rank_one or summary["passed"] != expected_passed:
        raise ValueError(f"invalid seed {seed} gate result at {path}")


def finalize_gate(
    *,
    report_dir: Path,
    seeds: list[int],
    out_summary: Path | None = None,
) -> dict[str, Any]:
    """Load every expected seed summary and pass only when all seeds passed."""
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seed values are not allowed")
    report_dir = Path(report_dir)
    seed_summaries: list[dict[str, Any]] = []
    for seed in seeds:
        path = report_dir / f"seed{int(seed)}_summary.json"
        try:
            summary = _load_object(path, description=f"seed {int(seed)} summary")
            _validate_seed_summary(summary, seed=int(seed), path=path)
        except ValueError as exc:
            summary = {
                "seed": int(seed),
                "summary_json": str(path),
                "passed": False,
                "error": str(exc),
                "structures": [],
            }
        seed_summaries.append(summary)

    n_passed = sum(summary["passed"] for summary in seed_summaries)
    final = {
        "seed_ids": [int(seed) for seed in seeds],
        "n_seeds": len(seeds),
        "n_passed_seeds": n_passed,
        "passed": n_passed == len(seeds),
        "seeds": seed_summaries,
    }
    destination = Path(out_summary) if out_summary is not None else report_dir / "summary.json"
    _write_strict_json(destination, final)
    return final


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("duplicate seed values are not allowed")
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ranking-json", type=Path)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--expected-structures", type=int, default=5)
    parser.add_argument("--pass-threshold", type=int)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--seeds", type=_parse_seeds, default=[0, 1, 2])
    args = parser.parse_args(argv)

    if args.finalize:
        if args.report_dir is None:
            parser.error("--finalize requires --report-dir")
        summary = finalize_gate(
            report_dir=args.report_dir,
            seeds=args.seeds,
            out_summary=args.out_summary,
        )
        print(
            f"seeds={summary['n_seeds']} passed={summary['passed']} "
            f"passed_seeds={summary['n_passed_seeds']}/{summary['n_seeds']}"
        )
        return 0 if summary["passed"] else 1

    if args.seed is None or args.ranking_json is None:
        parser.error("seed mode requires --seed and --ranking-json")
    try:
        summary = write_seed_summary(
            seed=args.seed,
            ranking_json=args.ranking_json,
            out_summary=args.out_summary,
            expected_structures=args.expected_structures,
            pass_threshold=args.pass_threshold,
        )
    except ValueError as exc:
        summary = {
            "seed": args.seed,
            "ranking_json": str(args.ranking_json),
            "passed": False,
            "error": str(exc),
            "structures": [],
        }
        _write_strict_json(args.out_summary, summary)
        print(f"seed={args.seed} passed=False error={exc}")
        return 1
    print(
        f"seed={summary['seed']} passed={summary['passed']} "
        f"rank1={summary['n_gt_rank_1']}/{summary['expected_structures']} "
        f"threshold={summary['pass_threshold']}"
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
