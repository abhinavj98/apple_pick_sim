"""Validate CMA-ES Young's-modulus fit integrity and report multi-seed gate results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


_STOP_KINDS = frozenset({"generation_cap", "pycma", "both"})
_COMPONENT_COUNT = 3


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


def _structures_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize CLI dict-keyed or list structures into an ordered list."""
    raw = report.get("structures")
    if isinstance(raw, dict):
        structures: list[dict[str, Any]] = []
        for key, structure in raw.items():
            if not isinstance(structure, dict):
                raise ValueError("cmaes report structure must be an object")
            try:
                key_idx = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"cmaes report structures key {key!r} must be integer-like"
                ) from exc
            if isinstance(key, str) and str(key_idx) != key.lstrip("+"):
                if not (key.isdigit() and int(key) == key_idx):
                    raise ValueError(
                        f"cmaes report structures key {key!r} must be integer-like"
                    )
            structure_idx = structure.get("structure_idx")
            if type(structure_idx) is int and structure_idx != key_idx:
                raise ValueError(
                    f"cmaes report structures key {key!r} disagrees with "
                    f"structure_idx {structure_idx}"
                )
            structures.append(structure)
        structures.sort(
            key=lambda item: int(item.get("structure_idx", -1))
            if type(item.get("structure_idx")) is int
            else -1
        )
        return structures
    if isinstance(raw, list):
        return raw
    raise ValueError("cmaes report.structures must be an object or list")


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


def _finite_vector3(value: Any, *, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != _COMPONENT_COUNT:
        raise ValueError(f"{field} must be a length-{_COMPONENT_COUNT} list")
    return [_finite_json_number(item, field=field) for item in value]


def _per_direction_sinkhorn(value: Any, *, structure_idx: int) -> dict[str, float]:
    field = f"structure {structure_idx} final-mean per-direction Sinkhorn"
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
                direction.startswith("-") and str(direction_id) == direction
            ) and not (direction.isdigit() and int(direction) == direction_id):
                raise ValueError(f"{field} direction IDs must be integer-like")
        key = str(direction_id)
        if key in normalized:
            raise ValueError(f"{field} contains duplicate direction ID {key}")
        normalized[key] = _finite_json_number(score, field=field)
    return normalized


def _validate_matrix3(value: Any, *, field: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != _COMPONENT_COUNT:
        raise ValueError(f"{field} must be a {_COMPONENT_COUNT}x{_COMPONENT_COUNT} matrix")
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != _COMPONENT_COUNT:
            raise ValueError(
                f"{field} must be a {_COMPONENT_COUNT}x{_COMPONENT_COUNT} matrix"
            )
        rows.append([_finite_json_number(item, field=field) for item in row])
    return rows


def _validate_covariance(value: Any, *, structure_idx: int) -> None:
    field = f"structure {structure_idx} covariance"
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object when present")
    required = (
        "C",
        "sigma",
        "sigma_vec_scaling",
        "phenotype_std",
        "effective_unbounded_covariance",
    )
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"{field} is missing fields: {', '.join(missing)}")
    _validate_matrix3(value["C"], field=f"{field}.C")
    sigma = _finite_json_number(value["sigma"], field=f"{field}.sigma")
    if sigma <= 0.0:
        raise ValueError(f"{field}.sigma must be positive")
    _finite_vector3(
        value["sigma_vec_scaling"], field=f"{field}.sigma_vec_scaling"
    )
    _finite_vector3(value["phenotype_std"], field=f"{field}.phenotype_std")
    _validate_matrix3(
        value["effective_unbounded_covariance"],
        field=f"{field}.effective_unbounded_covariance",
    )


def _validate_evaluated_history_extrema(value: Any, *, structure_idx: int) -> bool:
    field = f"structure {structure_idx} evaluated history extrema"
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    keys = ("min_log10_e", "max_log10_e", "min_e_pa", "max_e_pa")
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{field} is missing fields: {', '.join(missing)}")

    if all(value[key] is None for key in keys):
        return False
    if any(value[key] is None for key in keys):
        raise ValueError(f"{field} must be all null or all finite vectors")

    min_log10 = _finite_vector3(value["min_log10_e"], field=f"{field}.min_log10_e")
    max_log10 = _finite_vector3(value["max_log10_e"], field=f"{field}.max_log10_e")
    min_e = _finite_vector3(value["min_e_pa"], field=f"{field}.min_e_pa")
    max_e = _finite_vector3(value["max_e_pa"], field=f"{field}.max_e_pa")
    for lo, hi in zip(min_log10, max_log10):
        if lo > hi:
            raise ValueError(f"{field} min_log10_e must be <= max_log10_e")
    for lo, hi in zip(min_e, max_e):
        if lo > hi:
            raise ValueError(f"{field} min_e_pa must be <= max_e_pa")
    for log_v, e_v, label in (
        (min_log10, min_e, "min"),
        (max_log10, max_e, "max"),
    ):
        for log_component, e_component in zip(log_v, e_v):
            expected = 10.0**log_component
            if not math.isclose(e_component, expected, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    f"{field} {label}_e_pa must match 10**{label}_log10_e"
                )
    return True


def _within_bounds(
    log10_e: list[float],
    e_pa: list[float],
    bounds: dict[str, Any],
    *,
    structure_idx: int,
) -> None:
    lower = _finite_vector3(
        bounds.get("log10_lower"),
        field=f"structure {structure_idx} bounds.log10_lower",
    )
    upper = _finite_vector3(
        bounds.get("log10_upper"),
        field=f"structure {structure_idx} bounds.log10_upper",
    )
    for value, lo, hi in zip(log10_e, lower, upper):
        if value < lo or value > hi:
            raise ValueError(
                f"structure {structure_idx} final mean log10-E is outside bounds"
            )
    physical_min = _finite_vector3(
        bounds.get("physical_min_pa"),
        field=f"structure {structure_idx} bounds.physical_min_pa",
    )
    physical_max = _finite_vector3(
        bounds.get("physical_max_pa"),
        field=f"structure {structure_idx} bounds.physical_max_pa",
    )
    for value, lo, hi in zip(e_pa, physical_min, physical_max):
        if value < lo or value > hi:
            raise ValueError(
                f"structure {structure_idx} final mean E is outside bounds"
            )
    for log_component, e_component in zip(log10_e, e_pa):
        expected = 10.0**log_component
        if not math.isclose(e_component, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"structure {structure_idx} final mean e_pa must match 10**log10_e"
            )


def extract_structure_evidence(structure: dict[str, Any]) -> dict[str, Any]:
    """Extract and validate integrity evidence from one CMA structure report."""
    if not isinstance(structure, dict):
        raise ValueError("cmaes report structure must be an object")
    try:
        structure_idx = _json_integer(
            structure["structure_idx"], field="structure_idx"
        )
    except (KeyError, ValueError) as exc:
        raise ValueError("cmaes report structure has invalid structure_idx") from exc

    status = structure.get("status")
    if status != "fitted":
        raise ValueError(f"structure {structure_idx} status must be fitted")

    bounds = structure.get("bounds")
    if not isinstance(bounds, dict):
        raise ValueError(f"structure {structure_idx} bounds must be an object")

    final_mean = structure.get("final_mean")
    if not isinstance(final_mean, dict):
        raise ValueError(f"structure {structure_idx} final_mean must be an object")
    log10_e = _finite_vector3(
        final_mean.get("log10_e"),
        field=f"structure {structure_idx} final_mean.log10_e",
    )
    e_pa = _finite_vector3(
        final_mean.get("e_pa"),
        field=f"structure {structure_idx} final_mean.e_pa",
    )
    _within_bounds(log10_e, e_pa, bounds, structure_idx=structure_idx)

    if "aggregate_sinkhorn" not in final_mean:
        raise ValueError(
            f"structure {structure_idx} final mean is missing aggregate Sinkhorn"
        )
    aggregate = _finite_json_number(
        final_mean["aggregate_sinkhorn"],
        field=f"structure {structure_idx} final mean aggregate Sinkhorn",
    )
    per_direction = _per_direction_sinkhorn(
        final_mean.get("per_direction_sinkhorn"),
        structure_idx=structure_idx,
    )

    population_size = _json_integer(
        structure.get("population_size"),
        field=f"structure {structure_idx} population_size",
    )
    if population_size < 1:
        raise ValueError(f"structure {structure_idx} population_size must be positive")
    completed_generations = _json_integer(
        structure.get("completed_generations"),
        field=f"structure {structure_idx} completed_generations",
    )
    if completed_generations < 0:
        raise ValueError(
            f"structure {structure_idx} completed_generations must be non-negative"
        )
    optimizer_samples_told = _json_integer(
        structure.get("optimizer_samples_told"),
        field=f"structure {structure_idx} optimizer_samples_told",
    )
    expected_samples = completed_generations * population_size
    if optimizer_samples_told != expected_samples:
        raise ValueError(
            f"structure {structure_idx} optimizer_samples_told is not coherent "
            f"with completed_generations * population_size"
        )
    final_mean_evaluations = _json_integer(
        structure.get("final_mean_evaluations"),
        field=f"structure {structure_idx} final_mean_evaluations",
    )
    if final_mean_evaluations != 1:
        raise ValueError(
            f"structure {structure_idx} final_mean_evaluations must be 1 for fitted"
        )
    replay_candidate_evaluations = _json_integer(
        structure.get("replay_candidate_evaluations"),
        field=f"structure {structure_idx} replay_candidate_evaluations",
    )
    if replay_candidate_evaluations < optimizer_samples_told + final_mean_evaluations:
        raise ValueError(
            f"structure {structure_idx} replay_candidate_evaluations is not coherent"
        )

    stop_kind = structure.get("stop_kind")
    if stop_kind not in _STOP_KINDS:
        raise ValueError(f"structure {structure_idx} stop_kind is incomplete")
    stop_conditions = structure.get("stop_conditions")
    if not isinstance(stop_conditions, dict):
        raise ValueError(f"structure {structure_idx} stop_conditions must be an object")
    if stop_kind in {"pycma", "both"} and not stop_conditions:
        raise ValueError(
            f"structure {structure_idx} stop evidence requires non-empty "
            "stop_conditions for pycma stops"
        )

    covariance = structure.get("covariance")
    covariance_present = covariance is not None
    if covariance_present:
        _validate_covariance(covariance, structure_idx=structure_idx)

    extrema = structure.get("evaluated_history_extrema")
    if extrema is None:
        raise ValueError(
            f"structure {structure_idx} evaluated_history_extrema must be an object"
        )
    extrema_present = _validate_evaluated_history_extrema(
        extrema, structure_idx=structure_idx
    )

    return {
        "structure_idx": structure_idx,
        "status": "fitted",
        "final_mean_log10_e": log10_e,
        "final_mean_e_pa": e_pa,
        "final_mean_aggregate_sinkhorn": aggregate,
        "final_mean_per_direction_sinkhorn": per_direction,
        "completed_generations": completed_generations,
        "optimizer_samples_told": optimizer_samples_told,
        "final_mean_evaluations": final_mean_evaluations,
        "stop_kind": stop_kind,
        "stop_conditions": dict(stop_conditions),
        "covariance_present": covariance_present,
        "evaluated_history_extrema_present": extrema_present,
    }


def write_seed_summary(
    *,
    seed: int,
    cmaes_json: Path,
    out_summary: Path,
    expected_structures: int = 5,
) -> dict[str, Any]:
    """Validate one CMA report, write integrity evidence, and apply the seed gate."""
    expected_structures = int(expected_structures)
    if expected_structures < 1:
        raise ValueError("expected_structures must be positive")

    report = _load_object(Path(cmaes_json), description="cmaes report")
    structures = _structures_from_report(report)
    aggregate = report.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError("cmaes report.aggregate must be an object")
    if len(structures) != expected_structures:
        raise ValueError(
            f"cmaes report evaluated {len(structures)} structures; "
            f"expected {expected_structures}"
        )

    requested = _json_integer(
        aggregate.get("requested_structures"),
        field="cmaes report aggregate requested_structures",
    )
    fitted = _json_integer(
        aggregate.get("fitted_structures"),
        field="cmaes report aggregate fitted_structures",
    )
    failed = _json_integer(
        aggregate.get("failed_structures"),
        field="cmaes report aggregate failed_structures",
    )
    if requested != expected_structures:
        raise ValueError("cmaes report aggregate requested_structures is inconsistent")
    if fitted + failed != expected_structures:
        raise ValueError(
            "cmaes report aggregate fitted/failed counts are inconsistent with "
            f"expected {expected_structures}"
        )

    evidence: list[dict[str, Any]] = []
    fitted_count = 0
    structure_indices: list[int] = []
    for structure in structures:
        if not isinstance(structure, dict):
            raise ValueError("cmaes report structure must be an object")
        try:
            structure_idx = _json_integer(
                structure["structure_idx"], field="structure_idx"
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "cmaes report structure has invalid structure_idx"
            ) from exc
        structure_indices.append(structure_idx)
        status = structure.get("status")
        if status == "fitted":
            fitted_count += 1
            evidence.append(extract_structure_evidence(structure))
        else:
            evidence.append(
                {
                    "structure_idx": structure_idx,
                    "status": status,
                    "fitted": False,
                }
            )

    if len(set(structure_indices)) != expected_structures:
        raise ValueError("cmaes report must contain unique evaluated structures")
    if fitted_count != fitted:
        raise ValueError(
            "cmaes report aggregate fitted_structures disagrees with structure statuses"
        )
    if failed != expected_structures - fitted_count:
        raise ValueError(
            "cmaes report aggregate failed_structures disagrees with structure statuses"
        )

    summary = {
        "seed": int(seed),
        "cmaes_json": str(cmaes_json),
        "expected_structures": expected_structures,
        "n_fitted": fitted_count,
        "passed": fitted_count == expected_structures,
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
    n_fitted = 0
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError(f"invalid seed {seed} structure evidence")
        structure_idx = _json_integer(
            item.get("structure_idx"), field=f"seed {seed} structure_idx"
        )
        structure_indices.append(structure_idx)
        status = item.get("status")
        if status == "fitted":
            _finite_vector3(
                item.get("final_mean_log10_e"),
                field=f"seed {seed} structure {structure_idx} final_mean_log10_e",
            )
            _finite_json_number(
                item.get("final_mean_aggregate_sinkhorn"),
                field=(
                    f"seed {seed} structure {structure_idx} "
                    "final_mean_aggregate_sinkhorn"
                ),
            )
            _per_direction_sinkhorn(
                item.get("final_mean_per_direction_sinkhorn"),
                structure_idx=structure_idx,
            )
            stop_kind = item.get("stop_kind")
            if stop_kind not in _STOP_KINDS:
                raise ValueError(
                    f"invalid seed {seed} structure {structure_idx} stop_kind"
                )
            stop_conditions = item.get("stop_conditions")
            if not isinstance(stop_conditions, dict):
                raise ValueError(
                    f"invalid seed {seed} structure {structure_idx} stop_conditions"
                )
            if stop_kind in {"pycma", "both"} and not stop_conditions:
                raise ValueError(
                    f"invalid seed {seed} structure {structure_idx} stop evidence"
                )
            if type(item.get("covariance_present")) is not bool:
                raise ValueError(
                    f"invalid seed {seed} structure {structure_idx} covariance_present"
                )
            if type(item.get("evaluated_history_extrema_present")) is not bool:
                raise ValueError(
                    f"invalid seed {seed} structure {structure_idx} "
                    "evaluated_history_extrema_present"
                )
            n_fitted += 1
        elif status in {"failed", "active", "stopped_pending_final_evaluation"}:
            if item.get("fitted") is not False:
                raise ValueError(
                    f"invalid seed {seed} structure {structure_idx} non-fitted evidence"
                )
        else:
            raise ValueError(f"invalid seed {seed} structure {structure_idx} status")

    if len(set(structure_indices)) != expected_structures:
        raise ValueError(f"invalid seed {seed} duplicate structure evidence")
    return n_fitted


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
    if not isinstance(summary.get("cmaes_json"), str):
        raise ValueError(f"invalid seed {seed} cmaes_json at {path}")

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
    reported_fitted = _json_integer(
        summary.get("n_fitted"),
        field=f"invalid seed {seed} n_fitted",
    )
    if expected < 1:
        raise ValueError(f"invalid seed {seed} gate dimensions at {path}")
    n_fitted = _validate_seed_evidence(
        structures, seed=seed, expected_structures=expected
    )
    expected_passed = n_fitted == expected
    if reported_fitted != n_fitted or summary["passed"] != expected_passed:
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
    destination = (
        Path(out_summary) if out_summary is not None else report_dir / "summary.json"
    )
    _write_strict_json(destination, final)
    return final


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("duplicate seed values are not allowed")
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cmaes-json", type=Path)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--expected-structures", type=int, default=5)
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

    if args.seed is None or args.cmaes_json is None:
        parser.error("seed mode requires --seed and --cmaes-json")
    try:
        summary = write_seed_summary(
            seed=args.seed,
            cmaes_json=args.cmaes_json,
            out_summary=args.out_summary,
            expected_structures=args.expected_structures,
        )
    except ValueError as exc:
        summary = {
            "seed": args.seed,
            "cmaes_json": str(args.cmaes_json),
            "passed": False,
            "error": str(exc),
            "structures": [],
        }
        _write_strict_json(args.out_summary, summary)
        print(f"seed={args.seed} passed=False error={exc}")
        return 1
    print(
        f"seed={summary['seed']} passed={summary['passed']} "
        f"fitted={summary['n_fitted']}/{summary['expected_structures']}"
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
