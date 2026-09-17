"""Summarize fitted holdout match metrics across CMA seeds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load_reports(paths: list[Path]) -> list[dict]:
    reports: list[dict] = []
    for path in paths:
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    return reports


def _metric_value(report: dict, *, direction: str, window: str, path: list[str]) -> float | None:
    node = report["directions"][direction][window]["fitted"]
    for key in path:
        node = node[key]
    value = node.get("mse")
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def summarize_reports(reports: list[dict]) -> str:
    """Return a plain-text table grouped by tree."""
    by_tree: dict[str, list[dict]] = {}
    for report in reports:
        by_tree.setdefault(str(report.get("tree", "unknown")), []).append(report)
    lines: list[str] = []
    for tree in sorted(by_tree):
        group = by_tree[tree]
        lines.append(f"tree {tree} (n={len(group)})")
        for direction in sorted(
            {d for report in group for d in report.get("directions", {})},
            key=int,
        ):
            full_vals = [
                v
                for report in group
                if (v := _metric_value(report, direction=direction, window="full", path=["force", "combined"]))
                is not None
            ]
            hold_vals = [
                v
                for report in group
                if (v := _metric_value(report, direction=direction, window="hold", path=["force", "combined"]))
                is not None
            ]
            if full_vals:
                mean = sum(full_vals) / len(full_vals)
                var = sum((x - mean) ** 2 for x in full_vals) / len(full_vals)
                lines.append(
                    f"  dir {direction} full force.combined mse: "
                    f"{mean:.6g} ± {math.sqrt(var):.6g}"
                )
            if hold_vals:
                mean = sum(hold_vals) / len(hold_vals)
                var = sum((x - mean) ** 2 for x in hold_vals) / len(hold_vals)
                lines.append(
                    f"  dir {direction} hold force.combined mse: "
                    f"{mean:.6g} ± {math.sqrt(var):.6g}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="Paths to match_metrics.json files (e.g. tmp/cma_s04_val03_seed56/match_metrics.json)",
    )
    args = parser.parse_args(argv)
    text = summarize_reports(_load_reports(list(args.reports)))
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
