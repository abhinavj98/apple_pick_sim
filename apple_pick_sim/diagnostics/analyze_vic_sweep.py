"""Analyze and rank merged zero-VIC stability sweep results."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


def merge_summary_csvs(dirs: list[Path]) -> list[dict[str, str]]:
    """Concatenate worker summary.csv files and deduplicate by config_id."""
    by_id: dict[int, dict[str, str]] = {}
    for d in dirs:
        path = d / "summary.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                by_id[int(row["config_id"])] = row
    return [by_id[k] for k in sorted(by_id)]


def rank_configs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Sort configs by pass rate desc, then drift asc."""
    return sorted(
        rows,
        key=lambda r: (
            -float(r.get("vic_pass_rate") or 0.0),
            float(r.get("max_apple_drift_m") or float("inf")),
            float(r.get("max_apple_path_m") or float("inf")),
            int(r.get("config_id") or 0),
        ),
    )


def issue_breakdown(rows: list[dict[str, str]]) -> dict[str, int]:
    """Placeholder aggregate from summary scalars (no per-env issue tags in CSV)."""
    counts: Counter[str] = Counter()
    for row in rows:
        pass_rate = float(row.get("vic_pass_rate") or 0.0)
        if pass_rate < 1.0 - 1e-9:
            drift = float(row.get("max_apple_drift_m") or 0.0)
            if drift > 0.02:
                counts["apple_drift"] += 1
            z_drop = float(row.get("max_apple_z_drop_m") or 0.0)
            if z_drop > 0.015:
                counts["apple_sag"] += 1
            path_m = float(row.get("max_apple_path_m") or 0.0)
            if path_m > 0.05:
                counts["apple_wander"] += 1
            pos_err = float(row.get("max_pos_err") or 0.0)
            if pos_err > 0.05:
                counts["pos_err"] += 1
            tcp = float(row.get("max_tcp_speed") or 0.0)
            if tcp > 0.05:
                counts["tcp_speed"] += 1
            harvest = float(row.get("max_harvest_n") or 0.0)
            if harvest > 200.0:
                counts["harvest_force"] += 1
    return dict(counts)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and rank VIC sweep summaries.")
    parser.add_argument(
        "--input-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Worker output directories containing summary.csv.",
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write ranked CSV (default: stdout table only).",
    )
    parser.add_argument(
        "--merged-output",
        type=str,
        default=None,
        help="Write merged (deduplicated) CSV path.",
    )
    return parser.parse_args(argv)


def _print_ranked(rows: list[dict[str, str]], top_n: int) -> None:
    ranked = rank_configs(rows)[:top_n]
    print(f"top {len(ranked)} / {len(rows)} configs:", flush=True)
    for i, row in enumerate(ranked):
        print(
            f"  {i + 1:3d}  id={row.get('config_id')}  "
            f"pass={100.0 * float(row.get('vic_pass_rate') or 0.0):5.1f}%  "
            f"drift={float(row.get('max_apple_drift_m') or 0.0):.4f} m  "
            f"tier={row.get('spur_elev_tier')}  "
            f"stem={row.get('stem_gain')}  "
            f"k={row.get('vic_linear_k')}  d={row.get('vic_linear_d')}  "
            f"ak={row.get('vic_angular_k')}  ad={row.get('vic_angular_d')}  "
            f"sp={row.get('spur_num_segs') or 'def'}  "
            f"st={row.get('stem_num_segs') or 'def'}",
            flush=True,
        )
    issues = issue_breakdown(rows)
    if issues:
        print("failure proxy counts:", flush=True)
        for key, count in sorted(issues.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {key}: {count}", flush=True)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    dirs = [Path(p) for p in args.input_dirs]
    merged = merge_summary_csvs(dirs)
    ranked = rank_configs(merged)
    _print_ranked(merged, int(args.top_n))

    if args.merged_output:
        out = Path(args.merged_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(out, merged)
        print(f"wrote merged {out} ({len(merged)} rows)", flush=True)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(out, ranked[: int(args.top_n)])
        print(f"wrote ranked {out}", flush=True)

    return 0


def main(argv: list[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
