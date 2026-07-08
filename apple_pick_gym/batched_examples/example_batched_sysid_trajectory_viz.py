"""Visualize recorded batched sys-ID trajectories (positions over time + movement direction).

Run from repo root::

    uv run python apple_pick_gym/batched_examples/example_batched_sysid_trajectory_viz.py \\
        --dataset tmp/batched_sysid_dataset_20260707T204005Z \\
        --output tmp/batched_sysid_trajectory_viz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from apple_pick_sim.system_id.batched_hold_quasi_static import (
    analyze_dataset_hold_quasi_static,
    format_hold_summary_text,
)
from apple_pick_sim.system_id.batched_trajectory_viz import write_dataset_trajectory_viz


def _parse_int_list(text: str | None) -> list[int] | None:
    if text is None or not str(text).strip():
        return None
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Batched sys-ID dataset directory (contains manifest.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for Plotly HTML outputs.",
    )
    parser.add_argument(
        "--structures",
        type=str,
        default=None,
        help="Comma-separated structure indices to plot (default: all).",
    )
    parser.add_argument(
        "--directions",
        type=str,
        default=None,
        help="Comma-separated direction indices to plot (default: all).",
    )
    parser.add_argument(
        "--no-hold-check",
        action="store_true",
        help="Skip hold quasi-static diagnostics and plots.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    written = write_dataset_trajectory_viz(
        args.dataset,
        args.output,
        structure_indices=_parse_int_list(args.structures),
        direction_indices=_parse_int_list(args.directions),
        check_hold=not args.no_hold_check,
    )
    print(f"Wrote {len(written)} file(s) under {Path(args.output).resolve()}")
    for path in written:
        print(f"  {path}")
    if not args.no_hold_check:
        summary = analyze_dataset_hold_quasi_static(
            args.dataset,
            structure_indices=_parse_int_list(args.structures),
            direction_indices=_parse_int_list(args.directions),
        )
        print()
        print(format_hold_summary_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
