#!/usr/bin/env python3
"""CLI: real-world parquet → batched_sysid_v1 metadata and/or full dataset.

Bit 1: ``--out`` writes episode metadata JSON (rebuild + grasp init).
Bit 2: ``--dataset-out`` writes a 1×1 batched_sysid_v1 dataset directory
(manifest + episodes/s00_d00.parquet) for trajectory viz / FR3 replay.

For pose-control wrench logs, Bit 2 packs 19D ``vic_pose`` actions from
``target_pose_4x4`` + ``dump.controller_gains`` (not the raw wrench).

Preferred real source: ``robot_replay/s02-d00.parquet``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a real-world sys-ID parquet using settle-viewer pre/post "
            "builders. Emit metadata JSON and/or a full batched_sysid_v1 dataset."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Real-world episode parquet path (prefer s02-d00.parquet)",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(
            "apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json"
        ),
        help="Variance fixture JSON for materials / num_segments midpoints",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Episode metadata JSON path (default: stdout if no --dataset-out)",
    )
    parser.add_argument(
        "--dataset-out",
        type=Path,
        default=None,
        help="Write 1×1 batched_sysid_v1 dataset directory (manifest + episodes/).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting a non-empty --dataset-out directory.",
    )
    parser.add_argument(
        "--allow-zero-action",
        action="store_true",
        help="Permit exporting when action is all zeros (not recommended).",
    )
    parser.add_argument(
        "--weld-direction-sign",
        type=float,
        default=1.0,
        help="Multiply weld_direction by this sign (default: 1.0)",
    )
    args = parser.parse_args(argv)

    from apple_pick_sim.system_id.real_to_batched_sysid import (
        build_episode_metadata_from_real,
        export_real_episode_to_batched_dataset,
    )

    if args.dataset_out is not None:
        out_dir = export_real_episode_to_batched_dataset(
            args.input,
            fixture_path=args.fixture,
            output_dir=args.dataset_out,
            weld_direction_sign=args.weld_direction_sign,
            overwrite=bool(args.overwrite),
            allow_zero_action=bool(args.allow_zero_action),
            command_argv=list(sys.argv if argv is None else ["convert", *argv]),
        )
        print(f"Wrote batched dataset {out_dir}", file=sys.stderr)

    if args.out is not None or args.dataset_out is None:
        meta = build_episode_metadata_from_real(
            args.input,
            fixture_path=args.fixture,
            weld_direction_sign=args.weld_direction_sign,
        )
        text = json.dumps(meta, indent=2, sort_keys=True) + "\n"
        if args.out is None:
            sys.stdout.write(text)
        else:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
            print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
