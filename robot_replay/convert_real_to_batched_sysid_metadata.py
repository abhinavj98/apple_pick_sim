#!/usr/bin/env python3
"""CLI: real-world parquet → batched-style episode metadata JSON (no frame remap).

Pre-grasp woody (non-bending) rebuilds fruiting_system geometry; post-grasp
settled apple + TCP supplies the weld attachment. See robot_replay/README.md
and docs/real-sysid-pre-post-grasp-fixes.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build batched_sysid_v1-compatible episode metadata from a real-world "
            "sys-ID parquet using the same pre/post builders as "
            "example_view_pre_grasp_settle.py (rebuild + grasp init only; no "
            "trajectory export)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Real-world episode parquet path",
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
        help="Output JSON path (default: stdout)",
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
    )

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
