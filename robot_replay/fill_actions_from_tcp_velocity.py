#!/usr/bin/env python3
"""Fill zero ``action`` columns from ``tcp_velocity`` for local bit-2 testing.

Temporary mitigation for ``real-replay-action-zero`` while real collection is
fixed. Does **not** change the long-term contract: future real parquets should
ship non-zero ``action`` (full trajectory).

Example::

    uv run python robot_replay/fill_actions_from_tcp_velocity.py \\
      --input robot_replay/s00-d03.parquet \\
      --out robot_replay/s00-d03_with_actions.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local real-episode parquet with action filled from "
            "tcp_velocity (temporary real-replay-action-zero mitigation)."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: <input_stem>_with_actions.parquet beside input)",
    )
    parser.add_argument(
        "--overwrite-all",
        action="store_true",
        help="Overwrite every action row from tcp_velocity (default: only zero actions).",
    )
    args = parser.parse_args(argv)
    out = args.out
    if out is None:
        out = args.input.with_name(f"{args.input.stem}_with_actions.parquet")

    from apple_pick_sim.system_id.real_action_fill import fill_actions_from_tcp_velocity

    stats = fill_actions_from_tcp_velocity(
        args.input,
        out,
        only_when_action_zero=not bool(args.overwrite_all),
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
