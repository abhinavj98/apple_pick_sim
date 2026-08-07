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

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def fill_actions_from_tcp_velocity(
    input_path: Path,
    output_path: Path,
    *,
    only_when_action_zero: bool = True,
    atol: float = 1e-12,
) -> dict:
    """Copy parquet; set ``action`` from ``tcp_velocity`` where needed."""
    table = pq.read_table(input_path)
    if "action" not in table.column_names:
        raise ValueError(f"{input_path}: missing action column")
    if "tcp_velocity" not in table.column_names:
        raise ValueError(f"{input_path}: missing tcp_velocity column")

    n = table.num_rows
    actions = np.stack([table.column("action")[i].as_py() for i in range(n)], axis=0)
    vels = np.stack([table.column("tcp_velocity")[i].as_py() for i in range(n)], axis=0)
    actions = np.asarray(actions, dtype=np.float64).reshape(n, 6)
    vels = np.asarray(vels, dtype=np.float64).reshape(n, 6)

    action_norms = np.linalg.norm(actions, axis=1)
    vel_norms = np.linalg.norm(vels, axis=1)
    if only_when_action_zero:
        mask = action_norms <= float(atol)
    else:
        mask = np.ones(n, dtype=bool)
    filled = int(mask.sum())
    out_actions = actions.copy()
    out_actions[mask] = vels[mask]

    # Rebuild action column as fixed-size list[6]
    action_array = pa.array(out_actions.tolist(), type=pa.list_(pa.float64(), 6))
    idx = table.schema.get_field_index("action")
    arrays = [table.column(i) if i != idx else action_array for i in range(table.num_columns)]
    new_table = pa.Table.from_arrays(arrays, schema=table.schema)

    # Annotate dataset_metadata
    schema = table.schema
    meta = dict(schema.metadata or {})
    blob = meta.get(b"dataset_metadata")
    if blob is not None:
        dm = json.loads(blob.decode("utf-8") if isinstance(blob, (bytes, bytearray)) else str(blob))
        if not isinstance(dm, dict):
            raise ValueError("dataset_metadata must be a JSON object")
        dm["drive_fill"] = {
            "method": "tcp_velocity_where_action_zero"
            if only_when_action_zero
            else "tcp_velocity_overwrite",
            "source_parquet": str(input_path),
            "rows_filled": filled,
            "rows_total": n,
            "note": (
                "Temporary local fixture for bit-2 replay testing "
                "(real-replay-action-zero mitigation). Not the long-term contract."
            ),
        }
        meta[b"dataset_metadata"] = json.dumps(dm).encode("utf-8")
        new_table = new_table.replace_schema_metadata(meta)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, output_path)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows_total": n,
        "rows_filled": filled,
        "action_nonzero_before": int((action_norms > atol).sum()),
        "action_nonzero_after": int(
            (np.linalg.norm(out_actions, axis=1) > atol).sum()
        ),
        "tcp_velocity_nonzero": int((vel_norms > atol).sum()),
    }


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
    stats = fill_actions_from_tcp_velocity(
        args.input,
        out,
        only_when_action_zero=not bool(args.overwrite_all),
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
