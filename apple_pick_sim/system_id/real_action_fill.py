"""Temporary fill of zero ``action`` from ``tcp_velocity`` (real-replay-action-zero)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def fill_actions_from_tcp_velocity(
    input_path: Path | str,
    output_path: Path | str,
    *,
    only_when_action_zero: bool = True,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Copy parquet; set ``action`` from ``tcp_velocity`` where needed.

    Stamps ``dataset_metadata.drive_fill``. Temporary mitigation only.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
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
    mask = action_norms <= float(atol) if only_when_action_zero else np.ones(n, dtype=bool)
    filled = int(mask.sum())
    out_actions = actions.copy()
    out_actions[mask] = vels[mask]

    action_array = pa.array(out_actions.tolist(), type=pa.list_(pa.float64(), 6))
    idx = table.schema.get_field_index("action")
    arrays = [table.column(i) if i != idx else action_array for i in range(table.num_columns)]
    new_table = pa.Table.from_arrays(arrays, schema=table.schema)

    schema = table.schema
    meta = dict(schema.metadata or {})
    blob = meta.get(b"dataset_metadata")
    if blob is not None:
        dm = json.loads(
            blob.decode("utf-8") if isinstance(blob, (bytes, bytearray)) else str(blob)
        )
        if not isinstance(dm, dict):
            raise ValueError("dataset_metadata must be a JSON object")
        dm["drive_fill"] = {
            "method": (
                "tcp_velocity_where_action_zero"
                if only_when_action_zero
                else "tcp_velocity_overwrite"
            ),
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
        "action_nonzero_after": int((np.linalg.norm(out_actions, axis=1) > atol).sum()),
        "tcp_velocity_nonzero": int((vel_norms > atol).sum()),
    }
