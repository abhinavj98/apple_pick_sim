from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from apple_pick_sim.system_id.real_action_fill import fill_actions_from_tcp_velocity


def _write_mini(path: Path) -> None:
    rows = [
        {
            "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "tcp_velocity": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        {
            "action": [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            "tcp_velocity": [9.0, 9.0, 0.0, 0.0, 0.0, 0.0],
        },
    ]
    table = pa.Table.from_pylist(rows)
    dm = {"episode_id": "fill-test"}
    meta = {b"dataset_metadata": json.dumps(dm).encode("utf-8")}
    pq.write_table(table.replace_schema_metadata(meta), path)


def test_fill_actions_from_tcp_velocity_only_zeros(tmp_path: Path):
    src = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    _write_mini(src)
    stats = fill_actions_from_tcp_velocity(src, out, only_when_action_zero=True)
    assert stats["rows_filled"] == 1
    assert stats["action_nonzero_after"] == 2

    table = pq.read_table(out)
    a0 = np.asarray(table.column("action")[0].as_py(), dtype=np.float64)
    a1 = np.asarray(table.column("action")[1].as_py(), dtype=np.float64)
    np.testing.assert_allclose(a0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(a1, [0.0, 2.0, 0.0, 0.0, 0.0, 0.0])

    dm = json.loads(table.schema.metadata[b"dataset_metadata"])
    assert dm["drive_fill"]["rows_filled"] == 1
    assert "real-replay-action-zero" in dm["drive_fill"]["note"]
