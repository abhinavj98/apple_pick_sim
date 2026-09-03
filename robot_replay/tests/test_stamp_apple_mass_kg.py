from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robot_replay.stamp_apple_mass_kg import stamp_apple_mass_kg


def _write_minimal_parquet(path: Path, apple: dict) -> None:
    dm = {
        "pre_grasp_geometry": {
            "parts": {
                "apple": apple,
            }
        }
    }
    meta = {b"dataset_metadata": json.dumps(dm).encode("utf-8")}
    table = pa.table({"x": [1]}).replace_schema_metadata(meta)
    pq.write_table(table, path)


def test_stamp_apple_mass_kg_computes_from_radius_density(tmp_path: Path):
    path = tmp_path / "s03-d00.parquet"
    _write_minimal_parquet(
        path,
        {"radius_m": 0.04, "density_kg_m3": 820.6, "apple_number": 2},
    )
    result = stamp_apple_mass_kg(path)
    assert result["status"] == "stamped"
    assert result["mass_g"] == pytest.approx(220.0, rel=0.01)

    table = pq.read_table(path)
    dm = json.loads(table.schema.metadata[b"dataset_metadata"])
    apple = dm["pre_grasp_geometry"]["parts"]["apple"]
    assert apple["mass_kg_source"] == "computed_from_radius_density"
    expected = (4.0 / 3.0) * math.pi * 0.04**3 * 820.6
    assert apple["mass_kg"] == pytest.approx(expected, rel=1e-6)
    assert dm["apple_mass_kg"] == apple["mass_kg"]


def test_stamp_apple_mass_kg_skips_when_present(tmp_path: Path):
    path = tmp_path / "s09-d00.parquet"
    _write_minimal_parquet(
        path,
        {"radius_m": 0.04, "density_kg_m3": 1081.8, "mass_kg": 0.29},
    )
    result = stamp_apple_mass_kg(path)
    assert result["status"] == "skipped"
