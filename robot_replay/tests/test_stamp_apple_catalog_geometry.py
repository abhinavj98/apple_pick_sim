from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robot_replay.stamp_apple_catalog_geometry import stamp_apple_catalog_geometry


def _write_minimal_parquet(path: Path, apple_number: int, mass_kg: float) -> None:
    dm = {
        "pre_grasp_geometry": {
            "parts": {
                "apple": {
                    "apple_number": apple_number,
                    "radius_m": 0.04,
                    "density_kg_m3": 820.6,
                    "mass_kg": mass_kg,
                },
                "stem": {"length_m": 0.015, "radius_m": 0.0005},
            }
        }
    }
    meta = {b"dataset_metadata": json.dumps(dm).encode("utf-8")}
    pq.write_table(pa.table({"x": [1]}).replace_schema_metadata(meta), path)


def test_stamp_apple2_rescales_density_keeps_mass(tmp_path: Path):
    path = tmp_path / "s03-d00.parquet"
    mass_kg = 0.219988559
    _write_minimal_parquet(path, apple_number=2, mass_kg=mass_kg)
    result = stamp_apple_catalog_geometry(path)
    assert result["status"] == "stamped"
    assert result["radius_m"] == pytest.approx(0.035)
    assert result["stem_length_m"] == pytest.approx(0.011)

    table = pq.read_table(path)
    dm = json.loads(table.schema.metadata[b"dataset_metadata"])
    apple = dm["pre_grasp_geometry"]["parts"]["apple"]
    stem = dm["pre_grasp_geometry"]["parts"]["stem"]
    assert apple["mass_kg"] == pytest.approx(mass_kg)
    assert apple["radius_m"] == pytest.approx(0.035)
    assert stem["length_m"] == pytest.approx(0.011)
    check_mass = (4.0 / 3.0) * math.pi * 0.035**3 * apple["density_kg_m3"]
    assert check_mass == pytest.approx(mass_kg, rel=1e-9)


def test_stamp_apple9_rescales_density_keeps_mass(tmp_path: Path):
    path = tmp_path / "s09-d00.parquet"
    mass_kg = 0.290011728
    _write_minimal_parquet(path, apple_number=9, mass_kg=mass_kg)
    result = stamp_apple_catalog_geometry(path)
    assert result["status"] == "stamped"
    assert result["radius_m"] == pytest.approx(0.036)
    assert result["stem_length_m"] == pytest.approx(0.013)
    table = pq.read_table(path)
    apple = json.loads(table.schema.metadata[b"dataset_metadata"])["pre_grasp_geometry"]["parts"]["apple"]
    assert apple["mass_kg"] == pytest.approx(mass_kg)
