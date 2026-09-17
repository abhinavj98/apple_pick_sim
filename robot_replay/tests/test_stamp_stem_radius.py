from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robot_replay.stamp_stem_radius import STEM_RADIUS_M, stamp_stem_radius


def _write_parquet(path: Path, dm: dict) -> None:
    meta = {b"dataset_metadata": json.dumps(dm).encode("utf-8")}
    pq.write_table(pa.table({"x": [1]}).replace_schema_metadata(meta), path)


def test_stamp_sets_pre_grasp_and_nested_catalog_placeholder(tmp_path: Path) -> None:
    path = tmp_path / "s04-d00.parquet"
    _write_parquet(
        path,
        {
            "pre_grasp_geometry": {
                "parts": {"stem": {"length_m": 0.01, "radius_m": 0.0005}}
            },
            "dump": {
                "runner_metadata": {
                    "dump": {
                        "structure_catalog_entry": {
                            "parts": {"stem": {"length_m": 0.015, "radius_m": 0.0005}}
                        }
                    },
                    "pre_grasp_geometry": {
                        "parts": {"stem": {"length_m": 0.015, "radius_m": 0.0005}}
                    },
                }
            },
            "source_metadata_summary": {
                "pre_grasp_geometry": {
                    "parts": {"stem": {"length_m": 0.015, "radius_m": 0.0005}}
                }
            },
        },
    )
    result = stamp_stem_radius(path)
    assert result["status"] == "stamped"
    assert result["n_updated"] == 4
    assert result["radius_m"] == pytest.approx(STEM_RADIUS_M)

    dm = json.loads(pq.read_table(path).schema.metadata[b"dataset_metadata"])
    assert dm["pre_grasp_geometry"]["parts"]["stem"]["radius_m"] == pytest.approx(0.0009)
    catalog = dm["dump"]["runner_metadata"]["dump"]["structure_catalog_entry"]["parts"]["stem"]
    assert catalog["radius_m"] == pytest.approx(0.0009)
    nested = dm["dump"]["runner_metadata"]["pre_grasp_geometry"]["parts"]["stem"]
    assert nested["radius_m"] == pytest.approx(0.0009)
    summary = dm["source_metadata_summary"]["pre_grasp_geometry"]["parts"]["stem"]
    assert summary["radius_m"] == pytest.approx(0.0009)


def test_stamp_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "s02-d00.parquet"
    _write_parquet(
        path,
        {"pre_grasp_geometry": {"parts": {"stem": {"radius_m": 0.0005}}}},
    )
    first = stamp_stem_radius(path)
    second = stamp_stem_radius(path)
    assert first["status"] == "stamped"
    assert first["n_updated"] == 1
    assert second["n_updated"] == 0
    assert second["status"] == "unchanged"
