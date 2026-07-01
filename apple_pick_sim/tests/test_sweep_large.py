"""Unit tests for large zero-VIC stability sweep helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from apple_pick_sim.diagnostics.analyze_vic_sweep import merge_summary_csvs
from apple_pick_sim.diagnostics.sweep_zero_vic_stability_large import (
    ANGLE_TIER_PRESETS,
    GridCell,
    _grid_cells_phase_a,
    _load_done_ids,
    _slice_for_worker,
    patch_ranges_for_cell,
)
from apple_pick_sim.fruiting_system import load_ranges, default_ranges_fixture_path


def _base_ranges() -> dict:
    return load_ranges(default_ranges_fixture_path())


def _sample_cell(
    *,
    spur_elev_tier: str = "gravity",
    spur_num_segs: int | None = None,
    stem_num_segs: int | None = None,
) -> GridCell:
    return GridCell(
        config_id=0,
        stem_gain=1.0,
        vic_linear_k=600.0,
        vic_linear_d=200.0,
        vic_angular_k=50.0,
        vic_angular_d=4.0,
        spur_elev_tier=spur_elev_tier,
        spur_num_segs=spur_num_segs,
        stem_num_segs=stem_num_segs,
    )


def test_patch_ranges_gravity():
    patched = patch_ranges_for_cell(_base_ranges(), _sample_cell(spur_elev_tier="gravity"))
    assert patched["spur"]["elevation_delta_deg"] == {"min": -90, "max": -10}
    assert patched["stem"]["elevation_delta_deg"] == {"min": -90, "max": -10}


def test_patch_ranges_num_segs():
    patched = patch_ranges_for_cell(
        _base_ranges(),
        _sample_cell(spur_num_segs=3, stem_num_segs=5),
    )
    assert patched["spur"]["num_segments"] == {"min": 3, "max": 3}
    assert patched["stem"]["num_segments"] == {"min": 5, "max": 5}


def test_patch_ranges_none_segs():
    base = _base_ranges()
    patched = patch_ranges_for_cell(base, _sample_cell())
    assert patched["spur"]["num_segments"] == base["spur"]["num_segments"]
    assert patched["stem"]["num_segments"] == base["stem"]["num_segments"]


def test_angle_tier_presets_valid():
    for name, (smn, smx, tmn, tmx) in ANGLE_TIER_PRESETS.items():
        assert smn < smx, name
        assert tmn < tmx, name
        for v in (smn, smx, tmn, tmx):
            assert -90 <= v <= 90, name


def test_resume_skips_done_ids(tmp_path: Path):
    summary = tmp_path / "summary.csv"
    with open(summary, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["config_id"])
        writer.writeheader()
        writer.writerow({"config_id": "1"})
        writer.writerow({"config_id": "3"})
    assert _load_done_ids(summary) == {1, 3}
    assert _load_done_ids(tmp_path / "missing.csv") == set()


def test_worker_slice_coverage():
    cells = list(range(17))
    slices = [_slice_for_worker(cells, w, 5) for w in range(5)]
    assert sum(len(s) for s in slices) == 17
    assert sorted(v for s in slices for v in s) == cells
    assert len({v for s in slices for v in s}) == 17


def test_grid_cell_count_phase_a():
    cells = _grid_cells_phase_a(
        stem_gains=[1.0, 0.95],
        vic_linear_ks=[180.0, 600.0, 1200.0, 2000.0],
        vic_linear_ds=[200.0, 400.0, 600.0],
        vic_angular_ks=[20.0, 50.0, 80.0],
        vic_angular_ds=[4.0, 8.0],
        angle_tiers=["gravity", "level", "overhead"],
    )
    assert len(cells) == 144 * 3


def test_merge_summary_csvs(tmp_path: Path):
    fields = ["config_id", "vic_pass_rate", "max_apple_drift_m"]
    d0 = tmp_path / "w0"
    d1 = tmp_path / "w1"
    d0.mkdir()
    d1.mkdir()
    with open(d0 / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow({"config_id": "0", "vic_pass_rate": "0.5", "max_apple_drift_m": "0.1"})
        w.writerow({"config_id": "2", "vic_pass_rate": "0.8", "max_apple_drift_m": "0.02"})
    with open(d1 / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow({"config_id": "1", "vic_pass_rate": "0.6", "max_apple_drift_m": "0.05"})
        w.writerow({"config_id": "2", "vic_pass_rate": "0.9", "max_apple_drift_m": "0.01"})
    merged = merge_summary_csvs([d0, d1])
    assert len(merged) == 3
    by_id = {int(r["config_id"]): r for r in merged}
    assert by_id[2]["vic_pass_rate"] == "0.9"
