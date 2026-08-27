"""Tared F/T must subtract the unloaded EMA column, not unfiltered raw."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from robot_replay.tare_unloaded_ema import (
    add_ema_tare_columns,
    match_baseline_wrench,
    tared_ft_from_loaded_and_baseline,
)


def test_match_baseline_wrench_uses_frame_index_without_interpolation():
    source = np.arange(54, dtype=np.float64).reshape(9, 6)
    loaded_hold = np.zeros(10, dtype=np.int32)
    loaded_phase = np.zeros(10, dtype=np.int32)
    loaded_step = np.arange(10, dtype=np.int32)
    base_hold = np.zeros(9, dtype=np.int32)
    base_phase = np.zeros(9, dtype=np.int32)
    base_step = np.arange(9, dtype=np.int32)

    matched = match_baseline_wrench(
        loaded_hold,
        loaded_phase,
        loaded_step,
        base_hold,
        base_phase,
        base_step,
        source,
        max_relative_difference=0.5,
    )

    np.testing.assert_array_equal(matched[:9], source)
    np.testing.assert_array_equal(matched[9], source[-1])


def test_tared_ft_subtracts_unloaded_ema_not_raw():
    loaded_raw = np.full((4, 6), 5.0, dtype=np.float64)
    unloaded_ema = np.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]] * 2,
        dtype=np.float64,
    )
    unloaded_raw = unloaded_ema + 10.0
    hold = np.zeros(4, dtype=np.int32)
    phase = np.array([0, 0, 1, 1], dtype=np.int32)
    step = np.array([0, 1, 0, 1], dtype=np.int32)

    ema_baseline, ema_tared = tared_ft_from_loaded_and_baseline(
        loaded_raw=loaded_raw,
        loaded_hold=hold,
        loaded_phase=phase,
        loaded_step=step,
        baseline_ema=unloaded_ema,
        baseline_raw=unloaded_raw,
        baseline_hold=hold,
        baseline_phase=phase,
        baseline_step=step,
    )

    np.testing.assert_allclose(ema_baseline, unloaded_ema)
    np.testing.assert_allclose(
        ema_tared,
        np.array(
            [
                [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            ]
        ),
    )
    assert not np.allclose(ema_tared, loaded_raw - unloaded_raw)


def _episode_and_baseline(tmp_path: Path) -> tuple[Path, Path]:
    episode = pa.table(
        {
            "timestamp": pa.array([0.0, 0.001, 0.002], type=pa.float64()),
            "hold_index": pa.array([0, 0, 0], type=pa.int32()),
            "phase": pa.array([0, 0, 1], type=pa.int8()),
            "phase_name": pa.array(["pull", "pull", "hold"]),
            "hold_step_idx": pa.array([0, 1, 0], type=pa.int32()),
            "ft_wrist": pa.array(
                [[4.0] * 6, [3.0] * 6, [4.0] * 6], type=pa.list_(pa.float32(), 6)
            ),
            "ft_wrist_raw": pa.array(
                [[5.0] * 6, [5.0] * 6, [5.0] * 6], type=pa.list_(pa.float32(), 6)
            ),
            "ft_wrist_baseline": pa.array(
                [[10.0] * 6, [12.0] * 6, [10.0] * 6], type=pa.list_(pa.float32(), 6)
            ),
        }
    )
    episode = episode.replace_schema_metadata(
        {b"dataset_metadata": json.dumps({"episode_id": "s09-d00"}).encode()}
    )
    episode_path = tmp_path / "s09-d00.parquet"
    pq.write_table(episode, episode_path)

    baseline = pa.table(
        {
            "row_kind": pa.array([None] * 3),
            "hold_index": pa.array([0, 0, 0], type=pa.int32()),
            "phase": pa.array([0, 0, 1], type=pa.int8()),
            "hold_step_idx": pa.array([0, 1, 0], type=pa.int32()),
            "ft_wrist": pa.array(
                [[1.0] * 6, [2.0] * 6, [1.0] * 6], type=pa.list_(pa.float32(), 6)
            ),
            "ft_wrist_raw": pa.array(
                [[10.0] * 6, [12.0] * 6, [10.0] * 6], type=pa.list_(pa.float32(), 6)
            ),
        }
    )
    baseline_path = tmp_path / "baseline_robot.parquet"
    pq.write_table(baseline, baseline_path)
    return episode_path, baseline_path


def test_add_ema_tare_columns_writes_loaded_minus_unloaded_ema(tmp_path):
    episode_path, baseline_path = _episode_and_baseline(tmp_path)
    out = add_ema_tare_columns(episode_path, baseline_path, output_path=tmp_path / "out.parquet")
    table = pq.read_table(out)
    assert "ft_wrist_ema_baseline" in table.column_names
    assert "ft_wrist_ema_tared" in table.column_names
    ema_tared = np.stack(table.column("ft_wrist_ema_tared").to_pylist())
    np.testing.assert_allclose(ema_tared, np.array([[4.0] * 6, [3.0] * 6, [4.0] * 6]))
    meta = json.loads(table.schema.metadata[b"dataset_metadata"])
    assert meta["episode_id"] == "s09-d00"
    assert meta["ema_tare"]["unloaded_column"] == "ft_wrist"
