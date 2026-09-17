"""Tests for CMA holdout force RMSE / Pearson reporting."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from apple_pick_sim.system_id.cma_rmse_pearson import (
    DEFAULT_FEATURES,
    extract_feature_series,
    force_rmse_and_pearson,
    load_run_feature_rmse_pearson,
    load_run_force_rmse_pearson,
    scalar_rmse_and_pearson,
    summarize_feature_runs,
    summarize_runs,
)


def test_identical_force_series_rmse_zero_pearson_one() -> None:
    t = np.linspace(0.0, 1.0, 50)
    force = np.column_stack([np.sin(t), np.cos(t), t])
    rmse, pearson = force_rmse_and_pearson(force, force)
    assert rmse == pytest.approx(0.0, abs=1e-12)
    assert pearson == pytest.approx(1.0, abs=1e-12)


def test_scaled_force_series_keeps_pearson_one() -> None:
    t = np.linspace(0.0, 2.0, 80)
    real = np.column_stack([np.sin(t), 0.5 * np.cos(t), t])
    sim = 2.0 * real
    rmse, pearson = force_rmse_and_pearson(real, sim)
    assert rmse > 0.0
    assert pearson == pytest.approx(1.0, abs=1e-12)


def test_anticorrelated_force_magnitudes_negative_pearson() -> None:
    t = np.linspace(0.0, 3.0, 100)
    real = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    sim = np.column_stack([3.0 - t, np.zeros_like(t), np.zeros_like(t)])
    _, pearson = force_rmse_and_pearson(real, sim)
    assert pearson == pytest.approx(-1.0, abs=1e-12)


def test_load_run_uses_match_metrics_and_npz(tmp_path: Path) -> None:
    run = tmp_path / "cma_s02_val03_seed56"
    holdout = run / "structure_000" / "holdout" / "fitted"
    holdout.mkdir(parents=True)
    t = np.linspace(0.0, 1.0, 40)
    real_f = np.column_stack([np.sin(t), np.cos(t), 0.1 * t])
    sim_f = 1.1 * real_f
    # state matrix: first 6 cols are ft_wrist
    real_state = np.zeros((40, 26), dtype=np.float32)
    sim_state = np.zeros((40, 26), dtype=np.float32)
    real_state[:, :3] = real_f
    sim_state[:, :3] = sim_f
    np.savez(
        holdout / "dir_00.npz",
        real_state=real_state,
        sim_state=sim_state,
        phase_real=np.ones(40, dtype=np.int8),
        phase_sim=np.ones(40, dtype=np.int8),
        stable_real=np.ones(40, dtype=bool),
        stable_sim=np.ones(40, dtype=bool),
    )
    (run / "match_metrics.json").write_text(
        json.dumps(
            {
                "tree": "s02",
                "cma_seed": 56,
                "val_direction_indices": [0],
                "directions": {
                    "0": {
                        "full": {
                            "fitted": {
                                "force": {
                                    "combined": {
                                        "phase_corrected_rmse": 0.05,
                                        "mse": 0.0025,
                                    }
                                }
                            }
                        }
                    }
                },
            }
        )
    )
    row = load_run_force_rmse_pearson(run)
    assert row.structure == 2
    assert row.seed == 56
    assert len(row.per_direction) == 1
    assert row.per_direction[0].direction == 0
    assert row.per_direction[0].rmse == pytest.approx(0.05, rel=1e-6, abs=1e-6)
    assert row.per_direction[0].pearson == pytest.approx(1.0, abs=1e-9)
    assert row.mean_rmse == pytest.approx(0.05, rel=1e-6, abs=1e-6)
    assert row.mean_pearson == pytest.approx(1.0, abs=1e-9)


def test_summarize_runs_groups_by_structure(tmp_path: Path) -> None:
    for seed in (56, 57):
        run = tmp_path / f"cma_s03_val03_seed{seed}"
        holdout = run / "structure_000" / "holdout" / "fitted"
        holdout.mkdir(parents=True)
        n = 30
        real_f = np.column_stack(
            [np.linspace(0, 1, n), np.zeros(n), np.zeros(n)]
        )
        sim_f = real_f * (1.0 + 0.1 * (seed - 56))
        real_state = np.zeros((n, 26), dtype=np.float32)
        sim_state = np.zeros((n, 26), dtype=np.float32)
        real_state[:, :3] = real_f
        sim_state[:, :3] = sim_f
        np.savez(
            holdout / "dir_00.npz",
            real_state=real_state,
            sim_state=sim_state,
            phase_real=np.ones(n, dtype=np.int8),
            phase_sim=np.ones(n, dtype=np.int8),
            stable_real=np.ones(n, dtype=bool),
            stable_sim=np.ones(n, dtype=bool),
        )
        rmse = float(np.sqrt(np.mean(np.sum((sim_f - real_f) ** 2, axis=1))))
        # For magnitude-based phase RMSE in vector3_match_stats it's different;
        # store a known value in match_metrics and assert loader prefers it.
        (run / "match_metrics.json").write_text(
            json.dumps(
                {
                    "tree": "s03",
                    "cma_seed": seed,
                    "val_direction_indices": [0],
                    "directions": {
                        "0": {
                            "full": {
                                "fitted": {
                                    "force": {
                                        "combined": {
                                            "phase_corrected_rmse": 0.1 * (seed - 55),
                                            "mse": rmse**2,
                                        }
                                    }
                                }
                            }
                        }
                    },
                }
            )
        )
    summary = summarize_runs(tmp_path, structures=(3,), seeds=(56, 57))
    assert len(summary) == 2
    assert {r.seed for r in summary} == {56, 57}
    assert all(r.structure == 3 for r in summary)


def test_identical_scalar_series_rmse_zero_pearson_one() -> None:
    t = np.linspace(0.0, 2.0, 60)
    rmse, pearson = scalar_rmse_and_pearson(t, t.copy())
    assert rmse == pytest.approx(0.0, abs=1e-12)
    assert pearson == pytest.approx(1.0, abs=1e-12)


def test_lagged_scalar_series_phase_corrects_pearson() -> None:
    t = np.linspace(0.0, 4.0, 120)
    real = np.sin(t)
    sim = np.concatenate([np.zeros(5), real[:-5]])
    rmse, pearson = scalar_rmse_and_pearson(real, sim, max_lag=10)
    assert rmse == pytest.approx(0.0, abs=1e-6)
    assert pearson == pytest.approx(1.0, abs=1e-6)


def test_extract_feature_series_force_torque_tcp() -> None:
    n = 20
    state = np.zeros((n, 26), dtype=np.float64)
    state[:, 0] = np.linspace(0.0, 1.0, n)  # fx
    state[:, 3] = np.linspace(0.0, 2.0, n)  # tx
    state[:, 12] = np.linspace(0.0, 0.1, n)  # tcp x
    state[:, 13] = 0.5  # tcp y
    state[:, 15] = np.linspace(0.0, 0.2, n)  # rotvec x
    pull = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    fx_r, fx_s = extract_feature_series(state, state, "fx")
    assert fx_r.shape == (n,)
    assert fx_r[-1] == pytest.approx(1.0)
    tx_r, _ = extract_feature_series(state, state, "tx")
    assert tx_r[-1] == pytest.approx(2.0)
    force_r, _ = extract_feature_series(state, state, "force_combined")
    assert force_r.shape == (n, 3)
    disp_r, _ = extract_feature_series(
        state, state, "tcp_disp_along_pull", pull_direction=pull
    )
    # y is constant 0.5, so along -Y displacement from frame 0 is 0
    assert disp_r.shape == (n,)
    assert np.allclose(disp_r, 0.0)
    # move along -Y
    state2 = state.copy()
    state2[:, 13] = 0.5 - np.linspace(0.0, 0.3, n)
    disp_r2, _ = extract_feature_series(
        state2, state2, "tcp_disp_along_pull", pull_direction=pull
    )
    assert disp_r2[-1] == pytest.approx(0.3, abs=1e-9)
    ori_r, _ = extract_feature_series(state, state, "tcp_orientation")
    assert ori_r.shape == (n, 3)
    assert ori_r[-1, 0] == pytest.approx(0.2)


def test_extract_woody_junction_series_from_state() -> None:
    n = 15
    state = np.zeros((n, 26), dtype=np.float64)
    # woody_part_start_pos: primary_spur @ 18:21, spur_stem @ 21:24
    state[:, 18] = np.linspace(0.0, 0.05, n)
    state[:, 21] = np.linspace(0.0, 0.08, n)
    state[:, 22] = 0.01
    spur_r, spur_s = extract_feature_series(state, state, "woody_primary_spur")
    stem_r, _ = extract_feature_series(state, state, "woody_spur_stem")
    assert spur_r.shape == (n, 3)
    assert spur_r[-1, 0] == pytest.approx(0.05)
    assert stem_r[-1, 0] == pytest.approx(0.08)
    assert stem_r[-1, 1] == pytest.approx(0.01)


def test_woody_displacement_mode_phase_metrics() -> None:
    from apple_pick_sim.system_id.cma_rmse_pearson import vector3_rmse_and_pearson

    t = np.linspace(0.0, 1.0, 50)
    # Absolute positions offset, but identical displacement from frame 0.
    real = np.column_stack([0.1 + 0.05 * t, np.zeros_like(t), np.zeros_like(t)])
    sim = np.column_stack([0.4 + 0.05 * t, np.zeros_like(t), np.zeros_like(t)])
    rmse, pearson = vector3_rmse_and_pearson(
        real, sim, peak_mode="displacement"
    )
    assert rmse == pytest.approx(0.0, abs=1e-12)
    assert pearson == pytest.approx(1.0, abs=1e-12)


def test_load_run_feature_rmse_pearson_per_feature(tmp_path: Path) -> None:
    run = tmp_path / "cma_s02_val03_seed56"
    holdout = run / "structure_000" / "holdout" / "fitted"
    holdout.mkdir(parents=True)
    n = 40
    t = np.linspace(0.0, 1.0, n)
    real_state = np.zeros((n, 26), dtype=np.float32)
    sim_state = np.zeros((n, 26), dtype=np.float32)
    real_state[:, 0] = np.sin(t)
    sim_state[:, 0] = 1.1 * np.sin(t)
    real_state[:, 3] = np.cos(t)
    sim_state[:, 3] = 1.05 * np.cos(t)
    real_state[:, 12:15] = np.column_stack([0.01 * t, -0.02 * t, np.zeros(n)])
    sim_state[:, 12:15] = real_state[:, 12:15] * 1.02
    real_state[:, 15:18] = np.column_stack([0.1 * t, np.zeros(n), np.zeros(n)])
    sim_state[:, 15:18] = real_state[:, 15:18]
    real_state[:, 18:21] = np.column_stack([0.02 * t, np.zeros(n), np.zeros(n)])
    sim_state[:, 18:21] = real_state[:, 18:21] * 1.1
    real_state[:, 21:24] = np.column_stack([0.03 * t, np.zeros(n), np.zeros(n)])
    sim_state[:, 21:24] = real_state[:, 21:24]
    np.savez(
        holdout / "dir_00.npz",
        real_state=real_state,
        sim_state=sim_state,
        phase_real=np.ones(n, dtype=np.int8),
        phase_sim=np.ones(n, dtype=np.int8),
        stable_real=np.ones(n, dtype=bool),
        stable_sim=np.ones(n, dtype=bool),
    )
    (run / "match_metrics.json").write_text(
        json.dumps(
            {
                "tree": "s02",
                "cma_seed": 56,
                "val_direction_indices": [0],
                "directions": {
                    "0": {
                        "full": {
                            "fitted": {
                                "force": {
                                    "fx": {"phase_corrected_rmse": 0.11},
                                    "fy": {"phase_corrected_rmse": 0.0},
                                    "fz": {"phase_corrected_rmse": 0.0},
                                    "combined": {"phase_corrected_rmse": 0.12},
                                },
                                "torque": {
                                    "tx": {"phase_corrected_rmse": 0.03},
                                    "ty": {"phase_corrected_rmse": 0.0},
                                    "tz": {"phase_corrected_rmse": 0.0},
                                    "combined": {"phase_corrected_rmse": 0.04},
                                },
                                "woody": {
                                    "primary_spur": {"phase_corrected_rmse": 0.002},
                                    "spur_stem": {"phase_corrected_rmse": 0.001},
                                },
                            }
                        }
                    }
                },
            }
        )
    )
    (run / "cmaes_report.json").write_text(
        json.dumps({"dataset": str(tmp_path / "real_batched_s02")})
    )
    ds = tmp_path / "real_batched_s02"
    ds.mkdir()
    (ds / "manifest.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "direction_idx": 0,
                        "structure_idx": 0,
                        "excluded": False,
                        "pull_direction": [0.0, -1.0, 0.0],
                    }
                ]
            }
        )
    )
    row = load_run_feature_rmse_pearson(run, features=DEFAULT_FEATURES)
    assert row.structure == 2
    assert row.seed == 56
    assert len(row.per_direction) == 1
    feats = {f.feature: f for f in row.per_direction[0].features}
    assert set(feats) == set(DEFAULT_FEATURES)
    assert feats["fx"].rmse == pytest.approx(0.11, abs=1e-9)
    assert feats["force_combined"].rmse == pytest.approx(0.12, abs=1e-9)
    assert feats["tx"].rmse == pytest.approx(0.03, abs=1e-9)
    assert feats["fx"].pearson == pytest.approx(1.0, abs=1e-9)
    assert feats["tcp_orientation"].rmse == pytest.approx(0.0, abs=1e-9)
    assert feats["tcp_orientation"].pearson == pytest.approx(1.0, abs=1e-9)
    assert math.isfinite(feats["tcp_disp_along_pull"].rmse)
    assert feats["woody_primary_spur"].rmse == pytest.approx(0.002, abs=1e-9)
    assert feats["woody_spur_stem"].rmse == pytest.approx(0.001, abs=1e-9)
    assert feats["woody_spur_stem"].pearson == pytest.approx(1.0, abs=1e-9)
    assert feats["woody_primary_spur"].pearson == pytest.approx(1.0, abs=1e-9)
    summary = summarize_feature_runs(
        tmp_path, structures=(2,), seeds=(56,), features=("fx", "tcp_disp_along_pull")
    )
    assert len(summary) == 1
    assert {f.feature for f in summary[0].per_direction[0].features} == {
        "fx",
        "tcp_disp_along_pull",
    }
