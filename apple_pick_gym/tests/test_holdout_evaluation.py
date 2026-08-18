"""Pure unit tests for holdout report builders and verification gates."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes
from apple_pick_gym.batched_envs.holdout_evaluation import (
    build_holdout_report,
    cartesian_ft_mae,
    direction_verification,
    holdout_gate_failures,
    write_holdout_report,
)


def _episode_with_holds(
    *,
    force_z: float = 2.0,
    torque_y: float = 0.2,
    tcp_z_hold0: float = 1.0,
    tcp_z_pull: float = 0.0,
    apple_z_hold0: float = 1.5,
    n_holds: int = 4,
    pull_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    force_scale: float = 1.0,
    torque_scale: float = 1.0,
    tcp_scale: float = 1.0,
    tcp_hold_step: float = 0.05,
    apple_scale: float = 1.0,
    flip_force: bool = False,
) -> dict:
    """Synthetic episode: pull frame(s) plus ``n_holds`` single-frame hold segments."""
    frames: list[dict] = []
    frames.append(
        {
            "phase": 0,
            "force_z": 999.0 * force_scale,
            "torque_y": 999.0 * torque_scale,
            "tcp_z": tcp_z_pull,
            "apple_z": 0.0,
        }
    )
    for hold_i in range(n_holds):
        if hold_i > 0:
            frames.append(
                {
                    "phase": 0,
                    "force_z": 50.0 * force_scale,
                    "torque_y": 5.0 * torque_scale,
                    "tcp_z": tcp_z_pull,
                    "apple_z": 0.0,
                }
            )
        fz = force_z * force_scale * (1.0 + 0.1 * hold_i)
        if flip_force:
            fz = -fz
        frames.append(
            {
                "phase": 1,
                "force_z": fz,
                "torque_y": torque_y * torque_scale * (1.0 + 0.05 * hold_i),
                "tcp_z": tcp_z_hold0 + tcp_hold_step * hold_i * tcp_scale,
                "apple_z": apple_z_hold0 + 0.04 * hold_i * apple_scale,
            }
        )

    n = len(frames)
    ft = np.zeros((n, 6), dtype=np.float64)
    tcp = np.zeros((n, 3), dtype=np.float64)
    apple = np.zeros((n, 3), dtype=np.float64)
    phase = np.zeros(n, dtype=np.int8)
    excitation = np.tile(np.asarray(pull_direction, dtype=np.float64), (n, 1))
    for i, row in enumerate(frames):
        phase[i] = int(row["phase"])
        ft[i, 2] = row["force_z"]
        ft[i, 4] = row["torque_y"]
        tcp[i, 2] = row["tcp_z"]
        apple[i, 2] = row["apple_z"]

    return {
        "action": np.zeros((n, 6), dtype=np.float32),
        "phase": phase,
        "dir_idx": np.zeros(n, dtype=np.int32),
        "ft_wrist": ft.astype(np.float32),
        "ft_wrist_lpf": (ft * 0.99).astype(np.float32),
        "tcp_pos": tcp.astype(np.float32),
        "apple_pos": apple.astype(np.float32),
        "excitation_direction": excitation.astype(np.float32),
    }


def test_cartesian_ft_mae_uses_hold_frames_only():
    real = _episode_with_holds(force_z=2.0, torque_y=0.2)
    fitted = _episode_with_holds(force_z=3.0, torque_y=0.3)
    # Corrupt pull frame only; hold frames differ by +1 N / +0.1 N·m on z/y.
    fitted["ft_wrist"][0, 2] = 1.0e6
    fitted["ft_wrist_lpf"][0, 2] = 1.0e6

    mae_f, mae_t = cartesian_ft_mae(real=real, fitted=fitted, direction=0)
    hold_idx = [
        i
        for i in range(len(real["phase"]))
        if int(real["phase"][i]) == 1
    ]
    from apple_pick_sim.system_id.mmd_features import scored_ft_wrist

    real_ft = np.asarray(scored_ft_wrist(real), dtype=np.float64)
    fit_ft = np.asarray(scored_ft_wrist(fitted), dtype=np.float64)
    expected_f = float(
        np.mean(np.linalg.norm(real_ft[hold_idx, :3] - fit_ft[hold_idx, :3], axis=1))
    )
    expected_t = float(
        np.mean(np.linalg.norm(real_ft[hold_idx, 3:6] - fit_ft[hold_idx, 3:6], axis=1))
    )
    assert mae_f == pytest.approx(expected_f)
    assert mae_t == pytest.approx(expected_t)

    # LPF column is preferred when present.
    real_no_lpf = dict(real)
    del real_no_lpf["ft_wrist_lpf"]
    real_no_lpf["ft_wrist"] = np.asarray(fitted["ft_wrist_lpf"], dtype=np.float32)
    fitted_lpf_only = dict(fitted)
    fitted_lpf_only["ft_wrist"] = np.zeros_like(fitted["ft_wrist"])
    mae_f_lpf, _ = cartesian_ft_mae(
        real=real_no_lpf, fitted=fitted_lpf_only, direction=0
    )
    assert mae_f_lpf == pytest.approx(0.0)


def test_direction_verification_passes_matching_signed_series():
    ep = _episode_with_holds()
    report = direction_verification(
        real=ep,
        fitted=ep,
        direction=0,
        pull_direction=(0.0, 0.0, 1.0),
    )
    assert report["force_magnitude_ok"] is True
    assert report["force_trend_ok"] is True
    assert report["tcp_pose_magnitude_ok"] is True
    assert report["tcp_pose_trend_ok"] is True
    for key in (
        "force_ratio",
        "torque_ratio",
        "force_pearson_r",
        "tcp_ratio",
        "tcp_pearson_r",
    ):
        assert key in report


def test_direction_verification_fails_flipped_force_sign():
    real = _episode_with_holds()
    fitted = _episode_with_holds(flip_force=True)
    report = direction_verification(
        real=real,
        fitted=fitted,
        direction=0,
        pull_direction=(0.0, 0.0, 1.0),
    )
    assert report["force_trend_ok"] is False
    assert report["force_magnitude_ok"] is True


def test_direction_verification_fails_ten_times_stiff_pose():
    real = _episode_with_holds(tcp_z_pull=0.0, tcp_z_hold0=1.0, tcp_hold_step=0.5)
    fitted = _episode_with_holds(
        tcp_z_pull=0.0, tcp_z_hold0=1.0, tcp_hold_step=0.5, tcp_scale=0.1
    )
    report = direction_verification(
        real=real,
        fitted=fitted,
        direction=0,
        pull_direction=(0.0, 0.0, 1.0),
    )
    assert report["tcp_pose_magnitude_ok"] is False


def test_direction_verification_fails_torque_magnitude():
    real = _episode_with_holds()
    fitted = _episode_with_holds(torque_scale=10.0)
    report = direction_verification(
        real=real,
        fitted=fitted,
        direction=0,
        pull_direction=(0.0, 0.0, 1.0),
    )
    assert report["force_magnitude_ok"] is False


def test_direction_verification_apple_pose_is_diagnostic_only():
    real = _episode_with_holds()
    fitted = _episode_with_holds(apple_scale=0.01)
    report = direction_verification(
        real=real,
        fitted=fitted,
        direction=0,
        pull_direction=(0.0, 0.0, 1.0),
    )
    assert report["force_magnitude_ok"] is True
    assert report["force_trend_ok"] is True
    assert report["tcp_pose_magnitude_ok"] is True
    assert report["tcp_pose_trend_ok"] is True
    assert "apple_ratio" in report


def test_build_holdout_report_has_required_keys():
    report = build_holdout_report(
        structure_idx=0,
        direction_split_seed=17,
        train_direction_indices=[5, 2, 7, 4, 6],
        val_direction_indices=[3, 0, 1],
        baseline_log10=[4.0, 9.5, 9.5],
        fitted_log10=[4.1, 9.4, 9.6],
        train_fitted={
            "eligible_mean_sinkhorn": 1.0,
            "mae_force_n": 0.1,
            "mae_torque_nm": 0.02,
            "per_direction_mae_force_n": {2: 0.1},
            "per_direction_mae_torque_nm": {2: 0.02},
        },
        val_baseline={
            "eligible_mean_sinkhorn": 2.0,
            "mae_force_n": 0.2,
            "mae_torque_nm": 0.03,
            "per_direction_mae_force_n": {0: 0.2},
            "per_direction_mae_torque_nm": {0: 0.03},
        },
        val_fitted={
            "eligible_mean_sinkhorn": 1.5,
            "mae_force_n": 0.15,
            "mae_torque_nm": 0.025,
            "per_direction_mae_force_n": {0: 0.15},
            "per_direction_mae_torque_nm": {0: 0.025},
        },
        train_eligible_means=[2.0, 1.0],
        val_overlay_paths={0: "/tmp/d0.html", 1: "/tmp/d1.html", 3: "/tmp/d3.html"},
        val_verification_by_direction={
            direction: {
                "force_magnitude_ok": True,
                "force_trend_ok": True,
                "tcp_pose_magnitude_ok": True,
                "tcp_pose_trend_ok": True,
                "force_ratio": 1.0,
                "torque_ratio": 1.0,
                "force_pearson_r": 1.0,
                "tcp_ratio": 1.0,
                "tcp_pearson_r": 1.0,
            }
            for direction in (0, 1, 3)
        },
    )
    assert set(report.keys()) == {
        "structure_idx",
        "direction_split_seed",
        "train_direction_indices",
        "val_direction_indices",
        "phenotype_log10",
        "train_fitted",
        "val_baseline",
        "val_fitted",
        "verification",
        "val_overlay_paths",
    }
    assert report["train_direction_indices"] == [2, 4, 5, 6, 7]
    assert report["val_direction_indices"] == [0, 1, 3]
    assert report["verification"]["train_sinkhorn_decreased"] is True
    assert report["verification"]["val_sinkhorn_improved"] is True
    assert report["verification"]["by_direction"]["0"]["force_magnitude_ok"] is True

    no_seed = build_holdout_report(
        structure_idx=0,
        direction_split_seed=None,
        train_direction_indices=[0],
        val_direction_indices=[1],
        baseline_log10=[4.0, 9.5, 9.5],
        fitted_log10=[4.1, 9.4, 9.6],
        train_fitted={
            "eligible_mean_sinkhorn": 1.0,
            "mae_force_n": 0.1,
            "mae_torque_nm": 0.02,
            "per_direction_mae_force_n": {0: 0.1},
            "per_direction_mae_torque_nm": {0: 0.02},
        },
        val_baseline={
            "eligible_mean_sinkhorn": 2.0,
            "mae_force_n": 0.2,
            "mae_torque_nm": 0.03,
            "per_direction_mae_force_n": {1: 0.2},
            "per_direction_mae_torque_nm": {1: 0.03},
        },
        val_fitted={
            "eligible_mean_sinkhorn": 1.5,
            "mae_force_n": 0.15,
            "mae_torque_nm": 0.025,
            "per_direction_mae_force_n": {1: 0.15},
            "per_direction_mae_torque_nm": {1: 0.025},
        },
        train_eligible_means=[2.0, 1.0],
        val_overlay_paths={1: "/tmp/d1.html"},
        val_verification_by_direction={
            1: {
                "force_magnitude_ok": True,
                "force_trend_ok": True,
                "tcp_pose_magnitude_ok": True,
                "tcp_pose_trend_ok": True,
            }
        },
    )
    assert "direction_split_seed" not in no_seed
    strict = cmaes.to_strict_jsonable(report)
    json.dumps(strict)


def test_build_holdout_report_requires_finite_metrics():
    with pytest.raises(ValueError, match="finite"):
        build_holdout_report(
            structure_idx=0,
            direction_split_seed=17,
            train_direction_indices=[2],
            val_direction_indices=[0],
            baseline_log10=[4.0, 9.5, 9.5],
            fitted_log10=[4.1, 9.4, 9.6],
            train_fitted={
                "eligible_mean_sinkhorn": float("nan"),
                "mae_force_n": 0.1,
                "mae_torque_nm": 0.02,
                "per_direction_mae_force_n": {2: 0.1},
                "per_direction_mae_torque_nm": {2: 0.02},
            },
            val_baseline={
                "eligible_mean_sinkhorn": 2.0,
                "mae_force_n": 0.2,
                "mae_torque_nm": 0.03,
                "per_direction_mae_force_n": {0: 0.2},
                "per_direction_mae_torque_nm": {0: 0.03},
            },
            val_fitted={
                "eligible_mean_sinkhorn": 1.5,
                "mae_force_n": 0.15,
                "mae_torque_nm": 0.025,
                "per_direction_mae_force_n": {0: 0.15},
                "per_direction_mae_torque_nm": {0: 0.025},
            },
            train_eligible_means=[2.0, 1.0],
            val_overlay_paths={0: "/tmp/d0.html"},
            val_verification_by_direction={
                0: {
                    "force_magnitude_ok": True,
                    "force_trend_ok": True,
                    "tcp_pose_magnitude_ok": True,
                    "tcp_pose_trend_ok": True,
                }
            },
        )


def test_write_holdout_report_is_atomic(tmp_path: Path):
    out = write_holdout_report(
        tmp_path,
        {
            "structure_idx": 0,
            "train_direction_indices": [0],
            "val_direction_indices": [1],
            "phenotype_log10": {"baseline": [1.0, 2.0, 3.0], "fitted": [1.1, 2.1, 3.1]},
            "train_fitted": {
                "eligible_mean_sinkhorn": 1.0,
                "mae_force_n": 0.1,
                "mae_torque_nm": 0.01,
            },
            "val_baseline": {
                "eligible_mean_sinkhorn": 2.0,
                "mae_force_n": 0.2,
                "mae_torque_nm": 0.02,
            },
            "val_fitted": {
                "eligible_mean_sinkhorn": 1.5,
                "mae_force_n": 0.15,
                "mae_torque_nm": 0.015,
            },
            "verification": {
                "train_sinkhorn_decreased": True,
                "val_sinkhorn_improved": True,
                "by_direction": {},
            },
            "val_overlay_paths": {"1": str(tmp_path / "d1.html")},
        },
    )
    assert out == tmp_path / "holdout_report.json"
    assert out.is_file()


def test_holdout_gate_failures_names_gate_and_direction():
    report = {
        "verification": {
            "train_sinkhorn_decreased": True,
            "val_sinkhorn_improved": False,
            "by_direction": {
                "3": {
                    "force_magnitude_ok": False,
                    "force_trend_ok": True,
                    "tcp_pose_magnitude_ok": True,
                    "tcp_pose_trend_ok": True,
                }
            },
        }
    }
    failures = holdout_gate_failures(report)
    assert "val_sinkhorn_improved" in failures[0]
    assert any("force_magnitude_ok" in msg and "3" in msg for msg in failures)
