from __future__ import annotations

import numpy as np
import pytest


class _Cand:
    def __init__(self, primary: float, secondary: float, spur: float, stem: float) -> None:
        self.primary = float(primary)
        self.secondary = float(secondary)
        self.spur = float(spur)
        self.stem = float(stem)


def _episode(*, phase: np.ndarray, tcp_shift: float, ft_shift: float, woody_shift: float = 0.0) -> dict:
    n = int(phase.shape[0])
    tcp = np.zeros((n, 3), dtype=np.float32)
    apple = np.zeros((n, 3), dtype=np.float32)
    ft = np.zeros((n, 6), dtype=np.float32)
    tcp[:, 0] = float(tcp_shift)
    apple[:, 0] = float(tcp_shift)
    ft[:, 0] = float(ft_shift)
    ft[:, 3] = float(ft_shift)
    junction_names = ["primary_spur", "spur_stem"]
    woody_start = {
        name: np.tile(np.array([float(woody_shift), 0.0, 0.0], dtype=np.float32), (n, 1))
        for name in junction_names
    }
    woody_end = {
        name: np.tile(np.array([float(woody_shift) + 1.0, 0.0, 0.0], dtype=np.float32), (n, 1))
        for name in junction_names
    }
    return {
        "step_idx": np.arange(n, dtype=np.int32),
        "phase": np.asarray(phase, dtype=np.int8),
        "tcp_pos": tcp,
        "apple_pos": apple,
        "ft_wrist": ft,
        "woody_part_start_pos": woody_start,
        "woody_part_end_pos": woody_end,
        "junction_names": junction_names,
    }


def test_build_grid_viz_rows_marks_gt_and_prefers_gt_errors():
    from apple_pick_gym.grid_viz_table import build_grid_viz_rows

    phase = np.array([0, 1, 1, 2], dtype=np.int8)
    recorded_eps = [_episode(phase=phase, tcp_shift=0.0, ft_shift=0.0)]

    gt = _Cand(1.0, 0.0, 2.0, 3.0)
    far = _Cand(10.0, 0.0, 2.0, 3.0)
    candidates = [gt, far]

    replay_eps_by_candidate = [
        [_episode(phase=phase, tcp_shift=0.0, ft_shift=0.0, woody_shift=0.0)],  # perfect
        [_episode(phase=phase, tcp_shift=1.0, ft_shift=2.0, woody_shift=1.0)],  # worse
    ]

    rows = build_grid_viz_rows(
        structure_idx=0,
        candidates=candidates,
        gt_candidate=gt,
        recorded_eps=recorded_eps,
        replay_eps_by_candidate=replay_eps_by_candidate,
        hold_phase_value=1,
        pos_weights=(1.0, 1.0),
        dist_keys=("primary", "spur", "stem"),
        hold_aggregation="none",
    )

    assert len(rows) == 2
    assert rows[0].gt_flag is True
    assert rows[1].gt_flag is False
    assert rows[0].dist_log_gt == pytest.approx(0.0)
    assert rows[1].dist_log_gt > 0.0

    assert rows[0].err_pos_all == pytest.approx(0.0)
    assert rows[0].err_pos_hold == pytest.approx(0.0)
    assert rows[0].err_force_all == pytest.approx(0.0)
    assert rows[0].err_torque_all == pytest.approx(0.0)

    assert rows[1].err_pos_all > rows[0].err_pos_all
    assert rows[1].err_pos_hold > rows[0].err_pos_hold
    assert rows[1].err_force_all > rows[0].err_force_all
    assert rows[1].err_torque_all > rows[0].err_torque_all

    # err_pos_* includes woody (equal weights): sum of tcp, apple, and mean woody MSE.
    assert rows[0].err_woody_pos_all == pytest.approx(0.0)
    assert rows[0].err_woody_pos_hold == pytest.approx(0.0)
    assert rows[1].err_woody_pos_all > rows[0].err_woody_pos_all
    assert set(rows[0].woody_pos_mse_all) == {"primary_spur", "spur_stem"}
    tcp_apple_only = rows[1].err_pos_all - rows[1].err_woody_pos_all
    assert tcp_apple_only > 0.0


def test_build_grid_viz_rows_marks_gt_within_tolerance():
    from apple_pick_gym.grid_viz_table import build_grid_viz_rows

    phase = np.array([0, 1, 1, 2], dtype=np.int8)
    recorded_eps = [_episode(phase=phase, tcp_shift=0.0, ft_shift=0.0)]

    gt = _Cand(1.0, 0.0, 2.0, 3.0)
    near_gt = _Cand(1.0, 1e-12, 2.0, 3.0)
    candidates = [near_gt]

    replay_eps_by_candidate = [
        [_episode(phase=phase, tcp_shift=0.0, ft_shift=0.0, woody_shift=0.0)],
    ]

    rows = build_grid_viz_rows(
        structure_idx=0,
        candidates=candidates,
        gt_candidate=gt,
        recorded_eps=recorded_eps,
        replay_eps_by_candidate=replay_eps_by_candidate,
        hold_phase_value=1,
        pos_weights=(1.0, 1.0),
        dist_keys=("primary", "spur", "stem"),
        hold_aggregation="none",
    )

    assert len(rows) == 1
    assert rows[0].gt_flag is True
    assert rows[0].dist_log_gt == pytest.approx(0.0)


def test_build_grid_viz_rows_without_woody_data_uses_nan_scalar():
    from apple_pick_gym.grid_viz_table import build_grid_viz_rows

    phase = np.array([0, 1, 1, 2], dtype=np.int8)
    recorded_eps = [
        {
            "step_idx": np.arange(4, dtype=np.int32),
            "phase": phase,
            "tcp_pos": np.zeros((4, 3), dtype=np.float32),
            "apple_pos": np.zeros((4, 3), dtype=np.float32),
            "ft_wrist": np.zeros((4, 6), dtype=np.float32),
        }
    ]
    gt = _Cand(1.0, 0.0, 2.0, 3.0)
    candidates = [gt]
    replay_eps_by_candidate = [recorded_eps]

    rows = build_grid_viz_rows(
        structure_idx=0,
        candidates=candidates,
        gt_candidate=gt,
        recorded_eps=recorded_eps,
        replay_eps_by_candidate=replay_eps_by_candidate,
        hold_phase_value=1,
        hold_aggregation="none",
    )

    assert rows[0].woody_pos_mse_all == {}
    assert rows[0].woody_pos_mse_hold == {}
    assert not np.isfinite(rows[0].err_woody_pos_all)
    assert not np.isfinite(rows[0].err_woody_pos_hold)
    # Without woody data, err_pos_* falls back to tcp+apple only.
    assert rows[0].err_pos_all == pytest.approx(0.0)
    assert rows[0].err_pos_hold == pytest.approx(0.0)

