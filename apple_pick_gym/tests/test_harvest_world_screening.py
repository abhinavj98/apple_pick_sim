"""Harvest world screening: pass/fail criteria (pure) and scripted probe actions."""

from __future__ import annotations

import numpy as np
import torch

from apple_pick_gym.batched_envs.harvest_world_screening import (
    PULL_AXES,
    ScreeningConfig,
    evaluate_screening,
    pull_axis_indices,
    pull_delta_schedule,
)


def _metrics(n=3, **over):
    m = {
        "rest_invalid": np.zeros(n, bool),
        "nonfinite": np.zeros(n, bool),
        "hold_max_wrist_n": np.full(n, 1.0),
        "hold_tcp_drift_m": np.full(n, 0.002),
        "pull_max_wrist_n": np.full(n, 12.0),
        "pull_max_track_err_m": np.full(n, 0.02),
        "pull_max_junction_n": np.full(n, 20.0),
        "pull_end_wrist_excess_n": np.full(n, 0.5),
    }
    m.update(over)
    return m


def test_healthy_worlds_pass_with_no_reasons():
    passed, reasons = evaluate_screening(_metrics(), ScreeningConfig())
    assert passed.tolist() == [True, True, True]
    assert reasons == [[], [], []]


def test_each_criterion_rejects_only_the_offending_world():
    cfg = ScreeningConfig()
    cases = {
        "rest_invalid": np.array([True, False, False]),
        "nonfinite": np.array([True, False, False]),
        "hold_max_wrist_n": np.array([cfg.max_hold_wrist_n + 1, 1.0, 1.0]),
        "hold_tcp_drift_m": np.array([cfg.max_hold_drift_m + 0.01, 0.0, 0.0]),
        "pull_max_wrist_n": np.array([cfg.max_pull_wrist_n + 1, 1.0, 1.0]),
        "pull_max_track_err_m": np.array([cfg.max_pull_track_err_m + 0.01, 0.0, 0.0]),
        "pull_max_junction_n": np.array([cfg.max_pull_junction_n + 1, 1.0, 1.0]),
        "pull_end_wrist_excess_n": np.array([cfg.max_pull_end_wrist_excess_n + 1, 0.0, 0.0]),
    }
    for key, val in cases.items():
        passed, reasons = evaluate_screening(_metrics(**{key: val}), cfg)
        assert passed.tolist() == [False, True, True], key
        assert reasons[0] and key in reasons[0][0], (key, reasons)


def test_nan_metrics_fail_rather_than_pass():
    passed, _ = evaluate_screening(_metrics(pull_max_wrist_n=np.array([np.nan, 1.0, 1.0])), ScreeningConfig())
    assert passed.tolist() == [False, True, True]


def test_pull_axes_cover_all_six_directions_across_envs_and_episodes():
    idx = pull_axis_indices(num_envs=2, num_episodes=3)
    assert idx.shape == (3, 2)
    assert sorted(idx.flatten().tolist()) == [0, 1, 2, 3, 4, 5]
    assert np.allclose(np.abs(PULL_AXES).sum(axis=1), 1.0)


def test_pull_schedule_moves_out_and_returns_to_start():
    cfg = ScreeningConfig()
    deltas = pull_delta_schedule(cfg)
    assert deltas.shape == (cfg.pull_episode_steps,)
    cum = np.cumsum(deltas)
    assert np.isclose(cum.max(), cfg.pull_amplitude_m)
    assert np.isclose(cum[-1], 0.0, atol=1e-9)
    assert np.all(np.abs(deltas) <= 0.02 + 1e-12)  # within the harvest action's per-step bound


def test_probe_action_layout_packs_axis_delta_and_real_gains():
    from apple_pick_gym.batched_envs.harvest_world_screening import probe_actions

    cfg = ScreeningConfig()
    axes = torch.as_tensor(PULL_AXES[[0, 5]], dtype=torch.float32)
    a = probe_actions(axes, delta_m=0.001, cfg=cfg, device="cpu")
    assert a.shape == (2, 13)
    torch.testing.assert_close(a[:, :3], axes * 0.001)
    assert torch.all(a[:, 3:6] == 0)
    assert torch.all(a[:, 6:9] == cfg.pull_linear_k)
    assert torch.all(a[:, 9:12] == cfg.pull_angular_k)
    assert torch.all(a[:, 12] == cfg.pull_zeta)
