"""Unit tests for zero-VIC hold stability metrics."""

from __future__ import annotations

import math

import pytest

from apple_pick_sim.diagnostics.zero_vic_stability_metrics import (
    compute_env_stability_metrics,
    parse_float_list,
    summarize_hold_metrics,
)


def _row(
    *,
    t_s: float,
    env: int,
    pos_err_m: float = 0.01,
    tcp_v: tuple[float, float, float] = (0.0, 0.0, 0.0),
    apple_v: tuple[float, float, float] = (0.0, 0.0, 0.0),
    apple_xyz: tuple[float, float, float] = (0.0, 0.0, 0.5),
    harvest_f: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict:
    return {
        "t_s": t_s,
        "env": env,
        "pos_err_m": pos_err_m,
        "tcp_vx": tcp_v[0],
        "tcp_vy": tcp_v[1],
        "tcp_vz": tcp_v[2],
        "apple_x": apple_xyz[0],
        "apple_y": apple_xyz[1],
        "apple_z": apple_xyz[2],
        "apple_vx": apple_v[0],
        "apple_vy": apple_v[1],
        "apple_vz": apple_v[2],
        "harvest_fx": harvest_f[0],
        "harvest_fy": harvest_f[1],
        "harvest_fz": harvest_f[2],
    }


def test_parse_float_list():
    assert parse_float_list("0.3,0.5,1.0") == pytest.approx([0.3, 0.5, 1.0])
    assert parse_float_list("600") == [600.0]


def test_stable_hold_passes():
    rows = [_row(t_s=t, env=0) for t in (0.0, 0.5, 1.0, 5.0)]
    m = compute_env_stability_metrics(rows, env=0, duration_max=5.0)
    assert m.is_finite
    assert m.is_stable
    assert m.max_apple_drift_m == pytest.approx(0.0)
    assert m.max_pos_err_m == pytest.approx(0.01)


def test_apple_small_oscillation_passes():
    rows = [
        _row(t_s=0.0, env=0, apple_xyz=(0.0, 0.0, 0.5)),
        _row(t_s=1.0, env=0, apple_xyz=(0.005, 0.0, 0.5)),
        _row(t_s=2.0, env=0, apple_xyz=(0.0, 0.0, 0.5)),
        _row(t_s=5.0, env=0, apple_xyz=(-0.005, 0.0, 0.5)),
    ]
    m = compute_env_stability_metrics(rows, env=0, duration_max=5.0)
    assert m.is_stable
    assert m.max_apple_drift_m == pytest.approx(0.005)
    assert m.apple_path_length_m == pytest.approx(0.015)


def test_apple_drift_fails():
    rows = [
        _row(t_s=0.0, env=0, apple_xyz=(0.0, 0.0, 0.5)),
        _row(t_s=5.0, env=0, apple_xyz=(0.05, 0.0, 0.5)),
    ]
    m = compute_env_stability_metrics(rows, env=0, duration_max=5.0)
    assert not m.is_stable
    assert "apple_drift" in m.issues


def test_apple_wander_fails():
    rows = [
        _row(t_s=0.0, env=0, apple_xyz=(0.0, 0.0, 0.5)),
        _row(t_s=1.0, env=0, apple_xyz=(0.02, 0.0, 0.5)),
        _row(t_s=2.0, env=0, apple_xyz=(0.0, 0.02, 0.5)),
        _row(t_s=3.0, env=0, apple_xyz=(-0.02, 0.0, 0.5)),
        _row(t_s=4.0, env=0, apple_xyz=(0.0, -0.02, 0.5)),
        _row(t_s=5.0, env=0, apple_xyz=(0.0, 0.0, 0.5)),
    ]
    m = compute_env_stability_metrics(rows, env=0, duration_max=5.0)
    assert not m.is_stable
    assert "apple_wander" in m.issues


def test_apple_sag_fails():
    rows = [
        _row(t_s=0.0, env=0, apple_xyz=(0.0, 0.0, 0.5)),
        _row(t_s=5.0, env=0, apple_xyz=(0.0, 0.0, 0.47)),
    ]
    m = compute_env_stability_metrics(rows, env=0, duration_max=5.0)
    assert not m.is_stable
    assert "apple_sag" in m.issues


def test_diverging_pos_err_fails():
    rows = [_row(t_s=t, env=0, pos_err_m=0.01 + 0.02 * t) for t in (0.0, 2.0, 5.0)]
    m = compute_env_stability_metrics(rows, env=0, duration_max=5.0)
    assert not m.is_stable
    assert "pos_err" in m.issues


def test_high_tcp_speed_fails():
    rows = [_row(t_s=0.0, env=0, tcp_v=(0.2, 0.0, 0.0))]
    m = compute_env_stability_metrics(rows, env=0, duration_max=5.0)
    assert not m.is_stable
    assert "tcp_speed" in m.issues


def test_harvest_spike_fails():
    rows = [_row(t_s=0.0, env=0, harvest_f=(250.0, 0.0, 0.0))]
    m = compute_env_stability_metrics(rows, env=0, duration_max=5.0)
    assert not m.is_stable
    assert "harvest_force" in m.issues


def test_apple_below_floor_fails():
    rows = [_row(t_s=0.0, env=0, apple_xyz=(0.0, 0.0, -0.01))]
    m = compute_env_stability_metrics(rows, env=0, duration_max=5.0)
    assert not m.is_stable
    assert "apple_floor" in m.issues


def test_nan_row_fails_finite():
    rows = [_row(t_s=0.0, env=0)]
    rows[0]["pos_err_m"] = float("nan")
    m = compute_env_stability_metrics(rows, env=0, duration_max=5.0)
    assert not m.is_finite
    assert not m.is_stable


def test_summarize_hold_metrics_pass_rate():
    metrics = [
        compute_env_stability_metrics([_row(t_s=0.0, env=0)], env=0, duration_max=5.0),
        compute_env_stability_metrics(
            [_row(t_s=0.0, env=1, pos_err_m=1.0)], env=1, duration_max=5.0
        ),
    ]
    summary = summarize_hold_metrics(metrics, settle_stable=(True, False), ik_inside=(True, True))
    assert summary.vic_pass_rate == pytest.approx(0.5)
    assert summary.settle_pass_rate == pytest.approx(0.5)
    assert summary.ik_pass_rate == pytest.approx(1.0)


@pytest.mark.slow
def test_run_zero_vic_hold_smoke():
    pytest.importorskip("torch")
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("FR3 assets not available")

    from apple_pick_sim.diagnostics.log_zero_vic_poses import ZeroVicHoldConfig, run_zero_vic_hold

    config = ZeroVicHoldConfig(
        num_envs=2,
        duration=0.5,
        log_interval=0.25,
        settle_substeps=100,
        write_trajectory=True,
        print_settle_report=False,
        print_vic_summary=False,
    )
    result = run_zero_vic_hold(config)
    assert len(result.per_env_metrics) == 2
    assert len(result.time_series) >= 2
    assert all(math.isfinite(m.max_apple_drift_m) for m in result.per_env_metrics)
    assert result.summary.num_envs == 2
