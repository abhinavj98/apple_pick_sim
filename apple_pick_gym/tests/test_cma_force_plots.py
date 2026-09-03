"""Unit tests for CMA real-vs-sim force plot generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apple_pick_sim.system_id.trajectory_store import PHASE_TO_INT

_HOLD = int(PHASE_TO_INT["hold"])
_MOVE = int(PHASE_TO_INT["move_out"])


_STATE_DIM_WRENCH = 6
_STATE_DIM_FULL = 26


def _write_role_bag(
    role_dir: Path,
    *,
    n_dirs: int = 2,
    n: int = 12,
    direction_indices: list[int] | None = None,
    state_dim: int = _STATE_DIM_WRENCH,
) -> None:
    role_dir.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    phase = np.full(n, _MOVE, dtype=np.int8)
    phase[n // 3 : 2 * n // 3] = _HOLD
    force_norms: dict[str, dict[str, float]] = {}
    torque_norms: dict[str, dict[str, float]] = {}
    dirs = list(direction_indices) if direction_indices is not None else list(range(n_dirs))
    for d in dirs:
        rs = np.zeros((n, state_dim), dtype=np.float64)
        ss = np.zeros((n, state_dim), dtype=np.float64)
        rs[:, 0] = 4.0 + 0.1 * d
        ss[:, 0] = 2.0 + 0.1 * d
        rs[:, 3] = 0.4 + 0.01 * d
        ss[:, 3] = 0.2 + 0.01 * d
        if state_dim >= _STATE_DIM_FULL:
            # TCP pos @ 12:15, woody primary_spur @ 18:21, spur_stem @ 21:24
            rs[:, 12:15] = 0.1 + 0.01 * d
            ss[:, 12:15] = 0.11 + 0.01 * d
            rs[:, 12] += np.linspace(0.0, 0.02, n)
            ss[:, 12] += np.linspace(0.0, 0.015, n)
            rs[:, 18:21] = 0.2 + 0.01 * d
            ss[:, 18:21] = 0.21 + 0.01 * d
            rs[:, 21:24] = 0.3 + 0.01 * d
            ss[:, 21:24] = 0.31 + 0.01 * d
        np.savez(
            role_dir / f"dir_{d:02d}.npz",
            sim_time_real=t,
            sim_time_sim=t,
            real_state=rs,
            sim_state=ss,
            phase_real=phase,
        )
        force_norms[str(d)] = {"real": float(rs[0, 0]), "sim": float(ss[0, 0])}
        torque_norms[str(d)] = {"real": float(rs[0, 3]), "sim": float(ss[0, 3])}
    (role_dir / "metadata.json").write_text(
        json.dumps(
            {
                "aggregate_sinkhorn": 12.5,
                "candidate_index": 3,
                "per_direction_mean_hold_force_norm_n": force_norms,
                "per_direction_mean_hold_torque_norm_nm": torque_norms,
            }
        ),
        encoding="utf-8",
    )


def test_list_persisted_generations_is_sorted(tmp_path: Path):
    from apple_pick_gym.viz.cma_force_plots import list_persisted_generations

    struct = tmp_path / "structure_000"
    _write_role_bag(struct / "generations" / "gen_02" / "best")
    _write_role_bag(struct / "generations" / "gen_00" / "best")
    assert list_persisted_generations(struct) == [0, 2]


def test_write_run_force_plots_writes_png_and_index(tmp_path: Path):
    from apple_pick_gym.viz.cma_force_plots import write_run_force_plots

    run = tmp_path / "cma_run"
    struct = run / "structure_000"
    _write_role_bag(struct / "generations" / "gen_00" / "best")
    out = write_run_force_plots(run, n_directions=2, write_html=False)
    assert out == run / "structure_000" / "force_plots"
    assert (out / "README.md").is_file()
    gen_dir = out / "gen_00"
    assert (gen_dir / "all_directions_force.png").is_file()
    assert (gen_dir / "dir_00_force.png").is_file()
    assert (gen_dir / "dir_01_force.png").is_file()
    index = (out / "README.md").read_text(encoding="utf-8")
    assert "gen 00" in index
    assert "12.5" in index


def test_write_run_force_plots_writes_torque_png(tmp_path: Path):
    from apple_pick_gym.viz.cma_force_plots import write_run_force_plots

    run = tmp_path / "cma_run"
    struct = run / "structure_000"
    _write_role_bag(struct / "generations" / "gen_00" / "best")
    out = write_run_force_plots(run, n_directions=2, write_html=False)
    gen_dir = out / "gen_00"
    assert (gen_dir / "all_directions_torque.png").is_file()
    assert (gen_dir / "dir_00_torque.png").is_file()
    assert (gen_dir / "dir_01_torque.png").is_file()
    gen_readme = (gen_dir / "README.md").read_text(encoding="utf-8")
    assert "dir_XX_torque.png" in gen_readme or "torque.png" in gen_readme
    index = (out / "README.md").read_text(encoding="utf-8")
    assert "all_directions_torque.png" in index


def test_lowpass_sim_force_preserves_dc_and_kills_high_frequency():
    from apple_pick_gym.viz.cma_force_plots import lowpass_sim_force

    fs = 30.0
    t = np.arange(0.0, 2.0, 1.0 / fs, dtype=np.float64)
    dc = 4.0
    x = np.zeros((t.size, 3), dtype=np.float64)
    x[:, 0] = dc + 2.0 * np.sin(2.0 * np.pi * 10.0 * t)
    y = lowpass_sim_force(x, t, cutoff_hz=5.0)
    mid = y[20:-20, 0]
    assert float(np.mean(mid)) == pytest.approx(dc, abs=0.2)
    assert float(np.std(mid)) < 0.5
    np.testing.assert_allclose(y[:, 1], 0.0, atol=1e-12)


def test_lowpass_sim_force_cutoff_none_is_identity():
    from apple_pick_gym.viz.cma_force_plots import lowpass_sim_force

    t = np.linspace(0.0, 1.0, 30, dtype=np.float64)
    x = np.random.default_rng(0).normal(size=(t.size, 3))
    y = lowpass_sim_force(x, t, cutoff_hz=None)
    np.testing.assert_array_equal(y, x)


def test_pull_directions_from_manifest(tmp_path: Path):
    from apple_pick_gym.viz.cma_force_plots import pull_directions_from_manifest

    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "episodes": [
                    {"direction_idx": 1, "pull_direction": [0.0, 1.0, 0.0]},
                    {"direction_idx": 0, "pull_direction": [1.0, 0.0, 0.0]},
                ]
            }
        ),
        encoding="utf-8",
    )
    pull = pull_directions_from_manifest(path)
    assert pull[0] == pytest.approx((1.0, 0.0, 0.0))
    assert pull[1] == pytest.approx((0.0, 1.0, 0.0))


def test_list_direction_indices_sorted_from_npz_names(tmp_path: Path):
    from apple_pick_gym.viz.cma_force_plots import list_direction_indices

    role = tmp_path / "best"
    _write_role_bag(role, direction_indices=[1, 4, 7])
    assert list_direction_indices(role) == [1, 4, 7]


def test_write_run_force_plots_handles_noncontiguous_direction_indices(tmp_path: Path):
    from apple_pick_gym.viz.cma_force_plots import write_run_force_plots

    run = tmp_path / "cma_holdout_run"
    struct = run / "structure_000"
    _write_role_bag(
        struct / "generations" / "gen_00" / "best",
        direction_indices=[1, 2, 4],
    )
    out = write_run_force_plots(run, write_html=False)
    gen_dir = out / "gen_00"
    assert (gen_dir / "dir_01_force.png").is_file()
    assert (gen_dir / "dir_02_force.png").is_file()
    assert (gen_dir / "dir_04_force.png").is_file()
    assert not (gen_dir / "dir_00_force.png").is_file()
    assert (gen_dir / "all_directions_force.png").is_file()
    assert (gen_dir / "dir_01_torque.png").is_file()
    assert (gen_dir / "dir_04_torque.png").is_file()


def _write_holdout_npz_only(
    role_dir: Path,
    *,
    direction_indices: list[int],
    state_dim: int = _STATE_DIM_WRENCH,
) -> None:
    """Holdout persist writes dir_XX.npz without metadata.json."""
    role_dir.mkdir(parents=True, exist_ok=True)
    n = 12
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    phase = np.full(n, _MOVE, dtype=np.int8)
    phase[n // 3 : 2 * n // 3] = _HOLD
    for d in direction_indices:
        rs = np.zeros((n, state_dim), dtype=np.float64)
        ss = np.zeros((n, state_dim), dtype=np.float64)
        rs[:, 0] = 3.0 + 0.1 * d
        ss[:, 0] = 1.5 + 0.1 * d
        rs[:, 3] = 0.3
        ss[:, 3] = 0.15
        if state_dim >= _STATE_DIM_FULL:
            rs[:, 12:15] = 0.1
            ss[:, 12:15] = 0.12
            rs[:, 18:21] = 0.2
            ss[:, 18:21] = 0.22
            rs[:, 21:24] = 0.3
            ss[:, 21:24] = 0.32
        np.savez(
            role_dir / f"dir_{d:02d}.npz",
            sim_time_real=t,
            sim_time_sim=t,
            real_state=rs,
            sim_state=ss,
            phase_real=phase,
        )


def test_write_holdout_force_plots_writes_fitted_and_baseline(tmp_path: Path):
    from apple_pick_gym.viz.cma_force_plots import write_holdout_force_plots

    run = tmp_path / "cma_s03_val03_seed57"
    holdout = run / "structure_000" / "holdout"
    _write_holdout_npz_only(holdout / "fitted", direction_indices=[0, 3])
    _write_holdout_npz_only(holdout / "baseline", direction_indices=[0, 3])
    out = write_holdout_force_plots(run, write_html=False)
    assert out == run / "structure_000" / "force_plots" / "holdout"
    assert (out / "fitted" / "dir_00_force.png").is_file()
    assert (out / "fitted" / "dir_03_force.png").is_file()
    assert (out / "fitted" / "all_directions_force.png").is_file()
    assert (out / "fitted" / "dir_00_torque.png").is_file()
    assert (out / "baseline" / "dir_03_force.png").is_file()
    assert (out / "README.md").is_file()
    index = (out / "README.md").read_text(encoding="utf-8")
    assert "fitted" in index
    assert "baseline" in index


def test_write_run_force_plots_skips_tcp_woody_for_wrench_only_bags(tmp_path: Path):
    from apple_pick_gym.viz.cma_force_plots import write_run_force_plots

    run = tmp_path / "cma_run"
    struct = run / "structure_000"
    _write_role_bag(struct / "generations" / "gen_00" / "best", state_dim=6)
    out = write_run_force_plots(run, n_directions=2, write_html=False)
    gen_dir = out / "gen_00"
    assert (gen_dir / "dir_00_force.png").is_file()
    assert (gen_dir / "dir_00_torque.png").is_file()
    assert not (gen_dir / "dir_00_tcp.png").is_file()
    assert not (gen_dir / "dir_00_woody_primary_spur.png").is_file()
    assert not (gen_dir / "dir_00_woody_spur_stem.png").is_file()
    assert not (gen_dir / "all_directions_tcp.png").is_file()
    assert not list(gen_dir.glob("*woody*"))


def test_write_run_force_plots_writes_tcp_and_woody_png(tmp_path: Path):
    from apple_pick_gym.viz.cma_force_plots import write_run_force_plots

    run = tmp_path / "cma_run_full"
    struct = run / "structure_000"
    _write_role_bag(
        struct / "generations" / "gen_00" / "best",
        state_dim=_STATE_DIM_FULL,
    )
    out = write_run_force_plots(run, n_directions=2, write_html=False)
    gen_dir = out / "gen_00"
    assert (gen_dir / "dir_00_tcp.png").is_file()
    assert (gen_dir / "dir_01_tcp.png").is_file()
    assert (gen_dir / "all_directions_tcp.png").is_file()
    assert (gen_dir / "dir_00_woody_primary_spur.png").is_file()
    assert (gen_dir / "dir_00_woody_spur_stem.png").is_file()
    assert (gen_dir / "all_directions_woody_primary_spur.png").is_file()
    assert (gen_dir / "all_directions_woody_spur_stem.png").is_file()
    gen_readme = (gen_dir / "README.md").read_text(encoding="utf-8")
    assert "all_directions_tcp.png" in gen_readme
    assert "woody_primary_spur" in gen_readme
    assert "woody_spur_stem" in gen_readme


def test_write_holdout_force_plots_writes_tcp_and_woody(tmp_path: Path):
    from apple_pick_gym.viz.cma_force_plots import write_holdout_force_plots

    run = tmp_path / "cma_holdout_full"
    holdout = run / "structure_000" / "holdout"
    _write_holdout_npz_only(
        holdout / "fitted", direction_indices=[0, 3], state_dim=_STATE_DIM_FULL
    )
    _write_holdout_npz_only(
        holdout / "baseline", direction_indices=[0, 3], state_dim=_STATE_DIM_FULL
    )
    out = write_holdout_force_plots(run, write_html=False)
    assert (out / "fitted" / "dir_00_tcp.png").is_file()
    assert (out / "fitted" / "dir_00_woody_primary_spur.png").is_file()
    assert (out / "fitted" / "dir_03_woody_spur_stem.png").is_file()
    assert (out / "baseline" / "all_directions_tcp.png").is_file()


def test_load_dir_arrays_lpf_only_wrench_columns(tmp_path: Path):
    from apple_pick_gym.viz.cma_force_plots import _load_dir_arrays

    role = tmp_path / "best"
    role.mkdir(parents=True)
    fs = 30.0
    t = np.arange(0.0, 2.0, 1.0 / fs, dtype=np.float64)
    n = t.size
    ss = np.zeros((n, _STATE_DIM_FULL), dtype=np.float64)
    ss[:, 0] = 4.0 + 2.0 * np.sin(2.0 * np.pi * 10.0 * t)
    ss[:, 12] = 0.5 + 0.1 * np.sin(2.0 * np.pi * 10.0 * t)
    ss[:, 18] = 0.2 + 0.05 * np.sin(2.0 * np.pi * 10.0 * t)
    ss[:, 21] = 0.3 + 0.05 * np.sin(2.0 * np.pi * 10.0 * t)
    rs = ss.copy()
    phase = np.full(n, _HOLD, dtype=np.int8)
    np.savez(
        role / "dir_00.npz",
        sim_time_real=t,
        sim_time_sim=t,
        real_state=rs,
        sim_state=ss,
        phase_real=phase,
    )
    _tr, _ts, _rs, ss_out, _phase = _load_dir_arrays(role, 0, sim_lpf_hz=5.0)
    np.testing.assert_allclose(ss_out[:, 6:], ss[:, 6:], atol=1e-12)
    mid = ss_out[20:-20, 0]
    assert float(np.std(mid)) < float(np.std(ss[20:-20, 0]))
