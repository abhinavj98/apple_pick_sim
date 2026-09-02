"""Unit tests for CMA real-vs-sim force plot generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apple_pick_sim.system_id.trajectory_store import PHASE_TO_INT

_HOLD = int(PHASE_TO_INT["hold"])
_MOVE = int(PHASE_TO_INT["move_out"])


def _write_role_bag(role_dir: Path, *, n_dirs: int = 2, n: int = 12) -> None:
    role_dir.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    phase = np.full(n, _MOVE, dtype=np.int8)
    phase[n // 3 : 2 * n // 3] = _HOLD
    force_norms: dict[str, dict[str, float]] = {}
    torque_norms: dict[str, dict[str, float]] = {}
    for d in range(n_dirs):
        rs = np.zeros((n, 6), dtype=np.float64)
        ss = np.zeros((n, 6), dtype=np.float64)
        rs[:, 0] = 4.0 + 0.1 * d
        ss[:, 0] = 2.0 + 0.1 * d
        rs[:, 3] = 0.4 + 0.01 * d
        ss[:, 3] = 0.2 + 0.01 * d
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
    assert "dir_00_torque.png" in gen_readme
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
