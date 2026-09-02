"""Tests for sparse CMA generation trajectory persist."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apple_pick_gym.batched_envs.batched_sysid_cmaes import (
    CmaDistributionSnapshot,
    CmaGenerationRecord,
    YoungsModulusCandidate,
    YoungsModulusCandidateScore,
    YoungsModulusEvaluation,
)
from apple_pick_gym.cma_generation_persist import (
    SPARSE_ROLES,
    build_direction_state_npz,
    make_generation_features_figure,
    persist_cma_generation_wave,
    select_sparse_generation_roles,
)


def _score(
    idx: int,
    *,
    sinkhorn: float,
    disqualified: bool = False,
    force_err: float | None = 1.0,
    directions: tuple[int, ...] = (0, 1),
) -> YoungsModulusCandidateScore:
    return YoungsModulusCandidateScore(
        candidate_index=int(idx),
        candidate=YoungsModulusCandidate(
            primary=5e10,
            spur=1e9,
            stem=1e9,
        ),
        aggregate_sinkhorn=float(sinkhorn),
        per_direction_sinkhorn={int(d): float(sinkhorn) for d in directions},
        instability_fraction=0.0,
        disqualified=bool(disqualified),
        disqualification_reason="x" if disqualified else None,
        rank=None,
        is_gt=False,
        mean_hold_force_err_n=force_err,
    )


def test_select_sparse_roles_best_mean_worst_force():
    scores = [
        _score(0, sinkhorn=100.0, force_err=5.0),
        _score(1, sinkhorn=50.0, force_err=2.0),
        _score(2, sinkhorn=200.0, force_err=8.0),
    ]
    fitness = [110.0, 55.0, 210.0]
    ask = [(4.0, 9.0, 9.0), (4.1, 9.05, 9.05), (3.9, 8.9, 8.9)]
    ask_mean = (4.05, 9.02, 9.02)
    roles = select_sparse_generation_roles(
        scores=scores,
        penalized_fitness=fitness,
        ask_samples_log10=ask,
        ask_mean_log10=ask_mean,
    )
    assert roles["best"] == 1
    assert roles["mean"] == 0
    assert roles["worst_force"] == 2


def test_select_sparse_roles_skips_disqualified_and_missing_force():
    scores = [
        _score(0, sinkhorn=100.0, disqualified=True, force_err=99.0),
        _score(1, sinkhorn=50.0, force_err=None),
        _score(2, sinkhorn=60.0, force_err=3.0),
    ]
    fitness = [999.0, 55.0, 65.0]
    ask = [(4.0, 9.0, 9.0), (4.1, 9.0, 9.0), (4.2, 9.0, 9.0)]
    roles = select_sparse_generation_roles(
        scores=scores,
        penalized_fitness=fitness,
        ask_samples_log10=ask,
        ask_mean_log10=(4.1, 9.0, 9.0),
    )
    assert roles["best"] == 1
    assert roles["worst_force"] == 2
    assert 0 not in (roles["best"], roles["mean"], roles["worst_force"])


def _episode(*, n: int = 5, direction: int = 0, ft_scale: float = 1.0) -> dict:
    n_j = 2
    junction_names = ["primary_spur", "spur_stem"]
    t = np.arange(n, dtype=np.float32)
    woody = {
        name: np.tile(np.array([0.1 * (i + 1), 0.0, 0.5], dtype=np.float32), (n, 1))
        for i, name in enumerate(junction_names)
    }
    ft = np.zeros((n, 6), dtype=np.float32)
    ft[:, 0] = ft_scale
    return {
        "action": np.zeros((n, 19), dtype=np.float32),
        "ft_wrist": ft,
        "ft_wrist_lpf": ft * 0.9,
        "tcp_velocity": np.zeros((n, 6), dtype=np.float32),
        "tcp_pos": np.tile(np.array([0.0, 0.0, 0.5], dtype=np.float32), (n, 1)),
        "tcp_quat": np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1)),
        "apple_pos": np.tile(np.array([0.0, 0.0, 0.6], dtype=np.float32), (n, 1)),
        "woody_part_start_pos": woody,
        "excitation_direction": np.zeros((n, 3), dtype=np.float32),
        "phase": np.ones(n, dtype=np.int8),
        "excitation_type": np.zeros(n, dtype=np.int8),
        "dir_idx": np.full(n, int(direction), dtype=np.int32),
        "junction_names": junction_names,
        "step_idx": np.arange(n, dtype=np.int32),
        "sim_time": t.astype(np.float64),
        "stable": np.ones(n, dtype=bool),
    }


def test_build_direction_state_npz_uses_lpf_and_state_width():
    real = _episode(n=4, ft_scale=2.0)
    sim = _episode(n=3, ft_scale=1.0)
    arrays = build_direction_state_npz(real=real, sim=sim)
    assert arrays["real_state"].shape == (4, 26)
    assert arrays["sim_state"].shape == (3, 26)
    assert arrays["sim_time_real"].shape == (4,)
    assert arrays["sim_time_sim"].shape == (3,)
    assert float(arrays["real_state"][0, 0]) == pytest.approx(2.0 * 0.9)


def test_build_direction_state_npz_aligns_sim_time_to_real_when_missing():
    real = _episode(n=4, ft_scale=2.0)
    real["sim_time"] = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    sim = _episode(n=4, ft_scale=1.0)
    sim.pop("sim_time")
    arrays = build_direction_state_npz(real=real, sim=sim)
    np.testing.assert_allclose(arrays["sim_time_real"], real["sim_time"])
    np.testing.assert_allclose(arrays["sim_time_sim"], real["sim_time"])


def test_make_generation_features_figure_trace_counts():
    real = _episode(n=3, direction=2)
    sim = _episode(n=3, direction=2)
    arrays = build_direction_state_npz(real=real, sim=sim)
    per_dir = {
        2: {
            **arrays,
            "block_errors": {
                "force_err_n": 0.5,
                "torque_err_nm": 0.1,
                "woody_start_m": 0.01,
                "woody_bend_rad": 0.02,
            },
            "sinkhorn": 123.0,
            "force_ratio": 0.9,
        }
    }
    fig = make_generation_features_figure(
        per_direction=per_dir,
        direction_indices=(2,),
        title="test",
    )
    scatter = [t for t in fig.data if getattr(t, "type", None) == "scatter"]
    # 26 STATE_VECTOR channels × real + sim per direction.
    assert len(scatter) == 2 * 26


def test_persist_cma_generation_wave_writes_roles_and_npz(tmp_path: Path):
    recorded = [_episode(n=4, direction=0), _episode(n=4, direction=1)]
    replay0 = [_episode(n=4, direction=0), _episode(n=4, direction=1)]
    replay1 = [_episode(n=4, direction=0, ft_scale=0.5), _episode(n=4, direction=1)]
    scores = [
        _score(0, sinkhorn=100.0, force_err=5.0),
        _score(1, sinkhorn=50.0, force_err=2.0),
    ]
    record = CmaGenerationRecord(
        generation_index=0,
        structure_idx=0,
        ask_samples_log10=((4.0, 9.0, 9.0), (4.1, 9.05, 9.05)),
        candidates=tuple(
            YoungsModulusCandidate(primary=5e10, spur=1e9, stem=1e9)
            for _ in range(2)
        ),
        raw_scores=tuple(scores),
        penalized_fitness=(110.0, 55.0),
        penalty_metadata=({}, {}),
        ask_distribution=CmaDistributionSnapshot(
            mean_log10=(4.05, 9.02, 9.02),
            sigma=0.2,
        ),
        post_tell_distribution=CmaDistributionSnapshot(
            mean_log10=(4.05, 9.02, 9.02),
            sigma=0.2,
        ),
    )
    evaluation = YoungsModulusEvaluation(
        structure_idx=0,
        gt_candidate=None,
        fixed_secondary_e_pa=None,
        direction_indices=(0, 1),
        scores=list(scores),
        replay_episodes=[replay0, replay1],
        applied_params=[],
    )
    summary = persist_cma_generation_wave(
        output_dir=tmp_path,
        structure_idx=0,
        record=record,
        evaluation=evaluation,
        recorded_episodes=recorded,
    )
    gen_dir = tmp_path / "structure_000" / "generations" / "gen_00"
    assert gen_dir.is_dir()
    roles = json.loads((gen_dir / "roles.json").read_text(encoding="utf-8"))
    assert set(roles["roles"]) >= {"best", "mean", "worst_force"}
    for role in SPARSE_ROLES:
        role_dir = gen_dir / role
        assert role_dir.is_dir(), role
        assert (role_dir / "features.html").is_file()
        assert (role_dir / "metadata.json").is_file()
        assert (role_dir / "dir_00.npz").is_file()
    assert summary["generation_index"] == 0


def test_persist_noop_when_disabled(tmp_path: Path):
    summary = persist_cma_generation_wave(
        output_dir=tmp_path,
        structure_idx=0,
        record=SimpleNamespace(generation_index=0),
        evaluation=SimpleNamespace(
            direction_indices=(0,),
            replay_episodes=[[]],
            scores=[],
        ),
        recorded_episodes=[],
        persist=False,
    )
    assert summary["skipped"] is True
    assert not (tmp_path / "structure_000").exists()
