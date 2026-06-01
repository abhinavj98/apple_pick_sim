"""Toy 1D primary bend-stiffness recovery (FD gradient verification)."""

from __future__ import annotations

import numpy as np
import pytest

from apple_pick_sim import fruiting_system as fs
from apple_pick_sim.identification.theta_recovery import (
    DEFAULT_MIN_J_NORM,
    FeatureConfig,
    brute_force_grid_loss,
    compute_y_star,
    evaluate_at_k,
    primary_bend_bounds,
    recover_primary_bend_stiffness,
)
from apple_pick_sim.tests.conftest import COUPLED_BASE_POS, NO_SELF_COLLISION_KW, RANGES_FIXTURE

SEED = 7
EPS = 0.02
SUB_DT = 1.0 / 1800.0
INSTANCE_SPACING = (0.0, 1.5, 0.0)
N_SUBSTEPS_FREE = 120
N_SUBSTEPS_WELDED = 30
WARMUP_WELDED = 300
MAX_ITER = 15
K0_SCALE_FREE = 0.95
K0_SCALE_WELDED = 0.95


@pytest.fixture(scope="module", autouse=True)
def _warp_init():
    import warp as wp

    wp.init()


def _fd_kw() -> dict:
    return {
        "base_pos": COUPLED_BASE_POS,
        "instance_spacing": INSTANCE_SPACING,
        **NO_SELF_COLLISION_KW,
    }


def _sample_base_params():
    ranges = fs.load_ranges(RANGES_FIXTURE)
    return ranges, fs.sample_params(ranges, seed=SEED)


def _feature_cfg(fix_to_apple: bool) -> FeatureConfig:
    if fix_to_apple:
        return FeatureConfig.from_fix_to_apple(True, warmup_substeps=WARMUP_WELDED)
    return FeatureConfig.from_fix_to_apple(False)


def _run_recovery(fix_to_apple: bool):
    ranges, base_params = _sample_base_params()
    feature_cfg = _feature_cfg(fix_to_apple)
    n_substeps = N_SUBSTEPS_WELDED if fix_to_apple else N_SUBSTEPS_FREE
    k0_scale = K0_SCALE_WELDED if fix_to_apple else K0_SCALE_FREE
    return recover_primary_bend_stiffness(
        base_params,
        ranges,
        epsilon=EPS,
        n_substeps=n_substeps,
        dt=SUB_DT,
        k0_scale=k0_scale,
        max_iter=MAX_ITER,
        fd_kw=_fd_kw(),
        feature_cfg=feature_cfg,
    )


@pytest.mark.parametrize("fix_to_apple", [False, True])
def test_theta_recovery_loss_decreases(fix_to_apple: bool):
    result = _run_recovery(fix_to_apple)
    assert len(result.loss_hist) >= 1
    assert min(result.loss_hist) < result.loss_hist[0]


@pytest.mark.parametrize("fix_to_apple", [False, True])
def test_theta_recovery_converges_within_tolerance(fix_to_apple: bool):
    result = _run_recovery(fix_to_apple)
    assert result.rel_err <= 0.10


@pytest.mark.parametrize("fix_to_apple", [False, True])
def test_theta_recovery_brute_force_grid_agrees(fix_to_apple: bool):
    ranges, base_params = _sample_base_params()
    feature_cfg = _feature_cfg(fix_to_apple)
    n_substeps = N_SUBSTEPS_WELDED if fix_to_apple else N_SUBSTEPS_FREE
    assert base_params.primary is not None
    k_star = float(base_params.primary.bend_stiffness)
    y_star = compute_y_star(
        k_star,
        base_params,
        n_substeps=n_substeps,
        dt=SUB_DT,
        fd_kw=_fd_kw(),
        feature_cfg=feature_cfg,
    )
    k_min, k_max = primary_bend_bounds(ranges)
    k_grid = np.linspace(k_min, k_max, num=9)
    k_grid_best, _ = brute_force_grid_loss(
        base_params,
        y_star,
        k_star,
        k_grid=k_grid,
        epsilon=EPS,
        n_substeps=n_substeps,
        dt=SUB_DT,
        fd_kw=_fd_kw(),
        feature_cfg=feature_cfg,
    )
    result = recover_primary_bend_stiffness(
        base_params,
        ranges,
        k_star=k_star,
        epsilon=EPS,
        n_substeps=n_substeps,
        dt=SUB_DT,
        k0_scale=K0_SCALE_WELDED if fix_to_apple else K0_SCALE_FREE,
        max_iter=MAX_ITER,
        fd_kw=_fd_kw(),
        feature_cfg=feature_cfg,
    )
    assert abs(result.k_final - k_grid_best) / k_star <= 0.15


def test_theta_recovery_welded_jacobian_wrench_rows_nonzero():
    ranges, base_params = _sample_base_params()
    feature_cfg = _feature_cfg(True)
    assert base_params.primary is not None
    k0 = K0_SCALE_WELDED * float(base_params.primary.bend_stiffness)
    _, j_col, _, _ = evaluate_at_k(
        k0,
        base_params,
        EPS,
        n_substeps=N_SUBSTEPS_WELDED,
        dt=SUB_DT,
        fd_kw=_fd_kw(),
        feature_cfg=feature_cfg,
    )
    assert float(np.linalg.norm(j_col)) >= DEFAULT_MIN_J_NORM
