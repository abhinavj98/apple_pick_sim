"""Tests for pairwise Bhattacharyya coefficient of CMA final Gaussians."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apple_pick_sim.system_id.cma_bhattacharyya import (
    DEFAULT_PHENOTYPE_DIM_NAMES,
    bhattacharyya_coefficient_gaussians,
    load_final_gaussian_from_cmaes_report,
    pairwise_bhattacharyya_coefficients,
    pairwise_bhattacharyya_for_structure_runs,
    resolve_phenotype_dim_indices,
    select_phenotype_dims,
)


def test_resolve_phenotype_dims_excludes_e_youngs() -> None:
    idxs = resolve_phenotype_dim_indices(exclude=("E_youngs_spur", "E_youngs_stem"))
    assert idxs == (0, 1, 2, 5)
    assert tuple(DEFAULT_PHENOTYPE_DIM_NAMES[i] for i in idxs) == (
        "support_kp",
        "E_flex_spur",
        "E_flex_stem",
        "support_roll_kp",
    )


def test_select_phenotype_dims_slices_mean_and_cov() -> None:
    mean = np.arange(6, dtype=np.float64)
    cov = np.arange(36, dtype=np.float64).reshape(6, 6)
    m2, c2 = select_phenotype_dims(mean, cov, dim_indices=(0, 1, 2))
    np.testing.assert_array_equal(m2, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(c2, cov[np.ix_([0, 1, 2], [0, 1, 2])])


def test_pairwise_without_e_youngs_ignores_youngs_shift(tmp_path: Path) -> None:
    for seed, youngs_shift in [(56, 0.0), (57, 5.0), (58, 10.0)]:
        run = tmp_path / f"cma_s02_val03_seed{seed}"
        run.mkdir()
        # Only E_youngs dims differ across seeds.
        mean = [2.3, 8.0, 8.1, 9.0 + youngs_shift, 8.0 + youngs_shift]
        (run / "cmaes_report.json").write_text(
            json.dumps(
                {
                    "structures": {
                        "0": {
                            "final_mean": {"log10_e": mean},
                            "covariance": {
                                "effective_unbounded_covariance": np.eye(5).tolist()
                            },
                        }
                    }
                }
            )
        )
    full = pairwise_bhattacharyya_for_structure_runs(
        tmp_path, structure=2, seeds=(56, 57, 58)
    )
    no_youngs = pairwise_bhattacharyya_for_structure_runs(
        tmp_path,
        structure=2,
        seeds=(56, 57, 58),
        dim_indices=(0, 1, 2),
    )
    assert min(p.coefficient for p in full.pairs) < 1e-6
    assert all(p.coefficient == pytest.approx(1.0) for p in no_youngs.pairs)


def test_identical_gaussians_have_coefficient_one() -> None:
    mean = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    cov = np.eye(3, dtype=np.float64)
    assert bhattacharyya_coefficient_gaussians(mean, cov, mean, cov) == pytest.approx(1.0)


def test_well_separated_isotropic_gaussians_have_small_coefficient() -> None:
    cov = np.eye(2, dtype=np.float64)
    mean_a = np.array([0.0, 0.0], dtype=np.float64)
    mean_b = np.array([10.0, 0.0], dtype=np.float64)
    bc = bhattacharyya_coefficient_gaussians(mean_a, cov, mean_b, cov)
    # DB = (1/8)*100 + 0 = 12.5 → BC = exp(-12.5)
    assert bc == pytest.approx(float(np.exp(-12.5)), rel=1e-12)
    assert bc < 1e-5


def test_bhattacharyya_is_symmetric() -> None:
    mean_a = np.array([0.0, 1.0], dtype=np.float64)
    mean_b = np.array([2.0, -1.0], dtype=np.float64)
    cov_a = np.array([[1.0, 0.2], [0.2, 1.5]], dtype=np.float64)
    cov_b = np.array([[2.0, -0.1], [-0.1, 0.8]], dtype=np.float64)
    ab = bhattacharyya_coefficient_gaussians(mean_a, cov_a, mean_b, cov_b)
    ba = bhattacharyya_coefficient_gaussians(mean_b, cov_b, mean_a, cov_a)
    assert ab == pytest.approx(ba, rel=1e-12)


def test_pairwise_returns_upper_triangle_pairs() -> None:
    means = [
        np.zeros(2, dtype=np.float64),
        np.array([1.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
    ]
    covs = [np.eye(2, dtype=np.float64)] * 3
    labels = ["a", "b", "c"]
    pairs = pairwise_bhattacharyya_coefficients(means, covs, labels=labels)
    assert [(p.label_i, p.label_j) for p in pairs] == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]
    assert all(0.0 < p.coefficient <= 1.0 for p in pairs)


def test_load_final_gaussian_from_report(tmp_path: Path) -> None:
    report = {
        "structures": {
            "0": {
                "final_mean": {"log10_e": [2.3, 8.0, 8.1, 9.0, 8.5]},
                "covariance": {
                    "effective_unbounded_covariance": [
                        [0.01, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.02, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.03, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.04, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.05],
                    ]
                },
            }
        }
    }
    path = tmp_path / "cmaes_report.json"
    path.write_text(json.dumps(report))
    gaussian = load_final_gaussian_from_cmaes_report(path)
    np.testing.assert_allclose(gaussian.mean_log10, [2.3, 8.0, 8.1, 9.0, 8.5])
    assert gaussian.covariance.shape == (5, 5)
    assert gaussian.covariance[1, 1] == pytest.approx(0.02)


def test_pairwise_for_structure_runs_uses_report_dirs(tmp_path: Path) -> None:
    for seed, shift in [(56, 0.0), (57, 0.5), (58, 1.0)]:
        run = tmp_path / f"cma_s02_val03_seed{seed}"
        run.mkdir()
        mean = [2.3, 8.0 + shift, 8.0, 9.0, 8.0]
        cov = np.eye(5, dtype=np.float64).tolist()
        (run / "cmaes_report.json").write_text(
            json.dumps(
                {
                    "structures": {
                        "0": {
                            "final_mean": {"log10_e": mean},
                            "covariance": {"effective_unbounded_covariance": cov},
                        }
                    }
                }
            )
        )
    result = pairwise_bhattacharyya_for_structure_runs(
        tmp_path,
        structure=2,
        seeds=(56, 57, 58),
    )
    assert result.structure == 2
    assert len(result.pairs) == 3
    assert result.pairs[0].label_i == "seed56"
    assert result.pairs[0].label_j == "seed57"
    # farther seeds → smaller BC
    bc_56_57 = next(
        p.coefficient for p in result.pairs if p.label_i == "seed56" and p.label_j == "seed57"
    )
    bc_56_58 = next(
        p.coefficient for p in result.pairs if p.label_i == "seed56" and p.label_j == "seed58"
    )
    assert bc_56_58 < bc_56_57
