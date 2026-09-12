"""Bhattacharyya coefficient between CMA final-generation Gaussians.

Uses the fitted phenotype mean ``final_mean.log10_e`` and the recorded
``covariance.effective_unbounded_covariance`` (σ²-scaled CMA shape in
optimizer / unbounded phenotype coordinates).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_RUN_DIR_RE = re.compile(
    r"^cma_s(?P<structure>\d+)_val\d+_seed(?P<seed>\d+)$"
)

# Matches CMA_SEARCH_PARAMS phenotype order in example_youngs_modulus_cmaes.py.
DEFAULT_PHENOTYPE_DIM_NAMES: tuple[str, ...] = (
    "support_kp",
    "E_flex_spur",
    "E_flex_stem",
    "E_youngs_spur",
    "E_youngs_stem",
    "support_roll_kp",
)


def resolve_phenotype_dim_indices(
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    dim_names: Sequence[str] = DEFAULT_PHENOTYPE_DIM_NAMES,
) -> tuple[int, ...]:
    """Resolve phenotype coordinate indices from include/exclude name lists."""
    names = tuple(dim_names)
    name_to_idx = {name: i for i, name in enumerate(names)}
    if include is not None and exclude is not None:
        raise ValueError("pass only one of include or exclude")
    if include is not None:
        missing = [n for n in include if n not in name_to_idx]
        if missing:
            raise KeyError(f"unknown phenotype dims: {missing}; known={list(names)}")
        return tuple(name_to_idx[n] for n in include)
    if exclude is not None:
        missing = [n for n in exclude if n not in name_to_idx]
        if missing:
            raise KeyError(f"unknown phenotype dims: {missing}; known={list(names)}")
        skip = {name_to_idx[n] for n in exclude}
        return tuple(i for i in range(len(names)) if i not in skip)
    return tuple(range(len(names)))


def select_phenotype_dims(
    mean_log10: np.ndarray,
    covariance: np.ndarray,
    *,
    dim_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Slice mean/covariance to the selected phenotype coordinates."""
    idxs = np.asarray(list(dim_indices), dtype=np.int64)
    if idxs.size == 0:
        raise ValueError("dim_indices must be non-empty")
    mean = np.asarray(mean_log10, dtype=np.float64).reshape(-1)
    cov = np.asarray(covariance, dtype=np.float64)
    if np.any(idxs < 0) or np.any(idxs >= mean.shape[0]):
        raise IndexError(
            f"dim_indices {dim_indices} out of range for mean dim {mean.shape[0]}"
        )
    return mean[idxs].copy(), cov[np.ix_(idxs, idxs)].copy()


@dataclass(frozen=True)
class CmaFinalGaussian:
    """Final CMA Gaussian in log10 phenotype coordinates."""

    mean_log10: np.ndarray
    covariance: np.ndarray
    source_path: Path | None = None


@dataclass(frozen=True)
class PairwiseBhattacharyya:
    """One undirected pair of labeled Gaussians."""

    label_i: str
    label_j: str
    coefficient: float


@dataclass(frozen=True)
class StructurePairwiseBhattacharyya:
    """Pairwise BC among seeds for one structure."""

    structure: int
    pairs: tuple[PairwiseBhattacharyya, ...]
    run_paths: tuple[Path, ...]


def bhattacharyya_coefficient_gaussians(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Bhattacharyya coefficient BC ∈ (0, 1] for two multivariate Gaussians.

    For ``N(μ₁, Σ₁)`` and ``N(μ₂, Σ₂)`` with ``Σ = (Σ₁ + Σ₂) / 2``::

        D_B = (1/8)(μ₁-μ₂)ᵀ Σ⁻¹ (μ₁-μ₂)
            + (1/2) ln(det(Σ) / √(det(Σ₁) det(Σ₂)))
        BC = exp(-D_B)
    """
    mu1 = np.asarray(mean_a, dtype=np.float64).reshape(-1)
    mu2 = np.asarray(mean_b, dtype=np.float64).reshape(-1)
    s1 = np.asarray(cov_a, dtype=np.float64)
    s2 = np.asarray(cov_b, dtype=np.float64)
    if mu1.shape != mu2.shape:
        raise ValueError(f"mean shapes differ: {mu1.shape} vs {mu2.shape}")
    dim = int(mu1.shape[0])
    if s1.shape != (dim, dim) or s2.shape != (dim, dim):
        raise ValueError(
            f"covariance shapes must be ({dim}, {dim}); got {s1.shape}, {s2.shape}"
        )

    sigma = 0.5 * (s1 + s2)
    # Stabilize SPD inverses / logs on near-singular CMA covariances.
    eye = np.eye(dim, dtype=np.float64)
    sigma = sigma + eps * eye
    s1 = s1 + eps * eye
    s2 = s2 + eps * eye

    delta = mu1 - mu2
    # Solve Σ x = δ instead of forming Σ⁻¹ explicitly.
    mahal = float(delta @ np.linalg.solve(sigma, delta))
    sign_s, logdet_s = np.linalg.slogdet(sigma)
    sign_1, logdet_1 = np.linalg.slogdet(s1)
    sign_2, logdet_2 = np.linalg.slogdet(s2)
    if sign_s <= 0 or sign_1 <= 0 or sign_2 <= 0:
        raise ValueError("covariances must be positive definite after eps jitter")

    db = 0.125 * mahal + 0.5 * (logdet_s - 0.5 * (logdet_1 + logdet_2))
    # Clamp tiny negative D_B from numerical noise when distributions match.
    return float(np.exp(-max(db, 0.0)))


def pairwise_bhattacharyya_coefficients(
    means: Sequence[np.ndarray],
    covs: Sequence[np.ndarray],
    *,
    labels: Sequence[str] | None = None,
) -> list[PairwiseBhattacharyya]:
    """Upper-triangle pairwise BC for an ordered list of Gaussians."""
    n = len(means)
    if n != len(covs):
        raise ValueError("means and covs must have the same length")
    if labels is None:
        labels = [str(i) for i in range(n)]
    if len(labels) != n:
        raise ValueError("labels must match means length")

    out: list[PairwiseBhattacharyya] = []
    for i in range(n):
        for j in range(i + 1, n):
            bc = bhattacharyya_coefficient_gaussians(
                means[i], covs[i], means[j], covs[j]
            )
            out.append(
                PairwiseBhattacharyya(
                    label_i=str(labels[i]),
                    label_j=str(labels[j]),
                    coefficient=bc,
                )
            )
    return out


def load_final_gaussian_from_cmaes_report(
    report_path: Path | str,
    *,
    structure_key: str = "0",
) -> CmaFinalGaussian:
    """Load final-mean Gaussian from a ``cmaes_report.json``."""
    path = Path(report_path)
    payload = json.loads(path.read_text())
    structures = payload.get("structures")
    if not isinstance(structures, dict) or structure_key not in structures:
        raise KeyError(f"{path}: missing structures[{structure_key!r}]")
    block = structures[structure_key]
    final_mean = block.get("final_mean") or {}
    log10 = final_mean.get("log10_e")
    if log10 is None:
        raise KeyError(f"{path}: missing final_mean.log10_e")
    cov_block = block.get("covariance") or {}
    cov = cov_block.get("effective_unbounded_covariance")
    if cov is None:
        raise KeyError(
            f"{path}: missing covariance.effective_unbounded_covariance"
        )
    mean_arr = np.asarray(log10, dtype=np.float64).reshape(-1)
    cov_arr = np.asarray(cov, dtype=np.float64)
    if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
        raise ValueError(f"{path}: covariance must be square; got {cov_arr.shape}")
    if cov_arr.shape[0] != mean_arr.shape[0]:
        raise ValueError(
            f"{path}: mean dim {mean_arr.shape[0]} != cov {cov_arr.shape}"
        )
    return CmaFinalGaussian(
        mean_log10=mean_arr,
        covariance=cov_arr,
        source_path=path,
    )


def discover_structure_run_dirs(
    root: Path | str,
    *,
    structure: int,
    seeds: Iterable[int] | None = None,
) -> list[tuple[int, Path]]:
    """Find ``cma_sXX_val*_seedYY`` dirs under ``root`` for one structure."""
    root_path = Path(root)
    seed_filter = None if seeds is None else {int(s) for s in seeds}
    found: list[tuple[int, Path]] = []
    for child in sorted(root_path.iterdir()):
        if not child.is_dir():
            continue
        match = _RUN_DIR_RE.match(child.name)
        if match is None:
            continue
        struct = int(match.group("structure"))
        seed = int(match.group("seed"))
        if struct != int(structure):
            continue
        if seed_filter is not None and seed not in seed_filter:
            continue
        report = child / "cmaes_report.json"
        if report.is_file():
            found.append((seed, child))
    return found


def pairwise_bhattacharyya_for_structure_runs(
    root: Path | str,
    *,
    structure: int,
    seeds: Sequence[int] | None = None,
    dim_indices: Sequence[int] | None = None,
) -> StructurePairwiseBhattacharyya:
    """Pairwise BC among final Gaussians for one structure's seed runs."""
    runs = discover_structure_run_dirs(root, structure=structure, seeds=seeds)
    if len(runs) < 2:
        raise FileNotFoundError(
            f"need ≥2 cmaes_report.json runs for structure {structure:02d} under {root}; "
            f"found {len(runs)}"
        )
    gaussians = [
        load_final_gaussian_from_cmaes_report(run_dir / "cmaes_report.json")
        for _, run_dir in runs
    ]
    means: list[np.ndarray] = []
    covs: list[np.ndarray] = []
    for g in gaussians:
        if dim_indices is None:
            means.append(g.mean_log10)
            covs.append(g.covariance)
        else:
            m, c = select_phenotype_dims(
                g.mean_log10, g.covariance, dim_indices=dim_indices
            )
            means.append(m)
            covs.append(c)
    labels = [f"seed{seed}" for seed, _ in runs]
    pairs = pairwise_bhattacharyya_coefficients(means, covs, labels=labels)
    return StructurePairwiseBhattacharyya(
        structure=int(structure),
        pairs=tuple(pairs),
        run_paths=tuple(run_dir for _, run_dir in runs),
    )


def pairwise_bhattacharyya_for_all_structures(
    root: Path | str,
    *,
    structures: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    dim_indices: Sequence[int] | None = None,
) -> list[StructurePairwiseBhattacharyya]:
    """Compute pairwise BC for each structure present under ``root``."""
    root_path = Path(root)
    if structures is None:
        found_structs: set[int] = set()
        for child in root_path.iterdir():
            if not child.is_dir():
                continue
            match = _RUN_DIR_RE.match(child.name)
            if match is not None:
                found_structs.add(int(match.group("structure")))
        structures = tuple(sorted(found_structs))
    return [
        pairwise_bhattacharyya_for_structure_runs(
            root_path, structure=s, seeds=seeds, dim_indices=dim_indices
        )
        for s in structures
    ]
