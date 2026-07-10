"""Result serialization and plotting for MMD grid diagnostics."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROD_SEGMENTS: tuple[str, ...] = ("primary", "secondary", "spur", "stem")


@dataclass(frozen=True)
class MmdCandidateResult:
    """MMD loss result for one stiffness-grid candidate."""

    candidate_index: int
    stiffnesses: dict[str, float]
    aggregate_mmd2: float
    per_direction_mmd2: dict[int, float]

    def __post_init__(self) -> None:
        if not self.per_direction_mmd2:
            raise ValueError("MMD candidate result must include at least one direction")


def rank_results(results: list[MmdCandidateResult]) -> list[MmdCandidateResult]:
    """Return results ordered by increasing aggregate MMD^2."""

    return sorted(results, key=lambda result: (result.aggregate_mmd2, result.candidate_index))


def _all_directions(results: list[MmdCandidateResult]) -> list[int]:
    return sorted({direction for result in results for direction in result.per_direction_mmd2.keys()})


def _ensure_output_dir(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_results_csv(
    results: list[MmdCandidateResult],
    output_dir: str | Path,
) -> Path:
    """Write ranked MMD grid results to ``mmd_results.csv``."""

    output = _ensure_output_dir(output_dir)
    path = output / "mmd_results.csv"
    directions = _all_directions(results)
    fieldnames = [
        "rank",
        "candidate_index",
        "primary_bend_stiffness",
        "secondary_bend_stiffness",
        "spur_bend_stiffness",
        "stem_bend_stiffness",
        "aggregate_mmd2",
        "n_directions",
    ] + [f"dir_idx_{int(direction)}_mmd2" for direction in directions]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(rank_results(results), start=1):
            row: dict[str, object] = {
                "rank": rank,
                "candidate_index": result.candidate_index,
                "primary_bend_stiffness": result.stiffnesses["primary"],
                "secondary_bend_stiffness": result.stiffnesses["secondary"],
                "spur_bend_stiffness": result.stiffnesses["spur"],
                "stem_bend_stiffness": result.stiffnesses["stem"],
                "aggregate_mmd2": result.aggregate_mmd2,
                "n_directions": len(result.per_direction_mmd2),
            }
            for direction in directions:
                col_name = f"dir_idx_{int(direction)}_mmd2"
                value = result.per_direction_mmd2.get(direction)
                row[col_name] = "" if value is None else value
            writer.writerow(row)
    return path


def write_ranked_loss_plot(
    results: list[MmdCandidateResult],
    output_dir: str | Path,
) -> Path:
    """Write a ranked aggregate MMD loss PNG."""

    output = _ensure_output_dir(output_dir)
    path = output / "mmd_ranked_loss.png"

    plt = _import_pyplot()

    ranked = rank_results(results)
    ranks = list(range(1, len(ranked) + 1))
    losses = [result.aggregate_mmd2 for result in ranked]
    labels = [str(result.candidate_index) for result in ranked]

    fig, ax = plt.subplots(figsize=(max(6.0, 0.5 * len(ranked)), 4.0))
    ax.plot(ranks, losses, marker="o")
    ax.set_xlabel("Rank (tick label = candidate index)")
    ax.set_ylabel("Aggregate biased MMD^2")
    ax.set_title("Ranked MMD grid candidates")
    ax.set_xticks(ranks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _direction_loss_matrix(
    results: list[MmdCandidateResult],
) -> tuple[list[MmdCandidateResult], list[tuple[float, float, float]], np.ndarray]:
    ranked = rank_results(results)
    directions = _all_directions(ranked)
    matrix = np.full((len(directions), len(ranked)), np.nan, dtype=np.float64)
    for col, result in enumerate(ranked):
        for row, direction in enumerate(directions):
            value = result.per_direction_mmd2.get(direction)
            if value is not None:
                matrix[row, col] = float(value)
    return ranked, directions, matrix


def write_direction_heatmap_plot(
    results: list[MmdCandidateResult],
    output_dir: str | Path,
) -> Path:
    """Write a per-direction MMD heatmap over ranked candidates."""

    output = _ensure_output_dir(output_dir)
    path = output / "mmd_direction_heatmap.png"
    plt = _import_pyplot()

    ranked, directions, matrix = _direction_loss_matrix(results)
    masked = np.ma.masked_invalid(matrix)
    width = max(6.0, 0.45 * max(1, len(ranked)))
    height = max(3.5, 0.4 * max(1, len(directions)))
    fig, ax = plt.subplots(figsize=(width, height))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")
    image = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_xlabel("Ranked candidates (candidate index)")
    ax.set_ylabel("Excitation direction")
    ax.set_title("Per-direction MMD^2 by ranked candidate")
    ax.set_xticks(range(len(ranked)))
    ax.set_xticklabels(
        [str(result.candidate_index) for result in ranked],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(directions)))
    ax.set_yticklabels([f"({direction[0]:+.3f},{direction[1]:+.3f},{direction[2]:+.3f})" for direction in directions])
    fig.colorbar(image, ax=ax, label="Biased MMD^2")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _group_losses_by_stiffness(
    results: list[MmdCandidateResult],
    segment: str,
) -> tuple[list[float], list[float], list[float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for result in results:
        grouped[float(result.stiffnesses[segment])].append(float(result.aggregate_mmd2))

    values = sorted(grouped)
    mins = [float(np.min(grouped[value])) for value in values]
    medians = [float(np.median(grouped[value])) for value in values]
    return values, mins, medians


def write_stiffness_sensitivity_plot(
    results: list[MmdCandidateResult],
    output_dir: str | Path,
) -> Path:
    """Write aggregate MMD summaries grouped by each bend-stiffness axis."""

    output = _ensure_output_dir(output_dir)
    path = output / "mmd_stiffness_sensitivity.png"
    plt = _import_pyplot()

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharey=True)
    for ax, segment in zip(axes.reshape(-1), ROD_SEGMENTS, strict=True):
        values, mins, medians = _group_losses_by_stiffness(results, segment)
        ax.plot(values, medians, marker="o", label="median")
        ax.scatter(values, mins, marker="v", label="min")
        ax.set_title(f"{segment}.bend_stiffness")
        ax.set_xlabel("Candidate value")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0, 0].set_ylabel("Aggregate biased MMD^2")
    axes[1, 0].set_ylabel("Aggregate biased MMD^2")
    axes[0, 0].legend(loc="best")
    fig.suptitle("Grid sensitivity by bend-stiffness axis")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_diagnostic_plots(
    results: list[MmdCandidateResult],
    output_dir: str | Path,
) -> list[Path]:
    """Write the compact local MMD diagnostic plot bundle."""

    return [
        write_ranked_loss_plot(results, output_dir),
        write_direction_heatmap_plot(results, output_dir),
        write_stiffness_sensitivity_plot(results, output_dir),
    ]
