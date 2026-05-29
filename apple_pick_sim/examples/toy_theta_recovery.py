"""Toy 1D primary bend-stiffness recovery — FD gradient verification with plots.

Run from repository root::

    PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/toy_theta_recovery.py
    PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/toy_theta_recovery.py --fix-to-apple
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from apple_pick_sim import fruiting_system as fs
from apple_pick_sim.identification.theta_recovery import (
    FeatureConfig,
    brute_force_grid_loss,
    primary_bend_bounds,
    recover_primary_bend_stiffness,
)

DEFAULT_SEED = 7
DEFAULT_EPS = 0.02
DEFAULT_SUB_DT = 1.0 / 1800.0
DEFAULT_BASE_POS = (0.5, 0.5, 0.5)
DEFAULT_INSTANCE_SPACING = (0.0, 1.5, 0.0)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent.parent / "diagnostics" / "toy_theta_recovery"
)
N_SUBSTEPS_FREE = 120
N_SUBSTEPS_WELDED = 30
WARMUP_WELDED = 300
K0_SCALE_FREE = 0.95
K0_SCALE_WELDED = 1.05


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPS)
    parser.add_argument("--n-substeps", type=int, default=None)
    parser.add_argument("--warmup-substeps", type=int, default=WARMUP_WELDED)
    parser.add_argument("--max-iter", type=int, default=15)
    parser.add_argument("--k0-scale", type=float, default=None)
    parser.add_argument("--fix-to-apple", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--grid-points", type=int, default=21)
    return parser.parse_args()


def _fd_kw() -> dict:
    return {
        "base_pos": DEFAULT_BASE_POS,
        "instance_spacing": DEFAULT_INSTANCE_SPACING,
        "enable_self_collisions": False,
    }


def _plot_results(
    result,
    *,
    k_grid: np.ndarray,
    grid_losses: np.ndarray,
    output_dir: Path,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    iters = np.arange(len(result.loss_hist))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(iters, result.loss_hist, marker="o")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$||y(k)-y^*||$")
    ax.set_title("Loss vs iteration")
    fig.tight_layout()
    fig.savefig(output_dir / "loss_vs_iter.png", dpi=120)
    if show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(result.k_hist)), result.k_hist, marker="o", label="k")
    ax.axhline(result.k_star, color="C1", linestyle="--", label=r"$k^*$")
    ax.set_xlabel("iteration")
    ax.set_ylabel("primary.bend_stiffness")
    ax.set_title("Parameter vs iteration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "k_vs_iter.png", dpi=120)
    if show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_grid, grid_losses, label="brute-force loss")
    ax.axvline(result.k_star, color="C1", linestyle="--", label=r"$k^*$")
    ax.axvline(result.k_final, color="C2", linestyle=":", label="GN final")
    ax.set_xlabel("primary.bend_stiffness")
    ax.set_ylabel(r"$||y(k)-y^*||$")
    ax.set_title("Loss vs k (grid)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_vs_k_grid.png", dpi=120)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    import warp as wp

    wp.init()

    args = _parse_args()
    fix_to_apple = bool(args.fix_to_apple)
    mode = "welded" if fix_to_apple else "free"
    output_dir = args.output_dir / mode

    n_substeps = args.n_substeps
    if n_substeps is None:
        n_substeps = N_SUBSTEPS_WELDED if fix_to_apple else N_SUBSTEPS_FREE
    k0_scale = args.k0_scale
    if k0_scale is None:
        k0_scale = K0_SCALE_WELDED if fix_to_apple else K0_SCALE_FREE

    fixture = (
        Path(__file__).resolve().parent.parent
        / "fixtures"
        / "fruiting_system_ranges_straight_rod_test.json"
    )
    ranges = fs.load_ranges(fixture)
    base_params = fs.sample_params(ranges, seed=args.seed)
    feature_cfg = FeatureConfig.from_fix_to_apple(
        fix_to_apple,
        warmup_substeps=args.warmup_substeps,
    )
    fd_kw = _fd_kw()

    result = recover_primary_bend_stiffness(
        base_params,
        ranges,
        epsilon=args.epsilon,
        n_substeps=n_substeps,
        dt=DEFAULT_SUB_DT,
        k0_scale=k0_scale,
        max_iter=args.max_iter,
        fd_kw=fd_kw,
        feature_cfg=feature_cfg,
    )

    k_min, k_max = primary_bend_bounds(ranges)
    k_grid = np.linspace(k_min, k_max, num=int(args.grid_points))
    grid_losses = np.array(
        [
            brute_force_grid_loss(
                base_params,
                result.y_star,
                result.k_star,
                k_grid=np.array([k_val]),
                epsilon=args.epsilon,
                n_substeps=n_substeps,
                dt=DEFAULT_SUB_DT,
                fd_kw=fd_kw,
                feature_cfg=feature_cfg,
            )[1]
            for k_val in k_grid
        ],
        dtype=np.float64,
    )

    print(f"mode: fix_to_apple={fix_to_apple}")
    print(f"feature slice: {feature_cfg.feature_slice}")
    print(f"k* = {result.k_star:.6f}")
    print(f"k0 = {result.k_hist[0]:.6f}")
    print(f"k_final = {result.k_final:.6f}")
    print(f"relative error = {result.rel_err:.4%}")
    print(f"loss: {result.loss_hist[0]:.6e} -> {result.loss_hist[-1]:.6e}")
    print(f"plots -> {output_dir.resolve()}")

    _plot_results(
        result,
        k_grid=k_grid,
        grid_losses=grid_losses,
        output_dir=output_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
