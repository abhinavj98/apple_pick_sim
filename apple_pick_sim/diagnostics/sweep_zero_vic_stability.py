"""Grid sweep for zero-VIC hold stability (settle→weld, 5 s hold default).

Grid is over VIC linear K/D. ``--stem-gain`` defaults to ``1.0,0.95`` for
A/B comparison of stem harvest under-relaxation before changing the production
default (still 1.0 in :data:`DEFAULT_STEM_COUPLING_GAIN`).

Run from repository root::

    # Stem A/B at your viewer gains (20 envs, 5 s hold)
    uv run python apple_pick_sim/diagnostics/sweep_zero_vic_stability.py \\
      --num-envs 20 --seed 42 --duration 5 --log-interval 0.25 \\
      --settle-substeps 10000 --vic-angular-k 50 \\
      --stem-gain 1.0,0.95 --vic-linear-k 600 --vic-linear-d 200,400 \\
      --output-dir /tmp/vic_stem_ab

    # Full VIC grid with stem fixed to one value
    uv run python apple_pick_sim/diagnostics/sweep_zero_vic_stability.py \\
      --num-envs 500 --seed 42 --duration 5 --log-interval 0.25 \\
      --output-dir /tmp/vic_sweep_full \\
      --stem-gain 0.95 \\
      --vic-linear-k 180,600,2000 \\
      --vic-linear-d 100,200,400

Requires PyTorch for VIC joint torques (``uv sync --extra vic``).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from dataclasses import dataclass, replace
from pathlib import Path

from apple_pick_sim.diagnostics.log_zero_vic_poses import (
    ZeroVicHoldConfig,
    run_zero_vic_hold,
    write_trajectory_csv,
)
from apple_pick_sim.diagnostics.zero_vic_stability_metrics import (
    StabilityThresholds,
    parse_float_list,
)
from apple_pick_sim.fruiting_system import default_ranges_fixture_path

SUMMARY_FIELDS = (
    "config_id",
    "stem_gain",
    "vic_linear_k",
    "vic_linear_d",
    "sim_substeps",
    "seed",
    "num_envs",
    "settle_pass_rate",
    "ik_pass_rate",
    "vic_pass_rate",
    "max_apple_drift_m",
    "max_apple_z_drop_m",
    "max_apple_path_m",
    "max_pos_err",
    "max_tcp_speed",
    "max_harvest_n",
    "wall_time_s",
)


@dataclass(frozen=True)
class GridCell:
    """One hyperparameter combination in the sweep grid."""

    config_id: int
    stem_gain: float
    vic_linear_k: float
    vic_linear_d: float


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid sweep for zero-VIC hold stability.")
    parser.add_argument("--json", type=str, default=None, help="DR ranges JSON path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=500)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--log-interval", type=float, default=0.25)
    parser.add_argument("--settle-substeps", type=int, default=5000)
    parser.add_argument("--sim-substeps", type=int, default=15)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--stem-gain",
        type=str,
        default="1.0,0.95",
        help="Comma-separated stem_coupling_gain values (A/B compare before default change).",
    )
    parser.add_argument(
        "--vic-linear-k",
        type=str,
        default="180,600,2000",
        help="Comma-separated VIC linear K [N/m].",
    )
    parser.add_argument(
        "--vic-linear-d",
        type=str,
        default="100,200,400",
        help="Comma-separated VIC linear D [N·s/m].",
    )
    parser.add_argument("--vic-angular-k", type=float, default=20.0)
    parser.add_argument("--vic-angular-d", type=float, default=4.0)
    parser.add_argument("--stem-force-cap-n", type=float, default=200.0)
    parser.add_argument("--stem-torque-cap-nm", type=float, default=50.0)
    parser.add_argument("--max-apple-drift-m", type=float, default=0.02)
    parser.add_argument("--max-apple-z-drop-m", type=float, default=0.015)
    parser.add_argument("--max-apple-path-m", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for summary.csv and optional trajectory CSVs.",
    )
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Write per-config trajectory CSV under output-dir.",
    )
    parser.add_argument(
        "--save-failures-only",
        action="store_true",
        help="With --save-trajectories, only write configs with vic_pass_rate < 1.",
    )
    parser.add_argument("--enable-self-collision", action="store_true")
    return parser.parse_args(argv)


def _grid_cells(args: argparse.Namespace) -> list[GridCell]:
    stem_gains = parse_float_list(args.stem_gain)
    vic_ks = parse_float_list(args.vic_linear_k)
    vic_ds = parse_float_list(args.vic_linear_d)
    cells: list[GridCell] = []
    for idx, (sg, vk, vd) in enumerate(itertools.product(stem_gains, vic_ks, vic_ds)):
        cells.append(
            GridCell(config_id=idx, stem_gain=sg, vic_linear_k=vk, vic_linear_d=vd)
        )
    return cells


def _hold_config(args: argparse.Namespace, cell: GridCell) -> ZeroVicHoldConfig:
    ranges_path = Path(args.json) if args.json else default_ranges_fixture_path()
    thresholds = StabilityThresholds(
        max_apple_drift_m=float(args.max_apple_drift_m),
        max_apple_z_drop_m=float(args.max_apple_z_drop_m),
        max_apple_path_length_m=float(args.max_apple_path_m),
        max_harvest_force_n=float(args.stem_force_cap_n),
    )
    save_traj = bool(args.save_trajectories)
    return ZeroVicHoldConfig(
        seed=int(args.seed),
        num_envs=int(args.num_envs),
        ranges_path=ranges_path,
        device=args.device,
        settle_substeps=int(args.settle_substeps),
        duration=float(args.duration),
        log_interval=float(args.log_interval),
        sim_substeps=int(args.sim_substeps),
        enable_self_collision=bool(args.enable_self_collision),
        stem_coupling_gain=float(cell.stem_gain),
        stem_force_cap_n=float(args.stem_force_cap_n),
        stem_torque_cap_nm=float(args.stem_torque_cap_nm),
        vic_linear_k=float(cell.vic_linear_k),
        vic_linear_d=float(cell.vic_linear_d),
        vic_angular_k=float(args.vic_angular_k),
        vic_angular_d=float(args.vic_angular_d),
        thresholds=thresholds,
        write_trajectory=save_traj,
        print_settle_report=False,
        print_vic_summary=False,
    )


def _summary_row(args: argparse.Namespace, cell: GridCell, wall_time_s: float, result) -> dict:
    s = result.summary
    return {
        "config_id": cell.config_id,
        "stem_gain": cell.stem_gain,
        "vic_linear_k": cell.vic_linear_k,
        "vic_linear_d": cell.vic_linear_d,
        "sim_substeps": int(args.sim_substeps),
        "seed": int(args.seed),
        "num_envs": int(args.num_envs),
        "settle_pass_rate": "" if s.settle_pass_rate is None else s.settle_pass_rate,
        "ik_pass_rate": "" if s.ik_pass_rate is None else s.ik_pass_rate,
        "vic_pass_rate": s.vic_pass_rate,
        "max_apple_drift_m": s.max_apple_drift_m,
        "max_apple_z_drop_m": s.max_apple_z_drop_m,
        "max_apple_path_m": s.max_apple_path_length_m,
        "max_pos_err": s.max_pos_err_m,
        "max_tcp_speed": s.max_tcp_speed_m_s,
        "max_harvest_n": s.max_harvest_force_n,
        "wall_time_s": wall_time_s,
    }


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    cells = _grid_cells(args)
    total = len(cells)

    print(
        f"sweep: {total} configs  num_envs={args.num_envs}  duration={args.duration}s  "
        f"seed={args.seed}  output={out_dir}",
        flush=True,
    )

    best_pass = -1.0
    best_cell: GridCell | None = None
    best_drift = float("inf")

    with open(summary_path, "w", newline="", encoding="utf-8") as summary_fh:
        writer = csv.DictWriter(summary_fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()

        for i, cell in enumerate(cells):
            config = _hold_config(args, cell)
            if i == 0:
                config = replace(config, print_settle_report=True)
            t0 = time.perf_counter()
            result = run_zero_vic_hold(config)
            wall_time_s = time.perf_counter() - t0

            row = _summary_row(args, cell, wall_time_s, result)
            writer.writerow(row)
            summary_fh.flush()

            pass_rate = result.summary.vic_pass_rate
            max_drift = result.summary.max_apple_drift_m
            n_stable = sum(1 for m in result.per_env_metrics if m.is_stable)
            print(
                f"config {i + 1}/{total}  stem={cell.stem_gain:g}  "
                f"k={cell.vic_linear_k:g}  d={cell.vic_linear_d:g}  "
                f"pass={n_stable}/{args.num_envs} ({100.0 * pass_rate:.1f}%)  "
                f"max_drift={max_drift:.4f} m  wall={wall_time_s:.1f}s",
                flush=True,
            )

            save_traj = bool(args.save_trajectories)
            if save_traj and args.save_failures_only and pass_rate >= 1.0 - 1e-9:
                save_traj = False
            if save_traj and result.time_series:
                traj_path = out_dir / (
                    f"trajectory_c{cell.config_id}_sg{cell.stem_gain}_"
                    f"k{cell.vic_linear_k}_d{cell.vic_linear_d}.csv"
                )
                write_trajectory_csv(result.time_series, str(traj_path))

            if pass_rate > best_pass or (
                abs(pass_rate - best_pass) < 1e-9 and max_drift < best_drift
            ):
                best_pass = pass_rate
                best_drift = max_drift
                best_cell = cell

    if best_cell is not None:
        print(
            f"best: config_id={best_cell.config_id}  stem_gain={best_cell.stem_gain}  "
            f"vic_linear_k={best_cell.vic_linear_k}  vic_linear_d={best_cell.vic_linear_d}  "
            f"vic_pass_rate={100.0 * best_pass:.1f}%  max_drift={best_drift:.4f} m",
            flush=True,
        )
    print(f"wrote {summary_path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
