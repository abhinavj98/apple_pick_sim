"""Large phased grid sweep for zero-VIC hold stability.

Supports controller × angle-tier × topology grids with worker sharding and resume.

Run from repository root::

    # Phase A: 5 parallel workers (432 cells @ 20 envs)
    for W in 0 1 2 3 4; do
      uv run python -u apple_pick_sim/diagnostics/sweep_zero_vic_stability_large.py \\
        --phase A --worker-id $W --num-workers 5 --resume \\
        --num-envs 20 --seed 42 --duration 5 --settle-substeps 10000 \\
        --stem-gain 1.0,0.95 \\
        --vic-linear-k 180,600,1200,2000 --vic-linear-d 200,400,600 \\
        --vic-angular-k 20,50,80 --vic-angular-d 4,8 \\
        --angle-tiers gravity,level,overhead \\
        --output-dir sweep_runs/vic/A/worker_$W &
    done

Requires PyTorch for VIC joint torques (``uv sync --extra vic``).
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from apple_pick_sim.diagnostics.log_zero_vic_poses import (
    ZeroVicHoldConfig,
    run_zero_vic_hold,
    write_trajectory_csv,
)
from apple_pick_sim.diagnostics.zero_vic_stability_metrics import (
    StabilityThresholds,
    parse_float_list,
)
from apple_pick_sim.fruiting_system import default_ranges_fixture_path, load_ranges

ANGLE_TIER_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "gravity": (-90.0, -10.0, -90.0, -10.0),
    "level": (-20.0, 20.0, -20.0, 20.0),
    "overhead": (10.0, 90.0, 10.0, 90.0),
    "full": (-90.0, 90.0, -90.0, 90.0),
}

SUMMARY_FIELDS = (
    "config_id",
    "phase",
    "stem_gain",
    "vic_linear_k",
    "vic_linear_d",
    "vic_angular_k",
    "vic_angular_d",
    "spur_elev_tier",
    "spur_num_segs",
    "stem_num_segs",
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
    vic_angular_k: float
    vic_angular_d: float
    spur_elev_tier: str
    spur_num_segs: int | None = None
    stem_num_segs: int | None = None


def patch_ranges_for_cell(ranges_base: dict, cell: GridCell) -> dict:
    """Deep-copy fixture ranges and apply per-cell angle / segment overrides."""
    r = copy.deepcopy(ranges_base)
    if cell.spur_elev_tier not in ANGLE_TIER_PRESETS:
        raise ValueError(
            f"unknown angle tier {cell.spur_elev_tier!r}; "
            f"expected one of {sorted(ANGLE_TIER_PRESETS)}"
        )
    smn, smx, tmn, tmx = ANGLE_TIER_PRESETS[cell.spur_elev_tier]
    r["spur"]["elevation_delta_deg"] = {"min": smn, "max": smx}
    r["stem"]["elevation_delta_deg"] = {"min": tmn, "max": tmx}
    if cell.spur_num_segs is not None:
        r["spur"]["num_segments"] = {"min": cell.spur_num_segs, "max": cell.spur_num_segs}
    if cell.stem_num_segs is not None:
        r["stem"]["num_segments"] = {"min": cell.stem_num_segs, "max": cell.stem_num_segs}
    return r


def _slice_for_worker(cells: Sequence[Any], worker_id: int, num_workers: int) -> list[Any]:
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError(f"worker_id must be in [0, {num_workers}), got {worker_id}")
    return list(cells[worker_id::num_workers])


def _load_done_ids(summary_path: Path) -> set[int]:
    if not summary_path.exists():
        return set()
    with open(summary_path, newline="", encoding="utf-8") as fh:
        return {int(row["config_id"]) for row in csv.DictReader(fh)}


def _grid_cells_phase_a(
    *,
    stem_gains: list[float],
    vic_linear_ks: list[float],
    vic_linear_ds: list[float],
    vic_angular_ks: list[float],
    vic_angular_ds: list[float],
    angle_tiers: list[str],
) -> list[GridCell]:
    cells: list[GridCell] = []
    idx = 0
    for tier in angle_tiers:
        for sg, vk, vd, ak, ad in itertools.product(
            stem_gains, vic_linear_ks, vic_linear_ds, vic_angular_ks, vic_angular_ds
        ):
            cells.append(
                GridCell(
                    config_id=idx,
                    stem_gain=sg,
                    vic_linear_k=vk,
                    vic_linear_d=vd,
                    vic_angular_k=ak,
                    vic_angular_d=ad,
                    spur_elev_tier=tier,
                )
            )
            idx += 1
    return cells


def _row_to_cell(row: dict[str, str], config_id: int) -> GridCell:
    def _opt_int(key: str) -> int | None:
        raw = row.get(key, "")
        if raw is None or str(raw).strip() == "":
            return None
        return int(float(raw))

    return GridCell(
        config_id=config_id,
        stem_gain=float(row["stem_gain"]),
        vic_linear_k=float(row["vic_linear_k"]),
        vic_linear_d=float(row["vic_linear_d"]),
        vic_angular_k=float(row["vic_angular_k"]),
        vic_angular_d=float(row["vic_angular_d"]),
        spur_elev_tier=str(row["spur_elev_tier"]),
        spur_num_segs=_opt_int("spur_num_segs"),
        stem_num_segs=_opt_int("stem_num_segs"),
    )


def _rank_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda r: (
            -float(r.get("vic_pass_rate") or 0.0),
            float(r.get("max_apple_drift_m") or float("inf")),
            float(r.get("max_apple_path_m") or float("inf")),
            int(r.get("config_id") or 0),
        ),
    )


def _grid_cells_phase_b(
    refine_from: Path,
    *,
    top_n: int,
    spur_segs: list[int],
    stem_segs: list[int],
) -> list[GridCell]:
    with open(refine_from, newline="", encoding="utf-8") as fh:
        ranked = _rank_rows(list(csv.DictReader(fh)))
    winners = ranked[:top_n]
    cells: list[GridCell] = []
    idx = 0
    for row in winners:
        base = _row_to_cell(row, config_id=idx)
        for spur_n, stem_n in itertools.product(spur_segs, stem_segs):
            cells.append(
                GridCell(
                    config_id=idx,
                    stem_gain=base.stem_gain,
                    vic_linear_k=base.vic_linear_k,
                    vic_linear_d=base.vic_linear_d,
                    vic_angular_k=base.vic_angular_k,
                    vic_angular_d=base.vic_angular_d,
                    spur_elev_tier=base.spur_elev_tier,
                    spur_num_segs=spur_n,
                    stem_num_segs=stem_n,
                )
            )
            idx += 1
    return cells


def _grid_cells_phase_c(refine_from: Path, *, top_n: int) -> list[GridCell]:
    with open(refine_from, newline="", encoding="utf-8") as fh:
        ranked = _rank_rows(list(csv.DictReader(fh)))
    return [_row_to_cell(row, config_id=i) for i, row in enumerate(ranked[:top_n])]


def _build_cells(args: argparse.Namespace) -> list[GridCell]:
    phase = str(args.phase).upper()
    if phase == "A":
        return _grid_cells_phase_a(
            stem_gains=parse_float_list(args.stem_gain),
            vic_linear_ks=parse_float_list(args.vic_linear_k),
            vic_linear_ds=parse_float_list(args.vic_linear_d),
            vic_angular_ks=parse_float_list(args.vic_angular_k),
            vic_angular_ds=parse_float_list(args.vic_angular_d),
            angle_tiers=[t.strip() for t in args.angle_tiers.split(",") if t.strip()],
        )
    if phase == "B":
        if not args.refine_from:
            raise ValueError("phase B requires --refine-from")
        spur_segs = [int(x) for x in parse_float_list(args.spur_segs)]
        stem_segs = [int(x) for x in parse_float_list(args.stem_segs)]
        return _grid_cells_phase_b(
            Path(args.refine_from),
            top_n=int(args.top_n),
            spur_segs=spur_segs,
            stem_segs=stem_segs,
        )
    if phase == "C":
        if not args.refine_from:
            raise ValueError("phase C requires --refine-from")
        return _grid_cells_phase_c(Path(args.refine_from), top_n=int(args.top_n))
    raise ValueError(f"unknown phase {args.phase!r}; expected A, B, or C")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Large phased zero-VIC stability sweep.")
    parser.add_argument("--json", type=str, default=None, help="DR ranges JSON path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--log-interval", type=float, default=0.25)
    parser.add_argument("--settle-substeps", type=int, default=10000)
    parser.add_argument("--sim-substeps", type=int, default=15)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--phase", type=str, default="A", choices=["A", "B", "C", "a", "b", "c"])
    parser.add_argument("--stem-gain", type=str, default="1.0,0.95")
    parser.add_argument("--vic-linear-k", type=str, default="180,600,1200,2000")
    parser.add_argument("--vic-linear-d", type=str, default="200,400,600")
    parser.add_argument("--vic-angular-k", type=str, default="20,50,80")
    parser.add_argument("--vic-angular-d", type=str, default="4,8")
    parser.add_argument(
        "--angle-tiers",
        type=str,
        default="gravity,level,overhead",
        help="Comma-separated angle tier names (phase A).",
    )
    parser.add_argument(
        "--spur-segs",
        type=str,
        default="3,4,5",
        help="Comma-separated spur num_segments (phase B).",
    )
    parser.add_argument(
        "--stem-segs",
        type=str,
        default="3,4,5",
        help="Comma-separated stem num_segments (phase B).",
    )
    parser.add_argument("--stem-force-cap-n", type=float, default=200.0)
    parser.add_argument("--stem-torque-cap-nm", type=float, default=50.0)
    parser.add_argument("--max-apple-drift-m", type=float, default=0.02)
    parser.add_argument("--max-apple-z-drop-m", type=float, default=0.015)
    parser.add_argument("--max-apple-path-m", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--save-failures-only", action="store_true")
    parser.add_argument("--enable-self-collision", action="store_true")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--refine-from",
        type=str,
        default=None,
        help="Merged summary.csv from prior phase (phases B and C).",
    )
    parser.add_argument("--top-n", type=int, default=50, help="Top configs to refine (B/C).")
    return parser.parse_args(argv)


def _hold_config(
    args: argparse.Namespace,
    cell: GridCell,
    ranges_base: dict,
) -> ZeroVicHoldConfig:
    ranges_path = Path(args.json) if args.json else default_ranges_fixture_path()
    thresholds = StabilityThresholds(
        max_apple_drift_m=float(args.max_apple_drift_m),
        max_apple_z_drop_m=float(args.max_apple_z_drop_m),
        max_apple_path_length_m=float(args.max_apple_path_m),
        max_harvest_force_n=float(args.stem_force_cap_n),
    )
    save_traj = bool(args.save_trajectories)
    return ZeroVicHoldConfig(
        seed=int(args.seed) + int(cell.config_id),
        num_envs=int(args.num_envs),
        ranges_path=ranges_path,
        ranges_override=patch_ranges_for_cell(ranges_base, cell),
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
        vic_angular_k=float(cell.vic_angular_k),
        vic_angular_d=float(cell.vic_angular_d),
        thresholds=thresholds,
        write_trajectory=save_traj,
        print_settle_report=False,
        print_vic_summary=False,
    )


def _summary_row(
    args: argparse.Namespace,
    cell: GridCell,
    wall_time_s: float,
    result,
) -> dict[str, Any]:
    s = result.summary
    return {
        "config_id": cell.config_id,
        "phase": str(args.phase).upper(),
        "stem_gain": cell.stem_gain,
        "vic_linear_k": cell.vic_linear_k,
        "vic_linear_d": cell.vic_linear_d,
        "vic_angular_k": cell.vic_angular_k,
        "vic_angular_d": cell.vic_angular_d,
        "spur_elev_tier": cell.spur_elev_tier,
        "spur_num_segs": "" if cell.spur_num_segs is None else cell.spur_num_segs,
        "stem_num_segs": "" if cell.stem_num_segs is None else cell.stem_num_segs,
        "sim_substeps": int(args.sim_substeps),
        "seed": int(args.seed) + int(cell.config_id),
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


def _trajectory_path(out_dir: Path, cell: GridCell) -> Path:
    spur = "d" if cell.spur_num_segs is None else str(cell.spur_num_segs)
    stem = "d" if cell.stem_num_segs is None else str(cell.stem_num_segs)
    return out_dir / (
        f"trajectory_c{cell.config_id}_sg{cell.stem_gain}_"
        f"k{cell.vic_linear_k}_d{cell.vic_linear_d}_"
        f"ak{cell.vic_angular_k}_ad{cell.vic_angular_d}_"
        f"tier{cell.spur_elev_tier}_sp{spur}_st{stem}.csv"
    )


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"

    all_cells = _build_cells(args)
    worker_cells = _slice_for_worker(all_cells, int(args.worker_id), int(args.num_workers))
    done_ids = _load_done_ids(summary_path) if args.resume else set()
    pending = [c for c in worker_cells if c.config_id not in done_ids]

    ranges_base = load_ranges(Path(args.json) if args.json else default_ranges_fixture_path())

    print(
        f"sweep phase={args.phase.upper()} worker={args.worker_id}/{args.num_workers}  "
        f"total_cells={len(all_cells)}  worker_cells={len(worker_cells)}  "
        f"pending={len(pending)}  num_envs={args.num_envs}  output={out_dir}",
        flush=True,
    )

    write_header = not summary_path.exists() or summary_path.stat().st_size == 0
    mode = "a" if summary_path.exists() and not write_header else "w"

    best_pass = -1.0
    best_cell: GridCell | None = None
    best_drift = float("inf")

    with open(summary_path, mode, newline="", encoding="utf-8") as summary_fh:
        writer = csv.DictWriter(summary_fh, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()

        for i, cell in enumerate(pending):
            config = _hold_config(args, cell, ranges_base)
            if i == 0 and not done_ids:
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
                f"config {cell.config_id} ({i + 1}/{len(pending)})  "
                f"tier={cell.spur_elev_tier}  stem={cell.stem_gain:g}  "
                f"k={cell.vic_linear_k:g}  d={cell.vic_linear_d:g}  "
                f"ak={cell.vic_angular_k:g}  ad={cell.vic_angular_d:g}  "
                f"sp={cell.spur_num_segs}  st={cell.stem_num_segs}  "
                f"pass={n_stable}/{args.num_envs} ({100.0 * pass_rate:.1f}%)  "
                f"max_drift={max_drift:.4f} m  wall={wall_time_s:.1f}s",
                flush=True,
            )

            save_traj = bool(args.save_trajectories)
            if save_traj and args.save_failures_only and pass_rate >= 1.0 - 1e-9:
                save_traj = False
            if save_traj and result.time_series:
                write_trajectory_csv(result.time_series, str(_trajectory_path(out_dir, cell)))

            if pass_rate > best_pass or (
                abs(pass_rate - best_pass) < 1e-9 and max_drift < best_drift
            ):
                best_pass = pass_rate
                best_drift = max_drift
                best_cell = cell

    if best_cell is not None:
        print(
            f"best: config_id={best_cell.config_id}  tier={best_cell.spur_elev_tier}  "
            f"stem_gain={best_cell.stem_gain}  vic_linear_k={best_cell.vic_linear_k}  "
            f"vic_linear_d={best_cell.vic_linear_d}  vic_angular_k={best_cell.vic_angular_k}  "
            f"vic_pass_rate={100.0 * best_pass:.1f}%  max_drift={best_drift:.4f} m",
            flush=True,
        )
    print(f"wrote {summary_path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
