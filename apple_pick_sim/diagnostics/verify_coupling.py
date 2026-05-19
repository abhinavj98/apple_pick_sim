"""Headless diagnostic for staggered MuJoCo + VBD proxy coupling.

Run from repository root::

    PYTHONPATH=$(pwd) uv run --directory newton python \\
      ../apple_pick_sim/diagnostics/verify_coupling.py \\
      --json apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json \\
      --seed 42 --num-substeps 600 --max-force 5 --max-torque 1
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from apple_pick_sim.coupling_force_debug import read_tcp_wrench, wrench_magnitudes
from apple_pick_sim.coupled_fruiting import build_coupled_fruiting_placeholder
from apple_pick_sim.fruiting_system import load_ranges


def _quat_angle_error_rad(q_a: np.ndarray, q_b: np.ndarray) -> float:
    qa = q_a[3:7] / (np.linalg.norm(q_a[3:7]) + 1e-12)
    qb = q_b[3:7] / (np.linalg.norm(q_b[3:7]) + 1e-12)
    dot = float(np.clip(abs(np.dot(qa, qb)), -1.0, 1.0))
    return 2.0 * float(np.arccos(dot))


@dataclass
class CouplingStepRecord:
    step: int
    pos_err: float
    ang_err: float
    f_applied: float
    tau_applied: float
    f_harvest: float
    tau_harvest: float


def _default_ranges_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "fixtures"
        / "fruiting_system_ranges_straight_rod_test.json"
    )


def run_verification(
    ranges_path: Path,
    seed: int,
    *,
    num_substeps: int,
    dt: float,
    max_pose_err: float,
    max_force: float,
    max_torque: float,
    preview_rows: int,
) -> tuple[list[CouplingStepRecord], bool]:
    wp.init()
    ranges = load_ranges(ranges_path)
    scene = build_coupled_fruiting_placeholder(
        ranges,
        seed=seed,
        device="cpu",
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    tcp = scene.tcp_body_index
    proxy = scene.cable.gripper_proxy_body

    records: list[CouplingStepRecord] = []
    ok = True

    for step in range(num_substeps):
        cache_before = read_tcp_wrench(scene.coupling_forces_cache, tcp)
        scene.coupled_substep(dt)
        harvest = read_tcp_wrench(scene.proxy_forces, tcp)

        rq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp]
        pq = scene.cable.state_0.body_q.numpy().reshape(-1, 7)[proxy]
        pos_err = float(np.linalg.norm(rq[:3] - pq[:3]))
        ang_err = _quat_angle_error_rad(rq, pq)

        f_a, t_a = wrench_magnitudes(cache_before)
        f_h, t_h = wrench_magnitudes(harvest)

        rec = CouplingStepRecord(
            step=step,
            pos_err=pos_err,
            ang_err=ang_err,
            f_applied=f_a,
            tau_applied=t_a,
            f_harvest=f_h,
            tau_harvest=t_h,
        )
        records.append(rec)

        if pos_err > max_pose_err or ang_err > max_pose_err:
            ok = False
        if f_h > max_force or t_h > max_torque:
            ok = False

    _print_report(records, preview_rows=preview_rows)
    return records, ok


def _print_report(records: list[CouplingStepRecord], *, preview_rows: int) -> None:
    if not records:
        print("No substeps recorded.")
        return

    def _row(r: CouplingStepRecord) -> str:
        return (
            f"{r.step:5d}  pos_err={r.pos_err:.2e}  ang_err={r.ang_err:.2e}  "
            f"|F|_app={r.f_applied:6.3f}  |τ|_app={r.tau_applied:6.3f}  "
            f"|F|_harv={r.f_harvest:6.3f}  |τ|_harv={r.tau_harvest:6.3f}"
        )

    print("Coupling verification trace (first / last rows):")
    for r in records[:preview_rows]:
        print(_row(r))
    if len(records) > 2 * preview_rows:
        print("  ...")
    for r in records[-preview_rows:]:
        print(_row(r))

    f_harv = np.array([r.f_harvest for r in records], dtype=np.float64)
    t_harv = np.array([r.tau_harvest for r in records], dtype=np.float64)
    pos = np.array([r.pos_err for r in records], dtype=np.float64)

    print("\nSummary:")
    print(f"  substeps:     {len(records)}")
    print(f"  |F|_harvest:  max={f_harv.max():.4f}  mean={f_harv.mean():.4f}  final={f_harv[-1]:.4f}")
    print(f"  |τ|_harvest:  max={t_harv.max():.4f}  mean={t_harv.mean():.4f}  final={t_harv[-1]:.4f}")
    print(f"  pos_err:      max={pos.max():.4e}  mean={pos.mean():.4e}  final={pos[-1]:.4e}")


def _write_csv(path: Path, records: list[CouplingStepRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "pos_err",
                "ang_err",
                "f_applied",
                "tau_applied",
                "f_harvest",
                "tau_harvest",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.step,
                    r.pos_err,
                    r.ang_err,
                    r.f_applied,
                    r.tau_applied,
                    r.f_harvest,
                    r.tau_harvest,
                ]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify staggered MuJoCo+VBD coupling.")
    parser.add_argument("--json", type=Path, default=None, help="Fruiting ranges JSON.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-substeps", type=int, default=600)
    parser.add_argument(
        "--dt",
        type=float,
        default=(1.0 / 60.0) / 30.0,
        help="Substep dt [s]; default matches example_coupled_fruiting sim_dt.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--preview-rows", type=int, default=10)
    parser.add_argument("--max-pose-err", type=float, default=1e-2)
    parser.add_argument("--max-force", type=float, default=5.0, help="Max |F| harvest [N].")
    parser.add_argument("--max-torque", type=float, default=1.0, help="Max |τ| harvest [N·m].")
    args = parser.parse_args(argv)

    ranges_path = args.json if args.json is not None else _default_ranges_path()
    records, ok = run_verification(
        ranges_path,
        args.seed,
        num_substeps=args.num_substeps,
        dt=args.dt,
        max_pose_err=args.max_pose_err,
        max_force=args.max_force,
        max_torque=args.max_torque,
        preview_rows=args.preview_rows,
    )

    if args.csv is not None:
        _write_csv(args.csv, records)
        print(f"\nWrote CSV: {args.csv}")

    if ok:
        print("\nPASS: coupling metrics within thresholds.")
        return 0
    print("\nFAIL: coupling metrics exceeded thresholds.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
