"""Headless benchmark for staggered MuJoCo + VBD ``coupled_substep`` timing.

Run from repository root::

    PYTHONPATH=$(pwd) uv run --directory newton python \\
      ../apple_pick_sim/diagnostics/benchmark_coupling.py \\
      --robot placeholder --warmup-substeps 30 --bench-substeps 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import warp as wp

from apple_pick_sim.coupled_fruiting.builders import (
    build_coupled_fruiting_fr3,
    build_coupled_fruiting_placeholder,
)
from apple_pick_sim.fruiting_system import GripperProxyConfig, load_ranges
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.sim_device import resolve_sim_device
from apple_pick_sim.sim_mujoco_device import resolve_mujoco_use_cpu


def _default_ranges_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "fixtures"
        / "fruiting_system_ranges_straight_rod_test.json"
    )


def _sync_device(device: str) -> None:
    if "cuda" in device:
        wp.synchronize()


def _build_scene(
    robot: str,
    ranges: dict,
    seed: int,
    device: str,
    *,
    fix_to_apple: bool,
    mujoco_use_cpu: bool,
):
    mj_kw: dict = {"disable_contacts": True, "use_mujoco_cpu": mujoco_use_cpu}
    gripper = GripperProxyConfig(fix_to_apple=fix_to_apple)
    if robot == "fr3":
        if not fr3_robot.fr3_assets_available():
            raise SystemExit("FR3 assets missing; see assets/fr3/README.md")
        return build_coupled_fruiting_fr3(
            ranges,
            seed,
            device=device,
            gripper_proxy=gripper,
            mujoco_solver_kwargs=mj_kw,
            mujoco_use_cpu=mujoco_use_cpu,
        )
    return build_coupled_fruiting_placeholder(
        ranges,
        seed,
        device=device,
        gripper_proxy=gripper,
        mujoco_solver_kwargs=mj_kw,
        mujoco_use_cpu=mujoco_use_cpu,
    )


def run_benchmark(
    *,
    robot: str,
    device: str,
    seed: int,
    warmup_substeps: int,
    bench_substeps: int,
    sim_substeps_per_frame: int,
    fix_to_apple: bool,
    mujoco_use_cpu: bool,
) -> dict[str, float]:
    wp.init()
    ranges = load_ranges(_default_ranges_path())
    scene = _build_scene(
        robot,
        ranges,
        seed,
        device,
        fix_to_apple=fix_to_apple,
        mujoco_use_cpu=mujoco_use_cpu,
    )
    dt = (1.0 / 60.0) / sim_substeps_per_frame

    for _ in range(warmup_substeps):
        scene.coupled_substep(dt)
    _sync_device(device)

    t0 = time.perf_counter()
    for _ in range(bench_substeps):
        scene.coupled_substep(dt)
    _sync_device(device)
    elapsed = time.perf_counter() - t0

    ms_per_substep = 1000.0 * elapsed / bench_substeps
    substeps_per_s = bench_substeps / elapsed
    ms_per_frame = ms_per_substep * sim_substeps_per_frame
    frames_per_s = 1000.0 / ms_per_frame if ms_per_frame > 0 else 0.0

    return {
        "ms_per_substep": ms_per_substep,
        "substeps_per_s": substeps_per_s,
        "ms_per_frame": ms_per_frame,
        "frames_per_s": frames_per_s,
        "bench_substeps": float(bench_substeps),
        "dt": dt,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark coupled_substep throughput.")
    parser.add_argument(
        "--robot",
        choices=("placeholder", "fr3"),
        default="placeholder",
        help="Robot model to build (default: placeholder).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Warp device (default: cuda:0 when CUDA is available, else cpu).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-substeps", type=int, default=30)
    parser.add_argument("--bench-substeps", type=int, default=300)
    parser.add_argument(
        "--sim-substeps",
        type=int,
        default=30,
        help="Substeps per frame for derived frame timing (default: 30).",
    )
    parser.add_argument(
        "--fix-to-apple",
        action="store_true",
        help="Weld gripper proxy to apple (stem-harvest coupling path).",
    )
    parser.add_argument(
        "--mujoco-cpu",
        action="store_true",
        help="Force MuJoCo CPU solver (required for --device cpu).",
    )
    parser.add_argument(
        "--mujoco-gpu",
        action="store_true",
        help="Use MuJoCo Warp on CUDA (requires --device cuda:0).",
    )
    args = parser.parse_args(argv)
    device = resolve_sim_device(args.device)
    if args.mujoco_cpu and args.mujoco_gpu:
        raise SystemExit("--mujoco-cpu and --mujoco-gpu are mutually exclusive")
    mujoco_override = True if args.mujoco_cpu else (False if args.mujoco_gpu else None)
    mujoco_use_cpu = resolve_mujoco_use_cpu(device, mujoco_override)

    stats = run_benchmark(
        robot=args.robot,
        device=device,
        seed=args.seed,
        warmup_substeps=args.warmup_substeps,
        bench_substeps=args.bench_substeps,
        sim_substeps_per_frame=args.sim_substeps,
        fix_to_apple=args.fix_to_apple,
        mujoco_use_cpu=mujoco_use_cpu,
    )

    harvest = "stem" if args.fix_to_apple else "velocity-delta"
    mj_backend = "cpu" if mujoco_use_cpu else "warp"
    print(
        f"robot={args.robot} device={device} seed={args.seed} "
        f"mujoco={mj_backend} harvest={harvest}"
    )
    print(f"warmup={args.warmup_substeps} bench={args.bench_substeps} dt={stats['dt']:.6f} s")
    print(f"ms/substep: {stats['ms_per_substep']:.4f}")
    print(f"substeps/s: {stats['substeps_per_s']:.2f}")
    print(
        f"frame @ {args.sim_substeps} substeps: "
        f"{stats['ms_per_frame']:.2f} ms ({stats['frames_per_s']:.2f} fps)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
