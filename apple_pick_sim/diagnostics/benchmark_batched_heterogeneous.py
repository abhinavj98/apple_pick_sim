"""Headless profiler for batched heterogeneous coupled fruiting pipeline.

Run from repository root::

    uv run python apple_pick_sim/diagnostics/benchmark_batched_heterogeneous.py \\
      --num-envs 4 --seed 42 --robot fr3

Nsight Systems (optional NVTX ranges when ``nvtx`` is installed)::

    nsys profile --trace=cuda,nvtx --output=profile_report \\
      uv run python apple_pick_sim/diagnostics/benchmark_batched_heterogeneous.py \\
      --num-envs 4 --seed 42 --robot fr3 --bench-frames 60
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import warp as wp

from apple_pick_sim.coupled_fruiting import (
    build_coupled_fruiting_fr3,
    build_coupled_fruiting_placeholder,
    build_heterogeneous_coupled_fruiting_fr3,
    build_heterogeneous_coupled_fruiting_placeholder,
    quiet_all_cable_bodies,
    seed_fix_to_apple_from_settled,
    settle_vbd_substeps,
)
from apple_pick_sim.fruiting_system import (
    PLACEHOLDER_EE_MASS_KG,
    GripperProxyConfig,
    default_ranges_fixture_path,
    load_ranges,
    sample_heterogeneous_params_list,
)
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.sim_device import resolve_sim_device
from apple_pick_sim.sim_mujoco_device import resolve_mujoco_use_cpu

try:
    import nvtx as _nvtx
except ImportError:
    _nvtx = None

_PHASE_NAMES = (
    "build_settled",
    "settle",
    "build_welded",
    "ik_bootstrap",
    "warmup",
    "step_bench",
)


def _default_ranges_path() -> Path:
    return default_ranges_fixture_path()


def _sync_device(device: str) -> None:
    if "cuda" in device:
        wp.synchronize()


@contextlib.contextmanager
def _phase(name: str, results: dict[str, float], device: str):
    """Wall-clock phase timer with optional NVTX range (soft dep)."""
    if _nvtx is not None:
        _nvtx.push_range(name)
    try:
        _sync_device(device)
        t0 = time.perf_counter()
        yield
        _sync_device(device)
        results[name] = (time.perf_counter() - t0) * 1000.0
    finally:
        if _nvtx is not None:
            _nvtx.pop_range()


def _gripper_for_robot(robot: str, *, fix_to_apple: bool) -> GripperProxyConfig:
    if robot == "fr3":
        return GripperProxyConfig(
            mass=fr3_robot.EE_MASS_KG,
            fix_to_apple=fix_to_apple,
            robot_facing_weld=fix_to_apple,
        )
    return GripperProxyConfig(
        mass=PLACEHOLDER_EE_MASS_KG,
        fix_to_apple=fix_to_apple,
        robot_facing_weld=fix_to_apple,
    )


def _build_fn(robot: str, *, num_envs: int):
    if int(num_envs) == 1:
        return (
            build_coupled_fruiting_fr3
            if robot == "fr3"
            else build_coupled_fruiting_placeholder
        )
    return (
        build_heterogeneous_coupled_fruiting_fr3
        if robot == "fr3"
        else build_heterogeneous_coupled_fruiting_placeholder
    )


def _build_scene(
    build_fn,
    *,
    ranges: dict,
    seed: int,
    per_env_params: list,
    build_kw: dict[str, Any],
    gripper: GripperProxyConfig,
    num_envs: int,
    vbd_only: bool,
    skip_ik_bootstrap: bool = False,
    defer_template_robot_bootstrap: bool = False,
):
    if int(num_envs) == 1:
        single_kw = {k: v for k, v in build_kw.items() if k != "env_spacing"}
        kw = {**single_kw, "gripper_proxy": gripper, "vbd_only": vbd_only}
        if (
            not vbd_only
            and skip_ik_bootstrap
            and build_fn is build_coupled_fruiting_fr3
        ):
            kw["skip_ik_bootstrap"] = True
        return build_fn(ranges, int(seed), **kw)
    kw = {
        **build_kw,
        "gripper_proxy": gripper,
        "vbd_only": vbd_only,
        "skip_ik_bootstrap": skip_ik_bootstrap,
        "defer_template_robot_bootstrap": defer_template_robot_bootstrap,
    }
    return build_fn(ranges, per_env_params, **kw)


def _run_frames(scene: Any, *, frames: int, sim_substeps: int, sim_dt: float) -> None:
    for _ in range(int(frames)):
        for _ in range(int(sim_substeps)):
            scene.coupled_substep(sim_dt)


def _phase_entry(total_ms: float, *, calls: int = 1, extra: dict[str, float] | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"total_ms": float(total_ms), "calls": int(calls)}
    if extra:
        entry.update(extra)
    return entry


def run_profile(
    *,
    ranges_path: Path | str | None = None,
    seed: int = 42,
    num_envs: int = 4,
    robot: str = "placeholder",
    device: str | None = None,
    env_spacing: tuple[float, float, float] = (2.0, 2.0, 2.0),
    settle_substeps: int = 5000,
    sim_substeps: int = 15,
    warmup_frames: int = 30,
    bench_frames: int = 200,
    enable_self_collisions: bool = False,
    mujoco_use_cpu: bool | None = None,
    write_json: bool = True,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Profile build → settle → weld → IK bootstrap → warmup → step bench."""
    wp.init()
    resolved_device = resolve_sim_device(device)
    resolved_mujoco_cpu = resolve_mujoco_use_cpu(resolved_device, mujoco_use_cpu)
    ranges = load_ranges(Path(ranges_path) if ranges_path is not None else _default_ranges_path())
    per_env_params = sample_heterogeneous_params_list(
        ranges, topology_seed=int(seed), num_envs=int(num_envs)
    )

    build_fn = _build_fn(robot, num_envs=int(num_envs))
    gripper_welded = _gripper_for_robot(robot, fix_to_apple=True)
    gripper_free = dataclasses.replace(gripper_welded, fix_to_apple=False, robot_facing_weld=False)
    build_kw = dict(
        device=resolved_device,
        env_spacing=env_spacing,
        enable_self_collisions=enable_self_collisions,
        mujoco_use_cpu=resolved_mujoco_cpu,
        mujoco_solver_kwargs={"disable_contacts": True, "use_mujoco_cpu": resolved_mujoco_cpu},
    )
    sim_dt = (1.0 / 60.0) / int(sim_substeps)

    timings: dict[str, float] = {}
    settled = None

    with _phase("build_settled", timings, resolved_device):
        settled = _build_scene(
            build_fn,
            ranges=ranges,
            seed=int(seed),
            per_env_params=per_env_params,
            build_kw=build_kw,
            gripper=gripper_free,
            num_envs=int(num_envs),
            vbd_only=True,
        )

    with _phase("settle", timings, resolved_device):
        settle_vbd_substeps(settled, substeps=int(settle_substeps), dt=sim_dt)
        quiet_all_cable_bodies(settled.cable)

    with _phase("build_welded", timings, resolved_device):
        scene = _build_scene(
            build_fn,
            ranges=ranges,
            seed=int(seed),
            per_env_params=per_env_params,
            build_kw=build_kw,
            gripper=gripper_welded,
            num_envs=int(num_envs),
            vbd_only=False,
            skip_ik_bootstrap=True,
            defer_template_robot_bootstrap=True,
        )

    with _phase("ik_bootstrap", timings, resolved_device):
        seed_fix_to_apple_from_settled(
            welded_scene=scene,
            settled_scene=settled,
            quiet_apple_proxy=True,
            per_env_ik=robot == "fr3" and int(num_envs) > 1,
            per_world_proxy_offsets=getattr(scene, "per_world_proxy_offsets", None),
        )

    with _phase("warmup", timings, resolved_device):
        _run_frames(scene, frames=warmup_frames, sim_substeps=sim_substeps, sim_dt=sim_dt)

    with _phase("step_bench", timings, resolved_device):
        _run_frames(scene, frames=bench_frames, sim_substeps=sim_substeps, sim_dt=sim_dt)

    bench_ms = timings["step_bench"]
    total_substeps = int(bench_frames) * int(sim_substeps)
    ms_per_frame = bench_ms / max(int(bench_frames), 1)
    ms_per_substep = bench_ms / max(total_substeps, 1)
    fps = 1000.0 / ms_per_frame if ms_per_frame > 0.0 else 0.0

    report: dict[str, Any] = {
        "num_envs": int(num_envs),
        "seed": int(seed),
        "device": resolved_device,
        "robot": robot,
        "mujoco_use_cpu": resolved_mujoco_cpu,
        "settle_substeps": int(settle_substeps),
        "sim_substeps": int(sim_substeps),
        "warmup_frames": int(warmup_frames),
        "bench_frames": int(bench_frames),
        "sim_dt": sim_dt,
        "phases": {
            "build_settled": _phase_entry(timings["build_settled"]),
            "settle": _phase_entry(timings["settle"]),
            "build_welded": _phase_entry(timings["build_welded"]),
            "ik_bootstrap": _phase_entry(timings["ik_bootstrap"]),
            "warmup": _phase_entry(timings["warmup"]),
            "step_bench": _phase_entry(
                bench_ms,
                extra={
                    "ms_per_frame": ms_per_frame,
                    "ms_per_substep": ms_per_substep,
                    "fps": fps,
                },
            ),
        },
    }

    if write_json:
        out = Path(output_path) if output_path is not None else _default_output_path(num_envs, seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["output_path"] = str(out)

    return report


def _default_output_path(num_envs: int, seed: int) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(f"profile_{num_envs}envs_{seed}_{ts}.json")


def _print_summary(report: dict[str, Any]) -> None:
    print(
        f"robot={report['robot']} device={report['device']} "
        f"num_envs={report['num_envs']} seed={report['seed']} "
        f"mujoco={'cpu' if report['mujoco_use_cpu'] else 'warp'}"
    )
    print(
        f"settle_substeps={report['settle_substeps']} "
        f"warmup_frames={report['warmup_frames']} bench_frames={report['bench_frames']} "
        f"sim_substeps={report['sim_substeps']} dt={report['sim_dt']:.6f} s"
    )
    for name in _PHASE_NAMES:
        phase = report["phases"][name]
        line = f"{name}: {phase['total_ms']:.2f} ms"
        if name == "step_bench":
            line += (
                f"  ({phase['ms_per_frame']:.2f} ms/frame, "
                f"{phase['ms_per_substep']:.4f} ms/substep, "
                f"{phase['fps']:.2f} fps)"
            )
        print(line)
    if "output_path" in report:
        print(f"Wrote {report['output_path']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile batched heterogeneous coupled fruiting pipeline phases."
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path to fruiting-system range JSON (default: real-world proxy variance fixture).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--env-spacing",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[2.0, 2.0, 2.0],
    )
    parser.add_argument("--device", default=None, help="Warp device (default: cuda:0 when available).")
    parser.add_argument("--robot", choices=("placeholder", "fr3"), default="fr3")
    parser.add_argument("--settle-substeps", type=int, default=5000)
    parser.add_argument("--sim-substeps", type=int, default=15)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--bench-frames", type=int, default=200)
    parser.add_argument("--enable-self-collision", action="store_true")
    parser.add_argument("--mujoco-cpu", action="store_true")
    parser.add_argument("--mujoco-gpu", action="store_true")
    parser.add_argument("--output", type=str, default=None, help="JSON report path.")
    parser.add_argument("--no-output", action="store_true", help="Do not write JSON report.")
    args = parser.parse_args(argv)

    if args.mujoco_cpu and args.mujoco_gpu:
        raise SystemExit("--mujoco-cpu and --mujoco-gpu are mutually exclusive")

    robot = str(args.robot)
    if robot == "fr3" and not fr3_robot.fr3_assets_available():
        raise SystemExit("FR3 assets missing; see assets/fr3/README.md")

    device = resolve_sim_device(args.device)
    mujoco_override = True if args.mujoco_cpu else (False if args.mujoco_gpu else None)
    mujoco_use_cpu = resolve_mujoco_use_cpu(device, mujoco_override)

    report = run_profile(
        ranges_path=args.json,
        seed=int(args.seed),
        num_envs=int(args.num_envs),
        robot=robot,
        device=device,
        env_spacing=tuple(float(v) for v in args.env_spacing),
        settle_substeps=int(args.settle_substeps),
        sim_substeps=int(args.sim_substeps),
        warmup_frames=int(args.warmup_frames),
        bench_frames=int(args.bench_frames),
        enable_self_collisions=bool(args.enable_self_collision),
        mujoco_use_cpu=mujoco_use_cpu,
        write_json=not bool(args.no_output),
        output_path=args.output,
    )
    _print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
