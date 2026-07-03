"""Sweep settle duration; report stability after settle, after weld, and after post-weld hold.

For each ``--settle-substeps`` value: build free-proxy cable → VBD settle → stability
report → settle-then-weld with per-env IK → stability + envelope report → coupled hold
for ``--post-weld-frames`` → final stability report.

Run from repository root::

    uv run python apple_pick_sim/diagnostics/sweep_settle_weld_stability.py \\
      --num-envs 4 --seed 42 --settle-substeps 1000,5000,10000 \\
      --post-weld-frames 120 --robot fr3

    uv run python apple_pick_sim/diagnostics/sweep_settle_weld_stability.py \\
      --num-envs 8 --seed 42 --settle-substeps 5000,10000 --robot fr3 \\
      --output-dir /tmp/settle_sweep

Requires FR3 assets (``--robot fr3``). Settle-then-weld IK bootstrap is not supported
for placeholder TCP in batched scenes.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
import time
from pathlib import Path
from typing import Any

import warp as wp

from apple_pick_sim.coupled_fruiting import (
    build_coupled_fruiting_fr3,
    build_coupled_fruiting_placeholder,
    build_heterogeneous_coupled_fruiting_fr3,
    build_heterogeneous_coupled_fruiting_placeholder,
    print_envelope_coverage_report,
    print_settle_stability_report,
    quiet_all_cable_bodies,
    seed_fix_to_apple_from_settled,
    settle_stability_reports_from_cable,
    settle_vbd_substeps,
)
from apple_pick_sim.coupled_fruiting.settle_quasi_static import SettleStabilityReport
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

SUMMARY_FIELDS = (
    "settle_substeps",
    "settle_sim_time_s",
    "post_settle_stable_rate",
    "post_weld_stable_rate",
    "post_weld_ik_inside_rate",
    "post_hold_stable_rate",
    "post_settle_max_branch_speed_m_s",
    "post_hold_max_branch_speed_m_s",
    "wall_time_s",
)


@dataclasses.dataclass(frozen=True)
class SettleWeldSweepConfig:
    """One settle-duration sweep run."""

    seed: int = 42
    num_envs: int = 4
    settle_substeps_list: tuple[int, ...] = (1000, 5000, 10000)
    post_weld_frames: int = 120
    sim_substeps: int = 15
    ranges_path: Path | None = None
    env_spacing: tuple[float, float, float] = (2.0, 2.0, 2.0)
    device: str | None = None
    robot: str = "fr3"
    settle_max_speed: float = 0.05
    settle_gravity_ramp: bool = True
    enable_self_collision: bool = False
    verbose: bool = True


@dataclasses.dataclass(frozen=True)
class SettleWeldTrialResult:
    """Stability outcomes for one settle_substeps value."""

    settle_substeps: int
    settle_sim_time_s: float
    post_settle_reports: tuple[SettleStabilityReport, ...]
    post_weld_reports: tuple[SettleStabilityReport, ...]
    post_hold_reports: tuple[SettleStabilityReport, ...]
    ik_results: tuple[tuple[float, float, bool], ...]
    wall_time_s: float

    @property
    def post_settle_stable_rate(self) -> float:
        return _stable_rate(self.post_settle_reports)

    @property
    def post_weld_stable_rate(self) -> float:
        return _stable_rate(self.post_weld_reports)

    @property
    def post_hold_stable_rate(self) -> float:
        return _stable_rate(self.post_hold_reports)

    @property
    def post_weld_ik_inside_rate(self) -> float:
        if not self.ik_results:
            return float("nan")
        inside = sum(1 for _, _, ok in self.ik_results if ok)
        return inside / len(self.ik_results)

    def summary_row(self) -> dict[str, Any]:
        return {
            "settle_substeps": self.settle_substeps,
            "settle_sim_time_s": self.settle_sim_time_s,
            "post_settle_stable_rate": self.post_settle_stable_rate,
            "post_weld_stable_rate": self.post_weld_stable_rate,
            "post_weld_ik_inside_rate": self.post_weld_ik_inside_rate,
            "post_hold_stable_rate": self.post_hold_stable_rate,
            "post_settle_max_branch_speed_m_s": _max_branch_speed(
                self.post_settle_reports
            ),
            "post_hold_max_branch_speed_m_s": _max_branch_speed(self.post_hold_reports),
            "wall_time_s": self.wall_time_s,
        }


def _stable_rate(reports: tuple[SettleStabilityReport, ...]) -> float:
    if not reports:
        return float("nan")
    return sum(1 for report in reports if report.is_stable) / len(reports)


def _max_branch_speed(reports: tuple[SettleStabilityReport, ...]) -> float:
    if not reports:
        return float("nan")
    return max(report.max_branch_speed_m_s for report in reports)


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


def _run_hold_frames(scene: Any, *, frames: int, sim_substeps: int, sim_dt: float) -> None:
    for _ in range(int(frames)):
        for _ in range(int(sim_substeps)):
            scene.coupled_substep(sim_dt)


def run_settle_weld_trial(
    config: SettleWeldSweepConfig,
    *,
    settle_substeps: int,
    ranges: dict,
    per_env_params: list,
    build_fn,
    build_kw: dict[str, Any],
    sim_dt: float,
) -> SettleWeldTrialResult:
    """Settle → weld → post-weld hold for one settle_substeps value."""
    t0 = time.perf_counter()
    num_envs = int(config.num_envs)
    gripper_welded = _gripper_for_robot(config.robot, fix_to_apple=True)
    gripper_free = dataclasses.replace(
        gripper_welded, fix_to_apple=False, robot_facing_weld=False
    )

    settled = _build_scene(
        build_fn,
        ranges=ranges,
        seed=int(config.seed),
        per_env_params=per_env_params,
        build_kw=build_kw,
        gripper=gripper_free,
        num_envs=num_envs,
        vbd_only=True,
    )
    n_settle = int(settle_substeps)
    settle_vbd_substeps(
        settled,
        substeps=n_settle,
        dt=sim_dt,
        gravity_ramp=bool(config.settle_gravity_ramp),
    )
    post_settle_reports = tuple(
        settle_stability_reports_from_cable(
            settled.cable,
            per_env_params,
            max_branch_speed_m_s=float(config.settle_max_speed),
        )
    )
    quiet_all_cable_bodies(settled.cable)

    scene = _build_scene(
        build_fn,
        ranges=ranges,
        seed=int(config.seed),
        per_env_params=per_env_params,
        build_kw=build_kw,
        gripper=gripper_welded,
        num_envs=num_envs,
        vbd_only=False,
        skip_ik_bootstrap=True,
        defer_template_robot_bootstrap=True,
    )
    seed_fix_to_apple_from_settled(
        welded_scene=scene,
        settled_scene=settled,
        quiet_apple_proxy=True,
        per_env_ik=config.robot == "fr3" and num_envs > 1,
        per_world_proxy_offsets=getattr(scene, "per_world_proxy_offsets", None),
    )
    ik_results = tuple(getattr(scene, "settle_ik_envelope_results", None) or [])
    post_weld_reports = tuple(
        settle_stability_reports_from_cable(
            scene.cable,
            per_env_params,
            max_branch_speed_m_s=float(config.settle_max_speed),
        )
    )

    if config.verbose:
        print(
            f"\n--- settle_substeps={n_settle} "
            f"(sim {n_settle * sim_dt:.3f} s) ---",
            flush=True,
        )
        print("After settle (free proxy):", flush=True)
        print_settle_stability_report(post_settle_reports, prefix="  ", verbose=True)
        print("After weld + IK bootstrap:", flush=True)
        print_envelope_coverage_report(
            ik_results,
            stability_reports=post_weld_reports,
            prefix="  ",
            verbose=True,
        )

    _run_hold_frames(
        scene,
        frames=int(config.post_weld_frames),
        sim_substeps=int(config.sim_substeps),
        sim_dt=sim_dt,
    )
    post_hold_reports = tuple(
        settle_stability_reports_from_cable(
            scene.cable,
            per_env_params,
            max_branch_speed_m_s=float(config.settle_max_speed),
        )
    )

    if config.verbose:
        hold_sim_s = int(config.post_weld_frames) * int(config.sim_substeps) * sim_dt
        print(
            f"After post-weld hold ({config.post_weld_frames} frames, "
            f"{hold_sim_s:.3f} s sim):",
            flush=True,
        )
        print_settle_stability_report(post_hold_reports, prefix="  ", verbose=True)

    return SettleWeldTrialResult(
        settle_substeps=n_settle,
        settle_sim_time_s=n_settle * sim_dt,
        post_settle_reports=post_settle_reports,
        post_weld_reports=post_weld_reports,
        post_hold_reports=post_hold_reports,
        ik_results=ik_results,
        wall_time_s=time.perf_counter() - t0,
    )


def run_settle_weld_sweep(config: SettleWeldSweepConfig) -> list[SettleWeldTrialResult]:
    """Run all settle_substeps values and return per-trial results."""
    wp.init()
    device = resolve_sim_device(config.device)
    mujoco_use_cpu = resolve_mujoco_use_cpu(device)
    ranges_path = config.ranges_path or default_ranges_fixture_path()
    ranges = load_ranges(ranges_path)
    num_envs = int(config.num_envs)
    per_env_params = sample_heterogeneous_params_list(
        ranges, topology_seed=int(config.seed), num_envs=num_envs
    )
    sim_substeps = int(config.sim_substeps)
    sim_dt = (1.0 / 60.0) / sim_substeps
    build_fn = _build_fn(config.robot, num_envs=num_envs)
    build_kw = dict(
        device=device,
        env_spacing=config.env_spacing,
        enable_self_collisions=config.enable_self_collision,
        mujoco_use_cpu=mujoco_use_cpu,
        mujoco_solver_kwargs={"disable_contacts": True, "use_mujoco_cpu": mujoco_use_cpu},
    )

    if config.verbose:
        print(f"ranges: {ranges_path}", flush=True)
        print(
            f"robot={config.robot} device={device} num_envs={num_envs} seed={config.seed}",
            flush=True,
        )
        print(
            f"settle_substeps={list(config.settle_substeps_list)} "
            f"post_weld_frames={config.post_weld_frames} "
            f"settle_gravity_ramp={config.settle_gravity_ramp} "
            f"sim_dt={sim_dt:.6f} s",
            flush=True,
        )

    results: list[SettleWeldTrialResult] = []
    for settle_substeps in config.settle_substeps_list:
        results.append(
            run_settle_weld_trial(
                config,
                settle_substeps=int(settle_substeps),
                ranges=ranges,
                per_env_params=per_env_params,
                build_fn=build_fn,
                build_kw=build_kw,
                sim_dt=sim_dt,
            )
        )
    return results


def write_summary_csv(path: Path, results: list[SettleWeldTrialResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for trial in results:
            writer.writerow(trial.summary_row())


def _print_sweep_summary(results: list[SettleWeldTrialResult]) -> None:
    print("\nSweep summary:", flush=True)
    header = (
        f"{'settle_substeps':>15}  {'settle_s':>8}  "
        f"{'post_settle':>11}  {'post_weld':>11}  {'ik_inside':>11}  "
        f"{'post_hold':>11}  {'wall_s':>8}"
    )
    print(header, flush=True)
    for trial in results:
        row = trial.summary_row()
        print(
            f"{row['settle_substeps']:>15d}  "
            f"{row['settle_sim_time_s']:8.3f}  "
            f"{row['post_settle_stable_rate']:11.1%}  "
            f"{row['post_weld_stable_rate']:11.1%}  "
            f"{row['post_weld_ik_inside_rate']:11.1%}  "
            f"{row['post_hold_stable_rate']:11.1%}  "
            f"{row['wall_time_s']:8.2f}",
            flush=True,
        )


def _parse_settle_substeps(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise ValueError("expected at least one settle_substeps value")
    out = tuple(int(float(p)) for p in parts)
    if any(n <= 0 for n in out):
        raise ValueError("settle_substeps values must be positive integers")
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep settle substeps; report stability after settle, weld, and post-weld hold."
        ),
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Fruiting DR ranges JSON (default: real_world_proxy_variance fixture).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--settle-substeps",
        type=str,
        default="1000,5000,10000",
        help="Comma-separated VBD settle substep counts to sweep.",
    )
    parser.add_argument(
        "--post-weld-frames",
        type=int,
        default=120,
        help="Coupled macro-frames to run after weld before final stability check.",
    )
    parser.add_argument(
        "--env-spacing",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[2.0, 2.0, 2.0],
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--robot", choices=("fr3",), default="fr3")
    parser.add_argument("--sim-substeps", type=int, default=15)
    parser.add_argument("--settle-max-speed", type=float, default=0.05)
    parser.add_argument(
        "--settle-gravity-ramp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Linear 0→−9.81 m/s² gravity ramp during each settle phase.",
    )
    parser.add_argument("--enable-self-collision", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="If set, write summary.csv under this directory.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print sweep summary table.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    robot = str(args.robot)
    if robot == "fr3" and not fr3_robot.fr3_assets_available():
        print("FR3 assets missing; use --robot placeholder or install assets/fr3.", file=sys.stderr)
        return 1

    config = SettleWeldSweepConfig(
        seed=int(args.seed),
        num_envs=int(args.num_envs),
        settle_substeps_list=_parse_settle_substeps(args.settle_substeps),
        post_weld_frames=int(args.post_weld_frames),
        sim_substeps=int(args.sim_substeps),
        ranges_path=Path(args.json) if args.json else None,
        env_spacing=tuple(float(v) for v in args.env_spacing),
        device=args.device,
        robot=robot,
        settle_max_speed=float(args.settle_max_speed),
        settle_gravity_ramp=bool(args.settle_gravity_ramp),
        enable_self_collision=bool(args.enable_self_collision),
        verbose=not bool(args.quiet),
    )
    results = run_settle_weld_sweep(config)
    _print_sweep_summary(results)
    if args.output_dir:
        out = Path(args.output_dir) / "summary.csv"
        write_summary_csv(out, results)
        print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
