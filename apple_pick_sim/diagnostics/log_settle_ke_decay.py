"""Log branch KE timeseries during VBD settle and report envelope decay stability.

Run from repository root::

    uv run python apple_pick_sim/diagnostics/log_settle_ke_decay.py \\
      --num-envs 4 --seed 42 --settle-substeps 15000 --settle-gravity-ramp \\
      --output-dir /tmp/settle_ke_seed42
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from pathlib import Path

import warp as wp

from apple_pick_sim.coupled_fruiting import (
    build_heterogeneous_coupled_fruiting_placeholder,
    print_settle_ke_decay_report,
    print_settle_stability_report,
    settle_stability_reports_from_cable,
)
from apple_pick_sim.coupled_fruiting.settle_ke_decay import (
    DEFAULT_KE_ANALYSIS_TAIL_FRACTION,
    DEFAULT_KE_MIN_PEAKS,
    DEFAULT_KE_PEAK_DECAY_RTOL,
    DEFAULT_KE_SAMPLE_EVERY,
    SettleKeAnalysisConfig,
    SettleKeRecorder,
    _tail_samples,
    find_ke_peaks,
    peak_rows_from_report,
)
from apple_pick_sim.coupled_fruiting.settle_then_weld import (
    apply_settle_gravity_for_substep,
    quiet_all_cable_bodies,
)
from apple_pick_sim.fruiting_system import (
    PLACEHOLDER_EE_MASS_KG,
    GripperProxyConfig,
    default_ranges_fixture_path,
    load_ranges,
    sample_heterogeneous_params_list,
)
from apple_pick_sim.sim_device import resolve_sim_device

TIMESERIES_FIELDS = ("env", "substep", "sim_time_s", "ke_j", "max_branch_speed_m_s")
PEAK_FIELDS = ("env", "peak_idx", "sim_time_s", "peak_ke_j")
SUMMARY_FIELDS = (
    "env",
    "is_ke_decay_stable",
    "is_envelope_decaying",
    "is_ke_below_threshold",
    "num_peaks",
    "first_peak_ke_j",
    "final_peak_ke_j",
    "ke_peak_threshold_j",
    "issues",
)


@dataclasses.dataclass(frozen=True)
class SettleKeLogConfig:
    seed: int = 42
    num_envs: int = 4
    settle_substeps: int = 15000
    settle_gravity_ramp: bool = True
    settle_max_speed: float = 0.05
    env_spacing: tuple[float, float, float] = (2.0, 2.0, 2.0)
    ranges_path: Path | None = None
    device: str | None = None
    fps: float = 30.0
    sim_substeps: int = 60
    ke_sample_every: int = DEFAULT_KE_SAMPLE_EVERY
    ke_analysis_tail_fraction: float = DEFAULT_KE_ANALYSIS_TAIL_FRACTION
    ke_min_peaks: int = DEFAULT_KE_MIN_PEAKS
    ke_peak_decay_rtol: float = DEFAULT_KE_PEAK_DECAY_RTOL
    ke_peak_threshold_j: float | None = None
    output_dir: Path | None = None
    plot: bool = True


def _analysis_config(config: SettleKeLogConfig) -> SettleKeAnalysisConfig:
    return SettleKeAnalysisConfig(
        analysis_tail_fraction=float(config.ke_analysis_tail_fraction),
        min_peaks=int(config.ke_min_peaks),
        peak_decay_rtol=float(config.ke_peak_decay_rtol),
        speed_threshold_m_s=float(config.settle_max_speed),
        ke_peak_threshold_j=config.ke_peak_threshold_j,
    )


def run_settle_ke_log(config: SettleKeLogConfig):
    """Build free-proxy cable, settle with KE recording, return scene + recorder + reports."""
    wp.init()
    device = resolve_sim_device(config.device)
    ranges_path = config.ranges_path or default_ranges_fixture_path()
    ranges = load_ranges(ranges_path)
    per_env_params = sample_heterogeneous_params_list(
        ranges, topology_seed=int(config.seed), num_envs=int(config.num_envs)
    )
    gripper = GripperProxyConfig(
        mass=PLACEHOLDER_EE_MASS_KG,
        fix_to_apple=False,
        robot_facing_weld=False,
    )
    scene = build_heterogeneous_coupled_fruiting_placeholder(
        ranges,
        per_env_params,
        device=device,
        env_spacing=config.env_spacing,
        enable_self_collisions=False,
        gripper_proxy=gripper,
        vbd_only=True,
    )
    sim_dt = (1.0 / float(config.fps)) / int(config.sim_substeps)
    n = int(config.settle_substeps)
    recorder = SettleKeRecorder(
        num_envs=int(config.num_envs),
        sample_every=int(config.ke_sample_every),
    )
    print(
        f"VBD settle KE log: {n} substeps ({n * sim_dt:.3f} s sim), "
        f"sample_every={config.ke_sample_every}, "
        f"gravity_ramp={'on' if config.settle_gravity_ramp else 'off'}",
        flush=True,
    )
    for substep_idx in range(n):
        apply_settle_gravity_for_substep(
            scene,
            substep_idx,
            n,
            gravity_ramp=bool(config.settle_gravity_ramp),
        )
        scene.vbd_substep(sim_dt)
        recorder.record_substep(
            scene.cable,
            per_env_params,
            substep_idx,
            sim_dt,
            sample_every=int(config.ke_sample_every),
        )

    analysis = _analysis_config(config)
    ke_reports = recorder.reports(config=analysis)
    speed_reports = settle_stability_reports_from_cable(
        scene.cable,
        per_env_params,
        max_branch_speed_m_s=float(config.settle_max_speed),
    )
    print_settle_stability_report(speed_reports, prefix="  ")
    print_settle_ke_decay_report(ke_reports, prefix="  ")
    quiet_all_cable_bodies(scene.cable)
    return scene, recorder, ke_reports, analysis


def write_timeseries_csv(path: Path, recorder: SettleKeRecorder) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TIMESERIES_FIELDS)
        writer.writeheader()
        for env, samples in sorted(recorder.all_timeseries().items()):
            for sample in samples:
                writer.writerow(
                    {
                        "env": env,
                        "substep": sample.substep,
                        "sim_time_s": sample.sim_time_s,
                        "ke_j": sample.ke_j,
                        "max_branch_speed_m_s": sample.max_branch_speed_m_s,
                    }
                )


def write_peaks_csv(
    path: Path,
    recorder: SettleKeRecorder,
    reports: list,
    *,
    analysis: SettleKeAnalysisConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PEAK_FIELDS)
        writer.writeheader()
        for report in reports:
            samples = recorder.timeseries(report.world)
            for peak_idx, sim_time_s, peak_ke_j in peak_rows_from_report(
                report,
                samples,
                analysis_tail_fraction=analysis.analysis_tail_fraction,
            ):
                writer.writerow(
                    {
                        "env": report.world,
                        "peak_idx": peak_idx,
                        "sim_time_s": sim_time_s,
                        "peak_ke_j": peak_ke_j,
                    }
                )


def write_summary_csv(path: Path, reports: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for report in reports:
            first_peak = report.peak_ke_j[0] if report.peak_ke_j else float("nan")
            writer.writerow(
                {
                    "env": report.world,
                    "is_ke_decay_stable": int(report.is_ke_decay_stable),
                    "is_envelope_decaying": int(report.is_envelope_decaying),
                    "is_ke_below_threshold": int(report.is_ke_below_threshold),
                    "num_peaks": len(report.peak_ke_j),
                    "first_peak_ke_j": first_peak,
                    "final_peak_ke_j": report.final_peak_ke_j,
                    "ke_peak_threshold_j": report.ke_peak_threshold_j,
                    "issues": ",".join(report.issues),
                }
            )


def write_ke_plots(
    output_dir: Path,
    recorder: SettleKeRecorder,
    reports: list,
    *,
    analysis: SettleKeAnalysisConfig,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping KE plots.", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for report in reports:
        samples = recorder.timeseries(report.world)
        if not samples:
            continue
        tail = _tail_samples(samples, analysis_tail_fraction=analysis.analysis_tail_fraction)
        times = [s.sim_time_s for s in tail]
        ke_vals = [s.ke_j for s in tail]
        ke_arr = [s.ke_j for s in tail]
        peak_idx = find_ke_peaks(ke_arr)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(times, ke_vals, label="branch KE [J]", linewidth=1.0)
        if peak_idx:
            ax.scatter(
                [times[i] for i in peak_idx],
                [ke_arr[i] for i in peak_idx],
                color="tab:red",
                s=24,
                label="peaks",
                zorder=3,
            )
        ax.axhline(
            report.ke_peak_threshold_j,
            color="tab:orange",
            linestyle="--",
            linewidth=1.0,
            label="peak threshold",
        )
        status = "KE_DECAY_STABLE" if report.is_ke_decay_stable else "KE_DECAY_UNSTABLE"
        ax.set_title(f"env{report.world}: {status}")
        ax.set_xlabel("sim time [s]")
        ax.set_ylabel("branch linear KE [J]")
        ax.legend(loc="upper right")
        fig.tight_layout()
        out = output_dir / f"settle_ke_env{report.world}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)


def write_outputs(
    output_dir: Path,
    recorder: SettleKeRecorder,
    reports: list,
    *,
    analysis: SettleKeAnalysisConfig,
    plot: bool,
) -> None:
    write_timeseries_csv(output_dir / "settle_ke_timeseries.csv", recorder)
    write_peaks_csv(output_dir / "settle_ke_peaks.csv", recorder, reports, analysis=analysis)
    write_summary_csv(output_dir / "settle_ke_decay_summary.csv", reports)
    if plot:
        write_ke_plots(output_dir, recorder, reports, analysis=analysis)
    print(f"Wrote KE settle artifacts under {output_dir}", flush=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log branch KE envelope decay during VBD settle.",
    )
    parser.add_argument("--json", type=str, default=None, help="Fruiting DR ranges JSON.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--settle-substeps", type=int, default=15000)
    parser.add_argument(
        "--settle-gravity-ramp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--settle-max-speed", type=float, default=0.05)
    parser.add_argument(
        "--env-spacing",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[2.0, 2.0, 2.0],
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hz", type=float, default=30.0, help="Macro frame rate for sim_dt.")
    parser.add_argument("--sim-substeps", type=int, default=60)
    parser.add_argument("--ke-sample-every", type=int, default=DEFAULT_KE_SAMPLE_EVERY)
    parser.add_argument(
        "--ke-analysis-tail-fraction",
        type=float,
        default=DEFAULT_KE_ANALYSIS_TAIL_FRACTION,
    )
    parser.add_argument("--ke-min-peaks", type=int, default=DEFAULT_KE_MIN_PEAKS)
    parser.add_argument("--ke-peak-decay-rtol", type=float, default=DEFAULT_KE_PEAK_DECAY_RTOL)
    parser.add_argument(
        "--ke-peak-threshold-j",
        type=float,
        default=None,
        help="Override peak KE threshold [J]; default derives from branch mass and settle-max-speed.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for CSV/PNG outputs.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write settle_ke_env{N}.png when matplotlib is available.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = SettleKeLogConfig(
        seed=int(args.seed),
        num_envs=int(args.num_envs),
        settle_substeps=int(args.settle_substeps),
        settle_gravity_ramp=bool(args.settle_gravity_ramp),
        settle_max_speed=float(args.settle_max_speed),
        env_spacing=tuple(float(v) for v in args.env_spacing),
        ranges_path=Path(args.json) if args.json else None,
        device=args.device,
        fps=float(args.hz),
        sim_substeps=int(args.sim_substeps),
        ke_sample_every=int(args.ke_sample_every),
        ke_analysis_tail_fraction=float(args.ke_analysis_tail_fraction),
        ke_min_peaks=int(args.ke_min_peaks),
        ke_peak_decay_rtol=float(args.ke_peak_decay_rtol),
        ke_peak_threshold_j=(
            float(args.ke_peak_threshold_j) if args.ke_peak_threshold_j is not None else None
        ),
        output_dir=Path(args.output_dir),
        plot=bool(args.plot),
    )
    _scene, recorder, reports, analysis = run_settle_ke_log(config)
    write_outputs(
        Path(config.output_dir),
        recorder,
        reports,
        analysis=analysis,
        plot=config.plot,
    )
    stable = sum(1 for r in reports if r.is_ke_decay_stable)
    return 0 if stable == len(reports) else 1


if __name__ == "__main__":
    sys.exit(main())
