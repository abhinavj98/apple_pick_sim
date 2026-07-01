"""Log TCP / apple poses under zero VIC teleop (hold setpoint, zero twist).

Runs settle-then-weld, configures batched VIC joint torques, applies zero
``EEVelocity`` each frame, and writes one CSV row per env per log interval.

Run from repository root::

    uv run python apple_pick_sim/diagnostics/log_zero_vic_poses.py \\
      --num-envs 4 --seed 42 --duration 30 --log-interval 1.0

Requires PyTorch for VIC joint torques (``uv sync --extra vic``).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import warp as wp

from apple_pick_sim.coupled_fruiting.scene import DEFAULT_STEM_COUPLING_GAIN
from apple_pick_sim.coupled_fruiting import (
    build_heterogeneous_coupled_fruiting_fr3,
    quiet_all_cable_bodies,
    seed_fix_to_apple_from_settled,
    settle_stability_reports_from_cable,
    settle_vbd_substeps,
)
from apple_pick_sim.coupled_fruiting.settle_quasi_static import (
    SettleStabilityReport,
    print_settle_stability_report,
)
from apple_pick_sim.diagnostics.zero_vic_stability_metrics import (
    EnvStabilityMetrics,
    HoldSummary,
    StabilityThresholds,
    compute_env_stability_metrics,
    summarize_hold_metrics,
)
from apple_pick_sim.fruiting_system import (
    GripperProxyConfig,
    default_ranges_fixture_path,
    load_ranges,
    sample_heterogeneous_params_list,
)
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import ImpedanceGains
from apple_pick_sim.robot.fr3_robot.controllers.keyboard import EEVelocity
from apple_pick_sim.sim_device import resolve_sim_device
from apple_pick_sim.sim_mujoco_device import resolve_mujoco_use_cpu

_ZERO_VELOCITY = EEVelocity()

CSV_FIELDS = (
    "t_s",
    "env",
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "tcp_qx",
    "tcp_qy",
    "tcp_qz",
    "tcp_qw",
    "tcp_vx",
    "tcp_vy",
    "tcp_vz",
    "target_tcp_x",
    "target_tcp_y",
    "target_tcp_z",
    "pos_err_m",
    "apple_x",
    "apple_y",
    "apple_z",
    "apple_vx",
    "apple_vy",
    "apple_vz",
    "harvest_fx",
    "harvest_fy",
    "harvest_fz",
)


@dataclasses.dataclass
class ZeroVicHoldConfig:
    """Configuration for settle→weld→zero-VIC hold."""

    seed: int = 42
    num_envs: int = 4
    ranges_path: Path | None = None
    ranges_override: dict | None = None
    env_spacing: tuple[float, float, float] = (2.0, 2.0, 2.0)
    device: str | None = None
    settle_substeps: int = 5000
    settle_max_speed: float = 0.05
    quiet_settle: bool = True
    duration: float = 30.0
    log_interval: float = 1.0
    hz: float = 30.0
    sim_substeps: int = 15
    enable_self_collision: bool = False
    stem_coupling_gain: float = DEFAULT_STEM_COUPLING_GAIN
    stem_force_cap_n: float = 200.0
    stem_torque_cap_nm: float = 50.0
    vic_linear_k: float = 600.0
    vic_linear_d: float = 200.0
    vic_angular_k: float = 20.0
    vic_angular_d: float = 4.0
    thresholds: StabilityThresholds = dataclasses.field(default_factory=StabilityThresholds)
    write_trajectory: bool = True
    print_settle_report: bool = True
    print_vic_summary: bool = True


@dataclasses.dataclass
class ZeroVicHoldResult:
    """Outcome of a zero-VIC hold trial."""

    config: ZeroVicHoldConfig
    time_series: list[dict[str, float | int]]
    settle_reports: list[SettleStabilityReport]
    ik_results: list[tuple[float, float, bool]]
    per_env_metrics: list[EnvStabilityMetrics]
    summary: HoldSummary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log poses every N seconds with zero VIC teleop velocity.",
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
        "--env-spacing",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[2.0, 2.0, 2.0],
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--settle-substeps", type=int, default=5000)
    parser.add_argument("--settle-max-speed", type=float, default=0.05)
    parser.add_argument("--quiet-settle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--duration", type=float, default=30.0, help="Sim duration [s].")
    parser.add_argument(
        "--log-interval",
        type=float,
        default=1.0,
        help="Sim-time interval between pose log rows [s].",
    )
    parser.add_argument("--hz", type=float, default=30.0, help="Teleop / log frame rate [Hz].")
    parser.add_argument("--sim-substeps", type=int, default=15)
    parser.add_argument(
        "--stem-coupling-gain",
        type=float,
        default=DEFAULT_STEM_COUPLING_GAIN,
        help="Stem harvest under-relaxation (production default: 1.0).",
    )
    parser.add_argument("--stem-force-cap-n", type=float, default=200.0)
    parser.add_argument("--stem-torque-cap-nm", type=float, default=50.0)
    parser.add_argument("--vic-linear-k", type=float, default=600.0)
    parser.add_argument("--vic-linear-d", type=float, default=200.0)
    parser.add_argument("--vic-angular-k", type=float, default=20.0)
    parser.add_argument("--vic-angular-d", type=float, default=4.0)
    parser.add_argument("--max-apple-drift-m", type=float, default=0.02)
    parser.add_argument("--max-apple-z-drop-m", type=float, default=0.015)
    parser.add_argument("--max-apple-path-m", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=str,
        default="-",
        help="CSV output path ('-' for stdout).",
    )
    parser.add_argument("--enable-self-collision", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ZeroVicHoldConfig:
    """Build :class:`ZeroVicHoldConfig` from CLI namespace."""
    ranges_path = Path(args.json) if args.json else default_ranges_fixture_path()
    thresholds = StabilityThresholds(
        max_apple_drift_m=float(args.max_apple_drift_m),
        max_apple_z_drop_m=float(args.max_apple_z_drop_m),
        max_apple_path_length_m=float(args.max_apple_path_m),
        max_harvest_force_n=float(args.stem_force_cap_n),
    )
    return ZeroVicHoldConfig(
        seed=int(args.seed),
        num_envs=int(args.num_envs),
        ranges_path=ranges_path,
        env_spacing=tuple(float(v) for v in args.env_spacing),
        device=getattr(args, "device", None),
        settle_substeps=int(args.settle_substeps),
        settle_max_speed=float(args.settle_max_speed),
        quiet_settle=bool(args.quiet_settle),
        duration=float(args.duration),
        log_interval=float(args.log_interval),
        hz=float(args.hz),
        sim_substeps=int(args.sim_substeps),
        enable_self_collision=bool(args.enable_self_collision),
        stem_coupling_gain=float(args.stem_coupling_gain),
        stem_force_cap_n=float(args.stem_force_cap_n),
        stem_torque_cap_nm=float(args.stem_torque_cap_nm),
        vic_linear_k=float(args.vic_linear_k),
        vic_linear_d=float(args.vic_linear_d),
        vic_angular_k=float(args.vic_angular_k),
        vic_angular_d=float(args.vic_angular_d),
        thresholds=thresholds,
        write_trajectory=True,
    )


def _gripper_for_robot(*, fix_to_apple: bool) -> GripperProxyConfig:
    return GripperProxyConfig(
        mass=fr3_robot.EE_MASS_KG,
        fix_to_apple=fix_to_apple,
        robot_facing_weld=fix_to_apple,
    )


def _configure_vic(scene: Any, *, gains: ImpedanceGains) -> fr3_robot.Fr3BatchedEEImpedanceController:
    from apple_pick_sim.coupled_fruiting.vic_joint_torques import _require_torch

    _require_torch()
    ik_kw = fr3_robot.batched_ik_teleop_kwargs(scene)
    if not ik_kw:
        raise RuntimeError("batched scene missing template IK layout (FR3 required for VIC)")
    scene.robot_kinematic_mode = False
    scene.vic_use_joint_torques = True
    vic = fr3_robot.Fr3BatchedEEImpedanceController(
        scene.robot_model,
        linear_speed=0.5,
        angular_speed=1.0,
        **ik_kw,
    )
    scene.vic_controller = vic
    scene.vic_gains = gains
    fr3_robot.init_mujoco_actuator_targets_from_model(scene.robot_model, scene.robot_control)
    fr3_robot.configure_vic_joint_torques_arm_batched(
        scene.robot_model,
        scene.robot_state_0,
        scene.robot_control,
        scene.mj_solver,
        scene=scene,
        layout=scene.layout,
    )
    scene.vic_joint_torques_configured = True
    vic.sync_target_from_state(scene.robot_state_0)
    vic.stage_targets_to_scene(scene)
    scene.vic_target_twist = _ZERO_VELOCITY
    return vic


def pose_row(
    *,
    t_s: float,
    env: int,
    layout: Any,
    scene: Any,
    vic: fr3_robot.Fr3BatchedEEImpedanceController,
) -> dict[str, float | int]:
    """One CSV/log row for env ``env`` at sim time ``t_s``."""
    tcp_idx = int(layout.tcp_body_indices[env])
    apple_idx = int(layout.apple_body_indices[env])

    robot_bq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)
    robot_bqd = scene.robot_state_0.body_qd.numpy().reshape(-1, 6)
    cable_bq = scene.cable.state_0.body_q.numpy().reshape(-1, 7)
    cable_bqd = scene.cable.state_0.body_qd.numpy().reshape(-1, 6)

    tcp = robot_bq[tcp_idx]
    tcp_qd = robot_bqd[tcp_idx]
    target_pos = vic._target_pos_wp.numpy()[env]
    pos_err = float(np.linalg.norm(target_pos - tcp[:3]))

    harvest = np.zeros(3, dtype=np.float64)
    if scene.proxy_forces is not None:
        w = scene.proxy_forces.numpy().reshape(-1, 6)[tcp_idx, :3]
        harvest = np.asarray(w, dtype=np.float64)

    apple_pos = np.zeros(3, dtype=np.float64)
    apple_vel = np.zeros(3, dtype=np.float64)
    if apple_idx >= 0:
        apple_pos = cable_bq[apple_idx, :3]
        apple_vel = cable_bqd[apple_idx, :3]

    return {
        "t_s": float(t_s),
        "env": int(env),
        "tcp_x": float(tcp[0]),
        "tcp_y": float(tcp[1]),
        "tcp_z": float(tcp[2]),
        "tcp_qx": float(tcp[3]),
        "tcp_qy": float(tcp[4]),
        "tcp_qz": float(tcp[5]),
        "tcp_qw": float(tcp[6]),
        "tcp_vx": float(tcp_qd[0]),
        "tcp_vy": float(tcp_qd[1]),
        "tcp_vz": float(tcp_qd[2]),
        "target_tcp_x": float(target_pos[0]),
        "target_tcp_y": float(target_pos[1]),
        "target_tcp_z": float(target_pos[2]),
        "pos_err_m": pos_err,
        "apple_x": float(apple_pos[0]),
        "apple_y": float(apple_pos[1]),
        "apple_z": float(apple_pos[2]),
        "apple_vx": float(apple_vel[0]),
        "apple_vy": float(apple_vel[1]),
        "apple_vz": float(apple_vel[2]),
        "harvest_fx": float(harvest[0]),
        "harvest_fy": float(harvest[1]),
        "harvest_fz": float(harvest[2]),
    }


def _build_kw(config: ZeroVicHoldConfig, *, device: str, mujoco_use_cpu: bool) -> dict[str, Any]:
    return dict(
        device=device,
        env_spacing=config.env_spacing,
        enable_self_collisions=config.enable_self_collision,
        mujoco_use_cpu=mujoco_use_cpu,
        mujoco_solver_kwargs={"disable_contacts": True, "use_mujoco_cpu": mujoco_use_cpu},
        stem_coupling_gain=float(config.stem_coupling_gain),
        stem_force_cap_N=float(config.stem_force_cap_n),
        stem_torque_cap_Nm=float(config.stem_torque_cap_nm),
    )


def run_zero_vic_hold(config: ZeroVicHoldConfig) -> ZeroVicHoldResult:
    """Settle→weld, run zero-VIC hold, return time series and stability metrics."""
    wp.init()
    device = resolve_sim_device(config.device)
    mujoco_use_cpu = resolve_mujoco_use_cpu(device)
    ranges_path = config.ranges_path or default_ranges_fixture_path()
    ranges = (
        config.ranges_override
        if config.ranges_override is not None
        else load_ranges(ranges_path)
    )
    num_envs = int(config.num_envs)
    per_env_params = sample_heterogeneous_params_list(
        ranges, topology_seed=int(config.seed), num_envs=num_envs
    )
    sim_substeps = int(config.sim_substeps)
    frame_dt = 1.0 / float(config.hz)
    sim_dt = (1.0 / 60.0) / sim_substeps
    log_interval = float(config.log_interval)
    duration = float(config.duration)
    build_kw = _build_kw(config, device=device, mujoco_use_cpu=mujoco_use_cpu)

    gripper_welded = _gripper_for_robot(fix_to_apple=True)
    gripper_free = dataclasses.replace(gripper_welded, fix_to_apple=False, robot_facing_weld=False)

    if config.print_settle_report:
        print(f"ranges: {ranges_path}", flush=True)
        print(f"device: {device}  num_envs: {num_envs}  seed: {config.seed}", flush=True)

    settled = build_heterogeneous_coupled_fruiting_fr3(
        ranges,
        per_env_params,
        **{**build_kw, "gripper_proxy": gripper_free, "vbd_only": True},
    )
    settle_vbd_substeps(settled, substeps=int(config.settle_substeps), dt=sim_dt)
    settle_reports = settle_stability_reports_from_cable(
        settled.cable,
        per_env_params,
        max_branch_speed_m_s=float(config.settle_max_speed),
    )
    if config.print_settle_report:
        print_settle_stability_report(settle_reports, verbose=True)
    if config.quiet_settle:
        quiet_all_cable_bodies(settled.cable)

    scene = build_heterogeneous_coupled_fruiting_fr3(
        ranges,
        per_env_params,
        **{
            **build_kw,
            "gripper_proxy": gripper_welded,
            "skip_ik_bootstrap": True,
            "defer_template_robot_bootstrap": True,
        },
    )
    seed_fix_to_apple_from_settled(
        welded_scene=scene,
        settled_scene=settled,
        quiet_apple_proxy=True,
        per_env_ik=True,
        per_world_proxy_offsets=getattr(scene, "per_world_proxy_offsets", None),
    )
    layout = scene.layout
    if layout is None:
        raise RuntimeError("expected batched layout on welded scene")

    ik_results: list[tuple[float, float, bool]] = list(
        getattr(scene, "settle_ik_envelope_results", None) or []
    )

    gains = ImpedanceGains(
        linear_k=float(config.vic_linear_k),
        linear_d=float(config.vic_linear_d),
        angular_k=float(config.vic_angular_k),
        angular_d=float(config.vic_angular_d),
    )
    vic = _configure_vic(scene, gains=gains)
    if config.print_settle_report:
        print(
            f"VIC zero-action hold: stem_gain={config.stem_coupling_gain:g} "
            f"K=({gains.linear_k:g}, {gains.angular_k:g}) "
            f"D=({gains.linear_d:g}, {gains.angular_d:g}); "
            f"log every {log_interval:g}s for {duration:g}s sim time",
            flush=True,
        )

    time_series: list[dict[str, float | int]] = []
    sim_time = 0.0
    next_log_t = 0.0
    while sim_time <= duration + 1e-9:
        for w in range(layout.num_envs):
            time_series.append(
                pose_row(t_s=sim_time, env=w, layout=layout, scene=scene, vic=vic)
            )

        if sim_time >= duration - 1e-9:
            break
        next_log_t += log_interval
        while sim_time < next_log_t - 1e-9 and sim_time < duration - 1e-9:
            scene.update_fr3_ee_teleop(frame_dt, vic, velocity=_ZERO_VELOCITY)
            for _ in range(sim_substeps):
                scene.coupled_substep(sim_dt)
            sim_time += frame_dt

    per_env_metrics = [
        compute_env_stability_metrics(
            time_series,
            env=w,
            duration_max=duration,
            thresholds=config.thresholds,
        )
        for w in range(layout.num_envs)
    ]
    settle_stable = [r.is_stable for r in settle_reports]
    ik_inside = [inside for _, _, inside in ik_results] if ik_results else None
    summary = summarize_hold_metrics(
        per_env_metrics,
        settle_stable=settle_stable if len(settle_stable) == len(per_env_metrics) else None,
        ik_inside=ik_inside,
    )

    if config.print_vic_summary:
        _print_vic_stability_summary(per_env_metrics, summary)

    return ZeroVicHoldResult(
        config=config,
        time_series=time_series if config.write_trajectory else [],
        settle_reports=settle_reports,
        ik_results=ik_results,
        per_env_metrics=per_env_metrics,
        summary=summary,
    )


def _print_vic_stability_summary(
    metrics: list[EnvStabilityMetrics],
    summary: HoldSummary,
) -> None:
    stable_count = sum(1 for m in metrics if m.is_stable)
    print(
        f"Zero-VIC hold stability (apple drift + secondary gates): "
        f"{stable_count}/{len(metrics)} envs stable "
        f"({100.0 * summary.vic_pass_rate:.1f}%)",
        flush=True,
    )
    for m in metrics:
        if m.is_stable:
            continue
        issue_text = ", ".join(m.issues) if m.issues else "unknown"
        print(
            f"  env{m.env}: UNSTABLE  drift={m.max_apple_drift_m:.4f} m  "
            f"sag={m.max_apple_z_drop_m:.4f} m  path={m.apple_path_length_m:.4f} m  "
            f"pos_err={m.max_pos_err_m:.4f} m  issues: {issue_text}",
            flush=True,
        )


def write_trajectory_csv(
    rows: list[dict[str, float | int]],
    output: TextIO | str,
) -> None:
    """Write pose time series to CSV."""
    if isinstance(output, str):
        if output == "-":
            writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            return
        with open(output, "w", newline="", encoding="utf-8") as fh:
            write_trajectory_csv(rows, fh)
        print(f"wrote {output}", flush=True)
        return
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def run(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    result = run_zero_vic_hold(config)
    output_path = str(args.output)
    if output_path != "-":
        write_trajectory_csv(result.time_series, output_path)
    elif config.write_trajectory:
        write_trajectory_csv(result.time_series, "-")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
