"""Heterogeneous batched coupled fruiting — per-env physics DR, vectorized GPU stepping.

Each env shares topology (``num_segments``) but has independently sampled stiffness,
damping, rod geometry, and apple size. Default path: settle all worlds in parallel,
then weld with per-env robot-facing grasp direction and per-env IK bootstrap.

Run from the repository root::

    uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_fruiting.py \\
      --num-envs 4 --viewer gl --seed 42 --mark-endpoints --tcp-force-arrow

Per-env scripted actions (RL scatter-path demo)::

    uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_fruiting.py \\
      --num-envs 4 --viewer gl --demo-per-env-actions --seed 42

Variable-impedance teleop (requires PyTorch; ``uv sync --extra vic``)::

    uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_fruiting.py \\
      --num-envs 4 --viewer gl --controller vic --seed 42

Default ranges: ``fruiting_system_ranges_real_world_proxy_variance.json`` (real-world
bench proxy with per-env DR). Stretch axial VBD knobs are fixed via per-segment
``vbd_stretch_fixed`` in that fixture; per-env variation is ``youngs_modulus_pa`` and
``damping_ratio`` (bend stiffness and bend damping). Robot base at origin; fruiting
chain at (0, 0.5, 0.95) m.

Cable-only stepping (no MuJoCo robot)::

    uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_fruiting.py \\
      --num-envs 4 --viewer gl --only-vbd --seed 42
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import secrets
import sys
import warnings
from pathlib import Path

import numpy as np
import newton
import warp as wp
import newton.examples

from apple_pick_sim.sim_device import resolve_sim_device
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.coupled_fruiting.settle_quasi_static import print_settle_stability_report
from apple_pick_sim.coupled_fruiting.settle_ke_decay import (
    DEFAULT_KE_ANALYSIS_TAIL_FRACTION,
    DEFAULT_KE_MIN_PEAKS,
    DEFAULT_KE_PEAK_DECAY_RTOL,
    DEFAULT_KE_SAMPLE_EVERY,
    SettleKeAnalysisConfig,
    SettleKeRecorder,
    per_env_branch_ke_j_from_cable,
    print_settle_checkpoint_report,
    print_settle_ke_decay_report,
)
from apple_pick_sim.coupled_fruiting.settle_then_weld import apply_settle_gravity_for_substep
from apple_pick_sim.coupled_fruiting import (
    CoupledFruitingScene,
    broadcast_joint_q_from_world0,
    build_heterogeneous_coupled_fruiting_fr3,
    build_heterogeneous_coupled_fruiting_placeholder,
    print_envelope_coverage_report,
    quiet_all_cable_bodies,
    seed_fix_to_apple_from_settled,
    settle_stability_reports_from_cable,
)

_SETTLE_REPORT_INTERVAL = 1000
# Sag under gravity can exceed straight rest sum; stable when path ≤ 1.05× nominal.
_SETTLE_PATH_MAX_OVER_NOMINAL = 1.05
_SETTLE_PATH_RTOL = _SETTLE_PATH_MAX_OVER_NOMINAL - 1.0
from apple_pick_sim.fruiting_system import (
    FruitingSystemParams,
    GripperProxyConfig,
    PLACEHOLDER_EE_MASS_KG,
    default_ranges_fixture_path,
    load_ranges,
    sample_heterogeneous_params_list,
    set_fruiting_joint_angular_kd_batched,
    set_fruiting_joint_angular_kp_batched,
)

# Real-world bench proxy EE: 50 mm radius, 140 mm length (docs/real-world-proxy.md).
from apple_pick_sim.batched_obs import gather_batched_obs, make_batched_obs_buffers
from apple_pick_sim.batched_viz import (
    log_batched_endpoints,
    log_batched_tcp_force_arrows,
    print_batched_obs_debug,
    print_mark_endpoints_startup,
)
from apple_pick_sim.coupled_fruiting.batched_robot_status import print_batched_robot_status


# Batched heterogeneous teleop: smaller steps than 1.0 m/s so template IK keeps up at 30 Hz.
_FR3_TELEOP_LINEAR_SPEED = 0.1
_FR3_TELEOP_ANGULAR_SPEED = 0.1
_FR3_TELEOP_IK_ITERATIONS = 128

# Per-role FIXED-joint angular kd overrides (see docs/damping-tuning.md §3)
# Later divided by dt
_DEFAULT_JOINT_ANGULAR_KD_OVERRIDES: dict[str, float] = {
    "support": 1.0,
    "primary_spur": 1.0,
    "stem_apple": 5e-2,
}

VIC_DEFAULT_LINEAR_K = 600.0
VIC_DEFAULT_LINEAR_D = 200.0
VIC_DEFAULT_ANGULAR_K = 20.0
VIC_DEFAULT_ANGULAR_D = 4.0


def _default_ranges_path() -> Path:
    return default_ranges_fixture_path()


def _fix_to_apple_from_args(args: argparse.Namespace | None) -> bool:
    return bool(getattr(args, "fix_to_apple", True)) if args else True


def _enable_self_collisions_from_args(args: argparse.Namespace | None) -> bool:
    return bool(getattr(args, "enable_self_collision", False)) if args else False


def _enable_apple_woody_collisions_from_args(args: argparse.Namespace | None) -> bool:
    return bool(getattr(args, "apple_woody_collision", True)) if args else True


def _enable_proxy_woody_collisions_from_args(args: argparse.Namespace | None) -> bool:
    return bool(getattr(args, "proxy_woody_collision", True)) if args else True


def _gripper_proxy_from_args(
    args: argparse.Namespace | None,
    *,
    robot_kind: str,
) -> GripperProxyConfig:
    fix = _fix_to_apple_from_args(args)
    if robot_kind == "fr3":
        return GripperProxyConfig(
            mass=PLACEHOLDER_EE_MASS_KG,
            fix_to_apple=fix,
            robot_facing_weld=fix,
        )
    return GripperProxyConfig(fix_to_apple=fix, robot_facing_weld=fix)


def _print_vbd_settle_start(
    *,
    substeps: int,
    sim_dt: float,
    gravity_ramp: bool,
) -> None:
    """Log settle plan to the terminal (including gravity-ramp mode)."""
    if substeps <= 0:
        return
    sim_time_s = int(substeps) * float(sim_dt)
    if gravity_ramp:
        print(
            f"VBD settle: {substeps} substeps ({sim_time_s:.3f} s sim), "
            "gravity ramp 0 → −9.81 m/s² over all substeps",
            flush=True,
        )
    else:
        print(
            f"VBD settle: {substeps} substeps ({sim_time_s:.3f} s sim), "
            "instant full gravity (ramp disabled)",
            flush=True,
        )


def _print_settle_checkpoint(
    scene: CoupledFruitingScene,
    per_env_params,
    *,
    substep_idx: int,
    sim_dt: float,
    settle_max_speed: float,
    brief: bool = False,
    include_ke: bool = True,
) -> None:
    """Log per-env settle stability (+ optional branch KE) at a substep boundary."""
    reports = settle_stability_reports_from_cable(
        scene.cable,
        per_env_params,
        max_branch_speed_m_s=float(settle_max_speed),
        path_rtol=_SETTLE_PATH_RTOL,
    )
    sim_time_s = int(substep_idx) * float(sim_dt)
    if include_ke:
        branch_ke_j = per_env_branch_ke_j_from_cable(scene.cable)
        print_settle_checkpoint_report(
            reports,
            branch_ke_j,
            substep_idx=substep_idx,
            sim_time_s=sim_time_s,
            prefix="  ",
            verbose=not brief,
        )
        return
    print(
        f"Settle checkpoint @ substep {substep_idx} ({sim_time_s:.3f} s sim):",
        flush=True,
    )
    print_settle_stability_report(reports, prefix="  ", verbose=not brief)


def _settle_ke_enabled(args: argparse.Namespace | None) -> bool:
    return bool(getattr(args, "settle_ke_decay", True)) if args else True


def _settle_ke_analysis_config(
    args: argparse.Namespace | None,
    *,
    settle_max_speed: float,
) -> SettleKeAnalysisConfig:
    threshold_raw = getattr(args, "ke_peak_threshold_j", None) if args else None
    return SettleKeAnalysisConfig(
        analysis_tail_fraction=float(
            getattr(args, "ke_analysis_tail_fraction", DEFAULT_KE_ANALYSIS_TAIL_FRACTION)
            if args
            else DEFAULT_KE_ANALYSIS_TAIL_FRACTION
        ),
        min_peaks=int(getattr(args, "ke_min_peaks", DEFAULT_KE_MIN_PEAKS) if args else DEFAULT_KE_MIN_PEAKS),
        peak_decay_rtol=float(
            getattr(args, "ke_peak_decay_rtol", DEFAULT_KE_PEAK_DECAY_RTOL)
            if args
            else DEFAULT_KE_PEAK_DECAY_RTOL
        ),
        speed_threshold_m_s=float(settle_max_speed),
        ke_peak_threshold_j=float(threshold_raw) if threshold_raw is not None else None,
    )


def _settle_ke_sample_every(args: argparse.Namespace | None) -> int:
    return int(getattr(args, "ke_sample_every", DEFAULT_KE_SAMPLE_EVERY) if args else DEFAULT_KE_SAMPLE_EVERY)


def _print_post_settle_diagnostics(
    *,
    stability_reports: list,
    ke_decay_reports: list,
    ik_results: list[tuple[float, float, bool]] | None = None,
    brief: bool = False,
) -> None:
    """Print instant-speed stability, optional KE decay, and IK envelope summary."""
    print_envelope_coverage_report(
        ik_results or [],
        stability_reports=stability_reports,
        verbose=not brief,
    )
    if ke_decay_reports:
        print_settle_ke_decay_report(ke_decay_reports, prefix="  ", verbose=not brief)


def _run_vbd_settle_on_scene(
    scene: CoupledFruitingScene,
    *,
    substeps: int,
    sim_dt: float,
    gravity_ramp: bool,
    per_env_params,
    settle_max_speed: float,
    report_brief: bool = False,
    num_envs: int = 1,
    ke_enabled: bool = True,
    ke_config: SettleKeAnalysisConfig | None = None,
    ke_sample_every: int = DEFAULT_KE_SAMPLE_EVERY,
) -> tuple[list, list]:
    """Run VBD settle (+ optional stability report) on an existing scene."""
    n = int(substeps)
    if n <= 0:
        return [], []
    _print_vbd_settle_start(substeps=n, sim_dt=sim_dt, gravity_ramp=gravity_ramp)
    h = float(sim_dt)
    analysis = ke_config if ke_config is not None else SettleKeAnalysisConfig(
        speed_threshold_m_s=float(settle_max_speed),
    )
    recorder = (
        SettleKeRecorder(num_envs=int(num_envs), sample_every=int(ke_sample_every))
        if ke_enabled
        else None
    )
    for substep_idx in range(n):
        apply_settle_gravity_for_substep(
            scene,
            substep_idx,
            n,
            gravity_ramp=gravity_ramp,
        )
        scene.vbd_substep(h)
        if recorder is not None:
            recorder.record_substep(
                scene.cable,
                per_env_params,
                substep_idx,
                h,
                sample_every=int(ke_sample_every),
            )
        completed = substep_idx + 1
        if completed % _SETTLE_REPORT_INTERVAL == 0:
            _print_settle_checkpoint(
                scene,
                per_env_params,
                substep_idx=completed,
                sim_dt=h,
                settle_max_speed=settle_max_speed,
                brief=report_brief,
                include_ke=ke_enabled,
            )
    stability_reports = settle_stability_reports_from_cable(
        scene.cable,
        per_env_params,
        max_branch_speed_m_s=float(settle_max_speed),
        path_rtol=_SETTLE_PATH_RTOL,
    )
    ke_decay_reports = recorder.reports(config=analysis) if recorder is not None else []
    quiet_all_cable_bodies(scene.cable)
    return stability_reports, ke_decay_reports


def _defer_settle_to_viewer(
    viewer,
    *,
    fix_to_apple: bool,
    settle_substeps: int,
) -> bool:
    """True when settle should animate in the GL viewer instead of during __init__."""
    if fix_to_apple or int(settle_substeps) <= 0:
        return False
    import newton

    return isinstance(viewer, newton.viewer.ViewerGL)


def _resolve_step_mode(args: argparse.Namespace | None) -> str:
    """Return ``"coupled"``, ``"vbd"``, or ``"mjc"`` from CLI flags."""
    only_vbd = bool(getattr(args, "only_vbd", False)) if args else False
    only_mjc = bool(getattr(args, "only_mjc", False)) if args else False
    if only_vbd and only_mjc:
        raise SystemExit("--only-vbd and --only-mjc are mutually exclusive")
    if only_mjc:
        raise SystemExit(
            "--only-mjc is not supported in the heterogeneous batched example "
            "(num_envs > 1 does not support mujoco_only)."
        )
    if only_vbd:
        return "vbd"
    return "coupled"


def _resolve_robot_kind(args: argparse.Namespace) -> str:
    robot_kind = str(getattr(args, "robot", "fr3"))
    if robot_kind == "fr3" and not fr3_robot.fr3_assets_available():
        print(
            "Warning: FR3 assets not found under assets/fr3/; falling back to placeholder TCP.",
            file=sys.stderr,
        )
        return "placeholder"
    return robot_kind


def _print_per_env_params(params_list: list[FruitingSystemParams]) -> None:
    print("Per-env fruiting params (topology shared, continuous θ differs):")
    for w, p in enumerate(params_list):
        print(f"  env{w}:")
        for seg_name in ("primary", "secondary", "spur", "stem"):
            rod = getattr(p, seg_name)
            if rod is None:
                continue
            print(
                f"    {seg_name}: E={rod.youngs_modulus_pa:.4g} Pa  "
                f"zeta={rod.damping_ratio:.4g}  "
                f"k_bend={rod.bend_stiffness:.4g} N·m/rad  "
                f"c_bend={rod.bend_damping:.4g} N·m·s/rad  "
                f"k_stretch={rod.stretch_stiffness:.4g} N/m  "
                f"c_stretch={rod.stretch_damping:.4g} N·s/m"
            )
        radius = float(p.apple_radius) if p.apple_radius is not None else float("nan")
        density = float(p.apple_density) if p.apple_density is not None else float("nan")
        print(f"    apple: r={radius:.4g} m  rho={density:.4g} kg/m³")


def _settle_inspect_continue_requested(
    viewer,
    *,
    graphical: bool,
    paused_before: bool,
) -> bool:
    """Return True when the user pressed SPACE to leave settled-scene inspection."""
    if not graphical:
        return False
    if hasattr(viewer, "is_key_down") and viewer.is_key_down("space"):
        return True
    if hasattr(viewer, "is_paused") and not paused_before and viewer.is_paused():
        return True
    return False


def _make_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help=(
            "Path to fruiting-system range JSON (default: "
            "apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json)."
        ),
    )
    parser.add_argument("--hz", type=float, default=30.0, help="Target frame rate [Hz].")
    parser.add_argument("--seed", type=int, default=None, help="Topology + DR seed.")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of coupled worlds.")
    parser.add_argument(
        "--env-spacing",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[2.0, 2.0, 2.0],
        help="Viewer grid spacing [m] (sim worlds are co-located).",
    )
    parser.add_argument("--enable-self-collision", action="store_true")
    parser.add_argument(
        "--apple-woody-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="AVBD contact between apple and woody segments (default: on).",
    )
    parser.add_argument(
        "--proxy-woody-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="AVBD contact between gripper proxy and woody segments (default: on).",
    )
    parser.add_argument(
        "--only-vbd",
        action="store_true",
        help="Cable SolverVBD only (gripper proxy at spawn; no MuJoCo robot).",
    )
    parser.add_argument(
        "--only-mjc",
        action="store_true",
        help="Not supported for heterogeneous batched builds (num_envs > 1).",
    )
    parser.add_argument("--robot", type=str, choices=("placeholder", "fr3"), default="fr3")
    parser.add_argument(
        "--controller",
        type=str,
        choices=("direct", "ee", "vic"),
        default="direct",
    )
    parser.add_argument("--fr3-keyboard", action="store_true")
    parser.add_argument(
        "--fix-to-apple",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Settle-then-weld (default: on). Use --no-fix-to-apple for velocity-delta harvest.",
    )
    parser.add_argument(
        "--settle-substeps",
        type=int,
        default=5000,
        help=(
            "VBD substeps before runtime (default: 5000). Runs in all modes when > 0 "
            "(settle-then-weld, --no-fix-to-apple, and --only-vbd)."
        ),
    )
    parser.add_argument(
        "--settle-gravity-ramp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Linear 0→−9.81 m/s² gravity ramp over all settle substeps (default: off).",
    )
    parser.add_argument(
        "--inspect-settle",
        action="store_true",
        help=(
            "After settling, render the free-proxy settled cable in the viewer and wait "
            "for SPACE before building the welded scene and starting teleop."
        ),
    )
    parser.add_argument(
        "--settle-report-brief",
        action="store_true",
        help="Only print per-env settle lines for unstable worlds (default: all envs).",
    )
    parser.add_argument(
        "--settle-max-speed",
        type=float,
        default=0.05,
        help="Residual branch speed threshold [m/s] for post-settle stability.",
    )
    parser.add_argument(
        "--settle-ke-decay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print branch KE envelope decay report after settle (default: on).",
    )
    parser.add_argument(
        "--ke-sample-every",
        type=int,
        default=DEFAULT_KE_SAMPLE_EVERY,
        help="Sample branch KE every N VBD substeps during settle.",
    )
    parser.add_argument(
        "--ke-analysis-tail-fraction",
        type=float,
        default=DEFAULT_KE_ANALYSIS_TAIL_FRACTION,
        help="Analyze KE decay over the last fraction of settle samples.",
    )
    parser.add_argument(
        "--ke-min-peaks",
        type=int,
        default=DEFAULT_KE_MIN_PEAKS,
        help="Minimum KE peaks required for envelope decay gate.",
    )
    parser.add_argument(
        "--ke-peak-decay-rtol",
        type=float,
        default=DEFAULT_KE_PEAK_DECAY_RTOL,
        help="Relative drop first→last KE peak required for decay gate.",
    )
    parser.add_argument(
        "--ke-peak-threshold-j",
        type=float,
        default=None,
        help="Override peak KE threshold [J]; default derives from branch mass and settle-max-speed.",
    )
    parser.add_argument(
        "--scripted-ee-vel",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "VZ"),
        default=[0.05, 0.0, 0.0],
    )
    parser.add_argument(
        "--demo-per-env-actions",
        action="store_true",
        help="Scale scripted TCP velocity per env to demonstrate per-arm IK scatter.",
    )
    parser.add_argument("--status-every", type=int, default=60)
    parser.add_argument("--print-robot-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--noisy-action",
        action="store_true",
        help="Independent Gaussian noise per env (FR3 only; uses per-env RNG).",
    )
    parser.add_argument("--noisy-action-std", type=float, default=0.02)
    parser.add_argument("--tcp-force-arrow", action="store_true")
    parser.add_argument("--tcp-force-scale", type=float, default=0.02)
    parser.add_argument("--tcp-force-arrow-gain", type=float, default=1.0)
    parser.add_argument("--tcp-force-min-length", type=float, default=0.08)
    parser.add_argument("--tcp-force-max-length", type=float, default=1.5)
    parser.add_argument(
        "--mark-endpoints",
        action="store_true",
        help=(
            "Debug endpoint viz + console obs: apple COM→grasp arrow; "
            "colored woody start points + junction force arrows; "
            "prints color→label map and gather_batched_obs on --status-every interval."
        ),
    )
    parser.add_argument("--mujoco-viewer", action="store_true")
    parser.add_argument(
        "--vic-linear-k", type=float, default=VIC_DEFAULT_LINEAR_K, help="VIC linear K [N/m]."
    )
    parser.add_argument(
        "--vic-linear-d", type=float, default=VIC_DEFAULT_LINEAR_D, help="VIC linear D [N·s/m]."
    )
    parser.add_argument(
        "--vic-angular-k",
        type=float,
        default=VIC_DEFAULT_ANGULAR_K,
        help="VIC angular K [N·m/rad].",
    )
    parser.add_argument(
        "--vic-angular-d",
        type=float,
        default=VIC_DEFAULT_ANGULAR_D,
        help="VIC angular D [N·m·s/rad].",
    )
    return parser


class ExampleBatchedHeterogeneousCoupledFruiting:
    """N heterogeneous coupled stacks; per-env DR at build, vectorized runtime stepping."""

    def __init__(self, viewer, args: argparse.Namespace | None = None):
        self.viewer = viewer
        self.args = args
        self._step_mode = _resolve_step_mode(args)

        self.fps = float(getattr(args, "hz", 30.0)) if args else 30.0
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 60
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self._frame = 0
        self._settled_scene: CoupledFruitingScene | None = None
        self._settle_stability_reports: list = []
        self._settle_ke_decay_reports: list = []
        self._settle_ik_envelope_results: list[tuple[float, float, bool]] = []
        self._pending_settle_substeps = 0
        self._pending_settle_gravity_ramp = False
        self._pending_settle_max_speed = 0.05
        self._pending_settle_ke_enabled = True

        json_path = getattr(args, "json", None) if args else None
        ranges_path = Path(json_path) if json_path else _default_ranges_path()
        self.ranges = load_ranges(ranges_path)

        seed = getattr(args, "seed", None) if args else None
        if seed is None:
            seed = secrets.randbelow(2**31 - 1)
        self._seed = int(seed)

        self.num_envs = int(getattr(args, "num_envs", 4))
        env_spacing_raw = getattr(args, "env_spacing", [2.0, 2.0, 2.0]) if args else [2.0, 2.0, 2.0]
        self.env_spacing = tuple(float(v) for v in env_spacing_raw)

        sim_device = resolve_sim_device(getattr(args, "device", None) if args else None)
        robot_kind = _resolve_robot_kind(args or argparse.Namespace())
        if robot_kind == "placeholder":
            warnings.warn(
                "Placeholder robot uses .numpy() host round-trips in simulate() "
                "(broadcast_joint_q_from_world0, _nudge_placeholder_world0); "
                "GPU parallelism is not fully utilized. "
                "Switch to --robot fr3 for a fully GPU hot path.",
                UserWarning,
                stacklevel=2,
            )
        controller_mode = str(getattr(args, "controller", "direct"))
        fix_to_apple = _fix_to_apple_from_args(args)
        enable_self = _enable_self_collisions_from_args(args)
        enable_apple_woody = _enable_apple_woody_collisions_from_args(args)
        enable_proxy_woody = _enable_proxy_woody_collisions_from_args(args)
        settle_substeps = int(getattr(args, "settle_substeps", 5000))
        settle_gravity_ramp = bool(getattr(args, "settle_gravity_ramp", False))
        settle_max_speed = float(getattr(args, "settle_max_speed", 0.05))
        settle_ke_enabled = _settle_ke_enabled(args)
        settle_ke_config = _settle_ke_analysis_config(args, settle_max_speed=settle_max_speed)
        settle_ke_sample_every = _settle_ke_sample_every(args)
        report_brief = bool(getattr(args, "settle_report_brief", False))
        defer_settle_to_viewer = _defer_settle_to_viewer(
            viewer,
            fix_to_apple=fix_to_apple,
            settle_substeps=settle_substeps,
        )

        self.per_env_params = sample_heterogeneous_params_list(
            self.ranges, topology_seed=self._seed, num_envs=self.num_envs
        )

        print(f"Heterogeneous batched fruiting ranges: {ranges_path}")
        print(f"Topology seed: {self._seed}")
        print(f"Warp device: {sim_device}")
        if self._step_mode == "coupled":
            label = "FR3+EE" if robot_kind == "fr3" else "placeholder TCP"
            print(
                f"M1 cable + {label} MuJoCo (staggered coupling); "
                "Newton viewer shows cable model."
            )
        else:
            print("Cable SolverVBD only (--only-vbd).")
        _print_per_env_params(self.per_env_params)
        coupling_label = (
            "stem-harvest / settle-then-weld"
            if fix_to_apple and self._step_mode != "vbd"
            else "velocity-delta"
            if not fix_to_apple
            else "ignored with --only-vbd"
        )
        print(
            f"Gripper proxy fix_to_apple={fix_to_apple} ({coupling_label} coupling)."
        )
        print(
            "AVBD cable collisions: "
            f"self_collision={enable_self} "
            f"apple↔woody={enable_apple_woody} "
            f"proxy↔woody={enable_proxy_woody}",
            flush=True,
        )

        build_fn = (
            build_heterogeneous_coupled_fruiting_fr3
            if robot_kind == "fr3"
            else build_heterogeneous_coupled_fruiting_placeholder
        )
        gripper = _gripper_proxy_from_args(args, robot_kind=robot_kind)
        build_kw = dict(
            device=sim_device,
            env_spacing=self.env_spacing,
            enable_self_collisions=enable_self,
            enable_apple_woody_collisions=enable_apple_woody,
            enable_proxy_woody_collisions=enable_proxy_woody,
            gripper_proxy=gripper,
            vbd_only=(self._step_mode == "vbd"),
        )
        if robot_kind == "fr3":
            fr3_robot.enable_ik_bootstrap_warnings_for_examples()

        if fix_to_apple and self._step_mode != "vbd":
            settled = build_fn(
                self.ranges,
                self.per_env_params,
                **{
                    **build_kw,
                    "vbd_only": True,
                    "gripper_proxy": dataclasses.replace(
                        gripper, fix_to_apple=False, robot_facing_weld=False
                    ),
                },
            )
            self._settle_stability_reports, self._settle_ke_decay_reports = _run_vbd_settle_on_scene(
                settled,
                substeps=settle_substeps,
                sim_dt=self.sim_dt,
                gravity_ramp=settle_gravity_ramp,
                per_env_params=self.per_env_params,
                settle_max_speed=settle_max_speed,
                report_brief=report_brief,
                num_envs=self.num_envs,
                ke_enabled=settle_ke_enabled,
                ke_config=settle_ke_config,
                ke_sample_every=settle_ke_sample_every,
            )
            if bool(getattr(args, "inspect_settle", False)):
                self._settled_scene = settled
            self.scene = build_fn(
                self.ranges,
                self.per_env_params,
                **{
                    **build_kw,
                    "gripper_proxy": dataclasses.replace(
                        gripper, fix_to_apple=True, robot_facing_weld=True
                    ),
                    "skip_ik_bootstrap": True,
                    "defer_template_robot_bootstrap": True,
                },
            )
            seed_fix_to_apple_from_settled(
                welded_scene=self.scene,
                settled_scene=settled,
                quiet_apple_proxy=True,
                per_env_ik=True,
                per_world_proxy_offsets=self.scene.per_world_proxy_offsets,
            )
            ik_results = getattr(self.scene, "settle_ik_envelope_results", None)
            self._settle_ik_envelope_results = ik_results or []
            _print_post_settle_diagnostics(
                stability_reports=self._settle_stability_reports,
                ke_decay_reports=self._settle_ke_decay_reports,
                ik_results=self._settle_ik_envelope_results,
                brief=report_brief,
            )
        else:
            self.scene: CoupledFruitingScene = build_fn(
                self.ranges, self.per_env_params, **build_kw
            )
            if settle_substeps > 0:
                if defer_settle_to_viewer:
                    self._pending_settle_substeps = settle_substeps
                    self._pending_settle_gravity_ramp = settle_gravity_ramp
                    self._pending_settle_max_speed = settle_max_speed
                    self._pending_settle_ke_enabled = settle_ke_enabled
                    print(
                        "Settle deferred to GL viewer (nominal geometry → sag under gravity ramp).",
                        flush=True,
                    )
                else:
                    self._settle_stability_reports, self._settle_ke_decay_reports = _run_vbd_settle_on_scene(
                        self.scene,
                        substeps=settle_substeps,
                        sim_dt=self.sim_dt,
                        gravity_ramp=settle_gravity_ramp,
                        per_env_params=self.per_env_params,
                        settle_max_speed=settle_max_speed,
                        report_brief=report_brief,
                        num_envs=self.num_envs,
                        ke_enabled=settle_ke_enabled,
                        ke_config=settle_ke_config,
                        ke_sample_every=settle_ke_sample_every,
                    )
                    _print_post_settle_diagnostics(
                        stability_reports=self._settle_stability_reports,
                        ke_decay_reports=self._settle_ke_decay_reports,
                        brief=report_brief,
                    )

        self.layout = self.scene.layout
        if self.layout is None:
            raise RuntimeError("batched scene missing layout")

        set_fruiting_joint_angular_kd_batched(
            self.scene.cable.solver,
            self.scene.cable.fruiting_fixed_joints,
            _DEFAULT_JOINT_ANGULAR_KD_OVERRIDES,
            num_envs=self.layout.num_envs,
            joints_per_world=self.layout.joints_per_world,
        )

        robot_world_count = (
            self.scene.robot_model.world_count
            if self.scene.robot_model is not None
            else 0
        )
        print(
            f"Heterogeneous batched fruiting: num_envs={self.layout.num_envs} "
            f"spacing={self.env_spacing} "
            f"cable_world_count={self.scene.cable.model.world_count} "
            f"robot_world_count={robot_world_count}"
        )

        self._controller_mode = controller_mode
        self._robot_kind = robot_kind
        self._ee_ctrl = None
        self._scripted_velocity = fr3_robot.EEVelocity(
            linear=tuple(float(v) for v in getattr(args, "scripted_ee_vel", [0.05, 0.0, 0.0])),
        )
        self._teleop_velocity = self._scripted_velocity
        self._use_keyboard = bool(getattr(args, "fr3_keyboard", False))
        self._noisy_action_std = float(getattr(args, "noisy_action_std", 0.02))
        self._per_env_noise_rng = [
            np.random.default_rng(self._seed + 1000 + w) for w in range(self.num_envs)
        ]
        noisy_requested = bool(getattr(args, "noisy_action", False))
        if noisy_requested and robot_kind != "fr3":
            print("Warning: --noisy-action requires FR3; ignoring.", file=sys.stderr)
        self._noisy_action = noisy_requested and robot_kind == "fr3"
        self._demo_per_env_actions = bool(getattr(args, "demo_per_env_actions", False))

        if robot_kind == "fr3" and self.scene.robot_model is not None and self._step_mode != "vbd":
            self._ee_ctrl = self._configure_fr3_controller(controller_mode)
        elif controller_mode == "ee":
            print("Note: --controller ee requires FR3; running without teleop.", file=sys.stderr)
        elif self._step_mode == "vbd" and controller_mode != "direct":
            print(
                "Note: --only-vbd skips robot teleop; --controller is ignored.",
                file=sys.stderr,
            )

        if self._use_keyboard and robot_kind == "fr3" and hasattr(self.viewer, "is_key_down"):
            fr3_robot.print_fr3_keyboard_bindings()

        self._status_every = int(getattr(args, "status_every", 60))
        self._print_robot_state = bool(getattr(args, "print_robot_state", True))

        if self._print_robot_state and self.scene.robot_model is not None and self.layout is not None:
            if self.scene.mj_solver is not None:
                fr3_robot.sync_mujoco_visual_state(
                    self.scene.mj_solver,
                    self.scene.robot_model,
                    self.scene.robot_state_0,
                )
            print("Initial batched robot state (post-build):", flush=True)
            print_batched_robot_status(self.scene, self.layout, prefix="")

        self._tcp_force_arrow = bool(getattr(args, "tcp_force_arrow", False))
        self._tcp_force_scale = float(getattr(args, "tcp_force_scale", 0.02))
        self._tcp_force_gain = float(getattr(args, "tcp_force_arrow_gain", 1.0))
        self._tcp_force_min_length = float(getattr(args, "tcp_force_min_length", 0.08))
        self._tcp_force_max_length = float(getattr(args, "tcp_force_max_length", 1.5))
        self._mark_endpoints = bool(getattr(args, "mark_endpoints", False))

        need_obs_bufs = self._tcp_force_arrow or self._mark_endpoints
        if need_obs_bufs and self.layout is not None:
            self._obs_bufs = make_batched_obs_buffers(
                self.layout,
                self.scene.cable,
                str(self.scene.cable.model.device),
            )
        else:
            self._obs_bufs = None

        self.viewer.set_model(self.scene.cable.model)
        graphical = isinstance(viewer, newton.viewer.ViewerGL)
        self._mujoco_viewer = (
            self.scene.robot_model is not None
            and bool(getattr(args, "mujoco_viewer", False))
            and graphical
        )
        if self._tcp_force_arrow and self._step_mode != "coupled":
            print(
                "Note: --tcp-force-arrow needs full coupled stepping "
                "(omit --only-vbd).",
                flush=True,
            )
        elif self._tcp_force_arrow and graphical:
            cap = (
                f"{self._tcp_force_max_length:.2f} m max"
                if self._tcp_force_max_length > 0.0
                else "no max"
            )
            print(
                "TCP force arrows: yellow at each env's robot TCP; "
                f"scale={self._tcp_force_scale:.4f} m/N × gain {self._tcp_force_gain:g}, "
                f"min {self._tcp_force_min_length:.2f} m, {cap}.",
                flush=True,
            )
        if self._mark_endpoints:
            print_mark_endpoints_startup(self.scene.cable, status_every=self._status_every)
        if graphical and self.num_envs > 1:
            self.viewer.set_world_offsets(self.env_spacing)

        self._viz_contacts = self.scene.cable.model.collide(
            self.scene.cable.state_0,
            collision_pipeline=self.scene.cable_collision_pipeline,
        )

    def _configure_fr3_controller(self, mode: str):
        ik_kw = fr3_robot.batched_ik_teleop_kwargs(self.scene)
        if not ik_kw:
            raise RuntimeError("batched FR3 scene missing template IK layout")
        use_per_env = self._demo_per_env_actions or self._noisy_action
        velocity_for_world = self._velocity_for_world if use_per_env else (lambda w: self._teleop_velocity)  # noqa: ARG005
        if self._demo_per_env_actions:
            print("Heterogeneous demo: per-env scripted velocity scales (IK scatter).")
        if self._noisy_action:
            print(
                f"Heterogeneous noisy teleop: per-env Gaussian std={self._noisy_action_std}."
            )
        if mode == "ee":
            self.scene.robot_kinematic_mode = False
            ctrl = fr3_robot.Fr3BatchedEEVelocityController(
                self.scene.robot_model,
                linear_speed=_FR3_TELEOP_LINEAR_SPEED,
                angular_speed=_FR3_TELEOP_ANGULAR_SPEED,
                ik_iterations=_FR3_TELEOP_IK_ITERATIONS,
                velocity_for_world=velocity_for_world,
                **ik_kw,
            )
        elif mode == "vic":
            return self._configure_fr3_vic(ik_kw, velocity_for_world)
        else:
            self.scene.robot_kinematic_mode = True
            ctrl = fr3_robot.Fr3BatchedEEDirectJointController(
                self.scene.robot_model,
                linear_speed=_FR3_TELEOP_LINEAR_SPEED,
                angular_speed=_FR3_TELEOP_ANGULAR_SPEED,
                ik_iterations=_FR3_TELEOP_IK_ITERATIONS,
                velocity_for_world=velocity_for_world,
                **ik_kw,
            )
        ctrl.sync_target_from_state(self.scene.robot_state_0)
        return ctrl

    def _configure_fr3_vic(self, ik_kw: dict, velocity_for_world):
        from apple_pick_sim.coupled_fruiting.vic_joint_torques import _require_torch

        _require_torch()
        self.scene.robot_kinematic_mode = False
        fr3_robot.init_mujoco_actuator_targets_from_model(
            self.scene.robot_model, self.scene.robot_control
        )
        print("Batched FR3 dynamic-arm mode: MuJoCo integrates lagged plant wrenches on TCP body_f.")
        self.scene.vic_use_joint_torques = True
        vic = fr3_robot.Fr3BatchedEEImpedanceController(
            self.scene.robot_model,
            linear_speed=_FR3_TELEOP_LINEAR_SPEED,
            angular_speed=_FR3_TELEOP_ANGULAR_SPEED,
            velocity_for_world=velocity_for_world,
            **ik_kw,
        )
        self.scene.vic_controller = vic
        self.scene.vic_gains = fr3_robot.ImpedanceGains(
            linear_k=float(getattr(self.args, "vic_linear_k", VIC_DEFAULT_LINEAR_K)),
            linear_d=float(getattr(self.args, "vic_linear_d", VIC_DEFAULT_LINEAR_D)),
            angular_k=float(getattr(self.args, "vic_angular_k", VIC_DEFAULT_ANGULAR_K)),
            angular_d=float(getattr(self.args, "vic_angular_d", VIC_DEFAULT_ANGULAR_D)),
        )
        fr3_robot.configure_vic_joint_torques_arm_batched(
            self.scene.robot_model,
            self.scene.robot_state_0,
            self.scene.robot_control,
            self.scene.mj_solver,
            scene=self.scene,
            layout=self.scene.layout,
        )
        self.scene.vic_joint_torques_configured = True
        vic.sync_target_from_state(self.scene.robot_state_0)
        vic.stage_targets_to_scene(self.scene)
        self.scene.vic_target_twist = fr3_robot.EEVelocity()
        g = self.scene.vic_gains
        print(
            f"Batched VIC enabled (joint torques, joint PD off): "
            f"K=({g.linear_k:g}, {g.angular_k:g}) "
            f"D=({g.linear_d:g}, {g.angular_d:g}) N/m, N·m/rad."
        )
        return vic

    def _velocity_for_world(self, world: int) -> fr3_robot.EEVelocity:
        if self._demo_per_env_actions:
            base = self._scripted_velocity if not self._use_keyboard else self._teleop_velocity
            angle = 2.0 * math.pi * world / max(self.num_envs, 1)
            speed = float(base.linear[0]) * (1.0 + 0.2 * world)
            return fr3_robot.EEVelocity(
                linear=(speed * math.cos(angle), speed * math.sin(angle), float(base.linear[2])),
                angular=base.angular,
            )
        if self._noisy_action:
            return fr3_robot.add_gaussian_noise_to_ee_velocity(
                self._teleop_velocity,
                rng=self._per_env_noise_rng[world],
                std=self._noisy_action_std,
            )
        return self._teleop_velocity

    def _teleop_world0(self) -> None:
        if self._robot_kind == "fr3" and self._ee_ctrl is not None:
            velocity = self._scripted_velocity
            if self._use_keyboard and hasattr(self.viewer, "is_key_down"):
                velocity = fr3_robot.read_keyboard_ee_velocity(
                    self.viewer,
                    linear_speed=self._ee_ctrl.linear_speed,
                    angular_speed=self._ee_ctrl.angular_speed,
                )
            self._teleop_velocity = velocity
            if self._controller_mode == "direct":
                self.scene.update_fr3_ee_teleop_direct(
                    self.frame_dt,
                    self._ee_ctrl,
                    viewer=self.viewer if self._use_keyboard else None,
                    velocity=velocity,
                )
            else:
                self.scene.update_fr3_ee_teleop(
                    self.frame_dt,
                    self._ee_ctrl,
                    viewer=self.viewer if self._use_keyboard else None,
                    velocity=velocity,
                )
        elif self._robot_kind == "placeholder":
            self._nudge_placeholder_world0()

    def _nudge_placeholder_world0(self) -> None:
        layout = self.layout
        assert layout is not None
        model = self.scene.robot_model
        if model is None or self.scene.robot_state_0 is None:
            return
        vx = float(self._scripted_velocity.linear[0])
        jq = model.joint_q.numpy().copy()
        sl = layout.joint_q_slice(0)
        jq[sl][0] += vx * self.frame_dt
        model.joint_q.assign(jq)
        self.scene.robot_state_0.joint_q.assign(jq)

    def simulate(self) -> None:
        if self._step_mode != "vbd":
            self._teleop_world0()
            if self._robot_kind == "placeholder":
                broadcast_joint_q_from_world0(self.scene, self.layout)
        for _ in range(self.sim_substeps):
            if self._step_mode == "vbd":
                self.scene.vbd_substep(self.sim_dt)
            else:
                self.scene.coupled_substep(self.sim_dt)

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt
        self._frame += 1
        if self._status_every > 0 and self._frame % self._status_every == 0:
            self._print_status()

    def _print_status(self) -> None:
        layout = self.layout
        if layout is None:
            return
        body_q = self.scene.cable.state_0.body_q.numpy().reshape(-1, 7)
        apple0 = layout.apple_body_indices[0]
        if apple0 >= 0:
            z0 = float(body_q[apple0, 2])
            print(f"t={self.sim_time:6.2f}s world-0 apple z={z0:.4f}", flush=True)
        if self.scene.robot_model is None:
            return
        for w in range(layout.num_envs):
            vel = self._command_velocity_for_world(w)
            tx, ty, tz = self._tcp_world_position(w)
            lx, ly, lz = vel.linear
            print(
                f"  env{w}: v=({lx:+.3f},{ly:+.3f},{lz:+.3f}) m/s  "
                f"tcp=({tx:.3f},{ty:.3f},{tz:.3f}) m",
                flush=True,
            )
        if self._mark_endpoints and self._obs_bufs is not None:
            gather_batched_obs(
                self._obs_bufs,
                self.scene,
                self.sim_dt,
                include_robot=self.scene.robot_state_0 is not None,
                include_forces=self.scene.proxy_forces is not None,
            )
            print_batched_obs_debug(
                self._obs_bufs,
                frame=self._frame,
                sim_time=self.sim_time,
                cable=self.scene.cable,
            )

    def _command_velocity_for_world(self, world: int) -> fr3_robot.EEVelocity:
        if self._ee_ctrl is not None:
            return self._ee_ctrl.command_velocity_for_world(
                world, fallback=self._teleop_velocity
            )
        return self._teleop_velocity

    def _tcp_world_position(self, world: int) -> tuple[float, float, float]:
        layout = self.layout
        assert layout is not None
        if self._ee_ctrl is not None:
            pos = wp.transform_get_translation(self._ee_ctrl.tcp_world_pose(world))
            return float(pos[0]), float(pos[1]), float(pos[2])
        state = self.scene.robot_state_0
        if state is None:
            return (0.0, 0.0, 0.0)
        tcp_idx = layout.tcp_body_indices[world]
        bq = state.body_q.numpy().reshape(-1, 7)[tcp_idx]
        return float(bq[0]), float(bq[1]), float(bq[2])

    def render(self) -> None:
        if self.scene.last_vbd_contacts is not None:
            self._viz_contacts = self.scene.last_vbd_contacts
        else:
            self._viz_contacts = self.scene.cable.model.collide(
                self.scene.cable.state_0,
                collision_pipeline=self.scene.cable_collision_pipeline,
            )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.scene.cable.state_0)
        self.viewer.log_contacts(self._viz_contacts, self.scene.cable.state_0)
        if self._obs_bufs is not None:
            gather_batched_obs(
                self._obs_bufs,
                self.scene,
                self.sim_dt,
                include_robot=self.scene.robot_state_0 is not None,
                include_forces=self.scene.proxy_forces is not None,
            )
        if self._tcp_force_arrow:
            log_batched_tcp_force_arrows(
                self.viewer,
                self.scene,
                self.layout,
                scale_per_newton=self._tcp_force_scale,
                gain=self._tcp_force_gain,
                min_length=self._tcp_force_min_length,
                max_length=self._tcp_force_max_length,
                bufs=self._obs_bufs,
            )
        if self._mark_endpoints:
            log_batched_endpoints(
                self.viewer,
                self.scene,
                self.layout,
                bufs=self._obs_bufs,
                woody_force_scale=self._tcp_force_scale,
                woody_force_gain=self._tcp_force_gain,
                woody_force_min_length=self._tcp_force_min_length,
                woody_force_max_length=self._tcp_force_max_length,
            )
        self.viewer.end_frame()
        if self._mujoco_viewer and self.scene.mj_solver is not None:
            fr3_robot.sync_mujoco_visual_state(
                self.scene.mj_solver,
                self.scene.robot_model,
                self.scene.robot_state_0,
            )
            self.scene.mj_solver.render_mujoco_viewer()

    def cleanup(self) -> None:
        if self._mujoco_viewer and self.scene.mj_solver is not None:
            self.scene.mj_solver.close_mujoco_viewer()

    def run_visible_settle(self) -> None:
        """Animate VBD settle in the GL viewer (``--no-fix-to-apple`` path)."""
        import time

        n = int(self._pending_settle_substeps)
        if n <= 0:
            return
        _print_vbd_settle_start(
            substeps=n,
            sim_dt=self.sim_dt,
            gravity_ramp=self._pending_settle_gravity_ramp,
        )
        print(
            "Settle animation: cable sag under gravity in the viewer, then teleop starts.",
            flush=True,
        )
        brief = bool(getattr(self.args, "settle_report_brief", False)) if self.args else False
        ke_enabled = bool(self._pending_settle_ke_enabled)
        settle_max_speed = float(self._pending_settle_max_speed)
        ke_config = _settle_ke_analysis_config(
            self.args,
            settle_max_speed=settle_max_speed,
        )
        ke_sample_every = _settle_ke_sample_every(self.args)
        recorder = (
            SettleKeRecorder(num_envs=int(self.num_envs), sample_every=int(ke_sample_every))
            if ke_enabled
            else None
        )
        substep_idx = 0
        while substep_idx < n and self.viewer.is_running():
            for _ in range(self.sim_substeps):
                if substep_idx >= n:
                    break
                apply_settle_gravity_for_substep(
                    self.scene,
                    substep_idx,
                    n,
                    gravity_ramp=self._pending_settle_gravity_ramp,
                )
                self.scene.vbd_substep(self.sim_dt)
                if recorder is not None:
                    recorder.record_substep(
                        self.scene.cable,
                        self.per_env_params,
                        substep_idx,
                        self.sim_dt,
                        sample_every=int(ke_sample_every),
                    )
                substep_idx += 1
                if substep_idx % _SETTLE_REPORT_INTERVAL == 0:
                    _print_settle_checkpoint(
                        self.scene,
                        self.per_env_params,
                        substep_idx=substep_idx,
                        sim_dt=self.sim_dt,
                        settle_max_speed=settle_max_speed,
                        brief=brief,
                        include_ke=ke_enabled,
                    )
            self.sim_time = substep_idx * self.sim_dt
            self.render()
            time.sleep(max(0.0, self.frame_dt))

        self._pending_settle_substeps = 0
        self._settle_stability_reports = settle_stability_reports_from_cable(
            self.scene.cable,
            self.per_env_params,
            max_branch_speed_m_s=settle_max_speed,
            path_rtol=_SETTLE_PATH_RTOL,
        )
        self._settle_ke_decay_reports = (
            recorder.reports(config=ke_config) if recorder is not None else []
        )
        quiet_all_cable_bodies(self.scene.cable)
        _print_post_settle_diagnostics(
            stability_reports=self._settle_stability_reports,
            ke_decay_reports=self._settle_ke_decay_reports,
            brief=brief,
        )

    def inspect_settled_scene(self) -> None:
        """Render settled free-proxy cable until SPACE (GL viewer) or viewer closes."""
        settled = self._settled_scene
        if settled is None:
            return
        import time

        import newton

        cable = settled.cable
        self.viewer.set_model(cable.model)
        graphical = isinstance(self.viewer, newton.viewer.ViewerGL)
        if graphical and self.num_envs > 1:
            self.viewer.set_world_offsets(self.env_spacing)
        print(
            "Settled-scene inspection: review stability report above; "
            "press SPACE to continue to welded build + teleop.",
            flush=True,
        )
        paused_before = (
            graphical and hasattr(self.viewer, "is_paused") and self.viewer.is_paused()
        )
        while self.viewer.is_running():
            viz_contacts = cable.model.collide(
                cable.state_0,
                collision_pipeline=settled.cable_collision_pipeline,
            )
            self.viewer.begin_frame(0.0)
            self.viewer.log_state(cable.state_0)
            self.viewer.log_contacts(viz_contacts, cable.state_0)
            if self._mark_endpoints and self.layout is not None:
                inspect_bufs = self._obs_bufs
                if inspect_bufs is not None:
                    gather_batched_obs(
                        inspect_bufs,
                        settled,
                        self.sim_dt,
                        include_robot=False,
                        include_forces=False,
                    )
                log_batched_endpoints(
                    self.viewer,
                    settled,
                    self.layout,
                    bufs=inspect_bufs,
                    woody_force_scale=self._tcp_force_scale,
                    woody_force_gain=self._tcp_force_gain,
                    woody_force_min_length=self._tcp_force_min_length,
                    woody_force_max_length=self._tcp_force_max_length,
                )
            self.viewer.end_frame()
            if _settle_inspect_continue_requested(
                self.viewer, graphical=graphical, paused_before=paused_before
            ):
                if hasattr(self.viewer, "_paused"):
                    self.viewer._paused = False
                break
            time.sleep(max(0.0, self.frame_dt))
        self.viewer.set_model(self.scene.cable.model)
        if graphical and self.num_envs > 1:
            self.viewer.set_world_offsets(self.env_spacing)

    def test_final(self, tolerance: float = 0.05) -> None:
        layout = self.layout
        if layout is None:
            return
        body_q = self.scene.cable.state_0.body_q.numpy().reshape(-1, 7)
        for w, apple_idx in enumerate(layout.apple_body_indices):
            if apple_idx < 0:
                continue
            z = float(body_q[apple_idx, 2])
            assert z > -tolerance, f"world {w} apple fell: z={z}"
        if self._settle_stability_reports or self._settle_ke_decay_reports or self._settle_ik_envelope_results:
            _print_post_settle_diagnostics(
                stability_reports=self._settle_stability_reports,
                ke_decay_reports=self._settle_ke_decay_reports,
                ik_results=self._settle_ik_envelope_results,
            )


if __name__ == "__main__":
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--num-frames", "200"])
            print(
                "No DISPLAY/WAYLAND_DISPLAY: using --viewer null --num-frames 200."
            )

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)
    example = ExampleBatchedHeterogeneousCoupledFruiting(viewer, args)

    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    if example._pending_settle_substeps > 0:
        example.run_visible_settle()

    if bool(getattr(args, "inspect_settle", False)):
        example.inspect_settled_scene()

    print("Starting heterogeneous batched coupled simulation…")
    try:
        while viewer.is_running():
            example.step()
            example.render()
            import time

            time.sleep(max(0.0, example.frame_dt))
    finally:
        example.cleanup()

    example.test_final()
