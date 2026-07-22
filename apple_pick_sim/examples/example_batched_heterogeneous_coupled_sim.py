"""Primary entry point — thin CLI + viewer for ``BatchedHeterogeneousCoupledSim`` (V.3.2).

Sim build (VIC gains, settle substeps, joint kd overrides, control_hz, …) is configured via module
constants in this file, not CLI flags.

Run from the repository root::

    uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py \\
      --num-envs 4 --viewer gl --seed 42 --mark-endpoints --tcp-force-arrow

Headless smoke (no DISPLAY)::

    uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py \\
      --viewer null --num-frames 10 --settle-substeps 50 --num-envs 2
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Literal

import newton
import newton.examples
import torch

from apple_pick_sim.batched_viz import (
    log_batched_endpoints,
    log_batched_tcp_force_arrows,
    print_batched_obs_debug,
    print_mark_endpoints_startup,
)
from apple_pick_sim.coupled_fruiting import print_envelope_coverage_report
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    EXAMPLE_JOINT_ANGULAR_KD_OVERRIDES,
    EXAMPLE_JOINT_ANGULAR_KP_OVERRIDES,
    EXAMPLE_JOINT_LINEAR_KD_OVERRIDES,
    EXAMPLE_JOINT_LINEAR_KP_OVERRIDES,
    ObsConfig,
    RuntimeConfig,
    SettleDiagnosticsConfig,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_build import print_per_env_params
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_coupled_sim import (
    BatchedHeterogeneousCoupledSim,
)
from apple_pick_sim.coupled_fruiting.batched_robot_status import print_batched_robot_status
from apple_pick_sim.fruiting_system import (
    FruitingSystemParams,
    default_ranges_fixture_path,
    load_ranges,
    parse_sim_build,
    sample_heterogeneous_params_list,
)
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import ImpedanceGains
from apple_pick_sim.sim_device import resolve_sim_device

# Fallbacks when ranges JSON has no ``sim_build`` (match variance fixture values).
_VIC_DEFAULT_LINEAR_K = 200.0
_VIC_DEFAULT_LINEAR_D = 10.0
_VIC_DEFAULT_ANGULAR_K = 10.0
_VIC_DEFAULT_ANGULAR_D = 1.0
_PHYSICS_SUB_DT = 1.0 / 1800.0

# --- Sim build fallbacks (used when ranges omit ``sim_build``) ---
JOINT_ANGULAR_KD_OVERRIDES = EXAMPLE_JOINT_ANGULAR_KD_OVERRIDES
JOINT_LINEAR_KD_OVERRIDES = EXAMPLE_JOINT_LINEAR_KD_OVERRIDES
JOINT_ANGULAR_KP_OVERRIDES = EXAMPLE_JOINT_ANGULAR_KP_OVERRIDES
JOINT_LINEAR_KP_OVERRIDES = EXAMPLE_JOINT_LINEAR_KP_OVERRIDES


def _resolve_sim_build_knobs(ranges: dict) -> tuple[
    ImpedanceGains,
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    float | None,
]:
    sb = parse_sim_build(ranges)
    if sb is None:
        return (
            ImpedanceGains(
                linear_k=_VIC_DEFAULT_LINEAR_K,
                linear_d=_VIC_DEFAULT_LINEAR_D,
                angular_k=_VIC_DEFAULT_ANGULAR_K,
                angular_d=_VIC_DEFAULT_ANGULAR_D,
            ),
            dict(JOINT_ANGULAR_KD_OVERRIDES),
            dict(JOINT_LINEAR_KD_OVERRIDES),
            dict(JOINT_ANGULAR_KP_OVERRIDES),
            dict(JOINT_LINEAR_KP_OVERRIDES),
            None,
        )
    return (
        ImpedanceGains(
            linear_k=sb.vic_gains.linear_k,
            linear_d=sb.vic_gains.linear_d,
            angular_k=sb.vic_gains.angular_k,
            angular_d=sb.vic_gains.angular_d,
        ),
        dict(sb.joint_angular_kd_overrides),
        dict(sb.joint_linear_kd_overrides),
        dict(sb.joint_angular_kp_overrides),
        dict(sb.joint_linear_kp_overrides),
        sb.joint_damping_ratio,
    )


@dataclasses.dataclass(frozen=True)
class VizSettings:
    """Viewer debug overlays (--tcp-force-arrow, --mark-endpoints)."""

    tcp_force_arrow: bool = False
    tcp_force_scale: float = 0.02
    tcp_force_gain: float = 1.0
    tcp_force_min_length: float = 0.08
    tcp_force_max_length: float = 1.5
    mark_endpoints: bool = False

    @property
    def needs_obs_buffers(self) -> bool:
        return self.tcp_force_arrow or self.mark_endpoints


def _validate_tcp_force_viz_args(args: argparse.Namespace) -> None:
    scale = float(args.tcp_force_scale)
    gain = float(args.tcp_force_arrow_gain)
    min_len = float(args.tcp_force_min_length)
    max_len = float(args.tcp_force_max_length)
    if scale <= 0.0:
        raise ValueError("--tcp-force-scale must be positive")
    if gain <= 0.0:
        raise ValueError("--tcp-force-arrow-gain must be positive")
    if min_len < 0.0:
        raise ValueError("--tcp-force-min-length must be >= 0")
    if max_len < 0.0:
        raise ValueError("--tcp-force-max-length must be >= 0")


def _viz_settings_from_args(args: argparse.Namespace) -> VizSettings:
    _validate_tcp_force_viz_args(args)
    return VizSettings(
        tcp_force_arrow=bool(args.tcp_force_arrow),
        tcp_force_scale=float(args.tcp_force_scale),
        tcp_force_gain=float(args.tcp_force_arrow_gain),
        tcp_force_min_length=float(args.tcp_force_min_length),
        tcp_force_max_length=float(args.tcp_force_max_length),
        mark_endpoints=bool(args.mark_endpoints),
    )


def _resolve_step_mode(args: argparse.Namespace) -> Literal["coupled", "vbd"]:
    if args.only_vbd and args.only_mjc:
        raise SystemExit("--only-vbd and --only-mjc are mutually exclusive")
    if args.only_mjc:
        raise SystemExit(
            "--only-mjc is not supported for heterogeneous batched builds "
            "(num_envs > 1 does not support mujoco_only)."
        )
    if args.only_vbd:
        return "vbd"
    return "coupled"


def _require_fr3_assets() -> None:
    if not fr3_robot.fr3_assets_available():
        raise SystemExit(
            "Bundled FR3 assets missing under assets/fr3/; "
            "see assets/fr3/README.md and install usd-core."
        )


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
    parser.add_argument(
        "--controller",
        type=str,
        choices=("direct", "ee", "vic"),
        default="vic",
        help=(
            "FR3 teleop: vic (default, joint-torque impedance), "
            "ee (velocity IK + MuJoCo PD), or direct (kinematic joint_q)."
        ),
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
        help="VBD substeps before runtime (default: 5000).",
    )
    parser.add_argument(
        "--settle-gravity-ramp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Linear 0→−9.81 m/s² gravity ramp over all settle substeps (default: off).",
    )
    parser.add_argument(
        "--settle-quiet-every",
        type=int,
        default=300,
        metavar="N",
        help=(
            "Zero all fruiting-system body twists every N VBD settle substeps "
            "(device-side; default: 300)."
        ),
    )
    parser.add_argument(
        "--use-settle-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse on-disk settled body_q when config matches (default: off).",
    )
    parser.add_argument(
        "--force-settle",
        action="store_true",
        help="Always run full VBD settle even when a matching cache file exists.",
    )
    parser.add_argument(
        "--show-settling",
        action="store_true",
        help=(
            "Render VBD settle substeps in the viewer (default: off; show post-settle "
            "state only). Implies --force-settle and disables settle cache for this run."
        ),
    )
    parser.add_argument(
        "--scripted-ee-vel",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "VZ"),
        default=[0.05, 0.0, 0.0],
    )
    parser.add_argument("--status-every", type=int, default=60, help="Print status every N frames.")
    parser.add_argument(
        "--print-robot-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-env robot diagnostics once after build (default: off).",
    )
    parser.add_argument(
        "--print-per-env-params",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print all per-env fruiting params at startup (default: off).",
    )
    parser.add_argument(
        "--vic-linear-k",
        type=float,
        default=None,
        help="VIC linear K [N/m] (default: ranges sim_build or 200).",
    )
    parser.add_argument(
        "--vic-linear-d",
        type=float,
        default=None,
        help="VIC linear D [N·s/m] (default: ranges sim_build or 10).",
    )
    parser.add_argument(
        "--vic-angular-k",
        type=float,
        default=None,
        help="VIC angular K [N·m/rad] (default: ranges sim_build or 10).",
    )
    parser.add_argument(
        "--vic-angular-d",
        type=float,
        default=None,
        help="VIC angular D [N·m·s/rad] (default: ranges sim_build or 1).",
    )
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
    return parser


def _config_from_args(args: argparse.Namespace) -> BatchedHeterogeneousCoupledSimConfig:
    step_mode = _resolve_step_mode(args)
    fix_to_apple = bool(args.fix_to_apple)
    settle_substeps = int(args.settle_substeps)
    viz = _viz_settings_from_args(args)

    ranges_path = Path(args.json) if args.json else default_ranges_fixture_path()
    ranges = load_ranges(ranges_path)
    (
        fixture_vic,
        joint_angular_kd,
        joint_linear_kd,
        joint_angular_kp,
        joint_linear_kp,
        joint_damping_ratio,
    ) = _resolve_sim_build_knobs(ranges)
    vic_gains = ImpedanceGains(
        linear_k=float(args.vic_linear_k)
        if args.vic_linear_k is not None
        else fixture_vic.linear_k,
        linear_d=float(args.vic_linear_d)
        if args.vic_linear_d is not None
        else fixture_vic.linear_d,
        angular_k=float(args.vic_angular_k)
        if args.vic_angular_k is not None
        else fixture_vic.angular_k,
        angular_d=float(args.vic_angular_d)
        if args.vic_angular_d is not None
        else fixture_vic.angular_d,
    )
    seed = getattr(args, "_resolved_seed", None)

    base = BatchedHeterogeneousCoupledSimConfig.defaults()
    config = dataclasses.replace(
        base,
        runtime=dataclasses.replace(
            base.runtime,
            num_envs=int(args.num_envs),
            env_spacing=tuple(float(v) for v in args.env_spacing),
            device=resolve_sim_device(getattr(args, "device", None)),
            control_hz=float(args.hz),
            sub_dt=_PHYSICS_SUB_DT,
        ),
        robot=dataclasses.replace(
            base.robot,
            step_mode="vbd_only" if step_mode == "vbd" else "coupled",
            fix_to_apple=fix_to_apple,
        ),
        scene=dataclasses.replace(
            base.scene,
            settle_substeps=settle_substeps,
            settle_gravity_ramp=bool(args.settle_gravity_ramp),
            settle_quiet_every=(
                int(args.settle_quiet_every)
                if args.settle_quiet_every is not None
                else None
            ),
            enable_self_collisions=bool(args.enable_self_collision),
            enable_apple_woody_collisions=bool(args.apple_woody_collision),
            enable_proxy_woody_collisions=bool(args.proxy_woody_collision),
        ),
        domain_randomization=dataclasses.replace(
            base.domain_randomization,
            ranges_path=ranges_path,
            topology_seed=int(seed) if seed is not None else None,
        ),
        controller=dataclasses.replace(
            base.controller,
            mode=str(args.controller),  # type: ignore[arg-type]
            vic_gains=vic_gains,
        ),
        fruiting_system=dataclasses.replace(
            base.fruiting_system,
            joint_angular_kd_overrides=joint_angular_kd,
            joint_linear_kd_overrides=joint_linear_kd,
            joint_angular_kp_overrides=joint_angular_kp,
            joint_linear_kp_overrides=joint_linear_kp,
            joint_damping_ratio=joint_damping_ratio,
        ),
        settle_diagnostics=SettleDiagnosticsConfig() if settle_substeps > 0 else None,
        obs=(
            ObsConfig(allocate_buffers=True, include_robot=True, include_forces=True)
            if viz.needs_obs_buffers
            else None
        ),
    )
    config.validate()
    return config


def _print_startup(
    config: BatchedHeterogeneousCoupledSimConfig,
    *,
    ranges_path: Path,
    seed: int,
    per_env_params: list[FruitingSystemParams],
    print_per_env_params_flag: bool = False,
) -> None:
    print(f"Heterogeneous batched fruiting ranges: {ranges_path}")
    print(f"Topology seed: {seed}")
    print(f"Warp device: {config.resolve_device()}")
    step_mode = config.robot.step_mode
    if step_mode == "coupled":
        print("M1 cable + FR3+EE MuJoCo (staggered coupling); Newton viewer shows cable model.")
    else:
        print("Cable SolverVBD only (--only-vbd).")
    if print_per_env_params_flag:
        print_per_env_params(per_env_params)
    fix_to_apple = config.robot.fix_to_apple
    coupling_label = (
        "stem-harvest / settle-then-weld"
        if fix_to_apple and step_mode != "vbd_only"
        else "velocity-delta"
        if not fix_to_apple
        else "ignored with --only-vbd"
    )
    print(f"Gripper proxy fix_to_apple={fix_to_apple} ({coupling_label} coupling).")
    scene = config.scene
    print(
        "AVBD cable collisions: "
        f"self_collision={scene.enable_self_collisions} "
        f"apple↔woody={scene.enable_apple_woody_collisions} "
        f"proxy↔woody={scene.enable_proxy_woody_collisions}",
        flush=True,
    )


def _print_settle_diagnostics(sim: BatchedHeterogeneousCoupledSim) -> None:
    br = sim.build_result
    if br.settle_stability_reports is None and br.ik_envelope_results is None:
        return
    diag = sim.config.settle_diagnostics
    brief = bool(diag.report_brief) if diag is not None else False
    stability = list(br.settle_stability_reports or ())
    print_envelope_coverage_report(
        list(br.ik_envelope_results or ()),
        stability_reports=stability,
        verbose=not brief,
    )
    unstable = [r.world for r in stability if not r.is_stable]
    if unstable:
        print_per_env_params(
            sim.per_env_params,
            env_indices=unstable,
            heading=(
                f"Unstable settle envs {unstable} — fruiting params "
                "(topology shared, continuous θ differs):"
            ),
        )


def _read_scripted_or_keyboard_velocity(
    viewer,
    args: argparse.Namespace,
    sim: BatchedHeterogeneousCoupledSim,
) -> fr3_robot.EEVelocity:
    linear = tuple(float(v) for v in args.scripted_ee_vel)
    angular = (0.0, 0.0, 0.0)
    if args.fr3_keyboard and hasattr(viewer, "is_key_down"):
        return fr3_robot.read_keyboard_ee_velocity(
            viewer,
            linear_speed=sim.config.controller.linear_speed,
            angular_speed=sim.config.controller.angular_speed,
        )
    return fr3_robot.EEVelocity(linear=linear, angular=angular)


def _build_frame_actions(
    sim: BatchedHeterogeneousCoupledSim,
    viewer,
    args: argparse.Namespace,
) -> torch.Tensor:
    vel = _read_scripted_or_keyboard_velocity(viewer, args, sim)
    row = torch.tensor(
        [*vel.linear, *vel.angular],
        dtype=torch.float32,
        device=sim.device,
    )
    return row.unsqueeze(0).expand(sim.num_envs, -1).contiguous()


def _setup_viewer(viewer, sim: BatchedHeterogeneousCoupledSim) -> bool:
    graphical = isinstance(viewer, newton.viewer.ViewerGL)
    viewer.set_model(sim.scene.cable.model)
    if graphical and sim.num_envs > 1:
        viewer.set_world_offsets(tuple(sim.config.runtime.env_spacing))
    return graphical


def _print_viz_startup(
    sim: BatchedHeterogeneousCoupledSim,
    viz: VizSettings,
    *,
    graphical: bool,
    status_every: int,
    step_mode: str,
) -> None:
    if viz.tcp_force_arrow and step_mode != "coupled":
        print(
            "Note: --tcp-force-arrow needs full coupled stepping (omit --only-vbd).",
            flush=True,
        )
    elif viz.tcp_force_arrow and graphical:
        cap = (
            f"{viz.tcp_force_max_length:.2f} m max"
            if viz.tcp_force_max_length > 0.0
            else "no max"
        )
        print(
            "TCP force arrows: yellow at each env's robot TCP; "
            f"scale={viz.tcp_force_scale:.4f} m/N × gain {viz.tcp_force_gain:g}, "
            f"min {viz.tcp_force_min_length:.2f} m, {cap}.",
            flush=True,
        )
    if viz.mark_endpoints:
        print_mark_endpoints_startup(sim.scene.cable, status_every=status_every)


def _gather_obs_for_viz(sim: BatchedHeterogeneousCoupledSim) -> None:
    if sim.obs_bufs is None:
        return
    sim.gather_obs()


def _render_frame(
    viewer,
    sim: BatchedHeterogeneousCoupledSim,
    sim_time: float,
    viz: VizSettings,
) -> None:
    scene = sim.scene
    layout = sim.layout
    if scene.last_vbd_contacts is not None:
        contacts = scene.last_vbd_contacts
    else:
        contacts = scene.cable.model.collide(
            scene.cable.state_0,
            collision_pipeline=scene.cable_collision_pipeline,
        )
    viewer.begin_frame(sim_time)
    viewer.log_state(scene.cable.state_0)
    viewer.log_contacts(contacts, scene.cable.state_0)
    if viz.needs_obs_buffers:
        _gather_obs_for_viz(sim)
    if viz.tcp_force_arrow and layout is not None:
        log_batched_tcp_force_arrows(
            viewer,
            scene,
            layout,
            scale_per_newton=viz.tcp_force_scale,
            gain=viz.tcp_force_gain,
            min_length=viz.tcp_force_min_length,
            max_length=viz.tcp_force_max_length,
            bufs=sim.obs_bufs,
        )
    if viz.mark_endpoints and layout is not None:
        log_batched_endpoints(
            viewer,
            scene,
            layout,
            bufs=sim.obs_bufs,
            woody_force_scale=viz.tcp_force_scale,
            woody_force_gain=viz.tcp_force_gain,
            woody_force_min_length=viz.tcp_force_min_length,
            woody_force_max_length=viz.tcp_force_max_length,
        )
    viewer.end_frame()


def _print_minimal_status(
    sim: BatchedHeterogeneousCoupledSim,
    frame: int,
    viz: VizSettings,
) -> None:
    layout = sim.layout
    if layout is None:
        return
    body_q = sim.scene.cable.state_0.body_q.numpy().reshape(-1, 7)
    parts = []
    for w, apple_idx in enumerate(layout.apple_body_indices):
        if apple_idx < 0:
            continue
        z = float(body_q[apple_idx, 2])
        parts.append(f"env{w} apple_z={z:.4f}")
    if parts:
        print(f"frame {frame}: " + "  ".join(parts), flush=True)
    if viz.mark_endpoints and sim.obs_bufs is not None:
        _gather_obs_for_viz(sim)
        print_batched_obs_debug(
            sim.obs_bufs,
            frame=frame,
            sim_time=sim.sim_time,
            cable=sim.scene.cable,
        )


def test_final(sim: BatchedHeterogeneousCoupledSim, tolerance: float = 0.05) -> None:
    layout = sim.layout
    if layout is None:
        return
    body_q = sim.scene.cable.state_0.body_q.numpy().reshape(-1, 7)
    for w, apple_idx in enumerate(layout.apple_body_indices):
        if apple_idx < 0:
            continue
        z = float(body_q[apple_idx, 2])
        assert z > -tolerance, f"world {w} apple fell: z={z}"


def main() -> None:
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--num-frames", "200"])
            print("No DISPLAY/WAYLAND_DISPLAY: using --viewer null --num-frames 200.")

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)

    _require_fr3_assets()
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31 - 1)
    args._resolved_seed = int(seed)

    config = _config_from_args(args)
    viz = _viz_settings_from_args(args)
    ranges_path = config.domain_randomization.resolved_ranges_path()
    ranges = load_ranges(ranges_path)
    per_env_params = sample_heterogeneous_params_list(
        ranges,
        topology_seed=int(seed),
        num_envs=config.runtime.num_envs,
    )

    _print_startup(
        config,
        ranges_path=ranges_path,
        seed=int(seed),
        per_env_params=per_env_params,
        print_per_env_params_flag=bool(args.print_per_env_params),
    )

    graphical = isinstance(viewer, newton.viewer.ViewerGL)
    show_settling = graphical and bool(args.show_settling)
    build_viewer = viewer if show_settling else None
    use_settle_cache = bool(args.use_settle_cache) and not show_settling
    force_settle = bool(args.force_settle) or show_settling
    if show_settling:
        print(
            "Settling visualization enabled: running full VBD settle and rendering substeps.",
            flush=True,
        )

    if config.robot.step_mode != "vbd_only":
        fr3_robot.enable_ik_bootstrap_warnings_for_examples()

    sim = BatchedHeterogeneousCoupledSim(
        config,
        per_env_params,
        ranges,
        viewer=build_viewer,
        use_settle_cache=use_settle_cache,
        force_settle=force_settle,
    )

    _setup_viewer(viewer, sim)
    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    _print_settle_diagnostics(sim)

    step_mode = "coupled" if config.robot.step_mode != "vbd_only" else "vbd"
    _print_viz_startup(
        sim,
        viz,
        graphical=graphical,
        status_every=int(args.status_every),
        step_mode=step_mode,
    )

    if args.print_robot_state and sim.layout is not None:
        print_batched_robot_status(sim.scene, sim.layout)

    vbd_only = config.robot.step_mode == "vbd_only"
    frame = 0
    print("Starting heterogeneous batched coupled simulation…", flush=True)
    # With --show-settling off we skip rendering settle substeps during build. Render one
    # frame before any runtime stepping so the viewer shows the post-settle state.
    _render_frame(viewer, sim, sim.sim_time, viz)
    while viewer.is_running():
        print(f"frame {frame} sim_time={sim.sim_time:.3f}s")
        actions = None if vbd_only else _build_frame_actions(sim, viewer, args)
        sim.step(actions)
        _render_frame(viewer, sim, sim.sim_time, viz)
        if args.status_every and frame % int(args.status_every) == 0:
            _print_minimal_status(sim, frame, viz)
        frame += 1
        if graphical:
            time.sleep(max(0.0, sim.frame_dt))

    test_final(sim)
    print(f"Done (sim_time={sim.sim_time:.3f}s).", flush=True)


if __name__ == "__main__":
    main()
