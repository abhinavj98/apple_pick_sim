"""Parallel batched sys-ID data collection (V.4.2).

Collect ``num_structures × num_directions`` quasi-static episodes in one GPU
batched run and write a ``batched_sysid_v1`` dataset (``manifest.json`` +
``episodes/s{s}_d{d}.parquet``).

Run from the repository root::

    uv run python apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py \\
      --viewer gl --show-pull-direction --num-structures 2 --num-directions 3 \\
      --max-steps 200 --output /tmp/batched_sysid_dataset

Headless smoke::

    uv run python apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py \\
      --viewer null --num-structures 1 --num-directions 1 --max-steps 80 \\
      --output /tmp/batched_sysid_smoke

Sim build (VIC gains, settle substeps, control_hz, …) is configured via module
constants in this file as fallbacks; when the ranges JSON includes optional
``sim_build``, those values win. Settle phase knobs also accept
``--settle-substeps``, ``--settle-gravity-ramp``, and ``--settle-quiet-every``.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time

import numpy as np

import newton.examples
import newton.viewer

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ControllerConfig,
    DomainRandomizationConfig,
    EXAMPLE_JOINT_ANGULAR_KD_OVERRIDES,
    EXAMPLE_JOINT_ANGULAR_KP_OVERRIDES,
    EXAMPLE_JOINT_LINEAR_KD_OVERRIDES,
    EXAMPLE_JOINT_LINEAR_KP_OVERRIDES,
    FruitingSystemConfig,
    MujocoConfig,
    ObsConfig,
    RobotConfig,
    RuntimeConfig,
    SceneSettleCollisionConfig,
)
from apple_pick_sim.coupled_fruiting.scene import (
    DEFAULT_FR3_MUJOCO_SOLVER_KWARGS,
    DEFAULT_STEM_COUPLING_GAIN,
    DEFAULT_STEM_FORCE_CAP_N,
    DEFAULT_STEM_TORQUE_CAP_NM,
)
from apple_pick_sim.fruiting_system import default_ranges_fixture_path, load_ranges, parse_sim_build
from apple_pick_sim.fruiting_system.params import PLACEHOLDER_EE_MASS_KG, GripperProxyConfig
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import ImpedanceGains
from apple_pick_sim.system_id import QuasiStaticStepConfig
from apple_pick_sim.system_id.quasi_static_trajectory import derive_n_steps
from apple_pick_sim.system_id.trajectory_store import phase_to_int

# --- Sim build fallbacks (used when ranges omit ``sim_build``) ---
CONTROL_HZ = 30.0
SUB_DT = 1.0 / 1800.0
ENV_SPACING = (2.0, 2.0, 2.0)
SETTLE_SUBSTEPS = 5000
SETTLE_GRAVITY_RAMP = False
SETTLE_QUIET_EVERY: int | None = None
VIC_GAINS = ImpedanceGains(
    linear_k=200.0,
    linear_d=10.0,
    angular_k=10.0,
    angular_d=1.0,
)
JOINT_ANGULAR_KD_OVERRIDES = EXAMPLE_JOINT_ANGULAR_KD_OVERRIDES
JOINT_LINEAR_KD_OVERRIDES = EXAMPLE_JOINT_LINEAR_KD_OVERRIDES
JOINT_ANGULAR_KP_OVERRIDES = EXAMPLE_JOINT_ANGULAR_KP_OVERRIDES
JOINT_LINEAR_KP_OVERRIDES = EXAMPLE_JOINT_LINEAR_KP_OVERRIDES
GRIPPER_PROXY = GripperProxyConfig(mass=PLACEHOLDER_EE_MASS_KG)


def _resolve_sim_build_knobs(ranges: dict) -> tuple[
    ImpedanceGains,
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    sb = parse_sim_build(ranges)
    if sb is None:
        return (
            VIC_GAINS,
            dict(JOINT_ANGULAR_KD_OVERRIDES),
            dict(JOINT_LINEAR_KD_OVERRIDES),
            dict(JOINT_ANGULAR_KP_OVERRIDES),
            dict(JOINT_LINEAR_KP_OVERRIDES),
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
    )


def summarize_trajectory_for_debug(config: QuasiStaticStepConfig) -> str:
    """Human-readable move/hold frame counts for ``--debug`` startup logging."""
    import math

    n_steps = derive_n_steps(
        movement_per_step_m=config.movement_per_step_m,
        total_movement_m=config.total_movement_m,
    )
    hz = float(config.control_hz)
    move_frames = max(
        1,
        int(math.ceil(config.movement_per_step_m / config.move_speed_mps * hz)),
    )
    hold_frames = max(0, int(math.ceil(config.hold_duration_s * hz)))
    per_dir_frames = n_steps * (move_frames + hold_frames)
    if not config.skip_return:
        return_frames = max(
            1,
            int(math.ceil(config.total_movement_m / config.move_speed_mps * hz)),
        )
        per_dir_frames += return_frames
    return (
        f"trajectory: {n_steps} increments × ({move_frames} move_out + {hold_frames} hold) "
        f"= {per_dir_frames} frames/direction @ {hz:.0f} Hz "
        f"(step={config.movement_per_step_m * 1000:.1f} mm, "
        f"total={config.total_movement_m * 100:.1f} cm, "
        f"speed={config.move_speed_mps:.3f} m/s, hold={config.hold_duration_s:.3f} s)"
    )


def format_trajectory_step_debug(
    *,
    step_idx: int,
    phase: str,
    sim_time: float,
    amplitude_m: float = 0.0,
) -> str:
    """One-line per-step debug log (phase name + Parquet int code)."""
    if phase == "init":
        phase_int = -1
    elif phase == "pre_weld":
        phase_int = phase_to_int("pre_weld")
    else:
        phase_int = phase_to_int(phase)
    return (
        f"step={step_idx:4d}  phase={phase:8s}({phase_int})  "
        f"amp={amplitude_m * 1000:.2f}mm  t={sim_time:.3f}s"
    )


def build_trajectory_config(args: argparse.Namespace) -> QuasiStaticStepConfig:
    return QuasiStaticStepConfig(
        movement_per_step_m=float(args.movement_per_step_m),
        total_movement_m=float(args.total_movement_m),
        move_speed_mps=float(args.move_speed_mps),
        hold_duration_s=float(args.hold_duration_s),
        control_hz=CONTROL_HZ,
        skip_return=bool(args.skip_return),
    )


def build_sim_config(
    *,
    num_envs: int,
    settle_substeps: int | None = None,
    settle_gravity_ramp: bool | None = None,
    settle_quiet_every: int | None = None,
    device: str | None = None,
    ranges: dict | None = None,
) -> BatchedHeterogeneousCoupledSimConfig:
    if ranges is None:
        ranges = load_ranges(default_ranges_fixture_path())
    (
        vic_gains,
        joint_angular_kd,
        joint_linear_kd,
        joint_angular_kp,
        joint_linear_kp,
    ) = _resolve_sim_build_knobs(ranges)
    gym_cfg = BatchedHeterogeneousCoupledSimConfig.gym_defaults(num_envs=int(num_envs))
    settle = SETTLE_SUBSTEPS if settle_substeps is None else int(settle_substeps)
    gravity_ramp = SETTLE_GRAVITY_RAMP if settle_gravity_ramp is None else bool(settle_gravity_ramp)
    quiet_every = SETTLE_QUIET_EVERY if settle_quiet_every is None else settle_quiet_every
    return dataclasses.replace(
        gym_cfg,
        runtime=dataclasses.replace(
            gym_cfg.runtime,
            control_hz=CONTROL_HZ,
            sub_dt=SUB_DT,
            env_spacing=ENV_SPACING,
            device=device,
        ),
        scene=dataclasses.replace(
            gym_cfg.scene,
            settle_substeps=settle,
            settle_gravity_ramp=gravity_ramp,
            settle_quiet_every=quiet_every,
        ),
        controller=dataclasses.replace(
            gym_cfg.controller,
            vic_gains=vic_gains,
        ),
        fruiting_system=dataclasses.replace(
            gym_cfg.fruiting_system,
            joint_angular_kd_overrides=joint_angular_kd,
            joint_linear_kd_overrides=joint_linear_kd,
            joint_angular_kp_overrides=joint_angular_kp,
            joint_linear_kp_overrides=joint_linear_kp,
        ),
    )


def _make_parser() -> argparse.ArgumentParser:
    p = newton.examples.create_parser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-structures", type=int, default=1)
    p.add_argument("--num-directions", type=int, default=1)
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Cap env steps (0 = full single-direction trajectory).",
    )
    p.add_argument(
        "--topology-seed",
        type=int,
        default=42,
        help="Fix segment topology; material params vary per structure index.",
    )
    p.add_argument(
        "--hold-duration-s",
        type=float,
        default=1.5,
        help="Zero-velocity hold after each increment [s] (default 1.5).",
    )
    p.add_argument(
        "--movement-per-step-m",
        type=float,
        default=0.02,
        help="Distance per fast move burst [m] (default 0.02 = 2 cm).",
    )
    p.add_argument(
        "--total-movement-m",
        type=float,
        default=0.10,
        help="Total push excursion per direction [m] (default 0.10 = 10 cm).",
    )
    p.add_argument(
        "--move-speed-mps",
        type=float,
        default=0.2,
        help="Linear speed during move bursts [m/s] (default 0.2).",
    )
    p.add_argument(
        "--skip-return",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Unused for single-direction envs; kept for Parquet metadata parity.",
    )
    p.add_argument(
        "--ranges-path",
        type=str,
        default=None,
        help="Fruiting-system ranges JSON for structure sampling.",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to write Parquet trajectory dataset.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing dataset at --output instead of appending a timestamp.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Log trajectory summary and per-step phase (name + Parquet int).",
    )
    p.add_argument(
        "--show-pull-direction",
        action="store_true",
        help="Draw cyan pull-direction arrows at each TCP (requires --viewer gl).",
    )
    p.add_argument(
        "--save-snapshot",
        action="store_true",
        help=(
            "Write post-weld initial_states/sXX_dYY.npz per episode (diagnostic only; "
            "default off)."
        ),
    )
    p.add_argument(
        "--settle-substeps",
        type=int,
        default=None,
        help=f"VBD substeps before runtime (default: {SETTLE_SUBSTEPS}).",
    )
    p.add_argument(
        "--settle-gravity-ramp",
        action=argparse.BooleanOptionalAction,
        default=SETTLE_GRAVITY_RAMP,
        help="Linear 0→−9.81 m/s² gravity ramp over all settle substeps (default: off).",
    )
    p.add_argument(
        "--settle-quiet-every",
        type=int,
        default=SETTLE_QUIET_EVERY,
        metavar="N",
        help=(
            "Zero all fruiting-system body twists every N VBD settle substeps "
            "(device-side; default: off)."
        ),
    )
    return p


def _render_frame(
    viewer: object,
    env: object,
    sim_time: float,
    *,
    obs: dict | None = None,
    show_pull_direction: bool = False,
) -> None:
    sim = env._sim
    scene = sim.scene
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
    if show_pull_direction and obs is not None:
        layout = sim.layout
        excitation = obs.get("excitation_direction")
        if layout is not None and excitation is not None:
            from apple_pick_sim.batched_viz import log_batched_movement_direction_arrows

            if hasattr(excitation, "detach"):
                directions = excitation.detach().cpu().numpy()
            else:
                directions = np.asarray(excitation, dtype=np.float64)
            log_batched_movement_direction_arrows(
                viewer,
                scene,
                layout,
                directions=directions,
                bufs=sim.obs_bufs,
            )
    viewer.end_frame()


def main() -> None:
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--num-frames", "1"])

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)

    from apple_pick_gym.batched_envs import ApplePickBatchedSysIdEnv
    from apple_pick_gym.batched_envs.batched_stability_monitor import StabilityThresholds
    from apple_pick_gym.batched_envs.batched_sysid_collect import (
        collect_batched_quasi_static_dataset,
        sample_and_broadcast_structure_params,
    )
    from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset

    stability_thresholds = StabilityThresholds()

    num_structures = int(args.num_structures)
    num_directions = int(args.num_directions)
    if num_structures < 1 or num_directions < 1:
        raise SystemExit("--num-structures and --num-directions must be >= 1")

    num_envs = num_structures * num_directions
    ranges_path = args.ranges_path or str(default_ranges_fixture_path())
    ranges = load_ranges(ranges_path)
    config = build_trajectory_config(args)
    per_env_params = sample_and_broadcast_structure_params(
        ranges_path,
        topology_seed=int(args.topology_seed),
        num_structures=num_structures,
        num_directions=num_directions,
    )

    device = args.device
    if device == "cuda":
        device = "cuda:0"
    sim_config = build_sim_config(
        num_envs=num_envs,
        device=device,
        settle_substeps=args.settle_substeps,
        settle_gravity_ramp=args.settle_gravity_ramp,
        settle_quiet_every=args.settle_quiet_every,
        ranges=ranges,
    )

    env = ApplePickBatchedSysIdEnv(
        num_envs=num_envs,
        max_episode_steps=max(int(args.max_steps), 1) if int(args.max_steps) > 0 else 4096,
        ranges_path=ranges_path,
        topology_seed=int(args.topology_seed),
        use_settle_cache=False,
        per_env_params=per_env_params,
        control_hz=float(config.control_hz),
        sim_config=sim_config,
    )

    sim = env._sim
    graphical = isinstance(viewer, newton.viewer.ViewerGL)
    viewer.set_model(sim.scene.cable.model)
    if graphical and env.num_envs > 1:
        viewer.set_world_offsets(tuple(sim.config.runtime.env_spacing))
    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    frame_dt = 1.0 / float(config.control_hz)
    use_viewer = graphical or getattr(args, "viewer", None) != "null"
    show_pull_direction = bool(args.show_pull_direction) and graphical

    def on_step(
        *,
        step_idx: int,
        phase: str,
        sim_time: float,
        obs,
        amplitude_m: float = 0.0,
        **_kwargs,
    ) -> bool:
        if hasattr(viewer, "is_running") and not viewer.is_running():
            return False
        if use_viewer:
            _render_frame(
                viewer,
                env,
                sim_time,
                obs=obs,
                show_pull_direction=show_pull_direction,
            )
            if graphical:
                time.sleep(max(0.0, frame_dt))
        if bool(args.debug):
            print(
                format_trajectory_step_debug(
                    step_idx=step_idx,
                    phase=phase,
                    sim_time=sim_time,
                    amplitude_m=float(amplitude_m),
                )
            )
        return True

    if bool(args.debug):
        print(summarize_trajectory_for_debug(config))

    progress = None
    try:
        out = collect_batched_quasi_static_dataset(
            env,
            num_structures=num_structures,
            num_directions=num_directions,
            config=config,
            output_dir=args.output,
            seed=int(args.seed),
            ranges_path=ranges_path,
            max_steps=int(args.max_steps),
            progress=progress,
            on_step=on_step,
            command_argv=sys.argv,
            overwrite=bool(args.overwrite),
            stability_thresholds=stability_thresholds,
            save_snapshot=bool(args.save_snapshot),
        )
        unstable_frames = 0
        total_frames = 0
        dataset = BatchedSysIdDataset(out)
        for ep in dataset.episode_entries():
            arrays = dataset.load_episode_obs_arrays(
                int(ep["structure_idx"]),
                int(ep["direction_idx"]),
            )
            stable = np.asarray(arrays["stable"], dtype=bool).reshape(-1)
            unstable_frames += int(np.count_nonzero(~stable))
            total_frames += int(stable.size)
        print(
            f"stability summary: {unstable_frames}/{total_frames} unstable frames "
            f"across {len(dataset.episode_entries())} episodes"
        )
        if bool(args.debug):
            print(f"Saved batched sys-ID dataset to {out}")
    finally:
        env.close()
        if hasattr(viewer, "close"):
            viewer.close()


if __name__ == "__main__":
    main()
