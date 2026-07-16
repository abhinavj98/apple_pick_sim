"""Collect a Young's-modulus E-grid and write a faceted overlay Plotly HTML.

Treats each ``YoungsModulusCandidate`` as a structure (shared topology/geometry,
different ``E``), fans out directions like
``example_batched_collect_sysid_data.py``, soft-disables unstable envs, then
writes one comparison HTML (norms + move-vs-pull; see
``youngs_modulus_overlay_viz`` hygiene rules).

Run from repo root::

    uv run python apple_pick_gym/batched_examples/example_batched_youngs_modulus_collect_viz.py \\
        --viewer null --num-directions 2 --max-steps 80 \\
        --log10-e-primary 8.0,8.5 --log10-e-spur 7.5 --log10-e-stem 7.0 \\
        --output /tmp/youngs_e_grid_smoke --overwrite

"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

import newton.examples
import newton.viewer

from apple_pick_gym.batched_examples._youngs_e_grid_cli import (
    candidate_log10_triples,
    candidates_from_log10_cli,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    EXAMPLE_JOINT_ANGULAR_KD_OVERRIDES,
    EXAMPLE_JOINT_ANGULAR_KP_OVERRIDES,
    EXAMPLE_JOINT_LINEAR_KD_OVERRIDES,
    EXAMPLE_JOINT_LINEAR_KP_OVERRIDES,
)
from apple_pick_sim.fruiting_system import (
    default_ranges_fixture_path,
    load_ranges,
    parse_sim_build,
    sample_params,
)
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import ImpedanceGains
from apple_pick_sim.system_id import QuasiStaticStepConfig

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
    p.add_argument("--topology-seed", type=int, default=42)
    p.add_argument("--num-directions", type=int, default=2)
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Cap trajectory frames (0 = full trajectory estimate).",
    )
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--ranges-path", type=str, default=None)
    p.add_argument("--log10-e-primary", type=str, default="8.0,8.5")
    p.add_argument("--log10-e-spur", type=str, default="7.5")
    p.add_argument("--log10-e-stem", type=str, default="7.0")
    p.add_argument(
        "--max-overlay-candidates",
        type=int,
        default=8,
        help="Refuse overlay if more distinct E candidates than this.",
    )
    p.add_argument(
        "--overlay-html",
        type=str,
        default=None,
        help="Overlay HTML path (default: <output>/youngs_modulus_overlay.html).",
    )
    p.add_argument("--movement-per-step-m", type=float, default=0.02)
    p.add_argument("--total-movement-m", type=float, default=0.10)
    p.add_argument("--move-speed-mps", type=float, default=0.2)
    p.add_argument("--hold-duration-s", type=float, default=1.5)
    p.add_argument(
        "--skip-return",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--settle-substeps", type=int, default=None)
    p.add_argument(
        "--settle-gravity-ramp",
        action=argparse.BooleanOptionalAction,
        default=SETTLE_GRAVITY_RAMP,
    )
    p.add_argument("--settle-quiet-every", type=int, default=SETTLE_QUIET_EVERY)
    p.add_argument(
        "--show-pull-direction",
        action="store_true",
        help="Draw cyan pull-direction arrows (requires --viewer gl).",
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
            import numpy as np

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
            sys.argv.extend(["--viewer", "null"])

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)

    from apple_pick_gym.batched_envs import ApplePickBatchedSysIdEnv
    from apple_pick_gym.batched_envs.batched_stability_monitor import StabilityThresholds
    from apple_pick_gym.batched_envs.batched_sysid_collect import (
        broadcast_structure_params,
        collect_batched_quasi_static_dataset,
    )
    from apple_pick_gym.youngs_modulus_overlay_viz import (
        overlay_episodes_from_batched_dataset,
        write_youngs_modulus_overlay_html,
    )
    from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset

    candidates = candidates_from_log10_cli(
        log10_e_primary=str(args.log10_e_primary),
        log10_e_spur=str(args.log10_e_spur),
        log10_e_stem=str(args.log10_e_stem),
    )
    if len(candidates) < 1:
        raise SystemExit("need at least one YoungsModulusCandidate")
    if len(candidates) > int(args.max_overlay_candidates):
        raise SystemExit(
            f"{len(candidates)} candidates exceeds --max-overlay-candidates="
            f"{int(args.max_overlay_candidates)}; shrink the log10-E grid"
        )

    num_structures = len(candidates)
    num_directions = int(args.num_directions)
    if num_directions < 1:
        raise SystemExit("--num-directions must be >= 1")

    num_envs = num_structures * num_directions
    ranges_path = args.ranges_path or str(default_ranges_fixture_path())
    ranges = load_ranges(ranges_path)
    base = sample_params(ranges, seed=int(args.topology_seed), omit=("secondary",))
    structure_params = [c.apply_to(base) for c in candidates]
    per_env_params = broadcast_structure_params(structure_params, num_directions)
    config = build_trajectory_config(args)

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

    print(f"candidates={num_structures} directions={num_directions} envs={num_envs}")
    for i, c in enumerate(candidates):
        print(f"  s{i:02d} {c.short_label()} E=({c.primary:.3g},{c.spur:.3g},{c.stem:.3g})")

    def on_step(
        *,
        step_idx: int,
        phase: str,
        sim_time: float,
        obs,
        amplitude_m: float = 0.0,
        **_kwargs,
    ) -> bool:
        del step_idx, phase, amplitude_m
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
        return True

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
            on_step=on_step,
            command_argv=sys.argv,
            overwrite=bool(args.overwrite),
            shared_pull_directions_across_structures=True,
            stability_thresholds=StabilityThresholds(),
            save_snapshot=False,
        )
    finally:
        env.close()

    dataset = BatchedSysIdDataset(out)
    labels = [c.short_label() for c in candidates]
    log10_triples = candidate_log10_triples(candidates)
    episodes = overlay_episodes_from_batched_dataset(
        dataset,
        candidate_labels=labels,
        candidate_log10_e=log10_triples,
    )
    overlay_path = (
        Path(args.overlay_html)
        if args.overlay_html
        else Path(args.output) / "youngs_modulus_overlay.html"
    )
    written = write_youngs_modulus_overlay_html(
        episodes,
        overlay_path,
        max_overlay_candidates=int(args.max_overlay_candidates),
        title="Young's modulus E-grid overlay",
    )
    print(f"dataset={out}")
    print(f"overlay={written}")


if __name__ == "__main__":
    main()
