"""Parallel batched sys-ID data collection (V.4.2).

Collect ``num_structures × num_directions`` quasi-static episodes in one GPU
batched run and write a ``batched_sysid_v1`` dataset (``manifest.json`` +
``episodes/s{s}_d{d}.parquet``).

Run from the repository root::

    uv run python apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py \\
      --viewer gl --show-pull-direction --num-structures 2 --num-directions 3 \\
      --max-steps 200 --output /tmp/batched_sysid_dataset

Headless smoke (single structure, single direction)::

    uv run python apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py \\
      --viewer null --num-structures 1 --num-directions 1 --max-steps 80 \\
      --output /tmp/batched_sysid_smoke
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

import newton.examples
import newton.viewer


def _trajectory_config_from_args(args: argparse.Namespace):
    from apple_pick_sim.system_id import QuasiStaticStepConfig

    return QuasiStaticStepConfig(
        movement_per_step_m=float(args.movement_per_step_m),
        total_movement_m=float(args.total_movement_m),
        move_speed_mps=float(args.move_speed_mps),
        hold_duration_s=float(args.hold_duration_s),
        skip_return=bool(args.skip_return),
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
        help="Print collection progress messages.",
    )
    p.add_argument(
        "--show-pull-direction",
        action="store_true",
        help="Draw cyan pull-direction arrows at each TCP (requires --viewer gl).",
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
    from apple_pick_gym.batched_envs.batched_sysid_collect import (
        collect_batched_quasi_static_dataset,
        sample_and_broadcast_structure_params,
    )
    from apple_pick_sim.fruiting_system import default_ranges_fixture_path

    num_structures = int(args.num_structures)
    num_directions = int(args.num_directions)
    if num_structures < 1 or num_directions < 1:
        raise SystemExit("--num-structures and --num-directions must be >= 1")

    num_envs = num_structures * num_directions
    ranges_path = args.ranges_path or str(default_ranges_fixture_path())
    config = _trajectory_config_from_args(args)
    per_env_params = sample_and_broadcast_structure_params(
        ranges_path,
        topology_seed=int(args.topology_seed),
        num_structures=num_structures,
        num_directions=num_directions,
    )

    env = ApplePickBatchedSysIdEnv(
        num_envs=num_envs,
        max_episode_steps=max(int(args.max_steps), 1) if int(args.max_steps) > 0 else 4096,
        ranges_path=ranges_path,
        topology_seed=int(args.topology_seed),
        use_settle_cache=False,
        per_env_params=per_env_params,
        control_hz=float(config.control_hz),
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

    def on_step(*, step_idx: int, phase: str, sim_time: float, obs, **_kwargs) -> bool:
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
        if bool(args.debug) and step_idx >= 0 and step_idx % 20 == 0:
            print(f"step {step_idx} phase={phase}")
        return True

    progress = print if bool(args.debug) else None
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
        )
        if bool(args.debug):
            print(f"Saved batched sys-ID dataset to {out}")
    finally:
        env.close()
        if hasattr(viewer, "close"):
            viewer.close()


if __name__ == "__main__":
    main()
