"""Quasi-static sys-ID demo for ``ApplePickSysId-v0`` (M3.0 §2.1).

Runs the Fibonacci-hemisphere push–hold–return trajectory through the coupled VIC
env and prints mean steady-state ``ft_wrist`` force per direction during holds.

Run from the repository root::

    uv sync --extra gym --extra vic --extra dev
    uv run python apple_pick_gym/examples/example_gym_sysid.py --viewer null

With Newton GL viewer (requires a display)::

    uv run python apple_pick_gym/examples/example_gym_sysid.py --viewer gl

Headless smoke (first direction only)::

    uv run python apple_pick_gym/examples/example_gym_sysid.py --viewer null \\
      --n-directions 1 --max-steps 200
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero vector cannot be normalized")
    return (v / n).astype(np.float64)


def _action_from_velocity(vel) -> np.ndarray:
    lin = np.asarray(vel.linear, dtype=np.float32)
    ang = np.asarray(vel.angular, dtype=np.float32)
    return np.concatenate([lin, ang])


def _estimate_trajectory_frames(n_directions: int, config) -> int:
    move_frames = max(1, int(math.ceil(config.step_size_m / config.move_speed_mps * config.control_hz)))
    return_frames = max(
        1,
        int(
            math.ceil(
                config.n_steps * config.step_size_m / config.move_speed_mps * config.control_hz
            )
        ),
    )
    hold_frames = max(0, int(math.ceil(config.hold_duration_s * config.control_hz)))
    per_dir = config.n_steps * move_frames + hold_frames + return_frames
    return n_directions * per_dir


def _make_parser() -> argparse.ArgumentParser:
    import newton.examples

    p = newton.examples.create_parser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-directions", type=int, default=10, help="Fibonacci hemisphere samples")
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Cap env steps (0 = full trajectory for --n-directions)",
    )
    p.add_argument(
        "--max-tcp-force-n",
        type=float,
        default=1000.0,
        help="Wrench safety stop [N]; post-grasp transients can exceed 500 N on step 0",
    )
    p.add_argument(
        "--hz",
        type=float,
        default=30.0,
        help="Viewer refresh rate when --viewer gl (env still steps at control_hz).",
    )
    p.add_argument(
        "--fix-to-apple",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Weld gripper to apple for post-grasp sys-ID (default on).",
    )
    p.add_argument(
        "--fix-to-apple-warmup-substeps",
        type=int,
        default=1800,
        help="VBD settle substeps before welding (fix_to_apple only).",
    )
    p.add_argument(
        "--mujoco-viewer",
        action="store_true",
        help="Open MuJoCo passive viewer for the FR3 arm (requires a GUI session).",
    )
    return p


def _maybe_log_forces(viewer: object, obs: dict, info: dict, *, amplitude_m: float) -> None:
    log = getattr(viewer, "log_scalar", None)
    if log is None:
        return

    ft = obs.get("ft_wrist")
    if ft is not None:
        w = np.asarray(ft, dtype=np.float64).reshape(6)
        log("SysID |F| wrist [N]", float(np.linalg.norm(w[:3])), smoothing=3)
        log("SysID amp [m]", float(amplitude_m), smoothing=3)

    ee = info.get("end_effector_wrench")
    if ee is not None:
        w = np.asarray(ee, dtype=np.float64).reshape(6)
        log("SysID |F| TCP harvest [N]", float(np.linalg.norm(w[:3])), smoothing=3)


def main() -> None:
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--num-frames", "1"])
            print("No DISPLAY/WAYLAND_DISPLAY: using --viewer null (override with --viewer gl).")

    parser = _make_parser()
    import newton.examples

    viewer, args = newton.examples.init(parser=parser)

    import gymnasium as gym
    import apple_pick_gym  # noqa: F401 — registers ApplePickSysId-v0
    from apple_pick_gym.envs import ApplePickSysIdEnv
    from apple_pick_sim.system_id import (
        ExcitationContext,
        QuasiStaticStepConfig,
        QuasiStaticTrajectory,
        sample_fibonacci_hemisphere,
    )

    config = QuasiStaticStepConfig()
    n_directions = int(args.n_directions)
    max_steps = int(args.max_steps)
    if max_steps <= 0:
        max_steps = _estimate_trajectory_frames(n_directions, config) + 64

    env = gym.make(
        "ApplePickSysId-v0",
        render_mode=None,
        max_episode_steps=max_steps,
        max_tcp_force_n=float(args.max_tcp_force_n),
        fix_to_apple=bool(args.fix_to_apple),
        fix_to_apple_warmup_substeps=int(args.fix_to_apple_warmup_substeps),
        control_hz=config.control_hz,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    obs, info = env.reset(seed=int(args.seed))

    scene = env.unwrapped._scene
    if scene is None:
        raise RuntimeError("Env did not create a scene; did reset() succeed?")

    tcp_target = np.asarray(env.unwrapped._controller.target_tf[:3], dtype=np.float64)
    stem_dir = _normalize(np.asarray(obs["apple_pos"], dtype=np.float64) - tcp_target)
    directions = sample_fibonacci_hemisphere(n_directions, stem_dir)
    traj = QuasiStaticTrajectory(directions, config)

    viewer.set_model(scene.cable.model)
    sim_time = 0.0
    render_dt = 1.0 / float(args.hz)

    mujoco_viewer = bool(getattr(args, "mujoco_viewer", False))
    if mujoco_viewer and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("Suppressing --mujoco-viewer (no DISPLAY/WAYLAND_DISPLAY).")
        mujoco_viewer = False

    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    hold_forces: dict[int, list[np.ndarray]] = defaultdict(list)
    prev_phase: str | None = None
    dir_idx = -1

    print(f"Quasi-static sys-ID: {n_directions} directions, up to {max_steps} env steps")
    print(f"Stem direction (TCP → apple): ({stem_dir[0]:+.3f}, {stem_dir[1]:+.3f}, {stem_dir[2]:+.3f})")

    def _render_frame(
        frame_obs: dict,
        frame_info: dict,
        t: float,
        phase: str,
        linear_velocity: tuple[float, float, float] | None = None,
    ) -> None:
        if scene.last_vbd_contacts is not None:
            viz_contacts = scene.last_vbd_contacts
        else:
            viz_contacts = scene.cable.model.collide(
                scene.cable.state_0,
                collision_pipeline=scene.cable_collision_pipeline,
            )

        viewer.begin_frame(t)
        viewer.log_state(scene.cable.state_0)
        viewer.log_contacts(viz_contacts, scene.cable.state_0)
        ApplePickSysIdEnv.log_woody_part_markers(viewer, frame_obs, scene=scene)
        ApplePickSysIdEnv.log_junction_force_arrows(viewer, frame_obs, scene=scene)
        ApplePickSysIdEnv.log_ft_wrist_arrow(
            viewer,
            frame_obs,
            scene=scene,
            max_length=0.75,
        )
        ApplePickSysIdEnv.log_movement_direction_arrow(
            viewer,
            frame_obs,
            scene=scene,
            linear_velocity=linear_velocity,
        )
        _maybe_log_forces(viewer, frame_obs, frame_info, amplitude_m=traj.current_amplitude_m)
        viewer.end_frame()

        amp = traj.current_amplitude_m
        ft = np.asarray(frame_obs.get("ft_wrist", np.zeros(6)), dtype=np.float64)
        print(
            f"  step phase={phase:8s} dir={dir_idx:2d} amp={amp*100:5.1f} cm"
            f"  |F|={np.linalg.norm(ft[:3]):6.2f} N",
            end="\r",
            flush=True,
        )

    try:
        _render_frame(obs, info, sim_time, "init")

        for step_idx, (phase, vel) in enumerate(traj.iter_frames()):
            if step_idx >= max_steps:
                print(f"\nStopped at --max-steps={max_steps}")
                break
            if not viewer.is_running():
                print("\nViewer closed.")
                break

            if phase == "move_out" and prev_phase != "move_out":
                dir_idx += 1
                print(f"\n--- Direction {dir_idx}: ({directions[dir_idx][0]:+.3f}, "
                      f"{directions[dir_idx][1]:+.3f}, {directions[dir_idx][2]:+.3f}) ---")

            ctx = ExcitationContext(
                type="quasi_static",
                f_inst=0.0,
                direction=traj.current_direction,
            )
            env.unwrapped.set_excitation_context(ctx)

            action = _action_from_velocity(vel)
            obs, _reward, terminated, truncated, info = env.step(action)
            sim_time += 1.0 / config.control_hz

            scene = env.unwrapped._scene
            if scene is None:
                break

            if phase == "hold" and dir_idx >= 0:
                hold_forces[dir_idx].append(np.asarray(obs["ft_wrist"][:3], dtype=np.float64))

            _render_frame(obs, info, sim_time, phase, linear_velocity=vel.linear)
            prev_phase = phase

            if mujoco_viewer and scene.robot_model is not None:
                from apple_pick_sim.robot import fr3_robot

                fr3_robot.sync_mujoco_visual_state(
                    scene.mj_solver,
                    scene.robot_model,
                    scene.robot_state_0,
                )
                scene.mj_solver.render_mujoco_viewer()

            if getattr(args, "viewer", None) != "null":
                time.sleep(max(0.0, render_dt - (1.0 / config.control_hz)))

            if terminated:
                print(f"\nTerminated (wrench guard at step {step_idx}).")
                break
            if truncated:
                print(f"\nTruncated at step {step_idx} (max_episode_steps={max_steps}).")
                break

        print("\nMean steady-state wrist force [N] per direction (hold phases):")
        for idx in sorted(hold_forces):
            samples = hold_forces[idx]
            if not samples:
                print(f"  dir {idx:2d}: (no hold samples)")
                continue
            mean_f = np.mean(np.stack(samples, axis=0), axis=0)
            d = directions[idx]
            print(
                f"  dir {idx:2d}  d=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})"
                f"  F=({mean_f[0]:+.3f},{mean_f[1]:+.3f},{mean_f[2]:+.3f})"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
