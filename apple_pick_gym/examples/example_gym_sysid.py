"""Quasi-static sys-ID demo for ``ApplePickSysId-v0`` (M3.0 §2.1).

Runs the Fibonacci-hemisphere push–hold–return trajectory through the coupled VIC
env and prints mean steady-state ``ft_wrist`` force per direction during holds.

Run from the repository root::

    uv sync --extra gym --extra vic --extra dev
    uv run python apple_pick_gym/examples/example_gym_sysid.py --viewer null

With Newton GL viewer (requires a display)::

    uv run python apple_pick_gym/examples/example_gym_sysid.py --viewer gl

Canonical one-direction run (2 cm increments, 10 cm total)::

    uv run python apple_pick_gym/examples/example_gym_sysid.py \\
      --n-directions 1 --movement-per-step-m 0.02 --total-movement-m 0.10 \\
      --move-speed-mps 0.2

Headless smoke (cap env steps)::

    uv run python apple_pick_gym/examples/example_gym_sysid.py --viewer null \\
      --n-directions 1 --max-steps 200

Dataset collection writes Parquet observations by default; add ``--save-snapshot``
only when a privileged ``initial_states/*.npz`` baseline is needed::

    uv run python apple_pick_gym/examples/example_gym_sysid.py --viewer null \\
      --n-directions 1 --max-steps 200 --output /tmp/sysid_dataset

    uv run python apple_pick_gym/examples/example_gym_sysid.py --viewer null \\
      --n-directions 1 --max-steps 200 --save-snapshot \\
      --output /tmp/sysid_dataset_with_snapshot

Per-step trajectory and summary prints (off by default)::

    uv run python apple_pick_gym/examples/example_gym_sysid.py --debug --viewer null \\
      --n-directions 1 --max-steps 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

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


def _fmt_pos_cm(pos: np.ndarray) -> str:
    p = np.asarray(pos, dtype=np.float64).reshape(3)
    return (
        f"({p[0]*100:+.1f},{p[1]*100:+.1f},{p[2]*100:+.1f})"
    )


def _fmt_vel_mps(vel: np.ndarray) -> str:
    v = np.asarray(vel, dtype=np.float64).reshape(3)
    return f"({v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f})"


def _fmt_force_n(force: np.ndarray) -> str:
    f = np.asarray(force, dtype=np.float64).reshape(3)
    return f"({f[0]:+.3f},{f[1]:+.3f},{f[2]:+.3f})"


def _debug_flag_in_argv() -> bool:
    return "--debug" in sys.argv


def _make_debug_printer(enabled: bool):
    def debug_print(*args, **kwargs) -> None:
        if enabled:
            print(*args, **kwargs)

    return debug_print


def _fmt_dir(v: np.ndarray, *, threshold: float = 1e-6) -> str:
    u = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(u))
    if n < threshold:
        return "n/a"
    u = u / n
    return f"({u[0]:+.3f},{u[1]:+.3f},{u[2]:+.3f})"


def _tcp_target_pos(env) -> np.ndarray:
    return np.asarray(env._controller.target_tf[:3], dtype=np.float64)


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
        "--hold-duration-s",
        type=float,
        default=1.5,
        help="Zero-velocity hold after each increment [s] (default 1.5).",
    )
    p.add_argument(
        "--movement-per-step-m",
        type=float,
        default=0.05,
        help="Distance per fast move burst [m] (default 0.05 = 5 cm).",
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
        help="Linear speed during move/return bursts [m/s] (default 0.2).",
    )
    p.add_argument(
        "--skip-return",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Teleport to grasp pose between directions instead of physical return (default on).",
    )
    p.add_argument(
        "--mujoco-viewer",
        action="store_true",
        help="Open MuJoCo passive viewer for the FR3 arm (requires a GUI session).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to write Parquet trajectory dataset (metadata + per-episode frames).",
    )
    p.add_argument(
        "--save-snapshot",
        action="store_true",
        help="Also write privileged initial_states/*.npz baseline data (default off).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print per-step trajectory logs, summaries, and operational notices.",
    )
    return p


def _maybe_log_forces(viewer: object, obs: dict, info: dict, *, amplitude_m: float) -> None:
    log = getattr(viewer, "log_scalar", None)
    if log is None:
        return

    ft = obs.get("ft_wrist")
    if ft is not None:
        w = np.asarray(ft, dtype=np.float64).reshape(6)
        f = w[:3]
        log("SysID |F| wrist [N]", float(np.linalg.norm(f)), smoothing=3)
        log("SysID Fx wrist [N]", float(f[0]), smoothing=3)
        log("SysID Fy wrist [N]", float(f[1]), smoothing=3)
        log("SysID Fz wrist [N]", float(f[2]), smoothing=3)
        log("SysID amp [m]", float(amplitude_m), smoothing=3)

    ee = info.get("end_effector_wrench")
    if ee is not None:
        w = np.asarray(ee, dtype=np.float64).reshape(6)
        log("SysID |F| TCP harvest [N]", float(np.linalg.norm(w[:3])), smoothing=3)


def main() -> None:
    debug_print = _make_debug_printer(_debug_flag_in_argv())

    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--num-frames", "1"])
            debug_print(
                "No DISPLAY/WAYLAND_DISPLAY: using --viewer null (override with --viewer gl)."
            )

    parser = _make_parser()
    import newton.examples

    viewer, args = newton.examples.init(parser=parser)
    debug_print = _make_debug_printer(bool(args.debug))

    from apple_pick_gym.envs import ApplePickSysIdEnv
    from apple_pick_sim.fruiting_system import fruiting_params_to_json
    from apple_pick_sim.system_id import (
        EpisodeMeta,
        ExcitationContext,
        QuasiStaticTrajectory,
        TrajectoryWriter,
        estimate_trajectory_frames,
        grasp_snapshot_from_env,
        sample_robot_facing_pull_directions,
    )

    config = _trajectory_config_from_args(args)
    n_directions = int(args.n_directions)
    max_steps = int(args.max_steps)
    if max_steps <= 0:
        max_steps = estimate_trajectory_frames(config, n_directions) + 64

    trajectory_writer = TrajectoryWriter(episode_id=str(uuid4())) if args.output else None

    env = ApplePickSysIdEnv(
        render_mode=None,
        max_episode_steps=max_steps,
        fix_to_apple=bool(args.fix_to_apple),
        fix_to_apple_warmup_substeps=int(args.fix_to_apple_warmup_substeps),
        control_hz=config.control_hz,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    obs, info = env.reset(seed=int(args.seed))
    reset_obs = obs
    reset_info = dict(info)
    _reset_weld_direction = reset_info.get("weld_direction")

    if trajectory_writer is not None and bool(args.save_snapshot):
        snapshot = grasp_snapshot_from_env(
            env,
            obs=reset_obs,
            weld_direction=_reset_weld_direction,
        )
        snapshot_path = trajectory_writer.save_initial_state(Path(args.output), snapshot)
        debug_print(f"Saved initial state snapshot to {snapshot_path}")

    scene = env._scene
    if scene is None:
        raise RuntimeError("Env did not create a scene; did reset() succeed?")

    tcp_target = np.asarray(env._controller.target_tf[:3], dtype=np.float64)
    apple_pos = np.asarray(obs["apple_pos"], dtype=np.float64)
    grasp_axis = _normalize(apple_pos - tcp_target)
    robot_base_pos = np.asarray(reset_info["robot_base_pos"], dtype=np.float64)
    robot_vec = robot_base_pos - apple_pos

    cable = scene.cable
    stem_bodies = cable.stem_bodies
    assert len(stem_bodies) >= 2
    body_q = cable.state_0.body_q.numpy().reshape(-1, 7)
    physical_stem_dir = _normalize(
        body_q[int(stem_bodies[-1]), :3] - body_q[int(stem_bodies[-2]), :3]
    )
    directions = sample_robot_facing_pull_directions(n_directions, physical_stem_dir, robot_vec)
    traj = QuasiStaticTrajectory(directions, config)

    viewer.set_model(scene.cable.model)
    sim_time = 0.0
    render_dt = 1.0 / float(args.hz)

    mujoco_viewer = bool(getattr(args, "mujoco_viewer", False))
    if mujoco_viewer and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        debug_print("Suppressing --mujoco-viewer (no DISPLAY/WAYLAND_DISPLAY).")
        mujoco_viewer = False

    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    hold_forces: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    prev_phase: str | None = None
    dir_idx = -1
    recorded_steps = 0

    debug_print(f"Quasi-static sys-ID: {n_directions} directions, up to {max_steps} env steps")
    debug_print(
        f"Trajectory: {config.movement_per_step_m*100:.1f} cm/step, "
        f"{config.total_movement_m*100:.1f} cm total, "
        f"{config.move_speed_mps:.2f} m/s burst, {config.hold_duration_s:.1f} s hold"
    )
    debug_print(
        f"Physical stem (base → tip): "
        f"({physical_stem_dir[0]:+.3f}, {physical_stem_dir[1]:+.3f}, {physical_stem_dir[2]:+.3f})"
    )
    debug_print(
        f"Grasp axis (TCP → apple): "
        f"({grasp_axis[0]:+.3f}, {grasp_axis[1]:+.3f}, {grasp_axis[2]:+.3f})"
    )

    wall_start = time.perf_counter()

    def _render_frame(
        frame_obs: dict,
        frame_info: dict,
        t: float,
        phase: str,
        *,
        expected_pos: np.ndarray,
        command_vel: np.ndarray,
        actual_pos: np.ndarray,
        linear_velocity: tuple[float, float, float] | None = None,
        step_wall_s: float | None = None,
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
        pos_err_mm = float(np.linalg.norm(expected_pos - actual_pos)) * 1000.0
        f_wrist = ft[:3]
        excitation_dir = frame_obs.get("excitation_direction")
        cmd_dir = (
            np.asarray(excitation_dir, dtype=np.float64).reshape(3)
            if excitation_dir is not None
            else command_vel
        )
        wall_elapsed_s = time.perf_counter() - wall_start
        step_wall_str = (
            f"  step_wall={step_wall_s:.3f}s"
            if step_wall_s is not None
            else ""
        )
        debug_print(
            f"  step phase={phase:8s} dir={dir_idx:2d} step={traj.current_step_index:1d}, \n"
            f"amp={amp*100:5.1f} cm  |F|={np.linalg.norm(f_wrist):6.2f} N"
            f"  F={_fmt_force_n(f_wrist)} N  F_dir={_fmt_dir(f_wrist)}, \n"
            f"cmd_d={_fmt_dir(cmd_dir)}  "
            f"exp_cm={_fmt_pos_cm(expected_pos)}, \n"
            f"cmd_mps={_fmt_vel_mps(command_vel)}, \n"
            f"act_cm={_fmt_pos_cm(actual_pos)}, \n"
            f"err={pos_err_mm:5.1f} mm  sim={t:.3f}s  wall={wall_elapsed_s:.3f}s{step_wall_str}",
            flush=True,
        )

    try:
        init_expected = _tcp_target_pos(env)
        init_actual = np.asarray(obs["tcp_pos"], dtype=np.float64)
        _render_frame(
            obs,
            info,
            sim_time,
            "init",
            expected_pos=init_expected,
            command_vel=np.zeros(3, dtype=np.float64),
            actual_pos=init_actual,
        )

        for step_idx, (phase, vel) in enumerate(traj.iter_frames()):
            if step_idx >= max_steps:
                debug_print(f"\nStopped at --max-steps={max_steps}")
                break
            if not viewer.is_running():
                debug_print("\nViewer closed.")
                break

            if phase == "move_out" and prev_phase != "move_out":
                new_direction = False
                if prev_phase in (None, "init", "return"):
                    new_direction = True
                elif (
                    config.skip_return
                    and prev_phase == "hold"
                    and traj.current_step_index == 0
                ):
                    new_direction = True
                    env.restore_grasp_pose()

                if new_direction:
                    dir_idx += 1
                    debug_print(
                        f"\n--- Direction {dir_idx}: ({directions[dir_idx][0]:+.3f}, "
                        f"{directions[dir_idx][1]:+.3f}, {directions[dir_idx][2]:+.3f}) ---"
                    )

            ctx = ExcitationContext(
                type="quasi_static",
                f_inst=0.0,
                direction=traj.current_direction,
            )
            env.set_excitation_context(ctx)

            action = _action_from_velocity(vel)
            step_wall_start = time.perf_counter()
            obs, _reward, terminated, truncated, info = env.step(action)
            sim_time += 1.0 / config.control_hz

            if trajectory_writer is not None:
                trajectory_writer.record_step(
                    step_idx=step_idx,
                    sim_time=sim_time,
                    phase=phase,
                    dir_idx=dir_idx,
                    amplitude_m=traj.current_amplitude_m,
                    action=action,
                    obs=obs,
                )
                recorded_steps += 1

            scene = env._scene
            if scene is None:
                break

            if phase == "hold" and dir_idx >= 0:
                key = (dir_idx, traj.current_step_index)
                hold_forces[key].append(np.asarray(obs["ft_wrist"][:3], dtype=np.float64))

            _render_frame(
                obs,
                info,
                sim_time,
                phase,
                expected_pos=_tcp_target_pos(env),
                command_vel=np.asarray(action[:3], dtype=np.float64),
                actual_pos=np.asarray(obs["tcp_pos"], dtype=np.float64),
                linear_velocity=vel.linear,
                step_wall_s=time.perf_counter() - step_wall_start,
            )
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
                debug_print(f"\nTerminated at step {step_idx}.")
                break
            if truncated:
                debug_print(f"\nTruncated at step {step_idx} (max_episode_steps={max_steps}).")
                break

        debug_print("\nMean steady-state wrist force [N] per direction and increment (hold phases):")
        for dir_idx in sorted({k[0] for k in hold_forces}):
            d = directions[dir_idx]
            for step_idx in sorted(k[1] for k in hold_forces if k[0] == dir_idx):
                samples = hold_forces[(dir_idx, step_idx)]
                if not samples:
                    continue
                mean_f = np.mean(np.stack(samples, axis=0), axis=0)
                amp_cm = (step_idx + 1) * config.movement_per_step_m * 100.0
                debug_print(
                    f"  dir {dir_idx:2d} step {step_idx} amp={amp_cm:4.1f} cm"
                    f"  cmd_d=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})"
                    f"  F=({mean_f[0]:+.3f},{mean_f[1]:+.3f},{mean_f[2]:+.3f})"
                    f"  F_dir={_fmt_dir(mean_f)}"
                )

        if trajectory_writer is not None and recorded_steps > 0:
            weld = reset_info.get("weld_direction")
            if weld is None:
                weld = _reset_weld_direction
            if weld is None:
                weld_arr = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                weld_arr = np.asarray(weld, dtype=np.float64).reshape(3)
            weld_norm = float(np.linalg.norm(weld_arr))
            if weld_norm < 1e-12:
                weld_tuple = (0.0, 0.0, 1.0)
            else:
                weld_tuple = (
                    float(weld_arr[0] / weld_norm),
                    float(weld_arr[1] / weld_norm),
                    float(weld_arr[2] / weld_norm),
                )
            params_fp = reset_info.get("params_fingerprint", {})
            meta = EpisodeMeta(
                episode_id=trajectory_writer.episode_id,
                weld_direction=weld_tuple,
                excitation_type="quasi_static",
                n_woody_parts=int(reset_info.get("n_woody_parts", 0)),
                junction_names=list(env.unwrapped.junction_names),
                params_fingerprint=json.dumps(params_fp, sort_keys=True),
                fruiting_system_params=fruiting_params_to_json(scene.cable.params),
                control_hz=float(config.control_hz),
                timestamp=datetime.now(timezone.utc).isoformat(),
                seed=int(args.seed),
                n_directions=n_directions,
                initial_tcp_pos=tuple(
                    float(x) for x in np.asarray(reset_obs["tcp_pos"]).reshape(3)
                ),
                initial_tcp_quat=tuple(
                    float(x) for x in np.asarray(reset_obs["tcp_quat"]).reshape(4)
                ),
                initial_apple_pos=tuple(
                    float(x) for x in np.asarray(reset_obs["apple_pos"]).reshape(3)
                ),
                initial_apple_quat=tuple(
                    float(x) for x in np.asarray(reset_obs["apple_quat"]).reshape(4)
                ),
                initial_robot_joint_q=tuple(
                    float(x) for x in np.asarray(reset_obs["robot_joint_q"]).reshape(-1)
                ),
                fixture_path=str(env.unwrapped._fixture_ranges_path()),
                fruiting_base_pos=(
                    None
                    if reset_info.get("fruiting_base_pos") is None
                    else tuple(
                        float(x) for x in np.asarray(reset_info["fruiting_base_pos"]).reshape(3)
                    )
                ),
                apple_radius=(
                    None
                    if scene.cable.params.apple_radius is None
                    else float(scene.cable.params.apple_radius)
                ),
                rod_radii=reset_info.get("rod_radii"),
                weld_reference_pos=tuple(
                    float(x) for x in np.asarray(reset_obs["apple_pos"]).reshape(3)
                ),
                weld_reference_quat=tuple(
                    float(x) for x in np.asarray(reset_obs["apple_quat"]).reshape(4)
                ),
                movement_per_step_m=float(config.movement_per_step_m),
                total_movement_m=float(config.total_movement_m),
                hold_duration_s=float(config.hold_duration_s),
                move_speed_mps=float(config.move_speed_mps),
                skip_return=bool(config.skip_return),
            )
            output_dir = Path(args.output)
            frames_path = trajectory_writer.save(output_dir, meta)
            debug_print(
                f"\nSaved {recorded_steps} frames to {frames_path}\n"
                f"Metadata appended to {output_dir / 'metadata.parquet'}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
