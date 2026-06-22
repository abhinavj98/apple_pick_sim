"""Run §2.1 quasi-static stepped stiffness mapping (headless smoke).

From the repository root::

    uv run python apple_pick_sim/system_id/run_quasi_static.py --viewer null
"""

from __future__ import annotations

import argparse
import sys
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


def _trajectory_config_from_args(args: argparse.Namespace):
    from apple_pick_sim.system_id import QuasiStaticStepConfig

    return QuasiStaticStepConfig(
        movement_per_step_m=float(args.movement_per_step_m),
        total_movement_m=float(args.total_movement_m),
        move_speed_mps=float(args.move_speed_mps),
        hold_duration_s=float(args.hold_duration_s),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quasi-static stiffness mapping smoke run")
    parser.add_argument("--viewer", default="null", choices=("null",), help="Viewer backend (null only)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-directions", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = run full trajectory")
    parser.add_argument(
        "--movement-per-step-m",
        type=float,
        default=0.02,
        help="Distance per fast move burst [m] (default 0.02 = 2 cm).",
    )
    parser.add_argument(
        "--total-movement-m",
        type=float,
        default=0.10,
        help="Total push excursion per direction [m] (default 0.10 = 10 cm).",
    )
    parser.add_argument(
        "--move-speed-mps",
        type=float,
        default=0.2,
        help="Linear speed during move/return bursts [m/s] (default 0.2).",
    )
    parser.add_argument(
        "--hold-duration-s",
        type=float,
        default=1.5,
        help="Zero-velocity hold after each increment [s] (default 1.5).",
    )
    args = parser.parse_args(argv)

    if args.viewer != "null":
        print("Only --viewer null is supported for this smoke script.", file=sys.stderr)
        return 2

    from apple_pick_gym.envs import ApplePickSysIdEnv
    from apple_pick_sim.system_id import (
        ExcitationContext,
        QuasiStaticTrajectory,
        estimate_trajectory_frames,
        sample_robot_facing_pull_directions,
    )
    from apple_pick_sim.tests.conftest import COUPLED_ROBOT_BASE_POS

    config = _trajectory_config_from_args(args)
    max_steps = int(args.max_steps)
    if max_steps <= 0:
        max_steps = estimate_trajectory_frames(config, int(args.n_directions)) + 64

    env = ApplePickSysIdEnv(
        render_mode=None,
        max_episode_steps=max_steps,
        mujoco_solver_kwargs={"disable_contacts": True},
        control_hz=config.control_hz,
    )
    obs, _ = env.reset(seed=args.seed)

    apple_pos = np.asarray(obs["apple_pos"], dtype=np.float64)
    robot_vec = np.asarray(COUPLED_ROBOT_BASE_POS, dtype=np.float64) - apple_pos

    cable = env._scene.cable
    stem_bodies = cable.stem_bodies
    assert len(stem_bodies) >= 2
    body_q = cable.state_0.body_q.numpy().reshape(-1, 7)
    physical_stem = body_q[int(stem_bodies[-1]), :3] - body_q[int(stem_bodies[-2]), :3]
    physical_stem = _normalize(physical_stem)

    directions = sample_robot_facing_pull_directions(args.n_directions, physical_stem, robot_vec)

    traj = QuasiStaticTrajectory(directions, config)

    print(
        f"Quasi-static smoke: {args.n_directions} directions, up to {max_steps} env steps"
    )
    print(
        f"Trajectory: {config.movement_per_step_m*100:.1f} cm/step, "
        f"{config.total_movement_m*100:.1f} cm total, "
        f"{config.move_speed_mps:.2f} m/s burst, {config.hold_duration_s:.1f} s hold"
    )

    hold_forces: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    prev_phase: str | None = None
    dir_idx = -1

    for step_idx, (phase, vel) in enumerate(traj.iter_frames()):
        if step_idx >= max_steps:
            print(f"Stopped at --max-steps={max_steps}")
            break

        if phase == "move_out" and prev_phase != "move_out":
            new_direction = False
            if prev_phase in (None, "return"):
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

        direction = traj.current_direction
        ctx = ExcitationContext(type="quasi_static", f_inst=0.0, direction=direction)
        env.set_excitation_context(ctx)

        action = _action_from_velocity(vel)
        obs, _, terminated, truncated, _ = env.step(action)

        if phase == "hold" and dir_idx >= 0:
            hold_forces[(dir_idx, traj.current_step_index)].append(
                np.asarray(obs["ft_wrist"][:3], dtype=np.float64)
            )

        prev_phase = phase

        if terminated or truncated:
            force_n = float(np.linalg.norm(obs["ft_wrist"][:3]))
            print(
                f"Stopped early at step {step_idx}: terminated={terminated} truncated={truncated} "
                f"|F|={force_n:.1f} N"
            )
            break

    print("Mean steady-state wrist force [N] per direction and increment (hold phases):")
    for d_idx in sorted({k[0] for k in hold_forces}):
        direction = directions[d_idx]
        for step_idx in sorted(k[1] for k in hold_forces if k[0] == d_idx):
            samples = hold_forces[(d_idx, step_idx)]
            if not samples:
                continue
            mean_f = np.mean(np.stack(samples, axis=0), axis=0)
            amp_cm = (step_idx + 1) * config.movement_per_step_m * 100.0
            print(
                f"  dir {d_idx:2d} step {step_idx} amp={amp_cm:4.1f} cm"
                f"  d=({direction[0]:+.3f},{direction[1]:+.3f},{direction[2]:+.3f})"
                f"  F=({mean_f[0]:+.3f},{mean_f[1]:+.3f},{mean_f[2]:+.3f})"
            )

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
