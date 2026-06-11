"""Run §2.1 quasi-static stepped stiffness mapping (headless smoke).

From the repository root::

    uv run python apple_pick_sim/system_id/run_quasi_static.py --viewer null
"""

from __future__ import annotations

import argparse
import math
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


def _estimate_trajectory_frames(n_directions: int, config: "QuasiStaticStepConfig") -> int:
    move_frames = max(
        1, int(math.ceil(config.step_size_m / config.move_speed_mps * config.control_hz))
    )
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quasi-static stiffness mapping smoke run")
    parser.add_argument("--viewer", default="null", choices=("null",), help="Viewer backend (null only)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-directions", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = run full trajectory")
    parser.add_argument(
        "--max-tcp-force-n",
        type=float,
        default=1000.0,
        help="Wrench safety stop [N]; post-grasp transients can exceed 500 N on step 0",
    )
    args = parser.parse_args(argv)

    if args.viewer != "null":
        print("Only --viewer null is supported for this smoke script.", file=sys.stderr)
        return 2

    from apple_pick_gym.envs import ApplePickSysIdEnv
    from apple_pick_sim.system_id import (
        ExcitationContext,
        QuasiStaticStepConfig,
        QuasiStaticTrajectory,
        sample_fibonacci_hemisphere,
    )

    config = QuasiStaticStepConfig()
    max_steps = int(args.max_steps)
    if max_steps <= 0:
        max_steps = _estimate_trajectory_frames(int(args.n_directions), config) + 64

    env = ApplePickSysIdEnv(
        render_mode=None,
        max_episode_steps=max_steps,
        max_tcp_force_n=float(args.max_tcp_force_n),
        mujoco_solver_kwargs={"disable_contacts": True},
        control_hz=config.control_hz,
    )
    obs, _ = env.reset(seed=args.seed)

    tcp_target = np.asarray(env._controller.target_tf[:3], dtype=np.float64)
    stem_dir = _normalize(np.asarray(obs["apple_pos"], dtype=np.float64) - tcp_target)
    directions = sample_fibonacci_hemisphere(args.n_directions, stem_dir)

    traj = QuasiStaticTrajectory(directions, config)

    print(
        f"Quasi-static smoke: {args.n_directions} directions, up to {max_steps} env steps, "
        f"wrench guard {args.max_tcp_force_n:.0f} N"
    )

    hold_forces: dict[int, list[np.ndarray]] = defaultdict(list)
    prev_phase: str | None = None
    dir_idx = -1

    for step_idx, (phase, vel) in enumerate(traj.iter_frames()):
        if step_idx >= max_steps:
            print(f"Stopped at --max-steps={max_steps}")
            break

        if phase == "move_out" and prev_phase != "move_out":
            dir_idx += 1

        direction = traj.current_direction
        ctx = ExcitationContext(type="quasi_static", f_inst=0.0, direction=direction)
        env.set_excitation_context(ctx)

        action = _action_from_velocity(vel)
        obs, _, terminated, truncated, _ = env.step(action)

        if phase == "hold" and dir_idx >= 0:
            hold_forces[dir_idx].append(np.asarray(obs["ft_wrist"][:3], dtype=np.float64))

        prev_phase = phase

        if terminated or truncated:
            force_n = float(np.linalg.norm(obs["ft_wrist"][:3]))
            print(
                f"Stopped early at step {step_idx}: terminated={terminated} truncated={truncated} "
                f"|F|={force_n:.1f} N (guard={args.max_tcp_force_n:.0f} N)"
            )
            break

    print("Mean steady-state wrist force [N] per direction (hold phases):")
    for dir_idx in sorted(hold_forces):
        samples = hold_forces[dir_idx]
        if not samples:
            print(f"  dir {dir_idx:2d}: (no hold samples)")
            continue
        mean_f = np.mean(np.stack(samples, axis=0), axis=0)
        direction = directions[dir_idx]
        print(
            f"  dir {dir_idx:2d}  d=({direction[0]:+.3f},{direction[1]:+.3f},{direction[2]:+.3f})"
            f"  F=({mean_f[0]:+.3f},{mean_f[1]:+.3f},{mean_f[2]:+.3f})"
        )

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
