"""Parallel batched sys-ID data collection (V.4.2).

Collect ``num_structures × num_directions`` quasi-static episodes in one GPU
batched run and write Parquet compatible with ``example_gym_replay_overrides.py``.

Run from the repository root::

    uv run python apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py \\
      --viewer null --num-structures 2 --num-directions 3 \\
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

import newton.examples


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
        "--debug",
        action="store_true",
        help="Print collection progress messages.",
    )
    return p


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
        )
        if bool(args.debug):
            print(f"Saved batched sys-ID dataset to {out}")
    finally:
        env.close()
        if hasattr(viewer, "close"):
            viewer.close()


if __name__ == "__main__":
    main()
