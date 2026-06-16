"""Visualize Fibonacci pull directions from the live ApplePickSysIdEnv code path.

Pull directions use the same sampling as ``example_gym_sysid.py``: pole =
``stem_perpendicular_robot_pole(physical_stem, robot_vec)`` with a default
90° hemisphere (``max_polar_angle=π/2``) around that pole.

Run from the repository root::

    uv run python apple_pick_gym/examples/visualize_pull_directions.py \\
        --seed 0 --n-directions 10 --output pull_directions.png

Multiple resets (different welds per row)::

    uv run python apple_pick_gym/examples/visualize_pull_directions.py \\
        --seed 0 --n-directions 10 --n-resets 8 \\
        --fix-to-apple-warmup-substeps 0 --output multi_reset.png

Interactive display::

    uv run python apple_pick_gym/examples/visualize_pull_directions.py --show
"""

from __future__ import annotations

import argparse
import sys


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualize Fibonacci pull directions from ApplePickSysIdEnv.",
    )
    p.add_argument("--seed", type=int, default=0, help="Env reset seed (first reset only)")
    p.add_argument(
        "--n-directions",
        type=int,
        default=10,
        help="Fibonacci hemisphere samples per reset",
    )
    p.add_argument(
        "--n-resets",
        type=int,
        default=1,
        help="Number of env.reset() calls (cycles weld direction; default 1)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="pull_directions.png",
        help="PNG output path (default: pull_directions.png in cwd)",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Also open an interactive matplotlib window",
    )
    p.add_argument(
        "--fix-to-apple",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Weld gripper to apple (default on; weld arrow omitted when off)",
    )
    p.add_argument(
        "--fix-to-apple-warmup-substeps",
        type=int,
        default=1800,
        help="VBD settle substeps before welding (fix_to_apple only)",
    )
    p.add_argument(
        "--max-polar-angle-deg",
        type=float,
        default=90.0,
        help="Polar cap half-angle around pull pole (default 90° hemisphere)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)

    from apple_pick_gym.envs import ApplePickSysIdEnv
    import numpy as np
    from apple_pick_sim.system_id.pull_direction_viz import (
        collect_pull_direction_geometry,
        render_multi_reset_figure,
        render_pull_direction_figure,
    )

    n_resets = max(1, int(args.n_resets))
    max_polar_angle = float(np.deg2rad(args.max_polar_angle_deg))
    env = ApplePickSysIdEnv(
        render_mode=None,
        max_episode_steps=2,
        fix_to_apple=bool(args.fix_to_apple),
        fix_to_apple_warmup_substeps=int(args.fix_to_apple_warmup_substeps),
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    try:
        geoms = []
        for reset_idx in range(n_resets):
            if reset_idx == 0:
                obs, info = env.reset(seed=int(args.seed))
            else:
                obs, info = env.reset()
            geoms.append(
                collect_pull_direction_geometry(
                    env,
                    obs,
                    n_directions=int(args.n_directions),
                    reset_index=reset_idx,
                    max_polar_angle=max_polar_angle,
                )
            )

        title_suffix = f"seed={args.seed}, n_dirs={args.n_directions}"
        if n_resets > 1:
            title_suffix += f", n_resets={n_resets}"
            render_multi_reset_figure(
                geoms,
                args.output,
                show=bool(args.show),
                title_suffix=title_suffix,
            )
        else:
            geom = geoms[0]
            if "weld_direction" in info:
                weld = info["weld_direction"]
                title_suffix += (
                    f", weld=({weld[0]:+.2f},{weld[1]:+.2f},{weld[2]:+.2f})"
                )
            render_pull_direction_figure(
                geom,
                args.output,
                show=bool(args.show),
                title_suffix=title_suffix,
            )
            _print_geometry_summary(geom)
    finally:
        env.close()

    return 0


def _print_geometry_summary(geom) -> None:
    print(
        f"Physical stem (base → tip): "
        f"({geom.physical_stem_dir[0]:+.3f}, {geom.physical_stem_dir[1]:+.3f}, {geom.physical_stem_dir[2]:+.3f})"
    )
    print(
        f"Grasp axis (TCP → apple): "
        f"({geom.stem_dir[0]:+.3f}, {geom.stem_dir[1]:+.3f}, {geom.stem_dir[2]:+.3f})"
    )
    print(
        f"Pull pole (stem⊥, robot-facing): "
        f"({geom.robot_dir[0]:+.3f}, {geom.robot_dir[1]:+.3f}, {geom.robot_dir[2]:+.3f})"
    )
    if geom.weld_dir is not None:
        print(
            f"Weld direction (env): "
            f"({geom.weld_dir[0]:+.3f}, {geom.weld_dir[1]:+.3f}, {geom.weld_dir[2]:+.3f})"
        )


if __name__ == "__main__":
    sys.exit(main())
