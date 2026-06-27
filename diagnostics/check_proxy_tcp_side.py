"""Diagnostic: check whether the gripper proxy and robot TCP are on the same side of the apple.

Run with:
  PYTHONPATH=$(pwd) uv run diagnostics/check_proxy_tcp_side.py \
      --robot fr3 --fix-to-apple \
      --json apple_pick_sim/fixtures/fruiting_system_ranges_example_variance_soft.json
"""

import argparse
import sys
import numpy as np

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--robot", choices=["fr3", "placeholder"], default="fr3")
    parser.add_argument("--fix-to-apple", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robot-base-pos", type=float, nargs=3, default=None)
    args = parser.parse_args()

    from apple_pick_sim.fruiting_system.params import load_ranges
    from apple_pick_sim.fruiting_system import GripperProxyConfig
    from apple_pick_sim.coupled_fruiting.builders import (
        build_coupled_fruiting_fr3,
        build_coupled_fruiting_placeholder,
    )
    from apple_pick_sim.robot import fr3_robot

    ranges = load_ranges(args.json)

    robot_base_pos = tuple(args.robot_base_pos) if args.robot_base_pos else None

    proxy_cfg = GripperProxyConfig(
        fix_to_apple=args.fix_to_apple,
        robot_facing_weld=args.fix_to_apple,
        **({"mass": fr3_robot.EE_MASS_KG} if args.robot == "fr3" else {}),
    )

    print(f"Building scene (robot={args.robot}, fix_to_apple={args.fix_to_apple}, seed={args.seed}) ...")

    if args.robot == "fr3":
        scene = build_coupled_fruiting_fr3(
            ranges,
            args.seed,
            gripper_proxy=proxy_cfg,
            robot_base_pos=robot_base_pos,
        )
    else:
        scene = build_coupled_fruiting_placeholder(
            ranges,
            args.seed,
            gripper_proxy=proxy_cfg,
            robot_base_pos=robot_base_pos,
        )

    cable = scene.cable

    if cable.apple_body is None:
        print("ERROR: No apple body in scene. Check JSON has apple ranges.")
        sys.exit(1)

    bq = cable.state_0.body_q.numpy().reshape(-1, 7)
    apple_pos  = bq[cable.apple_body][:3]
    proxy_pos  = bq[cable.gripper_proxy_body][:3]
    tcp_pos    = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[scene.tcp_body_index][:3]

    apple_to_proxy = proxy_pos - apple_pos
    apple_to_tcp   = tcp_pos   - apple_pos

    dot = float(np.dot(apple_to_proxy, apple_to_tcp))
    norm_proxy = float(np.linalg.norm(apple_to_proxy))
    norm_tcp   = float(np.linalg.norm(apple_to_tcp))
    cos_angle  = dot / (norm_proxy * norm_tcp + 1e-12)
    angle_deg  = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    print()
    print("=" * 60)
    print(f"apple  pos  : ({apple_pos[0]:.4f}, {apple_pos[1]:.4f}, {apple_pos[2]:.4f})")
    print(f"proxy  pos  : ({proxy_pos[0]:.4f}, {proxy_pos[1]:.4f}, {proxy_pos[2]:.4f})")
    print(f"tcp    pos  : ({tcp_pos[0]:.4f}, {tcp_pos[1]:.4f}, {tcp_pos[2]:.4f})")
    print()
    print(f"apple→proxy : ({apple_to_proxy[0]:.4f}, {apple_to_proxy[1]:.4f}, {apple_to_proxy[2]:.4f})  |len={norm_proxy:.4f} m")
    print(f"apple→tcp   : ({apple_to_tcp[0]:.4f}, {apple_to_tcp[1]:.4f}, {apple_to_tcp[2]:.4f})  |len={norm_tcp:.4f} m")
    print()
    print(f"dot(proxy, tcp) = {dot:.4f}")
    print(f"angle between   = {angle_deg:.1f} deg")
    print()
    if dot > 0:
        print(f"✓ SAME SIDE  (angle={angle_deg:.1f}° < 90°)")
    else:
        print(f"✗ OPPOSITE SIDES  (angle={angle_deg:.1f}° > 90°)  ← BUG")
    print("=" * 60)


if __name__ == "__main__":
    main()
