"""Keyboard + render demo for the Gymnasium env `ApplePickCoupled-v0`.

Run from repo root:

    cd newton && uv sync --extra examples --extra dev && cd ..
    PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_gym/examples/example_gym_keyboard.py --viewer gl

Keys (terminal, not the viewer window):
  i/k: ±X, j/l: ±Y, r/f: ±Z
  z/x: ±rotX, t/g: ±rotY, u/o: ±rotZ
  space: noop
  q: quit
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager

import newton
import newton.examples
import numpy as np


@contextmanager
def _raw_terminal_mode():
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key_nonblocking() -> str | None:
    if not sys.stdin.isatty():
        return None
    r, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not r:
        return None
    ch = sys.stdin.read(1)
    return ch


def _key_to_action(ch: str | None) -> int | None:
    if ch is None:
        return None
    c = ch.lower()
    if c == "q":
        return -1
    if c == " ":
        return 12
    # Match the FR3 keyboard conventions in apple_pick_sim (world-frame).
    return {
        "i": 0,  # +X
        "k": 1,  # -X
        "j": 2,  # +Y
        "l": 3,  # -Y
        "r": 4,  # +Z
        "f": 5,  # -Z
        "z": 6,  # +rotX
        "x": 7,  # -rotX
        "t": 8,  # +rotY
        "g": 9,  # -rotY
        "u": 10,  # +rotZ
        "o": 11,  # -rotZ
    }.get(c)


def _make_parser() -> argparse.ArgumentParser:
    p = newton.examples.create_parser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--hz", type=float, default=30.0, help="Gym steps per second (render rate).")
    p.add_argument(
        "--fix-to-apple",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Weld the proxy to the apple so the apple co-teleports with robot motion.",
    )
    p.add_argument(
        "--mujoco-viewer",
        action="store_true",
        help="Open MuJoCo passive viewer to render the FR3 arm (requires a GUI session).",
    )
    return p


def _maybe_log_forces(viewer: object, info: dict) -> None:
    """Log force telemetry into the viewer time-series graph (if supported)."""
    log = getattr(viewer, "log_scalar", None)
    if log is None:
        return

    ee = info.get("end_effector_wrench", None)
    if ee is not None:
        w = np.asarray(ee, dtype=np.float64).reshape(6)
        fmag = float(np.linalg.norm(w[:3]))
        tmag = float(np.linalg.norm(w[3:]))
        log("Gym EE |F| [N]", fmag, smoothing=3)
        log("Gym EE |τ| [N·m]", tmag, smoothing=3)
        for axis, idx in zip("xyz", range(3), strict=True):
            log(f"Gym EE F{axis} [N]", float(w[idx]), smoothing=3)
            log(f"Gym EE τ{axis} [N·m]", float(w[3 + idx]), smoothing=3)

    links = info.get("fruiting_link_forces", None)
    if not isinstance(links, dict):
        return
    for key, rec in links.items():
        f = np.asarray(rec.get("force_world", (0.0, 0.0, 0.0)), dtype=np.float64).reshape(3)
        t = np.asarray(
            rec.get("torque_at_child_com_world", (0.0, 0.0, 0.0)), dtype=np.float64
        ).reshape(3)
        fmag = float(np.linalg.norm(f))
        tmag = float(np.linalg.norm(t))
        log(f"Gym FJ {key} |F| [N]", fmag, smoothing=3)
        log(f"Gym FJ {key} |τ| [N·m]", tmag, smoothing=3)


def main() -> None:
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--num-frames", "1"])
            print("No DISPLAY/WAYLAND_DISPLAY: using --viewer null (override with --viewer gl).")

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)

    import gymnasium as gym
    import apple_pick_gym  # noqa: F401 (registers env)

    env = gym.make(
        "ApplePickCoupled-v0",
        render_mode=None,
        max_episode_steps=int(args.max_steps),
        enable_self_collisions=False,
        fix_to_apple=bool(getattr(args, "fix_to_apple", False)),
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    env.reset(seed=int(args.seed))

    scene = env.unwrapped._scene
    if scene is None:
        raise RuntimeError("Env did not create a scene; did reset() succeed?")

    viewer.set_model(scene.cable.model)
    sim_time = 0.0
    dt = 1.0 / float(args.hz)

    mujoco_viewer = bool(getattr(args, "mujoco_viewer", False))
    if mujoco_viewer and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("Suppressing --mujoco-viewer (no DISPLAY/WAYLAND_DISPLAY).")
        mujoco_viewer = False

    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    print("Terminal keyboard control (press 'q' to quit):")
    print("  i/k ±X, j/l ±Y, r/f ±Z, z/x ±rotX, t/g ±rotY, u/o ±rotZ, space noop")

    try:
        with _raw_terminal_mode():
            for step in range(int(args.max_steps)):
                if not viewer.is_running():
                    break

                key = _read_key_nonblocking()
                action = _key_to_action(key)
                if action == -1:
                    break
                if action is None:
                    action = 12  # noop by default

                _obs, _reward, _terminated, _truncated, info = env.step(action)
                sim_time += dt

                scene = env.unwrapped._scene
                if scene is None:
                    break

                if scene.last_vbd_contacts is not None:
                    viz_contacts = scene.last_vbd_contacts
                else:
                    viz_contacts = scene.cable.model.collide(
                        scene.cable.state_0,
                        collision_pipeline=scene.cable_collision_pipeline,
                    )

                viewer.begin_frame(sim_time)
                viewer.log_state(scene.cable.state_0)
                viewer.log_contacts(viz_contacts, scene.cable.state_0)
                _maybe_log_forces(viewer, info)
                viewer.end_frame()

                if mujoco_viewer and scene.robot_model is not None:
                    from apple_pick_sim.robot import fr3_robot

                    fr3_robot.sync_mujoco_visual_state(
                        scene.mj_solver,
                        scene.robot_model,
                        scene.robot_state_0,
                    )
                    scene.mj_solver.render_mujoco_viewer()

                time.sleep(max(0.0, dt))
    finally:
        env.close()


if __name__ == "__main__":
    main()

