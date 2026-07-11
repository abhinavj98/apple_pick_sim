"""Keyboard teleop demo for :class:`ApplePickBatchedVicEnv`.

Runs the batched Gymnasium env and feeds the same world-frame TCP velocity command
to every parallel world. With ``--viewer gl``, focus the Newton window and use the
standard FR3 key map (``read_keyboard_ee_velocity``). Headless / ``--viewer null`` uses
the same keys in the terminal.

Run from repo root::

    uv run python apple_pick_gym/batched_examples/example_batched_gym_keyboard.py \\
        --viewer gl --num-envs 4 --seed 42

Headless smoke::

    uv run python apple_pick_gym/batched_examples/example_batched_gym_keyboard.py \\
        --viewer null --num-envs 2 --max-steps 60
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
import torch

from apple_pick_gym.batched_envs import ApplePickBatchedVicEnv
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.robot.fr3_robot.controllers.keyboard import EEVelocity, print_fr3_keyboard_bindings


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
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return None
    return sys.stdin.read(1)


def _drain_terminal_keys() -> list[str]:
    keys: list[str] = []
    while True:
        ch = _read_key_nonblocking()
        if ch is None:
            break
        keys.append(ch)
    return keys


def _axis_from_keys(keys: list[str], neg_key: str, pos_key: str) -> float:
    val = 0.0
    for ch in keys:
        c = ch.lower()
        if c == neg_key:
            val -= 1.0
        if c == pos_key:
            val += 1.0
    return val


_RESET_KEY = "p"


def _terminal_keys_to_velocity(
    keys: list[str],
    *,
    linear_speed: float,
    angular_speed: float,
) -> tuple[EEVelocity, bool, bool]:
    """Return (twist, quit_requested, reset_requested)."""
    if any(ch.lower() == "q" for ch in keys):
        return EEVelocity(), True, False
    reset_requested = any(ch.lower() == _RESET_KEY for ch in keys)
    lin = (
        _axis_from_keys(keys, "k", "i") * linear_speed,
        _axis_from_keys(keys, "l", "j") * linear_speed,
        _axis_from_keys(keys, "f", "r") * linear_speed,
    )
    ang = (
        _axis_from_keys(keys, "x", "z") * angular_speed,
        _axis_from_keys(keys, "g", "t") * angular_speed,
        _axis_from_keys(keys, "o", "u") * angular_speed,
    )
    return EEVelocity(linear=lin, angular=ang), False, reset_requested


def _viewer_reset_requested(viewer: object, reset_key_was_down: bool) -> tuple[bool, bool]:
    """Return (reset_requested, reset_key_down_now) on rising edge."""
    if not hasattr(viewer, "is_key_down"):
        return False, False
    fr3_robot.poll_viewer_events(viewer)
    down = bool(viewer.is_key_down(_RESET_KEY))
    return down and not reset_key_was_down, down


def _read_frame_input(
    viewer: object,
    env: ApplePickBatchedVicEnv,
    terminal_keys: list[str] | None,
    *,
    reset_key_was_down: bool,
) -> tuple[EEVelocity, bool, bool, bool]:
    """Return (twist, quit_requested, reset_requested, reset_key_down_now)."""
    ctrl = env._sim.config.controller
    if hasattr(viewer, "is_key_down"):
        vel = fr3_robot.read_keyboard_ee_velocity(
            viewer,
            linear_speed=float(ctrl.linear_speed),
            angular_speed=float(ctrl.angular_speed),
            poll_events=False,
        )
        reset_requested, reset_key_down = _viewer_reset_requested(viewer, reset_key_was_down)
        return vel, False, reset_requested, reset_key_down
    vel, quit_requested, reset_requested = _terminal_keys_to_velocity(
        terminal_keys or [],
        linear_speed=float(ctrl.linear_speed),
        angular_speed=float(ctrl.angular_speed),
    )
    return vel, quit_requested, reset_requested, False


def _velocity_to_actions(vel: EEVelocity, env: ApplePickBatchedVicEnv) -> torch.Tensor:
    row = torch.tensor(
        [*vel.linear, *vel.angular],
        dtype=torch.float32,
        device=env.device,
    )
    return row.unsqueeze(0).expand(env.num_envs, -1).contiguous()


def _render_frame(viewer: object, env: ApplePickBatchedVicEnv, sim_time: float) -> None:
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
    viewer.end_frame()


def _make_parser() -> argparse.ArgumentParser:
    p = newton.examples.create_parser()
    p.add_argument("--num-envs", type=int, default=4, help="Parallel batched worlds.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument(
        "--use-settle-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse settle snapshots across runs (default: off).",
    )
    p.add_argument(
        "--linear-speed",
        type=float,
        default=None,
        help="Override controller linear speed (m/s); default uses gym_defaults.",
    )
    p.add_argument(
        "--angular-speed",
        type=float,
        default=None,
        help="Override controller angular speed (rad/s); default uses gym_defaults.",
    )
    return p


def _make_env(args: argparse.Namespace) -> ApplePickBatchedVicEnv:
    import dataclasses

    from apple_pick_sim.coupled_fruiting import BatchedHeterogeneousCoupledSimConfig

    cfg = BatchedHeterogeneousCoupledSimConfig.gym_defaults(num_envs=int(args.num_envs))
    if args.linear_speed is not None or args.angular_speed is not None:
        ctrl = cfg.controller
        cfg = dataclasses.replace(
            cfg,
            controller=dataclasses.replace(
                ctrl,
                linear_speed=float(ctrl.linear_speed if args.linear_speed is None else args.linear_speed),
                angular_speed=float(
                    ctrl.angular_speed if args.angular_speed is None else args.angular_speed
                ),
            ),
        )
    return ApplePickBatchedVicEnv(
        num_envs=int(args.num_envs),
        max_episode_steps=int(args.max_steps),
        topology_seed=int(args.seed),
        use_settle_cache=bool(args.use_settle_cache),
        sim_config=cfg,
    )


def main() -> None:
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--max-steps", "120"])
            print("No DISPLAY/WAYLAND_DISPLAY: using --viewer null --max-steps 120.")

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)

    env = _make_env(args)
    obs, info = env.reset(seed=int(args.seed))
    sim = env._sim

    graphical = isinstance(viewer, newton.viewer.ViewerGL)
    viewer.set_model(sim.scene.cable.model)
    if graphical and env.num_envs > 1:
        viewer.set_world_offsets(tuple(sim.config.runtime.env_spacing))

    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    frame_dt = float(sim.frame_dt)
    sim_time = 0.0

    if graphical:
        print_fr3_keyboard_bindings()
        print(f"  {_RESET_KEY}: reset episode")
    else:
        print("Terminal keyboard control (press 'q' to quit):")
        print("  i/k ±X, j/l ±Y, r/f ±Z, z/x ±rotX, t/g ±rotY, u/o ±rotZ, space noop")
        print(f"  {_RESET_KEY}: reset episode")
    print(f"num_envs={env.num_envs} device={env.device} obs_layout={info['obs_layout']}")

    try:
        _render_frame(viewer, env, sim_time)

        terminal_mode = not hasattr(viewer, "is_key_down")
        ctx = _raw_terminal_mode() if terminal_mode else _null_context()
        with ctx:
            step = -1
            reset_key_was_down = False
            for step in range(int(args.max_steps)):
                if not viewer.is_running():
                    break

                terminal_keys = _drain_terminal_keys() if terminal_mode else None
                vel, quit_requested, reset_requested, reset_key_was_down = _read_frame_input(
                    viewer,
                    env,
                    terminal_keys,
                    reset_key_was_down=reset_key_was_down,
                )
                if quit_requested:
                    break
                if reset_requested:
                    obs, info = env.reset()
                    sim_time = 0.0
                    print(f"Episode reset at step {step + 1} (manual '{_RESET_KEY}')")
                    _render_frame(viewer, env, sim_time)
                    if graphical:
                        time.sleep(max(0.0, frame_dt))
                    continue

                actions = _velocity_to_actions(vel, env)
                obs, _reward, terminated, truncated, info = env.step(actions)
                del obs
                sim_time += frame_dt
                _render_frame(viewer, env, sim_time)

                if bool(terminated.any()) or bool(truncated.any()):
                    obs, info = env.reset()
                    print(f"Episode reset at step {step + 1} (truncated={bool(truncated.any())})")

                if graphical:
                    time.sleep(max(0.0, frame_dt))
    finally:
        env.close()

    print(f"Done ({step + 1} steps, sim_time={sim_time:.3f}s).")


@contextmanager
def _null_context():
    yield


if __name__ == "__main__":
    main()
