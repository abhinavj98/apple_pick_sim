"""Keyboard teleop over a Young's-modulus E-grid (shared topology).

Samples one shared-topology base fruiting system, applies a Cartesian product of
``log10(E)`` candidates for primary/spur/stem, and feeds the same world-frame TCP
velocity to every parallel world. Soft-disables unstable envs (sticky zero
actions) like the batched sys-ID grid.

Run from repo root::

    uv run python apple_pick_gym/batched_examples/example_batched_youngs_modulus_keyboard.py \\
        --viewer gl \\
        --log10-e-primary 8.0,8.5 --log10-e-spur 7.5 --log10-e-stem 7.0

Headless smoke::

    uv run python apple_pick_gym/batched_examples/example_batched_youngs_modulus_keyboard.py \\
        --viewer null --max-steps 60 \\
        --log10-e-primary 8.0,8.5 --log10-e-spur 7.5 --log10-e-stem 7.0
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from pathlib import Path

import newton
import newton.examples
import torch

from apple_pick_gym.batched_envs import ApplePickBatchedVicEnv
from apple_pick_gym.batched_envs.batched_stability_monitor import (
    BatchedStabilityMonitor,
    hard_blowup_mask,
    ik_bootstrap_unstable_mask,
)
from apple_pick_gym.batched_envs.env_disable_controller import EnvDisableController
from apple_pick_gym.batched_examples._youngs_e_grid_cli import candidates_from_log10_cli
from apple_pick_sim.fruiting_system import default_ranges_fixture_path, load_ranges, sample_params
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


@contextmanager
def _null_context():
    yield


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
    p.add_argument(
        "--log10-e-primary",
        type=str,
        default="8.0,8.5",
        help="Comma-separated log10(E_primary) grid values.",
    )
    p.add_argument(
        "--log10-e-spur",
        type=str,
        default="7.5",
        help="Comma-separated log10(E_spur) grid values.",
    )
    p.add_argument(
        "--log10-e-stem",
        type=str,
        default="7.0",
        help="Comma-separated log10(E_stem) grid values.",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for shared-topology base params.")
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument(
        "--ranges-path",
        type=str,
        default=None,
        help="Fruiting ranges JSON (default: project fixture).",
    )
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
        help="Override controller linear speed (m/s).",
    )
    p.add_argument(
        "--angular-speed",
        type=float,
        default=None,
        help="Override controller angular speed (rad/s).",
    )
    return p


def _make_env(args: argparse.Namespace) -> tuple[ApplePickBatchedVicEnv, list]:
    from apple_pick_sim.coupled_fruiting import BatchedHeterogeneousCoupledSimConfig

    candidates = candidates_from_log10_cli(
        log10_e_primary=str(args.log10_e_primary),
        log10_e_spur=str(args.log10_e_spur),
        log10_e_stem=str(args.log10_e_stem),
    )
    ranges_path = (
        Path(args.ranges_path) if args.ranges_path else default_ranges_fixture_path()
    )
    ranges = load_ranges(ranges_path)
    base = sample_params(ranges, seed=int(args.seed), omit=("secondary",))
    per_env_params = [c.apply_to(base) for c in candidates]
    num_envs = len(per_env_params)

    cfg = BatchedHeterogeneousCoupledSimConfig.gym_defaults(num_envs=num_envs)
    if args.linear_speed is not None or args.angular_speed is not None:
        ctrl = cfg.controller
        cfg = dataclasses.replace(
            cfg,
            controller=dataclasses.replace(
                ctrl,
                linear_speed=float(
                    ctrl.linear_speed if args.linear_speed is None else args.linear_speed
                ),
                angular_speed=float(
                    ctrl.angular_speed if args.angular_speed is None else args.angular_speed
                ),
            ),
        )
    env = ApplePickBatchedVicEnv(
        num_envs=num_envs,
        max_episode_steps=int(args.max_steps),
        topology_seed=int(args.seed),
        ranges_path=ranges_path,
        use_settle_cache=bool(args.use_settle_cache),
        sim_config=cfg,
        per_env_params=per_env_params,
    )
    return env, candidates


def main() -> None:
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--max-steps", "120"])
            print("No DISPLAY/WAYLAND_DISPLAY: using --viewer null --max-steps 120.")

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)

    env, candidates = _make_env(args)
    obs, info = env.reset(seed=int(args.seed))
    sim = env._sim

    initial_unstable = ik_bootstrap_unstable_mask(env, env.num_envs)
    monitor = BatchedStabilityMonitor(
        env.num_envs,
        known_obs_keys=set(obs.keys()),
        initial_unstable=initial_unstable,
    )
    disable_ctrl = EnvDisableController(
        env.num_envs,
        device=env.device,
        initial_disabled=initial_unstable,
    )
    if bool(initial_unstable.any()):
        bad = torch.where(initial_unstable)[0].tolist()
        print(f"IK-bootstrap soft-disable envs={bad}")

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

    print(f"num_envs={env.num_envs} (= # YoungsModulusCandidate) device={env.device}")
    for i, c in enumerate(candidates):
        print(f"  env[{i}] {c.short_label()}  E=({c.primary:.3g},{c.spur:.3g},{c.stem:.3g}) Pa")
    print(f"obs_layout={info['obs_layout']}")

    prev_disabled = disable_ctrl.disabled.clone()

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

                actions = disable_ctrl.apply_actions(_velocity_to_actions(vel, env))
                obs, _reward, terminated, truncated, info = env.step(actions)
                step_report = monitor.check(obs, step_idx=int(step))
                disable_ctrl.update(hard_blowup_mask(step_report))
                newly = disable_ctrl.disabled & ~prev_disabled
                if bool(newly.any()):
                    print(
                        f"soft-disable envs={torch.where(newly)[0].tolist()} "
                        f"at step {step + 1}"
                    )
                    prev_disabled = disable_ctrl.disabled.clone()

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


if __name__ == "__main__":
    main()
