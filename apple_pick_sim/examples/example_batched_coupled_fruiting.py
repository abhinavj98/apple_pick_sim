"""Batched coupled fruiting — see ``docs/vectorized-coupled-fruiting.md`` for the flow.

Canonical steps: build N free worlds → VBD settle in parallel → build N welded worlds
and seed from settled state → keyboard teleop (FR3: per-env IK scatter; placeholder: world-0 broadcast).

Run from the repository root::

    uv run python apple_pick_sim/examples/example_batched_coupled_fruiting.py \\
      --num-envs 4 --fix-to-apple --fr3-keyboard --viewer gl --seed 42

Optional **MuJoCo** passive viewer (second window): pass ``--mujoco-viewer`` with a graphics session.

With per-env noisy teleop (scatter IK, not broadcast)::

    uv run python apple_pick_sim/examples/example_batched_coupled_fruiting.py \\
      --num-envs 4 --fix-to-apple --fr3-keyboard --noisy-action --viewer gl --seed 42
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import secrets
import sys
from pathlib import Path

import numpy as np
import newton
import warp as wp
import newton.examples

from apple_pick_sim.sim_device import resolve_sim_device
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.coupled_fruiting import (
    CoupledFruitingScene,
    broadcast_joint_q_from_world0,
    build_batched_coupled_fruiting_fr3,
    build_batched_coupled_fruiting_placeholder,
    seed_fix_to_apple_from_settled,
    quiet_all_cable_bodies,
    settle_vbd_substeps,
)
from apple_pick_sim.fruiting_system import GripperProxyConfig, default_ranges_fixture_path, load_ranges
from apple_pick_sim.batched_viz import log_batched_endpoints, log_batched_tcp_force_arrows
from apple_pick_sim.coupled_fruiting.batched_robot_status import print_batched_robot_status


def _default_ranges_path() -> Path:
    return default_ranges_fixture_path()


def _fix_to_apple_from_args(args: argparse.Namespace | None) -> bool:
    return bool(getattr(args, "fix_to_apple", False)) if args else False


def _enable_self_collisions_from_args(args: argparse.Namespace | None) -> bool:
    return bool(getattr(args, "enable_self_collision", False)) if args else False


def _gripper_proxy_from_args(
    args: argparse.Namespace | None,
    *,
    robot_kind: str,
) -> GripperProxyConfig:
    fix = _fix_to_apple_from_args(args)
    if robot_kind == "fr3":
        return GripperProxyConfig(
            mass=fr3_robot.EE_MASS_KG,
            fix_to_apple=fix,
            robot_facing_weld=fix,
        )
    return GripperProxyConfig(fix_to_apple=fix, robot_facing_weld=fix)


def _reject_unsupported_flags(args: argparse.Namespace) -> None:
    if getattr(args, "controller", "direct") == "vic":
        raise SystemExit("VIC is not supported in the batched example (use --controller direct or ee).")
    if bool(getattr(args, "only_vbd", False)) or bool(getattr(args, "only_mjc", False)):
        raise SystemExit("--only-vbd and --only-mjc are not supported in the batched example.")


def _resolve_robot_kind(args: argparse.Namespace) -> str:
    robot_kind = str(getattr(args, "robot", "fr3"))
    if robot_kind == "fr3" and not fr3_robot.fr3_assets_available():
        print(
            "Warning: FR3 assets not found under assets/fr3/; falling back to placeholder TCP.",
            file=sys.stderr,
        )
        return "placeholder"
    return robot_kind


def _make_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help=(
            "Path to fruiting-system range JSON (default: "
            "apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json)."
        ),
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=30.0,
        help="Target frame rate [Hz] (default: 30.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for the scene. Omit for a random seed on each run.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of replicated coupled worlds (default: 4).",
    )
    parser.add_argument(
        "--env-spacing",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[2.5, 2.5, 0.0],
        help="Spacing between replicated worlds [m] (default: 2.5 2.5 0).",
    )
    parser.add_argument(
        "--enable-self-collision",
        action="store_true",
        help="Enable intra-chain self collisions on the coupled cable scene (off by default).",
    )
    parser.add_argument(
        "--robot",
        type=str,
        choices=("placeholder", "fr3"),
        default="fr3",
        help=(
            "Robot model for Model A: bundled FR3+EE (default) or placeholder free TCP "
            "(assets/fr3; requires usd-core)."
        ),
    )
    parser.add_argument(
        "--controller",
        type=str,
        choices=("direct", "ee"),
        default="direct",
        help=(
            "World-0 teleop: direct (kinematic joint_q) or ee (velocity IK + MuJoCo PD). "
            "VIC is not supported in the batched example."
        ),
    )
    parser.add_argument(
        "--fr3-keyboard",
        action="store_true",
        help=(
            "Enable FR3 TCP keyboard teleop on world 0 (requires ``--viewer gl``; "
            "focus the window — I/K J/L R/F translate, U/O T/G Z/X rotate)."
        ),
    )
    parser.add_argument(
        "--fix-to-apple",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Weld gripper proxy to apple (stem-harvest + co-teleport apple with TCP). "
            "Uses settle-then-weld initialization. Default: off (velocity-delta harvest)."
        ),
    )
    parser.add_argument(
        "--settle-substeps",
        type=int,
        default=1000,
        help=(
            "VBD substeps for free-proxy settling before weld when --fix-to-apple "
            "(default: 1000, same as example_coupled_fruiting.py)."
        ),
    )
    parser.add_argument(
        "--scripted-ee-vel",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "VZ"),
        default=[0.05, 0.0, 0.0],
        help="Scripted world-frame TCP linear velocity [m/s] when keyboard is off (default: 0.05 0 0).",
    )
    parser.add_argument(
        "--status-every",
        type=int,
        default=60,
        help="Print world-0 apple z, per-env TCP command velocity/position every N frames (0=off, default 60).",
    )
    parser.add_argument(
        "--print-robot-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --status-every, print each env's robot base pose, Newton joint_q, "
            "MuJoCo qpos, and PD targets (default: on)."
        ),
    )
    parser.add_argument(
        "--noisy-action",
        action="store_true",
        help=(
            "Add Gaussian noise to the teleop velocity each frame (FR3 only). "
            "Noise is shared across envs; template IK scatter still runs per arm."
        ),
    )
    parser.add_argument(
        "--noisy-action-std",
        type=float,
        default=0.02,
        help="Gaussian std [m/s and rad/s per twist component] when --noisy-action (default: 0.02).",
    )
    parser.add_argument(
        "--tcp-force-arrow",
        action="store_true",
        help=(
            "Draw harvested TCP force as a yellow world-frame arrow at each env's robot TCP "
            "(requires --viewer gl or viser; use --tcp-force-scale to tune length)."
        ),
    )
    parser.add_argument(
        "--tcp-force-scale",
        type=float,
        default=0.02,
        help="Arrow length per newton [m/N] for --tcp-force-arrow (default: 0.02 → 50 N ≈ 1 m).",
    )
    parser.add_argument(
        "--tcp-force-arrow-gain",
        type=float,
        default=1.0,
        help="Dimensionless multiplier on --tcp-force-scale (default: 1).",
    )
    parser.add_argument(
        "--tcp-force-min-length",
        type=float,
        default=0.08,
        help="Minimum arrow length [m] when force is above threshold (default: 0.08).",
    )
    parser.add_argument(
        "--tcp-force-max-length",
        type=float,
        default=1.5,
        help="Maximum arrow length [m]; 0 disables cap (default: 1.5).",
    )
    parser.add_argument(
        "--mark-endpoints",
        action="store_true",
        help=(
            "Draw red XYZ crosses at each env's apple and gripper-proxy markers, "
            "plus red spheres at woody fixed-joint parent anchors "
            "(requires --viewer gl or viser; use --endpoint-radius for cross half-size / dot radius)."
        ),
    )
    parser.add_argument(
        "--endpoint-radius",
        type=float,
        default=0.05,
        help="Sphere radius [m] for --mark-endpoints (default: 0.05).",
    )
    parser.add_argument(
        "--mujoco-viewer",
        action="store_true",
        help=(
            "Open MuJoCo's passive viewer for the robot model after each Newton frame "
            "(separate window; requires --viewer gl and a GUI session)."
        ),
    )
    return parser


class ExampleBatchedCoupledFruiting:
    """N homogeneous coupled stacks; FR3 teleop scatters IK per env, placeholder broadcasts world 0."""

    def __init__(self, viewer, args: argparse.Namespace | None = None):
        self.viewer = viewer
        self.args = args
        _reject_unsupported_flags(args or argparse.Namespace())

        self.fps = float(getattr(args, "hz", 30.0)) if args else 30.0
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 30
        self.sim_dt = (1.0 / 60.0) / self.sim_substeps
        self.sim_time = 0.0
        self._frame = 0

        json_path = getattr(args, "json", None) if args else None
        ranges_path = Path(json_path) if json_path else _default_ranges_path()
        self.ranges = load_ranges(ranges_path)

        seed = getattr(args, "seed", None) if args else None
        if seed is None:
            seed = secrets.randbelow(2**31 - 1)

        self.num_envs = int(getattr(args, "num_envs", 4))
        env_spacing_raw = getattr(args, "env_spacing", [2.5, 2.5, 0.0]) if args else [2.5, 2.5, 0.0]
        self.env_spacing = tuple(float(v) for v in env_spacing_raw)

        sim_device = resolve_sim_device(getattr(args, "device", None) if args else None)
        robot_kind = _resolve_robot_kind(args or argparse.Namespace())
        controller_mode = str(getattr(args, "controller", "direct"))
        fix_to_apple = _fix_to_apple_from_args(args)
        enable_self = _enable_self_collisions_from_args(args)
        settle_substeps = int(getattr(args, "settle_substeps", 1000))

        print(f"Batched coupled fruiting ranges: {ranges_path}")
        print(f"Initial seed: {seed}")
        print(f"Warp device: {sim_device}")
        print(
            f"Gripper proxy fix_to_apple={fix_to_apple} "
            f"({'stem-harvest / settle-then-weld' if fix_to_apple else 'velocity-delta'} coupling)."
        )

        build_fn = (
            build_batched_coupled_fruiting_fr3
            if robot_kind == "fr3"
            else build_batched_coupled_fruiting_placeholder
        )
        gripper = _gripper_proxy_from_args(args, robot_kind=robot_kind)
        build_kw = dict(
            device=sim_device,
            num_envs=self.num_envs,
            env_spacing=self.env_spacing,
            enable_self_collisions=enable_self,
            gripper_proxy=gripper,
        )
        if robot_kind == "fr3" and not fix_to_apple:
            build_kw["robot_base_from_proxy"] = True
        if robot_kind == "fr3":
            fr3_robot.enable_ik_bootstrap_warnings_for_examples()

        if fix_to_apple:
            settled = build_fn(
                self.ranges,
                seed,
                **{
                    **build_kw,
                    "num_envs": self.num_envs,
                    "vbd_only": True,
                    "gripper_proxy": dataclasses.replace(
                        gripper, fix_to_apple=False, robot_facing_weld=False
                    ),
                },
            )
            settle_vbd_substeps(settled, substeps=settle_substeps, dt=self.sim_dt)
            quiet_all_cable_bodies(settled.cable)
            self.scene = build_fn(
                self.ranges,
                seed,
                **{
                    **build_kw,
                    "gripper_proxy": dataclasses.replace(
                        gripper, fix_to_apple=True, robot_facing_weld=True
                    ),
                    "skip_ik_bootstrap": True,
                    "defer_template_robot_bootstrap": True,
                },
            )
            seed_fix_to_apple_from_settled(
                welded_scene=self.scene,
                settled_scene=settled,
                quiet_apple_proxy=True,
            )
        else:
            self.scene: CoupledFruitingScene = build_fn(self.ranges, seed, **build_kw)

        self.layout = self.scene.layout
        if self.layout is None:
            raise RuntimeError("batched scene missing layout")

        print(
            f"Batched coupled fruiting: num_envs={self.layout.num_envs} "
            f"spacing={self.env_spacing} "
            f"registry_pairs={len(self.scene.proxy_registry.robot_to_proxy)} "
            f"cable_world_count={self.scene.cable.model.world_count} "
            f"robot_world_count={self.scene.robot_model.world_count}"
        )

        self._controller_mode = controller_mode
        self._robot_kind = robot_kind
        self._ee_ctrl: (
            fr3_robot.Fr3BatchedEEVelocityController
            | fr3_robot.Fr3BatchedEEDirectJointController
            | None
        ) = None
        self._scripted_velocity = fr3_robot.EEVelocity(
            linear=tuple(float(v) for v in getattr(args, "scripted_ee_vel", [0.05, 0.0, 0.0])),
        )
        self._teleop_velocity = self._scripted_velocity
        self._use_keyboard = bool(getattr(args, "fr3_keyboard", False))
        self._noisy_action_std = float(getattr(args, "noisy_action_std", 0.02))
        self._action_noise_rng = np.random.default_rng(int(seed) + 911)
        noisy_requested = bool(getattr(args, "noisy_action", False))
        if noisy_requested and robot_kind != "fr3":
            print("Warning: --noisy-action requires FR3; ignoring.", file=sys.stderr)
        self._noisy_action = noisy_requested and robot_kind == "fr3"

        if robot_kind == "fr3" and self.scene.robot_model is not None:
            self._ee_ctrl = self._configure_fr3_controller(controller_mode)
        elif controller_mode == "ee":
            print("Note: --controller ee requires FR3; running without teleop.", file=sys.stderr)

        if self._use_keyboard:
            if robot_kind != "fr3" or self._ee_ctrl is None:
                print("Warning: --fr3-keyboard requires FR3.", file=sys.stderr)
            elif not hasattr(self.viewer, "is_key_down"):
                print("Warning: --fr3-keyboard requires --viewer gl.", file=sys.stderr)
            else:
                fr3_robot.print_fr3_keyboard_bindings()

        self._status_every = int(getattr(args, "status_every", 60))
        self._print_robot_state = bool(getattr(args, "print_robot_state", True))

        if self._print_robot_state and self.scene.robot_model is not None and self.layout is not None:
            if self.scene.mj_solver is not None:
                fr3_robot.sync_mujoco_visual_state(
                    self.scene.mj_solver,
                    self.scene.robot_model,
                    self.scene.robot_state_0,
                )
            print("Initial batched robot state (post-build):", flush=True)
            print_batched_robot_status(self.scene, self.layout, prefix="")

        self._tcp_force_arrow = bool(getattr(args, "tcp_force_arrow", False))
        tcp_force_scale = float(getattr(args, "tcp_force_scale", 0.02))
        tcp_force_gain = float(getattr(args, "tcp_force_arrow_gain", 1.0))
        tcp_force_min_len = float(getattr(args, "tcp_force_min_length", 0.08))
        tcp_force_max_len = float(getattr(args, "tcp_force_max_length", 1.5))
        if tcp_force_scale <= 0.0:
            raise ValueError("--tcp-force-scale must be positive")
        if tcp_force_gain <= 0.0:
            raise ValueError("--tcp-force-arrow-gain must be positive")
        if tcp_force_min_len < 0.0:
            raise ValueError("--tcp-force-min-length must be >= 0")
        if tcp_force_max_len < 0.0:
            raise ValueError("--tcp-force-max-length must be >= 0")
        self._tcp_force_scale = tcp_force_scale
        self._tcp_force_gain = tcp_force_gain
        self._tcp_force_min_length = tcp_force_min_len
        self._tcp_force_max_length = tcp_force_max_len

        self._mark_endpoints = bool(getattr(args, "mark_endpoints", False))
        endpoint_radius = float(getattr(args, "endpoint_radius", 0.05))
        if endpoint_radius <= 0.0:
            raise ValueError("--endpoint-radius must be positive")
        self._endpoint_radius = endpoint_radius

        self.viewer.set_model(self.scene.cable.model)
        graphical = isinstance(viewer, newton.viewer.ViewerGL)
        has_robot = self.scene.robot_model is not None
        self._mujoco_viewer = (
            has_robot
            and bool(getattr(args, "mujoco_viewer", False))
            and graphical
        )
        if self._mujoco_viewer and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            print("Suppressing --mujoco-viewer (no DISPLAY/WAYLAND_DISPLAY).")
            self._mujoco_viewer = False

        if self._tcp_force_arrow and graphical:
            cap = (
                f"{self._tcp_force_max_length:.2f} m max"
                if self._tcp_force_max_length > 0.0
                else "no max"
            )
            print(
                "TCP force arrows: yellow at each env's robot TCP; "
                f"scale={self._tcp_force_scale:.4f} m/N × gain {self._tcp_force_gain:g}, "
                f"min {self._tcp_force_min_length:.2f} m, {cap}."
            )
        if self._mark_endpoints and graphical:
            print(
                f"Endpoint markers: red XYZ crosses at apple + proxy; "
                f"red dots at woody parent anchors "
                f"(radius/half-size={self._endpoint_radius:.3f} m)."
            )
        if graphical and self.num_envs > 1:
            self.viewer.set_world_offsets(self.env_spacing)

        self._viz_contacts = self.scene.cable.model.collide(
            self.scene.cable.state_0,
            collision_pipeline=self.scene.cable_collision_pipeline,
        )

    def _configure_fr3_controller(
        self, mode: str
    ) -> (
        fr3_robot.Fr3BatchedEEVelocityController
        | fr3_robot.Fr3BatchedEEDirectJointController
    ):
        ik_kw = fr3_robot.batched_ik_teleop_kwargs(self.scene)
        if not ik_kw:
            raise RuntimeError("batched FR3 scene missing template IK layout")
        if self._noisy_action:
            velocity_for_world = self._noisy_velocity_for_world
            print(
                f"Batched FR3 noisy teleop: per-env Gaussian std={self._noisy_action_std} "
                f"(template IK scatter)."
            )
        else:
            velocity_for_world = lambda w: self._teleop_velocity  # noqa: ARG005
        if mode == "ee":
            self.scene.robot_kinematic_mode = False
            ctrl = fr3_robot.Fr3BatchedEEVelocityController(
                self.scene.robot_model,
                linear_speed=1.0,
                angular_speed=5.0,
                velocity_for_world=velocity_for_world,
                **ik_kw,
            )
            print("Batched FR3 ee controller (template IK, all envs).")
        else:
            self.scene.robot_kinematic_mode = True
            ctrl = fr3_robot.Fr3BatchedEEDirectJointController(
                self.scene.robot_model,
                linear_speed=1.0,
                angular_speed=5.0,
                velocity_for_world=velocity_for_world,
                **ik_kw,
            )
            print("Batched FR3 direct controller (template IK, all envs).")
        ctrl.sync_target_from_state(self.scene.robot_state_0)
        return ctrl

    def _noisy_velocity_for_world(self, world: int) -> fr3_robot.EEVelocity:
        del world
        return fr3_robot.add_gaussian_noise_to_ee_velocity(
            self._teleop_velocity,
            rng=self._action_noise_rng,
            std=self._noisy_action_std,
        )

    def _teleop_world0(self) -> None:
        if self._robot_kind == "fr3" and self._ee_ctrl is not None:
            velocity = self._teleop_velocity
            if self._use_keyboard and hasattr(self.viewer, "is_key_down"):
                velocity = fr3_robot.read_keyboard_ee_velocity(
                    self.viewer,
                    linear_speed=self._ee_ctrl.linear_speed,
                    angular_speed=self._ee_ctrl.angular_speed,
                )
            else:
                velocity = self._scripted_velocity
            self._teleop_velocity = velocity
            if self._controller_mode == "direct":
                self.scene.update_fr3_ee_teleop_direct(
                    self.frame_dt,
                    self._ee_ctrl,
                    viewer=self.viewer if self._use_keyboard else None,
                    velocity=velocity,
                )
            else:
                self.scene.update_fr3_ee_teleop(
                    self.frame_dt,
                    self._ee_ctrl,
                    viewer=self.viewer if self._use_keyboard else None,
                    velocity=velocity,
                )
        elif self._robot_kind == "placeholder":
            self._nudge_placeholder_world0()

    def _nudge_placeholder_world0(self) -> None:
        """Scripted +X drift for the free-floating placeholder TCP on world 0."""
        layout = self.layout
        assert layout is not None
        model = self.scene.robot_model
        if model is None or self.scene.robot_state_0 is None:
            return
        vx = float(self._scripted_velocity.linear[0])
        jq = model.joint_q.numpy().copy()
        sl = layout.joint_q_slice(0)
        jq[sl][0] += vx * self.frame_dt
        model.joint_q.assign(jq)
        self.scene.robot_state_0.joint_q.assign(jq)

    def simulate(self) -> None:
        self._teleop_world0()
        if self._robot_kind == "placeholder":
            broadcast_joint_q_from_world0(self.scene, self.layout)
        for _ in range(self.sim_substeps):
            self.scene.coupled_substep(self.sim_dt)

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt
        self._frame += 1
        if self._status_every > 0 and self._frame % self._status_every == 0:
            self._print_status()

    def _print_status(self) -> None:
        layout = self.layout
        if layout is None or layout.num_envs < 1:
            return
        body_q = self.scene.cable.state_0.body_q.numpy().reshape(-1, 7)
        apple0 = layout.apple_body_indices[0]
        if apple0 >= 0:
            z0 = float(body_q[apple0, 2])
            msg = f"t={self.sim_time:6.2f}s world-0 apple z={z0:.4f}"
            if layout.num_envs > 1:
                apple1 = layout.apple_body_indices[1]
                if apple1 >= 0:
                    z1 = float(body_q[apple1, 2])
                    msg += f"  Δz(env1-env0)={z1 - z0:.4f}"
            print(msg, flush=True)
        if self.scene.robot_model is None:
            return
        for w in range(layout.num_envs):
            vel = self._command_velocity_for_world(w)
            tx, ty, tz = self._tcp_world_position(w)
            lx, ly, lz = vel.linear
            print(
                f"  env{w}: v=({lx:+.3f},{ly:+.3f},{lz:+.3f}) m/s  "
                f"tcp=({tx:.3f},{ty:.3f},{tz:.3f}) m",
                flush=True,
            )
        if self._print_robot_state:
            if self.scene.mj_solver is not None:
                fr3_robot.sync_mujoco_visual_state(
                    self.scene.mj_solver,
                    self.scene.robot_model,
                    self.scene.robot_state_0,
                )
            print_batched_robot_status(
                self.scene,
                layout,
                prefix=f"t={self.sim_time:6.2f}s ",
            )

    def _command_velocity_for_world(self, world: int) -> fr3_robot.EEVelocity:
        if self._ee_ctrl is not None:
            return self._ee_ctrl.command_velocity_for_world(
                world, fallback=self._teleop_velocity
            )
        return self._teleop_velocity

    def _tcp_world_position(self, world: int) -> tuple[float, float, float]:
        layout = self.layout
        assert layout is not None
        if self._ee_ctrl is not None:
            pos = wp.transform_get_translation(self._ee_ctrl.tcp_world_pose(world))
            return float(pos[0]), float(pos[1]), float(pos[2])
        state = self.scene.robot_state_0
        if state is None:
            return (0.0, 0.0, 0.0)
        tcp_idx = layout.tcp_body_indices[world]
        bq = state.body_q.numpy().reshape(-1, 7)[tcp_idx]
        return float(bq[0]), float(bq[1]), float(bq[2])

    def render(self) -> None:
        if self.scene.last_vbd_contacts is not None:
            self._viz_contacts = self.scene.last_vbd_contacts
        else:
            self._viz_contacts = self.scene.cable.model.collide(
                self.scene.cable.state_0,
                collision_pipeline=self.scene.cable_collision_pipeline,
            )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.scene.cable.state_0)
        self.viewer.log_contacts(self._viz_contacts, self.scene.cable.state_0)
        if self._tcp_force_arrow:
            log_batched_tcp_force_arrows(
                self.viewer,
                self.scene,
                self.layout,
                scale_per_newton=self._tcp_force_scale,
                gain=self._tcp_force_gain,
                min_length=self._tcp_force_min_length,
                max_length=self._tcp_force_max_length,
            )
        if self._mark_endpoints:
            log_batched_endpoints(
                self.viewer,
                self.scene,
                self.layout,
                radius=self._endpoint_radius,
            )
        self.viewer.end_frame()
        if self._mujoco_viewer and self.scene.robot_model is not None:
            fr3_robot.sync_mujoco_visual_state(
                self.scene.mj_solver,
                self.scene.robot_model,
                self.scene.robot_state_0,
            )
            self.scene.mj_solver.render_mujoco_viewer()

    def cleanup(self) -> None:
        if self._mujoco_viewer and self.scene.mj_solver is not None:
            self.scene.mj_solver.close_mujoco_viewer()

    def test_final(self, tolerance: float = 0.05) -> None:
        layout = self.layout
        if layout is None:
            return
        body_q = self.scene.cable.state_0.body_q.numpy().reshape(-1, 7)
        for w, apple_idx in enumerate(layout.apple_body_indices):
            if apple_idx < 0:
                continue
            z = float(body_q[apple_idx, 2])
            assert z > -tolerance, f"world {w} apple fell: z={z}"


if __name__ == "__main__":
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--num-frames", "2000"])
            print(
                "No DISPLAY/WAYLAND_DISPLAY: using --viewer null --num-frames 2000 "
                "(override with --viewer gl)."
            )

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)
    example = ExampleBatchedCoupledFruiting(viewer, args)

    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    print("Starting batched coupled simulation…")
    try:
        while viewer.is_running():
            example.step()
            example.render()
            import time

            time.sleep(max(0.0, example.frame_dt))
    finally:
        example.cleanup()

    example.test_final()
