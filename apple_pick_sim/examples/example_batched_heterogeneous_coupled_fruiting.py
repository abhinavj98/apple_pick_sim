"""Heterogeneous batched coupled fruiting — per-env physics DR, vectorized GPU stepping.

Each env shares topology (``num_segments``) but has independently sampled stiffness,
damping, rod geometry, and apple size. Default path: settle all worlds in parallel,
then weld with per-env robot-facing grasp direction and per-env IK bootstrap.

Run from the repository root::

    uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_fruiting.py \\
      --num-envs 4 --viewer gl --seed 42 --mark-endpoints --tcp-force-arrow

Per-env scripted actions (RL scatter-path demo)::

    uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_fruiting.py \\
      --num-envs 4 --viewer gl --demo-per-env-actions --seed 42

Variable-impedance teleop (requires PyTorch; ``uv sync --extra vic``)::

    uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_fruiting.py \\
      --num-envs 4 --viewer gl --controller vic --seed 42

Default ranges: ``fruiting_system_ranges_real_world_proxy_variance.json`` (real-world
bench proxy with per-env DR). Robot base at origin; fruiting chain at (0, 0.5, 0.95) m.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
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
    build_heterogeneous_coupled_fruiting_fr3,
    build_heterogeneous_coupled_fruiting_placeholder,
    seed_fix_to_apple_from_settled,
    settle_vbd_substeps,
)
from apple_pick_sim.fruiting_system import (
    FruitingSystemParams,
    GripperProxyConfig,
    PLACEHOLDER_EE_MASS_KG,
    default_ranges_fixture_path,
    load_ranges,
    sample_heterogeneous_params_list,
)

# Real-world bench proxy EE: 50 mm radius, 140 mm length (docs/real-world-proxy.md).
from apple_pick_sim.batched_viz import log_batched_endpoints, log_batched_tcp_force_arrows
from apple_pick_sim.coupled_fruiting.batched_robot_status import print_batched_robot_status


# Batched heterogeneous teleop: smaller steps than 1.0 m/s so template IK keeps up at 30 Hz.
_FR3_TELEOP_LINEAR_SPEED = 0.2
_FR3_TELEOP_ANGULAR_SPEED = 1.0
_FR3_TELEOP_IK_ITERATIONS = 128


def _default_ranges_path() -> Path:
    return default_ranges_fixture_path()


def _fix_to_apple_from_args(args: argparse.Namespace | None) -> bool:
    return bool(getattr(args, "fix_to_apple", True)) if args else True


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
            mass=PLACEHOLDER_EE_MASS_KG,
            fix_to_apple=fix,
            robot_facing_weld=fix,
        )
    return GripperProxyConfig(fix_to_apple=fix, robot_facing_weld=fix)


def _reject_unsupported_flags(args: argparse.Namespace) -> None:
    if bool(getattr(args, "only_vbd", False)) or bool(getattr(args, "only_mjc", False)):
        raise SystemExit("--only-vbd and --only-mjc are not supported.")


def _resolve_robot_kind(args: argparse.Namespace) -> str:
    robot_kind = str(getattr(args, "robot", "fr3"))
    if robot_kind == "fr3" and not fr3_robot.fr3_assets_available():
        print(
            "Warning: FR3 assets not found under assets/fr3/; falling back to placeholder TCP.",
            file=sys.stderr,
        )
        return "placeholder"
    return robot_kind


def _print_per_env_params(params_list: list[FruitingSystemParams]) -> None:
    print("Per-env fruiting params (topology shared, continuous θ differs):")
    for w, p in enumerate(params_list):
        bend = p.primary.bend_stiffness if p.primary is not None else float("nan")
        radius = float(p.apple_radius) if p.apple_radius is not None else float("nan")
        density = float(p.apple_density) if p.apple_density is not None else float("nan")
        print(
            f"  env{w}: primary_bend={bend:.4g}  apple_r={radius:.4g} m  "
            f"apple_rho={density:.4g} kg/m³"
        )


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
    parser.add_argument("--hz", type=float, default=30.0, help="Target frame rate [Hz].")
    parser.add_argument("--seed", type=int, default=None, help="Topology + DR seed.")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of coupled worlds.")
    parser.add_argument(
        "--env-spacing",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[2.5, 2.5, 0.0],
        help="Viewer grid spacing [m] (sim worlds are co-located).",
    )
    parser.add_argument("--enable-self-collision", action="store_true")
    parser.add_argument("--robot", type=str, choices=("placeholder", "fr3"), default="fr3")
    parser.add_argument(
        "--controller",
        type=str,
        choices=("direct", "ee", "vic"),
        default="direct",
    )
    parser.add_argument("--fr3-keyboard", action="store_true")
    parser.add_argument(
        "--fix-to-apple",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Settle-then-weld (default: on). Use --no-fix-to-apple for velocity-delta harvest.",
    )
    parser.add_argument("--settle-substeps", type=int, default=1000)
    parser.add_argument(
        "--scripted-ee-vel",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "VZ"),
        default=[0.05, 0.0, 0.0],
    )
    parser.add_argument(
        "--demo-per-env-actions",
        action="store_true",
        help="Scale scripted TCP velocity per env to demonstrate per-arm IK scatter.",
    )
    parser.add_argument("--status-every", type=int, default=60)
    parser.add_argument("--print-robot-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--noisy-action",
        action="store_true",
        help="Independent Gaussian noise per env (FR3 only; uses per-env RNG).",
    )
    parser.add_argument("--noisy-action-std", type=float, default=0.02)
    parser.add_argument("--tcp-force-arrow", action="store_true")
    parser.add_argument("--tcp-force-scale", type=float, default=0.02)
    parser.add_argument("--tcp-force-arrow-gain", type=float, default=1.0)
    parser.add_argument("--tcp-force-min-length", type=float, default=0.08)
    parser.add_argument("--tcp-force-max-length", type=float, default=1.5)
    parser.add_argument("--mark-endpoints", action="store_true")
    parser.add_argument("--endpoint-radius", type=float, default=0.05)
    parser.add_argument("--mujoco-viewer", action="store_true")
    parser.add_argument("--vic-linear-k", type=float, default=8000.0, help="VIC linear K [N/m].")
    parser.add_argument("--vic-linear-d", type=float, default=80.0, help="VIC linear D [N·s/m].")
    parser.add_argument("--vic-angular-k", type=float, default=40.0, help="VIC angular K [N·m/rad].")
    parser.add_argument("--vic-angular-d", type=float, default=4.0, help="VIC angular D [N·m·s/rad].")
    return parser


class ExampleBatchedHeterogeneousCoupledFruiting:
    """N heterogeneous coupled stacks; per-env DR at build, vectorized runtime stepping."""

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
        self._seed = int(seed)

        self.num_envs = int(getattr(args, "num_envs", 4))
        env_spacing_raw = getattr(args, "env_spacing", [2.5, 2.5, 0.0]) if args else [2.5, 2.5, 0.0]
        self.env_spacing = tuple(float(v) for v in env_spacing_raw)

        sim_device = resolve_sim_device(getattr(args, "device", None) if args else None)
        robot_kind = _resolve_robot_kind(args or argparse.Namespace())
        controller_mode = str(getattr(args, "controller", "direct"))
        fix_to_apple = _fix_to_apple_from_args(args)
        enable_self = _enable_self_collisions_from_args(args)
        settle_substeps = int(getattr(args, "settle_substeps", 1000))

        self.per_env_params = sample_heterogeneous_params_list(
            self.ranges, topology_seed=self._seed, num_envs=self.num_envs
        )

        print(f"Heterogeneous batched fruiting ranges: {ranges_path}")
        print(f"Topology seed: {self._seed}")
        print(f"Warp device: {sim_device}")
        _print_per_env_params(self.per_env_params)
        print(
            f"Gripper proxy fix_to_apple={fix_to_apple} "
            f"({'stem-harvest / settle-then-weld' if fix_to_apple else 'velocity-delta'} coupling)."
        )

        build_fn = (
            build_heterogeneous_coupled_fruiting_fr3
            if robot_kind == "fr3"
            else build_heterogeneous_coupled_fruiting_placeholder
        )
        gripper = _gripper_proxy_from_args(args, robot_kind=robot_kind)
        build_kw = dict(
            device=sim_device,
            env_spacing=self.env_spacing,
            enable_self_collisions=enable_self,
            gripper_proxy=gripper,
        )
        if robot_kind == "fr3":
            fr3_robot.enable_ik_bootstrap_warnings_for_examples()

        if fix_to_apple:
            settled = build_fn(
                self.ranges,
                self.per_env_params,
                **{
                    **build_kw,
                    "vbd_only": True,
                    "gripper_proxy": dataclasses.replace(
                        gripper, fix_to_apple=False, robot_facing_weld=False
                    ),
                },
            )
            settle_vbd_substeps(settled, substeps=settle_substeps, dt=self.sim_dt)
            self.scene = build_fn(
                self.ranges,
                self.per_env_params,
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
                per_env_ik=True,
                per_world_proxy_offsets=self.scene.per_world_proxy_offsets,
            )
        else:
            self.scene: CoupledFruitingScene = build_fn(
                self.ranges, self.per_env_params, **build_kw
            )

        self.layout = self.scene.layout
        if self.layout is None:
            raise RuntimeError("batched scene missing layout")

        print(
            f"Heterogeneous batched fruiting: num_envs={self.layout.num_envs} "
            f"spacing={self.env_spacing} "
            f"cable_world_count={self.scene.cable.model.world_count} "
            f"robot_world_count={self.scene.robot_model.world_count}"
        )

        self._controller_mode = controller_mode
        self._robot_kind = robot_kind
        self._ee_ctrl = None
        self._scripted_velocity = fr3_robot.EEVelocity(
            linear=tuple(float(v) for v in getattr(args, "scripted_ee_vel", [0.05, 0.0, 0.0])),
        )
        self._teleop_velocity = self._scripted_velocity
        self._use_keyboard = bool(getattr(args, "fr3_keyboard", False))
        self._noisy_action_std = float(getattr(args, "noisy_action_std", 0.02))
        self._per_env_noise_rng = [
            np.random.default_rng(self._seed + 1000 + w) for w in range(self.num_envs)
        ]
        noisy_requested = bool(getattr(args, "noisy_action", False))
        if noisy_requested and robot_kind != "fr3":
            print("Warning: --noisy-action requires FR3; ignoring.", file=sys.stderr)
        self._noisy_action = noisy_requested and robot_kind == "fr3"
        self._demo_per_env_actions = bool(getattr(args, "demo_per_env_actions", False))

        if robot_kind == "fr3" and self.scene.robot_model is not None:
            self._ee_ctrl = self._configure_fr3_controller(controller_mode)
        elif controller_mode == "ee":
            print("Note: --controller ee requires FR3; running without teleop.", file=sys.stderr)

        if self._use_keyboard and robot_kind == "fr3" and hasattr(self.viewer, "is_key_down"):
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
        self._tcp_force_scale = float(getattr(args, "tcp_force_scale", 0.02))
        self._tcp_force_gain = float(getattr(args, "tcp_force_arrow_gain", 1.0))
        self._tcp_force_min_length = float(getattr(args, "tcp_force_min_length", 0.08))
        self._tcp_force_max_length = float(getattr(args, "tcp_force_max_length", 1.5))
        self._mark_endpoints = bool(getattr(args, "mark_endpoints", False))
        self._endpoint_radius = float(getattr(args, "endpoint_radius", 0.05))

        self.viewer.set_model(self.scene.cable.model)
        graphical = isinstance(viewer, newton.viewer.ViewerGL)
        self._mujoco_viewer = (
            self.scene.robot_model is not None
            and bool(getattr(args, "mujoco_viewer", False))
            and graphical
        )
        if graphical and self.num_envs > 1:
            self.viewer.set_world_offsets(self.env_spacing)

        self._viz_contacts = self.scene.cable.model.collide(
            self.scene.cable.state_0,
            collision_pipeline=self.scene.cable_collision_pipeline,
        )

    def _configure_fr3_controller(self, mode: str):
        ik_kw = fr3_robot.batched_ik_teleop_kwargs(self.scene)
        if not ik_kw:
            raise RuntimeError("batched FR3 scene missing template IK layout")
        use_per_env = self._demo_per_env_actions or self._noisy_action
        velocity_for_world = self._velocity_for_world if use_per_env else (lambda w: self._teleop_velocity)  # noqa: ARG005
        if self._demo_per_env_actions:
            print("Heterogeneous demo: per-env scripted velocity scales (IK scatter).")
        if self._noisy_action:
            print(
                f"Heterogeneous noisy teleop: per-env Gaussian std={self._noisy_action_std}."
            )
        if mode == "ee":
            self.scene.robot_kinematic_mode = False
            ctrl = fr3_robot.Fr3BatchedEEVelocityController(
                self.scene.robot_model,
                linear_speed=_FR3_TELEOP_LINEAR_SPEED,
                angular_speed=_FR3_TELEOP_ANGULAR_SPEED,
                ik_iterations=_FR3_TELEOP_IK_ITERATIONS,
                velocity_for_world=velocity_for_world,
                **ik_kw,
            )
        elif mode == "vic":
            return self._configure_fr3_vic(ik_kw, velocity_for_world)
        else:
            self.scene.robot_kinematic_mode = True
            ctrl = fr3_robot.Fr3BatchedEEDirectJointController(
                self.scene.robot_model,
                linear_speed=_FR3_TELEOP_LINEAR_SPEED,
                angular_speed=_FR3_TELEOP_ANGULAR_SPEED,
                ik_iterations=_FR3_TELEOP_IK_ITERATIONS,
                velocity_for_world=velocity_for_world,
                **ik_kw,
            )
        ctrl.sync_target_from_state(self.scene.robot_state_0)
        return ctrl

    def _configure_fr3_vic(self, ik_kw: dict, velocity_for_world):
        from apple_pick_sim.coupled_fruiting.vic_joint_torques import _require_torch

        _require_torch()
        self.scene.robot_kinematic_mode = False
        fr3_robot.init_mujoco_actuator_targets_from_model(
            self.scene.robot_model, self.scene.robot_control
        )
        print("Batched FR3 dynamic-arm mode: MuJoCo integrates lagged plant wrenches on TCP body_f.")
        self.scene.vic_use_joint_torques = True
        vic = fr3_robot.Fr3BatchedEEImpedanceController(
            self.scene.robot_model,
            linear_speed=_FR3_TELEOP_LINEAR_SPEED,
            angular_speed=_FR3_TELEOP_ANGULAR_SPEED,
            velocity_for_world=velocity_for_world,
            **ik_kw,
        )
        self.scene.vic_controller = vic
        self.scene.vic_gains = fr3_robot.ImpedanceGains(
            linear_k=float(getattr(self.args, "vic_linear_k", 8000.0)),
            linear_d=float(getattr(self.args, "vic_linear_d", 80.0)),
            angular_k=float(getattr(self.args, "vic_angular_k", 40.0)),
            angular_d=float(getattr(self.args, "vic_angular_d", 4.0)),
        )
        fr3_robot.configure_vic_joint_torques_arm_batched(
            self.scene.robot_model,
            self.scene.robot_state_0,
            self.scene.robot_control,
            self.scene.mj_solver,
            scene=self.scene,
            layout=self.scene.layout,
        )
        self.scene.vic_joint_torques_configured = True
        vic.sync_target_from_state(self.scene.robot_state_0)
        vic.stage_targets_to_scene(self.scene)
        self.scene.vic_target_twist = fr3_robot.EEVelocity()
        g = self.scene.vic_gains
        print(
            f"Batched VIC enabled (joint torques, joint PD off): "
            f"K=({g.linear_k:g}, {g.angular_k:g}) "
            f"D=({g.linear_d:g}, {g.angular_d:g}) N/m, N·m/rad."
        )
        return vic

    def _velocity_for_world(self, world: int) -> fr3_robot.EEVelocity:
        if self._demo_per_env_actions:
            base = self._scripted_velocity if not self._use_keyboard else self._teleop_velocity
            angle = 2.0 * math.pi * world / max(self.num_envs, 1)
            speed = float(base.linear[0]) * (1.0 + 0.2 * world)
            return fr3_robot.EEVelocity(
                linear=(speed * math.cos(angle), speed * math.sin(angle), float(base.linear[2])),
                angular=base.angular,
            )
        if self._noisy_action:
            return fr3_robot.add_gaussian_noise_to_ee_velocity(
                self._teleop_velocity,
                rng=self._per_env_noise_rng[world],
                std=self._noisy_action_std,
            )
        return self._teleop_velocity

    def _teleop_world0(self) -> None:
        if self._robot_kind == "fr3" and self._ee_ctrl is not None:
            velocity = self._scripted_velocity
            if self._use_keyboard and hasattr(self.viewer, "is_key_down"):
                velocity = fr3_robot.read_keyboard_ee_velocity(
                    self.viewer,
                    linear_speed=self._ee_ctrl.linear_speed,
                    angular_speed=self._ee_ctrl.angular_speed,
                )
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
        if layout is None:
            return
        body_q = self.scene.cable.state_0.body_q.numpy().reshape(-1, 7)
        apple0 = layout.apple_body_indices[0]
        if apple0 >= 0:
            z0 = float(body_q[apple0, 2])
            print(f"t={self.sim_time:6.2f}s world-0 apple z={z0:.4f}", flush=True)
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
        if self._mujoco_viewer and self.scene.mj_solver is not None:
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
            sys.argv.extend(["--viewer", "null", "--num-frames", "200"])
            print(
                "No DISPLAY/WAYLAND_DISPLAY: using --viewer null --num-frames 200."
            )

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)
    example = ExampleBatchedHeterogeneousCoupledFruiting(viewer, args)

    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    print("Starting heterogeneous batched coupled simulation…")
    try:
        while viewer.is_running():
            example.step()
            example.render()
            import time

            time.sleep(max(0.0, example.frame_dt))
    finally:
        example.cleanup()

    example.test_final()
