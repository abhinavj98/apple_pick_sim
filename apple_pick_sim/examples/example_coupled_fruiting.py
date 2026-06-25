"""Interactive two-model coupling: **cable** (``SolverVBD``) + FR3 TCP (``SolverMuJoCo``).

The Newton **ViewerGL** shows the **cable** ``Model`` (tree + apple + gripper proxy that tracks the coupling).
Optional **MuJoCo** passive viewer (second window): pass ``--mujoco-viewer`` with a graphics session.

By default the example runs the full **MuJoCo + VBD** staggered loop with **FR3 variable-impedance
teleop via joint torques**. Use ``--only-vbd`` or ``--only-mjc`` to step one solver in isolation.

Pick/drag applies forces to the **cable** state (right-click drag), routed after ``clear_forces`` when VBD runs.

Run from the repository root (see README for ``PYTHONPATH``). VIC joint torques require PyTorch
(``cd newton && uv sync --extra torch-cu12``)::

    PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_coupled_fruiting.py

Options (Newton example parser + extras)::

    PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_coupled_fruiting.py \\
      --json apple_pick_sim/fixtures/fruiting_system_ranges_example_variance.json --seed 42

Intra-chain self collisions are **off** by default. Pass ``--enable-self-collision`` to set
``enable_self_collisions=True`` on the coupled cable scene (same semantics as P0 when collisions are on).

``--fix-to-apple`` / ``--no-fix-to-apple`` select stem-harvest + apple co-teleport vs the default
velocity-delta harvest (proxy-only sync).

Pass ``--fr3-keyboard`` with ``--viewer gl`` for TCP keyboard teleop.

Select the FR3 teleop controller with ``--controller``:

- ``vic`` (default) — variable-impedance teleop via joint torques (requires PyTorch).
- ``ee`` — TCP velocity + IK with MuJoCo joint PD actuators.
- ``direct`` — kinematic direct ``joint_q`` writes (testing / accurate pose hold).

For ``vic``, tune impedance with ``--vic-linear-k``, ``--vic-linear-d``, ``--vic-angular-k``, and
``--vic-angular-d`` (defaults 800/80 N/m and 40/4 N·m/rad).
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

import dataclasses
import numpy as np
import newton
import newton.examples
import warp as wp

from apple_pick_sim.sim_device import resolve_sim_device
from apple_pick_sim.coupling_force_debug import CouplingForceDebugRecorder
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.coupled_fruiting import (
    CoupledFruitingScene,
    build_coupled_fruiting_fr3,
    build_coupled_fruiting_placeholder,
    seed_fix_to_apple_from_settled,
    settle_vbd_substeps,
)
from apple_pick_sim.fruiting_system import (
    GripperProxyConfig,
    geometry_fingerprint,
    load_ranges,
)


def _default_ranges_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "fixtures"
        / "fruiting_system_ranges_example_variance_soft.json"
    )


def _fix_to_apple_from_args(args: argparse.Namespace | None) -> bool:
    """Return whether the gripper proxy is welded to the apple (stem-harvest path)."""
    return bool(getattr(args, "fix_to_apple", False)) if args else False


def _enable_self_collisions_from_args(args: argparse.Namespace | None) -> bool:
    """Return whether intra-chain self collisions are enabled on the coupled cable scene."""
    return bool(getattr(args, "enable_self_collision", False)) if args else False


def _gripper_proxy_from_args(
    args: argparse.Namespace | None,
    *,
    robot_kind: str,
) -> GripperProxyConfig:
    """Build :class:`GripperProxyConfig` from CLI ``--fix-to-apple`` / ``--no-fix-to-apple``."""
    fix = _fix_to_apple_from_args(args)
    if robot_kind == "fr3":
        return GripperProxyConfig(
            mass=fr3_robot.EE_MASS_KG,
            box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
            fix_to_apple=fix,
            robot_facing_weld=fix,
        )
    return GripperProxyConfig(fix_to_apple=fix, robot_facing_weld=fix)


def _resolve_step_mode(args: argparse.Namespace | None) -> str:
    """Return ``"coupled"``, ``"vbd"``, or ``"mjc"`` from CLI flags."""
    only_vbd = bool(getattr(args, "only_vbd", False)) if args else False
    only_mjc = bool(getattr(args, "only_mjc", False)) if args else False
    if only_vbd and only_mjc:
        raise SystemExit("--only-vbd and --only-mjc are mutually exclusive")
    if only_vbd:
        return "vbd"
    if only_mjc:
        return "mjc"
    return "coupled"


def _resolve_controller_mode(args: argparse.Namespace | None) -> str:
    """Return ``"vic"``, ``"ee"``, or ``"direct"`` from CLI ``--controller``."""
    return str(getattr(args, "controller", "vic") if args else "vic")


def _format_wp_transform(tf: wp.transform) -> str:
    pos = wp.transform_get_translation(tf)
    rot = wp.transform_get_rotation(tf)
    return (
        f"pos=({float(pos[0]):.4f}, {float(pos[1]):.4f}, {float(pos[2]):.4f}) "
        f"quat=({float(rot[0]):.4f}, {float(rot[1]):.4f}, {float(rot[2]):.4f}, {float(rot[3]):.4f})"
    )


def _proxy_expected_tf(scene: CoupledFruitingScene) -> wp.transform | None:
    """Cable gripper-proxy body pose — the TCP alignment target at bootstrap."""
    proxy_body = scene.cable.gripper_proxy_body
    if proxy_body is None:
        return None
    bq = scene.cable.state_0.body_q.numpy().reshape(-1, 7)[proxy_body]
    return wp.transform(
        wp.vec3(float(bq[0]), float(bq[1]), float(bq[2])),
        wp.quat(float(bq[3]), float(bq[4]), float(bq[5]), float(bq[6])),
    )


def _print_target_and_expected_tf(
    scene: CoupledFruitingScene,
    controller: (
        fr3_robot.Fr3EEImpedanceController
        | fr3_robot.Fr3EEVelocityController
        | fr3_robot.Fr3EEDirectJointController
    ),
    *,
    sim_time: float | None = None,
) -> None:
    expected_tf = _proxy_expected_tf(scene)
    prefix = f"t={sim_time:6.2f}s " if sim_time is not None else ""
    print(f"{prefix}target_tf:   {_format_wp_transform(controller.target_tf)}", flush=True)
    if expected_tf is not None:
        print(f"{prefix}expected_tf: {_format_wp_transform(expected_tf)}", flush=True)
        target_pos = wp.transform_get_translation(controller.target_tf)
        exp_pos = wp.transform_get_translation(expected_tf)
        err_mm = float(
            np.linalg.norm(
                np.array([float(target_pos[i]) for i in range(3)])
                - np.array([float(exp_pos[i]) for i in range(3)])
            )
        ) * 1000.0
        print(f"{prefix}target−expected err={err_mm:5.1f} mm", flush=True)
    else:
        print(f"{prefix}expected_tf: (no gripper proxy on cable scene)", flush=True)
    if scene.robot_state_0 is not None and scene.tcp_body_index is not None:
        tcp_q7 = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[int(scene.tcp_body_index)]
        print(
            f"{prefix}actual_tcp:  pos=({tcp_q7[0]:.4f}, {tcp_q7[1]:.4f}, {tcp_q7[2]:.4f})",
            flush=True,
        )


def _resolve_robot_kind(args: argparse.Namespace | None) -> str:
    robot_kind = getattr(args, "robot", "fr3") if args else "fr3"
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
            "apple_pick_sim/fixtures/fruiting_system_ranges_example_variance.json)."
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
        help="RNG seed for the first scene. Omit for a random seed on each run.",
    )
    parser.add_argument(
        "--enable-self-collision",
        action="store_true",
        help="Enable intra-chain self collisions on the coupled cable scene (off by default).",
    )
    parser.add_argument(
        "--only-vbd",
        action="store_true",
        help="Cable SolverVBD only (gripper proxy at spawn; no MuJoCo robot).",
    )
    parser.add_argument(
        "--only-mjc",
        action="store_true",
        help="MuJoCo robot + proxy sync only (cable tree not integrated).",
    )
    parser.add_argument(
        "--mujoco-viewer",
        action="store_true",
        help=(
            "Open MuJoCo's passive viewer for the robot model after each Newton frame "
            "(separate window; requires a GUI session)."
        ),
    )
    parser.add_argument(
        "--debug-coupling-forces",
        action="store_true",
        help=(
            "Plot lagged MuJoCo-applied vs VBD-harvested TCP coupling wrenches in the "
            "Newton viewer (ViewerGL / ViewerViser Plots panel)."
        ),
    )
    parser.add_argument(
        "--tcp-force-arrow",
        action="store_true",
        help=(
            "Draw harvested TCP force as a yellow world-frame arrow at the robot TCP "
            "(requires --viewer gl or viser; use --tcp-force-scale to tune length)."
        ),
    )
    parser.add_argument(
        "--tcp-force-scale",
        type=float,
        default=0.02,
        help="Arrow length per newton [m/N] for --tcp-force-arrow (default 0.02 → 50 N ≈ 1 m).",
    )
    parser.add_argument(
        "--tcp-force-arrow-gain",
        type=float,
        default=1.0,
        help="Dimensionless multiplier on --tcp-force-scale (default 1).",
    )
    parser.add_argument(
        "--tcp-force-min-length",
        type=float,
        default=0.08,
        help="Minimum arrow length [m] when force is above threshold (default 0.08).",
    )
    parser.add_argument(
        "--tcp-force-max-length",
        type=float,
        default=1.5,
        help="Maximum arrow length [m]; 0 disables cap (default 1.5).",
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
        choices=("vic", "ee", "direct"),
        default="vic",
        help=(
            "FR3 teleop controller: vic (default, joint-torque VIC), "
            "ee (velocity IK + MuJoCo PD), or direct (kinematic joint_q writes)."
        ),
    )
    parser.add_argument(
        "--fr3-keyboard",
        action="store_true",
        help=(
            "Enable FR3 TCP keyboard teleop (requires ``--viewer gl``; "
            "focus the window — I/K J/L R/F translate, U/O T/G Z/X rotate)."
        ),
    )
    parser.add_argument(
        "--fix-to-apple",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Weld gripper proxy to apple (stem-harvest + co-teleport apple with TCP). "
            "Default: off (velocity-delta harvest, proxy-only sync)."
        ),
    )
    parser.add_argument("--vic-linear-k", type=float, default=8000.0, help="VIC linear K [N/m].")
    parser.add_argument("--vic-linear-d", type=float, default=80.0, help="VIC linear D [N·s/m].")
    parser.add_argument("--vic-angular-k", type=float, default=40.0, help="VIC angular K [N·m/rad].")
    parser.add_argument("--vic-angular-d", type=float, default=4.0, help="VIC angular D [N·m·s/rad].")
    parser.add_argument(
        "--print-tcp-transforms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print target_tf / expected_tf / actual_tcp each frame (default on).",
    )
    parser.add_argument(
        "--world-axes",
        action="store_true",
        help=(
            "Draw world-frame X/Y/Z axes as RGB arrows in the Newton viewer "
            "(requires --viewer gl or viser)."
        ),
    )
    parser.add_argument(
        "--world-axes-length",
        type=float,
        default=0.3,
        help="Length [m] of each world-axis arrow (default 0.3).",
    )
    parser.add_argument(
        "--world-axes-origin",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[0.0, 0.0, 0.0],
        help="World-frame origin of the axes arrows [m] (default 0 0 0).",
    )
    return parser


class ExampleCoupledFruiting:
    """Newton GL viewer drives the coupled cable model; TCP robot steps in MuJoCo each substep."""

    def __init__(self, viewer, args: argparse.Namespace | None = None):
        self.viewer = viewer
        self.args = args

        self.fps = float(getattr(args, "hz", 30.0)) if args else 30.0
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 30
        self.sim_dt = (1.0 / 60.0) / self.sim_substeps
        self.sim_time = 0.0

        json_path = getattr(args, "json", None) if args else None
        self.ranges_path = Path(json_path) if json_path else _default_ranges_path()
        self.ranges = load_ranges(self.ranges_path)

        first_seed = getattr(args, "seed", None) if args else None
        if first_seed is None:
            first_seed = secrets.randbelow(2**31 - 1)

        graphical = isinstance(viewer, newton.viewer.ViewerGL)
        self._step_mode = _resolve_step_mode(args)
        print(f"Coupled fruiting ranges: {self.ranges_path}")
        print(f"Initial seed: {first_seed}")
        sim_device = resolve_sim_device(getattr(args, "device", None) if args else None)
        print(f"Warp device: {sim_device}")
        robot_kind = _resolve_robot_kind(args)
        if robot_kind == "fr3":
            fr3_robot.enable_ik_bootstrap_warnings_for_examples()
        if self._step_mode == "coupled":
            label = "FR3+EE" if robot_kind == "fr3" else "placeholder TCP"
            print(f"M1 cable + {label} MuJoCo (staggered coupling); Newton viewer shows cable model.")
        elif self._step_mode == "vbd":
            print("Cable SolverVBD only (--only-vbd).")
        else:
            print("MuJoCo robot + proxy sync only (--only-mjc); Newton viewer shows cable model.")

        enable_self = _enable_self_collisions_from_args(self.args)

        fix_to_apple = _fix_to_apple_from_args(args)
        print(
            f"Gripper proxy fix_to_apple={fix_to_apple} "
            f"({'stem-harvest' if fix_to_apple else 'velocity-delta'} coupling)."
        )

        build_fn = (
            build_coupled_fruiting_fr3
            if robot_kind == "fr3"
            else build_coupled_fruiting_placeholder
        )
        gripper = _gripper_proxy_from_args(args, robot_kind=robot_kind)
        build_kw = dict(
            device=sim_device,
            enable_self_collisions=enable_self,
            gripper_proxy=gripper,
            vbd_only=(self._step_mode == "vbd"),
            mujoco_only=(self._step_mode == "mjc"),
        )
        if robot_kind == "fr3" and not (fix_to_apple and self._step_mode != "vbd"):
            build_kw["robot_base_from_proxy"] = True
        if fix_to_apple and self._step_mode != "vbd":
            from apple_pick_sim.robot.fr3_robot.placement import IKBootstrapConvergenceError

            build_kw["ik_bootstrap_iterations"] = 256
            last_ik_exc: IKBootstrapConvergenceError | None = None
            scene_seed = first_seed
            for try_seed in (first_seed,):
                try:
                    settled = build_fn(
                        self.ranges,
                        try_seed,
                        **{
                            **build_kw,
                            "vbd_only": True,
                            "gripper_proxy": dataclasses.replace(
                                gripper, fix_to_apple=False, robot_facing_weld=False
                            ),
                        },
                    )
                    settle_vbd_substeps(settled, substeps=1000, dt=self.sim_dt)
                    self.scene = build_fn(
                        self.ranges,
                        try_seed,
                        **{
                            **build_kw,
                            "skip_ik_bootstrap": True,
                            "gripper_proxy": dataclasses.replace(
                                gripper, fix_to_apple=True, robot_facing_weld=True
                            ),
                        },
                    )
                    seed_fix_to_apple_from_settled(
                        welded_scene=self.scene,
                        settled_scene=settled,
                        quiet_apple_proxy=True,
                    )
                    scene_seed = try_seed
                    break
                except IKBootstrapConvergenceError as exc:
                    last_ik_exc = exc
            else:
                raise last_ik_exc  # type: ignore[misc]
            if scene_seed != first_seed:
                print(f"Settle-then-weld IK bootstrap used seed {scene_seed} (requested {first_seed})")
            first_seed = scene_seed
        else:
            self.scene: CoupledFruitingScene = build_fn(
                self.ranges,
                first_seed,
                **build_kw,
            )

        self._force_debug: CouplingForceDebugRecorder | None = None
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
        self._world_axes = bool(getattr(args, "world_axes", False))
        self._world_axes_length = float(getattr(args, "world_axes_length", 0.3))
        _origin_raw = getattr(args, "world_axes_origin", [0.0, 0.0, 0.0]) if args else [0.0, 0.0, 0.0]
        self._world_axes_origin = tuple(float(v) for v in _origin_raw)
        if self._world_axes and graphical:
            print(
                f"World axes: RGB = XYZ from origin {self._world_axes_origin}, "
                f"length={self._world_axes_length:.3f} m."
            )
        if (
            self._step_mode == "coupled"
            and bool(getattr(args, "debug_coupling_forces", False))
        ):
            self._force_debug = CouplingForceDebugRecorder()
            self.scene.force_debug = self._force_debug
            if graphical:
                print(
                    "Coupling force debug: open the Plots panel (ViewerGL) for "
                    "MuJoCo-applied vs VBD-harvested wrenches."
                )
        if self._tcp_force_arrow and self._step_mode != "coupled":
            print("Note: --tcp-force-arrow needs full coupled stepping (omit --only-vbd / --only-mjc).")
        elif self._tcp_force_arrow and graphical:
            cap = (
                f"{self._tcp_force_max_length:.2f} m max"
                if self._tcp_force_max_length > 0.0
                else "no max"
            )
            print(
                "TCP force arrow: yellow at robot TCP; "
                f"scale={self._tcp_force_scale:.4f} m/N × gain {self._tcp_force_gain:g}, "
                f"min {self._tcp_force_min_length:.2f} m, {cap}."
            )

        fp = geometry_fingerprint(self.scene.cable)
        apos = fp.get("apple_pos")
        ap_txt = f"apple at {apos}" if apos is not None else "apple=OFF"
        print(f"Geometry fingerprint ({ap_txt})")

        self.viewer.set_model(self.scene.cable.model)
        self._viz_contacts = self.scene.cable.model.collide(
            self.scene.cable.state_0,
            collision_pipeline=self.scene.cable_collision_pipeline,
        )

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

        self._controller_mode = _resolve_controller_mode(args)
        self._print_tcp_transforms = bool(getattr(args, "print_tcp_transforms", True))
        self._ee_ctrl: (
            fr3_robot.Fr3EEImpedanceController
            | fr3_robot.Fr3EEVelocityController
            | fr3_robot.Fr3EEDirectJointController
            | None
        ) = None
        if robot_kind == "fr3" and has_robot and self._step_mode != "vbd":
            self._ee_ctrl = self._configure_fr3_controller(args, self._controller_mode)
            if self._print_tcp_transforms:
                print("FR3 TCP transforms after controller sync:", flush=True)
                _print_target_and_expected_tf(self.scene, self._ee_ctrl)

        enable_kb = bool(getattr(args, "fr3_keyboard", False)) if args else False
        kb_ok = hasattr(self.viewer, "is_key_down")
        if enable_kb and robot_kind == "fr3" and self._ee_ctrl is not None:
            if not kb_ok:
                print(
                    "FR3 keyboard teleop unavailable: use --viewer gl (not viser/null).",
                    file=sys.stderr,
                )
            else:
                fr3_robot.print_fr3_keyboard_bindings()
        elif enable_kb and robot_kind != "fr3":
            print("Warning: --fr3-keyboard requires FR3 assets.", file=sys.stderr)

    def _configure_fr3_controller(
        self,
        args: argparse.Namespace | None,
        mode: str,
    ) -> (
        fr3_robot.Fr3EEImpedanceController
        | fr3_robot.Fr3EEVelocityController
        | fr3_robot.Fr3EEDirectJointController
    ):
        if mode == "vic":
            return self._configure_fr3_vic(args)
        if mode == "ee":
            return self._configure_fr3_ee(args)
        if mode == "direct":
            return self._configure_fr3_direct(args)
        raise ValueError(f"unknown controller mode: {mode!r}")

    def _configure_fr3_ee(
        self, args: argparse.Namespace | None
    ) -> fr3_robot.Fr3EEVelocityController:
        self.scene.robot_kinematic_mode = False
        ee = fr3_robot.Fr3EEVelocityController(
            self.scene.robot_model,
            int(self.scene.tcp_body_index),
            linear_speed=1.0,
            angular_speed=5.0,
        )
        ee.sync_target_from_state(self.scene.robot_state_0)
        print("FR3 ee controller: TCP velocity + IK with MuJoCo joint PD actuators.")
        return ee

    def _configure_fr3_direct(
        self, args: argparse.Namespace | None
    ) -> fr3_robot.Fr3EEDirectJointController:
        del args
        self.scene.robot_kinematic_mode = True
        direct = fr3_robot.Fr3EEDirectJointController(
            self.scene.robot_model,
            int(self.scene.tcp_body_index),
            linear_speed=1.0,
            angular_speed=5.0,
        )
        direct.sync_target_from_state(self.scene.robot_state_0)
        print("FR3 direct controller: kinematic joint_q writes (robot_kinematic_mode=True).")
        return direct

    def _configure_fr3_vic(self, args: argparse.Namespace | None) -> fr3_robot.Fr3EEImpedanceController:
        from apple_pick_sim.coupled_fruiting import vic_joint_torques

        vic_joint_torques._require_torch()
        self.scene.robot_kinematic_mode = False
        fr3_robot.init_mujoco_actuator_targets_from_model(
            self.scene.robot_model, self.scene.robot_control
        )
        print("FR3 dynamic-arm mode: MuJoCo integrates lagged plant wrenches on TCP body_f.")
        self.scene.vic_use_joint_torques = True
        vic = fr3_robot.Fr3EEImpedanceController(
            tcp_body_index=int(self.scene.tcp_body_index),
            linear_speed=1.0,
            angular_speed=5.0,
        )
        self.scene.vic_controller = vic
        self.scene.vic_gains = fr3_robot.ImpedanceGains(
            linear_k=float(getattr(args, "vic_linear_k", 800.0)),
            linear_d=float(getattr(args, "vic_linear_d", 80.0)),
            angular_k=float(getattr(args, "vic_angular_k", 40.0)),
            angular_d=float(getattr(args, "vic_angular_d", 4.0)),
        )
        fr3_robot.configure_vic_joint_torques_arm(
            self.scene.robot_model,
            self.scene.robot_state_0,
            self.scene.robot_control,
            self.scene.mj_solver,
            scene=self.scene,
        )
        self.scene.vic_joint_torques_configured = True
        vic.sync_target_from_state(self.scene.robot_state_0)
        self.scene.vic_target_tf = vic.target_tf
        self.scene.vic_target_twist = fr3_robot.EEVelocity()
        g = self.scene.vic_gains
        print(
            f"VIC enabled (joint torques, joint PD off): "
            f"K=({g.linear_k:g}, {g.angular_k:g}) "
            f"D=({g.linear_d:g}, {g.angular_d:g}) N/m, N·m/rad."
        )
        return vic

    def _pick_forces(self) -> None:
        self.viewer.apply_forces(self.scene.cable.state_0)

    def simulate(self) -> None:
        if self._ee_ctrl is not None:
            if self._controller_mode == "direct":
                self.scene.update_fr3_ee_teleop_direct(
                    self.frame_dt,
                    self._ee_ctrl,
                    viewer=self.viewer,
                )
            else:
                self.scene.update_fr3_ee_teleop(
                    self.frame_dt,
                    self._ee_ctrl,
                    viewer=self.viewer,
                )
        for _ in range(self.sim_substeps):
            if self._step_mode == "vbd":
                self.scene.vbd_substep(
                    self.sim_dt,
                    after_cable_clear_forces=self._pick_forces,
                )
            elif self._step_mode == "mjc":
                self.scene.mujoco_substep(self.sim_dt)
            else:
                self.scene.coupled_substep(
                    self.sim_dt,
                    after_cable_clear_forces=self._pick_forces,
                )
                if self._force_debug is not None:
                    self._force_debug.log_to_viewer(self.viewer)

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt
        if self._print_tcp_transforms and self._ee_ctrl is not None:
            _print_target_and_expected_tf(
                self.scene,
                self._ee_ctrl,
                sim_time=self.sim_time,
            )

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
        if self._tcp_force_arrow and self._step_mode == "coupled":
            from apple_pick_sim.tcp_force_viz import log_coupled_scene_tcp_force

            log_coupled_scene_tcp_force(
                self.viewer,
                self.scene,
                scale_per_newton=self._tcp_force_scale,
                gain=self._tcp_force_gain,
                min_length=self._tcp_force_min_length,
                max_length=self._tcp_force_max_length,
            )
        if self._world_axes:
            self._render_world_axes()
        self.viewer.end_frame()
        if self._mujoco_viewer and self.scene.robot_model is not None:
            fr3_robot.sync_mujoco_visual_state(
                self.scene.mj_solver,
                self.scene.robot_model,
                self.scene.robot_state_0,
            )
            self.scene.mj_solver.render_mujoco_viewer()

    def _render_world_axes(self) -> None:
        """Draw world-frame X (red), Y (green), Z (blue) axis arrows in the Newton viewer."""
        log_lines = getattr(self.viewer, "log_lines", None)
        log_arrows = getattr(self.viewer, "log_arrows", None)
        _log = log_arrows if log_arrows is not None else log_lines
        if _log is None:
            return
        dev = str(getattr(self.viewer, "device", "cpu"))
        o = np.array(self._world_axes_origin, dtype=np.float32)
        L = self._world_axes_length
        # X → red, Y → green, Z → blue
        axes = [
            ("x", np.array([L, 0.0, 0.0], dtype=np.float32), np.array([[1.0, 0.0, 0.0]], dtype=np.float32)),
            ("y", np.array([0.0, L, 0.0], dtype=np.float32), np.array([[0.0, 1.0, 0.0]], dtype=np.float32)),
            ("z", np.array([0.0, 0.0, L], dtype=np.float32), np.array([[0.0, 0.0, 1.0]], dtype=np.float32)),
        ]
        for label, tip_offset, color in axes:
            starts = wp.array(o.reshape(1, 3), dtype=wp.vec3, device=dev)
            ends = wp.array((o + tip_offset).reshape(1, 3), dtype=wp.vec3, device=dev)
            colors = wp.array(color, dtype=wp.vec3, device=dev)
            _log(f"/debug/world_axis_{label}", starts, ends, colors)

    def cleanup(self) -> None:
        if self._mujoco_viewer:
            self.scene.mj_solver.close_mujoco_viewer()

    def test_final(self, tolerance: float = 0.05) -> None:
        apple = getattr(self.scene.cable, "apple_body", None)
        if apple is None:
            return
        body_q = self.scene.cable.state_0.body_q.numpy()
        z = float(body_q.reshape(-1, 7)[apple, 2])
        assert z > -tolerance, f"Apple fell through ground: z={z}"


if __name__ == "__main__":
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--num-frames", "2000"])
            print(
                "No DISPLAY/WAYLAND_DISPLAY: using --viewer null --num-frames 2000 (override with --viewer gl)."
            )

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)
    example = ExampleCoupledFruiting(viewer, args)

    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    print("Starting simulation…")
    try:
        while viewer.is_running():
            example.step()
            example.render()
            import time
            time.sleep(max(0.0, example.frame_dt))
    finally:
        example.cleanup()

    example.test_final()
