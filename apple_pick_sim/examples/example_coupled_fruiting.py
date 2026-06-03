"""Interactive two-model coupling: **cable** (``SolverVBD``) + placeholder **TCP** (``SolverMuJoCo``).

The Newton **ViewerGL** shows the **cable** ``Model`` (tree + apple + gripper proxy that tracks the coupling).
Optional **MuJoCo** passive viewer (second window): pass ``--mujoco-viewer`` with a graphics session.

By default the example runs the full **MuJoCo + VBD** staggered loop. Use ``--only-vbd`` or ``--only-mjc`` to step
one solver in isolation.

Pick/drag applies forces to the **cable** state (right-click drag), routed after ``clear_forces`` when VBD runs.

Run from the repository root (see README for ``PYTHONPATH``)::

    PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_coupled_fruiting.py

Options (Newton example parser + extras)::

    PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_coupled_fruiting.py \\
      --json apple_pick_sim/fixtures/fruiting_system_ranges_example_variance.json --seed 42 --only-vbd

``--no-self-collision`` matches :func:`~apple_pick_sim.fruiting_system.generate_coupled_cable_scene` /
``enable_self_collisions=False``.

``--fix-to-apple`` / ``--no-fix-to-apple`` select stem-harvest + apple co-teleport vs the default
velocity-delta harvest (proxy-only sync).

Pass ``--fr3-direct-joints`` with ``--robot fr3`` (and ``--fr3-keyboard`` for teleop) to write IK
``joint_q`` directly and skip MuJoCo arm dynamics (testing / kinematic coupled runs).

Pass ``--fr3-keyboard`` with ``--robot fr3`` and ``--viewer gl`` for TCP keyboard teleop: each frame
IK writes ``robot_control.joint_target_pos`` / ``vel`` before MuJoCo substeps (same keys as
``example_fr3_keyboard.py``). **Verified interactively** with ``--only-mjc`` (MuJoCo + proxy sync only;
default ``fix_to_apple=False``). Full coupled teleop (no ``--only-mjc``) is not yet confirmed in the viewer.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

import dataclasses
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
        / "fruiting_system_ranges_example_variance.json"
    )


def _fix_to_apple_from_args(args: argparse.Namespace | None) -> bool:
    """Return whether the gripper proxy is welded to the apple (stem-harvest path)."""
    return bool(getattr(args, "fix_to_apple", False)) if args else False


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
        )
    return GripperProxyConfig(fix_to_apple=fix)


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
        "--no-self-collision",
        action="store_true",
        help="Disable intra-chain self collisions on the coupled cable scene (same as P0 viewer flag).",
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
        default="placeholder",
        help=(
            "Robot model for Model A: placeholder free TCP (default) or bundled FR3+EE "
            "(assets/fr3; requires usd-core)."
        ),
    )
    parser.add_argument(
        "--fr3-keyboard",
        action="store_true",
        help=(
            "Enable FR3 TCP keyboard teleop when ``--robot fr3`` (requires ``--viewer gl``; "
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
    parser.add_argument(
        "--fr3-direct-joints",
        action="store_true",
        help=(
            "FR3 testing mode: IK writes joint_q directly (kinematic arm). Requires "
            "--robot fr3. Use with --fr3-keyboard for teleop; skips MuJoCo arm integration "
            "and lagged TCP wrenches while cable VBD + proxy sync still run."
        ),
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
        robot_kind = getattr(args, "robot", "placeholder") if args else "placeholder"
        if robot_kind == "fr3":
            fr3_robot.enable_ik_bootstrap_warnings_for_examples()
        if robot_kind == "fr3" and not fr3_robot.fr3_assets_available():
            raise SystemExit(
                "FR3 assets not found under assets/fr3/; see assets/fr3/README.md"
            )
        if self._step_mode == "coupled":
            label = "FR3+EE" if robot_kind == "fr3" else "placeholder TCP"
            print(f"M1 cable + {label} MuJoCo (staggered coupling); Newton viewer shows cable model.")
        elif self._step_mode == "vbd":
            print("Cable SolverVBD only (--only-vbd).")
        else:
            print("MuJoCo robot + proxy sync only (--only-mjc); Newton viewer shows cable model.")

        enable_self = not (
            getattr(self.args, "no_self_collision", True) if self.args else True
        )

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
        if fix_to_apple and self._step_mode != "vbd":
            # Quiet start: settle with free apple dynamics, then rebuild welded and seed.
            settled = build_fn(
                self.ranges,
                first_seed,
                device=sim_device,
                enable_self_collisions=enable_self,
                gripper_proxy=dataclasses.replace(gripper, fix_to_apple=False),
                vbd_only=False,
                mujoco_only=False,
            )
            settle_vbd_substeps(settled, substeps=1800, dt=self.sim_dt)
            self.scene = build_fn(
                self.ranges,
                first_seed,
                device=sim_device,
                enable_self_collisions=enable_self,
                gripper_proxy=dataclasses.replace(gripper, fix_to_apple=True),
                vbd_only=(self._step_mode == "vbd"),
                mujoco_only=(self._step_mode == "mjc"),
            )
            seed_fix_to_apple_from_settled(
                welded_scene=self.scene,
                settled_scene=settled,
                quiet_apple_proxy=True,
            )
        else:
            self.scene: CoupledFruitingScene = build_fn(
                self.ranges,
                first_seed,
                device=sim_device,
                enable_self_collisions=enable_self,
                gripper_proxy=gripper,
                vbd_only=(self._step_mode == "vbd"),
                mujoco_only=(self._step_mode == "mjc"),
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

        direct_joints = bool(getattr(args, "fr3_direct_joints", False)) if args else False
        if direct_joints and robot_kind != "fr3":
            raise SystemExit("--fr3-direct-joints requires --robot fr3.")
        if direct_joints and self._step_mode == "vbd":
            raise SystemExit("--fr3-direct-joints requires a robot step mode (not --only-vbd).")
        if direct_joints and has_robot:
            self.scene.robot_kinematic_mode = True
            print(
                "FR3 direct-joint mode: kinematic arm (no MuJoCo integration, no lagged TCP wrench)."
            )

        self._ee_ctrl: (
            fr3_robot.Fr3EEVelocityController | fr3_robot.Fr3EEDirectJointController | None
        ) = None
        self._fr3_direct_joints = direct_joints
        enable_kb = bool(getattr(args, "fr3_keyboard", False)) if args else False
        kb_ok = hasattr(self.viewer, "is_key_down")
        want_fr3_ctrl = (enable_kb or direct_joints) and robot_kind == "fr3"
        if want_fr3_ctrl and self._step_mode != "vbd" and has_robot:
            ctrl_cls = (
                fr3_robot.Fr3EEDirectJointController
                if direct_joints
                else fr3_robot.Fr3EEVelocityController
            )
            if enable_kb and not kb_ok:
                print(
                    "FR3 keyboard teleop unavailable: use --viewer gl (not viser/null).",
                    file=sys.stderr,
                )
            else:
                self._ee_ctrl = ctrl_cls(
                    self.scene.robot_model,
                    self.scene.tcp_body_index,
                    linear_speed=1.0,
                    angular_speed=5.0,
                    ik_iterations=48,
                )
                self._ee_ctrl.sync_target_from_state(self.scene.robot_state_0)
                if enable_kb:
                    fr3_robot.print_fr3_keyboard_bindings()
                elif direct_joints:
                    print(
                        "FR3 direct-joint controller ready (no keyboard; pass velocity in tests).",
                        file=sys.stderr,
                    )
        elif enable_kb and robot_kind != "fr3":
            print("Warning: --fr3-keyboard requires --robot fr3.", file=sys.stderr)

    def _pick_forces(self) -> None:
        self.viewer.apply_forces(self.scene.cable.state_0)

    def simulate(self) -> None:
        # FR3 teleop: advance target + solve IK once per frame (not per substep).
        if self._ee_ctrl is not None:
            if self._fr3_direct_joints:
                self.scene.apply_fr3_ee_teleop_direct(
                    self.frame_dt,
                    self._ee_ctrl,
                    viewer=self.viewer,
                )
            else:
                self.scene.apply_fr3_ee_teleop(
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

    def step(self, warmup: bool = False) -> None:
        del warmup
        self.simulate()
        self.sim_time += self.frame_dt

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
        self.viewer.end_frame()
        if self._mujoco_viewer and self.scene.robot_model is not None:
            fr3_robot.sync_mujoco_visual_state(
                self.scene.mj_solver,
                self.scene.robot_model,
                self.scene.robot_state_0,
            )
            self.scene.mj_solver.render_mujoco_viewer()

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

    print("Warming up…")
    for _ in range(20):
        example.step(warmup=True)

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
