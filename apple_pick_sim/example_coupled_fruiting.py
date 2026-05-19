"""Interactive two-model coupling: **cable** (``SolverVBD``) + placeholder **TCP** (``SolverMuJoCo``).

The Newton **ViewerGL** shows the **cable** ``Model`` (tree + apple + gripper proxy that tracks the coupling).
Optional **MuJoCo** passive viewer (second window): pass ``--mujoco-viewer`` with a graphics session.

By default the example runs the full **MuJoCo + VBD** staggered loop. Use ``--only-vbd`` or ``--only-mjc`` to step
one solver in isolation.

Pick/drag applies forces to the **cable** state (right-click drag), routed after ``clear_forces`` when VBD runs.

Run from the repository root (see README for ``PYTHONPATH``)::

    PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/example_coupled_fruiting.py

Options (Newton example parser + extras)::

    PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/example_coupled_fruiting.py \\
      --json apple_pick_sim/fixtures/fruiting_system_ranges_example_variance.json --seed 42 --only-vbd

``--no-self-collision`` matches :func:`~apple_pick_sim.fruiting_system.generate_coupled_cable_scene` /
``enable_self_collisions=False``.

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

import newton
import newton.examples
import warp as wp

from apple_pick_sim.coupling_force_debug import CouplingForceDebugRecorder
from apple_pick_sim import fr3_robot
from apple_pick_sim.coupled_fruiting import (
    CoupledFruitingScene,
    build_coupled_fruiting_fr3,
    build_coupled_fruiting_placeholder,
)
from apple_pick_sim.fruiting_system import geometry_fingerprint, load_ranges


def _default_ranges_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "fruiting_system_ranges_example_variance.json"


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
    return parser


class ExampleCoupledFruiting:
    """Newton GL viewer drives the coupled cable model; TCP robot steps in MuJoCo each substep."""

    def __init__(self, viewer, args: argparse.Namespace | None = None):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 30
        self.sim_dt = self.frame_dt / self.sim_substeps
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
        robot_kind = getattr(args, "robot", "placeholder") if args else "placeholder"
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
            getattr(self.args, "no_self_collision", False) if self.args else False
        )

        build_fn = (
            build_coupled_fruiting_fr3
            if robot_kind == "fr3"
            else build_coupled_fruiting_placeholder
        )
        self.scene: CoupledFruitingScene = build_fn(
            self.ranges,
            first_seed,
            device=str(wp.get_device()),
            enable_self_collisions=enable_self,
            vbd_only=(self._step_mode == "vbd"),
            mujoco_only=(self._step_mode == "mjc"),
        )

        self._force_debug: CouplingForceDebugRecorder | None = None
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

        self._ee_ctrl: fr3_robot.Fr3EEVelocityController | None = None
        enable_kb = bool(getattr(args, "fr3_keyboard", False)) if args else False
        kb_ok = hasattr(self.viewer, "is_key_down")
        if (
            enable_kb
            and robot_kind == "fr3"
            and self._step_mode != "vbd"
            and has_robot
            and kb_ok
        ):
            self._ee_ctrl = fr3_robot.Fr3EEVelocityController(
                self.scene.robot_model,
                self.scene.tcp_body_index,
                linear_speed=0.15,
                angular_speed=0.8,
                ik_iterations=24,
            )
            self._ee_ctrl.sync_target_from_state(self.scene.robot_state_0)
            print(
                "FR3 keyboard teleop (focus this window): I/K J/L R/F translate, "
                "U/O T/G Z/X rotate (not W/S — camera)."
            )
        elif enable_kb and robot_kind != "fr3":
            print("Warning: --fr3-keyboard requires --robot fr3.", file=sys.stderr)
        elif enable_kb and has_robot and self._step_mode != "vbd" and not kb_ok:
            print(
                "FR3 keyboard teleop unavailable: use --viewer gl (not viser/null).",
                file=sys.stderr,
            )

    def _pick_forces(self) -> None:
        self.viewer.apply_forces(self.scene.cable.state_0)

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            if self._ee_ctrl is not None:
                self.scene.apply_fr3_ee_teleop(
                    self.frame_dt,
                    self._ee_ctrl,
                    viewer=self.viewer,
                )
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
        self._viz_contacts = self.scene.cable.model.collide(
            self.scene.cable.state_0,
            collision_pipeline=self.scene.cable_collision_pipeline,
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.scene.cable.state_0)
        self.viewer.log_contacts(self._viz_contacts, self.scene.cable.state_0)
        self.viewer.end_frame()
        if self._mujoco_viewer:
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
    finally:
        example.cleanup()

    example.test_final()
