"""Build and simulate a digital twin from field observations (``digital_twin_v1`` JSON).

Loads junction anchor positions and weld direction from an observation file, infers
straight-rod geometry, optionally settles the VBD cable, and opens the Newton GL viewer.

Run from the repository root::

    uv run python apple_pick_sim/examples/example_digital_twin.py \\
      --obs apple_pick_sim/fixtures/digital_twin_obs_example.json \\
      --base-fixture apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy.json

Optional ``--settle-substeps`` runs quasi-static VBD settling before the interactive loop.
Pass ``--no-fix-to-apple`` to keep a free gripper proxy (weld direction is ignored).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import newton
import newton.examples
import warp as wp

from apple_pick_sim.digital_twin import build_digital_twin_scene, load_digital_twin_obs
from apple_pick_sim.fruiting_system import (
    default_ranges_fixture_path,
    example_collision_pipeline,
    fixed_joint_anchors_world,
    geometry_fingerprint_coupled,
    run_rollout,
)
from apple_pick_sim.sim_device import resolve_sim_device


def _default_obs_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "fixtures"
        / "digital_twin_obs_example.json"
    )


def _default_base_fixture_path() -> Path:
    return default_ranges_fixture_path()


def _settle_cable_substeps(scene, *, substeps: int, dt: float) -> None:
    """Advance a cable scene by ``substeps`` VBD steps (no robot coupling)."""
    n = int(substeps)
    if n <= 0:
        return
    pipeline = example_collision_pipeline(scene.model)
    for _ in range(n):
        scene.state_0.clear_forces()
        contacts = scene.model.collide(scene.state_0, collision_pipeline=pipeline)
        scene.solver.step(scene.state_0, scene.state_1, scene.control, contacts, dt)
        scene.state_0, scene.state_1 = scene.state_1, scene.state_0


def _make_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--obs",
        type=str,
        default=None,
        help="Path to digital_twin_v1 observation JSON (default: fixtures/digital_twin_obs_example.json).",
    )
    parser.add_argument(
        "--base-fixture",
        type=str,
        default=None,
        help=(
            "Base range JSON for non-geometric parameters "
            "(default: fixtures/fruiting_system_ranges_real_world_proxy_variance.json)."
        ),
    )
    parser.add_argument(
        "--settle-substeps",
        type=int,
        default=0,
        help="VBD substeps to run before the viewer loop (quasi-static settling).",
    )
    parser.add_argument(
        "--fix-to-apple",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Weld gripper proxy to apple using obs weld_direction (default: on).",
    )
    parser.add_argument(
        "--enable-self-collision",
        action="store_true",
        help="Enable intra-chain self collisions on the cable model.",
    )
    return parser


class ExampleDigitalTwin:
    """Newton viewer example: digital twin from field observations."""

    def __init__(self, viewer, args: argparse.Namespace | None = None):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 30
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        obs_path = Path(args.obs) if args and args.obs else _default_obs_path()
        base_fixture = (
            Path(args.base_fixture) if args and args.base_fixture else _default_base_fixture_path()
        )
        device = resolve_sim_device(getattr(args, "device", None) if args else None)

        print(f"Observation file: {obs_path}")
        print(f"Base fixture: {base_fixture}")
        print(f"Warp device: {device}")

        obs = load_digital_twin_obs(obs_path)
        self._scene = build_digital_twin_scene(
            obs,
            base_fixture,
            device=device,
            fix_to_apple=bool(getattr(args, "fix_to_apple", True)) if args else True,
            enable_self_collisions=bool(getattr(args, "enable_self_collision", False))
            if args
            else False,
        )

        settle = int(getattr(args, "settle_substeps", 0) or 0) if args else 0
        if settle > 0:
            print(f"Settling cable for {settle} VBD substeps …")
            _settle_cable_substeps(self._scene, substeps=settle, dt=self.sim_dt)

        fp = geometry_fingerprint_coupled(self._scene)
        print(f"Geometry fingerprint: {fp}")

        parent, child = fixed_joint_anchors_world(
            self._scene.model,
            self._scene.state_0.body_q,
            self._scene.fruiting_fixed_joints,
        )
        print(f"Junctions: {[l.removeprefix('joint_') for _, l in self._scene.fruiting_fixed_joints]}")
        print(f"Rebuilt parent anchors (first junction): {parent[:3]}")

        self.model = self._scene.model
        self.state_0 = self._scene.state_0
        self.state_1 = self._scene.state_1
        self.control = self._scene.control
        self.solver = self._scene.solver
        self.collision_pipeline = example_collision_pipeline(self.model)

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(0.5, -0.8, 1.6), pitch=-20.0, yaw=45.0)

    def step(self):
        self.sim_time += self.frame_dt
        run_rollout(
            self._scene,
            num_steps=1,
            sim_substeps=self.sim_substeps,
            fps=self.fps,
            collision_pipeline=self.collision_pipeline,
        )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--num-frames", "2000"])
            print("No DISPLAY/WAYLAND_DISPLAY: using --viewer null --num-frames 2000 (override with --viewer gl).")

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)
    example = ExampleDigitalTwin(viewer, args)

    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    print("Starting simulation…")
    while viewer.is_running():
        example.step()
        example.render()
