"""Interactive variational fruiting-system example (P0).

The first built rod in the chain has its base **pinned** in world space (in the
default JSON workflow that is the **primary** segment). In the range file, JSON
``null`` for ``secondary``, ``spur``, ``stem``, or ``apple`` omits that piece from
:func:`apple_pick_sim.fruiting_system.sample_params` / :func:`apple_pick_sim.fruiting_system.generate_scene`;
the remaining segments stay connected in order.

Each **script run** uses a new random seed unless you pass ``--seed``.
From code, call :meth:`ExampleFruitingSystem.regenerate` to build another instance
with a new seed while keeping the same viewer.

Run from the repository root (see README)::

    uv run --directory newton python ../apple_pick_sim/example_fruiting_system.py

Optional arguments (Newton example parser + extras)::

    uv run --directory newton python ../apple_pick_sim/example_fruiting_system.py \\
        --json ../apple_pick_sim/fixtures/fruiting_system_ranges.json --seed 123

    ``--no-self-collision`` keeps **adjacent** link filtering only (joint defaults). It no longer
    registers filters for every chain body pair, so distant links may collide (ground unchanged).
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

from apple_pick_sim.fruiting_system import FruitingSystemScene, generate_scene, geometry_fingerprint, load_ranges


def _default_ranges_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "fruiting_system_ranges.json"


def _make_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path to fruiting-system range JSON (default: apple_pick_sim/fixtures/fruiting_system_ranges.json).",
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
        help="Adjacent chain links only (no longer filters all chain pairs; ground unchanged).",
    )
    return parser


class ExampleFruitingSystem:
    """Newton viewer example: variational primary→secondary→spur→stem→apple chain."""

    def __init__(self, viewer, args: argparse.Namespace | None = None):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 50
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        json_path = getattr(args, "json", None) if args else None
        self.ranges_path = Path(json_path) if json_path else _default_ranges_path()
        self.ranges = load_ranges(self.ranges_path)

        first_seed = getattr(args, "seed", None) if args else None
        if first_seed is None:
            first_seed = secrets.randbelow(2**31 - 1)
        print(f"Fruiting system ranges: {self.ranges_path}")
        print(f"Initial seed: {first_seed}")

        self._scene: FruitingSystemScene | None = None
        self.graph = None
        self.collision_pipeline = None
        self.contacts = None

        self.regenerate(first_seed)

    def _device_str(self) -> str:
        return str(wp.get_device())

    def regenerate(self, seed: int | None = None) -> int:
        """Build a new fruiting system from ``self.ranges`` using ``seed``.

        If ``seed`` is None, picks a fresh random seed.

        Returns:
            The seed used for this build.
        """
        if seed is None:
            seed = secrets.randbelow(2**31 - 1)
        print(f"Regenerating fruiting system (seed={seed}) …")
        enable_self = not (
            getattr(self.args, "no_self_collision", False) if self.args else False
        )
        self._scene = generate_scene(
            self.ranges,
            seed,
            device=self._device_str(),
            omit=[],
            enable_self_collisions=enable_self,
        )
        fp = geometry_fingerprint(self._scene)
        pb, sb = fp.get("primary_bend_stiffness"), fp.get("secondary_bend_stiffness")
        prim_txt = f"primary bend={pb:.1f}" if pb is not None else "primary=OFF"
        sec_txt = f"secondary bend={sb:.1f}" if sb is not None else "secondary=OFF"
        apos = fp.get("apple_pos")
        ap_txt = f"apple at {apos}" if apos is not None else "apple=OFF"
        print(f"Geometry fingerprint ({ap_txt}): {prim_txt}, {sec_txt}")

        self.model = self._scene.model
        self.state_0 = self._scene.state_0
        self.state_1 = self._scene.state_1
        self.control = self._scene.control
        self.solver = self._scene.solver

        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, self.args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)
        self.capture()
        self.sim_time = 0.0
        return seed

    def capture(self) -> None:
        if self.solver.device.is_cuda:
            self.capturing = True
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.capturing = False
            self.graph = cap.graph
        else:
            self.graph = None

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self, warmup: bool = False) -> None:
        del warmup  # unused; kept for API symmetry with example_apple_stem
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self, tolerance: float = 0.05) -> None:
        """Light sanity check: apple stays near rest height and above ground."""
        if self._scene is None:
            raise RuntimeError("No scene built")
        body_q = self.state_0.body_q.numpy()
        apple = self._scene.apple_body
        if apple is None:
            return
        z = float(body_q[apple, 2])
        assert z > -tolerance, f"Apple fell through ground: z={z}"


if __name__ == "__main__":
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--num-frames", "2000"])
            print("No DISPLAY/WAYLAND_DISPLAY: using --viewer null --num-frames 2000 (override with --viewer gl).")

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)

    example = ExampleFruitingSystem(viewer, args)

    # ``newton.examples.init()`` shows a GL loading splash; ``run()`` hides it.
    # We drive the loop manually, so clear it here (same as ``examples.run``).
    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    print("Warming up…")
    for _ in range(30):
        example.step(warmup=True)

    print("Starting simulation…")
    while viewer.is_running():
        example.step()
        example.render()

    example.test_final()
