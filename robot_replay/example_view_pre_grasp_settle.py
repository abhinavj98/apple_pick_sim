"""Build plant-only scene from real pre_grasp_geometry, settle, and visualize.

Run from the repository root::

    uv run python robot_replay/example_view_pre_grasp_settle.py \\
      --parquet robot_replay/s00-d00.parquet \\
      --fixture apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json \\
      --settle-substeps 5000 \\
      --settle-quiet-every 300 \\
      --viewer gl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import newton
import newton.examples
import warp as wp

from apple_pick_sim.coupled_fruiting.settle_then_weld import (
    quiet_all_cable_bodies,
    should_quiet_cable_bodies_at_settle_substep,
    warn_settle_quiet_every_alignment,
)
from apple_pick_sim.fruiting_system import (
    GripperProxyConfig,
    example_collision_pipeline,
    geometry_fingerprint_coupled,
    load_ranges,
)
from apple_pick_sim.fruiting_system.coupled import generate_coupled_cable_scene
from apple_pick_sim.fruiting_system.params import fruiting_params_to_dict, parse_fixture_args
from apple_pick_sim.sim_device import resolve_sim_device
from apple_pick_sim.system_id.real_pre_grasp_params import (
    format_pre_grasp_diagnostics,
    fruiting_params_from_pre_grasp_parquet,
)

SETTLE_QUIET_EVERY: int | None = 300


def _make_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(
            "apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json"
        ),
    )
    parser.add_argument(
        "--settle-substeps",
        type=int,
        default=5000,
        help="VBD substeps for visible settle (rendered in the viewer).",
    )
    parser.add_argument(
        "--settle-quiet-every",
        type=int,
        default=SETTLE_QUIET_EVERY,
        metavar="N",
        help=(
            "Zero all cable body twists every N VBD settle substeps "
            f"(device-side; default: {SETTLE_QUIET_EVERY}; <=0 disables)."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if pre-grasp woody_bending_angles are not ~0.",
    )
    parser.add_argument(
        "--dump-params",
        type=Path,
        default=None,
        help="Write fruiting_base_pos + fruiting_system_params + diagnostics JSON.",
    )
    return parser


class ExampleViewPreGraspSettle:
    """Newton viewer: rebuild plant from pre-grasp, settle visibly, keep simulating."""

    def __init__(self, viewer, args: argparse.Namespace):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 30
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        device = resolve_sim_device(getattr(args, "device", None))
        params, base_pos, diagnostics = fruiting_params_from_pre_grasp_parquet(
            args.parquet,
            fixture_path=args.fixture,
            strict=bool(args.strict),
        )
        print(format_pre_grasp_diagnostics(diagnostics))
        if args.dump_params is not None:
            args.dump_params.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "fruiting_base_pos": list(base_pos),
                "fruiting_system_params": fruiting_params_to_dict(params),
                "diagnostics": diagnostics,
            }
            args.dump_params.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {args.dump_params}")

        ranges = load_ranges(args.fixture)
        robot_base = parse_fixture_args(ranges).robot_base_pos
        self._scene = generate_coupled_cable_scene(
            ranges,
            seed=0,
            params=params,
            base_pos=base_pos,
            device=device,
            gripper_proxy=GripperProxyConfig(fix_to_apple=False),
            robot_base_pos=robot_base,
        )
        print(f"fruiting_base_pos (Branch / T-junction): {base_pos}")
        print(f"Geometry fingerprint: {geometry_fingerprint_coupled(self._scene)}")
        self._settle_total = max(0, int(args.settle_substeps))
        self._settle_remaining = self._settle_total
        self._settle_completed = 0
        quiet_raw = int(args.settle_quiet_every)
        self._quiet_every: int | None = quiet_raw if quiet_raw > 0 else None
        warn_settle_quiet_every_alignment(self._settle_total, self._quiet_every)
        print(
            f"Visible settle: {self._settle_remaining} VBD substeps "
            f"(quiet_every={self._quiet_every}; then continue simulating)."
        )

        self.model = self._scene.model
        self.state_0 = self._scene.state_0
        self.state_1 = self._scene.state_1
        self.control = self._scene.control
        self.solver = self._scene.solver
        self.collision_pipeline = example_collision_pipeline(self.model)
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(0.5, -0.8, 1.6), pitch=-20.0, yaw=45.0)

    def capture_video(self, duration_seconds: float = 0.0) -> None:
        return None

    def step(self) -> None:
        # During settle budget, run more substeps per frame so a large settle
        # finishes while still showing motion; afterward one frame of normal substeps.
        settling = self._settle_remaining > 0
        if settling:
            n = min(self.sim_substeps * 10, self._settle_remaining)
            self._settle_remaining -= n
        else:
            n = self.sim_substeps
        for _ in range(n):
            self._scene.state_0.clear_forces()
            contacts = self.model.collide(
                self._scene.state_0, collision_pipeline=self.collision_pipeline
            )
            self.solver.step(
                self._scene.state_0, self._scene.state_1, self.control, contacts, self.sim_dt
            )
            self._scene.state_0, self._scene.state_1 = self._scene.state_1, self._scene.state_0
            self.state_0 = self._scene.state_0
            self.state_1 = self._scene.state_1
            if settling:
                self._settle_completed += 1
                if should_quiet_cable_bodies_at_settle_substep(
                    self._settle_completed, self._quiet_every
                ):
                    quiet_all_cable_bodies(self._scene)
        if settling and self._settle_remaining == 0:
            print("Settle complete; continuing VBD simulation.")
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


def main() -> None:
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            sys.argv.extend(["--viewer", "null", "--num-frames", "30"])
            print("No display: using --viewer null --num-frames 30")
    viewer, args = newton.examples.init(parser=_make_parser())
    example = ExampleViewPreGraspSettle(viewer, args)
    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()
    print("Starting simulation…")
    while viewer.is_running():
        example.step()
        example.render()


if __name__ == "__main__":
    main()
