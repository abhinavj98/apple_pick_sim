"""Build plant-only scene from real pre_grasp_geometry, settle, optionally grasp, view.

Run from the repository root::

    uv run python robot_replay/example_view_pre_grasp_settle.py \\
      --parquet robot_replay/s00-d00.parquet \\
      --fixture apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json \\
      --settle-substeps 5000 \\
      --settle-quiet-every 300 \\
      --viewer gl

With post-grasp snap (true logged TCP + apple SE(3) weld)::

    uv run python robot_replay/example_view_pre_grasp_settle.py \\
      --parquet robot_replay/s00-d00.parquet \\
      --grasp-after-settle \\
      --post-grasp-settle-substeps 500 \\
      --viewer gl

Apple translation only (keep settle orientation; ignore logged apple quat)::

    uv run python robot_replay/example_view_pre_grasp_settle.py \\
      --parquet robot_replay/s00-d00.parquet \\
      --grasp-after-settle \\
      --apple-position-only \\
      --viewer gl

With robot-base RGB axes at fixture ``robot_base_pos``::

    uv run python robot_replay/example_view_pre_grasp_settle.py \\
      --parquet robot_replay/s00-d00.parquet \\
      --robot-base-axes \\
      --viewer gl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal

import newton
import newton.examples
import numpy as np
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
    load_dataset_metadata,
)
from apple_pick_sim.system_id.real_post_grasp_plan import (
    apply_post_grasp_after_settle,
    format_post_grasp_plan,
    post_grasp_plan_from_metadata,
)

SETTLE_QUIET_EVERY: int | None = 100
Phase = Literal["long_settle", "short_settle", "run"]


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
        default=2000,
        help="VBD substeps for visible free settle before grasp (rendered in the viewer).",
    )
    parser.add_argument(
        "--grasp-after-settle",
        action="store_true",
        help=(
            "After long settle, weld proxy at logged TCP SE(3) and apple at measured "
            "post-grasp pose (no catalog surface snap); then run --post-grasp-settle-substeps."
        ),
    )
    parser.add_argument(
        "--apple-position-only",
        action="store_true",
        help=(
            "With --grasp-after-settle: set apple translation from logged post-grasp "
            "data but keep apple orientation from the free settle (ignore logged "
            "apple quat). TCP SE(3) is unchanged."
        ),
    )
    parser.add_argument(
        "--post-grasp-settle-substeps",
        type=int,
        default=500,
        help="Short VBD settle after grasp (default: 500).",
    )
    parser.add_argument(
        "--tcp-radius-warn-m",
        type=float,
        default=0.02,
        help="Warn if ||tcp−apple|−r| or apple shift exceeds this (default: 0.02 m).",
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
    parser.add_argument(
        "--robot-base-axes",
        action="store_true",
        help=(
            "Draw robot-base-frame X/Y/Z axes as RGB arrows at robot_base_pos "
            "(requires --viewer gl or viser)."
        ),
    )
    parser.add_argument(
        "--robot-base-axes-length",
        type=float,
        default=0.3,
        help="Length [m] of each robot-base axis arrow (default 0.3).",
    )
    return parser


def log_robot_base_axes(
    viewer,
    origin: tuple[float, float, float] | None,
    length: float,
) -> None:
    """Draw world-aligned RGB axes at ``origin`` (robot base). No-op if origin is None."""
    if origin is None:
        return
    log_lines = getattr(viewer, "log_lines", None)
    log_arrows = getattr(viewer, "log_arrows", None)
    _log = log_arrows if log_arrows is not None else log_lines
    if _log is None:
        return
    dev = str(getattr(viewer, "device", "cpu"))
    o = np.array(origin, dtype=np.float32)
    L = float(length)
    axes = [
        ("x", np.array([L, 0.0, 0.0], dtype=np.float32), np.array([[1.0, 0.0, 0.0]], dtype=np.float32)),
        ("y", np.array([0.0, L, 0.0], dtype=np.float32), np.array([[0.0, 1.0, 0.0]], dtype=np.float32)),
        ("z", np.array([0.0, 0.0, L], dtype=np.float32), np.array([[0.0, 0.0, 1.0]], dtype=np.float32)),
    ]
    for label, tip_offset, color in axes:
        starts = wp.array(o.reshape(1, 3), dtype=wp.vec3, device=dev)
        ends = wp.array((o + tip_offset).reshape(1, 3), dtype=wp.vec3, device=dev)
        colors = wp.array(color, dtype=wp.vec3, device=dev)
        _log(f"/debug/robot_base_axis_{label}", starts, ends, colors)


class ExampleViewPreGraspSettle:
    """Newton viewer: pre-grasp settle, optional post-grasp true-TCP weld, keep simulating."""

    def __init__(self, viewer, args: argparse.Namespace):
        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 30
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self._device = resolve_sim_device(getattr(args, "device", None))
        self._params, self._base_pos, diagnostics = fruiting_params_from_pre_grasp_parquet(
            args.parquet,
            fixture_path=args.fixture,
            strict=bool(args.strict),
        )
        print(format_pre_grasp_diagnostics(diagnostics))
        if args.dump_params is not None:
            args.dump_params.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "fruiting_base_pos": list(self._base_pos),
                "fruiting_system_params": fruiting_params_to_dict(self._params),
                "diagnostics": diagnostics,
            }
            args.dump_params.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {args.dump_params}")

        self._ranges = load_ranges(args.fixture)
        self._robot_base = parse_fixture_args(self._ranges).robot_base_pos
        self._robot_base_axes = bool(args.robot_base_axes)
        self._robot_base_axes_length = float(args.robot_base_axes_length)
        if self._robot_base_axes and self._robot_base is None:
            print("Warning: --robot-base-axes set but robot_base_pos is None; skipping axes.")
            self._robot_base_axes = False
        self._scene = generate_coupled_cable_scene(
            self._ranges,
            seed=0,
            params=self._params,
            base_pos=self._base_pos,
            device=self._device,
            gripper_proxy=GripperProxyConfig(fix_to_apple=False),
            robot_base_pos=self._robot_base,
        )
        print(f"fruiting_base_pos (Branch / T-junction): {self._base_pos}")
        print(f"Geometry fingerprint: {geometry_fingerprint_coupled(self._scene)}")

        quiet_raw = int(args.settle_quiet_every)
        self._quiet_every: int | None = quiet_raw if quiet_raw > 0 else None
        self._grasp_after_settle = bool(args.grasp_after_settle)
        self._apple_position_only = bool(args.apple_position_only)
        if self._apple_position_only and not self._grasp_after_settle:
            print(
                "Warning: --apple-position-only has no effect without --grasp-after-settle."
            )
        self._tcp_radius_warn_m = float(args.tcp_radius_warn_m)
        self._post_grasp_settle_total = max(0, int(args.post_grasp_settle_substeps))

        self._long_settle_total = max(0, int(args.settle_substeps))
        self._settle_remaining = self._long_settle_total
        self._settle_completed = 0
        self._phase: Phase = "long_settle" if self._settle_remaining > 0 else (
            "short_settle" if self._grasp_after_settle else "run"
        )
        if self._phase == "short_settle":
            # No long settle requested but grasp enabled: apply immediately.
            self._apply_grasp()
            self._settle_remaining = self._post_grasp_settle_total
            self._settle_completed = 0

        warn_settle_quiet_every_alignment(self._long_settle_total, self._quiet_every)
        if self._grasp_after_settle:
            warn_settle_quiet_every_alignment(
                self._post_grasp_settle_total, self._quiet_every
            )
        print(
            f"Phase={self._phase}: long_settle={self._long_settle_total} "
            f"grasp={self._grasp_after_settle} "
            f"post_grasp_settle={self._post_grasp_settle_total} "
            f"(quiet_every={self._quiet_every})."
        )

        self._bind_scene_to_viewer()
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(0.5, -0.8, 1.6), pitch=-20.0, yaw=45.0)

    def _bind_scene_to_viewer(self) -> None:
        self.model = self._scene.model
        self.state_0 = self._scene.state_0
        self.state_1 = self._scene.state_1
        self.control = self._scene.control
        self.solver = self._scene.solver
        self.collision_pipeline = example_collision_pipeline(self.model)
        self.viewer.set_model(self.model)

    def _apply_grasp(self) -> None:
        meta = load_dataset_metadata(self.args.parquet)
        plan = post_grasp_plan_from_metadata(
            meta,
            apple_radius_m=float(self._params.apple_radius),
            warn_tol_m=self._tcp_radius_warn_m,
            emit_warnings=True,
        )
        print(format_post_grasp_plan(plan))
        mode = (
            "apple position-only (keep settle orientation)"
            if self._apple_position_only
            else "true TCP + apple SE(3)"
        )
        print(f"Applying post-grasp weld ({mode})…")
        self._scene = apply_post_grasp_after_settle(
            self._scene,
            plan,
            ranges=self._ranges,
            params=self._params,
            base_pos=self._base_pos,
            device=self._device,
            robot_base_pos=self._robot_base,
            proxy_tcp_pos_warn_m=self._tcp_radius_warn_m,
            emit_warnings=True,
            keep_apple_settle_orientation=self._apple_position_only,
        )
        print(f"Post-grasp fingerprint: {geometry_fingerprint_coupled(self._scene)}")
        self._bind_scene_to_viewer()

    def capture_video(self, duration_seconds: float = 0.0) -> None:
        return None

    def step(self) -> None:
        settling = self._phase in ("long_settle", "short_settle") and self._settle_remaining > 0
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
            if self._phase == "long_settle":
                print("Long settle complete.")
                if self._grasp_after_settle:
                    self._apply_grasp()
                    self._phase = "short_settle"
                    self._settle_remaining = self._post_grasp_settle_total
                    self._settle_completed = 0
                    if self._settle_remaining == 0:
                        print("Post-grasp settle skipped (0 substeps); continuing.")
                        self._phase = "run"
                    else:
                        print(
                            f"Short post-grasp settle: {self._settle_remaining} VBD substeps…"
                        )
                else:
                    print("Continuing VBD simulation.")
                    self._phase = "run"
            elif self._phase == "short_settle":
                print("Post-grasp settle complete; continuing VBD simulation.")
                self._phase = "run"

        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self._robot_base_axes:
            log_robot_base_axes(
                self.viewer, self._robot_base, self._robot_base_axes_length
            )
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
