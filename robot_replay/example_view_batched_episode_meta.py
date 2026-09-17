"""View plant settle/weld from converted batched-style episode metadata JSON.

Bit-1 eyeball companion to ``example_view_pre_grasp_settle.py`` (native parquet).
Load JSON produced by ``convert_real_to_batched_sysid_metadata.py``.

Run from the repository root::

    uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \\
      --input robot_replay/s00-d00.parquet \\
      --out /tmp/s00_d00_episode_meta.json

    uv run python robot_replay/example_view_batched_episode_meta.py \\
      --episode-meta /tmp/s00_d00_episode_meta.json \\
      --grasp-after-settle \\
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
from apple_pick_sim.fruiting_system.params import (
    fruiting_params_from_dict,
    parse_fixture_args,
)
from apple_pick_sim.sim_device import resolve_sim_device
from apple_pick_sim.system_id.real_post_grasp_plan import (
    PostGraspPlan,
    apply_post_grasp_after_settle,
    format_post_grasp_plan,
)

SETTLE_QUIET_EVERY: int | None = 100
Phase = Literal["long_settle", "short_settle", "run"]


def log_robot_base_axes(
    viewer,
    origin: tuple[float, float, float] | None,
    length: float,
) -> None:
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


def _make_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--episode-meta",
        type=Path,
        required=True,
        help="Batched-style episode metadata JSON from the real→batched converter.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Override fixture path (default: episode_meta.fixture_path).",
    )
    parser.add_argument("--settle-substeps", type=int, default=2000)
    parser.add_argument("--grasp-after-settle", action="store_true")
    parser.add_argument("--apple-position-only", action="store_true")
    parser.add_argument("--post-grasp-settle-substeps", type=int, default=500)
    parser.add_argument("--tcp-radius-warn-m", type=float, default=0.02)
    parser.add_argument(
        "--settle-quiet-every",
        type=int,
        default=SETTLE_QUIET_EVERY,
        metavar="N",
    )
    parser.add_argument("--robot-base-axes", action="store_true")
    parser.add_argument("--robot-base-axes-length", type=float, default=0.3)
    return parser


def _plan_from_episode_meta(meta: dict) -> PostGraspPlan:
    tcp = tuple(float(x) for x in meta["initial_tcp_pos"])
    tcp_q = tuple(float(x) for x in meta["initial_tcp_quat"])
    apple = tuple(float(x) for x in meta["initial_apple_pos"])
    apple_q = tuple(float(x) for x in meta["initial_apple_quat"])
    weld = tuple(float(x) for x in meta["weld_direction"])
    r = float(meta["apple_radius"])
    d = float(np.linalg.norm(np.asarray(tcp) - np.asarray(apple)))
    return PostGraspPlan(
        tcp_pos=tcp,
        tcp_quat_xyzw=tcp_q,
        apple_pos_measured=apple,
        apple_quat_xyzw=apple_q,
        apple_pos_welded=apple,
        weld_direction=weld,
        apple_radius_m=r,
        tcp_apple_distance_m=d,
        tcp_radius_residual_m=abs(d - r),
        apple_shift_m=0.0,
        tcp_approach_dot_weld=float("nan"),
    )


class ExampleViewBatchedEpisodeMeta:
    """Newton viewer: settle/weld from converted batched episode metadata."""

    def __init__(self, viewer, args: argparse.Namespace):
        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 30
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self._device = resolve_sim_device(getattr(args, "device", None))
        meta = json.loads(Path(args.episode_meta).read_text(encoding="utf-8"))
        self._meta = meta
        fixture = Path(args.fixture) if args.fixture is not None else Path(meta["fixture_path"])
        if not fixture.is_file():
            raise FileNotFoundError(f"fixture not found: {fixture}")

        self._params = fruiting_params_from_dict(meta["fruiting_system_params"])
        self._base_pos = tuple(float(x) for x in meta["fruiting_base_pos"])
        self._ranges = load_ranges(fixture)
        self._robot_base = parse_fixture_args(self._ranges).robot_base_pos
        self._robot_base_axes = bool(args.robot_base_axes)
        self._robot_base_axes_length = float(args.robot_base_axes_length)
        if self._robot_base_axes and self._robot_base is None:
            print("Warning: --robot-base-axes set but robot_base_pos is None; skipping.")
            self._robot_base_axes = False

        print(
            f"episode_id={meta.get('episode_id')!r} "
            f"fruiting_base_pos={self._base_pos} "
            f"apple_radius={meta.get('apple_radius')} "
            f"rod_radii={meta.get('rod_radii')}"
        )
        self._scene = generate_coupled_cable_scene(
            self._ranges,
            seed=0,
            params=self._params,
            base_pos=self._base_pos,
            device=self._device,
            gripper_proxy=GripperProxyConfig(fix_to_apple=False),
            robot_base_pos=self._robot_base,
        )
        print(f"Geometry fingerprint: {geometry_fingerprint_coupled(self._scene)}")

        quiet_raw = int(args.settle_quiet_every)
        self._quiet_every: int | None = quiet_raw if quiet_raw > 0 else None
        self._grasp_after_settle = bool(args.grasp_after_settle)
        self._apple_position_only = bool(args.apple_position_only)
        self._tcp_radius_warn_m = float(args.tcp_radius_warn_m)
        self._post_grasp_settle_total = max(0, int(args.post_grasp_settle_substeps))
        self._long_settle_total = max(0, int(args.settle_substeps))
        self._settle_remaining = self._long_settle_total
        self._settle_completed = 0
        self._phase: Phase = (
            "long_settle"
            if self._settle_remaining > 0
            else ("short_settle" if self._grasp_after_settle else "run")
        )
        if self._phase == "short_settle":
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
            f"post_grasp_settle={self._post_grasp_settle_total}"
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
        plan = _plan_from_episode_meta(self._meta)
        print(format_post_grasp_plan(plan))
        mode = (
            "apple position-only (keep settle orientation)"
            if self._apple_position_only
            else "true TCP + apple SE(3)"
        )
        print(f"Applying post-grasp weld from episode meta ({mode})…")
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
                        self._phase = "run"
                    else:
                        print(
                            f"Short post-grasp settle: {self._settle_remaining} VBD substeps…"
                        )
                else:
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
    example = ExampleViewBatchedEpisodeMeta(viewer, args)
    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()
    print("Starting simulation…")
    while viewer.is_running():
        example.step()
        example.render()


if __name__ == "__main__":
    main()
