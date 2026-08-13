#!/usr/bin/env python3
"""Real FR3+VIC replay test for an exported real→batched dataset.

Uses ``replay_batched_sysid_structure`` (same path as CMA / MMD grid) after free
settle → weld → post-grasp settle. Default ``--controller-mode vic_pose`` drives
19D pose+gains actions packed at convert time from real ``target_pose_4x4`` +
``dump.controller_gains`` (real parquet ``action`` is a pose-control wrench, not
an EE twist). Use ``--controller-mode vic`` only for legacy 6D-twist datasets.

Apple lifecycle matches ``example_view_pre_grasp_settle.py --grasp-after-settle``:
pre-grasp ``apple_quat_xyzw`` for free settle; logged post-grasp apple + TCP SE(3)
at weld (true ``weld_proxy_offset_in_apple_frame``).

Settle defaults match ``example_view_pre_grasp_settle.py``:
``--settle-substeps 5000``, ``--settle-quiet-every 300``,
``--post-grasp-settle-substeps 500``.

With ``--viewer gl``, renders trajectory frames after off-screen rebuild/settle
(same minimal ``on_step`` pattern as ``example_batched_sysid_mmd_grid.py``).
When episode metadata includes ``camera_to_base_4x4`` (from convert), places the
GL camera at the real recording-camera pose (position + OpenCV +Z look).

Geometry comes from converted episode metadata (same native rebuild as
``example_view_pre_grasp_settle`` / ``example_view_batched_episode_meta``):
oracle ``fruiting_system_params`` and episode ``fruiting_base_pos``. Arm
placement is **open-loop** from ``initial_robot_joint_q`` (skip IK). Sim
``control_hz`` comes from episode / collection metadata (real recording rate).
Physics uses ``gym_defaults`` + fixture ``sim_build`` on the default sim device.

Example (after export)::

    uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \\
      --input robot_replay/s02-d00.parquet \\
      --dataset-out /tmp/real_batched_s02_d00 --overwrite

    uv run python robot_replay/example_replay_real_batched.py \\
      --dataset /tmp/real_batched_s02_d00 --viewer gl --max-frames 0 \\
      --settle-substeps 5000 --settle-quiet-every 300 \\
      --post-grasp-settle-substeps 500 \\
      --print-woody-forces 5

Headless MP4 (requires ``imageio-ffmpeg`` via ``uv sync --extra gym``)::

    uv run python robot_replay/example_replay_real_batched.py \\
      --dataset /tmp/real_batched_s02_d00 --viewer gl --headless \\
      --record-video /tmp/replay.mp4 --max-frames 0
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import newton.examples
import warp as wp

from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
    gt_bend_stiffness_candidate_from_structure,
    replay_batched_sysid_structure,
)
from apple_pick_gym.batched_envs.real_batched_replay_build import (
    bootstrap_joint_q_from_episode_metadata,
    check_action_semantics,
    control_hz_from_episode_metadata,
    fruiting_base_pos_from_episode_metadata,
    make_real_replay_build_env_fn,
    real_replay_sim_config,
)
from apple_pick_sim.fruiting_system.params import load_ranges
from apple_pick_sim.system_id import BatchedSysIdDataset

# Allow ``uv run python robot_replay/example_replay_real_batched.py`` imports.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from robot_replay.gl_video_recorder import GlVideoRecorder  # noqa: E402

_DEFAULT_FIXTURE = Path(
    "apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json"
)
# Match example_view_pre_grasp_settle.py defaults.
_SETTLE_SUBSTEPS = 5000
_SETTLE_QUIET_EVERY: int | None = 300
_SETTLE_GRAVITY_RAMP = False
_POST_GRASP_SETTLE_SUBSTEPS = 500
_CONTROL_HZ_FALLBACK = 15.0
_DEFAULT_CONTROLLER_MODE = "vic_pose"

# CLI tests import these names from the example module.
_test_sim_config = real_replay_sim_config
_build_env_fn = make_real_replay_build_env_fn


def woody_forces_from_last_obs(
    last_obs: Mapping[str, Any] | None,
    junction_names: list[str] | tuple[str, ...],
    *,
    env_idx: int = 0,
) -> dict[str, tuple[float, float, float]]:
    """Linear force [N] on each woody FIXED junction child (env ``env_idx``)."""
    if last_obs is None:
        return {}
    info = last_obs.get("woody_part_info")
    if not isinstance(info, Mapping):
        return {}
    out: dict[str, tuple[float, float, float]] = {}
    for name in junction_names:
        part = info.get(name)
        if not isinstance(part, Mapping):
            continue
        wrench = part.get("anchor_force")
        if wrench is None:
            continue
        if hasattr(wrench, "detach"):
            row = wrench.detach().cpu().numpy()
        else:
            row = np.asarray(wrench)
        row = np.asarray(row, dtype=np.float64).reshape(-1, 6)
        if env_idx < 0 or env_idx >= row.shape[0]:
            continue
        fx, fy, fz = (float(row[env_idx, 0]), float(row[env_idx, 1]), float(row[env_idx, 2]))
        out[str(name)] = (fx, fy, fz)
    return out


def format_woody_force_lines(
    forces: Mapping[str, tuple[float, float, float] | list[float] | np.ndarray],
    *,
    frame_idx: int,
) -> list[str]:
    """Human-readable woody force dump for one control frame."""
    lines = [f"woody_forces frame={int(frame_idx)}"]
    for name, f_xyz in forces.items():
        arr = np.asarray(f_xyz, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(arr))
        lines.append(
            f"  {name} F=[{arr[0]:.3f}, {arr[1]:.3f}, {arr[2]:.3f}] |F|={norm:.3f}"
        )
    return lines


def gl_camera_from_camera_to_base(
    camera_to_base_4x4: object,
) -> tuple[tuple[float, float, float], float, float] | None:
    """Map camera→base SE(3) to Newton GL ``set_camera`` (pos, pitch_deg, yaw_deg).

    Look direction is the camera **+Z** axis in base (OpenCV optical axis).
    Newton GL has no roll; up remains world-Z.
    """
    try:
        arr = np.asarray(camera_to_base_4x4, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.size != 16:
        return None
    arr = arr.reshape(4, 4)
    pos = (float(arr[0, 3]), float(arr[1, 3]), float(arr[2, 3]))
    front = arr[:3, 2]
    n = float(np.linalg.norm(front))
    if n < 1e-12:
        return None
    d = front / n
    pitch = float(np.rad2deg(np.arcsin(np.clip(d[2], -1.0, 1.0))))
    yaw = float(np.rad2deg(np.arctan2(d[1], d[0])))
    return pos, pitch, yaw


def require_gl_frame_capture(viewer: object) -> None:
    """Raise ``SystemExit`` unless ``viewer`` supports ``get_frame`` (ViewerGL)."""
    if not hasattr(viewer, "get_frame"):
        raise SystemExit(
            "--record-video requires a GL viewer with get_frame(); "
            "pass --viewer gl (optionally --headless)."
        )


def make_replay_on_step(
    viewer: object,
    *,
    max_frames: int,
    control_hz_fallback: float = _CONTROL_HZ_FALLBACK,
    print_woody_forces_every: int = 0,
    camera_to_base_4x4: object | None = None,
    recorder: GlVideoRecorder | None = None,
) -> Callable[..., bool]:
    """MMD-grid-style render + optional frame cap for real replay.

    When ``print_woody_forces_every > 0``, print env-0 woody FIXED-joint linear
    forces every N control frames (world frame, force on child).

    When ``camera_to_base_4x4`` is set and the viewer supports ``set_camera``,
    place the GL eye at the real recording camera pose on first init.

    When ``recorder`` is set, capture an RGB frame after each ``end_frame`` at
    sim ``control_hz``.
    """
    viewer_state: dict[str, object] = {"initialized": False}
    every = int(print_woody_forces_every)

    def on_step(*, frame_idx: int, env: object) -> bool:
        if hasattr(viewer, "is_running") and not viewer.is_running():
            return False

        sim = getattr(env, "_sim", None)
        scene = getattr(sim, "scene", None) if sim is not None else None
        if scene is not None and hasattr(viewer, "begin_frame"):
            if not viewer_state["initialized"]:
                if hasattr(viewer, "set_model") and hasattr(scene, "cable"):
                    viewer.set_model(scene.cable.model)
                if (
                    hasattr(viewer, "set_world_offsets")
                    and getattr(env, "num_envs", 1) > 1
                ):
                    viewer.set_world_offsets(tuple(sim.config.runtime.env_spacing))
                if hasattr(viewer, "set_camera") and camera_to_base_4x4 is not None:
                    pose = gl_camera_from_camera_to_base(camera_to_base_4x4)
                    if pose is not None:
                        pos, pitch, yaw = pose
                        viewer.set_camera(wp.vec3(*pos), pitch, yaw)
                if hasattr(viewer, "hide_loading_splash"):
                    viewer.hide_loading_splash()
                viewer_state["initialized"] = True
            runtime = getattr(getattr(sim, "config", None), "runtime", None)
            hz = float(getattr(runtime, "control_hz", control_hz_fallback))
            sim_time = float(frame_idx) / max(hz, 1e-9)
            viewer.begin_frame(sim_time)
            if hasattr(viewer, "log_state") and hasattr(scene, "cable"):
                viewer.log_state(scene.cable.state_0)
            viewer.end_frame()
            if recorder is not None:
                if recorder.fps is None:
                    recorder.set_fps(hz)
                recorder.capture(viewer)

        if every > 0 and int(frame_idx) % every == 0:
            names = list(getattr(env, "junction_names", []) or [])
            forces = woody_forces_from_last_obs(
                getattr(env, "_last_obs", None), names, env_idx=0
            )
            if forces:
                print("\n".join(format_woody_force_lines(forces, frame_idx=frame_idx)))

        if max_frames <= 0:
            return True
        return int(frame_idx) + 1 < max_frames

    return on_step


def _make_parser() -> argparse.ArgumentParser:
    p = newton.examples.create_parser()
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Exported batched_sysid_v1 directory (manifest.json + episodes/).",
    )
    p.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Ranges fixture (default: dataset collection.ranges_path or variance fixture).",
    )
    p.add_argument("--structure-idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-frames",
        type=int,
        default=24,
        help="Stop replay after this many frames. Use <=0 for full episode.",
    )
    p.add_argument(
        "--settle-substeps",
        type=int,
        default=_SETTLE_SUBSTEPS,
        help=(
            "VBD substeps for free settle before weld "
            f"(default: {_SETTLE_SUBSTEPS}; matches example_view_pre_grasp_settle)."
        ),
    )
    p.add_argument(
        "--settle-quiet-every",
        type=int,
        default=_SETTLE_QUIET_EVERY if _SETTLE_QUIET_EVERY is not None else 0,
        metavar="N",
        help=(
            "Zero all cable body twists every N VBD settle substeps "
            f"(default: {_SETTLE_QUIET_EVERY}; <=0 disables)."
        ),
    )
    p.add_argument(
        "--settle-gravity-ramp",
        action=argparse.BooleanOptionalAction,
        default=_SETTLE_GRAVITY_RAMP,
        help="Linear 0→−9.81 m/s² gravity ramp over settle substeps (default: off).",
    )
    p.add_argument(
        "--post-grasp-settle-substeps",
        type=int,
        default=_POST_GRASP_SETTLE_SUBSTEPS,
        help=(
            "VBD substeps on the welded scene after fix_to_apple seed "
            f"(default: {_POST_GRASP_SETTLE_SUBSTEPS}; matches example_view_pre_grasp_settle)."
        ),
    )
    p.add_argument(
        "--controller-mode",
        choices=["vic", "vic_pose"],
        default=_DEFAULT_CONTROLLER_MODE,
        help=(
            "'vic' (recorded EE twist, 6D action) or 'vic_pose' (pose+gains, 19D action; "
            "dataset must already carry 19D actions, e.g. via pack_vic_pose_actions.py)."
        ),
    )
    p.add_argument(
        "--allow-wrench-as-twist",
        action="store_true",
        help=(
            "Permit replay when a legacy 6D episode action is a real pose-control "
            "wrench (incorrect physics under mode=vic; format/GL smoke only). "
            "Rejected for 19D vic_pose datasets."
        ),
    )
    p.add_argument(
        "--print-woody-forces",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Print env-0 woody FIXED-joint linear forces every N control frames "
            "(world frame, force on child). <=0 disables (default)."
        ),
    )
    p.add_argument(
        "--record-video",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write GL viewer frames to PATH.mp4 (requires --viewer gl; "
            "--headless OK). FPS matches sim control_hz."
        ),
    )
    return p


def _run(
    args: argparse.Namespace,
    viewer: object,
    *,
    recorder: GlVideoRecorder | None = None,
) -> int:
    dataset = BatchedSysIdDataset(args.dataset)
    collection = dataset.manifest.get("collection", {})
    ranges_path = Path(
        args.fixture
        if args.fixture is not None
        else collection.get("ranges_path") or _DEFAULT_FIXTURE
    )
    if not ranges_path.is_file():
        raise SystemExit(f"ranges fixture not found: {ranges_path}")

    ranges = load_ranges(ranges_path)
    structure_idx = int(args.structure_idx)
    try:
        episode_meta = dataset.load_episode_metadata(structure_idx, 0)
        fruiting_base_pos = fruiting_base_pos_from_episode_metadata(episode_meta)
        bootstrap_joint_q = bootstrap_joint_q_from_episode_metadata(episode_meta)
        control_hz = control_hz_from_episode_metadata(
            episode_meta, collection=collection
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    controller_mode = str(args.controller_mode)
    check_action_semantics(
        controller_mode=controller_mode,
        collection=collection,
        episode_meta=episode_meta,
        allow_wrench_as_twist=bool(args.allow_wrench_as_twist),
    )

    candidates = [gt_bend_stiffness_candidate_from_structure(dataset, structure_idx)]
    max_frames = int(args.max_frames)
    seed = int(args.seed)
    settle_substeps = int(args.settle_substeps)
    quiet_raw = int(args.settle_quiet_every)
    settle_quiet_every: int | None = quiet_raw if quiet_raw > 0 else None
    settle_gravity_ramp = bool(args.settle_gravity_ramp)
    post_grasp_settle_substeps = int(args.post_grasp_settle_substeps)

    build_kwargs = dict(
        ranges_path=ranges_path,
        ranges=ranges,
        topology_seed=seed,
        fruiting_base_pos=fruiting_base_pos,
        episode_meta=episode_meta,
        settle_substeps=settle_substeps,
        settle_quiet_every=settle_quiet_every,
        settle_gravity_ramp=settle_gravity_ramp,
        post_grasp_settle_substeps=post_grasp_settle_substeps,
        bootstrap_joint_q=bootstrap_joint_q,
        controller_mode=controller_mode,
        control_hz=control_hz,
    )
    sim_kwargs = dict(
        num_envs=1,
        topology_seed=seed,
        fruiting_base_pos=fruiting_base_pos,
        ranges=ranges,
        settle_substeps=settle_substeps,
        settle_quiet_every=settle_quiet_every,
        settle_gravity_ramp=settle_gravity_ramp,
        post_grasp_settle_substeps=post_grasp_settle_substeps,
        bootstrap_joint_q=bootstrap_joint_q,
        controller_mode=controller_mode,
        control_hz=control_hz,
    )

    collectors = replay_batched_sysid_structure(
        dataset=dataset,
        structure_idx=structure_idx,
        candidates=candidates,
        num_directions=1,
        seed=seed,
        build_env_fn=make_real_replay_build_env_fn(**build_kwargs),
        replay_sim_config=real_replay_sim_config(**sim_kwargs),
        on_step=make_replay_on_step(
            viewer,
            max_frames=max_frames,
            print_woody_forces_every=int(args.print_woody_forces),
            camera_to_base_4x4=episode_meta.get("camera_to_base_4x4"),
            recorder=recorder,
        ),
        use_oracle_params=True,
        action_dim=19 if controller_mode == "vic_pose" else 6,
    )
    tcp = np.asarray(collectors.to_arrays(0)["tcp_pos"], dtype=np.float64)
    motion_m = float(np.linalg.norm(tcp[-1] - tcp[0])) if tcp.shape[0] >= 2 else 0.0
    print(
        f"replay frames={tcp.shape[0]} tcp_motion_m={motion_m:.6g}",
        file=sys.stderr,
    )
    if recorder is not None:
        if recorder.frame_count <= 0:
            print(
                f"FAIL: --record-video requested but wrote 0 frames ({recorder.path})",
                file=sys.stderr,
            )
            return 1
        print(
            f"recorded video frames={recorder.frame_count} path={recorder.path}",
            file=sys.stderr,
        )
    if motion_m <= 1e-4:
        print("FAIL: TCP stationary (expected open-loop motion)", file=sys.stderr)
        return 1
    print("OK: TCP moved under recorded actions", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null"])

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)
    recorder: GlVideoRecorder | None = None
    if args.record_video is not None:
        require_gl_frame_capture(viewer)
        recorder = GlVideoRecorder(args.record_video)
    try:
        return _run(args, viewer, recorder=recorder)
    finally:
        if recorder is not None:
            recorder.close()
        if hasattr(viewer, "close"):
            viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
