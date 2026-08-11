#!/usr/bin/env python3
"""Real FR3+VIC replay test for an exported real→batched dataset.

Uses ``replay_batched_sysid_structure`` (same path as CMA / MMD grid) so FR3
tracks recorded **EE twist** actions after free settle → weld → post-grasp settle.

**Action semantics caveat:** many real parquets store pose-control **wrench**
``[Fx…Tz]`` in ``action`` (see ``dump.action_semantics``), not twists. Export
refuses those by default; ``--allow-wrench-as-twist`` only unlocks format/GL
smoke with **incorrect** physics. Correct drive needs pose/wrench mode
(``docs/superpowers/specs/2026-08-10-vic-pose-action-controller-design.md``).

Settle defaults match ``example_view_pre_grasp_settle.py``:
``--settle-substeps 5000``, ``--settle-quiet-every 300``,
``--post-grasp-settle-substeps 500``.

With ``--viewer gl``, renders trajectory frames after off-screen rebuild/settle
(same minimal ``on_step`` pattern as ``example_batched_sysid_mmd_grid.py``).

Geometry comes from converted episode metadata (same native rebuild as
``example_view_pre_grasp_settle`` / ``example_view_batched_episode_meta``):
oracle ``fruiting_system_params`` and episode ``fruiting_base_pos``. Arm
placement is **open-loop** from ``initial_robot_joint_q`` (skip IK). Sim physics
uses ``gym_defaults`` + fixture ``sim_build`` on the default sim device.

Example (after export)::

    uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \\
      --input robot_replay/s02-d00_action.parquet \\
      --dataset-out /tmp/real_batched_s02_d00 --overwrite

    uv run python robot_replay/example_replay_real_batched.py \\
      --dataset /tmp/real_batched_s02_d00 --viewer gl --max-frames 0 \\
      --settle-substeps 5000 --settle-quiet-every 300 \\
      --post-grasp-settle-substeps 500
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import newton.examples

from apple_pick_gym.batched_envs import ApplePickBatchedSysIdEnv
from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
    gt_bend_stiffness_candidate_from_structure,
    replay_batched_sysid_structure,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ObsConfig,
)
from apple_pick_sim.fruiting_system.params import (
    GripperProxyConfig,
    load_ranges,
    parse_sim_build,
)
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import ImpedanceGains
from apple_pick_sim.system_id import BatchedSysIdDataset

_DEFAULT_FIXTURE = Path(
    "apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json"
)
# Match example_view_pre_grasp_settle.py defaults.
_SETTLE_SUBSTEPS = 5000
_SETTLE_QUIET_EVERY: int | None = 300
_SETTLE_GRAVITY_RAMP = False
_POST_GRASP_SETTLE_SUBSTEPS = 500
_CONTROL_HZ_FALLBACK = 15.0
_DEFAULT_CONTROLLER_MODE = "vic"


def fruiting_base_pos_from_episode_metadata(
    meta: Mapping[str, Any],
) -> tuple[float, float, float]:
    """T-junction base from converted episode metadata (native rebuild)."""
    raw = meta.get("fruiting_base_pos")
    if raw is None:
        raise ValueError("episode metadata missing fruiting_base_pos")
    arr = np.asarray(raw, dtype=np.float64).reshape(3)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def bootstrap_joint_q_from_episode_metadata(
    meta: Mapping[str, Any],
) -> tuple[float, ...]:
    """Recorded grasp arm joints for open-loop FR3 placement (skip IK)."""
    raw = meta.get("initial_robot_joint_q")
    if raw is None:
        raise ValueError("episode metadata missing initial_robot_joint_q")
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size < 1:
        raise ValueError("initial_robot_joint_q must be non-empty")
    return tuple(float(x) for x in arr.tolist())


def _sim_build_knobs(ranges: dict) -> tuple[
    ImpedanceGains | None,
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    float | None,
]:
    """Fixture ``sim_build`` joint/VIC knobs (same contract as MMD-grid collect)."""
    sb = parse_sim_build(ranges)
    if sb is None:
        return None, {}, {}, {}, {}, None
    return (
        ImpedanceGains(
            linear_k=sb.vic_gains.linear_k,
            linear_d=sb.vic_gains.linear_d,
            angular_k=sb.vic_gains.angular_k,
            angular_d=sb.vic_gains.angular_d,
        ),
        dict(sb.joint_angular_kd_overrides),
        dict(sb.joint_linear_kd_overrides),
        dict(sb.joint_angular_kp_overrides),
        dict(sb.joint_linear_kp_overrides),
        sb.joint_damping_ratio,
    )


def _test_sim_config(
    *,
    num_envs: int,
    topology_seed: int,
    fruiting_base_pos: tuple[float, float, float],
    ranges: dict,
    settle_substeps: int = _SETTLE_SUBSTEPS,
    settle_quiet_every: int | None = _SETTLE_QUIET_EVERY,
    settle_gravity_ramp: bool = _SETTLE_GRAVITY_RAMP,
    post_grasp_settle_substeps: int = _POST_GRASP_SETTLE_SUBSTEPS,
    bootstrap_joint_q: tuple[float, ...] | None = None,
    controller_mode: str = _DEFAULT_CONTROLLER_MODE,
) -> BatchedHeterogeneousCoupledSimConfig:
    """Gym FR3+VIC config with fixture sim_build; episode fruiting_base_pos."""
    gym_cfg = BatchedHeterogeneousCoupledSimConfig.gym_defaults(num_envs=num_envs)
    (
        vic_gains,
        joint_angular_kd,
        joint_linear_kd,
        joint_angular_kp,
        joint_linear_kp,
        joint_damping_ratio,
    ) = _sim_build_knobs(ranges)
    controller = dataclasses.replace(
        gym_cfg.controller,
        mode=controller_mode,
        action_dim=19 if controller_mode == "vic_pose" else gym_cfg.controller.action_dim,
        linear_speed=1.0,
        angular_speed=1.0,
    )
    if vic_gains is not None:
        controller = dataclasses.replace(controller, vic_gains=vic_gains)
    return dataclasses.replace(
        gym_cfg,
        robot=dataclasses.replace(
            gym_cfg.robot,
            kind="fr3",
            step_mode="coupled",
            fix_to_apple=True,
            skip_ik_bootstrap=True,
            defer_template_robot_bootstrap=True,
            force_batched_layout=True,
            robot_base_pos=(0.0, 0.0, 0.0),
            per_env_ik=False,
            bootstrap_joint_q=bootstrap_joint_q,
        ),
        scene=dataclasses.replace(
            gym_cfg.scene,
            settle_substeps=int(settle_substeps),
            settle_quiet_every=settle_quiet_every,
            settle_gravity_ramp=bool(settle_gravity_ramp),
            post_grasp_settle_substeps=int(post_grasp_settle_substeps),
            fruiting_base_pos=fruiting_base_pos,
        ),
        controller=controller,
        fruiting_system=dataclasses.replace(
            gym_cfg.fruiting_system,
            joint_angular_kd_overrides=joint_angular_kd,
            joint_linear_kd_overrides=joint_linear_kd,
            joint_angular_kp_overrides=joint_angular_kp,
            joint_linear_kp_overrides=joint_linear_kp,
            joint_damping_ratio=joint_damping_ratio,
        ),
        domain_randomization=dataclasses.replace(
            gym_cfg.domain_randomization,
            topology_seed=int(topology_seed),
        ),
        obs=ObsConfig(allocate_buffers=True),
    )


def _build_env_fn(
    *,
    ranges_path: Path,
    ranges: dict,
    topology_seed: int,
    fruiting_base_pos: tuple[float, float, float],
    settle_substeps: int = _SETTLE_SUBSTEPS,
    settle_quiet_every: int | None = _SETTLE_QUIET_EVERY,
    settle_gravity_ramp: bool = _SETTLE_GRAVITY_RAMP,
    post_grasp_settle_substeps: int = _POST_GRASP_SETTLE_SUBSTEPS,
    bootstrap_joint_q: tuple[float, ...] | None = None,
    controller_mode: str = _DEFAULT_CONTROLLER_MODE,
) -> Callable[..., Any]:
    def build_env_fn(
        *,
        num_envs: int,
        per_env_params: list[Any],
        max_episode_steps: int,
        gripper: GripperProxyConfig | None = None,
        per_env_grippers: list[GripperProxyConfig] | None = None,
    ) -> ApplePickBatchedSysIdEnv:
        if gripper is not None and per_env_grippers is not None:
            raise ValueError(
                "scalar gripper and per_env_grippers cannot both be provided"
            )
        sim_config = _test_sim_config(
            num_envs=num_envs,
            topology_seed=topology_seed,
            fruiting_base_pos=fruiting_base_pos,
            ranges=ranges,
            settle_substeps=settle_substeps,
            settle_quiet_every=settle_quiet_every,
            settle_gravity_ramp=settle_gravity_ramp,
            post_grasp_settle_substeps=post_grasp_settle_substeps,
            bootstrap_joint_q=bootstrap_joint_q,
            controller_mode=controller_mode,
        )
        if gripper is not None:
            sim_config = dataclasses.replace(
                sim_config,
                robot=dataclasses.replace(sim_config.robot, gripper=gripper),
            )
        return ApplePickBatchedSysIdEnv(
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            ranges_path=ranges_path,
            topology_seed=topology_seed,
            use_settle_cache=False,
            sim_config=sim_config,
            per_env_params=per_env_params,
            per_env_grippers=per_env_grippers,
        )

    return build_env_fn


def make_replay_on_step(
    viewer: object,
    *,
    max_frames: int,
    control_hz_fallback: float = _CONTROL_HZ_FALLBACK,
) -> Callable[..., bool]:
    """MMD-grid-style render + optional frame cap for real replay."""
    viewer_state: dict[str, object] = {"initialized": False}

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
    return p


def check_action_semantics(
    *,
    controller_mode: str,
    collection: dict,
    episode_meta: dict,
    allow_wrench_as_twist: bool,
) -> None:
    """Raise ``SystemExit`` when ``action`` semantics do not match ``controller_mode``."""
    pose_packed = (
        collection.get("action_layout") == "vic_pose_v1"
        or episode_meta.get("action_layout") == "vic_pose_v1"
        or int(collection.get("action_dim") or 0) == 19
        or int(episode_meta.get("action_dim") or 0) == 19
    )
    if controller_mode == "vic" and allow_wrench_as_twist and pose_packed:
        raise SystemExit(
            "--allow-wrench-as-twist only applies to legacy 6D wrench-as-twist exports; "
            "this dataset already carries 19D vic_pose actions "
            "(action_layout=vic_pose_v1). Use --controller-mode vic_pose instead."
        )

    wrench_marked = (
        episode_meta.get("action_compatible_with_vic_twist") is False
        or collection.get("action_compatible_with_vic_twist") is False
    )
    if controller_mode == "vic" and wrench_marked and not allow_wrench_as_twist:
        raise SystemExit(
            "dataset action is a real pose-control wrench, not an EE twist for "
            "mode=vic. Refuse incorrect physics. Use --controller-mode vic_pose "
            "for 19D pose actions, or pass --allow-wrench-as-twist for format/GL "
            "smoke only."
        )


def _run(args: argparse.Namespace, viewer: object) -> int:
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

    collectors = replay_batched_sysid_structure(
        dataset=dataset,
        structure_idx=structure_idx,
        candidates=candidates,
        num_directions=1,
        seed=seed,
        build_env_fn=_build_env_fn(
            ranges_path=ranges_path,
            ranges=ranges,
            topology_seed=seed,
            fruiting_base_pos=fruiting_base_pos,
            settle_substeps=settle_substeps,
            settle_quiet_every=settle_quiet_every,
            settle_gravity_ramp=settle_gravity_ramp,
            post_grasp_settle_substeps=post_grasp_settle_substeps,
            bootstrap_joint_q=bootstrap_joint_q,
            controller_mode=controller_mode,
        ),
        replay_sim_config=_test_sim_config(
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
        ),
        on_step=make_replay_on_step(viewer, max_frames=max_frames),
        use_oracle_params=True,
        action_dim=19 if controller_mode == "vic_pose" else 6,
    )
    tcp = np.asarray(collectors.to_arrays(0)["tcp_pos"], dtype=np.float64)
    motion_m = float(np.linalg.norm(tcp[-1] - tcp[0])) if tcp.shape[0] >= 2 else 0.0
    print(
        f"replay frames={tcp.shape[0]} tcp_motion_m={motion_m:.6g}",
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
    try:
        return _run(args, viewer)
    finally:
        if hasattr(viewer, "close"):
            viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
