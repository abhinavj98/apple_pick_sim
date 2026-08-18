"""Shared real-replay env build helpers for vic_pose datasets."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from apple_pick_gym.batched_envs import ApplePickBatchedSysIdEnv
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ObsConfig,
)
from apple_pick_sim.fruiting_system.params import (
    GripperProxyConfig,
    parse_sim_build,
)
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import ImpedanceGains
from apple_pick_sim.system_id.batched_digital_twin_init import (
    apply_logged_post_grasp_se3_to_cable,
    gripper_proxy_for_real_batched_replay,
)

# Match example_view_pre_grasp_settle.py / example_replay_real_batched defaults.
_SETTLE_SUBSTEPS = 5000
_SETTLE_QUIET_EVERY: int | None = 300
_SETTLE_GRAVITY_RAMP = False
_POST_GRASP_SETTLE_SUBSTEPS = 500
_DEFAULT_CONTROLLER_MODE = "vic_pose"


def dataset_declares_vic_pose(
    collection: Mapping[str, Any],
    episode_meta: Mapping[str, Any] | None = None,
) -> bool:
    """True when collection or episode metadata carries 19D vic_pose actions."""
    if collection.get("action_layout") == "vic_pose_v1":
        return True
    if episode_meta is not None and episode_meta.get("action_layout") == "vic_pose_v1":
        return True
    if int(collection.get("action_dim") or 0) == 19:
        return True
    if episode_meta is not None and int(episode_meta.get("action_dim") or 0) == 19:
        return True
    return False


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


def control_hz_from_episode_metadata(
    meta: Mapping[str, Any],
    *,
    collection: Mapping[str, Any] | None = None,
) -> float:
    """Recorded control rate [Hz] from episode meta, else collection."""
    raw = meta.get("control_hz")
    if raw is None and collection is not None:
        raw = collection.get("control_hz")
    if raw is None:
        raise ValueError("episode metadata missing control_hz")
    hz = float(raw)
    if hz <= 0.0:
        raise ValueError(f"control_hz must be positive, got {hz}")
    return hz


def check_action_semantics(
    *,
    controller_mode: str,
    collection: Mapping[str, Any],
    episode_meta: Mapping[str, Any],
    allow_wrench_as_twist: bool,
) -> None:
    """Raise ``SystemExit`` when ``action`` semantics do not match ``controller_mode``."""
    pose_packed = dataset_declares_vic_pose(collection, episode_meta)
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


def real_replay_sim_config(
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
    control_hz: float | None = None,
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
    runtime = gym_cfg.runtime
    if control_hz is not None:
        runtime = dataclasses.replace(runtime, control_hz=float(control_hz))
    return dataclasses.replace(
        gym_cfg,
        runtime=runtime,
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


def make_real_replay_build_env_fn(
    *,
    ranges_path: Path,
    ranges: dict,
    topology_seed: int,
    fruiting_base_pos: tuple[float, float, float],
    episode_meta: Mapping[str, Any],
    settle_substeps: int = _SETTLE_SUBSTEPS,
    settle_quiet_every: int | None = _SETTLE_QUIET_EVERY,
    settle_gravity_ramp: bool = _SETTLE_GRAVITY_RAMP,
    post_grasp_settle_substeps: int = _POST_GRASP_SETTLE_SUBSTEPS,
    bootstrap_joint_q: tuple[float, ...] | None = None,
    controller_mode: str = _DEFAULT_CONTROLLER_MODE,
    control_hz: float | None = None,
) -> Callable[..., ApplePickBatchedSysIdEnv]:
    def build_env_fn(
        *,
        num_envs: int,
        per_env_params: list[Any],
        max_episode_steps: int,
        gripper: GripperProxyConfig | None = None,
        per_env_grippers: list[GripperProxyConfig] | None = None,
        per_env_episode_meta: Sequence[Mapping[str, Any]] | None = None,
    ) -> ApplePickBatchedSysIdEnv:
        if gripper is not None and per_env_grippers is not None:
            raise ValueError("scalar gripper and per_env_grippers cannot both be provided")
        if per_env_grippers is not None:
            grippers = list(per_env_grippers)
        elif gripper is not None:
            grippers = [gripper] * int(num_envs)
        else:
            real_g = gripper_proxy_for_real_batched_replay(dict(episode_meta))
            grippers = [real_g] * int(num_envs)

        sim_config = real_replay_sim_config(
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
            control_hz=control_hz,
        )
        robot_updates: dict[str, Any] = {"gripper": grippers[0]}
        if per_env_episode_meta is not None:
            robot_updates["per_world_bootstrap_joint_q"] = tuple(
                bootstrap_joint_q_from_episode_metadata(env_meta)
                for env_meta in per_env_episode_meta
            )
        sim_config = dataclasses.replace(
            sim_config,
            robot=dataclasses.replace(sim_config.robot, **robot_updates),
        )
        env = ApplePickBatchedSysIdEnv(
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            ranges_path=ranges_path,
            topology_seed=topology_seed,
            use_settle_cache=False,
            sim_config=sim_config,
            per_env_params=per_env_params,
            per_env_grippers=grippers,
            control_hz=None if control_hz is None else float(control_hz),
        )
        scene = env._sim.scene
        layout = getattr(scene, "layout", None)
        cable = getattr(scene, "cable", None)
        if cable is not None:
            if per_env_episode_meta is not None:
                apply_logged_post_grasp_se3_to_cable(
                    cable,
                    dict(episode_meta),
                    layout=layout,
                    per_env_meta=per_env_episode_meta,
                )
            else:
                apply_logged_post_grasp_se3_to_cable(
                    cable, dict(episode_meta), layout=layout
                )
        return env

    build_env_fn.wants_per_env_meta = True
    return build_env_fn
