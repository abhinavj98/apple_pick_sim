"""Observation-only initialization helpers for sysID Parquet datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from apple_pick_sim.coupled_fruiting.scene import init_robot_mujoco_step_buffers
from apple_pick_sim.digital_twin import DigitalTwinObs, infer_params_from_obs
from apple_pick_sim.fruiting_system import fruiting_params_from_json
from apple_pick_sim.fruiting_system.params import load_ranges, parse_fixture_args
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.system_id.trajectory_store import (
    TrajectoryDataset,
    stack_woody_pos_frame,
    target_tf_from_array,
)


def _tuple_or_none(value: Any, size: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != size:
        raise ValueError(f"expected length {size}, got {arr.size}")
    return tuple(float(x) for x in arr)


def _array_or_none(value: Any, size: int) -> np.ndarray | None:
    values = _tuple_or_none(value, size)
    if values is None:
        return None
    return np.asarray(values, dtype=np.float32)


def _first_frame_array_or_none(arrays: dict[str, Any], key: str, size: int) -> np.ndarray | None:
    value = arrays.get(key)
    if value is None or np.asarray(value).size < size:
        return None
    return np.asarray(value[0], dtype=np.float32).reshape(size)


def _rod_radii_from_meta(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("rod_radii metadata must be a JSON object or dict")
    return {str(name): float(radius) for name, radius in value.items()}


def _resolve_base_ranges_path(meta: dict[str, Any], base_ranges_path: str | Path | None) -> Path | None:
    if base_ranges_path is not None:
        return Path(base_ranges_path)
    fixture_path = meta.get("fixture_path")
    if not fixture_path:
        return None
    path = Path(str(fixture_path))
    return path if path.exists() else None


def _default_fruiting_base_pos(base_ranges_path: Path | None) -> tuple[float, float, float] | None:
    if base_ranges_path is None:
        return None
    ranges = load_ranges(base_ranges_path)
    base = parse_fixture_args(ranges).fruiting_base_pos
    if base is None:
        return None
    return tuple(float(x) for x in base)


def digital_twin_obs_from_episode(
    dataset: TrajectoryDataset,
    episode_id: str,
    *,
    base_ranges_path: str | Path | None = None,
) -> DigitalTwinObs:
    """Build a digital-twin observation bundle from metadata and frame 0."""
    meta = dataset.load_episode_meta(episode_id)
    arrays = dataset.load_episode_obs_arrays(episode_id)
    junction_names = list(arrays["junction_names"])
    if not junction_names:
        raise ValueError(f"episode {episode_id!r} has no junction names")

    resolved_ranges = _resolve_base_ranges_path(meta, base_ranges_path)
    fruiting_base_pos = _tuple_or_none(meta.get("fruiting_base_pos"), 3)
    if fruiting_base_pos is None:
        fruiting_base_pos = _default_fruiting_base_pos(resolved_ranges)
    if fruiting_base_pos is None:
        raise ValueError("fruiting_base_pos is required in metadata or fixture ranges")

    weld_direction = _tuple_or_none(meta.get("weld_direction"), 3)
    if weld_direction is None:
        raise ValueError("weld_direction is required in metadata")

    woody_start = stack_woody_pos_frame(
        arrays["woody_part_start_pos"], 0, junction_names
    )
    woody_end = stack_woody_pos_frame(
        arrays["woody_part_end_pos"], 0, junction_names
    )

    apple_radius = meta.get("apple_radius")
    return DigitalTwinObs(
        fruiting_base_pos=fruiting_base_pos,
        weld_direction=weld_direction,
        junction_names=junction_names,
        woody_part_start_pos=woody_start,
        woody_part_end_pos=woody_end,
        apple_radius=None if apple_radius is None else float(apple_radius),
        rod_radii=_rod_radii_from_meta(meta.get("rod_radii")),
    )


def observation_reset_options_from_parquet(
    dataset: TrajectoryDataset,
    episode_id: str,
    *,
    base_ranges_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return reset options derived only from observable Parquet metadata."""
    meta = dataset.load_episode_meta(episode_id)
    options: dict[str, Any] = {}

    for key, size in (
        ("fruiting_base_pos", 3),
        ("weld_direction", 3),
        ("weld_reference_pos", 3),
        ("weld_reference_quat", 4),
    ):
        value = _tuple_or_none(meta.get(key), size)
        if value is not None:
            options[key] = value

    resolved_ranges = _resolve_base_ranges_path(meta, base_ranges_path)
    if "fruiting_base_pos" not in options:
        default_base = _default_fruiting_base_pos(resolved_ranges)
        if default_base is not None:
            options["fruiting_base_pos"] = default_base

    serialized_params = meta.get("fruiting_system_params")
    if serialized_params is not None:
        options["params"] = fruiting_params_from_json(str(serialized_params))

    if resolved_ranges is not None:
        options["ranges_path"] = resolved_ranges
        if "params" not in options:
            obs = digital_twin_obs_from_episode(
                dataset,
                episode_id,
                base_ranges_path=resolved_ranges,
            )
            try:
                inferred_params = infer_params_from_obs(obs, resolved_ranges)
                if inferred_params.apple_radius is not None:
                    options["params"] = inferred_params
            except ValueError:
                # Synthetic and legacy datasets may have non-topological junction labels.
                # They can still reset from recorded weld metadata and replay actions.
                pass

    return options


def initialize_env_from_parquet(
    env: Any,
    dataset: TrajectoryDataset,
    episode_id: str,
    *,
    base_ranges_path: str | Path | None = None,
) -> None:
    """Apply reset observable state after a standard env reset."""
    del base_ranges_path
    meta = dataset.load_episode_meta(episode_id)
    arrays = dataset.load_episode_obs_arrays(episode_id)
    q = _array_or_none(meta.get("initial_robot_joint_q"), 7)
    if q is None:
        q = _first_frame_array_or_none(arrays, "robot_joint_q", 7)
    if q is None:
        return

    scene = getattr(env, "_scene", None)
    if scene is None or q.size != int(scene.robot_state_0.joint_q.numpy().reshape(-1).size):
        return

    import newton

    zeros_qd = np.zeros_like(scene.robot_state_0.joint_qd.numpy())
    scene.robot_state_0.joint_q.assign(q)
    scene.robot_state_0.joint_qd.assign(zeros_qd)
    scene.robot_model.joint_q.assign(q)
    scene.robot_model.joint_qd.assign(zeros_qd)
    newton.eval_fk(
        scene.robot_model,
        scene.robot_state_0.joint_q,
        scene.robot_state_0.joint_qd,
        scene.robot_state_0,
    )
    init_robot_mujoco_step_buffers(scene)
    fr3_robot.hold_mujoco_actuator_targets_at_state(
        scene.robot_model, scene.robot_state_0, scene.robot_control
    )
    if getattr(scene, "vic_jt_default_dof_pos", None) is not None:
        default_q = q.copy()
        if default_q.shape[0] > 6:
            default_q[6] = 0.0
        scene.vic_jt_default_dof_pos.assign(default_q)

    controller = getattr(env, "_controller", None)
    if controller is not None:
        controller.sync_target_from_state(scene.robot_state_0, int(scene.tcp_body_index))
        tcp_pos = _array_or_none(meta.get("initial_tcp_pos"), 3)
        tcp_quat = _array_or_none(meta.get("initial_tcp_quat"), 4)
        if tcp_pos is None:
            tcp_pos = _first_frame_array_or_none(arrays, "tcp_pos", 3)
        if tcp_quat is None:
            tcp_quat = _first_frame_array_or_none(arrays, "tcp_quat", 4)
        if tcp_pos is not None and tcp_quat is not None:
            recorded_target = target_tf_from_array(
                np.concatenate(
                    [
                        tcp_pos,
                        tcp_quat,
                    ]
                )
            )
            controller.target_tf = recorded_target
        scene.vic_target_tf = controller.target_tf
    scene.vic_target_twist = fr3_robot.EEVelocity()
