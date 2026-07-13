"""Digital-twin observation helpers for batched sys-ID datasets."""

from __future__ import annotations

from collections.abc import Sequence

import json
from pathlib import Path
from typing import Any

import numpy as np

from apple_pick_sim.coupled_fruiting.scene import init_robot_mujoco_step_buffers
from apple_pick_sim.digital_twin import DigitalTwinObs, infer_params_from_obs
from apple_pick_sim.fruiting_system import fruiting_params_from_json
from apple_pick_sim.fruiting_system.params import FruitingSystemParams, GripperProxyConfig, load_ranges, parse_fixture_args
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.system_id.batched_trajectory_store import (
    BatchedSysIdDataset,
    PRE_WELD_STEP_IDX,
    FIRST_TRAJECTORY_STEP_IDX,
    frame_index_for_step,
)
from apple_pick_sim.system_id.excitation_state import ExcitationContext
from apple_pick_sim.system_id.trajectory_store import stack_woody_pos_frame


def _tuple_or_none(value: Any, size: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != size:
        raise ValueError(f"expected length {size}, got {arr.size}")
    return tuple(float(x) for x in arr)


def _rod_radii_from_meta(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("rod_radii metadata must be a JSON object or dict")
    return {str(name): float(radius) for name, radius in value.items()}


def _resolve_fixture_path(meta: dict[str, Any]) -> Path | None:
    fixture_path = meta.get("fixture_path")
    if not fixture_path:
        return None
    path = Path(str(fixture_path))
    return path if path.exists() else None


def _default_fruiting_base_pos(fixture_path: Path | None) -> tuple[float, float, float] | None:
    if fixture_path is None:
        return None
    ranges = load_ranges(fixture_path)
    base = parse_fixture_args(ranges).fruiting_base_pos
    if base is None:
        return None
    return tuple(float(x) for x in base)


def _array_or_none(value: Any, size: int) -> np.ndarray | None:
    values = _tuple_or_none(value, size)
    if values is None:
        return None
    return np.asarray(values, dtype=np.float32)


def _first_frame_array_or_none(arrays: dict[str, Any], key: str, size: int) -> np.ndarray | None:
    value = arrays.get(key)
    if value is None or np.asarray(value).size < size:
        return None
    return np.asarray(value[0], dtype=np.float32).reshape(-1)[:size]


def _frame_array_at_step(
    arrays: dict[str, Any],
    key: str,
    step_idx: int,
    size: int,
) -> np.ndarray | None:
    value = arrays.get(key)
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size < size:
        return None
    row = frame_index_for_step(arrays, step_idx, fallback=0)
    return np.asarray(arr[row], dtype=np.float32).reshape(-1)[:size]


def _frame_array_or_meta(
    arrays: dict[str, Any],
    meta: dict[str, Any],
    *,
    array_key: str,
    meta_key: str,
    size: int,
    step_idx: int = FIRST_TRAJECTORY_STEP_IDX,
) -> np.ndarray | None:
    arr = _frame_array_at_step(arrays, array_key, step_idx, size)
    if arr is not None:
        return arr
    return _array_or_none(meta.get(meta_key), size)


def gripper_proxy_from_episode_metadata(
    meta: dict[str, Any],
    *,
    base: GripperProxyConfig | None = None,
) -> GripperProxyConfig:
    """Build replay gripper config from batched episode metadata (structure-0 direction-0)."""
    import dataclasses

    proxy = base or GripperProxyConfig()
    weld_direction = _tuple_or_none(meta.get("weld_direction"), 3)
    weld_reference_pos = _tuple_or_none(meta.get("weld_reference_pos"), 3)
    weld_reference_quat = _tuple_or_none(meta.get("weld_reference_quat"), 4)
    return dataclasses.replace(
        proxy,
        fix_to_apple=True,
        robot_facing_weld=weld_direction is None,
        weld_direction=weld_direction,
        weld_reference_pos=weld_reference_pos,
        weld_reference_quat=weld_reference_quat,
    )


def digital_twin_obs_from_batched_episode(
    dataset: BatchedSysIdDataset,
    structure_idx: int,
    direction_idx: int = 0,
    *,
    tree_step_idx: int = PRE_WELD_STEP_IDX,
) -> DigitalTwinObs:
    """Build a digital-twin observation bundle for geometry rebuild.

    Defaults to the post-settle ``pre_weld`` frame (``step_idx=-1``) when present.
    Falls back to the first post-weld pull frame otherwise.
    """
    meta = dataset.load_episode_metadata(structure_idx, direction_idx)
    arrays = dataset.load_episode_obs_arrays(structure_idx, direction_idx)
    junction_names = list(arrays["junction_names"])
    if not junction_names:
        raise ValueError(
            f"structure {structure_idx}, direction {direction_idx} has no junction names"
        )

    fixture_path = _resolve_fixture_path(meta)
    fruiting_base_pos = _tuple_or_none(meta.get("fruiting_base_pos"), 3)
    if fruiting_base_pos is None:
        fruiting_base_pos = _default_fruiting_base_pos(fixture_path)
    if fruiting_base_pos is None:
        raise ValueError("fruiting_base_pos is required in metadata or fixture ranges")

    weld_direction = _tuple_or_none(meta.get("weld_direction"), 3)
    if weld_direction is None:
        raise ValueError("weld_direction is required in metadata")

    frame_idx = frame_index_for_step(
        arrays,
        int(tree_step_idx),
        fallback=frame_index_for_step(arrays, FIRST_TRAJECTORY_STEP_IDX, fallback=0),
    )
    woody_start = stack_woody_pos_frame(
        arrays["woody_part_start_pos"], frame_idx, junction_names
    )
    woody_end = stack_woody_pos_frame(
        arrays["woody_part_end_pos"], frame_idx, junction_names
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


def infer_base_params_for_structure(
    dataset: BatchedSysIdDataset,
    structure_idx: int,
) -> FruitingSystemParams:
    """Infer base :class:`FruitingSystemParams` from pre-weld tree observations."""
    obs = digital_twin_obs_from_batched_episode(
        dataset,
        structure_idx,
        0,
        tree_step_idx=PRE_WELD_STEP_IDX,
    )
    meta = dataset.load_episode_metadata(structure_idx, 0)
    fixture_path = _resolve_fixture_path(meta)
    if fixture_path is None:
        raise ValueError("fixture_path is required in episode metadata")
    return infer_params_from_obs(obs, fixture_path)


def true_params_for_structure(
    dataset: BatchedSysIdDataset,
    structure_idx: int,
) -> FruitingSystemParams:
    """Load the exact :class:`FruitingSystemParams` used to build this structure."""
    meta = dataset.load_episode_metadata(structure_idx, 0)
    serialized = meta.get("fruiting_system_params")
    if not serialized:
        raise ValueError(
            f"structure {structure_idx} metadata has no fruiting_system_params "
            "(true params are only available for sim-to-sim datasets)"
        )
    return fruiting_params_from_json(str(serialized))


def initialize_batched_env_from_dataset(
    env: Any,
    dataset: BatchedSysIdDataset,
    *,
    structure_idx: int,
    num_directions: int,
    direction_indices: Sequence[int] | None = None,
) -> None:
    """Apply pre-weld joint and VIC target state from a batched sys-ID dataset.

    Uses episode metadata (``initial_robot_joint_q``, ``initial_tcp_pos``,
    ``initial_tcp_quat``) which captures the robot state right after ``env.reset()``
    but *before* any trajectory step.  This matches the cable-settle equilibrium and
    avoids the spurious force transient that would arise from initialising with
    ``step_idx=0`` data (which is recorded *after* the first move step).
    """
    import newton

    scene = env._sim.scene
    layout = env._sim.layout
    if layout is None:
        raise RuntimeError("batched scene missing layout")

    vic = scene.vic_controller
    target_pos = None
    target_rot = None
    if vic is not None:
        target_pos = vic._target_pos_wp.numpy().copy()
        target_rot = vic._target_rot_wp.numpy().copy()

    dirs = (
        [int(d) for d in direction_indices]
        if direction_indices is not None
        else list(range(int(num_directions)))
    )
    if not dirs:
        raise ValueError("direction_indices must be non-empty")
    if len(dirs) != int(num_directions):
        raise ValueError(
            f"len(direction_indices)={len(dirs)} != num_directions={int(num_directions)}"
        )

    for env_idx in range(env.num_envs):
        direction_idx = int(dirs[int(env_idx) % len(dirs)])
        arrays = dataset.load_episode_obs_arrays(structure_idx, direction_idx)
        meta = dataset.load_episode_metadata(structure_idx, direction_idx)
        world = int(env_idx)

        # Prefer metadata initial values (captured right after env.reset(), before any
        # trajectory step) so the initialized EE position matches the cable-settle
        # equilibrium.  Fall back to the first trajectory frame only when metadata keys
        # are absent (legacy datasets).
        q = _array_or_none(meta.get("initial_robot_joint_q"), 7)
        if q is None:
            q = _frame_array_at_step(arrays, "robot_joint_q", FIRST_TRAJECTORY_STEP_IDX, 7)
        if q is None:
            raise ValueError(
                "missing initial robot_joint_q for "
                f"structure={structure_idx} direction={direction_idx} "
                f"(env_idx={env_idx})"
            )

        q_slice = layout.joint_q_slice(world)
        qd_slice = layout.joint_qd_slice(world)
        q_flat = q.reshape(-1)
        zeros_qd = np.zeros(qd_slice.stop - qd_slice.start, dtype=np.float32)

        for target in (scene.robot_state_0, scene.robot_model):
            jq = target.joint_q.numpy().copy()
            jqd = target.joint_qd.numpy().copy()
            jq[q_slice] = q_flat
            jqd[qd_slice] = zeros_qd
            target.joint_q.assign(jq)
            target.joint_qd.assign(jqd)

        vic_default = getattr(scene, "vic_jt_default_dof_pos", None)
        if vic_default is not None:
            default_q = q_flat.copy()
            if default_q.shape[0] > 6:
                default_q[6] = 0.0
            vic_default.assign(default_q)

        vic_default_batched = getattr(scene, "vic_jt_default_dof_pos_batched", None)
        if vic_default_batched is not None:
            default_rows = vic_default_batched.numpy().copy()
            row = q_flat.copy()
            if row.shape[0] > 6:
                row[6] = 0.0
            default_rows[world, : row.shape[0]] = row
            vic_default_batched.assign(default_rows)

        if target_pos is not None and target_rot is not None:
            tcp_pos = _array_or_none(meta.get("initial_tcp_pos"), 3)
            if tcp_pos is None:
                tcp_pos = _frame_array_at_step(arrays, "tcp_pos", FIRST_TRAJECTORY_STEP_IDX, 3)
            tcp_quat = _array_or_none(meta.get("initial_tcp_quat"), 4)
            if tcp_quat is None:
                tcp_quat = _frame_array_at_step(arrays, "tcp_quat", FIRST_TRAJECTORY_STEP_IDX, 4)
            if tcp_pos is not None and tcp_quat is not None:
                target_pos[world] = tcp_pos
                target_rot[world] = tcp_quat

        excitation_direction = arrays["excitation_direction"][0]
        env.set_excitation_context(
            env_idx,
            ExcitationContext(
                type="quasi_static",
                f_inst=0.0,
                direction=excitation_direction,
            ),
        )

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

    if vic is not None and target_pos is not None and target_rot is not None:
        vic._target_pos_wp.assign(target_pos.astype(np.float32))
        vic._target_rot_wp.assign(target_rot.astype(np.float32))
        vic._sync_target_tf_from_device()
        vic.stage_targets_to_scene(scene)
    scene.vic_target_twist = fr3_robot.EEVelocity()
