"""Digital-twin observation helpers for batched sys-ID datasets."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.coupled_fruiting.proxy_coupling import (
    align_proxy_body_q_prev_for_vbd,
    sync_model_body_q_rest_from_state,
)
from apple_pick_sim.coupled_fruiting.scene import init_robot_mujoco_step_buffers
from apple_pick_sim.coupled_fruiting.settle_then_weld import _proxy_world_pose_from_apple
from apple_pick_sim.digital_twin import DigitalTwinObs, infer_params_from_obs
from apple_pick_sim.fruiting_system import fruiting_params_from_json
from apple_pick_sim.fruiting_system.params import (
    FruitingSystemParams,
    GripperProxyConfig,
    fruiting_params_from_dict,
    load_ranges,
    parse_fixture_args,
)
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.system_id.batched_trajectory_store import (
    BatchedSysIdDataset,
    PRE_WELD_STEP_IDX,
    FIRST_TRAJECTORY_STEP_IDX,
    frame_index_for_step,
)
from apple_pick_sim.system_id.excitation_state import ExcitationContext
from apple_pick_sim.system_id.real_post_grasp_plan import proxy_offset_from_apple_and_tcp
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


def gripper_proxy_for_real_batched_replay(
    meta: dict[str, Any],
    *,
    base: GripperProxyConfig | None = None,
) -> GripperProxyConfig:
    """Gripper for real batched replay: logged apple reference + true TCP offset.

    Extends :func:`gripper_proxy_from_episode_metadata` with
    ``weld_proxy_offset_in_apple_frame = X_apple^{-1} X_tcp`` from episode
    ``initial_apple_*`` / ``initial_tcp_*`` (same math as settle-viewer post-grasp).
    """
    import dataclasses

    proxy = gripper_proxy_from_episode_metadata(meta, base=base)
    apple_pos = _tuple_or_none(meta.get("initial_apple_pos"), 3)
    if apple_pos is None:
        apple_pos = _tuple_or_none(meta.get("weld_reference_pos"), 3)
    apple_quat = _tuple_or_none(meta.get("initial_apple_quat"), 4)
    if apple_quat is None:
        apple_quat = _tuple_or_none(meta.get("weld_reference_quat"), 4)
    tcp_pos = _tuple_or_none(meta.get("initial_tcp_pos"), 3)
    tcp_quat = _tuple_or_none(meta.get("initial_tcp_quat"), 4)
    if tcp_pos is None or tcp_quat is None:
        raise ValueError(
            "real batched replay gripper requires initial_tcp_pos and initial_tcp_quat "
            "in episode metadata"
        )
    if apple_pos is None or apple_quat is None:
        raise ValueError(
            "real batched replay gripper requires initial_apple_pos/quat "
            "(or weld_reference_*) in episode metadata"
        )
    offset = proxy_offset_from_apple_and_tcp(
        apple_pos=apple_pos,  # type: ignore[arg-type]
        apple_quat_xyzw=apple_quat,  # type: ignore[arg-type]
        tcp_pos=tcp_pos,  # type: ignore[arg-type]
        tcp_quat_xyzw=tcp_quat,  # type: ignore[arg-type]
    )
    return dataclasses.replace(
        proxy,
        weld_reference_pos=apple_pos,
        weld_reference_quat=apple_quat,
        weld_proxy_offset_in_apple_frame=offset,
    )


def apply_logged_post_grasp_se3_to_cable(
    cable: Any,
    meta: dict[str, Any],
    *,
    layout: BatchedEnvLayout | None = None,
) -> None:
    """Write logged post-grasp apple SE(3) and realign proxy; sync VBD rest.

    Call after settle→weld seed so free settle still used pre-grasp
    ``params.apple_quat_xyzw``, matching ``example_view_pre_grasp_settle``.
    """
    apple_pos = _tuple_or_none(meta.get("initial_apple_pos"), 3)
    apple_quat = _tuple_or_none(meta.get("initial_apple_quat"), 4)
    if apple_pos is None:
        apple_pos = _tuple_or_none(meta.get("weld_reference_pos"), 3)
    if apple_quat is None:
        apple_quat = _tuple_or_none(meta.get("weld_reference_quat"), 4)
    if apple_pos is None or apple_quat is None:
        raise ValueError(
            "apply_logged_post_grasp_se3_to_cable requires initial_apple_pos/quat "
            "(or weld_reference_*) in episode metadata"
        )
    apple_id = getattr(cable, "apple_body", None)
    proxy_id = getattr(cable, "gripper_proxy_body", None)
    if apple_id is None or proxy_id is None:
        raise ValueError("cable missing apple_body or gripper_proxy_body")
    offset = getattr(cable, "gripper_proxy_offset_in_apple_frame", None)
    if offset is None:
        tcp_pos = _tuple_or_none(meta.get("initial_tcp_pos"), 3)
        tcp_quat = _tuple_or_none(meta.get("initial_tcp_quat"), 4)
        if tcp_pos is None or tcp_quat is None:
            raise ValueError(
                "cable has no gripper_proxy_offset_in_apple_frame and meta lacks "
                "initial_tcp_pos/quat"
            )
        offset = proxy_offset_from_apple_and_tcp(
            apple_pos=apple_pos,  # type: ignore[arg-type]
            apple_quat_xyzw=apple_quat,  # type: ignore[arg-type]
            tcp_pos=tcp_pos,  # type: ignore[arg-type]
            tcp_quat_xyzw=tcp_quat,  # type: ignore[arg-type]
        )

    bq = cable.state_0.body_q.numpy().reshape(-1, 7).astype(np.float32).copy()
    bqd = cable.state_0.body_qd.numpy().reshape(-1, 6).astype(np.float32).copy()
    if layout is not None and int(layout.num_envs) > 1:
        pairs = list(zip(layout.apple_body_indices, layout.proxy_body_indices, strict=True))
    else:
        pairs = [(int(apple_id), int(proxy_id))]
    for aid, pid in pairs:
        if int(aid) < 0 or int(pid) < 0:
            continue
        bq[int(aid), 0:3] = np.asarray(apple_pos, dtype=np.float32)
        bq[int(aid), 3:7] = np.asarray(apple_quat, dtype=np.float32)
        proxy_pos, proxy_quat = _proxy_world_pose_from_apple(bq[int(aid)], offset)
        bq[int(pid), 0:3] = proxy_pos
        bq[int(pid), 3:7] = proxy_quat
        bqd[int(aid)] = 0.0
        bqd[int(pid)] = 0.0
    cable.state_0.body_q.assign(bq)
    cable.state_0.body_qd.assign(bqd)
    cable.state_1.body_q.assign(bq)
    cable.state_1.body_qd.assign(bqd)
    body_count = int(getattr(getattr(cable, "model", None), "body_count", bq.shape[0]))
    align_proxy_body_q_prev_for_vbd(cable, tuple(range(body_count)))
    sync_model_body_q_rest_from_state(cable)


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
    # Sim collect stores a JSON string; real export may nest a dict in schema JSON.
    if isinstance(serialized, dict):
        return fruiting_params_from_dict(serialized)
    return fruiting_params_from_json(str(serialized))


@dataclasses.dataclass(frozen=True)
class ReplayEpisodeSource:
    """Dataset episode used to initialize one replay environment."""

    structure_idx: int
    direction_idx: int


def initialize_batched_env_from_episode_sources(
    env: Any,
    dataset: BatchedSysIdDataset,
    sources: Sequence[ReplayEpisodeSource],
) -> None:
    """Apply each world's pre-weld state from its explicit dataset episode.

    Uses episode metadata (``initial_robot_joint_q``, ``initial_tcp_pos``,
    ``initial_tcp_quat``) which captures the robot state right after ``env.reset()``
    but *before* any trajectory step.  This matches the cable-settle equilibrium and
    avoids the spurious force transient that would arise from initialising with
    ``step_idx=0`` data (which is recorded *after* the first move step).
    """
    source_list = tuple(sources)
    if len(source_list) != int(env.num_envs):
        raise ValueError(
            f"sources length ({len(source_list)}) must match "
            f"env.num_envs ({env.num_envs})"
        )

    import newton

    scene = env._sim.scene
    layout = env._sim.layout
    if layout is None:
        raise RuntimeError("batched scene missing layout")

    robot_targets = (scene.robot_state_0, scene.robot_model)
    joint_buffers = [
        (target, target.joint_q.numpy().copy(), target.joint_qd.numpy().copy())
        for target in robot_targets
    ]
    vic = scene.vic_controller
    target_pos = None if vic is None else vic._target_pos_wp.numpy().copy()
    target_rot = None if vic is None else vic._target_rot_wp.numpy().copy()
    vic_default = getattr(scene, "vic_jt_default_dof_pos", None)
    legacy_default_row = None
    vic_default_batched = getattr(scene, "vic_jt_default_dof_pos_batched", None)
    default_rows = (
        None if vic_default_batched is None else vic_default_batched.numpy().copy()
    )

    for env_idx, source in enumerate(source_list):
        arrays = dataset.load_episode_obs_arrays(
            source.structure_idx, source.direction_idx
        )
        meta = dataset.load_episode_metadata(
            source.structure_idx, source.direction_idx
        )
        q = _array_or_none(meta.get("initial_robot_joint_q"), 7)
        if q is None:
            q = _frame_array_at_step(arrays, "robot_joint_q", FIRST_TRAJECTORY_STEP_IDX, 7)
        if q is None:
            raise ValueError(
                "missing initial robot_joint_q for "
                f"structure={source.structure_idx} "
                f"direction={source.direction_idx} "
                f"(env_idx={env_idx})"
            )
        q_flat = q.reshape(-1)
        q_slice = layout.joint_q_slice(env_idx)
        qd_slice = layout.joint_qd_slice(env_idx)
        for _target, joint_q, joint_qd in joint_buffers:
            joint_q[q_slice] = q_flat
            joint_qd[qd_slice] = 0.0

        if default_rows is not None:
            default_row = q_flat.copy()
            if default_row.shape[0] > 6:
                default_row[6] = 0.0
            default_rows[env_idx, : default_row.shape[0]] = default_row
        elif vic_default is not None and len(source_list) == 1:
            legacy_default_row = q_flat.copy()
            if legacy_default_row.shape[0] > 6:
                legacy_default_row[6] = 0.0

        if target_pos is not None and target_rot is not None:
            tcp_pos = _array_or_none(meta.get("initial_tcp_pos"), 3)
            if tcp_pos is None:
                tcp_pos = _frame_array_at_step(arrays, "tcp_pos", FIRST_TRAJECTORY_STEP_IDX, 3)
            tcp_quat = _array_or_none(meta.get("initial_tcp_quat"), 4)
            if tcp_quat is None:
                tcp_quat = _frame_array_at_step(arrays, "tcp_quat", FIRST_TRAJECTORY_STEP_IDX, 4)
            if tcp_pos is not None and tcp_quat is not None:
                target_pos[env_idx] = tcp_pos
                target_rot[env_idx] = tcp_quat

        env.set_excitation_context(
            env_idx,
            ExcitationContext(
                type="quasi_static",
                f_inst=0.0,
                direction=arrays["excitation_direction"][0],
            ),
        )

    for target, joint_q, joint_qd in joint_buffers:
        target.joint_q.assign(joint_q)
        target.joint_qd.assign(joint_qd)
    if vic_default_batched is not None and default_rows is not None:
        vic_default_batched.assign(default_rows)
    elif vic_default is not None and legacy_default_row is not None:
        vic_default.assign(legacy_default_row)

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


def initialize_batched_env_from_dataset(
    env: Any,
    dataset: BatchedSysIdDataset,
    *,
    structure_idx: int,
    num_directions: int,
    direction_indices: Sequence[int] | None = None,
) -> None:
    """Initialize replay worlds from one structure, cycling physical directions."""
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

    sources = tuple(
        ReplayEpisodeSource(
            structure_idx=int(structure_idx),
            direction_idx=int(dirs[env_idx % len(dirs)]),
        )
        for env_idx in range(int(env.num_envs))
    )
    initialize_batched_env_from_episode_sources(env, dataset, sources)
