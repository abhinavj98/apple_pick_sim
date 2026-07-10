"""Opt-in disk I/O for batched post-weld EpisodeStateSnapshot (settle/init diagnostic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.coupled_fruiting.episode_state_snapshot import EpisodeStateSnapshot
from apple_pick_sim.system_id.excitation_state import ExcitationContext

WORLD_FRAME = "world"

BODY_Q_KEYS: tuple[str, ...] = (
    "robot_body_q",
    "cable_body_q_0",
    "cable_body_q_1",
)

SNAPSHOT_ARRAY_KEYS: tuple[str, ...] = (
    "robot_body_q",
    "robot_body_qd",
    "robot_joint_q",
    "robot_joint_qd",
    "model_joint_q",
    "model_joint_qd",
    "cable_body_q_0",
    "cable_body_qd_0",
    "cable_body_q_1",
    "cable_body_qd_1",
    "vic_target_pos",
    "vic_target_rot",
    "vic_lin_vels",
    "vic_ang_vels",
    "vic_default_dof_pos_batched",
)


def initial_state_path(output_dir: Path | str, structure_idx: int, direction_idx: int) -> Path:
    root = Path(output_dir) / "initial_states"
    return root / f"s{int(structure_idx):02d}_d{int(direction_idx):02d}.npz"


def _robot_body_slice(layout: BatchedEnvLayout, world: int) -> slice:
    start = int(world) * int(layout.robot_bodies_per_world)
    end = start + int(layout.robot_bodies_per_world)
    return slice(start, end)


def _cable_body_slice(layout: BatchedEnvLayout, world: int) -> slice:
    start = int(world) * int(layout.bodies_per_world)
    end = start + int(layout.bodies_per_world)
    return slice(start, end)


def _slice_world_arrays(
    snap: EpisodeStateSnapshot,
    layout: BatchedEnvLayout,
    world: int,
) -> dict[str, np.ndarray]:
    rb = _robot_body_slice(layout, world)
    cb = _cable_body_slice(layout, world)
    jq = layout.joint_q_slice(world)
    jqd = layout.joint_qd_slice(world)
    out: dict[str, np.ndarray] = {
        "robot_body_q": snap.robot_body_q.numpy()[rb].copy(),
        "robot_body_qd": snap.robot_body_qd.numpy()[rb].copy(),
        "robot_joint_q": snap.robot_joint_q.numpy()[jq].copy(),
        "robot_joint_qd": snap.robot_joint_qd.numpy()[jqd].copy(),
        "model_joint_q": snap.model_joint_q.numpy()[jq].copy(),
        "model_joint_qd": snap.model_joint_qd.numpy()[jqd].copy(),
        "cable_body_q_0": snap.cable_body_q_0.numpy()[cb].copy(),
        "cable_body_qd_0": snap.cable_body_qd_0.numpy()[cb].copy(),
        "cable_body_q_1": snap.cable_body_q_1.numpy()[cb].copy(),
        "cable_body_qd_1": snap.cable_body_qd_1.numpy()[cb].copy(),
    }
    if snap.vic_target_pos is not None:
        out["vic_target_pos"] = snap.vic_target_pos.numpy()[world].copy()
    if snap.vic_target_rot is not None:
        out["vic_target_rot"] = snap.vic_target_rot.numpy()[world].copy()
    if snap.vic_lin_vels is not None:
        out["vic_lin_vels"] = snap.vic_lin_vels.numpy()[world].copy()
    if snap.vic_ang_vels is not None:
        out["vic_ang_vels"] = snap.vic_ang_vels.numpy()[world].copy()
    if snap.vic_default_dof_pos_batched is not None:
        out["vic_default_dof_pos_batched"] = snap.vic_default_dof_pos_batched.numpy()[world].copy()
    return out


def _write_world_slice(
    target: dict[str, np.ndarray],
    source: dict[str, np.ndarray],
    layout: BatchedEnvLayout,
    world: int,
) -> None:
    rb = _robot_body_slice(layout, world)
    cb = _cable_body_slice(layout, world)
    jq = layout.joint_q_slice(world)
    jqd = layout.joint_qd_slice(world)
    world_arrays = {key: np.asarray(source[key]).copy() for key in source}
    target["robot_body_q"][rb] = world_arrays["robot_body_q"]
    target["robot_body_qd"][rb] = world_arrays["robot_body_qd"]
    target["robot_joint_q"][jq] = world_arrays["robot_joint_q"]
    target["robot_joint_qd"][jqd] = world_arrays["robot_joint_qd"]
    target["model_joint_q"][jq] = world_arrays["model_joint_q"]
    target["model_joint_qd"][jqd] = world_arrays["model_joint_qd"]
    target["cable_body_q_0"][cb] = world_arrays["cable_body_q_0"]
    target["cable_body_qd_0"][cb] = world_arrays["cable_body_qd_0"]
    target["cable_body_q_1"][cb] = world_arrays["cable_body_q_1"]
    target["cable_body_qd_1"][cb] = world_arrays["cable_body_qd_1"]
    if "vic_target_pos" in world_arrays:
        target["vic_target_pos"][world] = world_arrays["vic_target_pos"]
    if "vic_target_rot" in world_arrays:
        target["vic_target_rot"][world] = world_arrays["vic_target_rot"]
    if "vic_lin_vels" in world_arrays:
        target["vic_lin_vels"][world] = world_arrays["vic_lin_vels"]
    if "vic_ang_vels" in world_arrays:
        target["vic_ang_vels"][world] = world_arrays["vic_ang_vels"]
    if "vic_default_dof_pos_batched" in world_arrays:
        target["vic_default_dof_pos_batched"][world] = world_arrays["vic_default_dof_pos_batched"]


def _allocate_merge_buffers(snap: EpisodeStateSnapshot) -> dict[str, np.ndarray]:
    vic_pos = snap.vic_target_pos.numpy().copy() if snap.vic_target_pos is not None else None
    vic_rot = snap.vic_target_rot.numpy().copy() if snap.vic_target_rot is not None else None
    vic_lin = snap.vic_lin_vels.numpy().copy() if snap.vic_lin_vels is not None else None
    vic_ang = snap.vic_ang_vels.numpy().copy() if snap.vic_ang_vels is not None else None
    vic_default = (
        snap.vic_default_dof_pos_batched.numpy().copy()
        if snap.vic_default_dof_pos_batched is not None
        else None
    )
    out: dict[str, np.ndarray] = {
        "robot_body_q": snap.robot_body_q.numpy().copy(),
        "robot_body_qd": snap.robot_body_qd.numpy().copy(),
        "robot_joint_q": snap.robot_joint_q.numpy().copy(),
        "robot_joint_qd": snap.robot_joint_qd.numpy().copy(),
        "model_joint_q": snap.model_joint_q.numpy().copy(),
        "model_joint_qd": snap.model_joint_qd.numpy().copy(),
        "cable_body_q_0": snap.cable_body_q_0.numpy().copy(),
        "cable_body_qd_0": snap.cable_body_qd_0.numpy().copy(),
        "cable_body_q_1": snap.cable_body_q_1.numpy().copy(),
        "cable_body_qd_1": snap.cable_body_qd_1.numpy().copy(),
    }
    if vic_pos is not None:
        out["vic_target_pos"] = vic_pos
    if vic_rot is not None:
        out["vic_target_rot"] = vic_rot
    if vic_lin is not None:
        out["vic_lin_vels"] = vic_lin
    if vic_ang is not None:
        out["vic_ang_vels"] = vic_ang
    if vic_default is not None:
        out["vic_default_dof_pos_batched"] = vic_default
    return out


def _snapshot_from_merge_buffers(
    merge: dict[str, np.ndarray],
    template: EpisodeStateSnapshot,
    device: Any,
) -> EpisodeStateSnapshot:
    dtype_by_key: dict[str, Any] = {
        "robot_body_q": template.robot_body_q.dtype,
        "robot_body_qd": template.robot_body_qd.dtype,
        "robot_joint_q": template.robot_joint_q.dtype,
        "robot_joint_qd": template.robot_joint_qd.dtype,
        "model_joint_q": template.model_joint_q.dtype,
        "model_joint_qd": template.model_joint_qd.dtype,
        "cable_body_q_0": template.cable_body_q_0.dtype,
        "cable_body_qd_0": template.cable_body_qd_0.dtype,
        "cable_body_q_1": template.cable_body_q_1.dtype,
        "cable_body_qd_1": template.cable_body_qd_1.dtype,
    }
    if template.vic_target_pos is not None:
        dtype_by_key["vic_target_pos"] = template.vic_target_pos.dtype
    if template.vic_target_rot is not None:
        dtype_by_key["vic_target_rot"] = template.vic_target_rot.dtype
    if template.vic_lin_vels is not None:
        dtype_by_key["vic_lin_vels"] = template.vic_lin_vels.dtype
    if template.vic_ang_vels is not None:
        dtype_by_key["vic_ang_vels"] = template.vic_ang_vels.dtype
    if template.vic_default_dof_pos_batched is not None:
        dtype_by_key["vic_default_dof_pos_batched"] = template.vic_default_dof_pos_batched.dtype

    def _to_wp(key: str) -> wp.array:
        return wp.array(merge[key], dtype=dtype_by_key[key], device=device)

    vic_pos = vic_rot = vic_lin = vic_ang = vic_default = None
    if "vic_target_pos" in merge:
        vic_pos = _to_wp("vic_target_pos")
    if "vic_target_rot" in merge:
        vic_rot = _to_wp("vic_target_rot")
    if "vic_lin_vels" in merge:
        vic_lin = _to_wp("vic_lin_vels")
    if "vic_ang_vels" in merge:
        vic_ang = _to_wp("vic_ang_vels")
    if "vic_default_dof_pos_batched" in merge:
        vic_default = _to_wp("vic_default_dof_pos_batched")

    return EpisodeStateSnapshot(
        robot_body_q=_to_wp("robot_body_q"),
        robot_body_qd=_to_wp("robot_body_qd"),
        robot_joint_q=_to_wp("robot_joint_q"),
        robot_joint_qd=_to_wp("robot_joint_qd"),
        model_joint_q=_to_wp("model_joint_q"),
        model_joint_qd=_to_wp("model_joint_qd"),
        cable_body_q_0=_to_wp("cable_body_q_0"),
        cable_body_qd_0=_to_wp("cable_body_qd_0"),
        cable_body_q_1=_to_wp("cable_body_q_1"),
        cable_body_qd_1=_to_wp("cable_body_qd_1"),
        vic_target_pos=vic_pos,
        vic_target_rot=vic_rot,
        vic_lin_vels=vic_lin,
        vic_ang_vels=vic_ang,
        vic_default_dof_pos_batched=vic_default,
    )


def save_per_env_episode_snapshots(
    sim: Any,
    *,
    output_dir: Path | str,
    num_directions: int,
) -> list[Path]:
    """Write one ``initial_states/sXX_dYY.npz`` per env after post-weld reset."""
    layout = sim.layout
    if layout is None:
        raise RuntimeError("batched sim missing layout")

    snap = sim.episode_snapshot
    if snap is None:
        snap = EpisodeStateSnapshot.capture(sim)

    written: list[Path] = []
    for env_idx in range(int(layout.num_envs)):
        structure_idx = int(env_idx) // int(num_directions)
        direction_idx = int(env_idx) % int(num_directions)
        path = initial_state_path(output_dir, structure_idx, direction_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = _slice_world_arrays(snap, layout, int(env_idx))
        np.savez(path, origin_frame=np.asarray(WORLD_FRAME), **arrays)
        written.append(path)
    return written


def load_npz_for_direction(
    dataset_dir: Path | str,
    *,
    structure_idx: int,
    direction_idx: int,
) -> dict[str, np.ndarray]:
    path = initial_state_path(dataset_dir, structure_idx, direction_idx)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing post-weld snapshot {path}; collect with --save-snapshot first"
        )
    with np.load(path) as data:
        if "origin_frame" not in data:
            raise ValueError(
                f"snapshot {path} missing origin_frame marker; "
                "re-collect with --save-snapshot"
            )
        frame = str(np.asarray(data["origin_frame"]).item())
        if frame != WORLD_FRAME:
            raise ValueError(
                f"snapshot {path} has unsupported origin_frame={frame!r}; "
                "re-collect with --save-snapshot"
            )
        return {
            key: np.asarray(data[key])
            for key in data.files
            if key != "origin_frame"
        }


def _set_excitation_from_dataset(
    env: Any,
    dataset: Any,
    *,
    structure_idx: int,
    num_directions: int,
) -> None:
    for env_idx in range(int(env.num_envs)):
        direction_idx = int(env_idx) % int(num_directions)
        arrays = dataset.load_episode_obs_arrays(structure_idx, direction_idx)
        excitation_direction = arrays["excitation_direction"][0]
        env.set_excitation_context(
            env_idx,
            ExcitationContext(
                type="quasi_static",
                f_inst=0.0,
                direction=excitation_direction,
            ),
        )


def load_and_restore_episode_snapshots(
    env: Any,
    dataset: Any,
    *,
    structure_idx: int,
    num_directions: int,
) -> None:
    """Restore post-weld state0/state1 from disk; skip metadata joint/VIC init."""
    sim = env._sim
    layout = sim.layout
    if layout is None:
        raise RuntimeError("batched sim missing layout")

    template = sim.episode_snapshot
    if template is None:
        template = EpisodeStateSnapshot.capture(sim)

    merge = _allocate_merge_buffers(template)
    for env_idx in range(int(env.num_envs)):
        direction_idx = int(env_idx) % int(num_directions)
        per_dir = load_npz_for_direction(
            dataset.dataset_dir,
            structure_idx=int(structure_idx),
            direction_idx=int(direction_idx),
        )
        _write_world_slice(merge, per_dir, layout, int(env_idx))

    restored = _snapshot_from_merge_buffers(merge, template, sim.device)
    restored.restore(sim)
    sim.capture_episode_snapshot()
    _set_excitation_from_dataset(
        env,
        dataset,
        structure_idx=int(structure_idx),
        num_directions=int(num_directions),
    )

    env._step_count = 0
    env._last_obs = env._gather_obs()
    if hasattr(env, "_refresh_per_env_reset_info"):
        env._refresh_per_env_reset_info()
