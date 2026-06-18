"""Parquet persistence for sysID trajectory rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from apple_pick_sim.system_id.episode_meta import EpisodeMeta

PHASE_TO_INT: dict[str, int] = {
    "move_out": 0,
    "hold": 1,
    "return": 2,
}

WOODY_START_PREFIX = "woody_start__"
WOODY_END_PREFIX = "woody_end__"

REQUIRED_FRAME_COLUMNS: tuple[str, ...] = (
    "episode_id",
    "step_idx",
    "phase",
    "excitation_type",
    "excitation_direction",
    "action",
    "tcp_velocity",
    "ft_wrist",
)

BONUS_FRAME_COLUMNS: tuple[str, ...] = (
    "sim_time",
    "dir_idx",
    "amplitude_m",
    "tcp_pos",
    "apple_pos",
    "woody_part_force",
)

FRAME_COLUMNS: tuple[str, ...] = REQUIRED_FRAME_COLUMNS + BONUS_FRAME_COLUMNS

INITIAL_STATE_KEYS: tuple[str, ...] = (
    "robot_body_q",
    "robot_body_qd",
    "robot_joint_q",
    "robot_joint_qd",
    "cable_body_q",
    "cable_body_qd",
    "vic_target_tf",
)

OPTIONAL_INITIAL_STATE_KEYS: tuple[str, ...] = (
    "cable_state_1_body_q",
    "cable_state_1_body_qd",
)

INITIAL_OBS_KEYS: tuple[str, ...] = (
    "obs_apple_pos",
    "obs_tcp_pos",
    "obs_ft_wrist",
    "obs_tcp_velocity",
    "obs_woody_start",
    "obs_woody_end",
)

SNAPSHOT_KEYS: tuple[str, ...] = (
    INITIAL_STATE_KEYS
    + OPTIONAL_INITIAL_STATE_KEYS
    + INITIAL_OBS_KEYS
    + ("weld_direction", "weld_reference_pos", "weld_reference_quat")
)

METADATA_COLUMNS: tuple[str, ...] = (
    "episode_id",
    "weld_direction",
    "excitation_type",
    "n_woody_parts",
    "junction_names",
    "params_fingerprint",
    "control_hz",
    "timestamp",
    "seed",
    "n_directions",
    "initial_tcp_pos",
    "fixture_path",
    "movement_per_step_m",
    "total_movement_m",
    "hold_duration_s",
    "move_speed_mps",
    "skip_return",
)


def woody_start_column(junction_name: str) -> str:
    return f"{WOODY_START_PREFIX}{junction_name}"


def woody_end_column(junction_name: str) -> str:
    return f"{WOODY_END_PREFIX}{junction_name}"


def junction_name_from_start_column(column_name: str) -> str | None:
    if column_name.startswith(WOODY_START_PREFIX):
        return column_name[len(WOODY_START_PREFIX) :]
    return None


def junction_names_from_frame_columns(column_names: list[str]) -> list[str]:
    return [
        name
        for col in column_names
        if (name := junction_name_from_start_column(col)) is not None
    ]


def stack_woody_pos_from_obs(
    woody_by_junction: dict[str, np.ndarray],
    junction_names: list[str],
) -> np.ndarray:
    """Concatenate per-junction ``(3,)`` positions from a single env observation."""
    parts = [
        np.asarray(woody_by_junction[name], dtype=np.float32).reshape(3)
        for name in junction_names
    ]
    return np.concatenate(parts, dtype=np.float32)


def stack_woody_pos_frame(
    woody_by_junction: dict[str, np.ndarray],
    frame_idx: int,
    junction_names: list[str],
) -> np.ndarray:
    """Concatenate per-junction ``(3,)`` positions into flat ``(N*3,)`` for one frame."""
    parts = [
        np.asarray(woody_by_junction[name][frame_idx], dtype=np.float32).reshape(3)
        for name in junction_names
    ]
    return np.concatenate(parts, dtype=np.float32)


def phase_to_int(phase: str) -> int:
    """Map trajectory phase name to stored int code."""
    try:
        return PHASE_TO_INT[phase]
    except KeyError as exc:
        raise ValueError(f"unknown phase: {phase!r}") from exc


def _as_f32_list(value: Any, *, size: int | None = None) -> list[float]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if size is not None and arr.size != size:
        raise ValueError(f"expected length {size}, got {arr.size}")
    return [float(x) for x in arr.tolist()]


def target_tf_to_array(target_tf: Any) -> np.ndarray:
    """Serialize a Warp transform as ``(7,)`` float32 ``[px, py, pz, qx, qy, qz, qw]``."""
    import warp as wp

    pos = wp.transform_get_translation(target_tf)
    rot = wp.transform_get_rotation(target_tf)
    return np.array(
        [pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]],
        dtype=np.float32,
    )


def target_tf_from_array(arr: np.ndarray) -> Any:
    """Deserialize :func:`target_tf_to_array` output back to a Warp transform."""
    import warp as wp

    values = np.asarray(arr, dtype=np.float64).reshape(7)
    return wp.transform(
        wp.vec3(float(values[0]), float(values[1]), float(values[2])),
        wp.quat(float(values[3]), float(values[4]), float(values[5]), float(values[6])),
    )


def grasp_snapshot_from_env(
    env: Any,
    *,
    obs: dict[str, Any] | None = None,
    weld_direction: tuple[float, float, float] | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build npz-ready post-reset physics arrays from a snapshotted sysID env."""
    scene = env._scene
    controller = env._controller
    if scene is None or controller is None:
        raise RuntimeError("Environment must be reset() before grasp_snapshot_from_env().")
    snapshot = {
        "robot_body_q": scene.robot_state_0.body_q.numpy().copy(),
        "robot_body_qd": scene.robot_state_0.body_qd.numpy().copy(),
        "robot_joint_q": scene.robot_state_0.joint_q.numpy().copy(),
        "robot_joint_qd": scene.robot_state_0.joint_qd.numpy().copy(),
        "cable_body_q": scene.cable.state_0.body_q.numpy().copy(),
        "cable_body_qd": scene.cable.state_0.body_qd.numpy().copy(),
        "cable_state_1_body_q": scene.cable.state_1.body_q.numpy().copy(),
        "cable_state_1_body_qd": scene.cable.state_1.body_qd.numpy().copy(),
        "vic_target_tf": target_tf_to_array(controller.target_tf),
    }
    apple_body = getattr(scene.cable, "apple_body", None)
    if apple_body is not None:
        apple_q = scene.cable.state_0.body_q.numpy().reshape(-1, 7)[int(apple_body)]
        snapshot["weld_reference_pos"] = np.asarray(apple_q[:3], dtype=np.float32).reshape(3)
        snapshot["weld_reference_quat"] = np.asarray(apple_q[3:7], dtype=np.float32).reshape(4)
    if obs is not None:
        junction_names = sorted(obs["woody_part_start_pos"].keys())
        snapshot["obs_apple_pos"] = np.asarray(obs["apple_pos"], dtype=np.float32).reshape(3)
        snapshot["obs_tcp_pos"] = np.asarray(obs["tcp_pos"], dtype=np.float32).reshape(3)
        snapshot["obs_ft_wrist"] = np.asarray(obs["ft_wrist"], dtype=np.float32).reshape(6)
        snapshot["obs_tcp_velocity"] = np.asarray(obs["tcp_velocity"], dtype=np.float32).reshape(6)
        snapshot["obs_woody_start"] = stack_woody_pos_from_obs(
            obs["woody_part_start_pos"], junction_names
        )
        snapshot["obs_woody_end"] = stack_woody_pos_from_obs(
            obs["woody_part_end_pos"], junction_names
        )
    if weld_direction is not None:
        snapshot["weld_direction"] = np.asarray(weld_direction, dtype=np.float32).reshape(3)
    return snapshot


def load_grasp_snapshot_into_env(env: Any, snapshot: dict[str, np.ndarray]) -> None:
    """Populate env grasp buffers from a saved snapshot and restore pose."""
    env._grasp_robot_body_q = np.asarray(snapshot["robot_body_q"])
    env._grasp_robot_body_qd = np.asarray(snapshot["robot_body_qd"])
    env._grasp_robot_joint_q = np.asarray(snapshot["robot_joint_q"])
    env._grasp_robot_joint_qd = np.asarray(snapshot["robot_joint_qd"])
    env._grasp_cable_body_q = np.asarray(snapshot["cable_body_q"])
    env._grasp_cable_body_qd = np.asarray(snapshot["cable_body_qd"])
    if "cable_state_1_body_q" in snapshot:
        env._grasp_cable_state_1_body_q = np.asarray(snapshot["cable_state_1_body_q"])
    if "cable_state_1_body_qd" in snapshot:
        env._grasp_cable_state_1_body_qd = np.asarray(snapshot["cable_state_1_body_qd"])
    if "weld_reference_pos" in snapshot:
        env._weld_reference_pos_override = tuple(
            float(x) for x in np.asarray(snapshot["weld_reference_pos"]).reshape(3)
        )
    if "weld_reference_quat" in snapshot:
        env._weld_reference_quat_override = tuple(
            float(x) for x in np.asarray(snapshot["weld_reference_quat"]).reshape(4)
        )
    env._grasp_target_tf = target_tf_from_array(snapshot["vic_target_tf"])
    env.restore_grasp_pose()


def _woody_pos_dict_from_obs(value: Any) -> dict[str, np.ndarray]:
    if not isinstance(value, dict):
        raise TypeError(
            "woody_part_start_pos and woody_part_end_pos must be dict[str, ndarray]"
        )
    return {str(name): np.asarray(pos, dtype=np.float32).reshape(3) for name, pos in value.items()}


class TrajectoryWriter:
    """Accumulate per-frame sysID records and write Parquet episode files."""

    def __init__(self, *, episode_id: str | None = None) -> None:
        self._episode_id = episode_id or str(uuid4())
        self._rows: list[dict[str, Any]] = []

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def save_initial_state(self, output_dir: Path | str, snapshot: dict[str, np.ndarray]) -> Path:
        """Save post-warmup physics state to ``initial_states/<episode_id>.npz``."""
        output_dir = Path(output_dir)
        out = output_dir / "initial_states" / f"{self._episode_id}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        allowed = set(SNAPSHOT_KEYS)
        arrays = {key: np.asarray(snapshot[key]) for key in snapshot if key in allowed}
        missing = set(INITIAL_STATE_KEYS) - set(arrays)
        if missing:
            raise ValueError(f"snapshot missing required keys: {sorted(missing)}")
        np.savez(out, **arrays)
        return out

    def record_step(
        self,
        *,
        step_idx: int,
        sim_time: float,
        phase: str,
        dir_idx: int,
        amplitude_m: float,
        action: np.ndarray,
        obs: dict[str, Any],
    ) -> None:
        """Append one env-step record."""
        start_by_name = _woody_pos_dict_from_obs(obs["woody_part_start_pos"])
        end_by_name = _woody_pos_dict_from_obs(obs["woody_part_end_pos"])
        if set(start_by_name) != set(end_by_name):
            raise ValueError("woody_part_start_pos and woody_part_end_pos keys must match")

        row: dict[str, Any] = {
            "episode_id": self._episode_id,
            "step_idx": int(step_idx),
            "phase": phase_to_int(phase),
            "excitation_type": int(obs["excitation_type"]),
            "excitation_direction": _as_f32_list(obs["excitation_direction"], size=3),
            "action": _as_f32_list(action, size=6),
            "tcp_velocity": _as_f32_list(obs["tcp_velocity"], size=6),
            "ft_wrist": _as_f32_list(obs["ft_wrist"], size=6),
            "sim_time": float(sim_time),
            "dir_idx": int(dir_idx),
            "amplitude_m": float(amplitude_m),
            "tcp_pos": _as_f32_list(obs.get("tcp_pos", np.zeros(3)), size=3),
            "apple_pos": _as_f32_list(obs.get("apple_pos", np.zeros(3)), size=3),
            "woody_part_force": _as_f32_list(obs.get("woody_part_force", np.zeros(0))),
        }
        for name in sorted(start_by_name):
            row[woody_start_column(name)] = _as_f32_list(start_by_name[name], size=3)
            row[woody_end_column(name)] = _as_f32_list(end_by_name[name], size=3)
        self._rows.append(row)

    def save(self, output_dir: Path, meta: EpisodeMeta) -> Path:
        """Write frames parquet and append metadata row."""
        if meta.episode_id != self._episode_id:
            raise ValueError(
                f"EpisodeMeta episode_id {meta.episode_id!r} does not match "
                f"writer episode_id {self._episode_id!r}"
            )
        if not self._rows:
            raise ValueError("cannot save trajectory with zero recorded frames")

        output_dir = Path(output_dir)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frames_path = frames_dir / f"{self._episode_id}.parquet"
        pq.write_table(pa.Table.from_pylist(self._rows), frames_path)

        meta_path = output_dir / "metadata.parquet"
        meta_table = pa.Table.from_pylist([meta.to_row()])
        if meta_path.exists():
            existing = pq.read_table(meta_path)
            meta_table = pa.concat_tables([existing, meta_table])
        pq.write_table(meta_table, meta_path)
        return frames_path


class TrajectoryDataset:
    """Read a sysID trajectory dataset directory."""

    def __init__(self, dataset_dir: Path | str) -> None:
        self._dataset_dir = Path(dataset_dir)
        self._meta_path = self._dataset_dir / "metadata.parquet"
        if not self._meta_path.exists():
            raise FileNotFoundError(f"metadata.parquet not found in {self._dataset_dir}")

    @property
    def dataset_dir(self) -> Path:
        return self._dataset_dir

    def _metadata_table(self) -> pa.Table:
        return pq.read_table(self._meta_path)

    def episode_ids(self) -> list[str]:
        table = self._metadata_table()
        return [str(x) for x in table.column("episode_id").to_pylist()]

    def load_episode_meta(self, episode_id: str) -> dict[str, Any]:
        table = self._metadata_table()
        ids = table.column("episode_id").to_pylist()
        try:
            row_idx = ids.index(episode_id)
        except ValueError as exc:
            raise KeyError(f"episode_id {episode_id!r} not found") from exc
        return {name: table.column(name)[row_idx].as_py() for name in table.column_names}

    def load_episode_frames(self, episode_id: str) -> pa.Table:
        frames_path = self._dataset_dir / "frames" / f"{episode_id}.parquet"
        if not frames_path.exists():
            raise FileNotFoundError(f"frames parquet not found: {frames_path}")
        return pq.read_table(frames_path)

    def load_episode_actions(self, episode_id: str) -> np.ndarray:
        """Return recorded actions as ``(T, 6)`` float32 array."""
        table = self.load_episode_frames(episode_id)
        actions = [np.asarray(row, dtype=np.float32) for row in table.column("action").to_pylist()]
        return np.stack(actions, axis=0)

    def load_initial_state(self, episode_id: str) -> dict[str, np.ndarray] | None:
        """Return saved physics snapshot or ``None`` if not present."""
        path = self._dataset_dir / "initial_states" / f"{episode_id}.npz"
        if not path.exists():
            return None
        data = np.load(path)
        return {key: np.asarray(data[key]) for key in data.files}

    def load_episode_obs_arrays(self, episode_id: str) -> dict[str, Any]:
        """Return recorded per-frame observations as stacked ``float32`` arrays."""
        table = self.load_episode_frames(episode_id)
        meta = self.load_episode_meta(episode_id)
        junction_names = meta.get("junction_names")
        if not junction_names:
            junction_names = junction_names_from_frame_columns(list(table.column_names))

        def _stack_column(name: str) -> np.ndarray:
            if name not in table.column_names:
                return np.zeros((0,), dtype=np.float32)
            rows = table.column(name).to_pylist()
            if not rows:
                return np.zeros((0,), dtype=np.float32)
            first = np.asarray(rows[0], dtype=np.float32).reshape(-1)
            if first.size == 0:
                return np.zeros((len(rows), 0), dtype=np.float32)
            return np.stack([np.asarray(row, dtype=np.float32).reshape(-1) for row in rows], axis=0)

        def _stack_woody(prefix: str) -> dict[str, np.ndarray]:
            out: dict[str, np.ndarray] = {}
            for name in junction_names:
                col = f"{prefix}{name}"
                rows = table.column(col).to_pylist()
                out[name] = np.stack(
                    [np.asarray(row, dtype=np.float32).reshape(3) for row in rows],
                    axis=0,
                )
            return out

        return {
            "step_idx": np.asarray(table.column("step_idx").to_pylist(), dtype=np.int32),
            "action": _stack_column("action").reshape(-1, 6),
            "ft_wrist": _stack_column("ft_wrist").reshape(-1, 6),
            "tcp_velocity": _stack_column("tcp_velocity").reshape(-1, 6),
            "woody_part_start_pos": _stack_woody(WOODY_START_PREFIX),
            "woody_part_end_pos": _stack_woody(WOODY_END_PREFIX),
            "tcp_pos": _stack_column("tcp_pos").reshape(-1, 3),
            "apple_pos": _stack_column("apple_pos").reshape(-1, 3),
            "junction_names": list(junction_names),
        }
