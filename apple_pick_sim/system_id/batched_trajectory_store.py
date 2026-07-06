"""Parquet persistence for batched quasi-static sys-ID datasets (v1 layout)."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from apple_pick_sim.system_id.trajectory_store import (
    METADATA_COLUMNS,
    WOODY_END_PREFIX,
    WOODY_START_PREFIX,
    build_sysid_frame_row,
    junction_names_from_frame_columns,
    woody_end_column,
    woody_start_column,
)

SCHEMA_VERSION = "batched_sysid_v1"

BATCHED_REQUIRED_FRAME_COLUMNS: tuple[str, ...] = (
    "step_idx",
    "phase",
    "excitation_type",
    "excitation_direction",
    "action",
    "tcp_velocity",
    "ft_wrist",
)

BATCHED_BONUS_FRAME_COLUMNS: tuple[str, ...] = (
    "sim_time",
    "amplitude_m",
    "raw_ft_wrist",
    "tcp_pos",
    "apple_pos",
    "tcp_quat",
    "apple_quat",
    "robot_joint_q",
    "woody_part_force",
)

EPISODE_METADATA_KEYS: tuple[str, ...] = (
    "schema_version",
    "episode_id",
    "structure_idx",
    "direction_idx",
    "env_idx",
    "pull_direction",
    "params_fingerprint",
    "fruiting_system_params",
    "excitation_type",
    "control_hz",
    "seed",
    "n_woody_parts",
    "junction_names",
    "initial_tcp_pos",
    "initial_tcp_quat",
    "initial_apple_pos",
    "initial_apple_quat",
    "initial_robot_joint_q",
    "fixture_path",
    "fruiting_base_pos",
    "apple_radius",
    "rod_radii",
    "weld_direction",
    "weld_reference_pos",
    "weld_reference_quat",
    "movement_per_step_m",
    "total_movement_m",
    "hold_duration_s",
    "move_speed_mps",
    "skip_return",
)


def episode_filename(structure_idx: int, direction_idx: int) -> str:
    """Relative path under the dataset root for one grid episode."""
    return f"episodes/s{int(structure_idx):02d}_d{int(direction_idx):02d}.parquet"


def batched_dataset_exists(output_dir: Path | str) -> bool:
    """Return True when ``output_dir/manifest.json`` is present."""
    return (Path(output_dir) / "manifest.json").is_file()


def resolve_batched_dataset_output_dir(
    output_dir: Path | str,
    *,
    overwrite: bool = False,
    append_timestamp: bool = True,
    now: datetime | None = None,
) -> Path:
    """Pick a writable dataset root when ``output_dir`` already holds a v1 dataset.

    - Fresh path: return ``output_dir`` unchanged.
    - Existing + ``overwrite``: return ``output_dir`` (caller replaces contents).
    - Existing + ``append_timestamp``: warn and return a sibling
      ``{name}_{YYYYMMDDTHHMMSSZ}`` directory.
    - Existing + neither: raise ``FileExistsError``.
    """
    out = Path(output_dir)
    if not batched_dataset_exists(out):
        return out
    if overwrite:
        return out
    if append_timestamp:
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
        redirected = out.parent / f"{out.name}_{stamp}"
        warnings.warn(
            f"Batched sys-ID dataset already exists at {out}; writing to {redirected}",
            UserWarning,
            stacklevel=2,
        )
        return redirected
    raise FileExistsError(
        f"batched sys-ID dataset already exists at {out} "
        "(pass overwrite=True or append_timestamp=True)"
    )


def episode_metadata_to_schema_metadata(metadata: dict[str, Any]) -> dict[bytes, bytes]:
    """Encode episode-level metadata for Parquet schema metadata."""
    out: dict[bytes, bytes] = {}
    for key in EPISODE_METADATA_KEYS:
        if key not in metadata:
            continue
        value = metadata[key]
        out[key.encode("utf-8")] = json.dumps(value, sort_keys=True).encode("utf-8")
    return out


def schema_metadata_to_episode_metadata(raw: dict[bytes, bytes] | None) -> dict[str, Any]:
    """Decode Parquet schema metadata into a Python dict."""
    if not raw:
        return {}
    out: dict[str, Any] = {}
    for key_bytes, value_bytes in raw.items():
        key = key_bytes.decode("utf-8")
        out[key] = json.loads(value_bytes.decode("utf-8"))
    return out


class BatchedEpisodeWriter:
    """Accumulate per-frame sys-ID records for one batched episode."""

    def __init__(self, *, episode_id: str | None = None) -> None:
        self._episode_id = episode_id or str(uuid4())
        self._rows: list[dict[str, Any]] = []

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def n_frames(self) -> int:
        return len(self._rows)

    def record_step(
        self,
        *,
        step_idx: int,
        sim_time: float,
        phase: str,
        amplitude_m: float,
        action: np.ndarray,
        obs: dict[str, Any],
    ) -> None:
        """Append one env-step record (no per-frame episode_id / dir_idx)."""
        self._rows.append(
            build_sysid_frame_row(
                step_idx=step_idx,
                sim_time=sim_time,
                phase=phase,
                amplitude_m=amplitude_m,
                action=action,
                obs=obs,
            )
        )

    def save(self, path: Path | str, episode_metadata: dict[str, Any]) -> Path:
        """Write episode parquet with schema metadata."""
        if not self._rows:
            raise ValueError("cannot save trajectory with zero recorded frames")

        meta = dict(episode_metadata)
        meta.setdefault("schema_version", SCHEMA_VERSION)
        meta.setdefault("episode_id", self._episode_id)

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(self._rows)
        table = table.replace_schema_metadata(episode_metadata_to_schema_metadata(meta))
        pq.write_table(table, out)
        return out


def write_manifest(
    output_dir: Path | str,
    *,
    command_argv: list[str],
    collection: dict[str, Any],
    structures: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    overwrite: bool = False,
) -> Path:
    """Write dataset-level manifest.json (call after all episode files)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists: {manifest_path}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command_argv": list(command_argv),
        "collection": collection,
        "structures": structures,
        "episodes": episodes,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


class BatchedSysIdDataset:
    """Read a batched sys-ID dataset directory (manifest.json + episodes/)."""

    def __init__(self, dataset_dir: Path | str) -> None:
        self._dataset_dir = Path(dataset_dir)
        self._manifest_path = self._dataset_dir / "manifest.json"
        if not self._manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {self._dataset_dir}")
        self._manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))

    @property
    def dataset_dir(self) -> Path:
        return self._dataset_dir

    @property
    def manifest(self) -> dict[str, Any]:
        return dict(self._manifest)

    def episode_entries(self) -> list[dict[str, Any]]:
        return list(self._manifest.get("episodes", []))

    def structure_summaries(self) -> list[dict[str, Any]]:
        return list(self._manifest.get("structures", []))

    def episode_path(self, structure_idx: int, direction_idx: int) -> Path:
        rel = episode_filename(structure_idx, direction_idx)
        return self._dataset_dir / rel

    def load_episode_metadata(self, structure_idx: int, direction_idx: int) -> dict[str, Any]:
        path = self.episode_path(structure_idx, direction_idx)
        if not path.exists():
            raise FileNotFoundError(f"episode parquet not found: {path}")
        meta = pq.read_metadata(path).schema.to_arrow_schema().metadata
        return schema_metadata_to_episode_metadata(meta)

    def load_episode_frames(self, structure_idx: int, direction_idx: int) -> pa.Table:
        path = self.episode_path(structure_idx, direction_idx)
        if not path.exists():
            raise FileNotFoundError(f"episode parquet not found: {path}")
        return pq.read_table(path)

    def load_episode_obs_arrays(self, structure_idx: int, direction_idx: int) -> dict[str, Any]:
        """Return recorded per-frame observations as stacked float32 arrays."""
        table = self.load_episode_frames(structure_idx, direction_idx)
        meta = self.load_episode_metadata(structure_idx, direction_idx)
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

        ft_wrist = _stack_column("ft_wrist").reshape(-1, 6)
        raw_ft_wrist = (
            _stack_column("raw_ft_wrist").reshape(-1, 6)
            if "raw_ft_wrist" in table.column_names
            else ft_wrist.copy()
        )
        arrays: dict[str, Any] = {
            "step_idx": np.asarray(table.column("step_idx").to_pylist(), dtype=np.int32),
            "phase": np.asarray(table.column("phase").to_pylist(), dtype=np.int8),
            "excitation_type": np.asarray(
                table.column("excitation_type").to_pylist(), dtype=np.int8
            ),
            "excitation_direction": _stack_column("excitation_direction").reshape(-1, 3),
            "action": _stack_column("action").reshape(-1, 6),
            "ft_wrist": ft_wrist,
            "raw_ft_wrist": raw_ft_wrist,
            "tcp_velocity": _stack_column("tcp_velocity").reshape(-1, 6),
            "woody_part_start_pos": _stack_woody(WOODY_START_PREFIX),
            "woody_part_end_pos": _stack_woody(WOODY_END_PREFIX),
            "tcp_pos": _stack_column("tcp_pos").reshape(-1, 3),
            "apple_pos": _stack_column("apple_pos").reshape(-1, 3),
            "tcp_quat": _stack_column("tcp_quat").reshape(-1, 4),
            "apple_quat": _stack_column("apple_quat").reshape(-1, 4),
            "robot_joint_q": _stack_column("robot_joint_q").reshape(-1, 7),
            "junction_names": list(junction_names),
        }
        if "sim_time" in table.column_names:
            arrays["sim_time"] = _stack_column("sim_time").reshape(-1)
        if "amplitude_m" in table.column_names:
            arrays["amplitude_m"] = _stack_column("amplitude_m").reshape(-1)
        return arrays


def materialize_legacy_episode_dir(
    dataset: BatchedSysIdDataset,
    *,
    structure_idx: int,
    direction_idx: int,
    output_dir: Path | str,
    overwrite: bool = False,
) -> Path:
    """Write one episode into legacy metadata.parquet + frames/ layout for replay tooling."""
    out = Path(output_dir)
    meta = dataset.load_episode_metadata(structure_idx, direction_idx)
    frames = dataset.load_episode_frames(structure_idx, direction_idx)
    episode_id = str(meta["episode_id"])

    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames_path = frames_dir / f"{episode_id}.parquet"
    if frames_path.exists() and not overwrite:
        raise FileExistsError(f"legacy frames file already exists: {frames_path}")

    frame_rows = frames.to_pylist()
    for row in frame_rows:
        row["episode_id"] = episode_id
        row["dir_idx"] = 0
    pq.write_table(pa.Table.from_pylist(frame_rows), frames_path)

    legacy_row: dict[str, Any] = {name: None for name in METADATA_COLUMNS}
    legacy_row.update(
        {
            "episode_id": episode_id,
            "weld_direction": meta.get("weld_direction"),
            "excitation_type": meta.get("excitation_type"),
            "n_woody_parts": meta.get("n_woody_parts"),
            "junction_names": meta.get("junction_names"),
            "params_fingerprint": meta.get("params_fingerprint"),
            "fruiting_system_params": meta.get("fruiting_system_params"),
            "control_hz": meta.get("control_hz"),
            "timestamp": dataset.manifest.get("created_at"),
            "seed": meta.get("seed"),
            "n_directions": 1,
            "initial_tcp_pos": meta.get("initial_tcp_pos"),
            "initial_tcp_quat": meta.get("initial_tcp_quat"),
            "initial_apple_pos": meta.get("initial_apple_pos"),
            "initial_apple_quat": meta.get("initial_apple_quat"),
            "initial_robot_joint_q": meta.get("initial_robot_joint_q"),
            "fixture_path": meta.get("fixture_path"),
            "fruiting_base_pos": meta.get("fruiting_base_pos"),
            "apple_radius": meta.get("apple_radius"),
            "rod_radii": meta.get("rod_radii"),
            "weld_reference_pos": meta.get("weld_reference_pos"),
            "weld_reference_quat": meta.get("weld_reference_quat"),
            "movement_per_step_m": meta.get("movement_per_step_m"),
            "total_movement_m": meta.get("total_movement_m"),
            "hold_duration_s": meta.get("hold_duration_s"),
            "move_speed_mps": meta.get("move_speed_mps"),
            "skip_return": meta.get("skip_return"),
        }
    )

    meta_path = out / "metadata.parquet"
    meta_rows = [legacy_row]
    if meta_path.exists():
        if not overwrite:
            existing = pq.read_table(meta_path)
            meta_rows = existing.to_pylist() + meta_rows
        else:
            meta_rows = [legacy_row]
    from apple_pick_sim.system_id.trajectory_store import _metadata_table_from_rows

    pq.write_table(_metadata_table_from_rows(meta_rows), meta_path)
    return frames_path
