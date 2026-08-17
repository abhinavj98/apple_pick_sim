"""Convert real-world sys-ID episodes toward batched_sysid_v1.

Bit 1 — metadata: shared pre/post builders → episode metadata JSON.
Bit 2 — trajectory: export a 1×1 ``batched_sysid_v1`` dataset directory
(manifest + ``episodes/s00_d00.parquet``) for trajectory viz / FR3 replay.

Contract: ``docs/superpowers/specs/2026-08-07-real-to-batched-metadata-parity-design.md``,
``docs/superpowers/plans/2026-08-07-real-batched-trajectory-replay-bit2.md``,
``robot_replay/README.md``.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from apple_pick_sim.fruiting_system.params import (
    FruitingSystemParams,
    _spur_attach_fraction_from_ranges,
    _spur_surface_offset_from_ranges,
    _stretch_kw_from_seg_ranges,
    fruiting_params_to_dict,
    load_ranges,
    params_fingerprint,
    rod_params_from_material,
)
from apple_pick_sim.system_id.batched_trajectory_store import SCHEMA_VERSION
from apple_pick_sim.system_id.mmd_features import CMA_WOODY_JUNCTIONS

SIM_JUNCTION_NAMES: tuple[str, str, str] = ("primary_spur", "spur_stem", "stem_apple")
ROD_NAMES: tuple[str, str, str] = ("primary", "spur", "stem")
_JUNCTION_TO_ROD: dict[str, str] = {
    "primary_spur": "primary",
    "spur_stem": "spur",
    "stem_apple": "stem",
}
_ZERO_CHORD_EPS = 1e-12
_DEFAULT_CONTROL_HZ = 15.0
DEFAULT_TARGET_CONTROL_HZ = 30.0
DEFAULT_FT_LPF_CUTOFF_HZ = 10.0
DEFAULT_FT_LPF_ORDER = 4
_TREE_PARQUET_RE = re.compile(r"(?P<tree>s\d+)-d(?P<dir>\d+)\.parquet")


def flat_woody_to_dicts(
    start9: Any,
    end9: Any,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Map flat length-9 woody start/end vectors to named junction xyz arrays."""
    start = np.asarray(start9, dtype=np.float64).reshape(9)
    end = np.asarray(end9, dtype=np.float64).reshape(9)
    starts: dict[str, np.ndarray] = {}
    ends: dict[str, np.ndarray] = {}
    for i, name in enumerate(SIM_JUNCTION_NAMES):
        sl = slice(3 * i, 3 * (i + 1))
        starts[name] = start[sl].copy()
        ends[name] = end[sl].copy()
    return starts, ends


def tag_poses_to_cma_woody(
    branch_pose_4x4: Any,
    spur_pose_4x4: Any,
    apple_pose_4x4: Any,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Branch/Spur/Apple SE(3) translations → CMA woody starts + apple_pos."""
    from apple_pick_sim.system_id.real_post_grasp_plan import pose_4x4_to_pos_quat

    branch_pos, _ = pose_4x4_to_pos_quat(branch_pose_4x4)
    spur_pos, _ = pose_4x4_to_pos_quat(spur_pose_4x4)
    apple_pos, _ = pose_4x4_to_pos_quat(apple_pose_4x4)
    return {
        "primary_spur": np.asarray(branch_pos, dtype=np.float32),
        "spur_stem": np.asarray(spur_pos, dtype=np.float32),
    }, np.asarray(apple_pos, dtype=np.float32)


_TAG_POSE_COLUMNS: tuple[str, str, str] = (
    "branch_pose_4x4",
    "spur_pose_4x4",
    "apple_pose_4x4",
)


def _require_tag_pose_columns(table: Any, path: Path) -> None:
    missing = [name for name in _TAG_POSE_COLUMNS if name not in table.column_names]
    if missing:
        raise ValueError(
            f"{path}: convert Sinkhorn woody/apple requires tag pose columns "
            f"{list(_TAG_POSE_COLUMNS)}; missing {missing}"
        )


def _pose_cell(table: Any, name: str, row_i: int, path: Path) -> Any:
    raw = table.column(name)[row_i].as_py()
    if raw is None:
        raise ValueError(f"{path}: {name} is null at row {row_i}")
    return raw


def rod_directions_from_woody(
    start_by_name: dict[str, np.ndarray],
    end_by_name: dict[str, np.ndarray],
) -> dict[str, tuple[float, float, float]]:
    """Unit rod directions from woody chords; keys are primary/spur/stem."""
    out: dict[str, tuple[float, float, float]] = {}
    for junction, rod in _JUNCTION_TO_ROD.items():
        if junction not in start_by_name or junction not in end_by_name:
            raise ValueError(f"missing woody junction {junction!r}")
        chord = np.asarray(end_by_name[junction], dtype=np.float64).reshape(3) - np.asarray(
            start_by_name[junction], dtype=np.float64
        ).reshape(3)
        norm = float(np.linalg.norm(chord))
        if norm < _ZERO_CHORD_EPS:
            raise ValueError(f"zero woody chord for rod {rod!r} (junction {junction!r})")
        unit = chord / norm
        out[rod] = (float(unit[0]), float(unit[1]), float(unit[2]))
    return out


def split_pregrasp_and_trajectory(step_idx: np.ndarray) -> tuple[int, int]:
    """Return ``(pregrasp_row_index, grasp_row_index)`` from ``step_idx`` column."""
    steps = np.asarray(step_idx, dtype=np.int32).reshape(-1)
    pre = np.where(steps == -1)[0]
    if pre.size != 1:
        raise ValueError(
            "expected exactly one pre-grasp row with step_idx=-1 "
            f"(found {int(pre.size)})"
        )
    pre_i = int(pre[0])
    after = np.where(np.arange(steps.size) > pre_i)[0]
    if after.size == 0:
        raise ValueError("missing grasp/trajectory rows after pre-grasp")
    return pre_i, int(after[0])


def range_midpoint(band: dict[str, Any]) -> float:
    """Return the midpoint of a fixture ``{min, max}`` band."""
    return 0.5 * (float(band["min"]) + float(band["max"]))


def world_wrench_from_ee_logged(ft_ee: Any, tcp_pose_4x4: Any) -> np.ndarray:
    """Rotate logged EE-frame wrench ``[F, τ]`` into world using ``R(tcp)``."""
    ft = np.asarray(ft_ee, dtype=np.float64).reshape(6)
    R = np.asarray(tcp_pose_4x4, dtype=np.float64).reshape(4, 4)[:3, :3]
    out = np.empty(6, dtype=np.float32)
    out[:3] = (R @ ft[:3]).astype(np.float32)
    out[3:] = (R @ ft[3:]).astype(np.float32)
    return out


def build_fruiting_params_from_real(
    *,
    ranges_path: str | Path,
    rod_geometry: dict[str, dict[str, float]],
    directions: dict[str, tuple[float, float, float]],
    apple_radius_m: float | None,
    apple_density_kg_m3: float | None = None,
    use_parts_density: bool = False,
) -> FruitingSystemParams:
    """Assemble ``FruitingSystemParams`` from measured L/r and fixture midpoints.

    When ``use_parts_density`` is True, each rod's ``density_kg_m3`` (and optional
    ``apple_density_kg_m3``) come from ``rod_geometry`` / the apple override;
    Young's modulus, bend damping ratio, and ``num_segments`` still use fixture
    midpoints. Axial stretch uses fixture ``vbd_stretch_force`` on the measured
    geometry (same helper as :func:`~apple_pick_sim.fruiting_system.params.sample_params`).
    """
    ranges = load_ranges(ranges_path)
    rods: dict[str, Any] = {}
    for name in ROD_NAMES:
        if name not in rod_geometry:
            raise ValueError(f"rod_geometry missing {name}")
        if name not in directions:
            raise ValueError(f"directions missing {name}")
        seg = ranges[name]
        if seg is None:
            raise ValueError(f"fixture segment {name!r} is null")
        geo = rod_geometry[name]
        if use_parts_density:
            if "density_kg_m3" not in geo:
                raise ValueError(f"rod_geometry[{name!r}] missing density_kg_m3")
            density = float(geo["density_kg_m3"])
        else:
            density = range_midpoint(seg["density"])
        length = float(geo["length_m"])
        radius = float(geo["radius_m"])
        num_segments = int(round(range_midpoint(seg["num_segments"])))
        rods[name] = rod_params_from_material(
            range_midpoint(seg["youngs_modulus_pa"]),
            range_midpoint(seg["damping_ratio"]),
            length,
            radius,
            density,
            num_segments,
            directions[name],
            **_stretch_kw_from_seg_ranges(
                seg,
                length=length,
                radius=radius,
                density=density,
                num_segments=num_segments,
            ),
        )
    apple_r = (
        float(apple_radius_m)
        if apple_radius_m is not None
        else range_midpoint(ranges["apple"]["radius"])
    )
    if apple_density_kg_m3 is not None:
        apple_d = float(apple_density_kg_m3)
    else:
        apple_d = range_midpoint(ranges["apple"]["density"])
    return FruitingSystemParams(
        primary=rods["primary"],
        secondary=None,
        spur=rods["spur"],
        stem=rods["stem"],
        apple_radius=apple_r,
        apple_density=apple_d,
        topology="t_junction",
        spur_attach_fraction=_spur_attach_fraction_from_ranges(ranges),
        spur_surface_offset=_spur_surface_offset_from_ranges(ranges),
    )


def _load_dataset_metadata(path: Path) -> dict[str, Any]:
    schema = pq.read_metadata(path).schema.to_arrow_schema()
    raw = schema.metadata or {}
    blob = raw.get(b"dataset_metadata")
    if blob is None:
        raise ValueError(f"{path}: missing schema metadata key dataset_metadata")
    text = blob.decode("utf-8") if isinstance(blob, (bytes, bytearray)) else str(blob)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("dataset_metadata must be a JSON object")
    return data


def _as_float_list(value: Any, size: int, *, field: str) -> list[float]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != size:
        raise ValueError(f"{field} must have length {size}, got {arr.size}")
    return [float(x) for x in arr.tolist()]


def _as_4x4(value: Any) -> list[list[float]] | None:
    """Parse a nested or flat 4×4 into row-major ``list[list[float]]``, else None."""
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.size != 16:
        return None
    arr = arr.reshape(4, 4)
    return [[float(arr[i, j]) for j in range(4)] for i in range(4)]


def camera_to_base_4x4_from_dataset_metadata(
    dm: dict[str, Any],
) -> list[list[float]] | None:
    """Extract camera→base SE(3) from real parquet ``dataset_metadata``.

    Prefers top-level ``camera_to_base_4x4_used``, else the first
    ``pre_grasp_geometry.*.camera_to_base_4x4`` snapshot entry.
    """
    parsed = _as_4x4(dm.get("camera_to_base_4x4_used"))
    if parsed is not None:
        return parsed
    pre = dm.get("pre_grasp_geometry")
    if isinstance(pre, dict):
        for snap in pre.values():
            if isinstance(snap, dict):
                parsed = _as_4x4(snap.get("camera_to_base_4x4"))
                if parsed is not None:
                    return parsed
    return None


def _unit_direction(vec: np.ndarray, *, field: str) -> list[float]:
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(v))
    if norm < _ZERO_CHORD_EPS:
        raise ValueError(f"{field}: zero-length vector")
    u = v / norm
    return [float(u[0]), float(u[1]), float(u[2])]


def _resolve_control_hz(dm: dict[str, Any]) -> float:
    """Prefer nested robot/dump control_hz; else default with no raise."""
    source_robot = (dm.get("source_metadata") or {}).get("robot") or {}
    if "control_hz" in source_robot:
        return float(source_robot["control_hz"])
    dump = dm.get("dump") or {}
    if isinstance(dump, dict) and "control_hz" in dump:
        return float(dump["control_hz"])
    summary_robot = ((dm.get("source_metadata_summary") or {}).get("robot_dump") or {})
    if isinstance(summary_robot, dict) and "control_hz" in summary_robot:
        return float(summary_robot["control_hz"])
    import warnings

    warnings.warn(
        f"control_hz missing in dataset_metadata; using default {_DEFAULT_CONTROL_HZ}",
        UserWarning,
        stacklevel=3,
    )
    return float(_DEFAULT_CONTROL_HZ)


def _resolve_episode_id(dm: dict[str, Any], path: Path) -> str:
    eid = dm.get("episode_id")
    if isinstance(eid, str) and eid.strip():
        return eid.strip()
    dump = dm.get("dump") or {}
    if isinstance(dump, dict):
        dump_eid = dump.get("episode_id")
        if isinstance(dump_eid, str) and dump_eid.strip():
            return dump_eid.strip()
    return path.stem


def _grasp_row_index(table: Any) -> int:
    """Index of the grasp/trajectory start row (post-grasp freeze-frame)."""
    names = set(table.column_names)
    if "step_idx" in names:
        step_idx = table.column("step_idx").to_numpy(zero_copy_only=False)
        steps = np.asarray(step_idx, dtype=np.int32).reshape(-1)
        pre = np.where(steps == -1)[0]
        if pre.size == 1:
            after = np.where(np.arange(steps.size) > int(pre[0]))[0]
            if after.size == 0:
                raise ValueError("missing grasp/trajectory rows after pre-grasp")
            return int(after[0])
        if pre.size > 1:
            raise ValueError(
                "expected at most one pre-grasp row with step_idx=-1 "
                f"(found {int(pre.size)})"
            )
    if table.num_rows < 1:
        raise ValueError("episode table has no rows")
    return 0


def _joint_q_from_table(table: Any, grasp_i: int) -> list[float]:
    if "robot_joint_q" in table.column_names:
        return _as_float_list(
            table.column("robot_joint_q")[grasp_i].as_py(), 7, field="robot_joint_q"
        )
    if "joint_pos" in table.column_names:
        return _as_float_list(
            table.column("joint_pos")[grasp_i].as_py(), 7, field="joint_pos"
        )
    raise ValueError("episode table requires robot_joint_q or joint_pos")


def build_episode_metadata_from_real(
    input_path: str | Path,
    *,
    fixture_path: str | Path,
    weld_direction_sign: float = 1.0,
) -> dict[str, Any]:
    """Build batched-style episode metadata from one real-world parquet episode.

    Uses the same pre/post builders as ``example_view_pre_grasp_settle.py`` so
    converted metadata matches the settle-viewer twin (rebuild + grasp init).
    """
    path = Path(input_path)
    fixture = Path(fixture_path)
    if not fixture.is_file():
        raise FileNotFoundError(f"fixture not found: {fixture}")

    # Lazy imports avoid a cycle with real_pre_grasp_params → this module.
    from apple_pick_sim.system_id.real_post_grasp_plan import post_grasp_plan_from_metadata
    from apple_pick_sim.system_id.real_pre_grasp_params import (
        fruiting_params_from_pre_grasp_parquet,
        load_dataset_metadata,
    )

    params, base_pos, _diagnostics = fruiting_params_from_pre_grasp_parquet(
        path, fixture_path=fixture
    )
    if params.apple_radius is None:
        raise ValueError("native pre-grasp params missing apple_radius")
    dm = load_dataset_metadata(path)
    plan = post_grasp_plan_from_metadata(
        dm,
        apple_radius_m=float(params.apple_radius),
        emit_warnings=True,
    )

    table = pq.read_table(path)
    grasp_i = _grasp_row_index(table)
    joint_q = _joint_q_from_table(table, grasp_i)

    sign = float(weld_direction_sign)
    weld_direction = _unit_direction(
        sign * np.asarray(plan.weld_direction, dtype=np.float64),
        field="weld_direction",
    )

    tcp_pos = [float(x) for x in plan.tcp_pos]
    tcp_quat = [float(x) for x in plan.tcp_quat_xyzw]
    apple_pos = [float(x) for x in plan.apple_pos_welded]
    apple_quat = [float(x) for x in plan.apple_quat_xyzw]

    pull_direction = None
    if "excitation_direction" in table.column_names:
        pull_direction = _as_float_list(
            table.column("excitation_direction")[grasp_i].as_py(),
            3,
            field="excitation_direction",
        )

    rod_radii = {
        "primary": float(params.primary.radius),
        "spur": float(params.spur.radius),
        "stem": float(params.stem.radius),
    }

    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": _resolve_episode_id(dm, path),
        "structure_idx": 0,
        "direction_idx": 0,
        "env_idx": 0,
        "pull_direction": pull_direction,
        "params_fingerprint": params_fingerprint(params),
        "fruiting_system_params": fruiting_params_to_dict(params),
        "excitation_type": "quasi_static",
        "control_hz": _resolve_control_hz(dm),
        "seed": None,
        "n_woody_parts": 2,
        "junction_names": list(CMA_WOODY_JUNCTIONS),
        "initial_tcp_pos": tcp_pos,
        "initial_tcp_quat": tcp_quat,
        "initial_apple_pos": apple_pos,
        "initial_apple_quat": apple_quat,
        "initial_robot_joint_q": joint_q,
        "fixture_path": str(fixture.resolve()),
        "fruiting_base_pos": [float(x) for x in base_pos],
        "apple_radius": float(params.apple_radius),
        "rod_radii": rod_radii,
        "weld_direction": weld_direction,
        "weld_reference_pos": list(apple_pos),
        "weld_reference_quat": list(apple_quat),
    }
    cam = camera_to_base_4x4_from_dataset_metadata(dm)
    if cam is not None:
        meta["camera_to_base_4x4"] = cam
    return meta


_REAL_PHASE_NAME_TO_BATCHED: dict[str, str] = {
    "pull": "move_out",
    "move_out": "move_out",
    "hold": "hold",
    "return": "return",
    "pre_weld": "pre_weld",
}


def _scalar_hold_number(value: Any, *, hold_index: Any = None) -> int:
    if hold_index is not None:
        try:
            return int(hold_index)
        except (TypeError, ValueError):
            pass
    if value is None:
        return -1
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return int(arr[0])
    if arr.size > 1:
        return int(np.argmax(arr))
    return -1


def _phase_name_for_row(table: Any, row_i: int) -> str:
    if "phase_name" in table.column_names:
        name = table.column("phase_name")[row_i].as_py()
        if isinstance(name, str) and name.strip():
            return name.strip().lower()
    if "phase" in table.column_names:
        raw = table.column("phase")[row_i].as_py()
        # Real logs: 0=pull, 1=hold. Batched ints use move_out/hold naming via map.
        try:
            p = int(raw)
        except (TypeError, ValueError):
            p = -999
        if p == 0:
            return "pull"
        if p == 1:
            return "hold"
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()
    return "move_out"


def _actions_all_near_zero(table: Any, *, atol: float = 1e-12) -> bool:
    n = table.num_rows
    if n < 1 or "action" not in table.column_names:
        return True
    acts = np.stack([table.column("action")[i].as_py() for i in range(n)], axis=0)
    arr = np.asarray(acts, dtype=np.float64).reshape(n, -1)
    return bool(np.linalg.norm(arr, axis=1).max() <= atol)


def real_action_semantics_label(dataset_metadata: dict[str, Any] | None) -> str | None:
    """Return the human-readable action semantics string from real parquet metadata."""
    if not isinstance(dataset_metadata, dict):
        return None
    dump = dataset_metadata.get("dump")
    if isinstance(dump, dict):
        raw = dump.get("action_semantics")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    layout = dataset_metadata.get("field_layout")
    if isinstance(layout, dict):
        action_layout = layout.get("action")
        if isinstance(action_layout, dict):
            desc = action_layout.get("description")
            if isinstance(desc, str) and desc.strip():
                return desc.strip()
    return None


def is_pose_control_wrench_semantics(dataset_metadata: dict[str, Any] | None) -> bool:
    """True when real ``action`` is a pose-PD wrench, not an EE twist for VIC.

    Real logs (``dump.action_semantics`` / ``field_layout.action``) store the
    commanded pose-control wrench ``[Fx…Tz]``. Batched twist VIC expects EE
    twists; treating wrench as twist is a silent physics bug.
    """
    if not isinstance(dataset_metadata, dict):
        return False
    label = real_action_semantics_label(dataset_metadata)
    if isinstance(label, str) and "wrench" in label.lower():
        return True
    layout = dataset_metadata.get("field_layout")
    if isinstance(layout, dict):
        action_layout = layout.get("action")
        if isinstance(action_layout, dict):
            order = action_layout.get("order")
            if isinstance(order, (list, tuple)):
                tokens = [str(x).strip().lower() for x in order]
                if tokens[:3] == ["fx", "fy", "fz"]:
                    return True
    return False


def real_pose_control_gains(
    dataset_metadata: dict[str, Any] | None,
) -> tuple[list[float], list[float]]:
    """Return ``(Kp, Kd)`` from ``dump.controller_gains`` task prop/deriv gains.

    Each gain is length-6 ``[Fx, Fy, Fz, Tx, Ty, Tz]``.
    """
    if not isinstance(dataset_metadata, dict):
        raise ValueError("dataset_metadata missing dump.controller_gains")
    dump = dataset_metadata.get("dump")
    if not isinstance(dump, dict):
        raise ValueError("dataset_metadata missing dump.controller_gains")
    gains = dump.get("controller_gains")
    if not isinstance(gains, dict):
        raise ValueError("dataset_metadata missing dump.controller_gains")
    if "task_prop_gains" not in gains:
        raise ValueError("dump.controller_gains missing task_prop_gains")
    if "task_deriv_gains" not in gains:
        raise ValueError("dump.controller_gains missing task_deriv_gains")
    kp = _as_float_list(gains["task_prop_gains"], 6, field="task_prop_gains")
    kd = _as_float_list(gains["task_deriv_gains"], 6, field="task_deriv_gains")
    return kp, kd


def _vic_pose_action_row(
    target_pose_4x4_flat16: Any,
    kp: list[float],
    kd: list[float],
) -> np.ndarray:
    """Pack ``[pos(3), quat_wxyz(4), Kp(6), Kd(6)]`` from a flat 4×4 target pose."""
    from apple_pick_sim.system_id.real_post_grasp_plan import pose_4x4_to_pos_quat

    pos, quat_xyzw = pose_4x4_to_pos_quat(target_pose_4x4_flat16)
    quat_wxyz = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])
    return np.asarray([*pos, *quat_wxyz, *kp, *kd], dtype=np.float32)


def decimation_window_size(source_hz: float, target_hz: float) -> int:
    """Samples per output frame; at least 1 (no upsample)."""
    source = float(source_hz)
    target = float(target_hz)
    if source <= 0.0 or target <= 0.0:
        raise ValueError(f"source_hz and target_hz must be positive, got {source_hz}, {target_hz}")
    return max(1, int(round(source / target)))


def last_sample_indices(n_frames: int, window: int) -> np.ndarray:
    """Index of the last source sample in each complete ``window``-sample block."""
    if window < 1:
        raise ValueError(f"window must be positive, got {window}")
    n_out = int(n_frames) // int(window)
    if n_out < 1:
        raise ValueError(
            f"need at least one full window of {window} samples, got n_frames={n_frames}"
        )
    return (np.arange(n_out, dtype=np.int64) + 1) * int(window) - 1


def block_mean_downsample(values: np.ndarray, window: int) -> np.ndarray:
    """Mean each complete block of ``window`` rows; drop a trailing remainder."""
    if window < 1:
        raise ValueError(f"window must be positive, got {window}")
    x = np.asarray(values, dtype=np.float64)
    squeeze = False
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    if x.ndim != 2:
        raise ValueError(f"values must be 1D or 2D, got shape {x.shape}")
    n_out = x.shape[0] // int(window)
    if n_out < 1:
        raise ValueError(
            f"need at least one full window of {window} samples, got n_frames={x.shape[0]}"
        )
    trimmed = x[: n_out * int(window)]
    out = trimmed.reshape(n_out, int(window), x.shape[1]).mean(axis=1)
    if squeeze:
        out = out[:, 0]
    return out.astype(np.float32)


def zero_phase_lowpass(
    values: np.ndarray,
    *,
    source_hz: float,
    cutoff_hz: float,
    order: int = DEFAULT_FT_LPF_ORDER,
) -> np.ndarray:
    """Forward-backward Butterworth; skip when cutoff is unusable or the series is too short."""
    filtered, _status = zero_phase_lowpass_with_status(
        values, source_hz=source_hz, cutoff_hz=cutoff_hz, order=order
    )
    return filtered


def zero_phase_lowpass_with_status(
    values: np.ndarray,
    *,
    source_hz: float,
    cutoff_hz: float,
    order: int = DEFAULT_FT_LPF_ORDER,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return filtered values plus ``applied`` / ``skip_reason`` provenance."""
    x = np.asarray(values, dtype=np.float64)
    if cutoff_hz <= 0.0 or source_hz <= 0.0:
        return x.copy(), {"applied": False, "skip_reason": "nonpositive_hz"}
    nyquist = 0.5 * float(source_hz)
    if float(cutoff_hz) >= nyquist:
        return x.copy(), {"applied": False, "skip_reason": "cutoff_at_or_above_nyquist"}
    from scipy.signal import butter, filtfilt

    sos_order = int(order)
    if sos_order < 1:
        raise ValueError(f"filter order must be positive, got {order}")
    b, a = butter(sos_order, float(cutoff_hz) / nyquist, btype="low")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if x.shape[0] <= padlen:
        return x.copy(), {"applied": False, "skip_reason": "series_shorter_than_padlen"}
    filtered = filtfilt(b, a, x, axis=0)
    return np.asarray(filtered, dtype=np.float64), {"applied": True}


@dataclasses.dataclass
class _ConvertedEpisode:
    path: Path
    direction_idx: int
    traj: Any
    episode_meta: dict[str, Any]
    ft_filter: dict[str, Any]
    junction_names: list[str]
    pull_direction: list[float] | None
    fruiting_base_pos: list[float]
    max_hold_number: int
    dm: dict[str, Any]


def _require_dump_direction_index(path: Path, dm: dict[str, Any], direction_idx: int) -> None:
    dump = dm.get("dump") or {}
    if not isinstance(dump, dict) or "direction_index" not in dump:
        return
    logged = int(dump["direction_index"])
    if logged != int(direction_idx):
        raise ValueError(
            f"{path}: dump.direction_index={logged} does not match filename "
            f"s*-d{int(direction_idx):02d}"
        )


def _discover_tree_parquets(input_dir: Path) -> list[tuple[Path, int]]:
    entries: list[tuple[Path, int]] = []
    tree_prefix: str | None = None
    seen_dirs: set[int] = set()
    for path in sorted(input_dir.glob("*.parquet")):
        match = _TREE_PARQUET_RE.fullmatch(path.name)
        if match is None:
            continue
        tree = match.group("tree")
        direction_idx = int(match.group("dir"))
        if tree_prefix is None:
            tree_prefix = tree
        elif tree != tree_prefix:
            raise ValueError(
                f"{input_dir}: mixed tree prefixes {tree_prefix!r} and {tree!r}"
            )
        if direction_idx in seen_dirs:
            raise ValueError(
                f"{input_dir}: duplicate direction index d{direction_idx:02d} in filenames"
            )
        seen_dirs.add(direction_idx)
        entries.append((path, direction_idx))
    if not entries:
        raise ValueError(f"{input_dir}: no compiled sXX-dNN.parquet files found")
    return entries


def _rod_geometry_signature(params_blob: Any) -> tuple[Any, ...]:
    if isinstance(params_blob, str):
        data = json.loads(params_blob)
    elif isinstance(params_blob, dict):
        data = params_blob
    else:
        raise ValueError("fruiting_system_params must be dict or JSON string")
    sig: list[Any] = []
    for rod in ROD_NAMES:
        block = data.get(rod) or {}
        sig.append(
            (
                float(block.get("length", 0.0)),
                float(block.get("radius", 0.0)),
                int(block.get("num_segments", 0)),
            )
        )
    return tuple(sig)


def _canonicalize_tree_geometry(
    episodes: list[_ConvertedEpisode],
    *,
    base_pos_tolerance_m: float,
) -> tuple[list[float], dict[str, Any], str]:
    if not episodes:
        raise ValueError("canonicalize_tree_geometry requires at least one episode")
    base_arr = np.stack(
        [np.asarray(ep.fruiting_base_pos, dtype=np.float64).reshape(3) for ep in episodes],
        axis=0,
    )
    mean_base = base_arr.mean(axis=0)
    spread = np.max(np.abs(base_arr - mean_base.reshape(1, 3)), axis=0)
    if float(np.max(spread)) > float(base_pos_tolerance_m):
        raise ValueError(
            "fruiting_base_pos spread exceeds tolerance "
            f"{float(base_pos_tolerance_m)} m (max axis delta {float(np.max(spread)):.6f} m)"
        )
    ref = min(episodes, key=lambda ep: ep.direction_idx)
    ref_junctions = list(ref.junction_names)
    ref_sig = _rod_geometry_signature(ref.episode_meta.get("fruiting_system_params"))
    for ep in episodes:
        if list(ep.junction_names) != ref_junctions:
            raise ValueError(
                f"{ep.path}: junction_names {ep.junction_names!r} != {ref_junctions!r}"
            )
        if _rod_geometry_signature(ep.episode_meta.get("fruiting_system_params")) != ref_sig:
            raise ValueError(f"{ep.path}: rod geometry mismatch within tree folder")
    mean_list = [float(x) for x in mean_base.tolist()]
    canonical_params = ref.episode_meta.get("fruiting_system_params")
    canonical_fp = str(ref.episode_meta.get("params_fingerprint"))
    for ep in episodes:
        ep.episode_meta = {
            **ep.episode_meta,
            "fruiting_base_pos": list(mean_list),
            "fruiting_system_params": canonical_params,
            "params_fingerprint": canonical_fp,
        }
        ep.fruiting_base_pos = list(mean_list)
    return mean_list, canonical_params, canonical_fp


def _manifest_sim_config_from_fixture(
    *,
    fixture_path: Path,
    topology_seed: int,
    fruiting_base_pos: tuple[float, float, float],
    control_hz: float,
) -> dict[str, Any]:
    from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
        BatchedHeterogeneousCoupledSimConfig,
        ObsConfig,
    )
    from apple_pick_sim.fruiting_system.params import load_ranges, parse_sim_build
    from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import ImpedanceGains
    from apple_pick_sim.system_id.manifest_sim_config import sim_config_to_manifest_dict

    ranges = load_ranges(fixture_path)
    sb = parse_sim_build(ranges)
    vic_gains = None
    joint_angular_kd: dict[str, float] = {}
    joint_linear_kd: dict[str, float] = {}
    joint_angular_kp: dict[str, float] = {}
    joint_linear_kp: dict[str, float] = {}
    joint_damping_ratio: float | None = None
    if sb is not None:
        vic_gains = ImpedanceGains(
            linear_k=sb.vic_gains.linear_k,
            linear_d=sb.vic_gains.linear_d,
            angular_k=sb.vic_gains.angular_k,
            angular_d=sb.vic_gains.angular_d,
        )
        joint_angular_kd = dict(sb.joint_angular_kd_overrides)
        joint_linear_kd = dict(sb.joint_linear_kd_overrides)
        joint_angular_kp = dict(sb.joint_angular_kp_overrides)
        joint_linear_kp = dict(sb.joint_linear_kp_overrides)
        joint_damping_ratio = sb.joint_damping_ratio

    gym_cfg = BatchedHeterogeneousCoupledSimConfig.gym_defaults(num_envs=1)
    controller = dataclasses.replace(
        gym_cfg.controller,
        mode="vic_pose",
        action_dim=19,
        linear_speed=1.0,
        angular_speed=1.0,
    )
    if vic_gains is not None:
        controller = dataclasses.replace(controller, vic_gains=vic_gains)
    runtime = dataclasses.replace(gym_cfg.runtime, control_hz=float(control_hz))
    config = dataclasses.replace(
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
        ),
        scene=dataclasses.replace(
            gym_cfg.scene,
            settle_substeps=5000,
            settle_quiet_every=300,
            settle_gravity_ramp=False,
            post_grasp_settle_substeps=500,
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
    return sim_config_to_manifest_dict(config)


def _build_real_episode(
    input_path: str | Path,
    *,
    fixture_path: str | Path,
    direction_idx: int,
    weld_direction_sign: float = 1.0,
    allow_zero_action: bool = False,
    control_hz: float | None = None,
    ft_lpf_hz: float = DEFAULT_FT_LPF_CUTOFF_HZ,
    ft_lpf_order: int = DEFAULT_FT_LPF_ORDER,
) -> _ConvertedEpisode:
    """Convert one real parquet into trajectory + metadata (no manifest write)."""
    from apple_pick_sim.system_id.batched_trajectory_store import BatchedEpisodeWriter
    from apple_pick_sim.system_id.real_post_grasp_plan import pose_4x4_to_pos_quat

    path = Path(input_path)
    fixture = Path(fixture_path)

    episode_meta = build_episode_metadata_from_real(
        path, fixture_path=fixture, weld_direction_sign=weld_direction_sign
    )
    episode_meta = {
        **episode_meta,
        "structure_idx": 0,
        "direction_idx": int(direction_idx),
        "env_idx": int(direction_idx),
    }
    params_blob = episode_meta.get("fruiting_system_params")
    if isinstance(params_blob, dict):
        episode_meta = {
            **episode_meta,
            "fruiting_system_params": json.dumps(params_blob, sort_keys=True),
        }
    table = pq.read_table(path)
    _require_tag_pose_columns(table, path)

    dm_raw = pq.read_metadata(path).schema.to_arrow_schema().metadata or {}
    dm_blob = dm_raw.get(b"dataset_metadata")
    dm: dict[str, Any] = {}
    if dm_blob is not None:
        dm = json.loads(
            dm_blob.decode("utf-8") if isinstance(dm_blob, (bytes, bytearray)) else str(dm_blob)
        )
    pack_vic_pose = is_pose_control_wrench_semantics(dm)
    kp: list[float] | None = None
    kd: list[float] | None = None
    if pack_vic_pose:
        if "target_pose_4x4" not in table.column_names:
            raise ValueError(
                f"{path}: pose-control wrench semantics require target_pose_4x4 "
                "to pack 19D vic_pose actions"
            )
        kp, kd = real_pose_control_gains(dm)
        label = real_action_semantics_label(dm) or "pose-control wrench"
        episode_meta = {
            **episode_meta,
            "action_semantics": label,
            "action_compatible_with_vic_twist": False,
            "action_dim": 19,
            "action_layout": "vic_pose_v1",
        }
    has_drive_fill = isinstance(dm.get("drive_fill"), dict)
    if (
        not pack_vic_pose
        and _actions_all_near_zero(table)
        and not allow_zero_action
        and not has_drive_fill
    ):
        raise ValueError(
            f"{path}: action column is all zeros (real-replay-action-zero). "
            "Use a pose-control log with target_pose_4x4 (vic_pose pack), a fixed "
            "real parquet (e.g. s02-d00.parquet), fill via "
            "robot_replay/fill_actions_from_tcp_velocity.py, or set allow_zero_action=True."
        )

    source_hz = float(episode_meta["control_hz"])
    if source_hz <= 0:
        raise ValueError(f"invalid source control_hz={source_hz}")
    target_hz = DEFAULT_TARGET_CONTROL_HZ if control_hz is None else float(control_hz)
    if target_hz <= 0:
        raise ValueError(f"invalid control_hz={target_hz}")
    window = decimation_window_size(source_hz, target_hz)
    output_hz = source_hz if window == 1 else target_hz

    n_src = int(table.num_rows)
    phases: list[str] = []
    actions: list[np.ndarray] = []
    tcp_vels: list[np.ndarray] = []
    fts: list[np.ndarray] = []
    raw_fts: list[np.ndarray] = []
    tcp_pos_rows: list[np.ndarray] = []
    tcp_quats: list[np.ndarray] = []
    apple_pos_rows: list[np.ndarray] = []
    apple_quats: list[np.ndarray] = []
    joint_qs: list[np.ndarray] = []
    woody_rows: list[dict[str, np.ndarray]] = []
    excitations: list[np.ndarray] = []
    amplitudes: list[float] = []
    hold_numbers: list[int] = []
    step_indices: list[int] = []

    for i in range(n_src):
        phase_raw = _phase_name_for_row(table, i)
        phases.append(_REAL_PHASE_NAME_TO_BATCHED.get(phase_raw, "move_out"))

        if pack_vic_pose:
            assert kp is not None and kd is not None
            action = _vic_pose_action_row(
                table.column("target_pose_4x4")[i].as_py(), kp, kd
            )
        else:
            action = np.asarray(
                table.column("action")[i].as_py(), dtype=np.float32
            ).reshape(6)
        actions.append(action)
        tcp_vels.append(
            np.asarray(table.column("tcp_velocity")[i].as_py(), dtype=np.float32).reshape(6)
        )
        ft = np.asarray(table.column("ft_wrist")[i].as_py(), dtype=np.float32).reshape(6)
        raw_ft = ft
        if "ft_wrist_raw" in table.column_names:
            raw_ft = np.asarray(
                table.column("ft_wrist_raw")[i].as_py(), dtype=np.float32
            ).reshape(6)

        tcp_pos = np.asarray(table.column("tcp_pos")[i].as_py(), dtype=np.float32).reshape(3)
        if pack_vic_pose:
            if "tcp_pose_4x4" not in table.column_names:
                raise ValueError(
                    f"{path}: pose-control wrench semantics require tcp_pose_4x4 "
                    "to rotate ft_wrist into the TCP world frame"
                )
            tcp_pose = table.column("tcp_pose_4x4")[i].as_py()
            ft = world_wrench_from_ee_logged(ft, tcp_pose)
            raw_ft = world_wrench_from_ee_logged(raw_ft, tcp_pose)
            _, tcp_quat = pose_4x4_to_pos_quat(tcp_pose)
        elif "tcp_pose_4x4" in table.column_names:
            _, tcp_quat = pose_4x4_to_pos_quat(table.column("tcp_pose_4x4")[i].as_py())
        else:
            tcp_quat = (0.0, 0.0, 0.0, 1.0)
        fts.append(np.asarray(ft, dtype=np.float32).reshape(6))
        raw_fts.append(np.asarray(raw_ft, dtype=np.float32).reshape(6))
        tcp_pos_rows.append(tcp_pos)
        tcp_quats.append(np.asarray(tcp_quat, dtype=np.float32))

        branch_pose = _pose_cell(table, "branch_pose_4x4", i, path)
        spur_pose = _pose_cell(table, "spur_pose_4x4", i, path)
        apple_pose = _pose_cell(table, "apple_pose_4x4", i, path)
        woody_starts, apple_pos = tag_poses_to_cma_woody(branch_pose, spur_pose, apple_pose)
        _, apple_quat = pose_4x4_to_pos_quat(apple_pose)
        woody_rows.append(woody_starts)
        apple_pos_rows.append(np.asarray(apple_pos, dtype=np.float32))
        apple_quats.append(np.asarray(apple_quat, dtype=np.float32))

        if "joint_pos" in table.column_names:
            joint_q = np.asarray(table.column("joint_pos")[i].as_py(), dtype=np.float32).reshape(7)
        elif "robot_joint_q" in table.column_names:
            joint_q = np.asarray(
                table.column("robot_joint_q")[i].as_py(), dtype=np.float32
            ).reshape(7)
        else:
            joint_q = np.zeros(7, dtype=np.float32)
        joint_qs.append(joint_q)

        if "excitation_direction" in table.column_names:
            exc = np.asarray(
                table.column("excitation_direction")[i].as_py(), dtype=np.float32
            ).reshape(3)
        else:
            pull = episode_meta.get("pull_direction") or [0.0, -1.0, 0.0]
            exc = np.asarray(pull, dtype=np.float32).reshape(3)
        excitations.append(exc)

        amp = 0.0
        if "amplitude_m" in table.column_names:
            amp_raw = table.column("amplitude_m")[i].as_py()
            amp = float(np.asarray(amp_raw, dtype=np.float64).reshape(-1)[0])
        amplitudes.append(amp)

        hold_idx = None
        if "hold_index" in table.column_names:
            hold_idx = table.column("hold_index")[i].as_py()
        hold_raw = None
        if "hold_number" in table.column_names:
            hold_raw = table.column("hold_number")[i].as_py()
        hold_numbers.append(_scalar_hold_number(hold_raw, hold_index=hold_idx))

        step_idx = i
        if "step_idx" in table.column_names:
            try:
                step_idx = int(table.column("step_idx")[i].as_py())
            except (TypeError, ValueError):
                step_idx = i
        step_indices.append(step_idx)

    ft_unfiltered = np.stack(fts, axis=0)
    raw_unfiltered = np.stack(raw_fts, axis=0)
    vel_unfiltered = np.stack(tcp_vels, axis=0)
    ft_lpf, lpf_info = zero_phase_lowpass_with_status(
        ft_unfiltered,
        source_hz=source_hz,
        cutoff_hz=float(ft_lpf_hz),
        order=int(ft_lpf_order),
    )
    ft_out = block_mean_downsample(ft_unfiltered, window)
    raw_ft_out = block_mean_downsample(raw_unfiltered, window)
    vel_out = block_mean_downsample(vel_unfiltered, window)
    ft_lpf_out = block_mean_downsample(ft_lpf, window)
    pick = last_sample_indices(n_src, window)

    ft_filter = {
        "method": "butterworth_filtfilt",
        "cutoff_hz": float(ft_lpf_hz),
        "order": int(ft_lpf_order),
        "source_hz": source_hz,
        "target_hz": output_hz,
        "tare": "ema_minus_ema",
        "column": "ft_wrist_lpf",
        "window": int(window),
        "applied": bool(lpf_info.get("applied")),
    }
    if not ft_filter["applied"] and lpf_info.get("skip_reason"):
        ft_filter["skip_reason"] = str(lpf_info["skip_reason"])
    episode_meta = {
        **episode_meta,
        "control_hz": float(output_hz),
        "ft_filter": dict(ft_filter),
    }

    traj = BatchedEpisodeWriter(episode_id=str(episode_meta["episode_id"]))
    junction_names = list(episode_meta["junction_names"])
    for out_i, src_i in enumerate(pick.tolist()):
        obs = {
            "excitation_type": 0,
            "excitation_direction": excitations[int(src_i)],
            "tcp_velocity": vel_out[out_i],
            "ft_wrist": ft_out[out_i],
            "ft_wrist_lpf": ft_lpf_out[out_i],
            "raw_ft_wrist": raw_ft_out[out_i],
            "tcp_pos": tcp_pos_rows[int(src_i)],
            "apple_pos": apple_pos_rows[int(src_i)],
            "tcp_quat": tcp_quats[int(src_i)],
            "apple_quat": apple_quats[int(src_i)],
            "robot_joint_q": joint_qs[int(src_i)],
            "woody_part_start_pos": woody_rows[int(src_i)],
            "woody_part_force": np.zeros(0, dtype=np.float32),
        }
        traj.record_step(
            step_idx=int(step_indices[int(src_i)]),
            sim_time=float(out_i) / float(output_hz),
            phase=phases[int(src_i)],
            amplitude_m=amplitudes[int(src_i)],
            action=actions[int(src_i)],
            obs=obs,
            stable=True,
            hold_number=hold_numbers[int(src_i)],
        )

    max_hold = max((int(h) for h in hold_numbers), default=-1)
    fruiting_base = [float(x) for x in episode_meta.get("fruiting_base_pos") or [0.0, 0.0, 0.0]]
    pull_dir = episode_meta.get("pull_direction")
    return _ConvertedEpisode(
        path=path,
        direction_idx=int(direction_idx),
        traj=traj,
        episode_meta=episode_meta,
        ft_filter=dict(ft_filter),
        junction_names=junction_names,
        pull_direction=list(pull_dir) if pull_dir is not None else None,
        fruiting_base_pos=fruiting_base,
        max_hold_number=max_hold,
        dm=dm,
    )


def _assert_identical_ft_filters(episodes: list[_ConvertedEpisode]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("ft_filter check requires at least one episode")
    ref = episodes[0].ft_filter
    for ep in episodes[1:]:
        if ep.ft_filter != ref:
            raise ValueError(f"{ep.path}: ft_filter differs across directions in one tree")
    return dict(ref)


def _write_batched_manifest(
    out: Path,
    *,
    fixture: Path,
    episodes: list[_ConvertedEpisode],
    command_argv: list[str],
    source_real_parquet: str | None = None,
    source_real_parquets: list[str] | None = None,
    num_directions: int,
    topology_seed: int | None = None,
    sim_config: dict[str, Any] | None = None,
    n_holds: int | None = None,
) -> None:
    from apple_pick_sim.system_id.batched_trajectory_store import write_manifest

    ref = min(episodes, key=lambda ep: ep.direction_idx)
    ft_filter = _assert_identical_ft_filters(episodes)
    output_hz = float(ref.episode_meta["control_hz"])
    max_steps = max(int(ep.traj.n_frames) for ep in episodes)
    structures = [
        {
            "structure_idx": 0,
            "params_fingerprint": ref.episode_meta.get("params_fingerprint"),
            "junction_names": list(ref.junction_names),
            "n_woody_parts": int(ref.episode_meta.get("n_woody_parts") or 2),
        }
    ]
    manifest_episodes = [
        {
            "structure_idx": 0,
            "direction_idx": int(ep.direction_idx),
            "env_idx": int(ep.direction_idx),
            "filename": f"episodes/s00_d{int(ep.direction_idx):02d}.parquet",
            "episode_id": ep.episode_meta["episode_id"],
            "pull_direction": ep.pull_direction,
            "n_frames": ep.traj.n_frames,
            "excluded": False,
            "excluded_reason": None,
        }
        for ep in sorted(episodes, key=lambda item: item.direction_idx)
    ]
    seed_raw = ref.episode_meta.get("seed")
    seed_fallback = int(topology_seed) if topology_seed is not None else 0
    collection: dict[str, Any] = {
        "seed": int(seed_raw) if seed_raw is not None else seed_fallback,
        "ranges_path": str(fixture.resolve()),
        "control_hz": output_hz,
        "ft_filter": dict(ft_filter),
        "num_structures": 1,
        "num_directions": int(num_directions),
        "max_steps": int(max_steps),
        "drive_fill": ref.dm.get("drive_fill"),
    }
    if topology_seed is not None:
        collection["topology_seed"] = int(topology_seed)
    if source_real_parquet is not None:
        collection["source_real_parquet"] = source_real_parquet
    if source_real_parquets is not None:
        collection["source_real_parquets"] = list(source_real_parquets)
    if sim_config is not None:
        collection["sim_config"] = sim_config
    if n_holds is not None:
        collection["n_holds"] = int(n_holds)
    if ref.episode_meta.get("action_layout") == "vic_pose_v1":
        collection["action_semantics"] = ref.episode_meta.get("action_semantics")
        collection["action_compatible_with_vic_twist"] = False
        collection["action_dim"] = int(ref.episode_meta.get("action_dim") or 19)
        collection["action_layout"] = "vic_pose_v1"
    write_manifest(
        out,
        command_argv=list(command_argv),
        collection=collection,
        structures=structures,
        episodes=manifest_episodes,
        overwrite=True,
    )


def export_real_episode_to_batched_dataset(
    input_path: str | Path,
    *,
    fixture_path: str | Path,
    output_dir: str | Path,
    weld_direction_sign: float = 1.0,
    overwrite: bool = False,
    allow_zero_action: bool = False,
    command_argv: list[str] | None = None,
    control_hz: float | None = None,
    ft_lpf_hz: float = DEFAULT_FT_LPF_CUTOFF_HZ,
    ft_lpf_order: int = DEFAULT_FT_LPF_ORDER,
) -> Path:
    """Write a 1×1 ``batched_sysid_v1`` dataset from one real-world parquet."""
    from apple_pick_sim.system_id.batched_trajectory_store import episode_filename

    path = Path(input_path)
    out = Path(output_dir)
    fixture = Path(fixture_path)
    converted = _build_real_episode(
        path,
        fixture_path=fixture,
        direction_idx=0,
        weld_direction_sign=weld_direction_sign,
        allow_zero_action=allow_zero_action,
        control_hz=control_hz,
        ft_lpf_hz=ft_lpf_hz,
        ft_lpf_order=ft_lpf_order,
    )
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"output_dir not empty (pass overwrite=True): {out}")
    out.mkdir(parents=True, exist_ok=True)
    ep_rel = episode_filename(0, 0)
    converted.traj.save(out / ep_rel, converted.episode_meta)
    _write_batched_manifest(
        out,
        fixture=fixture,
        episodes=[converted],
        command_argv=list(command_argv or ["export_real_episode_to_batched_dataset"]),
        source_real_parquet=str(path.resolve()),
        num_directions=1,
    )
    return out


def export_real_tree_folder_to_batched_dataset(
    input_dir: str | Path,
    *,
    fixture_path: str | Path,
    output_dir: str | Path,
    weld_direction_sign: float = 1.0,
    overwrite: bool = False,
    allow_zero_action: bool = False,
    command_argv: list[str] | None = None,
    control_hz: float | None = None,
    ft_lpf_hz: float = DEFAULT_FT_LPF_CUTOFF_HZ,
    ft_lpf_order: int = DEFAULT_FT_LPF_ORDER,
    base_pos_tolerance_m: float = 5e-3,
) -> Path:
    """Write a 1×N ``batched_sysid_v1`` dataset from one tree folder of parquets."""
    from apple_pick_sim.system_id.batched_trajectory_store import episode_filename

    src_dir = Path(input_dir)
    out = Path(output_dir)
    fixture = Path(fixture_path)
    discovered = _discover_tree_parquets(src_dir)
    converted: list[_ConvertedEpisode] = []
    for path, direction_idx in discovered:
        dm = _load_dataset_metadata(path)
        _require_dump_direction_index(path, dm, direction_idx)
        converted.append(
            _build_real_episode(
                path,
                fixture_path=fixture,
                direction_idx=direction_idx,
                weld_direction_sign=weld_direction_sign,
                allow_zero_action=allow_zero_action,
                control_hz=control_hz,
                ft_lpf_hz=ft_lpf_hz,
                ft_lpf_order=ft_lpf_order,
            )
        )
    _canonicalize_tree_geometry(converted, base_pos_tolerance_m=float(base_pos_tolerance_m))
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"output_dir not empty (pass overwrite=True): {out}")
    out.mkdir(parents=True, exist_ok=True)
    for ep in converted:
        rel = episode_filename(0, ep.direction_idx)
        ep.traj.save(out / rel, ep.episode_meta)
    num_directions = max(ep.direction_idx for ep in converted) + 1
    n_holds = max(ep.max_hold_number for ep in converted) + 1
    ref = min(converted, key=lambda ep: ep.direction_idx)
    topology_seed = int(ref.episode_meta.get("seed") or 0)
    mean_base = tuple(float(x) for x in ref.fruiting_base_pos)
    output_hz = float(ref.episode_meta["control_hz"])
    sim_config = _manifest_sim_config_from_fixture(
        fixture_path=fixture,
        topology_seed=topology_seed,
        fruiting_base_pos=mean_base,
        control_hz=output_hz,
    )
    _write_batched_manifest(
        out,
        fixture=fixture,
        episodes=converted,
        command_argv=list(command_argv or ["export_real_tree_folder_to_batched_dataset"]),
        source_real_parquets=[str(ep.path.resolve()) for ep in converted],
        num_directions=num_directions,
        topology_seed=topology_seed,
        sim_config=sim_config,
        n_holds=n_holds,
    )
    return out
