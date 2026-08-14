"""Convert real-world sys-ID episodes toward batched_sysid_v1.

Bit 1 — metadata: shared pre/post builders → episode metadata JSON.
Bit 2 — trajectory: export a 1×1 ``batched_sysid_v1`` dataset directory
(manifest + ``episodes/s00_d00.parquet``) for trajectory viz / FR3 replay.

Contract: ``docs/superpowers/specs/2026-08-07-real-to-batched-metadata-parity-design.md``,
``docs/superpowers/plans/2026-08-07-real-batched-trajectory-replay-bit2.md``,
``robot_replay/README.md``.
"""

from __future__ import annotations

import json
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


def export_real_episode_to_batched_dataset(
    input_path: str | Path,
    *,
    fixture_path: str | Path,
    output_dir: str | Path,
    weld_direction_sign: float = 1.0,
    overwrite: bool = False,
    allow_zero_action: bool = False,
    command_argv: list[str] | None = None,
) -> Path:
    """Write a 1×1 ``batched_sysid_v1`` dataset from one real-world parquet.

    Uses bit-1 metadata builders for episode schema metadata and maps trajectory
    rows into batched frame columns. Sinkhorn woody/apple come from
    ``branch_pose_4x4`` / ``spur_pose_4x4`` / ``apple_pose_4x4`` translations
    (source ``woody_part_*`` packing is ignored).

    When real ``action`` is a pose-control wrench, packs a 19D ``vic_pose``
    action from ``target_pose_4x4`` + ``dump.controller_gains`` instead of
    copying the wrench (which is not an EE twist for ``mode=vic``).
    """
    from apple_pick_sim.system_id.batched_trajectory_store import (
        BatchedEpisodeWriter,
        episode_filename,
        write_manifest,
    )
    from apple_pick_sim.system_id.real_post_grasp_plan import pose_4x4_to_pos_quat

    path = Path(input_path)
    out = Path(output_dir)
    fixture = Path(fixture_path)

    episode_meta = build_episode_metadata_from_real(
        path, fixture_path=fixture, weld_direction_sign=weld_direction_sign
    )
    # Match sim-collected batched_sysid_v1: fruiting_system_params is a JSON string.
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
    # Pose-wrench logs pack from target_pose_4x4; logged action/wrench is unused.
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

    control_hz = float(episode_meta["control_hz"])
    if control_hz <= 0:
        raise ValueError(f"invalid control_hz={control_hz}")

    traj = BatchedEpisodeWriter(episode_id=str(episode_meta["episode_id"]))
    junction_names = list(episode_meta["junction_names"])

    for i in range(table.num_rows):
        phase_raw = _phase_name_for_row(table, i)
        phase = _REAL_PHASE_NAME_TO_BATCHED.get(phase_raw, "move_out")

        if pack_vic_pose:
            assert kp is not None and kd is not None
            action = _vic_pose_action_row(
                table.column("target_pose_4x4")[i].as_py(), kp, kd
            )
        else:
            action = np.asarray(
                table.column("action")[i].as_py(), dtype=np.float32
            ).reshape(6)
        tcp_vel = np.asarray(table.column("tcp_velocity")[i].as_py(), dtype=np.float32).reshape(6)
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

        branch_pose = _pose_cell(table, "branch_pose_4x4", i, path)
        spur_pose = _pose_cell(table, "spur_pose_4x4", i, path)
        apple_pose = _pose_cell(table, "apple_pose_4x4", i, path)
        woody_starts, apple_pos = tag_poses_to_cma_woody(branch_pose, spur_pose, apple_pose)
        _, apple_quat = pose_4x4_to_pos_quat(apple_pose)

        if "joint_pos" in table.column_names:
            joint_q = np.asarray(table.column("joint_pos")[i].as_py(), dtype=np.float32).reshape(7)
        elif "robot_joint_q" in table.column_names:
            joint_q = np.asarray(
                table.column("robot_joint_q")[i].as_py(), dtype=np.float32
            ).reshape(7)
        else:
            joint_q = np.zeros(7, dtype=np.float32)

        if "excitation_direction" in table.column_names:
            exc = np.asarray(
                table.column("excitation_direction")[i].as_py(), dtype=np.float32
            ).reshape(3)
        else:
            pull = episode_meta.get("pull_direction") or [0.0, -1.0, 0.0]
            exc = np.asarray(pull, dtype=np.float32).reshape(3)

        amp = 0.0
        if "amplitude_m" in table.column_names:
            amp_raw = table.column("amplitude_m")[i].as_py()
            amp = float(np.asarray(amp_raw, dtype=np.float64).reshape(-1)[0])

        hold_idx = None
        if "hold_index" in table.column_names:
            hold_idx = table.column("hold_index")[i].as_py()
        hold_raw = None
        if "hold_number" in table.column_names:
            hold_raw = table.column("hold_number")[i].as_py()
        hold_number = _scalar_hold_number(hold_raw, hold_index=hold_idx)

        step_idx = i
        if "step_idx" in table.column_names:
            try:
                step_idx = int(table.column("step_idx")[i].as_py())
            except (TypeError, ValueError):
                step_idx = i

        obs = {
            "excitation_type": 0,
            "excitation_direction": exc,
            "tcp_velocity": tcp_vel,
            "ft_wrist": ft,
            "raw_ft_wrist": raw_ft,
            "tcp_pos": tcp_pos,
            "apple_pos": apple_pos,
            "tcp_quat": np.asarray(tcp_quat, dtype=np.float32),
            "apple_quat": np.asarray(apple_quat, dtype=np.float32),
            "robot_joint_q": joint_q,
            "woody_part_start_pos": woody_starts,
            "woody_part_force": np.zeros(0, dtype=np.float32),
        }
        traj.record_step(
            step_idx=step_idx,
            sim_time=float(i) / control_hz,
            phase=phase,
            amplitude_m=amp,
            action=action,
            obs=obs,
            stable=True,
            hold_number=hold_number,
        )

    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"output_dir not empty (pass overwrite=True): {out}")
    out.mkdir(parents=True, exist_ok=True)
    ep_rel = episode_filename(0, 0)
    ep_path = out / ep_rel
    traj.save(ep_path, episode_meta)

    structures = [
        {
            "structure_idx": 0,
            "params_fingerprint": episode_meta.get("params_fingerprint"),
            "junction_names": list(junction_names),
            "n_woody_parts": int(episode_meta.get("n_woody_parts") or 2),
        }
    ]
    episodes = [
        {
            "structure_idx": 0,
            "direction_idx": 0,
            "env_idx": 0,
            "filename": ep_rel,
            "episode_id": episode_meta["episode_id"],
            "pull_direction": episode_meta.get("pull_direction"),
            "n_frames": traj.n_frames,
            "excluded": False,
            "excluded_reason": None,
        }
    ]
    seed_raw = episode_meta.get("seed")
    collection = {
        # Replay helpers require an int seed; real logs often omit one.
        "seed": int(seed_raw) if seed_raw is not None else 0,
        "ranges_path": str(fixture.resolve()),
        "control_hz": control_hz,
        "num_structures": 1,
        "num_directions": 1,
        "max_steps": traj.n_frames,
        "source_real_parquet": str(path.resolve()),
        "drive_fill": dm.get("drive_fill"),
    }
    if episode_meta.get("action_layout") == "vic_pose_v1":
        collection["action_semantics"] = episode_meta.get("action_semantics")
        collection["action_compatible_with_vic_twist"] = False
        collection["action_dim"] = int(episode_meta.get("action_dim") or 19)
        collection["action_layout"] = "vic_pose_v1"
    write_manifest(
        out,
        command_argv=list(command_argv or ["export_real_episode_to_batched_dataset"]),
        collection=collection,
        structures=structures,
        episodes=episodes,
        overwrite=True,
    )
    return out
