"""Convert real-world sys-ID episodes toward batched_sysid_v1 metadata.

Metadata-only slice: assemble fruiting/weld/init episode metadata by calling the
same pre/post builders as ``robot_replay/example_view_pre_grasp_settle.py``
(``fruiting_params_from_pre_grasp_parquet``, ``post_grasp_plan_from_metadata``),
then packaging into batched episode keys.

Frame remapping and trajectory export are deferred. Contract:
``docs/superpowers/specs/2026-08-07-real-to-batched-metadata-parity-design.md``,
``docs/real-sysid-pre-post-grasp-fixes.md``, ``robot_replay/README.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from apple_pick_sim.fruiting_system.params import (
    FruitingSystemParams,
    _spur_surface_offset_from_ranges,
    fruiting_params_to_dict,
    load_ranges,
    params_fingerprint,
    rod_params_from_material,
)
from apple_pick_sim.system_id.batched_trajectory_store import SCHEMA_VERSION

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
    Young's modulus, damping, stretch, and ``num_segments`` still use fixture
    midpoints.
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
        kwargs: dict[str, Any] = {}
        fixed = seg.get("vbd_stretch_fixed")
        if isinstance(fixed, dict):
            kwargs["stretch_stiffness"] = float(fixed["stretch_stiffness"])
            kwargs["stretch_damping"] = float(fixed["stretch_damping"])
        if use_parts_density:
            if "density_kg_m3" not in geo:
                raise ValueError(f"rod_geometry[{name!r}] missing density_kg_m3")
            density = float(geo["density_kg_m3"])
        else:
            density = range_midpoint(seg["density"])
        rods[name] = rod_params_from_material(
            range_midpoint(seg["youngs_modulus_pa"]),
            range_midpoint(seg["damping_ratio"]),
            float(geo["length_m"]),
            float(geo["radius_m"]),
            density,
            int(round(range_midpoint(seg["num_segments"]))),
            directions[name],
            **kwargs,
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
        spur_attach_fraction=0.5,
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
        "n_woody_parts": 3,
        "junction_names": list(SIM_JUNCTION_NAMES),
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
    return meta
