"""Map real-episode pre_grasp_geometry into FruitingSystemParams + placement."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from apple_pick_sim.fruiting_system.params import FruitingSystemParams, load_ranges
from apple_pick_sim.system_id.real_to_batched_sysid import (
    build_fruiting_params_from_real,
    range_midpoint,
)

_ZERO_EPS = 1e-12
_BEND_EPS = 1e-3


def load_dataset_metadata(path: str | Path) -> dict[str, Any]:
    """Load JSON ``dataset_metadata`` from a parquet schema."""
    path = Path(path)
    schema = pq.read_schema(path)
    raw = schema.metadata or {}
    blob = raw.get(b"dataset_metadata")
    if blob is None:
        raise ValueError(f"{path}: missing schema metadata key dataset_metadata")
    text = blob.decode("utf-8") if isinstance(blob, (bytes, bytearray)) else str(blob)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("dataset_metadata must be a JSON object")
    return data


def coerce_xyz(value: Any, *, field: str) -> np.ndarray:
    """Parse a length-3 XYZ vector from a list or numpy ``__str__``."""
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        arr = np.fromstring(s, sep=" ", dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{field} must have length 3, got {arr.size}")
    return arr


def _unit(vec: np.ndarray, *, field: str) -> tuple[float, float, float]:
    n = float(np.linalg.norm(vec))
    if n < _ZERO_EPS:
        raise ValueError(f"{field}: zero-length vector")
    u = vec / n
    return (float(u[0]), float(u[1]), float(u[2]))


def _direction_from_angles(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    cos_el = math.cos(el)
    return (cos_el * math.cos(az), cos_el * math.sin(az), math.sin(el))


def primary_direction_from_fixture(fixture_path: str | Path) -> tuple[float, float, float]:
    """Fixture midpoint azimuth/elevation → unit primary direction (proxy: +X)."""
    ranges = load_ranges(fixture_path)
    primary = ranges.get("primary")
    if not isinstance(primary, dict):
        raise ValueError("fixture missing primary segment ranges")
    az = range_midpoint(primary["azimuth_deg"])
    el = range_midpoint(primary["elevation_deg"])
    return _direction_from_angles(az, el)


@dataclass(frozen=True)
class PreGraspMappedGeometry:
    """Geometry extracted from real pre_grasp_geometry for plant rebuild."""

    fruiting_base_pos: tuple[float, float, float]
    spur_direction: tuple[float, float, float]
    stem_direction: tuple[float, float, float]
    rod_geometry: dict[str, dict[str, float]]
    apple_radius_m: float | None
    apple_density_kg_m3: float | None
    woody_bending_angles: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _snapshot_has_woody(snap: Any) -> bool:
    return (
        isinstance(snap, dict)
        and "woody_part_start_pos" in snap
        and "woody_part_end_pos" in snap
        and "apple_pos" in snap
    )


def select_pre_grasp_woody_snapshot(pre: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Pick the non-bending woody+apple snapshot for plant rebuild.

    Prefer ``rest_snapshot_during_run`` (current compiler). Fall back to legacy
    ``snapshot`` when it still carries woody fields. ``settled_snapshot`` is not
    used for rebuild (different earlier camera settle).
    """
    rest = pre.get("rest_snapshot_during_run")
    if _snapshot_has_woody(rest):
        return rest, "rest_snapshot_during_run"
    snap = pre.get("snapshot")
    if _snapshot_has_woody(snap):
        return snap, "snapshot"
    raise ValueError(
        "pre_grasp_geometry missing woody+apple snapshot: expected "
        "rest_snapshot_during_run or snapshot with woody_part_* and apple_pos"
    )


def map_pre_grasp_geometry(
    meta: dict[str, Any],
    *,
    strict: bool = False,
) -> PreGraspMappedGeometry:
    """Map Branch/Spur/Apple shared_endpoints pre-grasp into rod geometry."""
    topo = meta.get("topology") or {}
    names = list(topo.get("junction_names") or [])
    if names != ["Branch", "Spur", "Apple"] or not topo.get("shared_endpoints"):
        raise ValueError(
            "unsupported topology: expected shared_endpoints Branch/Spur/Apple, "
            f"got junction_names={names!r} shared_endpoints={topo.get('shared_endpoints')!r}"
        )
    pre = meta.get("pre_grasp_geometry")
    if not isinstance(pre, dict):
        raise ValueError("missing pre_grasp_geometry")
    parts = pre.get("parts")
    if not isinstance(parts, dict):
        raise ValueError("pre_grasp_geometry requires parts")
    snap, snap_source = select_pre_grasp_woody_snapshot(pre)

    start9 = np.asarray(snap["woody_part_start_pos"], dtype=np.float64).reshape(9)
    end9 = np.asarray(snap["woody_part_end_pos"], dtype=np.float64).reshape(9)
    # part0 Branch→Spur (T → spur end), part2 Spur→Apple
    branch = start9[0:3]
    spur_end = end9[0:3]
    apple_chord_end = end9[6:9]
    if float(np.linalg.norm(spur_end - start9[6:9])) > 1e-4:
        raise ValueError("Spur endpoint mismatch between part0 end and part2 start")

    apple_pos = coerce_xyz(snap["apple_pos"], field="apple_pos")

    bend = np.asarray(
        snap.get("woody_bending_angles", [0.0, 0.0, 0.0]), dtype=np.float64
    ).reshape(3)
    if float(np.max(np.abs(bend))) > _BEND_EPS:
        msg = f"pre-grasp woody_bending_angles not ~0: {bend.tolist()}"
        if strict:
            raise ValueError(msg)

    spur_dir = _unit(spur_end - branch, field="spur_direction")
    stem_dir = _unit(apple_pos - spur_end, field="stem_direction")

    def _geo(name: str) -> dict[str, float]:
        block = parts[name]
        return {
            "length_m": float(block["length_m"]),
            "radius_m": float(block["radius_m"]),
            "density_kg_m3": float(block["density_kg_m3"]),
        }

    spur_chord = float(np.linalg.norm(spur_end - branch))
    apple_r = float(parts["apple"]["radius_m"]) if "apple" in parts else None
    apple_d = float(parts["apple"]["density_kg_m3"]) if "apple" in parts else None
    # Woody Apple junction is the fruit CoM; physical stem is spur→surface.
    spur_to_com = float(np.linalg.norm(apple_pos - spur_end))
    if apple_r is None:
        stem_chord = spur_to_com
    else:
        stem_chord = spur_to_com - float(apple_r)
    spur_L = float(parts["spur"]["length_m"])
    stem_L = float(parts["stem"]["length_m"])
    apple_vs_chord = apple_pos - apple_chord_end

    def _rel_err(catalog: float, measured: float) -> float:
        if abs(catalog) < _ZERO_EPS:
            return float("inf") if measured > _ZERO_EPS else 0.0
        return abs(measured - catalog) / abs(catalog)

    diagnostics: dict[str, Any] = {
        "spur_chord_length_m": spur_chord,
        "stem_spur_to_com_m": spur_to_com,
        "stem_chord_length_m": stem_chord,
        "stem_chord_formula": "‖spur_end−apple_CoM‖−apple_radius",
        "spur_catalog_length_m": spur_L,
        "stem_catalog_length_m": stem_L,
        "spur_length_abs_error_m": abs(spur_chord - spur_L),
        "stem_length_abs_error_m": abs(stem_chord - stem_L),
        "spur_length_rel_error": _rel_err(spur_L, spur_chord),
        "stem_length_rel_error": _rel_err(stem_L, stem_chord),
        "spur_length_error": {
            "catalog_m": spur_L,
            "chord_m": spur_chord,
            "abs_m": abs(spur_chord - spur_L),
            "rel": _rel_err(spur_L, spur_chord),
        },
        "stem_length_error": {
            "catalog_m": stem_L,
            "chord_m": stem_chord,
            "spur_to_com_m": spur_to_com,
            "apple_radius_m": apple_r,
            "abs_m": abs(stem_chord - stem_L),
            "rel": _rel_err(stem_L, stem_chord),
        },
        "apple_pos_vs_chord_end_m": apple_vs_chord.tolist(),
        "apple_pos_vs_chord_end_norm_m": float(np.linalg.norm(apple_vs_chord)),
        "spur_direction": list(spur_dir),
        "stem_direction": list(stem_dir),
        "fruiting_base_pos": [float(branch[0]), float(branch[1]), float(branch[2])],
        "pre_grasp_snapshot_source": snap_source,
    }

    return PreGraspMappedGeometry(
        fruiting_base_pos=(float(branch[0]), float(branch[1]), float(branch[2])),
        spur_direction=spur_dir,
        stem_direction=stem_dir,
        rod_geometry={
            "primary": _geo("primary"),
            "spur": _geo("spur"),
            "stem": _geo("stem"),
        },
        apple_radius_m=apple_r,
        apple_density_kg_m3=apple_d,
        woody_bending_angles=bend,
        diagnostics=diagnostics,
    )


def format_pre_grasp_diagnostics(diagnostics: dict[str, Any]) -> str:
    """Human-readable catalog-vs-chord / apple diagnostics."""
    lines = [
        "pre-grasp length diagnostics:",
        (
            f"  spur: catalog={diagnostics['spur_catalog_length_m']:.4f} m  "
            f"chord={diagnostics['spur_chord_length_m']:.4f} m  "
            f"abs_err={diagnostics['spur_length_abs_error_m']:.4f} m  "
            f"rel_err={diagnostics['spur_length_rel_error']:.3%}"
        ),
        (
            f"  stem: catalog={diagnostics['stem_catalog_length_m']:.4f} m  "
            f"chord={diagnostics['stem_chord_length_m']:.4f} m  "
            f"(‖spur−CoM‖={diagnostics['stem_spur_to_com_m']:.4f} m − r)  "
            f"abs_err={diagnostics['stem_length_abs_error_m']:.4f} m  "
            f"rel_err={diagnostics['stem_length_rel_error']:.3%}"
        ),
        (
            f"  apple_pos vs Spur→Apple chord end: "
            f"{diagnostics['apple_pos_vs_chord_end_norm_m']:.6e} m"
        ),
    ]
    return "\n".join(lines)


def fruiting_params_from_pre_grasp_meta(
    meta: dict[str, Any],
    *,
    fixture_path: str | Path,
    strict: bool = False,
) -> tuple[FruitingSystemParams, tuple[float, float, float], dict[str, Any]]:
    """Build ``FruitingSystemParams`` + base + diagnostics from metadata dict."""
    mapped = map_pre_grasp_geometry(meta, strict=strict)
    primary_dir = primary_direction_from_fixture(fixture_path)
    directions = {
        "primary": primary_dir,
        "spur": mapped.spur_direction,
        "stem": mapped.stem_direction,
    }
    params = build_fruiting_params_from_real(
        ranges_path=fixture_path,
        rod_geometry=mapped.rod_geometry,
        directions=directions,
        apple_radius_m=mapped.apple_radius_m,
        apple_density_kg_m3=mapped.apple_density_kg_m3,
        use_parts_density=True,
    )
    diagnostics = {
        **mapped.diagnostics,
        "primary_direction": list(primary_dir),
    }
    return params, mapped.fruiting_base_pos, diagnostics


def fruiting_params_from_pre_grasp_parquet(
    path: str | Path,
    *,
    fixture_path: str | Path,
    strict: bool = False,
) -> tuple[FruitingSystemParams, tuple[float, float, float], dict[str, Any]]:
    """Load parquet ``dataset_metadata`` and build plant params from pre-grasp."""
    meta = load_dataset_metadata(path)
    return fruiting_params_from_pre_grasp_meta(
        meta, fixture_path=fixture_path, strict=strict
    )
