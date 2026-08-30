"""Map real-episode pre_grasp_geometry into FruitingSystemParams + placement."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from apple_pick_sim.fruiting_system.params import (
    FruitingSystemParams,
    load_ranges,
)
from apple_pick_sim.system_id.real_to_batched_sysid import (
    build_fruiting_params_from_real,
    range_midpoint,
)

_ZERO_EPS = 1e-12
_BEND_EPS = 1e-3
# Catalog gimbal: spur clocks about the primary; stem leans about fruiting→robot.
# Proxy world: primary +X, robot reach +Y, hang −Z. Fruiting→robot is −Y.
_WORLD_DOWN = (0.0, 0.0, -1.0)
_FRUITING_TO_ROBOT = (0.0, -1.0, 0.0)


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


def _radial_hat(primary_dir: tuple[float, float, float], spur_dir: tuple[float, float, float]) -> np.ndarray:
    """Unit vector on the primary cross-section toward the spur (zero if parallel)."""
    axis = np.asarray(primary_dir, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), _ZERO_EPS)
    d = np.asarray(spur_dir, dtype=np.float64)
    d /= max(float(np.linalg.norm(d)), _ZERO_EPS)
    radial = d - axis * float(np.dot(d, axis))
    radial_len = float(np.linalg.norm(radial))
    if radial_len < 1e-6:
        return np.zeros(3, dtype=np.float64)
    return radial / radial_len


def surface_to_centerline(
    spur_start_surface: tuple[float, float, float] | np.ndarray,
    spur_dir: tuple[float, float, float],
    primary_dir: tuple[float, float, float],
    primary_radius_m: float,
) -> tuple[float, float, float]:
    """Map a dowel-surface spur-start junction to the primary centerline T-center."""
    surface = np.asarray(spur_start_surface, dtype=np.float64).reshape(3)
    radial_hat = _radial_hat(primary_dir, spur_dir)
    center = surface - float(primary_radius_m) * radial_hat
    return (float(center[0]), float(center[1]), float(center[2]))


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


def _optional_manual_angle_deg(block: dict[str, Any], key: str) -> float | None:
    raw = block.get(key)
    if raw is None:
        return None
    angle = float(raw)
    if not math.isfinite(angle):
        raise ValueError(f"{key} must be finite, got {raw!r}")
    return angle


def _unit3(vec: np.ndarray | tuple[float, float, float], *, field: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(arr))
    if n < _ZERO_EPS:
        raise ValueError(f"{field}: zero-length vector")
    return arr / n


def _rotate_about_axis(
    vec: np.ndarray,
    axis: np.ndarray | tuple[float, float, float],
    angle_deg: float,
) -> np.ndarray:
    """Right-hand Rodrigues rotation of ``vec`` about ``axis`` by ``angle_deg``."""
    k = _unit3(axis, field="rotation_axis")
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    theta = math.radians(float(angle_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return v * cos_t + np.cross(k, v) * sin_t + k * float(np.dot(k, v)) * (1.0 - cos_t)


def _spur_rest_direction(primary_dir: np.ndarray) -> np.ndarray:
    """Horizontal T-junction rest: ⟂ primary, toward robot reach (+Y when primary is +X)."""
    down = np.asarray(_WORLD_DOWN, dtype=np.float64)
    rest = np.cross(primary_dir, down)
    if float(np.linalg.norm(rest)) < 1e-6:
        rest = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        rest = rest - primary_dir * float(np.dot(rest, primary_dir))
    return _unit3(rest, field="spur_rest_direction")


def rod_directions_from_manual_catalog_angles(
    primary_dir: tuple[float, float, float],
    *,
    spur_angle_deg: float,
    stem_angle_deg: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Catalog junction angles → unit spur/stem directions.

    ``manual_spur_angle_deg`` clocks the T-junction spur about the **primary**
    axis (proxy +X). Rest is horizontal toward robot reach (+Y); −90° hangs
    toward −Z. ``manual_stem_angle_deg`` then leans the stem about
    **fruiting→robot** (proxy −Y), not world Z. Right-hand +60° after a 90°
    hang yields stem ``(sin 60, 0, −cos 60)``.
    """
    primary = _unit3(primary_dir, field="primary_dir")
    rest = _spur_rest_direction(primary)
    spur = _rotate_about_axis(rest, primary, -float(spur_angle_deg))
    stem = _rotate_about_axis(spur, _FRUITING_TO_ROBOT, float(stem_angle_deg))
    spur_u = _unit3(spur, field="spur_direction")
    stem_u = _unit3(stem, field="stem_direction")
    return (
        (float(spur_u[0]), float(spur_u[1]), float(spur_u[2])),
        (float(stem_u[0]), float(stem_u[1]), float(stem_u[2])),
    )


def _resolve_rod_directions(
    *,
    primary_dir: tuple[float, float, float],
    parts: dict[str, Any],
    chord_spur_dir: tuple[float, float, float],
    chord_stem_dir: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float], str]:
    spur_block = parts.get("spur")
    stem_block = parts.get("stem")
    if not isinstance(spur_block, dict) or not isinstance(stem_block, dict):
        return chord_spur_dir, chord_stem_dir, "woody_chords"
    spur_angle = _optional_manual_angle_deg(spur_block, "manual_spur_angle_deg")
    stem_angle = _optional_manual_angle_deg(stem_block, "manual_stem_angle_deg")
    if spur_angle is None and stem_angle is None:
        return chord_spur_dir, chord_stem_dir, "woody_chords"
    if spur_angle is None or stem_angle is None:
        raise ValueError(
            "manual catalog angles require both manual_spur_angle_deg and "
            "manual_stem_angle_deg in pre_grasp_geometry.parts"
        )
    spur_dir, stem_dir = rod_directions_from_manual_catalog_angles(
        primary_dir,
        spur_angle_deg=spur_angle,
        stem_angle_deg=stem_angle,
    )
    return spur_dir, stem_dir, "manual_catalog_angles"


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
    apple_quat_xyzw: tuple[float, float, float, float] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _apple_quat_from_snapshot(
    snap: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    """Prefer explicit ``apple_quat_xyzw``; else derive from ``apple_pose_4x4``."""
    raw = snap.get("apple_quat_xyzw")
    if raw is not None:
        arr = np.asarray(raw, dtype=np.float64).reshape(4)
        q = (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
        n = float(np.linalg.norm(arr))
        if n < _ZERO_EPS:
            raise ValueError("apple_quat_xyzw: zero-length quaternion")
        return q
    pose = snap.get("apple_pose_4x4")
    if pose is None:
        return None
    from apple_pick_sim.system_id.real_post_grasp_plan import pose_4x4_to_pos_quat

    _pos, quat = pose_4x4_to_pos_quat(pose)
    return quat


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
    primary_dir: tuple[float, float, float],
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
    spur_start_surface = start9[0:3]
    spur_end = end9[0:3]
    apple_chord_end = end9[6:9]
    if float(np.linalg.norm(spur_end - start9[6:9])) > 1e-4:
        raise ValueError("Spur endpoint mismatch between part0 end and part2 start")

    apple_pos = coerce_xyz(snap["apple_pos"], field="apple_pos")
    apple_quat_xyzw = _apple_quat_from_snapshot(snap)

    bend = np.asarray(
        snap.get("woody_bending_angles", [0.0, 0.0, 0.0]), dtype=np.float64
    ).reshape(3)
    if float(np.max(np.abs(bend))) > _BEND_EPS:
        msg = f"pre-grasp woody_bending_angles not ~0: {bend.tolist()}"
        if strict:
            raise ValueError(msg)

    spur_dir = _unit(spur_end - spur_start_surface, field="spur_direction")
    stem_dir = _unit(apple_pos - spur_end, field="stem_direction")
    chord_spur_dir = spur_dir
    chord_stem_dir = stem_dir
    spur_dir, stem_dir, direction_source = _resolve_rod_directions(
        primary_dir=primary_dir,
        parts=parts,
        chord_spur_dir=chord_spur_dir,
        chord_stem_dir=chord_stem_dir,
    )

    primary_r = float(parts["primary"]["radius_m"])
    fruiting_base = surface_to_centerline(
        spur_start_surface, spur_dir, primary_dir, primary_r
    )
    surface_to_centerline_m = float(
        np.linalg.norm(np.asarray(spur_start_surface, dtype=np.float64) - np.asarray(fruiting_base))
    )

    rod_density_diag: dict[str, dict[str, Any]] = {}

    def _geo(name: str) -> dict[str, float]:
        block = parts[name]
        length = float(block["length_m"])
        radius = float(block["radius_m"])
        catalog_density = float(block["density_kg_m3"])
        density = catalog_density
        raw_mass = block.get("mass_kg")
        if raw_mass is not None:
            mass = float(raw_mass)
            if not math.isfinite(mass) or mass <= 0.0:
                raise ValueError(
                    f"parts[{name!r}].mass_kg must be finite > 0, got {mass}"
                )
            volume = math.pi * radius * radius * length
            if volume < _ZERO_EPS:
                raise ValueError(
                    f"parts[{name!r}] cylinder volume too small to convert mass_kg"
                )
            density = mass / volume
            rod_density_diag[name] = {
                "source": "mass_kg",
                "mass_kg": mass,
                "catalog_density_kg_m3": catalog_density,
                "density_kg_m3": density,
            }
        return {
            "length_m": length,
            "radius_m": radius,
            "density_kg_m3": density,
        }

    spur_chord = float(np.linalg.norm(spur_end - spur_start_surface))
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
    rod_geometry = {
        "primary": _geo("primary"),
        "spur": _geo("spur"),
        "stem": _geo("stem"),
    }

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
        "rod_direction_source": direction_source,
        "chord_spur_direction": list(chord_spur_dir),
        "chord_stem_direction": list(chord_stem_dir),
        "spur_direction": list(spur_dir),
        "stem_direction": list(stem_dir),
        "spur_start_surface": [float(spur_start_surface[0]), float(spur_start_surface[1]), float(spur_start_surface[2])],
        "primary_surface_to_centerline_m": surface_to_centerline_m,
        "fruiting_base_pos_source": "spur_start_surface − r_primary·radial_hat",
        "fruiting_base_pos": list(fruiting_base),
        "pre_grasp_snapshot_source": snap_source,
        "rod_density": rod_density_diag,
    }
    spur_u = np.asarray(spur_dir, dtype=np.float64)
    stem_u = np.asarray(stem_dir, dtype=np.float64)
    chord_spur_u = np.asarray(chord_spur_dir, dtype=np.float64)
    chord_stem_u = np.asarray(chord_stem_dir, dtype=np.float64)
    diagnostics["built_spur_stem_angle_deg"] = math.degrees(
        math.acos(float(np.clip(np.dot(spur_u, stem_u), -1.0, 1.0)))
    )
    diagnostics["chord_spur_stem_angle_deg"] = math.degrees(
        math.acos(float(np.clip(np.dot(chord_spur_u, chord_stem_u), -1.0, 1.0)))
    )
    spur_block = parts.get("spur") if isinstance(parts.get("spur"), dict) else {}
    stem_block = parts.get("stem") if isinstance(parts.get("stem"), dict) else {}
    diagnostics["manual_spur_angle_deg"] = spur_block.get("manual_spur_angle_deg")
    diagnostics["manual_stem_angle_deg"] = stem_block.get("manual_stem_angle_deg")

    return PreGraspMappedGeometry(
        fruiting_base_pos=fruiting_base,
        spur_direction=spur_dir,
        stem_direction=stem_dir,
        rod_geometry=rod_geometry,
        apple_radius_m=apple_r,
        apple_density_kg_m3=apple_d,
        woody_bending_angles=bend,
        apple_quat_xyzw=apple_quat_xyzw,
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
    source = diagnostics.get("rod_direction_source")
    lines.extend(
        [
            "connection angles:",
            (
                f"  source={source}  "
                f"manual_spur_angle_deg={diagnostics.get('manual_spur_angle_deg')}  "
                f"manual_stem_angle_deg={diagnostics.get('manual_stem_angle_deg')}"
            ),
            (
                f"  built spur–stem angle="
                f"{float(diagnostics.get('built_spur_stem_angle_deg', float('nan'))):.1f}°  "
                f"chord spur–stem angle="
                f"{float(diagnostics.get('chord_spur_stem_angle_deg', float('nan'))):.1f}°"
            ),
            "  axes: spur about primary, stem about fruiting→robot (−Y)",
        ]
    )
    return "\n".join(lines)


def fruiting_params_from_pre_grasp_meta(
    meta: dict[str, Any],
    *,
    fixture_path: str | Path,
    strict: bool = False,
) -> tuple[FruitingSystemParams, tuple[float, float, float], dict[str, Any]]:
    """Build ``FruitingSystemParams`` + base + diagnostics from metadata dict."""
    primary_dir = primary_direction_from_fixture(fixture_path)
    mapped = map_pre_grasp_geometry(meta, primary_dir=primary_dir, strict=strict)
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
    if mapped.apple_quat_xyzw is not None:
        params = dataclasses.replace(params, apple_quat_xyzw=mapped.apple_quat_xyzw)
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
