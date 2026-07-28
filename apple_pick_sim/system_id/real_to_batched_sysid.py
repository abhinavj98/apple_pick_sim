"""Convert real-world sys-ID episodes toward batched_sysid_v1 metadata.

Metadata-only slice: assemble fruiting/weld/init episode metadata from:

- **Pre-grasp** woody chords (non-bending, gravity largely opposed) + measured
  rod L/r + variance-fixture midpoints → rebuild ``fruiting_system`` geometry.
- **Post-grasp** settled apple / TCP → weld attachment on the settled plant.

Frame remapping is deferred. Contract and collection fixes:
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


def build_episode_metadata_from_real(
    input_path: str | Path,
    *,
    fixture_path: str | Path,
    weld_direction_sign: float = 1.0,
) -> dict[str, Any]:
    """Build batched-style episode metadata from one real-world parquet episode."""
    path = Path(input_path)
    fixture = Path(fixture_path)
    if not fixture.is_file():
        raise FileNotFoundError(f"fixture not found: {fixture}")

    table = pq.read_table(path)
    required_cols = (
        "step_idx",
        "woody_part_start_pos",
        "woody_part_end_pos",
        "tcp_pos",
        "tcp_quat",
        "apple_pos",
        "apple_quat",
        "robot_joint_q",
    )
    missing = [c for c in required_cols if c not in table.column_names]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")

    dm = _load_dataset_metadata(path)
    rod_geometry = dm.get("rod_geometry")
    if not isinstance(rod_geometry, dict):
        raise ValueError("dataset_metadata.rod_geometry is required")

    step_idx = table.column("step_idx").to_numpy(zero_copy_only=False)
    pre_i, grasp_i = split_pregrasp_and_trajectory(step_idx)

    start9 = table.column("woody_part_start_pos")[pre_i].as_py()
    end9 = table.column("woody_part_end_pos")[pre_i].as_py()
    starts, ends = flat_woody_to_dicts(start9, end9)
    directions = rod_directions_from_woody(starts, ends)

    apple_radius_m = dm.get("apple_radius_m")
    if apple_radius_m is not None:
        apple_radius_m = float(apple_radius_m)

    params = build_fruiting_params_from_real(
        ranges_path=fixture,
        rod_geometry=rod_geometry,
        directions=directions,
        apple_radius_m=apple_radius_m,
    )
    params_dict = fruiting_params_to_dict(params)

    tcp_pos = _as_float_list(table.column("tcp_pos")[grasp_i].as_py(), 3, field="tcp_pos")
    tcp_quat = _as_float_list(table.column("tcp_quat")[grasp_i].as_py(), 4, field="tcp_quat")
    apple_pos = _as_float_list(table.column("apple_pos")[grasp_i].as_py(), 3, field="apple_pos")
    apple_quat = _as_float_list(
        table.column("apple_quat")[grasp_i].as_py(), 4, field="apple_quat"
    )
    joint_q = _as_float_list(
        table.column("robot_joint_q")[grasp_i].as_py(), 7, field="robot_joint_q"
    )

    sign = float(weld_direction_sign)
    weld_direction = _unit_direction(
        sign * (np.asarray(tcp_pos, dtype=np.float64) - np.asarray(apple_pos, dtype=np.float64)),
        field="weld_direction",
    )

    source_robot = (dm.get("source_metadata") or {}).get("robot") or {}
    if "control_hz" not in source_robot:
        raise ValueError("dataset_metadata.source_metadata.robot.control_hz is required")
    control_hz = float(source_robot["control_hz"])

    fruiting_base_pos: list[float]
    if isinstance(dm.get("pre_grasp_geometry"), dict):
        from apple_pick_sim.system_id.real_pre_grasp_params import (
            map_pre_grasp_geometry,
            primary_direction_from_fixture,
        )

        primary_dir = primary_direction_from_fixture(fixture)
        mapped = map_pre_grasp_geometry(dm, primary_dir=primary_dir)
        fruiting_base_pos = [float(x) for x in mapped.fruiting_base_pos]
    else:
        legacy_base = dm.get("fruiting_base_pos")
        if legacy_base is None:
            raise ValueError(
                "dataset_metadata requires pre_grasp_geometry or legacy fruiting_base_pos"
            )
        fruiting_base_pos = _as_float_list(legacy_base, 3, field="fruiting_base_pos")

    pull_direction = None
    if "excitation_direction" in table.column_names:
        pull_direction = _as_float_list(
            table.column("excitation_direction")[grasp_i].as_py(),
            3,
            field="excitation_direction",
        )

    episode_id = dm.get("episode_id")
    if episode_id is None:
        episode_id = path.stem

    rod_radii = {
        name: float(rod_geometry[name]["radius_m"])
        for name in ROD_NAMES
        if name in rod_geometry
    }

    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": str(episode_id),
        "structure_idx": 0,
        "direction_idx": 0,
        "env_idx": 0,
        "pull_direction": pull_direction,
        "params_fingerprint": params_fingerprint(params),
        "fruiting_system_params": params_dict,
        "excitation_type": "quasi_static",
        "control_hz": control_hz,
        "seed": None,
        "n_woody_parts": 3,
        "junction_names": list(SIM_JUNCTION_NAMES),
        "initial_tcp_pos": tcp_pos,
        "initial_tcp_quat": tcp_quat,
        "initial_apple_pos": apple_pos,
        "initial_apple_quat": apple_quat,
        "initial_robot_joint_q": joint_q,
        "fixture_path": str(fixture.resolve()),
        "fruiting_base_pos": fruiting_base_pos,
        "apple_radius": float(params.apple_radius) if params.apple_radius is not None else None,
        "rod_radii": rod_radii,
        "weld_direction": weld_direction,
        "weld_reference_pos": list(apple_pos),
        "weld_reference_quat": list(apple_quat),
    }
    return meta
