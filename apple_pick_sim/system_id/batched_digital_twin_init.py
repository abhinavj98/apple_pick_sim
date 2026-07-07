"""Digital-twin observation helpers for batched sys-ID datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from apple_pick_sim.digital_twin import DigitalTwinObs, infer_params_from_obs
from apple_pick_sim.fruiting_system.params import FruitingSystemParams, load_ranges, parse_fixture_args
from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset
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


def digital_twin_obs_from_batched_episode(
    dataset: BatchedSysIdDataset,
    structure_idx: int,
    direction_idx: int = 0,
) -> DigitalTwinObs:
    """Build a digital-twin observation bundle from batched episode metadata and frame 0."""
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

    woody_start = stack_woody_pos_frame(
        arrays["woody_part_start_pos"], 0, junction_names
    )
    woody_end = stack_woody_pos_frame(
        arrays["woody_part_end_pos"], 0, junction_names
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
    """Infer base :class:`FruitingSystemParams` for one structure from frame-0 observations."""
    obs = digital_twin_obs_from_batched_episode(dataset, structure_idx, 0)
    meta = dataset.load_episode_metadata(structure_idx, 0)
    fixture_path = _resolve_fixture_path(meta)
    if fixture_path is None:
        raise ValueError("fixture_path is required in episode metadata")
    return infer_params_from_obs(obs, fixture_path)
