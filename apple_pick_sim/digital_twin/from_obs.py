"""Infer fruiting-system parameters and build scenes from digital-twin observations."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from apple_pick_sim.digital_twin.obs_io import DigitalTwinObs
from apple_pick_sim.fruiting_system.coupled import CoupledCableScene, generate_coupled_cable_scene
from apple_pick_sim.fruiting_system.params import (
    FruitingSystemParams,
    GripperProxyConfig,
    RodParams,
    TOPOLOGY_LINEAR_CHAIN,
    TOPOLOGY_T_JUNCTION,
    load_ranges,
    parse_fixture_args,
    rod_params_from_material,
    _spur_attach_fraction_from_ranges,
    _spur_surface_offset_from_ranges,
    _stretch_kw_from_seg_ranges,
    _topology_from_ranges,
)

_ROD_NAMES = ("primary", "secondary", "spur", "stem")
_SUPPORT_JUNCTION_PREFIX = "primary_support_"


def _is_support_junction(name: str) -> bool:
    return name.startswith(_SUPPORT_JUNCTION_PREFIX)


def ordered_rod_segments_from_junctions(junction_names: list[str]) -> tuple[str, ...]:
    """Return enabled rod segment names in chain order from junction labels."""
    rod_junctions = [n for n in junction_names if not _is_support_junction(n)]
    if not rod_junctions:
        raise ValueError("junction_names must include at least one rod junction")
    rods: list[str] = []
    for i, name in enumerate(rod_junctions):
        if "_" not in name:
            raise ValueError(f"invalid junction name {name!r}")
        left, right = name.split("_", 1)
        if i == 0:
            if left not in _ROD_NAMES:
                raise ValueError(f"unknown rod segment {left!r} in {name!r}")
            rods.append(left)
        if right == "apple":
            break
        if right not in _ROD_NAMES:
            raise ValueError(f"unknown rod segment {right!r} in {name!r}")
        rods.append(right)
    return tuple(rods)


def junction_has_apple(junction_names: list[str]) -> bool:
    """True when the last rod junction (ignoring world supports) attaches to the apple."""
    rod_junctions = [n for n in junction_names if not _is_support_junction(n)]
    return bool(rod_junctions) and rod_junctions[-1].endswith("_apple")


def _median_range_scalar(block: dict, key: str) -> float:
    entry = block[key]
    if not isinstance(entry, dict) or "min" not in entry or "max" not in entry:
        raise ValueError(f"range block missing min/max for {key!r}")
    lo = float(entry["min"])
    hi = float(entry["max"])
    return 0.5 * (lo + hi)


def _median_range_int(block: dict, key: str) -> int:
    return int(round(_median_range_scalar(block, key)))


def _rod_params_from_range_median(seg_ranges: dict) -> RodParams:
    """Sample geometry/material at range midpoints with a placeholder direction."""
    length = _median_range_scalar(seg_ranges, "length")
    radius = _median_range_scalar(seg_ranges, "radius")
    density = _median_range_scalar(seg_ranges, "density")
    num_segments = max(2, _median_range_int(seg_ranges, "num_segments"))
    return rod_params_from_material(
        _median_range_scalar(seg_ranges, "youngs_modulus_pa"),
        _median_range_scalar(seg_ranges, "damping_ratio"),
        length,
        radius,
        density,
        num_segments,
        (1.0, 0.0, 0.0),
        **_stretch_kw_from_seg_ranges(
            seg_ranges,
            length=length,
            radius=radius,
            density=density,
            num_segments=num_segments,
        ),
    )


def params_from_ranges_median(ranges: dict) -> FruitingSystemParams:
    """Build :class:`FruitingSystemParams` using range midpoints for every enabled segment."""
    rods: dict[str, RodParams | None] = {}
    for name in _ROD_NAMES:
        block = ranges.get(name)
        rods[name] = None if block is None else _rod_params_from_range_median(block)

    apple_block = ranges.get("apple")
    apple_radius: float | None = None
    apple_density: float | None = None
    if apple_block is not None:
        apple_radius = _median_range_scalar(apple_block, "radius")
        apple_density = _median_range_scalar(apple_block, "density")

    if not any(rods[n] is not None for n in _ROD_NAMES):
        raise ValueError("at least one rod segment must be present in ranges")

    return FruitingSystemParams(
        primary=rods["primary"],
        secondary=rods["secondary"],
        spur=rods["spur"],
        stem=rods["stem"],
        apple_radius=apple_radius,
        apple_density=apple_density,
        topology=_topology_from_ranges(ranges),
        spur_attach_fraction=_spur_attach_fraction_from_ranges(ranges),
        spur_surface_offset=_spur_surface_offset_from_ranges(ranges),
    )


def infer_segment_geometry(
    obs: DigitalTwinObs,
) -> dict[str, tuple[tuple[float, float, float], float]]:
    """Recover per-rod unit direction and length from junction anchor positions."""
    rods = ordered_rod_segments_from_junctions(obs.junction_names)
    name_to_idx = {name: i for i, name in enumerate(obs.junction_names)}
    start_positions = obs.woody_part_start_pos.reshape(-1, 3)
    end_positions = obs.woody_part_end_pos.reshape(-1, 3)
    base = np.asarray(obs.fruiting_base_pos, dtype=np.float64)
    rod_junction_indices = [
        name_to_idx[name]
        for name in obs.junction_names
        if not _is_support_junction(name)
    ]

    geometry: dict[str, tuple[tuple[float, float, float], float]] = {}
    left_i = name_to_idx.get("primary_support_left")
    right_i = name_to_idx.get("primary_support_right")

    for i, rod_name in enumerate(rods):
        if i == 0 and left_i is not None and right_i is not None:
            origin = end_positions[left_i]
            target = end_positions[right_i]
        elif i == 0:
            origin = base
            target = start_positions[rod_junction_indices[0]]
        elif i < len(rod_junction_indices):
            origin = start_positions[rod_junction_indices[i - 1]]
            target = start_positions[rod_junction_indices[i]]
        else:
            origin = start_positions[rod_junction_indices[i - 1]]
            target = end_positions[rod_junction_indices[i - 1]]
        delta = target - origin
        length = float(np.linalg.norm(delta))
        if length < 1e-9:
            raise ValueError(f"zero-length segment inferred for {rod_name!r}")
        direction = tuple((delta / length).astype(np.float64).tolist())
        geometry[rod_name] = (direction, length)
    return geometry


def _resolve_apple_radius(obs: DigitalTwinObs, ranges: dict) -> float | None:
    if obs.apple_radius is not None:
        return float(obs.apple_radius)
    apple_block = ranges.get("apple")
    if apple_block is None:
        return None
    return _median_range_scalar(apple_block, "radius")


def infer_params_from_obs(
    obs: DigitalTwinObs,
    base_ranges_path: str | Path,
) -> FruitingSystemParams:
    """Infer :class:`FruitingSystemParams` geometry from obs; other scalars from range midpoints."""
    ranges = load_ranges(base_ranges_path)
    template = params_from_ranges_median(ranges)
    geometry = infer_segment_geometry(obs)
    enabled_rods = ordered_rod_segments_from_junctions(obs.junction_names)
    has_apple = junction_has_apple(obs.junction_names)

    def _rod_for(name: str) -> RodParams | None:
        if name not in enabled_rods:
            return None
        ref = getattr(template, name)
        if ref is None:
            raise ValueError(f"base fixture has no ranges for enabled segment {name!r}")
        direction, length = geometry[name]
        radius = (
            float(obs.rod_radii[name])
            if obs.rod_radii is not None and name in obs.rod_radii
            else ref.radius
        )
        return rod_params_from_material(
            ref.youngs_modulus_pa,
            ref.damping_ratio,
            length,
            radius,
            ref.density,
            ref.num_segments,
            direction,
        )

    apple_radius = _resolve_apple_radius(obs, ranges) if has_apple else None
    apple_density = template.apple_density if has_apple else None
    topology = (
        TOPOLOGY_T_JUNCTION
        if any(_is_support_junction(n) for n in obs.junction_names)
        else template.topology
    )

    return FruitingSystemParams(
        primary=_rod_for("primary"),
        secondary=_rod_for("secondary"),
        spur=_rod_for("spur"),
        stem=_rod_for("stem"),
        apple_radius=apple_radius,
        apple_density=apple_density,
        topology=topology,
        spur_attach_fraction=template.spur_attach_fraction,
        spur_surface_offset=template.spur_surface_offset,
    )


def build_digital_twin_scene(
    obs: DigitalTwinObs,
    base_ranges_path: str | Path,
    *,
    device: str | None = None,
    fix_to_apple: bool = True,
    enable_self_collisions: bool = False,
    robot_base_pos: tuple[float, float, float] | None = None,
) -> CoupledCableScene:
    """Build a VBD cable scene whose straight-rod geometry matches ``obs``."""
    ranges = load_ranges(base_ranges_path)
    params = infer_params_from_obs(obs, base_ranges_path)
    if robot_base_pos is None:
        robot_base_pos = parse_fixture_args(ranges).robot_base_pos

    if fix_to_apple:
        proxy = GripperProxyConfig(
            fix_to_apple=True,
            weld_direction=obs.weld_direction,
        )
    else:
        proxy = GripperProxyConfig(fix_to_apple=False)
    return generate_coupled_cable_scene(
        ranges,
        seed=0,
        params=params,
        base_pos=obs.fruiting_base_pos,
        device=device,
        enable_self_collisions=enable_self_collisions,
        gripper_proxy=proxy,
        robot_base_pos=robot_base_pos,
    )
