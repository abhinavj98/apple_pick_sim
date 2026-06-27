"""Fruiting-system parameters, sampling, and range validation."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Collection
from pathlib import Path
from typing import Literal

from apple_pick_sim.fruiting_system.gripper_proxy_shape import (
    GRIPPER_PROXY_CYLINDER_HALF_HEIGHT,
    GRIPPER_PROXY_CYLINDER_RADIUS,
)
import numpy as np

# Placeholder until hardware weigh-in; see docs/real-world-proxy.md.
PLACEHOLDER_EE_MASS_KG: float = 0.5

TOPOLOGY_T_JUNCTION = "t_junction"
TOPOLOGY_LINEAR_CHAIN = "linear_chain"
ALLOWED_TOPOLOGIES = frozenset({TOPOLOGY_T_JUNCTION, TOPOLOGY_LINEAR_CHAIN})
DEFAULT_TOPOLOGY = TOPOLOGY_T_JUNCTION
DEFAULT_SPUR_ATTACH_FRACTION = 0.5
REAL_WORLD_PROXY_VARIANCE_FIXTURE = "fruiting_system_ranges_real_world_proxy_variance.json"

@dataclasses.dataclass
class RodParams:
    """Sampled parameters for a single rod segment in the fruiting chain."""

    num_segments: int
    length: float
    radius: float
    bend_stiffness: float
    bend_damping: float
    stretch_stiffness: float
    density: float
    direction: tuple[float, float, float]  # unit vector in world space


@dataclasses.dataclass
class FruitingSystemParams:
    """All sampled parameters for a single fruiting-system instance.

    ``None`` on a rod field or on apple scalars means that piece is disabled
    (not built). At least one rod segment must be enabled.

    To turn off segments from code while keeping RNG and downstream directions
    consistent with JSON ``null``, use :func:`sample_params` ``omit=...`` rather
    than setting rod fields to ``None`` after a full sample (the latter can
    leave spur/stem directions wrong if an *intermediate* rod is removed).
    """

    primary: RodParams | None
    secondary: RodParams | None
    spur: RodParams | None
    stem: RodParams | None
    apple_radius: float | None
    apple_density: float | None
    topology: str = DEFAULT_TOPOLOGY
    spur_attach_fraction: float = DEFAULT_SPUR_ATTACH_FRACTION


FRUITING_SYSTEM_PARAMS_SCHEMA = "fruiting_system_params_v1"


def analytic_apple_mass_kg(params: FruitingSystemParams) -> float | None:
    """Solid-sphere mass from sampled ``apple_radius`` and ``apple_density`` [kg]."""
    if params.apple_radius is None or params.apple_density is None:
        return None
    r = float(params.apple_radius)
    rho = float(params.apple_density)
    return (4.0 / 3.0) * math.pi * r**3 * rho


def _rod_params_to_row(rod: RodParams | None) -> dict[str, Any] | None:
    if rod is None:
        return None
    return {
        "num_segments": int(rod.num_segments),
        "length": float(rod.length),
        "radius": float(rod.radius),
        "bend_stiffness": float(rod.bend_stiffness),
        "bend_damping": float(rod.bend_damping),
        "stretch_stiffness": float(rod.stretch_stiffness),
        "density": float(rod.density),
        "direction": [float(x) for x in rod.direction],
    }


def fruiting_params_to_dict(params: FruitingSystemParams) -> dict[str, Any]:
    """Return a lossless JSON-ready representation of sampled fruiting params."""
    return {
        "schema": FRUITING_SYSTEM_PARAMS_SCHEMA,
        "primary": _rod_params_to_row(params.primary),
        "secondary": _rod_params_to_row(params.secondary),
        "spur": _rod_params_to_row(params.spur),
        "stem": _rod_params_to_row(params.stem),
        "apple_radius": None if params.apple_radius is None else float(params.apple_radius),
        "apple_density": None if params.apple_density is None else float(params.apple_density),
        "topology": params.topology,
        "spur_attach_fraction": float(params.spur_attach_fraction),
    }


def fruiting_params_to_json(params: FruitingSystemParams) -> str:
    """Serialize sampled fruiting params for episode metadata storage."""
    return json.dumps(fruiting_params_to_dict(params), sort_keys=True, separators=(",", ":"))


def _expect_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _rod_params_from_row(value: Any, *, field: str) -> RodParams | None:
    if value is None:
        return None
    row = _expect_mapping(value, field=field)
    direction = row.get("direction")
    if not isinstance(direction, (list, tuple)) or len(direction) != 3:
        raise ValueError(f"{field}.direction must be [x, y, z]")
    return RodParams(
        num_segments=int(row["num_segments"]),
        length=float(row["length"]),
        radius=float(row["radius"]),
        bend_stiffness=float(row["bend_stiffness"]),
        bend_damping=float(row["bend_damping"]),
        stretch_stiffness=float(row["stretch_stiffness"]),
        density=float(row["density"]),
        direction=(float(direction[0]), float(direction[1]), float(direction[2])),
    )


def fruiting_params_from_dict(data: dict[str, Any]) -> FruitingSystemParams:
    """Deserialize :func:`fruiting_params_to_dict` output."""
    row = _expect_mapping(data, field="fruiting_system_params")
    schema = row.get("schema")
    if schema != FRUITING_SYSTEM_PARAMS_SCHEMA:
        raise ValueError(
            f"unsupported fruiting params schema {schema!r}; "
            f"expected {FRUITING_SYSTEM_PARAMS_SCHEMA!r}"
        )
    topology = row.get("topology", DEFAULT_TOPOLOGY)
    if topology not in ALLOWED_TOPOLOGIES:
        raise ValueError(
            f"unsupported topology {topology!r}; expected one of {sorted(ALLOWED_TOPOLOGIES)}"
        )
    spur_attach_fraction = float(
        row.get("spur_attach_fraction", DEFAULT_SPUR_ATTACH_FRACTION)
    )
    params = FruitingSystemParams(
        primary=_rod_params_from_row(row.get("primary"), field="primary"),
        secondary=_rod_params_from_row(row.get("secondary"), field="secondary"),
        spur=_rod_params_from_row(row.get("spur"), field="spur"),
        stem=_rod_params_from_row(row.get("stem"), field="stem"),
        apple_radius=None if row.get("apple_radius") is None else float(row["apple_radius"]),
        apple_density=None if row.get("apple_density") is None else float(row["apple_density"]),
        topology=topology,
        spur_attach_fraction=spur_attach_fraction,
    )
    if not any((params.primary, params.secondary, params.spur, params.stem)):
        raise ValueError("at least one rod segment must be present in fruiting params")
    return params


def fruiting_params_from_json(value: str) -> FruitingSystemParams:
    """Deserialize sampled fruiting params stored as episode metadata JSON."""
    return fruiting_params_from_dict(json.loads(value))


@dataclasses.dataclass(frozen=True)
class GripperProxyConfig:
    """Gripper proxy rigid body added on the cable ``Model`` for M1 coupling.

    The proxy is the VBD-side stand-in for the robot TCP: its pose tracks the MuJoCo
    body each substep, and contact/joint reactions on the proxy are harvested as wrenches
    fed back into ``robot_state.body_f`` on the next substep (see module docstring).
    Mass and collision shape should match the robot TCP link built in
    ``coupled_fruiting.build_placeholder_tcp_robot_model`` so velocity-delta harvest
    uses consistent inertia. Default shape is a cylinder (50 mm radius, 140 mm length)
    with the distal tip at the body origin (+Z).
    """

    mass: float = PLACEHOLDER_EE_MASS_KG
    shape: Literal["box", "cylinder"] = "cylinder"
    cylinder_radius: float = GRIPPER_PROXY_CYLINDER_RADIUS
    cylinder_half_height: float = GRIPPER_PROXY_CYLINDER_HALF_HEIGHT
    box_half_extents: tuple[float, float, float] = (0.05, 0.05, 0.05)
    label: str = "gripper_proxy"
    fix_to_apple: bool = False
    """If ``True``, weld the proxy to the apple with a FIXED joint at the exterior pole.

    Default ``False``: velocity-delta harvest + proxy-only sync. Set ``True`` for stem-harvest /
    apple co-teleport (see ``example_coupled_fruiting.py --fix-to-apple``).
    """
    robot_facing_weld: bool = False
    """If ``True`` and ``fix_to_apple=True``, weld to the apple face toward ``robot_base_pos``.

    Requires ``robot_base_pos`` at scene build time; otherwise placement uses a random
    exterior surface point (legacy behavior).
    """
    weld_direction: tuple[float, float, float] | None = None
    """Explicit unit approach vector for the weld pole (world frame, normalized at build time).

    When set and ``fix_to_apple=True``, replaces the default approach direction.
    When ``robot_facing_weld=True``, validated to lie on the robot-facing hemisphere
    (dot ≥ 0 with apple→robot unit vector); raises ``ValueError`` otherwise.
    """
    weld_reference_pos: tuple[float, float, float] | None = None
    """Optional apple-center override for robot-facing hemisphere checks and weld offset.

    Used by settle-then-weld workflows where the apple center moves before the welded
    scene is built. When ``None``, the nominal build-time apple center is used.
    """
    weld_reference_quat: tuple[float, float, float, float] | None = None
    """Apple orientation ``(x, y, z, w)`` paired with ``weld_reference_pos`` for apple-frame offset."""

@dataclasses.dataclass(frozen=True)
class FixtureArgs:
    """Scene placement from a ranges JSON ``args`` block."""

    fruiting_base_pos: tuple[float, float, float] | None = None
    robot_base_pos: tuple[float, float, float] | None = None


def _coerce_xyz_triplet(raw: object, *, field: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"args.{field} must be [x, y, z]")
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def parse_fixture_args(ranges: dict) -> FixtureArgs:
    """Return placement args from ``ranges['args']``, or empty defaults when absent."""
    block = ranges.get("args")
    if block is None:
        return FixtureArgs()
    if not isinstance(block, dict):
        raise ValueError("args must be a JSON object")
    fruiting = block.get("fruiting_base_pos")
    robot = block.get("robot_base_pos")
    return FixtureArgs(
        fruiting_base_pos=None if fruiting is None else _coerce_xyz_triplet(
            fruiting, field="fruiting_base_pos"
        ),
        robot_base_pos=None if robot is None else _coerce_xyz_triplet(
            robot, field="robot_base_pos"
        ),
    )


def resolve_fruiting_base_pos(
    ranges: dict,
    default: tuple[float, float, float],
    *,
    override: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    """Resolve fruiting chain base position: explicit override, JSON ``args``, then ``default``."""
    if override is not None:
        return override
    args = parse_fixture_args(ranges)
    if args.fruiting_base_pos is not None:
        return args.fruiting_base_pos
    return default


def resolve_robot_base_pos(
    ranges: dict,
    *,
    override: tuple[float, float, float] | None = None,
) -> tuple[float, float, float] | None:
    """Resolve FR3 root translation: explicit override, JSON ``args``, else ``None`` (auto placement)."""
    if override is not None:
        return override
    return parse_fixture_args(ranges).robot_base_pos


def default_ranges_fixture_path() -> Path:
    """Default range JSON for examples and gym env (real-world proxy domain randomization)."""
    return Path(__file__).resolve().parent.parent / "fixtures" / REAL_WORLD_PROXY_VARIANCE_FIXTURE


def load_ranges(path: str | Path) -> dict:
    """Load a fruiting-system range JSON file.

    Args:
        path: Path to the JSON range file.

    Returns:
        Dict with keys ``primary``, ``secondary``, ``spur``, ``stem``, ``apple``, and
        optionally ``args`` (``fruiting_base_pos`` / ``robot_base_pos`` as ``[x, y, z]``).
        A rod or ``apple`` entry may be JSON ``null`` (``None`` in Python) to omit that
        piece from sampling and scene construction (at least one rod must remain).
    """
    with open(path) as f:
        data = json.load(f)
    _validate_ranges(data)
    return data


def _coerce_omit(omit: Collection[str] | None) -> frozenset[str]:
    """Validate ``omit`` keys for :func:`sample_params` / :func:`generate_scene`."""
    if omit is None:
        return frozenset()
    allowed = frozenset({"primary", "secondary", "spur", "stem", "apple"})
    o = frozenset(omit)
    extra = o - allowed
    if extra:
        raise ValueError(
            f"omit contains unknown keys: {sorted(extra)}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )
    return o


def sample_params(
    ranges: dict,
    seed: int,
    *,
    omit: Collection[str] | None = None,
) -> FruitingSystemParams:
    """Sample fruiting-system parameters from ``ranges`` deterministically via ``seed``.

    Skips sampling for any rod segment (or apple) whose range entry is ``None``.
    Names in ``omit`` (e.g. ``{"secondary", "apple"}``) force that piece off even
    when the range entry is present—useful for toggling topology from code without
    editing JSON. Omission is applied **during** sampling so ``parent_dir`` for
    spur/stem matches omitting intermediate rods (same as setting that range to
    ``null``).

    When both **primary** and **secondary** are enabled, enforces
    ``primary.bend_stiffness >= secondary.bend_stiffness``.

    Args:
        ranges: Range dict as returned by :func:`load_ranges`.
        seed: Integer seed for the RNG.
        omit: Optional set of segment names to force to ``None`` in the result.

    Returns:
        A :class:`FruitingSystemParams` instance.

    Raises:
        ValueError: If ``omit`` contains unknown keys, or no rod segment remains enabled.
    """
    rng = np.random.default_rng(seed)
    omit_set = _coerce_omit(omit)

    def _s(seg_ranges: dict, key: str) -> float:
        return float(rng.uniform(seg_ranges[key]["min"], seg_ranges[key]["max"]))

    def _si(seg_ranges: dict, key: str) -> int:
        return int(rng.integers(seg_ranges[key]["min"], seg_ranges[key]["max"] + 1))

    # Parent direction for lateral segments: last built rod, or +X when primary is off.
    parent_dir: tuple[float, float, float] = (1.0, 0.0, 0.0)

    primary: RodParams | None = None
    pr = ranges.get("primary")
    if pr is not None and "primary" not in omit_set:
        primary_az = _s(pr, "azimuth_deg")
        primary_el = _s(pr, "elevation_deg")
        primary_dir = _direction_from_angles(primary_az, primary_el)
        primary_bend = _s(pr, "bend_stiffness")
        primary = RodParams(
            num_segments=max(2, _si(pr, "num_segments")),
            length=_s(pr, "length"),
            radius=_s(pr, "radius"),
            bend_stiffness=primary_bend,
            bend_damping=_s(pr, "bend_damping"),
            stretch_stiffness=_s(pr, "stretch_stiffness"),
            density=_s(pr, "density"),
            direction=primary_dir,
        )
        parent_dir = primary.direction

    secondary: RodParams | None = None
    sr = ranges.get("secondary")
    if sr is not None and "secondary" not in omit_set:
        if primary is not None:
            secondary_bend_max = min(sr["bend_stiffness"]["max"], primary.bend_stiffness)
            secondary_bend_min = min(sr["bend_stiffness"]["min"], secondary_bend_max)
            secondary_bend = float(rng.uniform(secondary_bend_min, secondary_bend_max))
        else:
            secondary_bend = float(
                rng.uniform(sr["bend_stiffness"]["min"], sr["bend_stiffness"]["max"])
            )
        secondary_el_delta = _s(sr, "elevation_delta_deg")
        secondary_lat_delta = _s(sr, "lateral_delta_deg")
        secondary_dir = _deflect_direction(parent_dir, secondary_el_delta, secondary_lat_delta)
        secondary = RodParams(
            num_segments=max(2, _si(sr, "num_segments")),
            length=_s(sr, "length"),
            radius=_s(sr, "radius"),
            bend_stiffness=secondary_bend,
            bend_damping=_s(sr, "bend_damping"),
            stretch_stiffness=_s(sr, "stretch_stiffness"),
            density=_s(sr, "density"),
            direction=secondary_dir,
        )
        parent_dir = secondary.direction

    spur: RodParams | None = None
    spr = ranges.get("spur")
    if spr is not None and "spur" not in omit_set:
        spur_el_delta = _s(spr, "elevation_delta_deg")
        spur_lat_delta = _s(spr, "lateral_delta_deg")
        spur_dir = _deflect_direction(parent_dir, spur_el_delta, spur_lat_delta)
        spur = RodParams(
            num_segments=max(2, _si(spr, "num_segments")),
            length=_s(spr, "length"),
            radius=_s(spr, "radius"),
            bend_stiffness=_s(spr, "bend_stiffness"),
            bend_damping=_s(spr, "bend_damping"),
            stretch_stiffness=_s(spr, "stretch_stiffness"),
            density=_s(spr, "density"),
            direction=spur_dir,
        )
        parent_dir = spur.direction

    stem: RodParams | None = None
    stem_r = ranges.get("stem")
    if stem_r is not None and "stem" not in omit_set:
        stem_el_delta = _s(stem_r, "elevation_delta_deg")
        stem_lat_delta = _s(stem_r, "lateral_delta_deg")
        stem_dir = _deflect_direction(parent_dir, stem_el_delta, stem_lat_delta)
        stem = RodParams(
            num_segments=max(2, _si(stem_r, "num_segments")),
            length=_s(stem_r, "length"),
            radius=_s(stem_r, "radius"),
            bend_stiffness=_s(stem_r, "bend_stiffness"),
            bend_damping=_s(stem_r, "bend_damping"),
            stretch_stiffness=_s(stem_r, "stretch_stiffness"),
            density=_s(stem_r, "density"),
            direction=stem_dir,
        )

    apple_radius: float | None = None
    apple_density: float | None = None
    ar = ranges.get("apple")
    if ar is not None and "apple" not in omit_set:
        apple_radius = _s(ar, "radius")
        apple_density = _s(ar, "density")

    if not any((primary, secondary, spur, stem)):
        raise ValueError(
            "At least one rod segment must be enabled (check ranges and omit)."
        )

    topology = _topology_from_ranges(ranges)
    spur_attach_fraction = _spur_attach_fraction_from_ranges(ranges)

    return FruitingSystemParams(
        primary=primary,
        secondary=secondary,
        spur=spur,
        stem=stem,
        apple_radius=apple_radius,
        apple_density=apple_density,
        topology=topology,
        spur_attach_fraction=spur_attach_fraction,
    )


def _fix_topology(
    p: FruitingSystemParams, topo: FruitingSystemParams
) -> FruitingSystemParams:
    """Return ``p`` with ``num_segments`` overridden to match ``topo`` (other params unchanged)."""

    def _rod(r: RodParams | None, t: RodParams | None) -> RodParams | None:
        if r is None or t is None:
            return None
        return dataclasses.replace(r, num_segments=t.num_segments)

    return dataclasses.replace(
        p,
        primary=_rod(p.primary, topo.primary),
        secondary=_rod(p.secondary, topo.secondary),
        spur=_rod(p.spur, topo.spur),
        stem=_rod(p.stem, topo.stem),
        topology=topo.topology,
        spur_attach_fraction=topo.spur_attach_fraction,
    )


def sample_heterogeneous_params_list(
    ranges: dict,
    topology_seed: int,
    num_envs: int,
    *,
    omit: Collection[str] | None = None,
) -> list[FruitingSystemParams]:
    """Sample ``num_envs`` param sets with segment topology fixed to ``topology_seed``."""
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")
    topo = sample_params(ranges, topology_seed, omit=omit)
    return [
        _fix_topology(sample_params(ranges, topology_seed + 1 + w, omit=omit), topo)
        for w in range(num_envs)
    ]


def copy_fruiting_params(params: FruitingSystemParams) -> FruitingSystemParams:
    """Deep-copy sampled params (geometry and stiffness scalars)."""
    def _rod(r: RodParams | None) -> RodParams | None:
        return None if r is None else dataclasses.replace(r)

    return FruitingSystemParams(
        primary=_rod(params.primary),
        secondary=_rod(params.secondary),
        spur=_rod(params.spur),
        stem=_rod(params.stem),
        apple_radius=params.apple_radius,
        apple_density=params.apple_density,
        topology=params.topology,
        spur_attach_fraction=params.spur_attach_fraction,
    )


def perturb_rod_stiffness(
    params: FruitingSystemParams,
    segment: str,
    *,
    bend_delta: float = 0.0,
    stretch_delta: float = 0.0,
) -> FruitingSystemParams:
    """Return a copy with stiffness deltas on one rod segment (geometry unchanged).

    Args:
        params: Nominal fruiting parameters.
        segment: One of ``primary``, ``secondary``, ``spur``, ``stem``.
        bend_delta: Added to ``bend_stiffness`` (must stay positive).
        stretch_delta: Added to ``stretch_stiffness`` (must stay positive).

    Raises:
        ValueError: If the segment is disabled or the result would be non-positive.
    """
    if segment not in ("primary", "secondary", "spur", "stem"):
        raise ValueError(f"Unknown segment {segment!r}")
    out = copy_fruiting_params(params)
    rod = getattr(out, segment)
    if rod is None:
        raise ValueError(f"Segment {segment!r} is disabled in params")
    new_bend = rod.bend_stiffness + bend_delta
    new_stretch = rod.stretch_stiffness + stretch_delta
    if new_bend <= 0.0 or new_stretch <= 0.0:
        raise ValueError(
            f"Stiffness perturbation on {segment!r} must keep bend and stretch positive "
            f"(got bend={new_bend}, stretch={new_stretch})"
        )
    setattr(
        out,
        segment,
        dataclasses.replace(
            rod,
            bend_stiffness=new_bend,
            stretch_stiffness=new_stretch,
        ),
    )
    # if (
    #     out.primary is not None
    #     and out.secondary is not None
    #     and out.primary.bend_stiffness < out.secondary.bend_stiffness
    # ):
    #     raise ValueError(
    #         "primary.bend_stiffness must be >= secondary.bend_stiffness after perturbation"
    #     )
    return out


def set_rod_bend_stiffness(
    params: FruitingSystemParams,
    segment: str,
    bend_stiffness: float,
) -> FruitingSystemParams:
    """Return a copy with absolute ``bend_stiffness`` on one rod segment."""
    if bend_stiffness <= 0.0:
        raise ValueError("bend_stiffness must be positive")
    if segment not in ("primary", "secondary", "spur", "stem"):
        raise ValueError(f"Unknown segment {segment!r}")
    out = copy_fruiting_params(params)
    rod = getattr(out, segment)
    if rod is None:
        raise ValueError(f"Segment {segment!r} is disabled in params")
    setattr(
        out,
        segment,
        dataclasses.replace(rod, bend_stiffness=bend_stiffness),
    )
    return out


def enabled_rod_segments(params: FruitingSystemParams) -> tuple[str, ...]:
    """Rod segment names present in ``params``."""
    return tuple(
        name
        for name in ("primary", "secondary", "spur", "stem")
        if getattr(params, name) is not None
    )


def params_fingerprint(params: FruitingSystemParams) -> dict:
    """Return a dict of scalar summaries from sampled params (no Newton model needed).

    This is cheaper than building the full scene and useful for quick determinism checks.
    Fields for disabled segments are ``None``.
    """
    p, s, sp, st = params.primary, params.secondary, params.spur, params.stem
    return {
        "primary_num_segments": None if p is None else p.num_segments,
        "primary_length": None if p is None else round(p.length, 9),
        "primary_radius": None if p is None else round(p.radius, 9),
        "primary_bend_stiffness": None if p is None else round(p.bend_stiffness, 6),
        "secondary_num_segments": None if s is None else s.num_segments,
        "secondary_length": None if s is None else round(s.length, 9),
        "secondary_radius": None if s is None else round(s.radius, 9),
        "secondary_bend_stiffness": None if s is None else round(s.bend_stiffness, 6),
        "spur_num_segments": None if sp is None else sp.num_segments,
        "spur_length": None if sp is None else round(sp.length, 9),
        "spur_bend_stiffness": None if sp is None else round(sp.bend_stiffness, 6),
        "stem_num_segments": None if st is None else st.num_segments,
        "stem_length": None if st is None else round(st.length, 9),
        "stem_bend_stiffness": None if st is None else round(st.bend_stiffness, 6),
        "apple_radius": None if params.apple_radius is None else round(params.apple_radius, 9),
        "apple_density": None if params.apple_density is None else round(params.apple_density, 6),
        "primary_dir_x": None if p is None else round(p.direction[0], 6),
        "secondary_dir_x": None if s is None else round(s.direction[0], 6),
        "spur_dir_z": None if sp is None else round(sp.direction[2], 6),
        "stem_dir_z": None if st is None else round(st.direction[2], 6),
        "topology": params.topology,
        "spur_attach_fraction": round(params.spur_attach_fraction, 9),
    }


def _topology_from_ranges(ranges: dict) -> str:
    topology = ranges.get("topology", DEFAULT_TOPOLOGY)
    if topology not in ALLOWED_TOPOLOGIES:
        raise ValueError(
            f"unsupported topology {topology!r}; expected one of {sorted(ALLOWED_TOPOLOGIES)}"
        )
    return topology


def _spur_attach_fraction_from_ranges(ranges: dict) -> float:
    raw = ranges.get("spur_attach_fraction", DEFAULT_SPUR_ATTACH_FRACTION)
    fraction = float(raw)
    if not (0.0 < fraction < 1.0):
        raise ValueError(
            f"spur_attach_fraction must lie in (0, 1), got {fraction}"
        )
    return fraction

def _direction_from_angles(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    """Convert azimuth + elevation angles (degrees) to a unit direction vector."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    cos_el = math.cos(el)
    return (cos_el * math.cos(az), cos_el * math.sin(az), math.sin(el))


def _deflect_direction(
    parent_dir: tuple[float, float, float],
    elevation_delta_deg: float,
    lateral_delta_deg: float,
) -> tuple[float, float, float]:
    """Deflect a parent direction by elevation and lateral angle deltas.

    The parent direction's azimuth is shifted by ``lateral_delta_deg`` and
    its elevation is increased by ``elevation_delta_deg``, clamped to [-90, 90] deg.
    """
    dx, dy, dz = float(parent_dir[0]), float(parent_dir[1]), float(parent_dir[2])
    az = math.atan2(dy, dx) + math.radians(lateral_delta_deg)
    el_parent = math.asin(max(-1.0, min(1.0, dz)))
    el_new = el_parent + math.radians(elevation_delta_deg)
    el_new = max(-math.pi / 2.0, min(math.pi / 2.0, el_new))
    cos_el = math.cos(el_new)
    return (cos_el * math.cos(az), cos_el * math.sin(az), math.sin(el_new))


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_ranges(data: dict) -> None:
    """Raise ValueError if the range dict is missing required keys or has invalid bounds."""
    required_segments = ("primary", "secondary", "spur", "stem", "apple")
    for seg in required_segments:
        if seg not in data:
            raise ValueError(f"Missing segment '{seg}' in range file")

    rod_required = (
        "num_segments",
        "length",
        "radius",
        "bend_stiffness",
        "bend_damping",
        "stretch_stiffness",
        "density",
    )
    for seg in ("primary", "secondary", "spur", "stem"):
        seg_data = data[seg]
        if seg_data is None:
            continue
        if not isinstance(seg_data, dict):
            raise ValueError(f"Segment '{seg}' must be a JSON object or null")
        for key in rod_required:
            if key not in seg_data:
                raise ValueError(f"Missing key '{key}' in segment '{seg}'")
            rng = seg_data[key]
            if "min" not in rng or "max" not in rng:
                raise ValueError(f"Range {seg}.{key} must have 'min' and 'max'")
            if rng["min"] > rng["max"]:
                raise ValueError(
                    f"Range {seg}.{key}: min ({rng['min']}) > max ({rng['max']})"
                )

        if seg == "primary":
            for key in ("azimuth_deg", "elevation_deg"):
                if key not in seg_data:
                    raise ValueError(f"Missing key '{key}' in segment 'primary'")
                rng = seg_data[key]
                if "min" not in rng or "max" not in rng:
                    raise ValueError(f"Range primary.{key} must have 'min' and 'max'")
                if rng["min"] > rng["max"]:
                    raise ValueError(
                        f"Range primary.{key}: min ({rng['min']}) > max ({rng['max']})"
                    )
        else:
            for key in ("elevation_delta_deg", "lateral_delta_deg"):
                if key not in seg_data:
                    raise ValueError(f"Missing key '{key}' in segment '{seg}'")
                rng = seg_data[key]
                if "min" not in rng or "max" not in rng:
                    raise ValueError(f"Range {seg}.{key} must have 'min' and 'max'")
                if rng["min"] > rng["max"]:
                    raise ValueError(
                        f"Range {seg}.{key}: min ({rng['min']}) > max ({rng['max']})"
                    )

    apple = data["apple"]
    if apple is not None:
        if not isinstance(apple, dict):
            raise ValueError("Segment 'apple' must be a JSON object or null")
        for key in ("radius", "density"):
            if key not in apple:
                raise ValueError(f"Missing key '{key}' in apple")
            rng = apple[key]
            if "min" not in rng or "max" not in rng:
                raise ValueError(f"Range apple.{key} must have 'min' and 'max'")
            if rng["min"] > rng["max"]:
                raise ValueError(
                    f"Range apple.{key}: min ({rng['min']}) > max ({rng['max']})"
                )

    any_rod = any(data.get(s) is not None for s in ("primary", "secondary", "spur", "stem"))
    if not any_rod:
        raise ValueError(
            "At least one rod segment (primary, secondary, spur, or stem) must be non-null in the range file"
        )

    topology = data.get("topology", DEFAULT_TOPOLOGY)
    if topology not in ALLOWED_TOPOLOGIES:
        raise ValueError(
            f"unsupported topology {topology!r}; expected one of {sorted(ALLOWED_TOPOLOGIES)}"
        )

    if "spur_attach_fraction" in data:
        fraction = float(data["spur_attach_fraction"])
        if not (0.0 < fraction < 1.0):
            raise ValueError(
                f"spur_attach_fraction must lie in (0, 1), got {fraction}"
            )

    args = data.get("args")
    if args is None:
        return
    if not isinstance(args, dict):
        raise ValueError("args must be a JSON object")
    for key in ("fruiting_base_pos", "robot_base_pos"):
        if key not in args:
            continue
        val = args[key]
        if val is None:
            continue
        _coerce_xyz_triplet(val, field=key)
