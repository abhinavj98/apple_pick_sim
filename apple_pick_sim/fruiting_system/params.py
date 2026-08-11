"""Fruiting-system parameters, sampling, and range validation."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Collection
from pathlib import Path
from typing import Any, Literal

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
DEFAULT_SPUR_SURFACE_OFFSET = True
OVERLAP_DIRECTION_THRESHOLD: float = 0.75
REAL_WORLD_PROXY_VARIANCE_FIXTURE = "fruiting_system_ranges_real_world_proxy_variance.json"

@dataclasses.dataclass
class RodParams:
    """Sampled parameters for a single rod segment in the fruiting chain."""

    num_segments: int
    length: float
    radius: float
    youngs_modulus_pa: float
    damping_ratio: float
    bend_stiffness: float
    bend_damping: float
    stretch_stiffness: float
    stretch_damping: float
    density: float
    direction: tuple[float, float, float]  # unit vector in world space


_LEGACY_ROD_STIFFNESS_RANGE_KEYS = frozenset(
    {"bend_stiffness", "bend_damping", "stretch_stiffness"}
)

_VBD_STRETCH_FORCE_KEYS = frozenset({"max_force_n", "damping_ratio"})

# Axial extension budget under max_force_n: δ = fraction * L_seg.
VBD_STRETCH_EXTENSION_FRACTION: float = 0.05


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
    apple_quat_xyzw: tuple[float, float, float, float] | None = None
    """Initial apple body orientation ``(x, y, z, w)``; ``None`` → identity.

    Real pre-grasp rebuild sets this from the logged tracker pose so the
    stem–apple FIXED child anchor is baked in the marker frame.
    """
    topology: str = DEFAULT_TOPOLOGY
    spur_attach_fraction: float = DEFAULT_SPUR_ATTACH_FRACTION
    spur_surface_offset: bool = DEFAULT_SPUR_SURFACE_OFFSET


FRUITING_SYSTEM_PARAMS_SCHEMA = "fruiting_system_params_v2"
FRUITING_SYSTEM_PARAMS_SCHEMA_V1 = "fruiting_system_params_v1"


def _segment_material_geometry(
    radius: float, length: float, num_segments: int, density: float
) -> tuple[float, float, float, float, float]:
    """Return ``(A, I, L_seg, m_seg, J_seg)`` for a circular rod segment."""
    r = float(radius)
    n = max(2, int(num_segments))
    l_seg = float(length) / n
    area = math.pi * r * r
    inertia = math.pi * r**4 / 4.0
    m_seg = float(density) * area * l_seg
    j_seg = m_seg * (3.0 * r * r + l_seg * l_seg) / 12.0
    return area, inertia, l_seg, m_seg, j_seg


# Axial stretch from max force + ζ (decoupled from bend youngs_modulus_pa).


def stretch_knobs_from_max_force(
    max_force_n: float,
    damping_ratio_axial: float,
    length: float,
    radius: float,
    density: float,
    num_segments: int,
    *,
    extension_fraction: float = VBD_STRETCH_EXTENSION_FRACTION,
) -> tuple[float, float]:
    """Derive axial VBD ``(stretch_stiffness, stretch_damping)`` from force budget.

    Uses ``k = F_max / (extension_fraction * L_seg)`` and
    ``c = 2 ζ_stretch √(k m_seg)``. Bend knobs stay on ``youngs_modulus_pa``.
    """
    if float(max_force_n) <= 0.0:
        raise ValueError("max_force_n must be positive")
    if float(damping_ratio_axial) <= 0.0:
        raise ValueError("damping_ratio_axial must be positive")
    if float(extension_fraction) <= 0.0:
        raise ValueError("extension_fraction must be positive")
    _area, _inertia, l_seg, m_seg, _j_seg = _segment_material_geometry(
        radius, length, num_segments, density
    )
    if l_seg <= 0.0:
        raise ValueError("segment length must be positive")
    k_stretch = float(max_force_n) / (float(extension_fraction) * l_seg)
    c_stretch = 2.0 * float(damping_ratio_axial) * math.sqrt(k_stretch * m_seg)
    return k_stretch, c_stretch


def rod_params_from_material(
    youngs_modulus_pa: float,
    damping_ratio: float,
    length: float,
    radius: float,
    density: float,
    num_segments: int,
    direction: tuple[float, float, float],
    *,
    stretch_stiffness: float | None = None,
    stretch_damping: float | None = None,
) -> RodParams:
    """Build :class:`RodParams` from material properties and geometry.

    Derives VBD stiffness/damping from circular-rod beam theory (see
    ``docs/material-parameter-sampling.md``). Optional ``stretch_stiffness`` and
    ``stretch_damping`` override the axial knobs (e.g. from JSON ``vbd_stretch_force``);
    bend knobs always follow sampled ``youngs_modulus_pa`` and ``damping_ratio``.
    """
    if youngs_modulus_pa <= 0.0:
        raise ValueError("youngs_modulus_pa must be positive")
    if damping_ratio < 0.0:
        raise ValueError("damping_ratio must be non-negative")
    n = max(2, int(num_segments))
    area, inertia, l_seg, m_seg, j_seg = _segment_material_geometry(
        radius, length, n, density
    )
    e = float(youngs_modulus_pa)
    zeta = float(damping_ratio)
    bend_stiffness = e * inertia / l_seg
    if stretch_stiffness is None:
        stretch_stiffness = e * area / l_seg
    if stretch_damping is None:
        stretch_damping = 2.0 * zeta * math.sqrt(stretch_stiffness * m_seg)
    bend_damping = 2.0 * zeta * math.sqrt(bend_stiffness * j_seg) 
    return RodParams(
        num_segments=n,
        length=float(length),
        radius=float(radius),
        youngs_modulus_pa=e,
        damping_ratio=zeta,
        bend_stiffness=bend_stiffness,
        bend_damping=bend_damping,
        stretch_stiffness=float(stretch_stiffness),
        stretch_damping=float(stretch_damping),
        density=float(density),
        direction=direction,
    )


def rod_params_from_vbd_targets(
    *,
    num_segments: int,
    length: float,
    radius: float,
    bend_stiffness: float,
    bend_damping: float,
    stretch_stiffness: float,
    density: float,
    direction: tuple[float, float, float],
    stretch_damping: float | None = None,
) -> RodParams:
    """Build :class:`RodParams` from explicit VBD targets (tests, legacy tooling).

    Back-computes ``youngs_modulus_pa`` and ``damping_ratio`` for storage; preserves
    the supplied bend/stretch stiffness and damping values exactly.
    """
    n = max(2, int(num_segments))
    area, inertia, l_seg, m_seg, j_seg = _segment_material_geometry(
        radius, length, n, density
    )
    e = bend_stiffness * l_seg / inertia if inertia > 0.0 else 0.0
    zeta = (
        bend_damping / (2.0 * math.sqrt(bend_stiffness * j_seg))
        if bend_stiffness > 0.0 and j_seg > 0.0
        else 0.0
    )
    if stretch_damping is None:
        stretch_damping = 2.0 * zeta * math.sqrt(stretch_stiffness * m_seg)
    return RodParams(
        num_segments=n,
        length=float(length),
        radius=float(radius),
        youngs_modulus_pa=e,
        damping_ratio=zeta,
        bend_stiffness=float(bend_stiffness),
        bend_damping=float(bend_damping),
        stretch_stiffness=float(stretch_stiffness),
        stretch_damping=float(stretch_damping),
        density=float(density),
        direction=direction,
    )


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
        "youngs_modulus_pa": float(rod.youngs_modulus_pa),
        "damping_ratio": float(rod.damping_ratio),
        "bend_stiffness": float(rod.bend_stiffness),
        "bend_damping": float(rod.bend_damping),
        "stretch_stiffness": float(rod.stretch_stiffness),
        "stretch_damping": float(rod.stretch_damping),
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
        "apple_quat_xyzw": (
            None
            if params.apple_quat_xyzw is None
            else [float(x) for x in params.apple_quat_xyzw]
        ),
        "topology": params.topology,
        "spur_attach_fraction": float(params.spur_attach_fraction),
        "spur_surface_offset": bool(params.spur_surface_offset),
    }


def fruiting_params_to_json(params: FruitingSystemParams) -> str:
    """Serialize sampled fruiting params for episode metadata storage."""
    return json.dumps(fruiting_params_to_dict(params), sort_keys=True, separators=(",", ":"))


def _expect_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _infer_material_from_vbd_row(row: dict[str, Any]) -> tuple[float, float, float]:
    """Back-compute ``(E, zeta, stretch_damping)`` from stored VBD scalars (v1 episodes)."""
    n = max(2, int(row["num_segments"]))
    r = float(row["radius"])
    length = float(row["length"])
    rho = float(row["density"])
    bend_stiffness = float(row["bend_stiffness"])
    bend_damping = float(row["bend_damping"])
    stretch_stiffness = float(row["stretch_stiffness"])
    if "youngs_modulus_pa" in row and "damping_ratio" in row:
        e = float(row["youngs_modulus_pa"])
        zeta = float(row["damping_ratio"])
        stretch_damping = float(row.get("stretch_damping", 0.0))
        return e, zeta, stretch_damping
    area, inertia, l_seg, m_seg, j_seg = _segment_material_geometry(r, length, n, rho)
    e = bend_stiffness * l_seg / inertia if inertia > 0.0 else 0.0
    zeta = (
        bend_damping / (2.0 * math.sqrt(bend_stiffness * j_seg))
        if bend_stiffness > 0.0 and j_seg > 0.0
        else 0.0
    )
    stretch_damping = float(row.get("stretch_damping", 2.0 * zeta * math.sqrt(stretch_stiffness * m_seg)))
    return e, zeta, stretch_damping


def _rod_params_from_row(value: Any, *, field: str) -> RodParams | None:
    if value is None:
        return None
    row = _expect_mapping(value, field=field)
    direction = row.get("direction")
    if not isinstance(direction, (list, tuple)) or len(direction) != 3:
        raise ValueError(f"{field}.direction must be [x, y, z]")
    e, zeta, stretch_damping = _infer_material_from_vbd_row(row)
    return RodParams(
        num_segments=int(row["num_segments"]),
        length=float(row["length"]),
        radius=float(row["radius"]),
        youngs_modulus_pa=e,
        damping_ratio=zeta,
        bend_stiffness=float(row["bend_stiffness"]),
        bend_damping=float(row["bend_damping"]),
        stretch_stiffness=float(row["stretch_stiffness"]),
        stretch_damping=stretch_damping,
        density=float(row["density"]),
        direction=(float(direction[0]), float(direction[1]), float(direction[2])),
    )


def fruiting_params_from_dict(data: dict[str, Any]) -> FruitingSystemParams:
    """Deserialize :func:`fruiting_params_to_dict` output."""
    row = _expect_mapping(data, field="fruiting_system_params")
    schema = row.get("schema")
    if schema not in (FRUITING_SYSTEM_PARAMS_SCHEMA, FRUITING_SYSTEM_PARAMS_SCHEMA_V1):
        raise ValueError(
            f"unsupported fruiting params schema {schema!r}; "
            f"expected {FRUITING_SYSTEM_PARAMS_SCHEMA!r} or "
            f"{FRUITING_SYSTEM_PARAMS_SCHEMA_V1!r}"
        )
    topology = row.get("topology", DEFAULT_TOPOLOGY)
    if topology not in ALLOWED_TOPOLOGIES:
        raise ValueError(
            f"unsupported topology {topology!r}; expected one of {sorted(ALLOWED_TOPOLOGIES)}"
        )
    spur_attach_fraction = float(
        row.get("spur_attach_fraction", DEFAULT_SPUR_ATTACH_FRACTION)
    )
    spur_surface_offset = bool(row.get("spur_surface_offset", False))
    apple_quat_raw = row.get("apple_quat_xyzw")
    apple_quat_xyzw: tuple[float, float, float, float] | None = None
    if apple_quat_raw is not None:
        if not isinstance(apple_quat_raw, (list, tuple)) or len(apple_quat_raw) != 4:
            raise ValueError("apple_quat_xyzw must be a length-4 list [x, y, z, w]")
        apple_quat_xyzw = (
            float(apple_quat_raw[0]),
            float(apple_quat_raw[1]),
            float(apple_quat_raw[2]),
            float(apple_quat_raw[3]),
        )
    params = FruitingSystemParams(
        primary=_rod_params_from_row(row.get("primary"), field="primary"),
        secondary=_rod_params_from_row(row.get("secondary"), field="secondary"),
        spur=_rod_params_from_row(row.get("spur"), field="spur"),
        stem=_rod_params_from_row(row.get("stem"), field="stem"),
        apple_radius=None if row.get("apple_radius") is None else float(row["apple_radius"]),
        apple_density=None if row.get("apple_density") is None else float(row["apple_density"]),
        apple_quat_xyzw=apple_quat_xyzw,
        topology=topology,
        spur_attach_fraction=spur_attach_fraction,
        spur_surface_offset=spur_surface_offset,
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
    uses consistent inertia. Default shape is a cylinder (50 mm radius, 180 mm length)
    with the distal tip at the body origin (TCP); bulk extends along local −Z toward
    the flange (+Z is tip-out, matching the USD / recorded TCP).
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
    weld_reference_stem_dir: tuple[float, float, float] | None = None
    """Optional settled stem direction for hemisphere checks with ``weld_reference_pos``.

    When set, robot-facing weld validation uses this unit vector instead of the
  nominal build-time ``proxy_placement_dir`` (settle-then-weld workflows).
    """
    weld_proxy_offset_in_apple_frame: (
        tuple[float, float, float, float, float, float, float] | None
    ) = None
    """Explicit FIXED ``parent_xform`` ``(px,py,pz,qx,qy,qz,qw)`` in the apple frame.

    When set with ``fix_to_apple=True``, bypasses look-at / ``weld_direction``
    orientation. Used by post-grasp replay to encode true TCP SE(3) relative to
    the apple (``X_offset = X_apple^{-1} X_tcp``). Requires ``fix_to_apple``.
    """

@dataclasses.dataclass(frozen=True)
class FixtureArgs:
    """Scene placement from a ranges JSON ``args`` block."""

    fruiting_base_pos: tuple[float, float, float] | None = None
    robot_base_pos: tuple[float, float, float] | None = None


_SIM_BUILD_JOINT_ROLES = frozenset({"support", "primary_spur", "spur_stem", "stem_apple"})
_SIM_BUILD_ALLOWED_KEYS = frozenset(
    {
        "vic_gains",
        "joint_damping_ratio",
        "joint_angular_kd_overrides",
        "joint_linear_kd_overrides",
        "joint_angular_kp_overrides",
        "joint_linear_kp_overrides",
    }
)
_VIC_GAIN_KEYS = ("linear_k", "linear_d", "angular_k", "angular_d")


@dataclasses.dataclass(frozen=True)
class VicGainsConfig:
    """TCP impedance gains from an optional ranges JSON ``sim_build.vic_gains`` block."""

    linear_k: float
    linear_d: float
    angular_k: float
    angular_d: float


@dataclasses.dataclass(frozen=True)
class SimBuildConfig:
    """Optional sim-build knobs from a ranges JSON ``sim_build`` block."""

    vic_gains: VicGainsConfig
    joint_angular_kd_overrides: dict[str, float] = dataclasses.field(default_factory=dict)
    joint_linear_kd_overrides: dict[str, float] = dataclasses.field(default_factory=dict)
    joint_angular_kp_overrides: dict[str, float] = dataclasses.field(default_factory=dict)
    joint_linear_kp_overrides: dict[str, float] = dataclasses.field(default_factory=dict)
    joint_damping_ratio: float | None = None


def _coerce_xyz_triplet(raw: object, *, field: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"args.{field} must be [x, y, z]")
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _coerce_nonnegative_float(raw: object, *, field: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite float >= 0") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{field} must be a finite float >= 0, got {raw!r}")
    return value


def _coerce_unit_interval_float(raw: object, *, field: str) -> float:
    """Finite float in ``[0, 1]``."""
    value = _coerce_nonnegative_float(raw, field=field)
    if value > 1.0:
        raise ValueError(f"{field} must be <= 1, got {raw!r}")
    return value


def _coerce_joint_overrides(raw: object, *, field: str) -> dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"sim_build.{field} must be a JSON object")
    out: dict[str, float] = {}
    for role, value in raw.items():
        if role not in _SIM_BUILD_JOINT_ROLES:
            raise ValueError(
                f"sim_build.{field} has unknown joint role {role!r}; "
                f"expected one of {sorted(_SIM_BUILD_JOINT_ROLES)}"
            )
        out[str(role)] = _coerce_nonnegative_float(
            value, field=f"sim_build.{field}.{role}"
        )
    return out


def _coerce_vic_gains(raw: object) -> VicGainsConfig:
    if not isinstance(raw, dict):
        raise ValueError("sim_build.vic_gains must be a JSON object")
    missing = [k for k in _VIC_GAIN_KEYS if k not in raw]
    if missing:
        raise ValueError(f"sim_build.vic_gains missing required keys: {missing}")
    unknown = sorted(set(raw) - set(_VIC_GAIN_KEYS))
    if unknown:
        raise ValueError(f"sim_build.vic_gains has unknown keys: {unknown}")
    return VicGainsConfig(
        **{
            key: _coerce_nonnegative_float(raw[key], field=f"sim_build.vic_gains.{key}")
            for key in _VIC_GAIN_KEYS
        }
    )


def _coerce_joint_damping_ratio(raw: object) -> float | None:
    if raw is None:
        return None
    return _coerce_nonnegative_float(raw, field="sim_build.joint_damping_ratio")


def _validate_sim_build(block: object) -> None:
    """Raise ValueError if ``sim_build`` is present but invalid."""
    if block is None:
        return
    if not isinstance(block, dict):
        raise ValueError("sim_build must be a JSON object")
    unknown = sorted(set(block) - _SIM_BUILD_ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"sim_build has unknown keys: {unknown}")
    if "vic_gains" not in block:
        raise ValueError("sim_build.vic_gains is required when sim_build is present")
    _coerce_vic_gains(block["vic_gains"])
    zeta = _coerce_joint_damping_ratio(block.get("joint_damping_ratio"))
    for field in (
        "joint_angular_kd_overrides",
        "joint_linear_kd_overrides",
        "joint_angular_kp_overrides",
        "joint_linear_kp_overrides",
    ):
        if field in block:
            _coerce_joint_overrides(block[field], field=field)
    ang_kd = _coerce_joint_overrides(
        block.get("joint_angular_kd_overrides"), field="joint_angular_kd_overrides"
    )
    lin_kd = _coerce_joint_overrides(
        block.get("joint_linear_kd_overrides"), field="joint_linear_kd_overrides"
    )
    if zeta is not None and (ang_kd or lin_kd):
        raise ValueError(
            "sim_build.joint_damping_ratio is mutually exclusive with "
            "joint_angular_kd_overrides / joint_linear_kd_overrides"
        )


def parse_sim_build(ranges: dict) -> SimBuildConfig | None:
    """Return ``sim_build`` knobs from ``ranges``, or ``None`` when absent."""
    block = ranges.get("sim_build")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError("sim_build must be a JSON object")
    _validate_sim_build(block)
    return SimBuildConfig(
        vic_gains=_coerce_vic_gains(block["vic_gains"]),
        joint_angular_kd_overrides=_coerce_joint_overrides(
            block.get("joint_angular_kd_overrides"), field="joint_angular_kd_overrides"
        ),
        joint_linear_kd_overrides=_coerce_joint_overrides(
            block.get("joint_linear_kd_overrides"), field="joint_linear_kd_overrides"
        ),
        joint_angular_kp_overrides=_coerce_joint_overrides(
            block.get("joint_angular_kp_overrides"), field="joint_angular_kp_overrides"
        ),
        joint_linear_kp_overrides=_coerce_joint_overrides(
            block.get("joint_linear_kp_overrides"), field="joint_linear_kp_overrides"
        ),
        joint_damping_ratio=_coerce_joint_damping_ratio(block.get("joint_damping_ratio")),
    )


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
        optionally ``args`` (``fruiting_base_pos`` / ``robot_base_pos`` as ``[x, y, z]``)
        and ``sim_build`` (VIC gains + joint kp/kd overrides).
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
    ``primary.youngs_modulus_pa >= secondary.youngs_modulus_pa``.

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
        primary_length = _s(pr, "length")
        primary_radius = _s(pr, "radius")
        primary_density = _s(pr, "density")
        primary_n = max(2, _si(pr, "num_segments"))
        primary = rod_params_from_material(
            _s(pr, "youngs_modulus_pa"),
            _s(pr, "damping_ratio"),
            primary_length,
            primary_radius,
            primary_density,
            primary_n,
            primary_dir,
            **_stretch_kw_from_seg_ranges(
                pr,
                length=primary_length,
                radius=primary_radius,
                density=primary_density,
                num_segments=primary_n,
            ),
        )
        parent_dir = primary.direction

    secondary: RodParams | None = None
    sr = ranges.get("secondary")
    if sr is not None and "secondary" not in omit_set:
        if primary is not None:
            secondary_e_max = min(
                sr["youngs_modulus_pa"]["max"], primary.youngs_modulus_pa
            )
            secondary_e_min = min(sr["youngs_modulus_pa"]["min"], secondary_e_max)
            secondary_e = float(rng.uniform(secondary_e_min, secondary_e_max))
        else:
            secondary_e = float(
                rng.uniform(
                    sr["youngs_modulus_pa"]["min"], sr["youngs_modulus_pa"]["max"]
                )
            )
        secondary_el_delta = _s(sr, "elevation_delta_deg")
        secondary_lat_delta = _s(sr, "lateral_delta_deg")
        secondary_dir = _deflect_direction(parent_dir, secondary_el_delta, secondary_lat_delta)
        secondary_length = _s(sr, "length")
        secondary_radius = _s(sr, "radius")
        secondary_density = _s(sr, "density")
        secondary_n = max(2, _si(sr, "num_segments"))
        secondary = rod_params_from_material(
            secondary_e,
            _s(sr, "damping_ratio"),
            secondary_length,
            secondary_radius,
            secondary_density,
            secondary_n,
            secondary_dir,
            **_stretch_kw_from_seg_ranges(
                sr,
                length=secondary_length,
                radius=secondary_radius,
                density=secondary_density,
                num_segments=secondary_n,
            ),
        )
        parent_dir = secondary.direction

    spur: RodParams | None = None
    spr = ranges.get("spur")
    if spr is not None and "spur" not in omit_set:
        spur_el_delta = _s(spr, "elevation_delta_deg")
        spur_lat_delta = _s(spr, "lateral_delta_deg")
        spur_dir = _deflect_direction(parent_dir, spur_el_delta, spur_lat_delta)
        spur_length = _s(spr, "length")
        spur_radius = _s(spr, "radius")
        spur_density = _s(spr, "density")
        spur_n = max(2, _si(spr, "num_segments"))
        spur = rod_params_from_material(
            _s(spr, "youngs_modulus_pa"),
            _s(spr, "damping_ratio"),
            spur_length,
            spur_radius,
            spur_density,
            spur_n,
            spur_dir,
            **_stretch_kw_from_seg_ranges(
                spr,
                length=spur_length,
                radius=spur_radius,
                density=spur_density,
                num_segments=spur_n,
            ),
        )
        parent_dir = spur.direction

    stem: RodParams | None = None
    stem_r = ranges.get("stem")
    if stem_r is not None and "stem" not in omit_set:
        stem_el_delta = _s(stem_r, "elevation_delta_deg")
        stem_lat_delta = _s(stem_r, "lateral_delta_deg")
        stem_dir = _deflect_direction(parent_dir, stem_el_delta, stem_lat_delta)
        stem_length = _s(stem_r, "length")
        stem_radius = _s(stem_r, "radius")
        stem_density = _s(stem_r, "density")
        stem_n = max(2, _si(stem_r, "num_segments"))
        stem = rod_params_from_material(
            _s(stem_r, "youngs_modulus_pa"),
            _s(stem_r, "damping_ratio"),
            stem_length,
            stem_radius,
            stem_density,
            stem_n,
            stem_dir,
            **_stretch_kw_from_seg_ranges(
                stem_r,
                length=stem_length,
                radius=stem_radius,
                density=stem_density,
                num_segments=stem_n,
            ),
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
    spur_surface_offset = _spur_surface_offset_from_ranges(ranges)

    return FruitingSystemParams(
        primary=primary,
        secondary=secondary,
        spur=spur,
        stem=stem,
        apple_radius=apple_radius,
        apple_density=apple_density,
        topology=topology,
        spur_attach_fraction=spur_attach_fraction,
        spur_surface_offset=spur_surface_offset,
    )


def _dot3(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def branches_overlap_by_direction(
    params: FruitingSystemParams,
    threshold: float = OVERLAP_DIRECTION_THRESHOLD,
) -> bool:
    """Return True when any child branch direction risks growing into its parent body.

    T-junction:
      spur vs primary — ``|dot| > threshold`` (parallel or anti-parallel both bad at mid-span)
      stem vs spur — ``dot < -threshold`` (anti-parallel only)

    Linear chain:
      each consecutive enabled pair — ``dot(child, parent) < -threshold`` only
    """
    thr = float(threshold)

    if params.topology == TOPOLOGY_T_JUNCTION:
        primary = params.primary
        spur = params.spur
        stem = params.stem
        if primary is not None and spur is not None:
            if abs(_dot3(spur.direction, primary.direction)) > thr:
                return True
        if spur is not None and stem is not None:
            if _dot3(stem.direction, spur.direction) < -thr:
                return True
        return False

    if params.topology != TOPOLOGY_LINEAR_CHAIN:
        raise ValueError(
            f"unsupported topology {params.topology!r}; expected one of "
            f"{sorted(ALLOWED_TOPOLOGIES)}"
        )

    parent: RodParams | None = None
    for name in ("primary", "secondary", "spur", "stem"):
        child = getattr(params, name)
        if child is None:
            continue
        if parent is not None and _dot3(child.direction, parent.direction) < -thr:
            return True
        parent = child
    return False


def sample_params_no_overlap(
    ranges: dict,
    seed: int,
    *,
    threshold: float = OVERLAP_DIRECTION_THRESHOLD,
    max_retries: int = 20,
    omit: Collection[str] | None = None,
) -> FruitingSystemParams:
    """Sample fruiting params, retrying with shifted seeds until directions do not overlap."""
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    for attempt in range(max_retries):
        candidate = sample_params(ranges, seed + attempt * 997, omit=omit)
        if not branches_overlap_by_direction(candidate, threshold=threshold):
            return candidate
    raise RuntimeError(
        f"Could not sample non-overlapping fruiting geometry in {max_retries} attempts "
        f"(seed={seed}, threshold={threshold})"
    )


def _sample_params_for_heterogeneous(
    ranges: dict,
    seed: int,
    *,
    omit: Collection[str] | None,
    overlap_threshold: float | None,
) -> FruitingSystemParams:
    if overlap_threshold is None:
        return sample_params(ranges, seed, omit=omit)
    return sample_params_no_overlap(ranges, seed, threshold=overlap_threshold, omit=omit)


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
        spur_surface_offset=topo.spur_surface_offset,
    )


def sample_heterogeneous_params_list(
    ranges: dict,
    topology_seed: int,
    num_envs: int,
    *,
    omit: Collection[str] | None = None,
    overlap_threshold: float | None = None,
) -> list[FruitingSystemParams]:
    """Sample ``num_envs`` param sets with segment topology fixed to ``topology_seed``."""
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")
    topo = sample_params(ranges, topology_seed, omit=omit)
    return [
        _fix_topology(
            _sample_params_for_heterogeneous(
                ranges,
                topology_seed + 1 + w,
                omit=omit,
                overlap_threshold=overlap_threshold,
            ),
            topo,
        )
        for w in range(num_envs)
    ]


def resample_heterogeneous_params_for_worlds(
    ranges: dict,
    topology_seed: int,
    params_list: list[FruitingSystemParams],
    worlds: Collection[int],
    *,
    resample_seed: int,
    omit: Collection[str] | None = None,
    overlap_threshold: float | None = None,
) -> list[FruitingSystemParams]:
    """Return a copy of ``params_list`` with DR re-drawn for the listed env indices."""
    if not worlds:
        return list(params_list)
    topo = sample_params(ranges, topology_seed, omit=omit)
    out = list(params_list)
    for i, world in enumerate(sorted(int(w) for w in worlds)):
        if world < 0 or world >= len(out):
            raise ValueError(f"world index {world} out of range for params_list len {len(out)}")
        out[world] = _fix_topology(
            _sample_params_for_heterogeneous(
                ranges,
                int(resample_seed) + i,
                omit=omit,
                overlap_threshold=overlap_threshold,
            ),
            topo,
        )
    return out


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
        apple_quat_xyzw=params.apple_quat_xyzw,
        topology=params.topology,
        spur_attach_fraction=params.spur_attach_fraction,
        spur_surface_offset=params.spur_surface_offset,
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
    _area, inertia, l_seg, _m_seg, j_seg = _segment_material_geometry(
        rod.radius, rod.length, rod.num_segments, rod.density
    )
    k = float(bend_stiffness)
    new_bend_damping = 2.0 * rod.damping_ratio * math.sqrt(k * j_seg)
    new_youngs_modulus_pa = k * l_seg / inertia if inertia > 0.0 else 0.0
    setattr(
        out,
        segment,
        dataclasses.replace(
            rod,
            bend_stiffness=k,
            bend_damping=new_bend_damping,
            youngs_modulus_pa=new_youngs_modulus_pa,
        ),
    )
    return out


def _rod_has_fixed_axial_stretch(rod: RodParams, *, rel_tol: float = 1e-5) -> bool:
    """True when stored stretch knobs differ from beam-theory stretch at current ``E``."""
    beam = rod_params_from_material(
        rod.youngs_modulus_pa,
        rod.damping_ratio,
        rod.length,
        rod.radius,
        rod.density,
        rod.num_segments,
        rod.direction,
    )
    return not (
        math.isclose(
            rod.stretch_stiffness,
            beam.stretch_stiffness,
            rel_tol=rel_tol,
            abs_tol=1e-9,
        )
        and math.isclose(
            rod.stretch_damping,
            beam.stretch_damping,
            rel_tol=rel_tol,
            abs_tol=1e-9,
        )
    )


def set_rod_youngs_modulus(
    params: FruitingSystemParams,
    segment: str,
    youngs_modulus_pa: float,
) -> FruitingSystemParams:
    """Return a copy with absolute Young's modulus on one rod segment.

    Re-derives bend (and, when stretch was beam-consistent, stretch) via
    :func:`rod_params_from_material`. Freezes geometry and ``damping_ratio``.
    If the base rod's axial stretch differs from beam theory (e.g. fixture
    ``vbd_stretch_force``), those stretch knobs are preserved.
    """
    if youngs_modulus_pa <= 0.0:
        raise ValueError("youngs_modulus_pa must be positive")
    if segment not in ("primary", "secondary", "spur", "stem"):
        raise ValueError(f"Unknown segment {segment!r}")
    out = copy_fruiting_params(params)
    rod = getattr(out, segment)
    if rod is None:
        raise ValueError(f"Segment {segment!r} is disabled in params")
    stretch_kw: dict[str, float] = {}
    if _rod_has_fixed_axial_stretch(rod):
        stretch_kw = {
            "stretch_stiffness": float(rod.stretch_stiffness),
            "stretch_damping": float(rod.stretch_damping),
        }
    new_rod = rod_params_from_material(
        float(youngs_modulus_pa),
        rod.damping_ratio,
        rod.length,
        rod.radius,
        rod.density,
        rod.num_segments,
        rod.direction,
        **stretch_kw,
    )
    setattr(out, segment, new_rod)
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
        "primary_youngs_modulus_pa": None if p is None else round(p.youngs_modulus_pa, 6),
        "primary_damping_ratio": None if p is None else round(p.damping_ratio, 8),
        "secondary_num_segments": None if s is None else s.num_segments,
        "secondary_length": None if s is None else round(s.length, 9),
        "secondary_radius": None if s is None else round(s.radius, 9),
        "secondary_youngs_modulus_pa": None if s is None else round(s.youngs_modulus_pa, 6),
        "secondary_damping_ratio": None if s is None else round(s.damping_ratio, 8),
        "spur_num_segments": None if sp is None else sp.num_segments,
        "spur_length": None if sp is None else round(sp.length, 9),
        "spur_youngs_modulus_pa": None if sp is None else round(sp.youngs_modulus_pa, 6),
        "spur_damping_ratio": None if sp is None else round(sp.damping_ratio, 8),
        "stem_num_segments": None if st is None else st.num_segments,
        "stem_length": None if st is None else round(st.length, 9),
        "stem_youngs_modulus_pa": None if st is None else round(st.youngs_modulus_pa, 6),
        "stem_damping_ratio": None if st is None else round(st.damping_ratio, 8),
        "apple_radius": None if params.apple_radius is None else round(params.apple_radius, 9),
        "apple_density": None if params.apple_density is None else round(params.apple_density, 6),
        "apple_quat_xyzw": (
            None
            if params.apple_quat_xyzw is None
            else [round(float(x), 9) for x in params.apple_quat_xyzw]
        ),
        "primary_dir_x": None if p is None else round(p.direction[0], 6),
        "secondary_dir_x": None if s is None else round(s.direction[0], 6),
        "spur_dir_z": None if sp is None else round(sp.direction[2], 6),
        "stem_dir_z": None if st is None else round(st.direction[2], 6),
        "topology": params.topology,
        "spur_attach_fraction": round(params.spur_attach_fraction, 9),
        "spur_surface_offset": params.spur_surface_offset,
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


def _spur_surface_offset_from_ranges(ranges: dict) -> bool:
    raw = ranges.get("spur_surface_offset", DEFAULT_SPUR_SURFACE_OFFSET)
    if not isinstance(raw, bool):
        raise ValueError(f"spur_surface_offset must be a boolean, got {raw!r}")
    return raw


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


def _parse_vbd_stretch_force(seg_data: dict) -> tuple[float, float] | None:
    """Return ``(max_force_n, damping_ratio)`` from ``vbd_stretch_force``, if present."""
    if "vbd_stretch_fixed" in seg_data:
        raise ValueError(
            "vbd_stretch_fixed is removed; use vbd_stretch_force with "
            "max_force_n and damping_ratio instead"
        )
    block = seg_data.get("vbd_stretch_force")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError("vbd_stretch_force must be a JSON object")
    missing = _VBD_STRETCH_FORCE_KEYS - block.keys()
    if missing:
        raise ValueError(
            f"vbd_stretch_force missing required keys: {sorted(missing)}"
        )
    extra = set(block.keys()) - _VBD_STRETCH_FORCE_KEYS
    if extra:
        raise ValueError(
            f"vbd_stretch_force has unknown keys: {sorted(extra)}"
        )
    f_max = float(block["max_force_n"])
    zeta = float(block["damping_ratio"])
    if f_max <= 0.0:
        raise ValueError("vbd_stretch_force.max_force_n must be positive")
    if zeta <= 0.0:
        raise ValueError("vbd_stretch_force.damping_ratio must be positive")
    return f_max, zeta


def _stretch_kw_from_seg_ranges(
    seg_data: dict,
    *,
    length: float,
    radius: float,
    density: float,
    num_segments: int,
) -> dict[str, float]:
    """Keyword args for :func:`rod_params_from_material` stretch overrides."""
    force = _parse_vbd_stretch_force(seg_data)
    if force is None:
        return {}
    k_stretch, c_stretch = stretch_knobs_from_max_force(
        force[0],
        force[1],
        length,
        radius,
        density,
        num_segments,
    )
    return {"stretch_stiffness": k_stretch, "stretch_damping": c_stretch}


def _validate_vbd_stretch_force(seg: str, seg_data: dict) -> None:
    """Validate optional ``vbd_stretch_force`` (raises on bad shape / legacy key)."""
    if "vbd_stretch_fixed" in seg_data:
        raise ValueError(
            f"Segment '{seg}' uses removed key vbd_stretch_fixed; "
            "use vbd_stretch_force instead"
        )
    if "vbd_stretch_force" not in seg_data:
        return
    _parse_vbd_stretch_force(seg_data)


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
        "youngs_modulus_pa",
        "damping_ratio",
        "density",
    )
    for seg in ("primary", "secondary", "spur", "stem"):
        seg_data = data[seg]
        if seg_data is None:
            continue
        if not isinstance(seg_data, dict):
            raise ValueError(f"Segment '{seg}' must be a JSON object or null")
        legacy = _LEGACY_ROD_STIFFNESS_RANGE_KEYS.intersection(seg_data)
        if legacy:
            raise ValueError(
                f"Segment '{seg}' uses deprecated keys {sorted(legacy)}; "
                "use youngs_modulus_pa and damping_ratio instead"
            )
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

        _validate_vbd_stretch_force(seg, seg_data)

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

    if "spur_surface_offset" in data and not isinstance(data["spur_surface_offset"], bool):
        raise ValueError(
            f"spur_surface_offset must be a boolean, got {data['spur_surface_offset']!r}"
        )

    if "sim_build" in data:
        _validate_sim_build(data["sim_build"])

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
