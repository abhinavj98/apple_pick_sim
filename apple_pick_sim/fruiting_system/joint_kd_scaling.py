"""Expand FIXED-joint damping ratio to absolute angular/linear kd.

Fixture joint damping may be specified as absolute ``joint_*_kd`` or as
``joint_damping_ratio`` (ζ) expanded via ``kd = ζ · 2 · √(k · I)`` /
``√(k · m)``. Weld kd is **not** scaled with Young's modulus: fixture ζ is
the constant weld damping ratio for every env.

Support angular stiffness is ``(3/4) * L_dowel^2 * k_linear`` with ``L_dowel``
the primary rod length (see :func:`support_angular_kp_from_linear`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

# Newton SolverVBD default rigid joint ke when a role has no kp override.
DEFAULT_RIGID_JOINT_KE = 1.0e5

JOINT_KD_ROLES: tuple[str, ...] = (
    "support",
    "primary_spur",
    "spur_stem",
    "stem_apple",
)

# Linear support k_p is N/m. Angular k_p is (3/4) L_dowel^2 k_linear (N·m/rad),
# with L_dowel the primary rod length between the T-junction world clamps.
SUPPORT_ANGULAR_KP_LENGTH_FACTOR: float = 0.75


def support_angular_kp_from_linear(linear_kp: float, dowel_length_m: float) -> float:
    """Map linear support k_p (N/m) to angular k_p (N·m/rad).

    k_ang = (3/4) * L^2 * k_lin where L is the primary (dowel) length in metres.
    """
    length = float(dowel_length_m)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError(
            f"dowel_length_m must be finite and > 0, got {dowel_length_m!r}"
        )
    return SUPPORT_ANGULAR_KP_LENGTH_FACTOR * (length**2) * float(linear_kp)


def map_support_angular_kp_overrides(
    angular_kp: Mapping[str, float],
    linear_kp: Mapping[str, float],
    *,
    dowel_length_m: float,
) -> dict[str, float]:
    """Overwrite ``support`` angular k_p from linear k_p when linear is set."""
    out = dict(angular_kp)
    if "support" in linear_kp:
        out["support"] = support_angular_kp_from_linear(
            linear_kp["support"], dowel_length_m
        )
    return out


def support_roll_kd_from_damping_ratio(
    *,
    zeta: float,
    roll_kp: float,
    child_inertia: np.ndarray,
) -> float:
    """Expand fixture ζ to revolute T-roll drive damping [N·m·s/rad]."""
    z = float(zeta)
    k = float(roll_kp)
    if not math.isfinite(z) or z < 0.0:
        raise ValueError(f"zeta must be >= 0, got {zeta!r}")
    if k <= 0.0:
        return 0.0
    i_max = _inertia_max_eigenvalue(child_inertia)
    return z * 2.0 * math.sqrt(k * i_max)


def support_dowel_length_m(params: object) -> float:
    """Primary rod length used as L in the support angular-k_p map."""
    primary = getattr(params, "primary", None)
    if primary is None:
        raise ValueError(
            "support angular kp requires params.primary.length (dowel length)"
        )
    length = float(getattr(primary, "length"))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError(
            f"params.primary.length must be finite and > 0, got {length!r}"
        )
    return length


def primary_length_midpoint_m(ranges: Mapping[str, object]) -> float:
    """Midpoint of fixture ``primary.length`` [min, max], for config-level maps."""
    primary = ranges.get("primary")
    if not isinstance(primary, Mapping):
        raise ValueError("ranges['primary'] is required to map support angular kp")
    band = primary.get("length")
    if not isinstance(band, Mapping) or "min" not in band or "max" not in band:
        raise ValueError("ranges['primary']['length'] must have min and max")
    lo = float(band["min"])
    hi = float(band["max"])
    mid = 0.5 * (lo + hi)
    if not math.isfinite(mid) or mid <= 0.0:
        raise ValueError(
            f"primary.length midpoint must be finite and > 0, got {mid!r}"
        )
    return mid


def _inertia_max_eigenvalue(inertia: np.ndarray) -> float:
    mat = np.asarray(inertia, dtype=np.float64)
    if mat.shape != (3, 3):
        raise ValueError(f"body inertia must be 3x3, got shape {mat.shape}")
    sym = 0.5 * (mat + mat.T)
    return float(np.max(np.linalg.eigvalsh(sym)))


def joint_kd_from_damping_ratio(
    *,
    zeta: float,
    fruiting_fixed_joints: Sequence[tuple[int, str]],
    body_mass: np.ndarray,
    body_inertia: np.ndarray,
    joint_child: np.ndarray,
    angular_kp_by_role: Mapping[str, float],
    linear_kp_by_role: Mapping[str, float],
    roles: Sequence[str] = JOINT_KD_ROLES,
    default_ke: float = DEFAULT_RIGID_JOINT_KE,
    body_offset: int = 0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Expand joint damping ratio to absolute angular/linear kd role maps.

    ``kd = ζ · 2 · √(k · I_child)`` (angular) and ``√(k · m_child)`` (linear),
    using intended ``kp`` per role (else ``default_ke``). Roles with no matching
    joint label are omitted.
    """
    z = float(zeta)
    if not math.isfinite(z) or z < 0.0:
        raise ValueError(f"zeta must be >= 0, got {zeta!r}")
    if float(default_ke) <= 0.0:
        raise ValueError(f"default_ke must be positive, got {default_ke}")

    mass = np.asarray(body_mass, dtype=np.float64).reshape(-1)
    inertia = np.asarray(body_inertia, dtype=np.float64)
    children = np.asarray(joint_child, dtype=np.int32).reshape(-1)
    if inertia.ndim != 3 or inertia.shape[1:] != (3, 3):
        raise ValueError(
            f"body_inertia must have shape (n, 3, 3), got {inertia.shape}"
        )

    angular: dict[str, float] = {}
    linear: dict[str, float] = {}
    for role in roles:
        matches = [(int(j), lab) for j, lab in fruiting_fixed_joints if role in lab]
        if not matches:
            continue
        joint_index, _label = matches[0]
        if joint_index < 0 or joint_index >= children.size:
            raise ValueError(
                f"joint index {joint_index} out of range for joint_child "
                f"(size {children.size})"
            )
        child_local = int(children[joint_index])
        child = int(body_offset) + child_local
        if child < 0 or child >= mass.size:
            raise ValueError(
                f"child body {child} out of range for body_mass (size {mass.size})"
            )
        if child >= inertia.shape[0]:
            raise ValueError(
                f"child body {child} out of range for body_inertia "
                f"(size {inertia.shape[0]})"
            )
        m = float(mass[child])
        if m < 0.0 or not math.isfinite(m):
            raise ValueError(f"body_mass[{child}] must be finite >= 0, got {m}")
        i_max = _inertia_max_eigenvalue(inertia[child])
        if i_max < 0.0 or not math.isfinite(i_max):
            raise ValueError(
                f"body_inertia[{child}] max eigenvalue must be finite >= 0, got {i_max}"
            )
        k_ang = float(angular_kp_by_role.get(role, default_ke))
        k_lin = float(linear_kp_by_role.get(role, default_ke))
        if k_ang <= 0.0 or k_lin <= 0.0:
            raise ValueError(
                f"kp for role {role!r} must be positive, got ang={k_ang} lin={k_lin}"
            )
        angular[role] = z * 2.0 * math.sqrt(k_ang * i_max)
        linear[role] = z * 2.0 * math.sqrt(k_lin * m)
    return angular, linear
