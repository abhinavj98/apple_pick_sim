"""Expand FIXED-joint damping ratio to absolute angular/linear kd.

Fixture joint damping may be specified as absolute ``joint_*_kd`` or as
``joint_damping_ratio`` (ζ) expanded via ``kd = ζ · 2 · √(k · I)`` /
``√(k · m)``. Weld kd is **not** scaled with Young's modulus: fixture ζ is
the constant weld damping ratio for every env.
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
