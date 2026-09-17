"""Pure geometry helpers for FR3 ee-cylinder tip/flange/TCP layout."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_UNIT_CYLINDER_Z_MIN = -0.5
_UNIT_CYLINDER_Z_MAX = 0.5
_UNIT_CYLINDER_XY_RADIUS = 0.5


@dataclass(frozen=True)
class EeCylinderLayout:
    tip_z_m: float
    flange_z_m: float
    tcp_z_m: float
    radius_m: float
    length_m: float


def ee_cylinder_layout_from_authored(
    *,
    ee_scale_xyz: tuple[float, float, float],
    mesh_translate_xyz: tuple[float, float, float],
    mesh_scale_xyz: tuple[float, float, float],
    mesh_z_min: float,
    mesh_z_max: float,
    tcp_translate_xyz: tuple[float, float, float],
) -> EeCylinderLayout:
    """Compute tip/flange/TCP positions in ee-local meters from authored USD xforms.

    Convention: USD ``xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]``
    on the mesh means parent-from-local for a point ``p`` is (orient = identity):

    ``p_ee_local = scale * (p + translate)``

    Then ee's own scale maps to meters along the tool axis (ee translate ignored for
    relative tip/flange/TCP):

    ``z_m = ee_scale_z * p_ee_local.z``
    ``r_m = ee_scale_x * mesh_scale_x * (mesh radial half-extent)``

    Mesh points extent uses ``z in [mesh_z_min, mesh_z_max]`` (unit cylinder ``[-0.5, 0.5]``).
    After mesh xform, the two face ``z`` values are known; **tip** is the face nearer
    to TCP and **flange** is the other (so either ee ``+Z`` or ``−Z`` tip-out works).

    Note: ``fr3_joint8`` applies ~180° about X, so world tip-out (away from link7)
    is **ee −Z**. Author the cylinder on negative ee ``z`` for correct viz.
    The TCP body uses **RotX(180°)** so TCP local **+Z** is tip-out (logged Franka /
    VBD proxy). TCP ``z_m = ee_scale_z * tcp_translate_z``.
    """
    sx, _sy, sz = ee_scale_xyz
    mx, _my, mz = mesh_scale_xyz
    _tx, _ty, tz = mesh_translate_xyz
    z0 = mz * (mesh_z_min + tz)
    z1 = mz * (mesh_z_max + tz)
    face0_m = sz * z0
    face1_m = sz * z1
    tcp_z_m = sz * tcp_translate_xyz[2]
    if abs(face0_m - tcp_z_m) <= abs(face1_m - tcp_z_m):
        tip_z_m, flange_z_m = face0_m, face1_m
    else:
        tip_z_m, flange_z_m = face1_m, face0_m
    radius_m = abs(sx * mx * _UNIT_CYLINDER_XY_RADIUS)
    length_m = abs(tip_z_m - flange_z_m)
    return EeCylinderLayout(
        tip_z_m=tip_z_m,
        flange_z_m=flange_z_m,
        tcp_z_m=tcp_z_m,
        radius_m=radius_m,
        length_m=length_m,
    )


def assert_tip_flange_tcp_contract(
    layout: EeCylinderLayout,
    *,
    tip_tol_m: float = 1e-3,
    flange_tol_m: float = 1e-3,
    expected_flange_z_m: float | None = None,
) -> None:
    """Assert tip↔TCP coincidence and proximal (flange) face placement.

    If ``expected_flange_z_m`` is set, proximal face must match it (e.g. link7
    mesh end relative to ``fr3_joint8``). Otherwise proximal face must be near
    ee origin (legacy flush-at-joint check).
    """
    if expected_flange_z_m is None:
        if abs(layout.flange_z_m) > flange_tol_m:
            raise AssertionError(
                f"flange face z={layout.flange_z_m} m must be ~0 (flush with ee origin)"
            )
    elif abs(layout.flange_z_m - expected_flange_z_m) > flange_tol_m:
        raise AssertionError(
            f"flange face z={layout.flange_z_m} m != expected {expected_flange_z_m} m "
            f"(tol={flange_tol_m})"
        )
    if abs(layout.tip_z_m - layout.tcp_z_m) > tip_tol_m:
        raise AssertionError(
            f"tip z={layout.tip_z_m} m != tcp z={layout.tcp_z_m} m (tol={tip_tol_m})"
        )


def scrape_ee_cylinder_authored(usd_path: Path) -> dict:
    """Return authored xform numbers for ``/fr3/ee`` from a USDA file."""
    text = usd_path.read_text(encoding="utf-8")
    ee_block = _extract_ee_block(text)
    ee_scale = _parse_xform_triple(ee_block, "xformOp:scale")
    _ = _parse_xform_triple(ee_block, "xformOp:translate")  # ee translate ignored for layout

    cylinder_block = _extract_child_block(ee_block, r'def\s+Mesh\s+"Cylinder"')
    mesh_scale = _parse_xform_triple(cylinder_block, "xformOp:scale")
    mesh_translate = _parse_xform_triple(cylinder_block, "xformOp:translate")
    mesh_z_min, mesh_z_max = _parse_mesh_z_extent(cylinder_block)

    tcp_block = _extract_child_block(ee_block, r'def\s+Xform\s+"tcp"')
    tcp_translate = _parse_xform_triple(tcp_block, "xformOp:translate")
    tcp_orient = _parse_xform_quat_wxyz(tcp_block, "xformOp:orient")

    return {
        "ee_scale_xyz": ee_scale,
        "mesh_translate_xyz": mesh_translate,
        "mesh_scale_xyz": mesh_scale,
        "mesh_z_min": mesh_z_min,
        "mesh_z_max": mesh_z_max,
        "tcp_translate_xyz": tcp_translate,
        "tcp_orient_wxyz": tcp_orient,
    }


def flange_com_to_ee_local(
    f_x_cee: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Map Franka flange F_x_Cee into USD ee local (fr3_joint8 RotX 180°)."""
    x, y, z = f_x_cee
    return (float(x), float(-y), float(-z))


def scrape_ee_mass_properties(usd_path: Path) -> dict:
    """Return mass/COM/inertia for ``/fr3/ee`` and ``/fr3/ee/tcp`` from a USDA file."""
    text = usd_path.read_text(encoding="utf-8")
    ee_block = _extract_ee_block(text)
    ee_own_attrs = _split_ee_own_attrs(ee_block)

    ee_mass_kg = _parse_physics_float(ee_own_attrs, "physics:mass")
    ee_com_xyz = _parse_physics_point3f(ee_own_attrs, "physics:centerOfMass")
    ee_inertia_diag = _parse_physics_float3(ee_own_attrs, "physics:diagonalInertia")

    tcp_block = _extract_child_block(ee_block, r'def\s+Xform\s+"tcp"')
    tcp_mass_kg = _parse_physics_float(tcp_block, "physics:mass")

    return {
        "ee_mass_kg": ee_mass_kg,
        "ee_com_xyz": ee_com_xyz,
        "ee_inertia_diag": ee_inertia_diag,
        "tcp_mass_kg": tcp_mass_kg,
    }


def _split_ee_own_attrs(ee_block: str) -> str:
    """Return ee prim own attributes (text before the first nested ``def``)."""
    match = re.search(r"\n        def ", ee_block)
    if match is None:
        return ee_block
    return ee_block[: match.start()]


def _parse_physics_float(block: str, attr_name: str) -> float:
    pattern = rf"float\s+{re.escape(attr_name)}\s*=\s*([^\s\n]+)"
    match = re.search(pattern, block)
    if match is None:
        raise ValueError(f"missing {attr_name} in block")
    return float(match.group(1))


def _parse_physics_point3f(block: str, attr_name: str) -> tuple[float, float, float]:
    pattern = rf"point3f\s+{re.escape(attr_name)}\s*=\s*\(([^)]+)\)"
    match = re.search(pattern, block)
    if match is None:
        raise ValueError(f"missing {attr_name} in block")
    parts = [float(part.strip()) for part in match.group(1).split(",")]
    if len(parts) != 3:
        raise ValueError(f"{attr_name} must have 3 components, got {parts!r}")
    return (parts[0], parts[1], parts[2])


def _parse_physics_float3(block: str, attr_name: str) -> tuple[float, float, float]:
    pattern = rf"float3\s+{re.escape(attr_name)}\s*=\s*\(([^)]+)\)"
    match = re.search(pattern, block)
    if match is None:
        raise ValueError(f"missing {attr_name} in block")
    parts = [float(part.strip()) for part in match.group(1).split(",")]
    if len(parts) != 3:
        raise ValueError(f"{attr_name} must have 3 components, got {parts!r}")
    return (parts[0], parts[1], parts[2])


def _extract_ee_block(text: str) -> str:
    match = re.search(r'def\s+Xform\s+"ee"\s*(?:\([^)]*\))?\s*\{', text)
    if match is None:
        raise ValueError(f'no def Xform "ee" block in {text[:80]!r}...')
    return _extract_braced_block(text, match.end() - 1)


def _extract_child_block(parent: str, child_pattern: str) -> str:
    match = re.search(child_pattern + r"\s*(?:\([^)]*\))?\s*\{", parent)
    if match is None:
        raise ValueError(f"no child block matching {child_pattern!r}")
    return _extract_braced_block(parent, match.end() - 1)


def _extract_braced_block(text: str, open_brace_index: int) -> str:
    if text[open_brace_index] != "{":
        raise ValueError("open_brace_index must point at '{'")
    depth = 0
    for index in range(open_brace_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_index : index + 1]
    raise ValueError("unbalanced braces in USD block")


def _parse_xform_triple(block: str, op_name: str) -> tuple[float, float, float]:
    pattern = rf"(?:double3|float3)\s+{re.escape(op_name)}\s*=\s*\(([^)]+)\)"
    match = re.search(pattern, block)
    if match is None:
        raise ValueError(f"missing {op_name} in block")
    parts = [float(part.strip()) for part in match.group(1).split(",")]
    if len(parts) != 3:
        raise ValueError(f"{op_name} must have 3 components, got {parts!r}")
    return (parts[0], parts[1], parts[2])


def _parse_xform_quat_wxyz(block: str, op_name: str) -> tuple[float, float, float, float]:
    """Parse USD ``quatd`` / ``quatf`` as ``(w, x, y, z)``."""
    pattern = rf"(?:quatd|quatf)\s+{re.escape(op_name)}\s*=\s*\(([^)]+)\)"
    match = re.search(pattern, block)
    if match is None:
        raise ValueError(f"missing {op_name} in block")
    parts = [float(part.strip()) for part in match.group(1).split(",")]
    if len(parts) != 4:
        raise ValueError(f"{op_name} must have 4 components, got {parts!r}")
    return (parts[0], parts[1], parts[2], parts[3])

def _parse_mesh_z_extent(cylinder_block: str) -> tuple[float, float]:
    extent_match = re.search(
        r"float3\[\]\s+extent\s*=\s*\[\(([^)]+)\),\s*\(([^)]+)\)\]",
        cylinder_block,
    )
    if extent_match:
        min_corner = [float(part.strip()) for part in extent_match.group(1).split(",")]
        max_corner = [float(part.strip()) for part in extent_match.group(2).split(",")]
        if len(min_corner) >= 3 and len(max_corner) >= 3:
            z_min = min(min_corner[2], max_corner[2])
            z_max = max(min_corner[2], max_corner[2])
            if abs(z_min - _UNIT_CYLINDER_Z_MIN) > 1e-6 or abs(z_max - _UNIT_CYLINDER_Z_MAX) > 1e-6:
                raise ValueError(
                    f"mesh extent z=[{z_min}, {z_max}] differs from unit cylinder "
                    f"[{_UNIT_CYLINDER_Z_MIN}, {_UNIT_CYLINDER_Z_MAX}]"
                )
            return z_min, z_max

    points_match = re.search(r"point3f\[\]\s+points\s*=\s*\[([^\]]+)\]", cylinder_block)
    if points_match:
        coords = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", points_match.group(1))]
        z_coords = coords[2::3]
        if z_coords:
            z_min = min(z_coords)
            z_max = max(z_coords)
            if abs(z_min - _UNIT_CYLINDER_Z_MIN) > 1e-6 or abs(z_max - _UNIT_CYLINDER_Z_MAX) > 1e-6:
                raise ValueError(
                    f"mesh points z extent [{z_min}, {z_max}] differs from unit cylinder "
                    f"[{_UNIT_CYLINDER_Z_MIN}, {_UNIT_CYLINDER_Z_MAX}]"
                )
            return z_min, z_max

    return _UNIT_CYLINDER_Z_MIN, _UNIT_CYLINDER_Z_MAX
