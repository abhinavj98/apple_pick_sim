"""Rewrite woody hang so the spur points toward the robot base (world origin).

Reads a real_static_sysid_episode parquet, rotates pre-/post-grasp (and rest)
woody geometry about the Branch/T so Branch→Spur aligns with Branch→robot,
and writes a sibling parquet. Table rows are copied unchanged.

Run from repo root::

    uv run python robot_replay/make_spur_toward_robot_parquet.py \\
      --input robot_replay/s00-d00.parquet \\
      --output robot_replay/s00-d00_spur_toward_robot.parquet
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _as_xyz(value: Any) -> np.ndarray:
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        return np.fromstring(s, sep=" ", dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _rotation_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3×3 rotation mapping unit vector ``a`` onto unit vector ``b``."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1.0 + 1e-12:
        # 180°: pick an orthogonal axis
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - a * np.dot(axis, a)
        axis /= np.linalg.norm(axis)
        K = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=np.float64,
        )
        return np.eye(3) + 2.0 * (K @ K)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = np.linalg.norm(v)
    K = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def _xform_about(branch: np.ndarray, R: np.ndarray, p: np.ndarray) -> np.ndarray:
    return branch + R @ (p - branch)


def _xform_woody9(
    start9: np.ndarray, end9: np.ndarray, branch: np.ndarray, R: np.ndarray
) -> tuple[list[float], list[float]]:
    s = start9.reshape(3, 3).copy()
    e = end9.reshape(3, 3).copy()
    for i in range(3):
        s[i] = _xform_about(branch, R, s[i])
        e[i] = _xform_about(branch, R, e[i])
    return s.reshape(9).tolist(), e.reshape(9).tolist()


def _spur_rotation(
    branch: np.ndarray,
    spur_end: np.ndarray,
    robot_base: np.ndarray,
    *,
    aim: str = "horizontal",
) -> np.ndarray:
    """Rotation mapping old Branch→Spur onto the toward-robot aim direction.

    ``aim="horizontal"`` (default): XY projection of Branch→robot_base (z=0),
    so the spur lies in a horizontal plane toward the robot (~90° from vertical).
    ``aim="origin_3d"``: full 3D Branch→robot_base (diagonal down if robot is lower).
    """
    old = spur_end - branch
    if float(np.linalg.norm(old)) < 1e-12:
        raise ValueError("zero-length spur chord")
    delta = robot_base - branch
    if aim == "horizontal":
        new = np.array([delta[0], delta[1], 0.0], dtype=np.float64)
    elif aim == "origin_3d":
        new = delta.astype(np.float64, copy=True)
    else:
        raise ValueError(f"unknown aim={aim!r}; expected 'horizontal' or 'origin_3d'")
    if float(np.linalg.norm(new)) < 1e-12:
        raise ValueError("Branch coincides with robot_base in aim plane; cannot aim spur")
    return _rotation_align(old, new)


def rewrite_metadata(
    meta: dict[str, Any],
    *,
    robot_base: tuple[float, float, float],
    aim: str = "horizontal",
) -> dict[str, Any]:
    out = copy.deepcopy(meta)
    rb = np.asarray(robot_base, dtype=np.float64)

    pre = out["pre_grasp_geometry"]
    snap = pre["snapshot"]
    start = np.asarray(snap["woody_part_start_pos"], dtype=np.float64).reshape(9)
    end = np.asarray(snap["woody_part_end_pos"], dtype=np.float64).reshape(9)
    branch = start[0:3].copy()
    spur_end = end[0:3].copy()
    R = _spur_rotation(branch, spur_end, rb, aim=aim)

    new_start, new_end = _xform_woody9(start, end, branch, R)
    apple = _xform_about(branch, R, _as_xyz(snap["apple_pos"]))
    snap["woody_part_start_pos"] = new_start
    snap["woody_part_end_pos"] = new_end
    snap["apple_pos"] = apple.tolist()
    snap["woody_bending_angles"] = [0.0, 0.0, 0.0]
    note = pre.get("note") or ""
    suffix = (
        "Synthetic: rotated hang about Branch so spur aims "
        f"{aim} toward robot_base={list(robot_base)}."
    )
    pre["note"] = f"{note} {suffix}".strip() if note else suffix
    pre["geometry_source"] = "synthetic_spur_toward_robot"

    # Rest reference matches non-bending pre-grasp hang.
    out["rest_woody_part_start_pos"] = list(new_start)
    out["rest_woody_part_end_pos"] = list(new_end)
    chords: list[float] = []
    s9 = np.asarray(new_start, dtype=np.float64).reshape(3, 3)
    e9 = np.asarray(new_end, dtype=np.float64).reshape(3, 3)
    for i in range(3):
        chords.extend((e9[i] - s9[i]).tolist())
    out["rest_chord_vectors"] = chords

    post = out.get("post_grasp_geometry")
    if isinstance(post, dict) and "woody_part_start_pos" in post:
        ps = np.asarray(post["woody_part_start_pos"], dtype=np.float64).reshape(9)
        pe = np.asarray(post["woody_part_end_pos"], dtype=np.float64).reshape(9)
        p_branch = ps[0:3].copy()
        # Same world rotation as pre (align original pre spur → toward robot).
        p_start, p_end = _xform_woody9(ps, pe, p_branch, R)
        post["woody_part_start_pos"] = p_start
        post["woody_part_end_pos"] = p_end
        if "apple_pos" in post:
            post["apple_pos"] = _xform_about(
                p_branch, R, _as_xyz(post["apple_pos"])
            ).tolist()

    dump = out.get("dump")
    if isinstance(dump, dict):
        dump["variant"] = "spur_toward_robot"
        dump["spur_toward_robot_base"] = list(robot_base)
        dump["spur_aim"] = aim

    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--robot-base",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="World-frame robot base the spur should point toward (default: origin).",
    )
    p.add_argument(
        "--aim",
        choices=("horizontal", "origin_3d"),
        default="horizontal",
        help=(
            "horizontal: spur in XY toward robot (~90° from vertical; default). "
            "origin_3d: aim at robot base in 3D (diagonal if base is lower)."
        ),
    )
    args = p.parse_args()

    table = pq.read_table(args.input)
    raw_meta = table.schema.metadata or {}
    if b"dataset_metadata" not in raw_meta:
        raise SystemExit(f"{args.input}: missing dataset_metadata")
    meta = json.loads(raw_meta[b"dataset_metadata"].decode("utf-8"))
    new_meta = rewrite_metadata(
        meta, robot_base=tuple(args.robot_base), aim=str(args.aim)
    )

    new_schema_meta = dict(raw_meta)
    new_schema_meta[b"dataset_metadata"] = json.dumps(new_meta).encode("utf-8")
    table_out = table.replace_schema_metadata(new_schema_meta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table_out, args.output)
    print(f"Wrote {args.output}")

    pre = new_meta["pre_grasp_geometry"]["snapshot"]
    branch = np.asarray(pre["woody_part_start_pos"][:3], dtype=np.float64)
    spur = np.asarray(pre["woody_part_end_pos"][:3], dtype=np.float64)
    u = spur - branch
    u /= np.linalg.norm(u)
    horiz = np.asarray(args.robot_base, dtype=np.float64) - branch
    horiz[2] = 0.0
    horiz /= np.linalg.norm(horiz)
    angle_from_vertical_deg = float(np.degrees(np.arccos(np.clip(-u[2], -1.0, 1.0))))
    print(f"Branch: {branch.tolist()}")
    print(f"Spur dir: {u.tolist()}")
    print(f"Horizontal toward robot: {horiz.tolist()}")
    print(f"dot(spur, horizontal_toward)={float(np.dot(u, horiz)):.6f}")
    print(f"angle from vertical: {angle_from_vertical_deg:.1f} deg (90 = horizontal)")


if __name__ == "__main__":
    main()
