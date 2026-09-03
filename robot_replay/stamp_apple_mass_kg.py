#!/usr/bin/env python3
"""One-shot: stamp ``pre_grasp_geometry.parts.apple.mass_kg`` on real episode parquets.

Computes solid-sphere mass from catalog ``radius_m`` and ``density_kg_m3``:

    m = (4/3) * pi * r^3 * rho

Skips files that already have a positive ``mass_kg``. Use ``--force`` to overwrite.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _apple_sphere_mass_kg(radius_m: float, density_kg_m3: float) -> float:
    r = float(radius_m)
    rho = float(density_kg_m3)
    if not math.isfinite(r) or r <= 0.0:
        raise ValueError(f"apple radius_m must be finite > 0, got {radius_m!r}")
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError(f"apple density_kg_m3 must be finite > 0, got {density_kg_m3!r}")
    return (4.0 / 3.0) * math.pi * r**3 * rho


def _is_episode_parquet(path: Path) -> bool:
    name = path.name
    if not name.endswith(".parquet"):
        return False
    if "_robot" in name or "_tracking" in name or name.startswith("pull_"):
        return False
    return "-d" in name


def stamp_apple_mass_kg(path: Path, *, force: bool = False) -> dict[str, Any]:
    table = pq.read_table(path)
    meta = dict(table.schema.metadata or {})
    blob = meta.get(b"dataset_metadata")
    if blob is None:
        raise ValueError(f"{path}: missing dataset_metadata")
    dm = json.loads(
        blob.decode("utf-8") if isinstance(blob, (bytes, bytearray)) else str(blob)
    )
    if not isinstance(dm, dict):
        raise ValueError(f"{path}: dataset_metadata must be a JSON object")
    pre = dm.get("pre_grasp_geometry")
    if not isinstance(pre, dict):
        raise ValueError(f"{path}: missing pre_grasp_geometry")
    parts = pre.get("parts")
    if not isinstance(parts, dict) or "apple" not in parts:
        raise ValueError(f"{path}: missing pre_grasp_geometry.parts.apple")
    apple = parts["apple"]
    if not isinstance(apple, dict):
        raise ValueError(f"{path}: parts.apple must be an object")

    existing = apple.get("mass_kg")
    if existing is not None and not force:
        mass = float(existing)
        if math.isfinite(mass) and mass > 0.0:
            return {
                "path": str(path),
                "status": "skipped",
                "mass_kg": mass,
                "mass_g": mass * 1000.0,
            }

    radius_m = apple.get("radius_m")
    density_kg_m3 = apple.get("density_kg_m3")
    if radius_m is None or density_kg_m3 is None:
        raise ValueError(f"{path}: apple missing radius_m or density_kg_m3")
    mass_kg = _apple_sphere_mass_kg(float(radius_m), float(density_kg_m3))
    apple["mass_kg"] = round(mass_kg, 9)
    apple["mass_kg_source"] = "computed_from_radius_density"
    dm["apple_mass_kg"] = apple["mass_kg"]

    meta[b"dataset_metadata"] = json.dumps(dm, sort_keys=True).encode("utf-8")
    out = table.replace_schema_metadata(meta)
    pq.write_table(out, path)
    return {
        "path": str(path),
        "status": "stamped",
        "apple_number": apple.get("apple_number"),
        "radius_m": float(radius_m),
        "density_kg_m3": float(density_kg_m3),
        "mass_kg": mass_kg,
        "mass_g": mass_kg * 1000.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("robot_replay")],
        help="Directories to scan for sXX-dNN.parquet episode files (default: robot_replay)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing mass_kg",
    )
    args = parser.parse_args(argv)

    paths: list[Path] = []
    for root in args.roots:
        root = root.resolve()
        if root.is_file() and root.suffix == ".parquet":
            paths.append(root)
            continue
        paths.extend(sorted(p for p in root.rglob("*.parquet") if _is_episode_parquet(p)))

    if not paths:
        print("No episode parquets found.")
        return 1

    for path in paths:
        result = stamp_apple_mass_kg(path, force=bool(args.force))
        status = result["status"]
        apple_num = result.get("apple_number", "?")
        mass_g = result["mass_g"]
        print(f"{status:7s}  {path}  apple={apple_num}  mass={mass_g:.1f} g")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
