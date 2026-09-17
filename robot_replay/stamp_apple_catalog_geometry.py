#!/usr/bin/env python3
"""One-shot: patch catalog apple radius, stem length; keep ``mass_kg``; update ``density_kg_m3``."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

# Catalog geometry keyed by ``apple_number`` (mass_kg preserved from parquet).
APPLE_CATALOG: dict[int, dict[str, float]] = {
    2: {"radius_m": 0.035, "stem_length_m": 0.011},
    9: {"radius_m": 0.036, "stem_length_m": 0.013},
}


def _density_for_mass_kg(mass_kg: float, radius_m: float) -> float:
    volume_m3 = (4.0 / 3.0) * math.pi * float(radius_m) ** 3
    if volume_m3 <= 0.0:
        raise ValueError(f"invalid radius_m={radius_m!r}")
    return float(mass_kg) / volume_m3


def _is_episode_parquet(path: Path) -> bool:
    name = path.name
    if not name.endswith(".parquet"):
        return False
    if "_robot" in name or "_tracking" in name or name.startswith("pull_"):
        return False
    return "-d" in name


def stamp_apple_catalog_geometry(path: Path) -> dict[str, Any]:
    table = pq.read_table(path)
    meta = dict(table.schema.metadata or {})
    blob = meta.get(b"dataset_metadata")
    if blob is None:
        raise ValueError(f"{path}: missing dataset_metadata")
    dm = json.loads(
        blob.decode("utf-8") if isinstance(blob, (bytes, bytearray)) else str(blob)
    )
    pre = dm.get("pre_grasp_geometry")
    if not isinstance(pre, dict):
        raise ValueError(f"{path}: missing pre_grasp_geometry")
    parts = pre.get("parts")
    if not isinstance(parts, dict):
        raise ValueError(f"{path}: missing pre_grasp_geometry.parts")
    apple = parts.get("apple")
    stem = parts.get("stem")
    if not isinstance(apple, dict) or not isinstance(stem, dict):
        raise ValueError(f"{path}: missing apple or stem in parts")

    apple_number = apple.get("apple_number")
    if apple_number is None:
        raise ValueError(f"{path}: apple missing apple_number")
    catalog = APPLE_CATALOG.get(int(apple_number))
    if catalog is None:
        return {"path": str(path), "status": "skipped", "apple_number": apple_number}

    mass_kg = apple.get("mass_kg")
    if mass_kg is None or not math.isfinite(float(mass_kg)) or float(mass_kg) <= 0.0:
        raise ValueError(f"{path}: apple missing positive mass_kg (run stamp_apple_mass_kg first)")
    mass_kg = float(mass_kg)

    radius_m = float(catalog["radius_m"])
    stem_length_m = float(catalog["stem_length_m"])
    density_kg_m3 = _density_for_mass_kg(mass_kg, radius_m)

    apple["radius_m"] = radius_m
    apple["density_kg_m3"] = round(density_kg_m3, 9)
    apple["mass_kg_source"] = "fixed_mass_rescaled_density"
    stem["length_m"] = stem_length_m

    dm["apple_mass_kg"] = mass_kg
    dm["apple_radius_m"] = radius_m
    dm["stem_catalog_length_m"] = stem_length_m

    meta[b"dataset_metadata"] = json.dumps(dm, sort_keys=True).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(meta), path)
    return {
        "path": str(path),
        "status": "stamped",
        "apple_number": int(apple_number),
        "radius_m": radius_m,
        "stem_length_m": stem_length_m,
        "density_kg_m3": density_kg_m3,
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
        result = stamp_apple_catalog_geometry(path)
        if result["status"] == "skipped":
            print(f"skipped  {path}  apple={result.get('apple_number', '?')}")
            continue
        print(
            f"stamped  {path}  apple={result['apple_number']}  "
            f"R={result['radius_m']*1000:.0f}mm  stem={result['stem_length_m']*1000:.0f}mm  "
            f"mass={result['mass_g']:.1f}g  rho={result['density_kg_m3']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
