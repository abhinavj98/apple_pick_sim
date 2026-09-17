#!/usr/bin/env python3
"""Stamp stem ``radius_m`` to 0.9 mm (``0.0009`` m) on real episode parquets.

Sets every ``parts.stem.radius_m`` in ``dataset_metadata`` (pre-grasp, nested
catalog dump, source_metadata_summary) to ``0.0009`` m so the 0.5 mm catalog
placeholder is gone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

STEM_RADIUS_M = 0.0009


def _is_episode_parquet(path: Path) -> bool:
    name = path.name
    if not name.endswith(".parquet"):
        return False
    if "_robot" in name or "_tracking" in name or name.startswith("pull_"):
        return False
    return "-d" in name


def _stamp_stem_radii(obj: Any) -> int:
    """Set every dict ``stem.radius_m`` under ``obj`` to ``STEM_RADIUS_M``."""
    n_updated = 0
    if isinstance(obj, dict):
        stem = obj.get("stem")
        if isinstance(stem, dict) and "radius_m" in stem:
            current = stem.get("radius_m")
            try:
                same = float(current) == float(STEM_RADIUS_M)
            except (TypeError, ValueError):
                same = False
            if not same:
                stem["radius_m"] = float(STEM_RADIUS_M)
                n_updated += 1
        for value in obj.values():
            n_updated += _stamp_stem_radii(value)
    elif isinstance(obj, list):
        for item in obj:
            n_updated += _stamp_stem_radii(item)
    return n_updated


def stamp_stem_radius(path: Path) -> dict[str, Any]:
    table = pq.read_table(path)
    meta = dict(table.schema.metadata or {})
    blob = meta.get(b"dataset_metadata")
    if blob is None:
        raise ValueError(f"{path}: missing dataset_metadata")
    dm = json.loads(
        blob.decode("utf-8") if isinstance(blob, (bytes, bytearray)) else str(blob)
    )
    n_updated = _stamp_stem_radii(dm)
    if n_updated == 0:
        return {
            "path": str(path),
            "status": "unchanged",
            "n_updated": 0,
            "radius_m": STEM_RADIUS_M,
        }
    meta[b"dataset_metadata"] = json.dumps(dm, sort_keys=True).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(meta), path)
    return {
        "path": str(path),
        "status": "stamped",
        "n_updated": int(n_updated),
        "radius_m": STEM_RADIUS_M,
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
        result = stamp_stem_radius(path)
        print(
            f"{result['status']:<9}  {path}  "
            f"updated={result['n_updated']}  R={result['radius_m']*1000:.1f}mm"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
