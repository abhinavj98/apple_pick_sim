"""Mark batched_sysid_v1 episodes excluded when unstable-frame fraction is high.

Default policy: exclude if ``excluded`` is already true, or if the fraction of
``stable=False`` frames exceeds ``DEFAULT_UNSTABLE_FRACTION_EXCLUDE`` (0.25).
A single force-cap hit must not wipe an episode.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from apple_pick_gym.batched_envs.batched_stability_monitor import (
    DEFAULT_UNSTABLE_FRACTION_THRESHOLD,
)
from apple_pick_sim.system_id import BatchedSysIdDataset

EXCLUDED_REASON_STABILITY_BLOWUP = "stability_blowup"
FILTERED_MANIFEST_NAME = "manifest.filtered.json"
PRE_EXCLUDE_BACKUP_NAME = "manifest.pre_exclude.json"


DEFAULT_UNSTABLE_FRACTION_EXCLUDE = DEFAULT_UNSTABLE_FRACTION_THRESHOLD


def _episode_unstable_fraction(dataset: BatchedSysIdDataset, episode: dict[str, Any]) -> float:
    arrays = dataset.load_episode_obs_arrays(
        int(episode["structure_idx"]),
        int(episode["direction_idx"]),
    )
    stable = np.asarray(arrays["stable"], dtype=bool).reshape(-1)
    if stable.size == 0:
        return 0.0
    return float((~stable).mean())


def _episode_should_exclude(
    dataset: BatchedSysIdDataset,
    episode: dict[str, Any],
    *,
    unstable_fraction_threshold: float = DEFAULT_UNSTABLE_FRACTION_EXCLUDE,
) -> bool:
    """Exclude only when already flagged or unstable-frame fraction exceeds threshold.

    A single force-cap hit is common under stiff VIC and must not wipe the episode.
    """
    if bool(episode.get("excluded", False)):
        return True
    return _episode_unstable_fraction(dataset, episode) > float(unstable_fraction_threshold)


def _annotate_episodes(dataset: BatchedSysIdDataset) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for raw in dataset.episode_entries():
        ep = dict(raw)
        if _episode_should_exclude(dataset, ep):
            ep["excluded"] = True
            ep["excluded_reason"] = ep.get("excluded_reason") or EXCLUDED_REASON_STABILITY_BLOWUP
        else:
            ep["excluded"] = False
            ep["excluded_reason"] = None
        episodes.append(ep)
    return episodes


def exclude_unstable_episodes(
    dataset_dir: Path | str,
    *,
    inplace: bool = False,
) -> Path:
    """Update exclusion flags from Parquet ``stable`` columns.

    When ``inplace`` is False, writes ``manifest.filtered.json`` and leaves
    ``manifest.json`` unchanged. When True, backs up to ``manifest.pre_exclude.json``
    then rewrites ``manifest.json``.
    """
    root = Path(dataset_dir)
    dataset = BatchedSysIdDataset(root)
    episodes = _annotate_episodes(dataset)
    payload = dataset.manifest
    payload["episodes"] = episodes

    if inplace:
        manifest_path = root / "manifest.json"
        backup = root / PRE_EXCLUDE_BACKUP_NAME
        if manifest_path.exists() and not backup.exists():
            shutil.copy2(manifest_path, backup)
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest_path

    out = root / FILTERED_MANIFEST_NAME
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Mark batched_sysid_v1 episodes excluded when the unstable-frame "
            f"fraction exceeds {DEFAULT_UNSTABLE_FRACTION_EXCLUDE} "
            "(already-excluded rows stay excluded)."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset root directory")
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Rewrite manifest.json (backs up to manifest.pre_exclude.json first)",
    )
    args = parser.parse_args(argv)
    out = exclude_unstable_episodes(args.dataset, inplace=bool(args.inplace))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
