"""Disk cache for free-proxy VBD settle state (settle-then-weld, V.3.1 step B)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
)
from apple_pick_sim.fruiting_system import FruitingSystemParams, params_fingerprint

SETTLE_CACHE_SCHEMA_VERSION = 1


def resolve_settle_cache_dir(override: Path | str | None) -> Path:
    """Resolve settle cache root directory."""
    if override is not None:
        return Path(override)
    env = os.environ.get("APPLE_PICK_SIM_SETTLE_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "apple_pick_sim" / "settled"


def ranges_content_fingerprint(ranges: dict) -> str:
    """Stable SHA-256 prefix from normalized ranges JSON."""
    payload = json.dumps(ranges, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _ranges_stem(config: BatchedHeterogeneousCoupledSimConfig, ranges: dict) -> str:
    path = config.domain_randomization.ranges_path
    if path is not None:
        return Path(path).stem
    return f"ranges_{ranges_content_fingerprint(ranges)[:8]}"


def _per_env_params_hash(per_env_params: Sequence[FruitingSystemParams]) -> str:
    fps = [params_fingerprint(p) for p in per_env_params]
    payload = json.dumps(fps, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:8]


def _topology_seed_value(config: BatchedHeterogeneousCoupledSimConfig) -> int:
    seed = config.domain_randomization.topology_seed
    return int(seed) if seed is not None else -1


def _fruiting_base_label(config: BatchedHeterogeneousCoupledSimConfig) -> str:
    pos = config.scene.fruiting_base_pos
    if pos is None:
        return "default"
    return ",".join(f"{float(v):.6g}" for v in pos)


def build_cache_key(
    config: BatchedHeterogeneousCoupledSimConfig,
    ranges: dict,
    per_env_params: Sequence[FruitingSystemParams],
) -> str:
    """Canonical cache key string for metadata validation."""
    scene = config.scene
    parts = [
        f"ranges={ranges_content_fingerprint(ranges)}",
        f"n={config.runtime.num_envs}",
        f"seed={_topology_seed_value(config)}",
        f"sub={scene.settle_substeps}",
        f"fix={1 if config.robot.fix_to_apple else 0}",
        f"params={_per_env_params_hash(per_env_params)}",
        f"grav_ramp={int(scene.settle_gravity_ramp)}",
        f"quiet_every={scene.settle_quiet_every if scene.settle_quiet_every is not None else 0}",
        f"max_speed={scene.settle_max_speed_m_s:.6g}",
        f"self_coll={int(scene.enable_self_collisions)}",
        f"apple_woody={int(scene.enable_apple_woody_collisions)}",
        f"proxy_woody={int(scene.enable_proxy_woody_collisions)}",
        f"base={_fruiting_base_label(config)}",
        f"schema={SETTLE_CACHE_SCHEMA_VERSION}",
    ]
    return "|".join(parts)


def settle_cache_path_for(
    config: BatchedHeterogeneousCoupledSimConfig,
    ranges: dict,
    per_env_params: Sequence[FruitingSystemParams],
    *,
    cache_dir: Path | str | None = None,
) -> Path | None:
    """Filesystem path for settle cache, or ``None`` when caching does not apply."""
    if config.robot.step_mode == "vbd_only":
        return None
    if not config.robot.fix_to_apple:
        return None
    if int(config.scene.settle_substeps) <= 0:
        return None

    stem = _ranges_stem(config, ranges)
    seed = _topology_seed_value(config)
    sub = int(config.scene.settle_substeps)
    fix = 1 if config.robot.fix_to_apple else 0
    params_hash = _per_env_params_hash(per_env_params)
    filename = f"{stem}__n{config.runtime.num_envs}__seed{seed}__sub{sub}__fix{fix}__{params_hash}.npz"
    return resolve_settle_cache_dir(cache_dir) / filename


def settle_cache_applicable(config: BatchedHeterogeneousCoupledSimConfig) -> bool:
    return (
        config.robot.step_mode != "vbd_only"
        and config.robot.fix_to_apple
        and int(config.scene.settle_substeps) > 0
    )


@dataclasses.dataclass(frozen=True)
class SettledCheckpoint:
    """Free-proxy cable state after VBD settle + quiet; seeds welded build."""

    body_q: np.ndarray
    metadata: dict[str, Any]

    @classmethod
    def from_build_context(
        cls,
        *,
        body_q: np.ndarray,
        config: BatchedHeterogeneousCoupledSimConfig,
        ranges: dict,
        per_env_params: Sequence[FruitingSystemParams],
    ) -> SettledCheckpoint:
        bq = np.asarray(body_q, dtype=np.float32).reshape(-1, 7)
        fps = [params_fingerprint(p) for p in per_env_params]
        metadata: dict[str, Any] = {
            "schema_version": SETTLE_CACHE_SCHEMA_VERSION,
            "cache_key": build_cache_key(config, ranges, per_env_params),
            "ranges_fingerprint": ranges_content_fingerprint(ranges),
            "topology_seed": _topology_seed_value(config),
            "num_envs": int(config.runtime.num_envs),
            "settle_substeps": int(config.scene.settle_substeps),
            "fix_to_apple": bool(config.robot.fix_to_apple),
            "per_env_params_fps": fps,
            "settle_gravity_ramp": bool(config.scene.settle_gravity_ramp),
            "settle_quiet_every": config.scene.settle_quiet_every,
            "settle_max_speed_m_s": float(config.scene.settle_max_speed_m_s),
            "enable_self_collisions": bool(config.scene.enable_self_collisions),
            "enable_apple_woody_collisions": bool(config.scene.enable_apple_woody_collisions),
            "enable_proxy_woody_collisions": bool(config.scene.enable_proxy_woody_collisions),
            "fruiting_base_pos": config.scene.fruiting_base_pos,
        }
        return cls(body_q=bq, metadata=metadata)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            body_q=self.body_q,
            schema_version=np.int32(self.metadata["schema_version"]),
            cache_key=np.array(self.metadata["cache_key"]),
            ranges_fingerprint=np.array(self.metadata["ranges_fingerprint"]),
            topology_seed=np.int32(self.metadata["topology_seed"]),
            num_envs=np.int32(self.metadata["num_envs"]),
            settle_substeps=np.int32(self.metadata["settle_substeps"]),
            fix_to_apple=np.bool_(self.metadata["fix_to_apple"]),
            per_env_params_fps=np.array(
                json.dumps(self.metadata["per_env_params_fps"], sort_keys=True)
            ),
            settle_gravity_ramp=np.bool_(self.metadata["settle_gravity_ramp"]),
            settle_max_speed_m_s=np.float32(self.metadata["settle_max_speed_m_s"]),
            enable_self_collisions=np.bool_(self.metadata["enable_self_collisions"]),
            enable_apple_woody_collisions=np.bool_(self.metadata["enable_apple_woody_collisions"]),
            enable_proxy_woody_collisions=np.bool_(self.metadata["enable_proxy_woody_collisions"]),
            fruiting_base_pos=np.array(
                json.dumps(self.metadata["fruiting_base_pos"], sort_keys=True)
            ),
        )

    @classmethod
    def load(cls, path: Path | str) -> SettledCheckpoint:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            body_q = np.asarray(data["body_q"], dtype=np.float32)
            if "metadata_json" in data:
                metadata = json.loads(str(data["metadata_json"].item()))
            else:
                metadata = {
                    "schema_version": int(data["schema_version"]),
                    "cache_key": str(data["cache_key"].item()),
                    "ranges_fingerprint": str(data["ranges_fingerprint"].item()),
                    "topology_seed": int(data["topology_seed"]),
                    "num_envs": int(data["num_envs"]),
                    "settle_substeps": int(data["settle_substeps"]),
                    "fix_to_apple": bool(data["fix_to_apple"]),
                    "per_env_params_fps": json.loads(str(data["per_env_params_fps"].item())),
                    "settle_gravity_ramp": bool(data["settle_gravity_ramp"]),
                    "settle_max_speed_m_s": float(data["settle_max_speed_m_s"]),
                    "enable_self_collisions": bool(data["enable_self_collisions"]),
                    "enable_apple_woody_collisions": bool(data["enable_apple_woody_collisions"]),
                    "enable_proxy_woody_collisions": bool(data["enable_proxy_woody_collisions"]),
                    "fruiting_base_pos": json.loads(str(data["fruiting_base_pos"].item())),
                }
        if int(metadata.get("schema_version", 0)) != SETTLE_CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported settle checkpoint schema_version "
                f"{metadata.get('schema_version')!r} (expected {SETTLE_CACHE_SCHEMA_VERSION})"
            )
        return cls(body_q=body_q.reshape(-1, 7), metadata=metadata)

    def validate_against(
        self,
        *,
        config: BatchedHeterogeneousCoupledSimConfig,
        ranges: dict,
        per_env_params: Sequence[FruitingSystemParams],
    ) -> None:
        if int(self.metadata.get("num_envs", -1)) != int(config.runtime.num_envs):
            raise ValueError("settled checkpoint num_envs mismatch")
        if int(self.metadata.get("settle_substeps", -1)) != int(config.scene.settle_substeps):
            raise ValueError("settled checkpoint settle_substeps mismatch")
        fps = [params_fingerprint(p) for p in per_env_params]
        if self.metadata.get("per_env_params_fps") != fps:
            raise ValueError("settled checkpoint per_env_params mismatch")
        expected_key = build_cache_key(config, ranges, per_env_params)
        if self.metadata.get("cache_key") != expected_key:
            raise ValueError(
                "settled checkpoint cache_key mismatch "
                f"(file {self.metadata.get('cache_key')!r} vs current {expected_key!r})"
            )
