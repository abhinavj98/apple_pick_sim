"""Per-world geometry and weld metadata for batched sys-ID envs."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from apple_pick_sim.coupled_fruiting.defaults import COUPLED_ROBOT_BASE_POS


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero vector cannot be normalized")
    return (v / n).astype(np.float64)


def physical_stem_dir_for_world(scene: Any, layout: Any, world: int) -> np.ndarray:
    """Unit stem direction (base → tip) for one batched world."""
    cable = scene.cable
    stem_bodies = cable.stem_bodies
    if len(stem_bodies) < 2:
        raise RuntimeError("cable scene missing stem bodies for pull-direction sampling")
    bq = cable.state_0.body_q.numpy().reshape(-1, 7)
    tip = bq[layout.body_index(int(world), int(stem_bodies[-1])), :3]
    base = bq[layout.body_index(int(world), int(stem_bodies[-2])), :3]
    return _normalize(tip - base)


def robot_base_pos_for_world(layout: Any, world: int) -> np.ndarray:
    """Fixture robot base position translated to one batched world's origin."""
    origin = np.asarray(layout.world_origin(int(world)), dtype=np.float64)
    base = np.asarray(COUPLED_ROBOT_BASE_POS, dtype=np.float64) + origin
    return base.astype(np.float64)


def weld_direction_for_world(scene: Any, layout: Any, world: int) -> np.ndarray:
    """Unit weld approach direction in world frame for one batched world."""
    cable = scene.cable
    apple_idx = int(layout.apple_body_indices[int(world)])
    if apple_idx < 0:
        raise RuntimeError(f"world {world} has no apple body index")

    bq = cable.state_0.body_q.numpy().reshape(-1, 7)
    apple_quat = bq[apple_idx, 3:7].astype(np.float64)

    per_offsets = getattr(scene, "per_world_proxy_offsets", None)
    offset = None
    if per_offsets is not None and int(world) < len(per_offsets):
        offset = per_offsets[int(world)]
    if offset is None:
        offset = cable.gripper_proxy_offset_in_apple_frame

    if offset is not None:
        off_pos = np.asarray(offset[:3], dtype=np.float64)
        norm = float(np.linalg.norm(off_pos))
        if norm >= 1e-9:
            approach_apple = off_pos / norm
            wq = wp.quat(
                float(apple_quat[0]),
                float(apple_quat[1]),
                float(apple_quat[2]),
                float(apple_quat[3]),
            )
            rotated = wp.quat_rotate(wq, wp.vec3(float(approach_apple[0]), float(approach_apple[1]), float(approach_apple[2])))
            return _normalize(np.asarray(rotated, dtype=np.float64))

    proxy_idx = int(layout.proxy_body_indices[int(world)])
    weld_vec = bq[proxy_idx, :3] - bq[apple_idx, :3]
    return _normalize(weld_vec)


def per_world_sysid_reset_info(scene: Any, layout: Any, world: int, params: Any) -> dict[str, Any]:
    """Build per-env reset metadata used by Parquet episode rows."""
    import apple_pick_sim.fruiting_system as fs

    cable = scene.cable
    apple_idx = int(layout.apple_body_indices[int(world)])
    bq = cable.state_0.body_q.numpy().reshape(-1, 7)
    apple_q = bq[apple_idx]

    robot_base = robot_base_pos_for_world(layout, world)
    info: dict[str, Any] = {
        "weld_direction": weld_direction_for_world(scene, layout, world).astype(np.float32),
        "robot_base_pos": robot_base.astype(np.float32),
        "weld_reference_pos": np.asarray(apple_q[:3], dtype=np.float32),
        "weld_reference_quat": np.asarray(apple_q[3:7], dtype=np.float32),
        "params_fingerprint": fs.params_fingerprint(params),
    }

    base_pos = getattr(params, "base_pos", None)
    if base_pos is not None:
        info["fruiting_base_pos"] = np.asarray(base_pos, dtype=np.float32).reshape(3)

    rod_radii: dict[str, float] = {}
    for segment in ("primary", "secondary", "spur", "stem"):
        rod = getattr(params, segment, None)
        if rod is not None and getattr(rod, "radius", None) is not None:
            rod_radii[segment] = float(rod.radius)
    if rod_radii:
        info["rod_radii"] = rod_radii

    apple_radius = getattr(params, "apple_radius", None)
    if apple_radius is not None:
        info["apple_radius"] = float(apple_radius)

    return info
