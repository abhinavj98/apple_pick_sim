"""Capture post-settle, pre-weld tree observations for batched sys-ID datasets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.digital_twin.record import fruiting_tree_fixed_joints
from apple_pick_sim.fruiting_system.scene import fixed_joint_anchors_world


def capture_pre_weld_tree_obs_for_world(
    cable: Any,
    layout: BatchedEnvLayout,
    *,
    world: int,
    junction_names: Sequence[str],
) -> dict[str, Any]:
    """Return woody/apple observations from one settled world before grasp weld."""
    joint_pairs = list(fruiting_tree_fixed_joints(cable))
    if not joint_pairs:
        raise ValueError("cable scene has no fruiting fixed joints")

    world_pairs = [
        (layout.joint_index(int(world), int(j_idx)), label) for j_idx, label in joint_pairs
    ]
    parent_flat, child_flat = fixed_joint_anchors_world(
        cable.model,
        cable.state_0.body_q,
        world_pairs,
    )
    parent = parent_flat.reshape(-1, 3)
    child = child_flat.reshape(-1, 3)
    labels = [label.removeprefix("joint_") for _, label in joint_pairs]

    woody_start: dict[str, np.ndarray] = {}
    woody_end: dict[str, np.ndarray] = {}
    for i, name in enumerate(labels):
        if name not in junction_names:
            raise ValueError(
                f"pre-weld junction {name!r} missing from junction_names {list(junction_names)!r}"
            )
        woody_start[name] = np.asarray(parent[i], dtype=np.float32)
        woody_end[name] = np.asarray(child[i], dtype=np.float32)

    apple_idx = int(layout.apple_body_indices[int(world)])
    if apple_idx < 0:
        raise ValueError(f"world {world} has no apple body")
    bq = cable.state_0.body_q.numpy().reshape(-1, 7)
    apple_q = np.asarray(bq[apple_idx], dtype=np.float32)

    return {
        "woody_part_start_pos": woody_start,
        "woody_part_end_pos": woody_end,
        "apple_pos": apple_q[:3].copy(),
        "apple_quat": apple_q[3:7].copy(),
    }


def capture_pre_weld_tree_obs_all_worlds(
    cable: Any,
    layout: BatchedEnvLayout,
    *,
    junction_names: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Capture settled tree observations for every batched world."""
    return tuple(
        capture_pre_weld_tree_obs_for_world(
            cable,
            layout,
            world=w,
            junction_names=junction_names,
        )
        for w in range(int(layout.num_envs))
    )


def complete_pre_weld_sysid_obs(
    tree_obs: dict[str, Any],
    *,
    pull_direction: Sequence[float],
) -> dict[str, Any]:
    """Fill sys-ID parquet columns for a pre-weld frame (no robot/TCP state yet)."""
    zeros6 = np.zeros(6, dtype=np.float32)
    zeros7 = np.zeros(7, dtype=np.float32)
    n_j = len(tree_obs["woody_part_start_pos"])
    return {
        **tree_obs,
        "excitation_type": 0,
        "excitation_direction": np.asarray(pull_direction, dtype=np.float32).reshape(3),
        "tcp_velocity": zeros6.copy(),
        "ft_wrist": zeros6.copy(),
        "raw_ft_wrist": zeros6.copy(),
        "tcp_pos": np.zeros(3, dtype=np.float32),
        "tcp_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot_joint_q": zeros7.copy(),
        "woody_part_force": np.zeros(n_j * 6, dtype=np.float32),
    }
