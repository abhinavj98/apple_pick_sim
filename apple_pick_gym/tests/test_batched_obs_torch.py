"""Tests for batched obs buffer → torch dict reshaping."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

from apple_pick_gym.batched_envs.obs_torch import legacy_v3_numpy_from_batched, obs_dict_from_bufs
from apple_pick_sim.batched_obs import BatchedObsBuffers, make_batched_obs_buffers
from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.fruiting_system.coupled import CoupledCableScene


class _FakeJoint:
    def __init__(self, parent: int, child: int) -> None:
        self.parent = parent
        self.child = child


class _FakeModel:
    def __init__(self) -> None:
        self.joint_parent = wp.array([0, 1], dtype=wp.int32, device="cpu")
        self.joint_child = wp.array([1, 2], dtype=wp.int32, device="cpu")
        self.joint_X_p = wp.array([wp.transform() for _ in range(2)], dtype=wp.transform, device="cpu")
        self.joint_X_c = wp.array([wp.transform() for _ in range(2)], dtype=wp.transform, device="cpu")


class _FakeCable:
    fruiting_fixed_joints = [(0, "joint_a_b"), (1, "joint_b_c")]

    def __init__(self) -> None:
        self.model = _FakeModel()
        self.solver = object()
        self.fruiting_fixed_joints = list(self.fruiting_fixed_joints)


def _layout() -> BatchedEnvLayout:
    return BatchedEnvLayout(
        num_envs=2,
        bodies_per_world=10,
        robot_bodies_per_world=8,
        joints_per_world=9,
        joint_coord_count_per_world=7,
        joint_dof_count_per_world=7,
        template_tcp_body=7,
        template_proxy_body=4,
        template_apple_body=5,
        tcp_body_indices=(7, 15),
        proxy_body_indices=(4, 14),
        apple_body_indices=(5, 15),
        env_spacing=(2.5, 2.5, 0.0),
    )


def test_obs_dict_from_bufs_shapes():
    layout = _layout()
    cable = _FakeCable()
    bufs = make_batched_obs_buffers(layout, cable, "cpu")
    names = ["a_b", "b_c"]

    # deterministic fill
    parent_np = np.arange(2 * 2 * 3, dtype=np.float32).reshape(4, 3)
    child_np = parent_np + 100.0
    force_np = np.ones((4, 3), dtype=np.float32)
    torque_np = np.full((4, 3), 2.0, dtype=np.float32)
    bufs.woody_parent_pos.assign(parent_np)
    bufs.woody_child_pos.assign(child_np)
    bufs.woody_force.assign(force_np)
    bufs.woody_torque.assign(torque_np)
    bufs.apple_pos.assign(np.ones((2, 3), dtype=np.float32))
    bufs.tcp_force.assign(np.full((2, 6), 3.0, dtype=np.float32))
    bufs.tcp_velocity.assign(np.full((2, 6), 4.0, dtype=np.float32))
    bufs.tcp_coupling_force.assign(np.full((2, 6), 5.0, dtype=np.float32))

    obs = obs_dict_from_bufs(bufs, names, torch.device("cpu"))
    assert obs["apple_pos"].shape == (2, 3)
    assert obs["woody_part_info"]["a_b"]["anchors_pos"].shape == (2, 6)
    assert obs["woody_part_info"]["a_b"]["anchor_force"].shape == (2, 6)
    torch.testing.assert_close(
        obs["woody_part_info"]["a_b"]["anchors_pos"][0, :3],
        torch.tensor(parent_np[0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        obs["woody_part_info"]["a_b"]["anchor_force"][1],
        torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=torch.float32),
    )


def test_legacy_v3_mapping_num_envs_one():
    names = ["stem_apple"]
    obs = {
        "woody_part_info": {
            "stem_apple": {
                "anchors_pos": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
                "anchor_force": torch.tensor([[7.0, 8.0, 9.0, 1.0, 2.0, 3.0]]),
            }
        },
        "apple_pos": torch.tensor([[0.1, 0.2, 0.3]]),
        "tcp_force": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "tcp_velocity": torch.zeros(1, 6),
        "ft_wrist": torch.ones(1, 6),
    }
    legacy = legacy_v3_numpy_from_batched(obs, names)
    np.testing.assert_allclose(legacy["woody_part_start_pos"]["stem_apple"], [1, 2, 3])
    np.testing.assert_allclose(legacy["woody_part_end_pos"]["stem_apple"], [4, 5, 6])
    np.testing.assert_allclose(legacy["woody_part_force"], [7, 8, 9, 1, 2, 3])
