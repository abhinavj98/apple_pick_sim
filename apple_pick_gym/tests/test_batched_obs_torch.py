"""Tests for batched obs buffer → torch dict reshaping."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

from apple_pick_gym.batched_envs.obs_torch import (
    download_batched_replay_obs_numpy,
    legacy_v3_numpy_from_batched,
    obs_dict_from_bufs,
    replay_obs_dict_from_batched_numpy_row,
    sysid_numpy_obs_from_batched,
)
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


def _batched_sysid_torch_obs(*, num_envs: int, junction_names: list[str]) -> dict:
    woody_part_info: dict[str, dict[str, torch.Tensor]] = {}
    for j, name in enumerate(junction_names):
        anchors = torch.zeros(num_envs, 6, dtype=torch.float32)
        forces = torch.zeros(num_envs, 6, dtype=torch.float32)
        for env_idx in range(num_envs):
            base = float(env_idx * 100 + j * 10)
            anchors[env_idx] = torch.arange(6, dtype=torch.float32) + base
            forces[env_idx] = torch.arange(6, dtype=torch.float32) + base + 0.5
        woody_part_info[name] = {
            "anchors_pos": anchors,
            "anchor_force": forces,
        }

    def _row_tensor(cols: int, scale: float) -> torch.Tensor:
        rows = torch.zeros(num_envs, cols, dtype=torch.float32)
        for env_idx in range(num_envs):
            rows[env_idx] = torch.full((cols,), float(env_idx * scale), dtype=torch.float32) + torch.arange(
                cols, dtype=torch.float32
            )
        return rows

    return {
        "woody_part_info": woody_part_info,
        "apple_pos": _row_tensor(3, scale=1.0),
        "tcp_force": _row_tensor(6, scale=2.0),
        "tcp_velocity": _row_tensor(6, scale=3.0),
        "ft_wrist": _row_tensor(6, scale=4.0),
        "raw_ft_wrist": _row_tensor(6, scale=5.0),
        "tcp_pos": _row_tensor(3, scale=6.0),
        "tcp_quat": _row_tensor(4, scale=7.0),
        "apple_quat": _row_tensor(4, scale=8.0),
        "robot_joint_q": _row_tensor(7, scale=9.0),
        "excitation_type": torch.tensor([env_idx % 3 for env_idx in range(num_envs)], dtype=torch.long),
        "excitation_f_inst": torch.tensor(
            [float(env_idx) for env_idx in range(num_envs)],
            dtype=torch.float32,
        ),
        "excitation_direction": _row_tensor(3, scale=10.0),
    }


def test_sysid_numpy_obs_from_batched_reads_requested_env_idx():
    junction_names = ["joint_a", "joint_b"]
    obs = _batched_sysid_torch_obs(num_envs=2, junction_names=junction_names)

    exported = sysid_numpy_obs_from_batched(obs, junction_names, env_idx=1)

    np.testing.assert_allclose(exported["apple_pos"], obs["apple_pos"][1].numpy())
    np.testing.assert_allclose(exported["ft_wrist"], obs["ft_wrist"][1].numpy())
    np.testing.assert_allclose(
        exported["woody_part_start_pos"]["joint_a"],
        obs["woody_part_info"]["joint_a"]["anchors_pos"][1, :3].numpy(),
    )
    assert exported["apple_pos"][0] != pytest.approx(float(obs["apple_pos"][0, 0].item()))


def test_download_batched_replay_obs_numpy_matches_per_env_sysid_export():
    junction_names = ["joint_a", "joint_b"]
    obs = _batched_sysid_torch_obs(num_envs=3, junction_names=junction_names)

    batched = download_batched_replay_obs_numpy(obs, junction_names)

    assert batched["ft_wrist"].shape == (3, 6)
    assert batched["woody_start"].shape == (3, 6)
    assert batched["woody_end"].shape == (3, 6)

    for env_idx in range(3):
        per_env = sysid_numpy_obs_from_batched(obs, junction_names, env_idx=env_idx)
        row = replay_obs_dict_from_batched_numpy_row(batched, env_idx=env_idx)
        np.testing.assert_allclose(row["ft_wrist"], per_env["ft_wrist"])
        np.testing.assert_allclose(row["tcp_velocity"], per_env["tcp_velocity"])
        np.testing.assert_allclose(row["tcp_pos"], per_env["tcp_pos"])
        np.testing.assert_allclose(row["apple_pos"], per_env["apple_pos"])
        np.testing.assert_allclose(
            row["woody_start"],
            np.concatenate(
                [per_env["woody_part_start_pos"][name] for name in junction_names],
                dtype=np.float32,
            ),
        )


def test_download_batched_replay_obs_numpy_copies_away_from_torch_buffer():
    """CPU numpy must not alias live torch obs (else replay collectors freeze)."""
    junction_names = ["joint_a"]
    obs = _batched_sysid_torch_obs(num_envs=1, junction_names=junction_names)
    snap = download_batched_replay_obs_numpy(obs, junction_names)
    tcp_before = snap["tcp_pos"].copy()
    obs["tcp_pos"].fill_(9.0)
    np.testing.assert_allclose(snap["tcp_pos"], tcp_before)
    assert not np.allclose(snap["tcp_pos"], 9.0)
