"""GPU-native batched observation gather tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import warp as wp

from apple_pick_sim.batched_obs import (
    BatchedObsBuffers,
    gather_batched_obs,
    launch_gather_positions,
    launch_gather_spatial_vectors,
    launch_gather_transforms,
    make_batched_obs_buffers,
)
from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout

_DEVICE = "cpu"


def _two_env_layout(*, template_apple_body: int | None = 5) -> BatchedEnvLayout:
    return BatchedEnvLayout(
        num_envs=2,
        bodies_per_world=10,
        robot_bodies_per_world=8,
        joints_per_world=9,
        joint_coord_count_per_world=7,
        joint_dof_count_per_world=7,
        template_tcp_body=7,
        template_proxy_body=4,
        template_apple_body=template_apple_body,
        tcp_body_indices=(7, 15),
        proxy_body_indices=(4, 14),
        apple_body_indices=(5, 15) if template_apple_body is not None else (-1, -1),
        env_spacing=(2.5, 2.5, 0.0),
    )


def _transforms_from_xyz_quat(rows: list[tuple[float, ...]]) -> wp.array:
    host = np.zeros((len(rows), 7), dtype=np.float32)
    for i, row in enumerate(rows):
        host[i, : len(row)] = row
        if len(row) == 3:
            host[i, 6] = 1.0
    return wp.array(host, dtype=wp.transform, device=_DEVICE)


def test_gather_positions_known_transforms():
    body_q = _transforms_from_xyz_quat(
        [
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
            (7.0, 8.0, 9.0),
        ]
    )
    indices = wp.array([2, 0], dtype=wp.int32, device=_DEVICE)
    out = wp.zeros((2, 3), dtype=wp.float32, device=_DEVICE)
    launch_gather_positions(body_q, indices, out)
    got = out.numpy()
    np.testing.assert_allclose(got[0], [7.0, 8.0, 9.0])
    np.testing.assert_allclose(got[1], [1.0, 2.0, 3.0])


def test_gather_transforms_quaternion_passthrough():
    body_q = _transforms_from_xyz_quat(
        [
            (1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9),
            (4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0),
        ]
    )
    indices = wp.array([1], dtype=wp.int32, device=_DEVICE)
    out = wp.zeros((1, 7), dtype=wp.float32, device=_DEVICE)
    launch_gather_transforms(body_q, indices, out)
    np.testing.assert_allclose(out.numpy()[0], [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0])


def test_gather_spatial_vectors_known_forces():
    host = np.zeros((3, 6), dtype=np.float32)
    host[2] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    wrenches = wp.array(host, dtype=wp.spatial_vector, device=_DEVICE)
    indices = wp.array([2, 0], dtype=wp.int32, device=_DEVICE)
    out = wp.zeros((2, 6), dtype=wp.float32, device=_DEVICE)
    launch_gather_spatial_vectors(wrenches, indices, out)
    got = out.numpy()
    np.testing.assert_allclose(got[0], host[2])
    np.testing.assert_allclose(got[1], 0.0)


def test_gather_positions_flat_for_woody():
    num_envs = 2
    num_junctions = 3
    n_bodies = 12
    host = np.zeros((n_bodies, 7), dtype=np.float32)
    host[:, 6] = 1.0
    for flat in range(num_envs * num_junctions):
        host[flat, :3] = [float(flat), float(flat) + 0.5, float(flat) + 1.0]
    body_q = wp.array(host, dtype=wp.transform, device=_DEVICE)
    indices = wp.array(np.arange(num_envs * num_junctions, dtype=np.int32), device=_DEVICE)
    out = wp.zeros((num_envs * num_junctions, 3), dtype=wp.float32, device=_DEVICE)
    launch_gather_positions(body_q, indices, out)
    reshaped = out.numpy().reshape(num_envs, num_junctions, 3)
    assert reshaped.shape == (2, 3, 3)
    np.testing.assert_allclose(reshaped[0, 1], [1.0, 1.5, 2.0])


def _mock_cable_for_buffers(layout: BatchedEnvLayout):
    joint_pairs = [
        (0, "joint_a_b"),
        (1, "joint_b_c"),
    ]
    parent_np = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    child_np = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)

    class _Model:
        joint_parent = wp.array(parent_np, dtype=wp.int32, device=_DEVICE)
        joint_child = wp.array(child_np, dtype=wp.int32, device=_DEVICE)

    class _Cable:
        model = _Model()
        fruiting_fixed_joints = tuple(joint_pairs)

    return _Cable()


def test_make_batched_obs_buffers_shapes():
    layout = _two_env_layout()
    cable = _mock_cable_for_buffers(layout)
    bufs = make_batched_obs_buffers(layout, cable, _DEVICE)
    assert isinstance(bufs, BatchedObsBuffers)
    assert bufs.num_envs == 2
    assert bufs.num_junctions == 2
    assert bufs.apple_pos.shape == (2, 3)
    assert bufs.proxy_pos.shape == (2, 3)
    assert bufs.tcp_pose.shape == (2, 7)
    assert bufs.tcp_force.shape == (2, 6)
    assert bufs.tcp_coupling_force.shape == (2, 6)
    assert bufs.woody_parent_pos.shape == (4, 3)
    assert bufs.woody_child_pos.shape == (4, 3)
    assert bufs.woody_force.shape == (4, 3)
    assert bufs.woody_torque.shape == (4, 3)
    assert bufs.apple_indices.shape == (2,)
    assert bufs.woody_joint_indices.shape == (4,)


def _mock_scene_for_gather(layout: BatchedEnvLayout, cable):
    n_cable = layout.bodies_per_world * layout.num_envs
    n_robot = layout.robot_bodies_per_world * layout.num_envs
    cable_bq = np.zeros((n_cable, 7), dtype=np.float32)
    cable_bq[:, 6] = 1.0
    for w in range(layout.num_envs):
        apple_idx = layout.apple_body_indices[w]
        proxy_idx = layout.proxy_body_indices[w]
        cable_bq[apple_idx, :3] = [1.0 + w, 2.0 + w, 3.0 + w]
        cable_bq[proxy_idx, :3] = [4.0 + w, 5.0 + w, 6.0 + w]
        for j in range(cable.fruiting_fixed_joints.__len__()):
            flat = w * len(cable.fruiting_fixed_joints) + j
            parent_idx = layout.body_index(w, int(cable.model.joint_parent.numpy()[j]))
            child_idx = layout.body_index(w, int(cable.model.joint_child.numpy()[j]))
            cable_bq[parent_idx, :3] = [10.0 + flat, 11.0 + flat, 12.0 + flat]
            cable_bq[child_idx, :3] = [20.0 + flat, 21.0 + flat, 22.0 + flat]

    robot_bq = np.zeros((n_robot, 7), dtype=np.float32)
    robot_bq[:, 6] = 1.0
    for w in range(layout.num_envs):
        tcp_idx = layout.tcp_body_indices[w]
        robot_bq[tcp_idx, :3] = [7.0 + w, 8.0 + w, 9.0 + w]
        robot_bq[tcp_idx, 3:7] = [0.0, 0.0, 0.0, 1.0]

    proxy_forces = np.zeros((n_robot, 6), dtype=np.float32)
    coupling_cache = np.zeros((n_robot, 6), dtype=np.float32)
    for w in range(layout.num_envs):
        tcp_idx = layout.tcp_body_indices[w]
        proxy_forces[tcp_idx] = [1.0 + w, 0.0, 0.0, 0.0, 2.0 + w, 0.0]
        coupling_cache[tcp_idx] = [0.0, 3.0 + w, 0.0, 0.0, 0.0, 4.0 + w]

    class _CableState:
        body_q = wp.array(cable_bq, dtype=wp.transform, device=_DEVICE)

    class _CableStatePrev:
        body_q = wp.array(cable_bq.copy(), dtype=wp.transform, device=_DEVICE)

    cable.state_0 = _CableState()
    cable.state_1 = _CableStatePrev()
    cable.solver = object()

    class _RobotState:
        body_q = wp.array(robot_bq, dtype=wp.transform, device=_DEVICE)

    return SimpleNamespace(
        cable=cable,
        robot_state_0=_RobotState(),
        proxy_forces=wp.array(proxy_forces, dtype=wp.spatial_vector, device=_DEVICE),
        coupling_forces_cache=wp.array(coupling_cache, dtype=wp.spatial_vector, device=_DEVICE),
    )


def test_gather_batched_obs_end_to_end():
    layout = _two_env_layout()
    cable = _mock_cable_for_buffers(layout)
    bufs = make_batched_obs_buffers(layout, cable, _DEVICE)
    scene = _mock_scene_for_gather(layout, cable)

    woody_force = np.zeros((bufs.num_envs * bufs.num_junctions, 3), dtype=np.float32)
    woody_torque = np.zeros_like(woody_force)
    for flat in range(woody_force.shape[0]):
        woody_force[flat] = [0.5 + flat, 0.0, 0.0]
        woody_torque[flat] = [0.0, 0.0, 1.0 + flat]

    def _fake_gather(*_args, **_kwargs):
        return (
            wp.array(woody_force, dtype=wp.vec3, device=_DEVICE),
            wp.array(woody_torque, dtype=wp.vec3, device=_DEVICE),
        )

    with patch(
        "apple_pick_sim.batched_obs.gather_joint_wrench_child_com_device",
        side_effect=_fake_gather,
    ):
        gather_batched_obs(bufs, scene, sim_dt=1.0 / 900.0)

    np.testing.assert_allclose(bufs.apple_pos.numpy()[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(bufs.apple_pos.numpy()[1], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(bufs.proxy_pos.numpy()[0], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(bufs.tcp_pose.numpy()[1, :3], [8.0, 9.0, 10.0])
    np.testing.assert_allclose(bufs.tcp_force.numpy()[0], [1.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    np.testing.assert_allclose(bufs.tcp_coupling_force.numpy()[1], [0.0, 4.0, 0.0, 0.0, 0.0, 5.0])
    np.testing.assert_allclose(bufs.woody_parent_pos.numpy()[0], [10.0, 11.0, 12.0])
    np.testing.assert_allclose(bufs.woody_child_pos.numpy()[1], [21.0, 22.0, 23.0])
    np.testing.assert_allclose(bufs.woody_force.numpy()[0], [0.5, 0.0, 0.0])
    np.testing.assert_allclose(bufs.woody_torque.numpy()[1], [0.0, 0.0, 2.0])
