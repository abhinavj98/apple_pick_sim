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
    launch_gather_contiguous_slices,
    launch_gather_fixed_joint_anchors,
    launch_gather_positions,
    launch_gather_spatial_vectors,
    launch_gather_transforms,
    make_batched_obs_buffers,
)
from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.fruiting_system.scene import fixed_joint_anchors_world

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


def _identity_transforms(n: int) -> wp.array:
    host = np.zeros((n, 7), dtype=np.float32)
    host[:, 6] = 1.0
    return wp.array(host, dtype=wp.transform, device=_DEVICE)


def _mock_cable_for_buffers(layout: BatchedEnvLayout):
    joint_pairs = [
        (0, "joint_a_b"),
        (1, "joint_b_c"),
        (2, "joint_apple_gripper_proxy"),
    ]
    tpl_parent = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    tpl_child = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    tpl_xp = np.zeros((9, 7), dtype=np.float32)
    tpl_xc = np.zeros((9, 7), dtype=np.float32)
    tpl_xp[:, 6] = 1.0
    tpl_xc[:, 6] = 1.0
    tpl_xp[0, :3] = [0.1, 0.0, 0.0]
    tpl_xc[0, :3] = [0.0, 0.2, 0.0]
    tpl_xp[1, :3] = [0.0, 0.1, 0.0]
    tpl_xc[1, :3] = [0.0, 0.0, 0.2]
    n_joints = 9 * layout.num_envs
    parent_np = np.tile(tpl_parent, layout.num_envs)
    child_np = np.tile(tpl_child, layout.num_envs)
    for w in range(1, layout.num_envs):
        offset = w * layout.bodies_per_world
        parent_np[w * 9 : (w + 1) * 9] = np.where(
            tpl_parent >= 0, tpl_parent + offset, tpl_parent
        )
        child_np[w * 9 : (w + 1) * 9] = tpl_child + offset
    xp = np.tile(tpl_xp, (layout.num_envs, 1))
    xc = np.tile(tpl_xc, (layout.num_envs, 1))

    class _Model:
        joint_parent = wp.array(parent_np, dtype=wp.int32, device=_DEVICE)
        joint_child = wp.array(child_np, dtype=wp.int32, device=_DEVICE)
        joint_X_p = wp.array(xp, dtype=wp.transform, device=_DEVICE)
        joint_X_c = wp.array(xc, dtype=wp.transform, device=_DEVICE)

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
    assert bufs.num_junctions == 2  # gripper_proxy joint excluded
    assert bufs.apple_pos.shape == (2, 3)
    assert bufs.apple_pose.shape == (2, 7)
    assert bufs.proxy_pos.shape == (2, 3)
    assert bufs.tcp_pose.shape == (2, 7)
    assert bufs.tcp_force.shape == (2, 6)
    assert bufs.tcp_coupling_force.shape == (2, 6)
    assert bufs.woody_parent_pos.shape == (4, 3)
    assert bufs.woody_child_pos.shape == (4, 3)
    assert bufs.woody_force.shape == (4, 3)
    assert bufs.woody_torque.shape == (4, 3)
    assert bufs.tcp_velocity.shape == (2, 6)
    assert bufs.joint_q.shape == (2, 7)
    assert bufs.joint_qd.shape == (2, 7)
    assert bufs.woody_force_scratch.shape == (4,)
    assert bufs.woody_torque_scratch.shape == (4,)
    assert bufs.joint_coord_count == 7
    assert bufs.joint_dof_count == 7
    assert bufs.apple_indices.shape == (2,)
    assert bufs.woody_joint_indices.shape == (4,)


def test_gather_woody_anchors_matches_cpu_reference():
    layout = _two_env_layout()
    cable = _mock_cable_for_buffers(layout)
    bufs = make_batched_obs_buffers(layout, cable, _DEVICE)
    n_cable = layout.bodies_per_world * layout.num_envs
    cable_bq = np.zeros((n_cable, 7), dtype=np.float32)
    cable_bq[:, 6] = 1.0
    for w in range(layout.num_envs):
        offset = float(w * layout.bodies_per_world)
        cable_bq[int(offset), :3] = [1.0, 2.0, 3.0]
        cable_bq[int(offset) + 1, :3] = [4.0, 5.0, 6.0]
        cable_bq[int(offset) + 2, :3] = [7.0, 8.0, 9.0]
    body_q = wp.array(cable_bq, dtype=wp.transform, device=_DEVICE)

    launch_gather_fixed_joint_anchors(
        body_q,
        cable.model,
        bufs.woody_joint_indices,
        bufs.woody_parent_pos,
        bufs.woody_child_pos,
    )

    joint_pairs = [(0, "joint_a_b"), (1, "joint_b_c")]
    for w in range(layout.num_envs):
        world_pairs = [(layout.joint_index(w, ji), label) for ji, label in joint_pairs]
        parent_cpu, child_cpu = fixed_joint_anchors_world(cable.model, body_q, world_pairs)
        parent_cpu = parent_cpu.reshape(-1, 3)
        child_cpu = child_cpu.reshape(-1, 3)
        base = w * len(joint_pairs)
        np.testing.assert_allclose(
            bufs.woody_parent_pos.numpy()[base : base + len(joint_pairs)],
            parent_cpu,
            rtol=0,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            bufs.woody_child_pos.numpy()[base : base + len(joint_pairs)],
            child_cpu,
            rtol=0,
            atol=1e-5,
        )


def _mock_scene_for_gather(layout: BatchedEnvLayout, cable):
    n_cable = layout.bodies_per_world * layout.num_envs
    n_robot = layout.robot_bodies_per_world * layout.num_envs
    cable_bq = np.zeros((n_cable, 7), dtype=np.float32)
    cable_bq[:, 6] = 1.0
    for w in range(layout.num_envs):
        apple_idx = layout.apple_body_indices[w]
        proxy_idx = layout.proxy_body_indices[w]
        cable_bq[apple_idx, :3] = [1.0 + w, 2.0 + w, 3.0 + w]
        cable_bq[apple_idx, 3:7] = [0.1 + w, 0.2, 0.3, 0.9]
        cable_bq[proxy_idx, :3] = [4.0 + w, 5.0 + w, 6.0 + w]
        offset = float(w * layout.bodies_per_world)
        cable_bq[int(offset), :3] = [10.0 + w, 11.0 + w, 12.0 + w]
        cable_bq[int(offset) + 1, :3] = [20.0 + w, 21.0 + w, 22.0 + w]
        cable_bq[int(offset) + 2, :3] = [30.0 + w, 31.0 + w, 32.0 + w]

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
        body_qd = wp.zeros(n_robot, dtype=wp.spatial_vector, device=_DEVICE)
        joint_q = wp.zeros(layout.num_envs * layout.joint_coord_count_per_world, dtype=wp.float32, device=_DEVICE)
        joint_qd = wp.zeros(layout.num_envs * layout.joint_dof_count_per_world, dtype=wp.float32, device=_DEVICE)

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

    def _fake_gather(*_args, **kwargs):
        n = bufs.num_envs * bufs.num_junctions
        woody_force = np.zeros((n, 3), dtype=np.float32)
        woody_torque = np.zeros_like(woody_force)
        for flat in range(n):
            woody_force[flat] = [0.5 + flat, 0.0, 0.0]
            woody_torque[flat] = [0.0, 0.0, 1.0 + flat]
        out_f = kwargs.get("out_f")
        out_t = kwargs.get("out_t")
        if out_f is not None and out_t is not None:
            out_f.assign(wp.array(woody_force, dtype=wp.vec3, device=_DEVICE))
            out_t.assign(wp.array(woody_torque, dtype=wp.vec3, device=_DEVICE))
            return out_f, out_t
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
    np.testing.assert_allclose(bufs.apple_pose.numpy()[0, :3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(bufs.apple_pose.numpy()[0, 3:7], [0.1, 0.2, 0.3, 0.9])
    np.testing.assert_allclose(bufs.apple_pose.numpy()[1, 3:7], [1.1, 0.2, 0.3, 0.9])
    np.testing.assert_allclose(bufs.proxy_pos.numpy()[0], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(bufs.tcp_pose.numpy()[1, :3], [8.0, 9.0, 10.0])
    np.testing.assert_allclose(bufs.tcp_force.numpy()[0], [1.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    np.testing.assert_allclose(bufs.tcp_coupling_force.numpy()[1], [0.0, 4.0, 0.0, 0.0, 0.0, 5.0])
    joint_pairs = [(0, "joint_a_b"), (1, "joint_b_c")]
    parent_cpu, child_cpu = fixed_joint_anchors_world(
        cable.model, cable.state_0.body_q, joint_pairs
    )
    parent_ref = parent_cpu.reshape(-1, 3)
    child_ref = child_cpu.reshape(-1, 3)
    np.testing.assert_allclose(bufs.woody_parent_pos.numpy()[0], parent_ref[0], atol=1e-5)
    np.testing.assert_allclose(bufs.woody_child_pos.numpy()[1], child_ref[1], atol=1e-5)
    np.testing.assert_allclose(bufs.woody_force.numpy()[0], [0.5, 0.0, 0.0])
    np.testing.assert_allclose(bufs.woody_torque.numpy()[1], [0.0, 0.0, 2.0])


def test_gather_batched_obs_cable_only_skips_robot():
    layout = _two_env_layout()
    cable = _mock_cable_for_buffers(layout)
    bufs = make_batched_obs_buffers(layout, cable, _DEVICE)
    n_cable = layout.bodies_per_world * layout.num_envs
    cable_bq = np.zeros((n_cable, 7), dtype=np.float32)
    cable_bq[:, 6] = 1.0
    cable_bq[layout.apple_body_indices[0], :3] = [1.0, 2.0, 3.0]

    class _CableState:
        body_q = wp.array(cable_bq, dtype=wp.transform, device=_DEVICE)

    class _CableStatePrev:
        body_q = wp.array(cable_bq.copy(), dtype=wp.transform, device=_DEVICE)

    cable.state_0 = _CableState()
    cable.state_1 = _CableStatePrev()
    cable.solver = object()
    scene = SimpleNamespace(
        cable=cable,
        robot_state_0=None,
        proxy_forces=None,
        coupling_forces_cache=None,
    )

    with patch(
        "apple_pick_sim.batched_obs.gather_joint_wrench_child_com_device",
        side_effect=lambda *_a, **_k: (
            wp.zeros(bufs.num_envs * bufs.num_junctions, dtype=wp.vec3, device=_DEVICE),
            wp.zeros(bufs.num_envs * bufs.num_junctions, dtype=wp.vec3, device=_DEVICE),
        ),
    ):
        gather_batched_obs(
            bufs,
            scene,
            sim_dt=1.0 / 900.0,
            include_robot=False,
            include_forces=False,
        )

    np.testing.assert_allclose(bufs.apple_pos.numpy()[0], [1.0, 2.0, 3.0])
    assert np.allclose(bufs.tcp_pose.numpy(), 0.0)


def _spatial_vectors_from_rows(rows: list[tuple[float, ...]]) -> wp.array:
    host = np.zeros((len(rows), 6), dtype=np.float32)
    for i, row in enumerate(rows):
        host[i, : len(row)] = row
    return wp.array(host, dtype=wp.spatial_vector, device=_DEVICE)


def test_gather_tcp_velocity_known_body_qd():
    layout = _two_env_layout()
    n_robot = layout.robot_bodies_per_world * layout.num_envs
    host = np.zeros((n_robot, 6), dtype=np.float32)
    for w in range(layout.num_envs):
        tcp_idx = layout.tcp_body_indices[w]
        host[tcp_idx] = [1.0 + w, 2.0 + w, 3.0 + w, 4.0 + w, 5.0 + w, 6.0 + w]
    body_qd = _spatial_vectors_from_rows([tuple(row) for row in host])
    indices = wp.array(list(layout.tcp_body_indices), dtype=wp.int32, device=_DEVICE)
    out = wp.zeros((layout.num_envs, 6), dtype=wp.float32, device=_DEVICE)
    launch_gather_spatial_vectors(body_qd, indices, out)
    got = out.numpy()
    np.testing.assert_allclose(got[0], host[layout.tcp_body_indices[0]])
    np.testing.assert_allclose(got[1], host[layout.tcp_body_indices[1]])


def test_gather_joint_q_known_flat_array():
    layout = _two_env_layout()
    coord = layout.joint_coord_count_per_world
    flat = np.arange(layout.num_envs * coord, dtype=np.float32) * 0.1
    src = wp.array(flat, dtype=wp.float32, device=_DEVICE)
    out = wp.zeros((layout.num_envs, coord), dtype=wp.float32, device=_DEVICE)
    launch_gather_contiguous_slices(src, coord, out)
    got = out.numpy()
    for w in range(layout.num_envs):
        np.testing.assert_allclose(got[w], flat[w * coord : (w + 1) * coord])


def test_gather_joint_qd_known_flat_array():
    layout = _two_env_layout()
    dof = layout.joint_dof_count_per_world
    flat = np.linspace(0.0, 1.0, layout.num_envs * dof, dtype=np.float32)
    src = wp.array(flat, dtype=wp.float32, device=_DEVICE)
    out = wp.zeros((layout.num_envs, dof), dtype=wp.float32, device=_DEVICE)
    launch_gather_contiguous_slices(src, dof, out)
    got = out.numpy()
    for w in range(layout.num_envs):
        np.testing.assert_allclose(got[w], flat[w * dof : (w + 1) * dof])


def test_gather_batched_obs_robot_state_fields():
    layout = _two_env_layout()
    cable = _mock_cable_for_buffers(layout)
    bufs = make_batched_obs_buffers(layout, cable, _DEVICE)
    scene = _mock_scene_for_gather(layout, cable)

    n_robot = layout.robot_bodies_per_world * layout.num_envs
    robot_bqd = np.zeros((n_robot, 6), dtype=np.float32)
    for w in range(layout.num_envs):
        tcp_idx = layout.tcp_body_indices[w]
        robot_bqd[tcp_idx] = [0.1 + w, 0.2 + w, 0.3 + w, 0.4 + w, 0.5 + w, 0.6 + w]

    coord = layout.joint_coord_count_per_world
    dof = layout.joint_dof_count_per_world
    joint_q_host = np.arange(layout.num_envs * coord, dtype=np.float32) * 0.05
    joint_qd_host = np.linspace(0.0, 0.5, layout.num_envs * dof, dtype=np.float32)

    class _RobotState:
        body_q = scene.robot_state_0.body_q
        body_qd = _spatial_vectors_from_rows([tuple(row) for row in robot_bqd])
        joint_q = wp.array(joint_q_host, dtype=wp.float32, device=_DEVICE)
        joint_qd = wp.array(joint_qd_host, dtype=wp.float32, device=_DEVICE)

    scene.robot_state_0 = _RobotState()
    scene.robot_model = object()

    with patch(
        "apple_pick_sim.batched_obs.gather_joint_wrench_child_com_device",
        side_effect=lambda *_a, **_k: (
            wp.zeros(bufs.num_envs * bufs.num_junctions, dtype=wp.vec3, device=_DEVICE),
            wp.zeros(bufs.num_envs * bufs.num_junctions, dtype=wp.vec3, device=_DEVICE),
        ),
    ):
        gather_batched_obs(bufs, scene, sim_dt=1.0 / 900.0)

    for w in range(layout.num_envs):
        tcp_idx = layout.tcp_body_indices[w]
        np.testing.assert_allclose(bufs.tcp_velocity.numpy()[w], robot_bqd[tcp_idx])
        np.testing.assert_allclose(
            bufs.joint_q.numpy()[w],
            joint_q_host[w * coord : (w + 1) * coord],
        )
        np.testing.assert_allclose(
            bufs.joint_qd.numpy()[w],
            joint_qd_host[w * dof : (w + 1) * dof],
        )


def test_gather_batched_obs_passes_wp_joint_indices_to_wrench_gather():
    layout = _two_env_layout()
    cable = _mock_cable_for_buffers(layout)
    bufs = make_batched_obs_buffers(layout, cable, _DEVICE)
    scene = _mock_scene_for_gather(layout, cable)
    captured: dict[str, object] = {}

    def _capture(*_args, **kwargs):
        captured["joint_indices"] = kwargs.get("joint_indices")
        captured["out_f"] = kwargs.get("out_f")
        captured["out_t"] = kwargs.get("out_t")
        n = bufs.num_envs * bufs.num_junctions
        return (
            wp.zeros(n, dtype=wp.vec3, device=_DEVICE),
            wp.zeros(n, dtype=wp.vec3, device=_DEVICE),
        )

    with patch(
        "apple_pick_sim.batched_obs.gather_joint_wrench_child_com_device",
        side_effect=_capture,
    ):
        gather_batched_obs(bufs, scene, sim_dt=1.0 / 900.0)

    assert isinstance(captured["joint_indices"], wp.array)
    assert captured["out_f"] is bufs.woody_force_scratch
    assert captured["out_t"] is bufs.woody_torque_scratch
