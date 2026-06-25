"""Unit tests for batched coupled-scene debug visualization."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from apple_pick_sim.batched_viz import (
    _apple_marker_position,
    _proxy_marker_position,
    _viewer_world_origin,
    _world_position,
    log_batched_endpoints,
    log_batched_tcp_force_arrows,
    log_batched_woody_part_endpoints,
)
from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout


class _MockViewer:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def log_arrows(self, name, starts, ends, colors, **kw) -> None:
        self.calls.append(("arrows", name, starts, ends, colors, kw))

    def log_lines(self, name, starts, ends, colors, **kw) -> None:
        self.calls.append(("lines", name, starts, ends, colors, kw))

    def log_points(self, name, points, **kw) -> None:
        self.calls.append(("points", name, points, kw))


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


def _mock_scene_with_forces(
    layout: BatchedEnvLayout,
    *,
    tcp_positions: list[tuple[float, float, float]],
    tcp_forces: list[tuple[float, float, float]],
) -> object:
    n_robot = layout.robot_bodies_per_world * layout.num_envs
    body_q = np.zeros((n_robot, 7), dtype=np.float64)
    for w, (x, y, z) in enumerate(tcp_positions):
        idx = layout.tcp_body_indices[w]
        body_q[idx, :3] = (x, y, z)
        body_q[idx, 6] = 1.0

    wrenches = np.zeros((n_robot, 6), dtype=np.float64)
    for w, force in enumerate(tcp_forces):
        idx = layout.tcp_body_indices[w]
        wrenches[idx, :3] = force

    class _State:
        pass

    _State.body_q = wp.array(body_q, dtype=wp.transform, device="cpu")

    class _Scene:
        proxy_forces = wp.array(wrenches, dtype=wp.spatial_vector, device="cpu")
        robot_state_0 = _State()
        robot_model = type("M", (), {"device": "cpu"})()
        env_spacing = layout.env_spacing

    return _Scene()


def _mock_scene_with_woody_joints(
    layout: BatchedEnvLayout,
    *,
    joint_pairs: list[tuple[int, str]] | None = None,
) -> object:
    n_cable = layout.bodies_per_world * layout.num_envs
    body_q = np.zeros((n_cable, 7), dtype=np.float64)
    body_q[:, 6] = 1.0

    class _CableState:
        pass

    _CableState.body_q = wp.array(body_q, dtype=wp.transform, device="cpu")

    if joint_pairs is None:
        joint_pairs = [
            (0, "joint_primary_secondary"),
            (1, "joint_stem_apple"),
            (2, "joint_apple_gripper_proxy"),
        ]

    class _Cable:
        state_0 = _CableState()
        model = type("M", (), {"device": "cpu"})()
        fruiting_fixed_joints = tuple(joint_pairs)

    class _Scene:
        cable = _Cable()
        env_spacing = layout.env_spacing

    return _Scene()


def _mock_scene_with_cable_bodies(
    layout: BatchedEnvLayout,
    *,
    apple_positions: list[tuple[float, float, float]],
    proxy_positions: list[tuple[float, float, float]],
) -> object:
    n_cable = layout.bodies_per_world * layout.num_envs
    body_q = np.zeros((n_cable, 7), dtype=np.float64)
    for w, pos in enumerate(apple_positions):
        if layout.template_apple_body is not None:
            idx = layout.apple_body_indices[w]
            body_q[idx, :3] = pos
            body_q[idx, 6] = 1.0
    for w, pos in enumerate(proxy_positions):
        idx = layout.proxy_body_indices[w]
        body_q[idx, :3] = pos
        body_q[idx, 6] = 1.0

    class _CableState:
        pass

    _CableState.body_q = wp.array(body_q, dtype=wp.transform, device="cpu")

    class _Cable:
        state_0 = _CableState()
        params = type("P", (), {"apple_radius": 0.04})()
        gripper_proxy_vis_offset = (0.0, 0.0, 0.08)
        gripper_proxy_config = type("C", (), {"box_half_extents": (0.03, 0.03, 0.03)})()

    class _Scene:
        cable = _Cable()
        env_spacing = layout.env_spacing

    return _Scene()


def test_log_batched_tcp_force_arrows_noop_when_proxy_forces_none():
    viewer = _MockViewer()
    layout = _two_env_layout()

    class _Scene:
        proxy_forces = None
        robot_state_0 = None
        robot_model = None

    log_batched_tcp_force_arrows(viewer, _Scene(), layout)
    assert len(viewer.calls) == 1
    kind, name, starts, ends, colors, _kw = viewer.calls[0]
    assert kind == "arrows"
    assert name == "/debug/tcp_force_arrow"
    assert starts is None and ends is None and colors is None


def test_world_position_adds_scene_viewer_origin():
    layout = _two_env_layout()
    scene = _mock_scene_with_cable_bodies(
        layout,
        apple_positions=[(0.0, 0.0, 0.0)],
        proxy_positions=[(0.0, 0.0, 0.0)],
    )
    local = np.array([0.1, 0.2, 1.5])
    w0 = _world_position(scene, layout, 0, local)
    w1 = _world_position(scene, layout, 1, local)
    np.testing.assert_allclose(w0, local + _viewer_world_origin(scene, layout, 0))
    np.testing.assert_allclose(w1, local + _viewer_world_origin(scene, layout, 1))
    assert not np.allclose(w0, w1)


def test_world_position_uses_scene_env_spacing_over_layout():
    layout = BatchedEnvLayout(
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
        env_spacing=(0.0, 0.0, 0.0),
    )

    class _Scene:
        env_spacing = (2.5, 2.5, 0.0)

    scene = _Scene()
    local = np.array([0.0, 0.0, 1.0])
    w1 = _world_position(scene, layout, 1, local)
    np.testing.assert_allclose(w1, local + _viewer_world_origin(scene, layout, 1))
    assert not np.allclose(w1, local)


def test_log_batched_tcp_force_arrows_draws_one_per_env():
    viewer = _MockViewer()
    layout = _two_env_layout()
    scene = _mock_scene_with_forces(
        layout,
        tcp_positions=[(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)],
        tcp_forces=[(10.0, 0.0, 0.0), (0.0, 20.0, 0.0)],
    )
    log_batched_tcp_force_arrows(
        viewer,
        scene,
        layout,
        scale_per_newton=0.02,
        min_length=0.08,
    )
    arrow_calls = [c for c in viewer.calls if c[0] == "arrows"]
    assert len(arrow_calls) == 1
    _kind, name, starts, ends, _colors, _kw = arrow_calls[0]
    assert name == "/debug/tcp_force_arrow"
    assert starts is not None and ends is not None
    starts_np = starts.numpy().reshape(-1, 3)
    ends_np = ends.numpy().reshape(-1, 3)
    assert starts_np.shape == (2, 3)
    assert ends_np.shape == (2, 3)
    np.testing.assert_allclose(
        starts_np[0],
        _world_position(scene, layout, 0, [0.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(
        starts_np[1],
        _world_position(scene, layout, 1, [0.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(ends_np[0], starts_np[0] + [0.2, 0.0, 0.0], rtol=0, atol=1e-5)
    np.testing.assert_allclose(ends_np[1], starts_np[1] + [0.0, 0.4, 0.0], rtol=0, atol=1e-5)


def test_log_batched_tcp_force_arrows_noop_without_log_lines():
    layout = _two_env_layout()
    scene = _mock_scene_with_forces(
        layout,
        tcp_positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        tcp_forces=[(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    )

    class _Viewer:
        pass

    log_batched_tcp_force_arrows(_Viewer(), scene, layout)


def test_log_batched_endpoints_draws_apple_and_proxy_red_crosses():
    viewer = _MockViewer()
    layout = _two_env_layout()
    scene = _mock_scene_with_cable_bodies(
        layout,
        apple_positions=[(0.1, 0.2, 1.5), (0.1, 0.2, 1.5)],
        proxy_positions=[(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)],
    )
    cable_bq = scene.cable.state_0.body_q.numpy().reshape(-1, 7)
    log_batched_endpoints(viewer, scene, layout, radius=0.05)
    line_calls = [c for c in viewer.calls if c[0] == "lines"]
    assert len(line_calls) == 2
    apple_call = next(c for c in line_calls if c[1] == "/debug/batched_apple_endpoints")
    proxy_call = next(c for c in line_calls if c[1] == "/debug/batched_proxy_endpoints")
    apple_starts = apple_call[2].numpy().reshape(-1, 3)
    proxy_starts = proxy_call[2].numpy().reshape(-1, 3)
    assert apple_starts.shape == (6, 3)
    assert proxy_starts.shape == (6, 3)
    apple_origin = _apple_marker_position(scene, layout, 0, cable_bq)
    proxy_origin = _proxy_marker_position(scene, layout, 0, cable_bq)
    np.testing.assert_allclose(apple_starts[1], apple_origin - [0.0, 0.05, 0.0])
    np.testing.assert_allclose(proxy_starts[2], proxy_origin - [0.0, 0.0, 0.05])
    apple_colors = apple_call[4].numpy().reshape(-1, 3)
    np.testing.assert_allclose(apple_colors, [[1.0, 0.0, 0.0]] * 6)


def test_log_batched_endpoints_skips_apple_when_none():
    viewer = _MockViewer()
    layout = _two_env_layout(template_apple_body=None)
    scene = _mock_scene_with_cable_bodies(
        layout,
        apple_positions=[],
        proxy_positions=[(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)],
    )
    log_batched_endpoints(viewer, scene, layout)
    line_calls = [c for c in viewer.calls if c[0] == "lines"]
    assert len(line_calls) == 1
    assert line_calls[0][1] == "/debug/batched_proxy_endpoints"


def test_log_batched_woody_part_endpoints_draws_red_points(monkeypatch):
    viewer = _MockViewer()
    layout = _two_env_layout()
    scene = _mock_scene_with_woody_joints(layout)
    n_fruiting_joints = 2  # gripper_proxy joint excluded

    def _fake_anchors(model, body_q, joint_pairs):
        del model, body_q
        n = len(joint_pairs)
        parent = np.array(
            [coord for i in range(n) for coord in (float(i), float(i) + 0.1, float(i) + 0.2)],
            dtype=np.float32,
        )
        child = parent + 0.5
        return parent, child

    monkeypatch.setattr(
        "apple_pick_sim.batched_viz.fixed_joint_anchors_world",
        _fake_anchors,
    )

    log_batched_woody_part_endpoints(viewer, scene, layout, radius=0.03)

    point_calls = [c for c in viewer.calls if c[0] == "points"]
    assert len(point_calls) == 1
    _kind, name, points, kw = point_calls[0]
    assert name == "/debug/batched_woody_endpoints"
    assert points is not None
    assert len(points) == n_fruiting_joints * layout.num_envs
    assert kw["radii"] is not None
    assert kw["colors"] is not None

    pts_np = np.array(
        [[float(p[0]), float(p[1]), float(p[2])] for p in points.numpy()],
        dtype=np.float64,
    )
    w0_j0 = _world_position(scene, layout, 0, [0.0, 0.1, 0.2])
    w0_j1 = _world_position(scene, layout, 0, [1.0, 1.1, 1.2])
    w1_j0 = _world_position(scene, layout, 1, [0.0, 0.1, 0.2])
    w1_j1 = _world_position(scene, layout, 1, [1.0, 1.1, 1.2])
    np.testing.assert_allclose(pts_np[0], w0_j0)
    np.testing.assert_allclose(pts_np[1], w0_j1)
    np.testing.assert_allclose(pts_np[2], w1_j0)
    np.testing.assert_allclose(pts_np[3], w1_j1)
    assert not np.allclose(pts_np[0], pts_np[2])


def test_log_batched_woody_part_endpoints_skips_gripper_proxy_joints(monkeypatch):
    viewer = _MockViewer()
    layout = _two_env_layout()
    scene = _mock_scene_with_woody_joints(
        layout,
        joint_pairs=[
            (0, "joint_primary_secondary"),
            (3, "joint_apple_gripper_proxy"),
        ],
    )
    captured: list[int] = []

    def _fake_anchors(model, body_q, joint_pairs):
        del model, body_q
        captured.append(len(joint_pairs))
        n = len(joint_pairs)
        parent = np.zeros(n * 3, dtype=np.float32)
        return parent, parent

    monkeypatch.setattr(
        "apple_pick_sim.batched_viz.fixed_joint_anchors_world",
        _fake_anchors,
    )

    log_batched_woody_part_endpoints(viewer, scene, layout)

    assert captured == [1, 1]
    point_calls = [c for c in viewer.calls if c[0] == "points"]
    assert len(point_calls) == 1
    assert len(point_calls[0][2]) == layout.num_envs


def test_log_batched_endpoints_includes_woody_dots(monkeypatch):
    viewer = _MockViewer()
    layout = _two_env_layout()
    scene = _mock_scene_with_cable_bodies(
        layout,
        apple_positions=[(0.1, 0.2, 1.5), (0.1, 0.2, 1.5)],
        proxy_positions=[(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)],
    )
    scene.cable.fruiting_fixed_joints = ((0, "joint_primary_secondary"),)
    scene.cable.model = type("M", (), {"device": "cpu"})()

    def _fake_anchors(model, body_q, joint_pairs):
        del model, body_q
        n = len(joint_pairs)
        parent = np.zeros(n * 3, dtype=np.float32)
        return parent, parent

    monkeypatch.setattr(
        "apple_pick_sim.batched_viz.fixed_joint_anchors_world",
        _fake_anchors,
    )

    log_batched_endpoints(viewer, scene, layout, radius=0.05)

    line_calls = [c for c in viewer.calls if c[0] == "lines"]
    point_calls = [c for c in viewer.calls if c[0] == "points"]
    assert len(line_calls) == 2
    assert len(point_calls) == 1
    assert point_calls[0][1] == "/debug/batched_woody_endpoints"
