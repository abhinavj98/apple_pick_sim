"""HarvestVideoRecorder + example action scaling: fake viewer/env, no GL or sim needed."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from apple_pick_gym.batched_envs.harvest_action import HarvestActionBounds
from apple_pick_gym.batched_envs.harvest_video import (
    HarvestVideoRecorder,
    fit_camera,
    pick_env,
    world_points,
)


class _Arr:
    def __init__(self, a):
        self._a = np.asarray(a)

    def numpy(self):
        return self._a


class _FakeViewer:
    def __init__(self, h=32, w=48):
        self.frame = np.full((h, w, 3), 128, dtype=np.uint8)
        self.visible: list[list[int]] = []
        self.cameras: list[tuple] = []
        self.models: list[object] = []
        self.offsets: list[tuple] = []
        self.frames_begun = 0
        self.closed = False
        self.camera = SimpleNamespace(fov=45.0)

    def set_model(self, m):
        self.models.append(m)

    def set_visible_worlds(self, worlds):
        self.visible.append(list(worlds))

    def set_world_offsets(self, spacing):
        self.offsets.append(tuple(spacing))

    def set_camera(self, pos, pitch, yaw):
        self.cameras.append((pos, pitch, yaw))

    def begin_frame(self, t):
        self.frames_begun += 1

    def log_state(self, s):
        pass

    def log_contacts(self, c, s):
        pass

    def end_frame(self):
        pass

    def get_frame(self):
        return self.frame

    def close(self):
        self.closed = True


class _FakeRecorder:
    def __init__(self, path, *, fps):
        self.path = path
        self.fps = fps
        self.frame_count = 0
        self.closed = False

    def capture(self, viewer):
        viewer.get_frame()
        self.frame_count += 1

    def close(self):
        self.closed = True


def _fake_env(num_envs=4, bodies_per_world=3):
    body_world = np.repeat(np.arange(num_envs), bodies_per_world)
    # world w's bodies sit at x = 10*w so per-world AABBs are distinguishable
    body_q = np.zeros((body_world.size, 7), dtype=np.float32)
    body_q[:, 0] = 10.0 * body_world + np.tile(np.arange(bodies_per_world), num_envs) * 0.1
    model = SimpleNamespace(body_world=_Arr(body_world))
    cable = SimpleNamespace(model=model, state_0=SimpleNamespace(body_q=_Arr(body_q)))
    scene = SimpleNamespace(cable=cable, last_vbd_contacts=object(), cable_collision_pipeline=None)
    sim = SimpleNamespace(
        scene=scene,
        config=SimpleNamespace(runtime=SimpleNamespace(control_hz=60.0, env_spacing=(2.0, 2.0, 2.0))),
    )
    return SimpleNamespace(_sim=sim, num_envs=num_envs)


def _recorder(tmp_path, viewer, **kw):
    return HarvestVideoRecorder(tmp_path, open_viewer=lambda: viewer, recorder_factory=_FakeRecorder, **kw)


def test_should_record_schedule(tmp_path):
    rec = HarvestVideoRecorder(tmp_path, record_every=5)
    assert [i for i in range(12) if rec.should_record(i)] == [0, 5, 10]
    assert all(HarvestVideoRecorder(tmp_path, record_every=1).should_record(i) for i in range(4))
    with pytest.raises(ValueError):
        HarvestVideoRecorder(tmp_path, record_every=0)


def test_pick_env_reproducible_and_in_range():
    picks = [pick_env(7, ep, 8) for ep in range(50)]
    assert picks == [pick_env(7, ep, 8) for ep in range(50)]
    assert all(0 <= p < 8 for p in picks)
    assert len(set(picks)) > 1


def test_world_points_returns_only_that_world():
    env = _fake_env()
    pts = world_points(env, 2)
    assert pts.shape == (3, 3)
    assert np.all(pts[:, 0] >= 20.0) and np.all(pts[:, 0] < 21.0)


def test_fit_camera_centers_on_points_and_zooms_with_extent():
    small = np.array([[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]])
    large = np.array([[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]])
    (ps, _, _), (pl, _, _) = fit_camera(small), fit_camera(large)
    d_small = np.linalg.norm(np.array(ps) - small.mean(axis=0))
    d_large = np.linalg.norm(np.array(pl) - large.mean(axis=0))
    assert d_large > d_small
    # min_extent keeps tiny objects from collapsing the camera onto them
    assert d_small > 0.3


def test_episode_restricts_viewer_to_one_world_and_records(tmp_path):
    viewer, env = _FakeViewer(), _fake_env()
    rec = _recorder(tmp_path, viewer, record_every=5, seed=3)
    env_idx = rec.start_episode(env, 0)
    assert viewer.visible == [[env_idx]]
    assert viewer.offsets == [(2.0, 2.0, 2.0)]
    pos, _, _ = viewer.cameras[0]
    assert abs(pos[0] - 10.0 * env_idx) < 6.0  # framed on the picked world, not on the batch
    assert rec.recording
    for k in range(4):
        rec.capture(env, k / 60.0)
    path = rec.end_episode()
    assert path is not None and f"env{env_idx}" in path.name and path.name.startswith("episode_00000_")
    assert viewer.frames_begun == 4
    assert not rec.recording


def test_viewer_and_model_opened_once_across_episodes(tmp_path):
    opened = []
    viewer, env = _FakeViewer(), _fake_env()

    def open_viewer():
        opened.append(1)
        return viewer

    rec = HarvestVideoRecorder(tmp_path, open_viewer=open_viewer, recorder_factory=_FakeRecorder)
    for ep in (0, 5):
        rec.start_episode(env, ep)
        rec.capture(env, 0.0)
        rec.end_episode()
    assert len(opened) == 1 and len(viewer.models) == 1 and len(viewer.visible) == 2


def test_no_frames_means_no_path(tmp_path):
    rec = _recorder(tmp_path, _FakeViewer())
    rec.start_episode(_fake_env(), 0)
    assert rec.end_episode() is None


def test_start_while_recording_is_an_error_and_close_is_idempotent(tmp_path):
    viewer, env = _FakeViewer(), _fake_env()
    rec = _recorder(tmp_path, viewer)
    rec.start_episode(env, 0)
    with pytest.raises(RuntimeError):
        rec.start_episode(env, 1)
    rec.close()
    rec.close()
    assert viewer.closed


def test_real_gl_video_recorder_writes_mp4(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    viewer, env = _FakeViewer(), _fake_env()
    rec = HarvestVideoRecorder(tmp_path, open_viewer=lambda: viewer)
    rec.start_episode(env, 0)
    for k in range(6):
        rec.capture(env, k / 60.0)
    path = rec.end_episode()
    assert path is not None and path.exists() and path.stat().st_size > 0


def test_action_scale_shrinks_only_pose_delta_dims():
    from gymnasium import spaces

    from apple_pick_gym.batched_examples.example_batched_vic_harvest_random_actions import (
        _sample_random_actions,
    )

    b = HarvestActionBounds()
    lin, ang = [b.linear_delta_m] * 3, [b.angular_delta_rad] * 3
    low = np.array(
        [-x for x in lin] + [-x for x in ang] + [b.k_lin_min] * 3 + [b.k_ang_min] * 3 + [b.zeta_min],
        dtype=np.float32,
    )
    high = np.array(
        lin + ang + [b.k_lin_max] * 3 + [b.k_ang_max] * 3 + [b.zeta_max], dtype=np.float32
    )
    env = SimpleNamespace(
        num_envs=64, device="cpu", action_space=spaces.Box(low=low, high=high, dtype=np.float32)
    )
    gen_a, gen_b = torch.Generator(), torch.Generator()
    gen_a.manual_seed(0)
    gen_b.manual_seed(0)
    full = _sample_random_actions(env, gen_a, 1.0)
    small = _sample_random_actions(env, gen_b, 0.25)
    torch.testing.assert_close(small[:, :6], full[:, :6] * 0.25)
    torch.testing.assert_close(small[:, 6:], full[:, 6:])
    assert small[:, :3].abs().max() <= 0.25 * b.linear_delta_m + 1e-6
    assert small[:, 3:6].abs().max() <= 0.25 * b.angular_delta_rad + 1e-6


def test_cpu_viewer_device_is_rejected_instead_of_segfaulting(tmp_path):
    viewer = _FakeViewer()
    viewer.device = SimpleNamespace(is_cuda=False)
    rec = _recorder(tmp_path, viewer)
    with pytest.raises(RuntimeError, match="CUDA"):
        rec.start_episode(_fake_env(), 0)
