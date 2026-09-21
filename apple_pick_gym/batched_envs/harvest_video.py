"""Occasional MP4 clips of one randomly picked env from a batched harvest env.

One headless ``ViewerGL`` is opened lazily and reused for the whole run (repeatedly creating
and destroying GL viewers is fragile in this repo). For each recorded episode a random env is
chosen (reproducibly from ``(seed, episode_idx)``), the viewer is restricted to that world with
``set_visible_worlds`` and the camera is fitted to that world's own bodies, so the clip shows a
single plant + gripper proxy + apple with no neighbouring worlds. Frames are encoded with the
existing :class:`robot_replay.gl_video_recorder.GlVideoRecorder`.

The cable-model viewer does not render the FR3 arm (same as the repo's other videos).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

_LOOK_PITCH_DEG = -20.0
_LOOK_YAW_DEG = 45.0
_PADDING = 1.2
_MIN_EXTENT_M = 0.25


def _gl_front(pitch_deg: float, yaw_deg: float) -> tuple[float, float, float]:
    """Newton GL look direction for Z-up (matches ``Camera.get_front``)."""
    pitch = max(min(float(pitch_deg), 89.0), -89.0)
    cp = math.cos(math.radians(pitch))
    f = (
        math.cos(math.radians(yaw_deg)) * cp,
        math.sin(math.radians(yaw_deg)) * cp,
        math.sin(math.radians(pitch)),
    )
    norm = math.sqrt(sum(c * c for c in f)) or 1.0
    return (f[0] / norm, f[1] / norm, f[2] / norm)


def fit_camera(
    points: np.ndarray,
    *,
    fov_deg: float = 45.0,
    pitch_deg: float = _LOOK_PITCH_DEG,
    yaw_deg: float = _LOOK_YAW_DEG,
    padding: float = _PADDING,
    min_extent: float = _MIN_EXTENT_M,
) -> tuple[tuple[float, float, float], float, float]:
    """Camera ``(pos, pitch, yaw)`` framing the AABB of ``points`` ``(M,3)``."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (lo + hi)
    extent = float(np.max(hi - lo))
    if not math.isfinite(extent) or extent < min_extent:
        extent = float(min_extent)
    fov = min(90.0, max(15.0, float(fov_deg)))
    distance = extent / (2.0 * math.tan(math.radians(fov) / 2.0)) * float(padding)
    front = _gl_front(pitch_deg, yaw_deg)
    pos = tuple(float(center[k] - front[k] * distance) for k in range(3))
    return pos, float(pitch_deg), float(yaw_deg)


def world_points(env: Any, env_idx: int) -> np.ndarray | None:
    """Body positions ``(M,3)`` of world ``env_idx`` from the cable state, or ``None``."""
    cable = env._sim.scene.cable
    body_q = cable.state_0.body_q.numpy().reshape(-1, 7)
    body_world = cable.model.body_world
    if body_world is None:
        return None
    mask = np.asarray(body_world.numpy()) == int(env_idx)
    if not mask.any():
        return None
    return np.asarray(body_q[mask, :3], dtype=np.float64)


def pick_env(seed: int, episode_idx: int, num_envs: int) -> int:
    return int(np.random.default_rng((int(seed), int(episode_idx))).integers(int(num_envs)))


def _open_headless_gl_viewer() -> Any:
    import newton
    import pyglet

    pyglet.app.event_loop.has_exit = False
    return newton.viewer.ViewerGL(headless=True)


class HarvestVideoRecorder:
    """Record one full episode of a randomly picked env every ``record_every`` episodes."""

    def __init__(
        self,
        video_dir: Path | str,
        *,
        record_every: int = 5,
        seed: int = 0,
        open_viewer: Callable[[], Any] = _open_headless_gl_viewer,
        recorder_factory: Callable[..., Any] | None = None,
    ) -> None:
        if int(record_every) < 1:
            raise ValueError(f"record_every must be >= 1, got {record_every}")
        self.video_dir = Path(video_dir)
        self.record_every = int(record_every)
        self.seed = int(seed)
        self._open_viewer = open_viewer
        self._recorder_factory = recorder_factory
        self._viewer: Any | None = None
        self._model: Any | None = None
        self._recorder: Any | None = None
        self.env_idx: int | None = None
        self.fps: float | None = None

    def should_record(self, episode_idx: int) -> bool:
        return int(episode_idx) == 0 or int(episode_idx) % self.record_every == 0

    @property
    def recording(self) -> bool:
        return self._recorder is not None

    def start_episode(self, env: Any, episode_idx: int) -> int:
        """Pick an env, restrict the viewer to it, frame the camera, open the MP4."""
        if self._recorder is not None:
            raise RuntimeError("start_episode called while an episode is still recording")
        sim = env._sim
        cable_model = sim.scene.cable.model
        if self._viewer is None:
            self._viewer = self._open_viewer()
            if not hasattr(self._viewer, "get_frame"):
                raise RuntimeError("video recording needs a GL viewer with get_frame()")
            device = getattr(self._viewer, "device", None)
            if device is not None and not getattr(device, "is_cuda", True):
                # ViewerGL.get_frame reads the GL framebuffer through CUDA-GL interop; on a
                # CPU warp device that launch segfaults instead of raising.
                raise RuntimeError(
                    f"video recording needs a CUDA warp device (viewer device is {device}); "
                    "run with --device cuda:0"
                )
        viewer = self._viewer
        if self._model is not cable_model:
            viewer.set_model(cable_model)
            if hasattr(viewer, "hide_loading_splash"):
                viewer.hide_loading_splash()
            self._model = cable_model

        env_idx = pick_env(self.seed, episode_idx, env.num_envs)
        viewer.set_visible_worlds([env_idx])
        viewer.set_world_offsets(tuple(sim.config.runtime.env_spacing))
        points = world_points(env, env_idx)
        if points is not None and hasattr(viewer, "set_camera"):
            fov = float(getattr(getattr(viewer, "camera", None), "fov", 45.0) or 45.0)
            pos, pitch, yaw = fit_camera(points, fov_deg=fov)
            viewer.set_camera(pos, pitch, yaw)

        self.fps = float(sim.config.runtime.control_hz)
        if self._recorder_factory is None:
            from robot_replay.gl_video_recorder import GlVideoRecorder as factory
        else:
            factory = self._recorder_factory
        path = self.video_dir / f"episode_{int(episode_idx):05d}_env{env_idx}.mp4"
        self._recorder = factory(path, fps=self.fps)
        self.env_idx = env_idx
        return env_idx

    def capture(self, env: Any, sim_time: float) -> None:
        if self._recorder is None or self._viewer is None:
            return
        scene = env._sim.scene
        if scene.last_vbd_contacts is not None:
            contacts = scene.last_vbd_contacts
        else:
            contacts = scene.cable.model.collide(
                scene.cable.state_0, collision_pipeline=scene.cable_collision_pipeline
            )
        v = self._viewer
        v.begin_frame(float(sim_time))
        v.log_state(scene.cable.state_0)
        v.log_contacts(contacts, scene.cable.state_0)
        v.end_frame()
        self._recorder.capture(v)

    def end_episode(self) -> Path | None:
        """Finalize the MP4; return its path if any frame was written."""
        rec, self._recorder = self._recorder, None
        if rec is None:
            return None
        rec.close()
        return rec.path if rec.frame_count > 0 else None

    def close(self) -> None:
        self.end_episode()
        viewer, self._viewer = self._viewer, None
        if viewer is not None and hasattr(viewer, "close"):
            viewer.close()
