"""Tests for robot_replay/gl_video_recorder.py (no real GL window)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from robot_replay.gl_video_recorder import GlVideoRecorder


class _FakeViewer:
    def __init__(self, frame: np.ndarray):
        self._frame = frame
        self.get_frame_calls = 0

    def get_frame(self):
        self.get_frame_calls += 1
        return self._frame


def test_capture_writes_mp4(tmp_path: Path):
    out = tmp_path / "out.mp4"
    frame = np.zeros((16, 24, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)
    viewer = _FakeViewer(frame)
    rec = GlVideoRecorder(out, fps=15.0)
    rec.capture(viewer)
    rec.capture(viewer)
    rec.close()
    assert rec.frame_count == 2
    assert viewer.get_frame_calls == 2
    assert out.is_file() and out.stat().st_size > 0


def test_capture_requires_get_frame(tmp_path: Path):
    rec = GlVideoRecorder(tmp_path / "x.mp4", fps=15.0)
    with pytest.raises(TypeError, match="get_frame"):
        rec.capture(SimpleNamespace())


def test_close_idempotent(tmp_path: Path):
    out = tmp_path / "out.mp4"
    rec = GlVideoRecorder(out, fps=15.0)
    rec.capture(_FakeViewer(np.zeros((8, 8, 3), dtype=np.uint8)))
    rec.close()
    rec.close()
    assert rec.frame_count == 1
    assert out.is_file()


def test_fps_set_on_first_capture(tmp_path: Path):
    out = tmp_path / "out.mp4"
    rec = GlVideoRecorder(out)  # fps deferred
    rec.set_fps(12.0)
    rec.capture(_FakeViewer(np.zeros((8, 8, 3), dtype=np.uint8)))
    rec.close()
    assert rec.fps == pytest.approx(12.0)
    assert rec.frame_count == 1
