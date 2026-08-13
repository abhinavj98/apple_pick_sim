"""Stream Newton GL viewer frames to an MP4 via imageio-ffmpeg."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _frame_to_numpy(frame: Any) -> np.ndarray:
    """Convert Warp/array-like RGB frame to contiguous uint8 HxWx3."""
    if hasattr(frame, "numpy") and callable(frame.numpy):
        arr = frame.numpy()
    else:
        arr = np.asarray(frame)
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected RGB frame shape (H, W, 3), got {arr.shape}")
    return arr


class GlVideoRecorder:
    """Capture ``viewer.get_frame()`` RGB buffers into a single MP4.

    FPS may be set at construction or deferred via :meth:`set_fps` before the
    first :meth:`capture` (typical when ``control_hz`` is known only after env
    construction).
    """

    def __init__(self, path: Path | str, *, fps: float | None = None) -> None:
        self.path = Path(path)
        self._fps: float | None = float(fps) if fps is not None else None
        self._writer: Any | None = None
        self._frame_count = 0
        self._closed = False

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def set_fps(self, fps: float) -> None:
        """Set output FPS before the writer is opened (first capture)."""
        if self._writer is not None:
            raise RuntimeError("cannot change fps after the writer has opened")
        rate = float(fps)
        if rate <= 0.0:
            raise ValueError(f"fps must be > 0, got {rate}")
        self._fps = rate

    def capture(self, viewer: object) -> None:
        """Append one RGB frame from ``viewer.get_frame()``."""
        if self._closed:
            raise RuntimeError("GlVideoRecorder is closed")
        if not hasattr(viewer, "get_frame"):
            raise TypeError(
                "viewer has no get_frame(); use --viewer gl "
                "(optionally --headless) for video recording"
            )
        frame = _frame_to_numpy(viewer.get_frame())
        self._ensure_writer(frame.shape[1], frame.shape[0])
        assert self._writer is not None
        self._writer.send(frame)
        self._frame_count += 1

    def close(self) -> None:
        """Finalize the MP4. Idempotent."""
        if self._closed:
            return
        self._closed = True
        writer = self._writer
        self._writer = None
        if writer is not None:
            writer.close()

    def __enter__(self) -> GlVideoRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _ensure_writer(self, width: int, height: int) -> None:
        if self._writer is not None:
            return
        if self._fps is None:
            raise RuntimeError("fps is unset; call set_fps() before capture()")
        try:
            import imageio_ffmpeg as ffmpeg
        except ImportError as exc:
            raise ImportError(
                "imageio-ffmpeg is required for --record-video; "
                "install with: uv sync --extra gym"
            ) from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        writer = ffmpeg.write_frames(
            str(self.path),
            size=(int(width), int(height)),
            fps=float(self._fps),
            codec="libx264",
            macro_block_size=8,
            quality=5,
        )
        writer.send(None)  # initialize
        self._writer = writer
