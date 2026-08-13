# GL viewer MP4 recording for real batched replay

| Field | Value |
| ----- | ----- |
| **Status** | Implemented |
| **Date** | 2026-08-12 |
| **Code** | `robot_replay/gl_video_recorder.py`, `robot_replay/example_replay_real_batched.py` |

## Purpose

Record the Newton GL viewer framebuffer during
`robot_replay/example_replay_real_batched.py` into a single MP4, so a replay can
be saved as a video (windowed or headless GL).

## Decisions

| Topic | Choice |
| ----- | ------ |
| Output | Single MP4 (`--record-video PATH`) |
| FPS | Sim `control_hz` (1 video second ≈ 1 control second) |
| Encode | `imageio-ffmpeg` (libx264) |
| Layout | Reusable helper `robot_replay/gl_video_recorder.py` + wire into the example |
| Rendering | Requires `ViewerGL` (`--viewer gl`), including `--headless` |

## Architecture

### `GlVideoRecorder` (`robot_replay/gl_video_recorder.py`)

- Construct with `path: Path` and `fps: float`.
- Lazy-open the `imageio_ffmpeg` writer on the first `capture()` once frame
  width/height are known from `viewer.get_frame()`.
- `capture(viewer)`:
  1. Require `hasattr(viewer, "get_frame")`; otherwise raise a clear error.
  2. `frame = viewer.get_frame()` → RGB `uint8` `(H, W, 3)`.
  3. Convert to contiguous CPU `numpy` and send to the writer.
- `close()`: finalize the writer; idempotent.
- Support context-manager use (`__enter__` / `__exit__` → `close()`).
- Expose `frame_count` for logging / tests.

### Example wire-up (`example_replay_real_batched.py`)

- CLI: `--record-video PATH` (optional `Path`).
- When set:
  - If viewer has no `get_frame`, `SystemExit` telling the user to pass
    `--viewer gl` (and optionally `--headless`).
  - Build `GlVideoRecorder(path, fps=control_hz)` once `control_hz` is known
    (same source as `sim_time` in `make_replay_on_step`), or pass a recorder
    that receives FPS on first step from env runtime `control_hz`.
  - Pass recorder into `make_replay_on_step`.
  - After each successful `end_frame()`, call `recorder.capture(viewer)`.
  - `close()` in `finally` after `_run` / `main` so early stops still finalize.
  - On success with ≥1 frame, print path + frame count to stderr.
  - If user requested recording and `frame_count == 0` after replay, fail
    (non-zero exit) with a clear message.

Preferred FPS wiring: resolve `control_hz` inside `on_step` on first capture
(same `runtime.control_hz` / fallback as today) and pass it into the recorder
if the writer is not yet open—avoids guessing FPS before the env exists.

### Dependency

Add `imageio-ffmpeg` to the optional **`gym`** extra in root `pyproject.toml`
(replay tooling already lives under that extra).

## Data flow

```text
replay on_step
  → viewer begin_frame / log_state / end_frame
  → GlVideoRecorder.capture(viewer)
       → get_frame() → numpy RGB → imageio_ffmpeg writer
  → (finally) recorder.close() → PATH.mp4
```

## Errors

| Condition | Behavior |
| --------- | -------- |
| `--record-video` + non-GL viewer | `SystemExit` with `--viewer gl` hint |
| Missing `imageio-ffmpeg` | Import error with install hint (`uv sync --extra gym`) |
| Writer / encode failure | Propagate; close best-effort |
| Zero frames after requested record | Non-zero exit |

## Testing

`robot_replay/tests/test_gl_video_recorder.py` (no real GL window):

- Fake viewer with `get_frame()` returning fixed RGB; assert N frames written,
  FPS used, `close()` idempotent.
- Capture without `get_frame` → clear error.
- Optional: `make_replay_on_step` with mock viewer + recorder asserts `capture`
  runs after `end_frame`.

## Non-goals

- PNG sequence / dual PNG+MP4 output
- RTX / Viser / `--viewer null` pixel capture
- Changing real-camera pose logic (`camera_to_base_4x4`)
- Recording settle / pre-replay phases (trajectory frames only, same as current
  GL render hook)

## Example usage

```bash
uv sync --extra gym --extra vic --extra dev

uv run python robot_replay/example_replay_real_batched.py \
  --dataset /tmp/real_batched_s02_d00 \
  --viewer gl --headless \
  --record-video /tmp/replay.mp4 \
  --max-frames 0
```
