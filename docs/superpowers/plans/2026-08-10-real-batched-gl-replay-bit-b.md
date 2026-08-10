# Real batched GL smoke replay (phase B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `robot_replay/example_replay_real_batched_smoke.py` so `--viewer gl` renders FR3+VIC open-loop trajectory frames after off-screen settle (phase B of the GL replay design).

**Architecture:** Keep C6 clamps and `replay_batched_sysid_structure`. Add a testable `make_smoke_on_step` that mirrors MMD-grid’s minimal render loop, compose it with `--max-frames` / `viewer.is_running`, and wire `newton.examples.init` for `--viewer {gl,null}`.

**Tech Stack:** Python, Newton ViewerGL, `replay_batched_sysid_structure`, pytest, uv.

## Global Constraints

- Trajectory-only render (settle stays off-screen).
- Do not change `replay_batched_sysid_structure` internals.
- Phase A (MMD-grid real clamps) is out of scope.
- Headless physics gate (`test_real_exported_s02_replay_moves_tcp`) must keep passing.
- Prefer `uv run` for all commands.

## File map

| File | Role |
| ---- | ---- |
| `robot_replay/example_replay_real_batched_smoke.py` | CLI + C6 + viewer `on_step` |
| `apple_pick_gym/tests/test_real_batched_replay_smoke_cli.py` | Parser + mock-viewer `on_step` unit tests |
| `robot_replay/README.md` | GL command |
| `docs/superpowers/specs/2026-08-10-real-batched-gl-replay-design.md` | Mark phase B implemented when done |

---

### Task 1: Failing CLI / on_step unit tests

**Files:**
- Create: `apple_pick_gym/tests/test_real_batched_replay_smoke_cli.py`
- Modify (later): `robot_replay/example_replay_real_batched_smoke.py`

**Interfaces:**
- Produces (expected after Task 2):
  - `_make_parser() -> argparse.ArgumentParser` with `--viewer` choices `gl`,`null`
  - `make_smoke_on_step(viewer, *, max_frames: int, control_hz_fallback: float = 30.0) -> Callable[..., bool]`

- [x] **Step 1: Write failing tests**

```python
"""CLI / viewer on_step unit tests for real batched smoke (no GPU window)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_SMOKE = Path(__file__).resolve().parents[2] / "robot_replay" / "example_replay_real_batched_smoke.py"


def _load_smoke():
    spec = importlib.util.spec_from_file_location("example_replay_real_batched_smoke", _SMOKE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parser_accepts_viewer_gl_and_null():
    mod = _load_smoke()
    p = mod._make_parser()
    gl = p.parse_args(["--dataset", "/tmp/ds", "--viewer", "gl"])
    assert gl.viewer == "gl"
    null = p.parse_args(["--dataset", "/tmp/ds", "--viewer", "null"])
    assert null.viewer == "null"


def test_make_smoke_on_step_renders_and_stops_at_max_frames():
    mod = _load_smoke()
    calls: list = []

    class Viewer:
        def set_model(self, model):
            calls.append(("set_model", model))

        def hide_loading_splash(self):
            calls.append("splash")

        def begin_frame(self, t):
            calls.append(("begin", t))

        def log_state(self, state):
            calls.append(("log", state))

        def end_frame(self):
            calls.append("end")

        def is_running(self):
            return True

    cable = SimpleNamespace(model="MODEL", state_0="STATE")
    scene = SimpleNamespace(cable=cable)
    sim = SimpleNamespace(scene=scene, config=SimpleNamespace(runtime=SimpleNamespace(control_hz=10.0)))
    env = SimpleNamespace(_sim=sim, num_envs=1)

    on_step = mod.make_smoke_on_step(Viewer(), max_frames=2, control_hz_fallback=30.0)
    assert on_step(frame_idx=0, env=env) is True
    assert on_step(frame_idx=1, env=env) is False
    assert ("set_model", "MODEL") in calls
    assert ("begin", 0.0) in calls
    assert ("log", "STATE") in calls
    assert "end" in calls
```

- [x] **Step 2: Run tests — expect FAIL**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_real_batched_replay_smoke_cli.py -q
```

Expected: FAIL (`choices` rejects `gl` and/or `make_smoke_on_step` missing).

- [x] **Step 3: Implement `make_smoke_on_step` + parser `gl|null` + wire `main`**

In `robot_replay/example_replay_real_batched_smoke.py`:

1. Change `--viewer` choices to `("gl", "null")`; update help.
2. Add:

```python
def make_smoke_on_step(
    viewer: object,
    *,
    max_frames: int,
    control_hz_fallback: float = 30.0,
):
    viewer_state: dict[str, object] = {"initialized": False}

    def on_step(*, frame_idx: int, env: object) -> bool:
        if hasattr(viewer, "is_running") and not viewer.is_running():
            return False

        sim = getattr(env, "_sim", None)
        scene = getattr(sim, "scene", None) if sim is not None else None
        if scene is not None and hasattr(viewer, "begin_frame"):
            if not viewer_state["initialized"]:
                if hasattr(viewer, "set_model") and hasattr(scene, "cable"):
                    viewer.set_model(scene.cable.model)
                if hasattr(viewer, "set_world_offsets") and getattr(env, "num_envs", 1) > 1:
                    viewer.set_world_offsets(tuple(sim.config.runtime.env_spacing))
                if hasattr(viewer, "hide_loading_splash"):
                    viewer.hide_loading_splash()
                viewer_state["initialized"] = True
            hz = float(
                getattr(getattr(getattr(sim, "config", None), "runtime", None), "control_hz", control_hz_fallback)
            )
            sim_time = float(frame_idx) / max(hz, 1e-9)
            viewer.begin_frame(sim_time)
            if hasattr(viewer, "log_state") and hasattr(scene, "cable"):
                viewer.log_state(scene.cable.state_0)
            viewer.end_frame()

        if max_frames <= 0:
            return True
        return int(frame_idx) + 1 < max_frames

    return on_step
```

3. Rewrite `main` to use `newton.examples.init` when run as `__main__`, or accept pre-parsed path for tests. Practical pattern matching other examples:

```python
def main(argv: list[str] | None = None, *, viewer: object | None = None) -> int:
    # If viewer is None and we're scripting: build parser, parse argv, use null stub
    # Prefer: _run(args, viewer) extracted from main body
```

Recommended structure (match MMD-grid lightly):

```python
def _run(args: argparse.Namespace, viewer: object) -> int:
    # existing dataset / clamp / replay body
    # always pass on_step = make_smoke_on_step(viewer, max_frames=...)
    # (null viewer without begin_frame still enforces max_frames)

def main(argv: list[str] | None = None) -> int:
    import os
    import newton.examples
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null"])
    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)
    try:
        return _run(args, viewer)
    finally:
        if hasattr(viewer, "close"):
            viewer.close()
```

Note: `newton.examples.init` may inject its own `--viewer` / `--num-frames`. Keep smoke’s `--max-frames` (do not rename). If `init` requires `--num-frames`, leave unused.

- [x] **Step 4: Re-run unit tests — expect PASS**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_real_batched_replay_smoke_cli.py -q
```

- [x] **Step 5: Docs + design status**

Update `robot_replay/README.md` smoke section with:

```bash
uv run python robot_replay/example_replay_real_batched_smoke.py \
  --dataset /tmp/real_batched_s02_d00 --viewer gl --max-frames 0
```

Set design spec status to “Phase B implemented; phase A follow-on”.

- [x] **Step 6: Regression physics smoke (if GPU / FR3 available)**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_real_batched_replay_smoke.py::test_real_exported_s02_replay_moves_tcp \
  -q -p no:launch_testing
```

Expected: PASS (or skip if missing assets).

---

## Spec coverage

| Spec item | Task |
| --------- | ---- |
| `--viewer gl\|null` via newton init | Task 1 |
| MMD-style on_step render | Task 1 |
| Keep C6 clamps / TCP exit | unchanged in `_run` |
| Unit test without GPU window | Task 1 |
| README GL command | Task 1 Step 5 |
| Phase A out of scope | honored |

## Self-review

- No placeholders.
- Phase A not included.
- `make_smoke_on_step` signature consistent across test and implementation.
