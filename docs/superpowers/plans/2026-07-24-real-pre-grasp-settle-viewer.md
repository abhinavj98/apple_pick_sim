# Real Pre-Grasp Settle Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert real-parquet `pre_grasp_geometry` into `FruitingSystemParams` (Branch = `fruiting_base_pos` T-junction; spur/stem from consecutive chords), build a plant-only VBD scene, settle, and visualize under `robot_replay/`.

**Architecture:** Library `real_pre_grasp_params.py` owns metadata parse + rod mapping + params assembly. CLI `example_view_pre_grasp_settle.py` builds via `generate_coupled_cable_scene(..., fix_to_apple=False)`, settles like `example_digital_twin.py`, then runs the Newton viewer. Params-first so a later dataset converter can embed the same JSON blob.

**Tech Stack:** Python, NumPy, PyArrow, Newton/Warp, `FruitingSystemParams`, `uv`, pytest.

**Spec:** `docs/superpowers/specs/2026-07-24-real-pre-grasp-settle-viewer-design.md`

## Global Constraints

- Plant-only: no FR3, no weld (`GripperProxyConfig(fix_to_apple=False)`).
- `fruiting_base_pos` = measured **Branch** xyz (`franka_base_o`), start of spur / T-junction (not fixture default).
- Primary = horizontal through T via **fixture azimuth/elevation midpoints** (proxy: +X); L/r from `parts`.
- Spur end tracker → `spur.direction` = Branch→Spur; `stem.direction` = Spur end → **`apple_pos`** (no stem-end tracker); report error vs Spur→Apple chord end if present.
- **Density from `parts.*.density_kg_m3`**; E/ζ/stretch/`num_segments` from fixture midpoints.
- **Lengths and radii from `parts.*`**; always print catalog-vs-chord length errors (never fail on them). `--strict` = bend≈0 only.
- Materials from fixture midpoints.
- TDD: failing tests before implementation.
- Run tests with `uv run --env-file pytest.env python -m pytest …` from repo root.
- Prefer a feature worktree before editing production code (`.cursor/rules/worktree-feature-dev.mdc`).

---

## File map

| File | Responsibility |
|------|----------------|
| `apple_pick_sim/system_id/real_pre_grasp_params.py` | Parse parquet metadata; map pre-grasp → `FruitingSystemParams` + `fruiting_base_pos` |
| `apple_pick_sim/tests/test_real_pre_grasp_params.py` | Unit tests for parse/map/params |
| `robot_replay/example_view_pre_grasp_settle.py` | CLI: build → settle → view |
| `robot_replay/README.md` | Document command |
| `docs/real-sysid-pre-post-grasp-fixes.md` | Link to shipped viewer path (short note) |

---

### Task 1: Pre-grasp metadata helpers + failing tests

**Files:**
- Create: `apple_pick_sim/system_id/real_pre_grasp_params.py`
- Create: `apple_pick_sim/tests/test_real_pre_grasp_params.py`

**Interfaces:**
- Produces (stubs ok until Task 2 fills them):
  - `load_dataset_metadata(path: Path) -> dict[str, Any]`
  - `coerce_xyz(value: Any, *, field: str) -> np.ndarray`  # shape (3,), handles list or numpy-str
  - `PreGraspMappedGeometry` dataclass with fields:
    - `fruiting_base_pos: tuple[float, float, float]`
    - `primary_direction: tuple[float, float, float]`
    - `spur_direction: tuple[float, float, float]`
    - `stem_direction: tuple[float, float, float]`
    - `rod_geometry: dict[str, dict[str, float]]`  # primary/spur/stem → length_m, radius_m
    - `apple_radius_m: float | None`
    - `woody_bending_angles: np.ndarray`  # shape (3,)

- [ ] **Step 1: Write failing tests for coerce + Branch T-junction base + spur/stem dirs**

```python
# apple_pick_sim/tests/test_real_pre_grasp_params.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from apple_pick_sim.system_id.real_pre_grasp_params import (
    PreGraspMappedGeometry,
    coerce_xyz,
    load_dataset_metadata,
    map_pre_grasp_geometry,
)


def test_coerce_xyz_list():
    np.testing.assert_allclose(coerce_xyz([1.0, 2.0, 3.0], field="p"), [1.0, 2.0, 3.0])


def test_coerce_xyz_numpy_string():
    out = coerce_xyz("[-0.00889757  0.94594489  0.40465398]", field="apple_pos")
    np.testing.assert_allclose(out, [-0.00889757, 0.94594489, 0.40465398], rtol=1e-6)


def _synthetic_pre_grasp_meta() -> dict:
    # Branch = T-junction (fruiting_base); Spur = spur end; Apple = fruit
    # Non-collinear hang so spur/stem directions differ.
    branch = [0.0, 0.0, 0.0]
    spur = [0.02, 0.0, -0.10]
    apple = [0.05, 0.0, -0.13]
    return {
        "topology": {
            "junction_names": ["Branch", "Spur", "Apple"],
            "start_nodes": ["Branch", "Branch", "Spur"],
            "end_nodes": ["Spur", "Apple", "Apple"],
            "shared_endpoints": True,
            "n_woody_parts": 3,
        },
        "pre_grasp_geometry": {
            "structure_name": "default_template",
            "parts": {
                "primary": {
                    "length_m": 0.2,
                    "radius_m": 0.0125,
                    "density_kg_m3": 660,
                    "shape": "cylinder",
                },
                "spur": {
                    "length_m": 0.1,
                    "radius_m": 0.0025,
                    "density_kg_m3": 1200,
                    "shape": "cylinder",
                },
                "stem": {
                    "length_m": 0.025,
                    "radius_m": 0.0005,
                    "density_kg_m3": 1000,
                    "shape": "cylinder",
                },
                "apple": {
                    "length_m": 0.08,
                    "radius_m": 0.04,
                    "density_kg_m3": 650,
                    "shape": "sphere",
                },
            },
            "snapshot": {
                "woody_part_start_pos": branch + branch + spur,
                "woody_part_end_pos": spur + apple + apple,
                "woody_bending_angles": [0.0, 0.0, 0.0],
                "apple_pos": apple,
            },
        },
    }


def test_map_pre_grasp_branch_is_fruiting_base_pos():
    mapped = map_pre_grasp_geometry(_synthetic_pre_grasp_meta())
    assert isinstance(mapped, PreGraspMappedGeometry)
    np.testing.assert_allclose(mapped.fruiting_base_pos, (0.0, 0.0, 0.0), atol=1e-9)
    # spur: Branch → Spur; stem: Spur → Apple (distinct)
    spur_u = np.array([0.02, 0.0, -0.10], dtype=np.float64)
    spur_u /= np.linalg.norm(spur_u)
    stem_u = np.array([0.03, 0.0, -0.03], dtype=np.float64)
    stem_u /= np.linalg.norm(stem_u)
    np.testing.assert_allclose(mapped.spur_direction, spur_u, atol=1e-6)
    np.testing.assert_allclose(mapped.stem_direction, stem_u, atol=1e-6)
    assert mapped.spur_direction != mapped.stem_direction
    assert mapped.rod_geometry["primary"]["length_m"] == pytest.approx(0.2)
    assert mapped.rod_geometry["spur"]["radius_m"] == pytest.approx(0.0025)
    assert mapped.apple_radius_m == pytest.approx(0.04)


def test_map_pre_grasp_strict_rejects_nonzero_bend():
    meta = _synthetic_pre_grasp_meta()
    meta["pre_grasp_geometry"]["snapshot"]["woody_bending_angles"] = [0.2, 0.0, 0.0]
    with pytest.raises(ValueError, match="bend"):
        map_pre_grasp_geometry(meta, strict=True)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_real_pre_grasp_params.py -q
```

Expected: FAIL (import / not defined).

- [ ] **Step 3: Implement coerce, load_dataset_metadata, map_pre_grasp_geometry**

Create `apple_pick_sim/system_id/real_pre_grasp_params.py`:

```python
"""Map real-episode pre_grasp_geometry into sim placement + rod geometry."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

_ZERO_EPS = 1e-12
_BEND_EPS = 1e-3


def load_dataset_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    schema = pq.read_schema(path)
    raw = schema.metadata or {}
    blob = raw.get(b"dataset_metadata")
    if blob is None:
        raise ValueError(f"{path}: missing schema metadata key dataset_metadata")
    text = blob.decode("utf-8") if isinstance(blob, (bytes, bytearray)) else str(blob)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("dataset_metadata must be a JSON object")
    return data


def coerce_xyz(value: Any, *, field: str) -> np.ndarray:
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        arr = np.fromstring(s, sep=" ", dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{field} must have length 3, got {arr.size}")
    return arr


def _unit(vec: np.ndarray, *, field: str) -> tuple[float, float, float]:
    n = float(np.linalg.norm(vec))
    if n < _ZERO_EPS:
        raise ValueError(f"{field}: zero-length vector")
    u = vec / n
    return (float(u[0]), float(u[1]), float(u[2]))


@dataclass(frozen=True)
class PreGraspMappedGeometry:
    fruiting_base_pos: tuple[float, float, float]
    primary_direction: tuple[float, float, float]
    spur_direction: tuple[float, float, float]
    stem_direction: tuple[float, float, float]
    rod_geometry: dict[str, dict[str, float]]
    apple_radius_m: float | None
    woody_bending_angles: np.ndarray


def map_pre_grasp_geometry(
    meta: dict[str, Any],
    *,
    strict: bool = False,
) -> PreGraspMappedGeometry:
    topo = meta.get("topology") or {}
    names = list(topo.get("junction_names") or [])
    if names != ["Branch", "Spur", "Apple"] or not topo.get("shared_endpoints"):
        raise ValueError(
            "unsupported topology: expected shared_endpoints Branch/Spur/Apple, "
            f"got junction_names={names!r} shared_endpoints={topo.get('shared_endpoints')!r}"
        )
    pre = meta.get("pre_grasp_geometry")
    if not isinstance(pre, dict):
        raise ValueError("missing pre_grasp_geometry")
    snap = pre.get("snapshot")
    parts = pre.get("parts")
    if not isinstance(snap, dict) or not isinstance(parts, dict):
        raise ValueError("pre_grasp_geometry requires snapshot and parts")

    start9 = np.asarray(snap["woody_part_start_pos"], dtype=np.float64).reshape(9)
    end9 = np.asarray(snap["woody_part_end_pos"], dtype=np.float64).reshape(9)
    # part0 Branch→Spur (T → spur end), part2 Spur→Apple (spur end → fruit)
    branch = start9[0:3]  # fruiting_base / spur-primary T-junction
    spur_end = end9[0:3]
    apple = end9[6:9]
    if float(np.linalg.norm(spur_end - start9[6:9])) > 1e-4:
        raise ValueError("Spur endpoint mismatch between part0 end and part2 start")

    bend = np.asarray(snap.get("woody_bending_angles", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    if float(np.max(np.abs(bend))) > _BEND_EPS:
        msg = f"pre-grasp woody_bending_angles not ~0: {bend.tolist()}"
        if strict:
            raise ValueError(msg)

    # Primary axis: proxy convention (+X); not Branch→Spur (that is the spur).
    primary_dir = (1.0, 0.0, 0.0)
    spur_dir = _unit(spur_end - branch, field="spur_direction")
    stem_dir = _unit(apple - spur_end, field="stem_direction")

    def _lr(name: str) -> dict[str, float]:
        block = parts[name]
        return {
            "length_m": float(block["length_m"]),
            "radius_m": float(block["radius_m"]),
        }

    apple_r = None
    if "apple" in parts and parts["apple"] is not None:
        apple_r = float(parts["apple"]["radius_m"])

    return PreGraspMappedGeometry(
        fruiting_base_pos=(float(branch[0]), float(branch[1]), float(branch[2])),
        primary_direction=primary_dir,
        spur_direction=spur_dir,
        stem_direction=stem_dir,
        rod_geometry={
            "primary": _lr("primary"),
            "spur": _lr("spur"),
            "stem": _lr("stem"),
        },
        apple_radius_m=apple_r,
        woody_bending_angles=bend,
    )
```

Note: use `np.testing` only in tests; in library use plain checks for Spur consistency:

```python
if float(np.linalg.norm(spur - start9[6:9])) > 1e-4:
    raise ValueError("Spur endpoint mismatch between part0 end and part2 start")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_real_pre_grasp_params.py::test_coerce_xyz_list \
  apple_pick_sim/tests/test_real_pre_grasp_params.py::test_coerce_xyz_numpy_string \
  apple_pick_sim/tests/test_real_pre_grasp_params.py::test_map_pre_grasp_branch_is_fruiting_base_pos \
  apple_pick_sim/tests/test_real_pre_grasp_params.py::test_map_pre_grasp_strict_rejects_nonzero_bend \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apple_pick_sim/system_id/real_pre_grasp_params.py \
  apple_pick_sim/tests/test_real_pre_grasp_params.py
git commit -m "Add pre-grasp geometry mapping helpers for real parquet."
```

---

### Task 2: Build FruitingSystemParams from mapped pre-grasp

**Files:**
- Modify: `apple_pick_sim/system_id/real_pre_grasp_params.py`
- Modify: `apple_pick_sim/tests/test_real_pre_grasp_params.py`
- Optionally reuse: `apple_pick_sim/system_id/real_to_batched_sysid.py` (`build_fruiting_params_from_real`, `range_midpoint`)

**Interfaces:**
- Consumes: `PreGraspMappedGeometry`, `build_fruiting_params_from_real`
- Produces:
  - `fruiting_params_from_pre_grasp_parquet(path, *, fixture_path, strict=False) -> tuple[FruitingSystemParams, tuple[float,float,float], dict]`
    (third value = diagnostics for stdout / `--dump-params`)

- [ ] **Step 1: Write failing test for params assembly**

```python
from apple_pick_sim.fruiting_system.params import fruiting_params_to_dict
from apple_pick_sim.system_id.real_pre_grasp_params import (
    fruiting_params_from_pre_grasp_meta,
)

VARIANCE = Path("apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json")


def test_fruiting_params_from_pre_grasp_meta():
    params, base = fruiting_params_from_pre_grasp_meta(
        _synthetic_pre_grasp_meta(),
        fixture_path=VARIANCE,
    )
    assert base == (0.0, 0.0, 0.0)
    assert params.topology == "t_junction"
    assert params.secondary is None
    assert params.primary is not None and params.spur is not None and params.stem is not None
    np.testing.assert_allclose(params.primary.direction, (1.0, 0.0, 0.0), atol=1e-6)
    spur_u = np.array([0.02, 0.0, -0.10]); spur_u /= np.linalg.norm(spur_u)
    stem_u = np.array([0.03, 0.0, -0.03]); stem_u /= np.linalg.norm(stem_u)
    np.testing.assert_allclose(params.spur.direction, spur_u, atol=1e-6)
    np.testing.assert_allclose(params.stem.direction, stem_u, atol=1e-6)
    assert params.primary.length == pytest.approx(0.2)
    assert params.apple_radius == pytest.approx(0.04)
    blob = fruiting_params_to_dict(params)
    assert blob["schema"] == "fruiting_system_params_v2"
    assert "youngs_modulus_pa" in blob["primary"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_real_pre_grasp_params.py::test_fruiting_params_from_pre_grasp_meta -q
```

Expected: FAIL (function missing).

- [ ] **Step 3: Implement fruiting_params_from_pre_grasp_meta / parquet wrapper**

```python
from apple_pick_sim.fruiting_system.params import FruitingSystemParams
from apple_pick_sim.system_id.real_to_batched_sysid import build_fruiting_params_from_real


def fruiting_params_from_pre_grasp_meta(
    meta: dict[str, Any],
    *,
    fixture_path: str | Path,
    strict: bool = False,
) -> tuple[FruitingSystemParams, tuple[float, float, float]]:
    mapped = map_pre_grasp_geometry(meta, strict=strict)
    directions = {
        "primary": mapped.primary_direction,
        "spur": mapped.spur_direction,
        "stem": mapped.stem_direction,
    }
    params = build_fruiting_params_from_real(
        ranges_path=fixture_path,
        rod_geometry=mapped.rod_geometry,
        directions=directions,
        apple_radius_m=mapped.apple_radius_m,
    )
    return params, mapped.fruiting_base_pos


def fruiting_params_from_pre_grasp_parquet(
    path: str | Path,
    *,
    fixture_path: str | Path,
    strict: bool = False,
) -> tuple[FruitingSystemParams, tuple[float, float, float]]:
    meta = load_dataset_metadata(path)
    return fruiting_params_from_pre_grasp_meta(
        meta, fixture_path=fixture_path, strict=strict
    )
```

If `real_to_batched_sysid.py` is not yet on the branch, either land it first (already present in worktree as untracked) or inline the small `build_fruiting_params_from_real` body into `real_pre_grasp_params.py` for this slice — prefer importing the existing helper to avoid duplication.

- [ ] **Step 4: Optional smoke on s00-d00 metadata**

```python
@pytest.mark.parametrize("parquet", [Path("robot_replay/s00-d00.parquet")])
def test_s00_d00_pre_grasp_params_smoke(parquet: Path):
    if not parquet.is_file():
        pytest.skip("missing robot_replay/s00-d00.parquet")
    params, base = fruiting_params_from_pre_grasp_parquet(
        parquet, fixture_path=VARIANCE
    )
    assert params.primary is not None
    assert math.isfinite(base[0]) and math.isfinite(base[1]) and math.isfinite(base[2])
```

- [ ] **Step 5: Run tests**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_real_pre_grasp_params.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apple_pick_sim/system_id/real_pre_grasp_params.py \
  apple_pick_sim/system_id/real_to_batched_sysid.py \
  apple_pick_sim/tests/test_real_pre_grasp_params.py
git commit -m "Build FruitingSystemParams from real pre-grasp geometry."
```

---

### Task 3: Settle viewer CLI in `robot_replay/`

**Files:**
- Create: `robot_replay/example_view_pre_grasp_settle.py`
- Modify: `robot_replay/README.md`
- Modify: `docs/real-sysid-pre-post-grasp-fixes.md` (one short “shipped viewer” pointer)
- Modify: `docs/superpowers/specs/2026-07-24-real-pre-grasp-settle-viewer-design.md` status → Implemented when done

**Interfaces:**
- Consumes: `fruiting_params_from_pre_grasp_parquet`, `generate_coupled_cable_scene`, `load_ranges`, `GripperProxyConfig`, `resolve_sim_device`

- [ ] **Step 1: Implement CLI script**

Mirror `apple_pick_sim/examples/example_digital_twin.py` settle + viewer loop, but build from params:

```python
"""Build plant-only scene from real pre_grasp_geometry, settle, and visualize.

Run from repo root::

    uv run python robot_replay/example_view_pre_grasp_settle.py \\
      --parquet robot_replay/s00-d00.parquet \\
      --fixture apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json \\
      --settle-substeps 5000 \\
      --viewer gl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import newton
import newton.examples
import warp as wp

from apple_pick_sim.fruiting_system import (
    GripperProxyConfig,
    example_collision_pipeline,
    geometry_fingerprint_coupled,
    load_ranges,
)
from apple_pick_sim.fruiting_system.coupled import generate_coupled_cable_scene
from apple_pick_sim.fruiting_system.params import fruiting_params_to_dict, parse_fixture_args
from apple_pick_sim.sim_device import resolve_sim_device
from apple_pick_sim.system_id.real_pre_grasp_params import (
    fruiting_params_from_pre_grasp_parquet,
)


def _settle_cable_substeps(scene, *, substeps: int, dt: float) -> None:
    n = int(substeps)
    if n <= 0:
        return
    pipeline = example_collision_pipeline(scene.model)
    for _ in range(n):
        scene.state_0.clear_forces()
        contacts = scene.model.collide(scene.state_0, collision_pipeline=pipeline)
        scene.solver.step(scene.state_0, scene.state_1, scene.control, contacts, dt)
        scene.state_0, scene.state_1 = scene.state_1, scene.state_0


def _make_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(
            "apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json"
        ),
    )
    parser.add_argument("--settle-substeps", type=int, default=5000,
                        help="VBD substeps for visible settle (rendered in viewer).")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dump-params", type=Path, default=None)
    return parser


class ExampleViewPreGraspSettle:
    def __init__(self, viewer, args: argparse.Namespace):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 30
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        device = resolve_sim_device(getattr(args, "device", None))
        params, base_pos, diagnostics = fruiting_params_from_pre_grasp_parquet(
            args.parquet,
            fixture_path=args.fixture,
            strict=bool(args.strict),
        )
        if args.dump_params is not None:
            args.dump_params.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "fruiting_base_pos": list(base_pos),
                "fruiting_system_params": fruiting_params_to_dict(params),
                "diagnostics": diagnostics,
            }
            args.dump_params.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {args.dump_params}")

        ranges = load_ranges(args.fixture)
        robot_base = parse_fixture_args(ranges).robot_base_pos
        self._scene = generate_coupled_cable_scene(
            ranges,
            seed=0,
            params=params,
            base_pos=base_pos,
            device=device,
            gripper_proxy=GripperProxyConfig(fix_to_apple=False),
            robot_base_pos=robot_base,
        )
        print(f"fruiting_base_pos (spur-primary T): {base_pos}")
        print(f"Geometry fingerprint: {geometry_fingerprint_coupled(self._scene)}")
        self._settle_remaining = max(0, int(args.settle_substeps))
        # Visible settle: consume settle_remaining inside step() while rendering.

        self.model = self._scene.model
        self.state_0 = self._scene.state_0
        self.state_1 = self._scene.state_1
        self.control = self._scene.control
        self.solver = self._scene.solver
        self.collision_pipeline = example_collision_pipeline(self.model)
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(0.5, -0.8, 1.6), pitch=-20.0, yaw=45.0)

    def capture_video(self, duration_seconds: float = 0.0) -> None:
        return None

    def step(self) -> None:
        # During settle budget, run more substeps per frame so 5000 finishes
        # while still showing motion; afterward one frame of normal substeps.
        if self._settle_remaining > 0:
            n = min(self.sim_substeps * 10, self._settle_remaining)
            self._settle_remaining -= n
        else:
            n = self.sim_substeps
        for _ in range(n):
            self._scene.state_0.clear_forces()
            contacts = self.model.collide(
                self._scene.state_0, collision_pipeline=self.collision_pipeline
            )
            self.solver.step(
                self._scene.state_0, self._scene.state_1, self.control, contacts, self.sim_dt
            )
            self._scene.state_0, self._scene.state_1 = self._scene.state_1, self._scene.state_0
            self.state_0 = self._scene.state_0
            self.state_1 = self._scene.state_1
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


def main() -> None:
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            sys.argv.extend(["--viewer", "null", "--num-frames", "30"])
            print("No display: using --viewer null --num-frames 30")
    viewer, args = newton.examples.init(parser=_make_parser())
    example = ExampleViewPreGraspSettle(viewer, args)
    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()
    while viewer.is_running():
        example.step()
        example.render()


if __name__ == "__main__":
    main()
```

Align `step`/`render` with whatever `newton.examples.run` pattern `example_digital_twin.py` uses in this repo (copy that main loop exactly if it differs).

- [ ] **Step 2: Update `robot_replay/README.md`** with the command and note that `--dump-params` writes the sim-native params blob for later dataset embedding.

- [ ] **Step 3: Headless smoke**

```bash
uv run python robot_replay/example_view_pre_grasp_settle.py \
  --parquet robot_replay/s00-d00.parquet \
  --settle-substeps 100 \
  --viewer null --num-frames 5 \
  --dump-params /tmp/s00_d00_params.json
```

Expected: exits 0; JSON contains `fruiting_base_pos` and `fruiting_system_params.primary`.

- [ ] **Step 4: Commit**

```bash
git add robot_replay/example_view_pre_grasp_settle.py robot_replay/README.md \
  docs/real-sysid-pre-post-grasp-fixes.md \
  docs/superpowers/specs/2026-07-24-real-pre-grasp-settle-viewer-design.md
git commit -m "Add plant-only pre-grasp settle viewer for real parquet."
```

---

## Spec coverage check

| Spec requirement | Task |
| ---------------- | ---- |
| Params-first from pre_grasp | Task 2 |
| Branch = fruiting_base_pos T-junction; spur/stem consecutive chords | Task 1 |
| Primary/hang directions + parts L/r | Task 1–2 |
| Fixture materials | Task 2 (`build_fruiting_params_from_real`) |
| Plant-only settle + viewer in `robot_replay/` | Task 3 |
| `--dump-params` | Task 3 |
| `--strict` bend check | Task 1–3 |
| Unit tests without GPU viewer | Task 1–2 |
| No robot/weld/trajectory | Global constraints |

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-24-real-pre-grasp-settle-viewer.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute tasks in this session with checkpoints  

Also confirm whether to create a sibling worktree (`feature/real-pre-grasp-settle-viewer`) before coding.

Which approach (and worktree yes/no)?
