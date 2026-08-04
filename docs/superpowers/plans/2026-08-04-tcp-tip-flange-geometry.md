# TCP Tip / Flange Geometry Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make FR3 USD cylinder tip coincide with `/ee/tcp`, flange-side face flush with the flange, match Ø100×140 mm tip-out contract, and update docs/README with warnings so gym/digital-twin look-at is not mistaken for true TCP SE(3).

**Architecture:** Add a small pure helper that evaluates ee-local cylinder tip/flange/TCP from authored USD numbers (no sim step). Drive TDD with that helper against `testfr3_resolved.usda` and `testfr3.usda`. Edit only the `ee` / `Cylinder` / `tcp` / `tcp_joint` transforms needed for tip=TCP and flange flush. Then land doc/README/supersession edits from the approved spec.

**Tech Stack:** Python, pytest, `uv`, optional `pxr` (usd-core) for import smoke, USDA text edits under `assets/`, existing `apple_pick_sim.robot.fr3_robot.paths` constants.

**Spec:** [docs/superpowers/specs/2026-08-04-tcp-tip-flange-geometry-design.md](../specs/2026-08-04-tcp-tip-flange-geometry-design.md)

## Global Constraints

- TCP / `/ee/tcp` = center of **distal tip face**
- **World tip-out** = away from link7. `fr3_joint8` ≈ RotX(180) ⇒ tip-out is **ee −Z** (do **not** author the mesh on ee `+Z` or it grows into the arm)
- Proximal cylinder face meets **link7 visual mesh** end (~6.2 mm past `ee` origin: joint `localPos0.z≈0.113`, mesh max-z≈0.107)
- Distal face flush with TCP; length **0.14 m**, radius **0.05 m** (`hh = 0.07`)
- Recorded / VBD proxy TCP frame: local **+Z** tip-out, bulk on −Z from tip
- Post-grasp replay already follows logged SE(3); do **not** change that path in this plan
- Gym / digital-twin true TCP SE(3) is **out of scope** — document warning only
- Prefer editing `assets/testfr3_resolved.usda` and `assets/testfr3.usda` transforms; do not regenerate unrelated FR3 links
- TDD: failing geometry tests before USD edits
- Run tests with `uv run --env-file pytest.env pytest …` from repo root
- Commits only if the user asks, or when an agent is explicitly executing a plan that includes commit steps the user approved

---

## File map

| File | Role |
|------|------|
| `apple_pick_sim/robot/fr3_robot/ee_cylinder_geometry.py` | Pure helpers: ee-scaled tip/flange/TCP from authored numbers; USDA attribute scrape for `/fr3/ee` subtree |
| `apple_pick_sim/tests/test_ee_cylinder_geometry.py` | Unit + asset regression tests |
| `assets/testfr3_resolved.usda` | Sim SoT FR3 + ee/tcp/cylinder |
| `assets/testfr3.usda` | Authoring FR3; must not disagree on tip/flange/TCP |
| `apple_pick_sim/robot/fr3_robot/paths.py` | `EE_CYLINDER_RADIUS`, `EE_CYLINDER_HALF_HEIGHT` (assert against) |
| `docs/real-world-proxy.md` | Radius + tip/flange contract + look-at warning |
| `docs/superpowers/specs/2026-08-04-true-tcp-pose-weld-design.md` | Look-at / gym warning |
| `docs/superpowers/specs/2026-07-24-real-post-grasp-viewer-design.md` | Obsolete orientation tables |
| `robot_replay/README.md` | True TCP SE(3) + follow-data copy |
| `docs/superpowers/specs/2026-08-04-tcp-tip-flange-geometry-design.md` | Mark Implemented when done |

---

### Task 1: Pure geometry helper + failing asset tests

**Files:**
- Create: `apple_pick_sim/robot/fr3_robot/ee_cylinder_geometry.py`
- Create: `apple_pick_sim/tests/test_ee_cylinder_geometry.py`
- Modify: `apple_pick_sim/robot/fr3_robot/__init__.py` only if other robot symbols are re-exported there (optional; not required for tests)

**Interfaces:**
- Produces:
  - `dataclass EeCylinderLayout` with fields `tip_z_m: float`, `flange_z_m: float`, `tcp_z_m: float`, `radius_m: float`, `length_m: float`
  - `def ee_cylinder_layout_from_authored(*, ee_scale_xyz: tuple[float,float,float], mesh_translate_xyz: tuple[float,float,float], mesh_scale_xyz: tuple[float,float,float], mesh_z_min: float, mesh_z_max: float, tcp_translate_xyz: tuple[float,float,float]) -> EeCylinderLayout`
  - `def scrape_ee_cylinder_authored(usd_path: Path) -> dict` returning the authored numbers above for `/fr3/ee` (regex/text scrape is OK if `pxr` stage traversal is awkward for overrides; prefer `pxr` if already used in `test_fr3_usd_import.py`)
  - `def assert_tip_flange_tcp_contract(layout: EeCylinderLayout, *, tip_tol_m: float = 1e-3, flange_tol_m: float = 1e-3) -> None` raising `AssertionError` with a clear message

**Convention for `ee_cylinder_layout_from_authored` (document in docstring):**

USD `xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]` on the mesh means parent-from-local for a point `p` is:

```text
p_ee_local = scale * (p + translate)   # orient = identity
```

Then ee’s own scale maps to meters along the tool axis (ee translate ignored for relative tip/flange/TCP):

```text
z_m = ee_scale_z * p_ee_local.z
r_m = ee_scale_x * mesh_scale_x * (mesh radial half-extent)
```

Use mesh points extent `z ∈ [mesh_z_min, mesh_z_max]` (resolved unit cylinder uses `[-0.5, 0.5]`). After mesh xform, **tip** = face nearer TCP, **flange** = other face (works for ee `−Z` tip-out). TCP `z_m = ee_scale_z * tcp_translate_z`.

**Corrected authored targets** (after joint-flip discovery; replaces early draft `(0,0,0.5)` / `(0,0,1)` which grew into link7):

| Prim | Field | Target |
|------|-------|--------|
| Cylinder | `xformOp:translate` | `(0, 0, ≈−0.4557)` so faces ≈ `[−0.9557, +0.0443]` pre-ee-scale |
| tcp | `xformOp:translate` | `(0, 0, ≈−0.9557)` → tip/tcp ≈ `−0.1338` m |
| tcp_joint | `physics:localPos0` | same as tcp translate z |
| Proximal face | meters | `≈ +0.0062` (meets link7 mesh) |
| Distal / TCP | meters | `≈ −0.1338` (length 0.14) |

- [x] **Step 1: Write failing tests (RED)**

```python
# apple_pick_sim/tests/test_ee_cylinder_geometry.py
from pathlib import Path

import pytest

from apple_pick_sim.robot.fr3_robot.ee_cylinder_geometry import (
    assert_tip_flange_tcp_contract,
    ee_cylinder_layout_from_authored,
    scrape_ee_cylinder_authored,
)
from apple_pick_sim.robot.fr3_robot.paths import (
    EE_CYLINDER_HALF_HEIGHT,
    EE_CYLINDER_RADIUS,
    TESTFR3_SCENE_USD,
)

REPO = Path(__file__).resolve().parents[2]
AUTHORING_USD = REPO / "assets" / "testfr3.usda"


def test_layout_math_tip_equals_tcp_when_authored_consistently():
    layout = ee_cylinder_layout_from_authored(
        ee_scale_xyz=(0.2, 0.2, 0.14),
        mesh_translate_xyz=(0.0, 0.0, 0.5),
        mesh_scale_xyz=(0.5, 0.5, 1.0),
        mesh_z_min=-0.5,
        mesh_z_max=0.5,
        tcp_translate_xyz=(0.0, 0.0, 1.0),
    )
    assert layout.length_m == pytest.approx(0.14, abs=1e-9)
    assert layout.radius_m == pytest.approx(0.05, abs=1e-9)
    assert_tip_flange_tcp_contract(layout)
    assert layout.flange_z_m == pytest.approx(0.0, abs=1e-9)
    assert layout.tip_z_m == pytest.approx(layout.tcp_z_m, abs=1e-9)


def test_resolved_usd_tip_flange_tcp_contract():
    authored = scrape_ee_cylinder_authored(TESTFR3_SCENE_USD)
    layout = ee_cylinder_layout_from_authored(**authored)
    assert layout.radius_m == pytest.approx(EE_CYLINDER_RADIUS, abs=1e-3)
    assert layout.length_m == pytest.approx(2.0 * EE_CYLINDER_HALF_HEIGHT, abs=1e-3)
    assert_tip_flange_tcp_contract(layout)


def test_authoring_usd_tip_flange_tcp_contract():
    authored = scrape_ee_cylinder_authored(AUTHORING_USD)
    layout = ee_cylinder_layout_from_authored(**authored)
    assert_tip_flange_tcp_contract(layout)
```

- [x] **Step 2: Run tests to verify RED**

```bash
uv run --env-file pytest.env pytest apple_pick_sim/tests/test_ee_cylinder_geometry.py -v --tb=short
```

Expected: `test_layout_math_*` may FAIL until helper exists (import error) or PASS once helper is correct; **`test_resolved_usd_*` and `test_authoring_usd_*` FAIL** on tip≠TCP and/or flange≠0 with current assets.

- [x] **Step 3: Implement helper (minimal)**

Create `ee_cylinder_geometry.py` with the dataclass, `ee_cylinder_layout_from_authored`, `assert_tip_flange_tcp_contract`, and `scrape_ee_cylinder_authored`.

Scrape strategy (keep deterministic, no Isaac):

1. Read USDA as text.
2. Locate the `def Xform "ee"` block under `fr3` (resolved uses nested `ee` with `Cylinder`, `tcp`, `tcp_joint`).
3. Parse `xformOp:scale` / `xformOp:translate` for `ee`, for child `Cylinder`, and for child `tcp`.
4. Hard-code mesh extent `(-0.5, 0.5)` for this asset family’s unit cylinder (document assumption; fail if points extent in file differs when present).

```python
@dataclass(frozen=True)
class EeCylinderLayout:
    tip_z_m: float
    flange_z_m: float
    tcp_z_m: float
    radius_m: float
    length_m: float


def ee_cylinder_layout_from_authored(...):
    sx, sy, sz = ee_scale_xyz
    mx, my, mz = mesh_scale_xyz
    tx, ty, tz = mesh_translate_xyz
    # p' = scale * (p + translate)
    z0 = mz * (mesh_z_min + tz)
    z1 = mz * (mesh_z_max + tz)
    tip_local = max(z0, z1)
    flange_local = min(z0, z1)
    tip_z_m = sz * tip_local
    flange_z_m = sz * flange_local
    tcp_z_m = sz * tcp_translate_xyz[2]
    radius_m = abs(sx * mx * 0.5)  # unit cylinder radius 0.5 in mesh local XY
    length_m = abs(tip_z_m - flange_z_m)
    return EeCylinderLayout(...)


def assert_tip_flange_tcp_contract(layout, *, tip_tol_m=1e-3, flange_tol_m=1e-3):
    if abs(layout.flange_z_m) > flange_tol_m:
        raise AssertionError(
            f"flange face z={layout.flange_z_m} m must be ~0 (flush with ee origin)"
        )
    if abs(layout.tip_z_m - layout.tcp_z_m) > tip_tol_m:
        raise AssertionError(
            f"tip z={layout.tip_z_m} m != tcp z={layout.tcp_z_m} m (tol={tip_tol_m})"
        )
```

- [x] **Step 4: Re-run tests**

```bash
uv run --env-file pytest.env pytest apple_pick_sim/tests/test_ee_cylinder_geometry.py -v --tb=short
```

Expected: math unit test **PASS**; resolved + authoring asset tests **FAIL** on contract (document the measured tip/tcp/flange in the failure message).

- [x] **Step 5: Commit** (only if user requested commits)

```bash
git add apple_pick_sim/robot/fr3_robot/ee_cylinder_geometry.py \
  apple_pick_sim/tests/test_ee_cylinder_geometry.py
git commit -m "$(cat <<'EOF'
Add ee cylinder tip/flange/TCP layout helper and failing asset tests.

EOF
)"
```

---

### Task 2: Fix `testfr3_resolved.usda`

**Files:**
- Modify: `assets/testfr3_resolved.usda` (`/fr3/ee` Cylinder translate, `/fr3/ee/tcp` translate, `tcp_joint` `physics:localPos0`)
- Test: `apple_pick_sim/tests/test_ee_cylinder_geometry.py::test_resolved_usd_tip_flange_tcp_contract`

**Interfaces:**
- Consumes: `ee_cylinder_layout_from_authored` / scrape from Task 1
- Target authored numbers (with current `ee` scale `(0.2, 0.2, 0.14)` and mesh scale `(0.5, 0.5, 1)`):

| Prim | Field | Target |
|------|-------|--------|
| Cylinder | `xformOp:translate` | `(0, 0, ≈−0.4557)` — tip-out on **ee −Z**, proximal meets link7 mesh |
| tcp | `xformOp:translate` | `(0, 0, ≈−0.9557)` → `tcp_z_m ≈ −0.1338` |
| tcp_joint | `physics:localPos0` | match tcp translate z |
| ee scale | unchanged | `(0.2, 0.2, 0.14)` → r=0.05, L=0.14 |

> **Do not** use `(0, 0, 0.5)` / `(0, 0, 1)`: `fr3_joint8` RotX(180) makes ee `+Z` point into the arm.

Do **not** change unrelated FR3 link geometry.

- [x] **Step 1: Confirm RED still fails on resolved**

```bash
uv run --env-file pytest.env pytest \
  apple_pick_sim/tests/test_ee_cylinder_geometry.py::test_resolved_usd_tip_flange_tcp_contract -v
```

Expected: FAIL tip≠tcp and/or flange≠0.

- [x] **Step 2: Edit resolved USDA**

In `assets/testfr3_resolved.usda`, inside `def Xform "ee"`:

1. Cylinder: set `double3 xformOp:translate = (0, 0, ≈−0.4557)` (tip-out ee −Z; proximal ≈ +6.2 mm).
2. tcp: set `double3 xformOp:translate = (0, 0, ≈−0.9557)`.
3. tcp_joint: set `point3f physics:localPos0` to the same z as tcp.

Leave mesh `xformOp:scale = (0.5, 0.5, 1)` and ee scale as-is.

- [x] **Step 3: GREEN resolved test**

```bash
uv run --env-file pytest.env pytest \
  apple_pick_sim/tests/test_ee_cylinder_geometry.py::test_resolved_usd_tip_flange_tcp_contract \
  apple_pick_sim/tests/test_ee_cylinder_geometry.py::test_layout_math_tip_equals_tcp_when_authored_consistently \
  -v
```

Expected: PASS.

- [x] **Step 4: FR3 import smoke still passes**

```bash
uv run --env-file pytest.env pytest apple_pick_sim/tests/test_fr3_usd_import.py -q --tb=line
```

Expected: PASS (or skip if usd-core missing — note in PR; do not delete the test).

- [x] **Step 5: Commit** (if requested)

```bash
git add assets/testfr3_resolved.usda
git commit -m "$(cat <<'EOF'
Align resolved FR3 ee cylinder tip with /ee/tcp and flange.

EOF
)"
```

---

### Task 3: Fix `testfr3.usda` (authoring)

**Files:**
- Modify: `assets/testfr3.usda` (`/fr3/ee` Cylinder + tcp; add `tcp_joint` localPos if present / needed)
- Test: `apple_pick_sim/tests/test_ee_cylinder_geometry.py::test_authoring_usd_tip_flange_tcp_contract`

**Interfaces:**
- Same contract: proximal meets link7 mesh (~+6.2 mm ee), tip z≈tcp z on ee −Z, Ø100×140.
- Authoring currently uses `ee` scale `(0.1, 0.1, 0.1)` and TCP at large **−Z**. Bring it onto the same physical contract as resolved:

**Preferred target (match resolved physics, avoid dual conventions):**

| Prim | Field | Target |
|------|-------|--------|
| ee | `xformOp:scale` | `(0.2, 0.2, 0.14)` |
| Cylinder | `xformOp:scale` | `(0.5, 0.5, 1)` |
| Cylinder | `xformOp:translate` | `(0, 0, ≈−0.4557)` |
| tcp | `xformOp:translate` | `(0, 0, ≈−0.9557)` |
| tcp_joint | `localPos0` if present | match tcp z |

If adding a joint is too invasive for authoring, at minimum fix tcp + cylinder transforms so scrape/layout contract passes; document any missing joint in the commit message.

- [x] **Step 1: RED authoring test**

```bash
uv run --env-file pytest.env pytest \
  apple_pick_sim/tests/test_ee_cylinder_geometry.py::test_authoring_usd_tip_flange_tcp_contract -v
```

Expected: FAIL.

- [x] **Step 2: Edit authoring USDA** to the table above (tip-out, flange flush, Ø100×140).

- [x] **Step 3: GREEN both asset tests**

```bash
uv run --env-file pytest.env pytest apple_pick_sim/tests/test_ee_cylinder_geometry.py -v
```

Expected: all PASS.

- [x] **Step 4: Commit** (if requested)

```bash
git add assets/testfr3.usda
git commit -m "$(cat <<'EOF'
Align authoring testfr3 ee TCP/cylinder with tip-out flange-flush contract.

EOF
)"
```

---

### Task 4: Documentation + README + supersession

**Files:**
- Modify: `docs/real-world-proxy.md`
- Modify: `docs/superpowers/specs/2026-08-04-true-tcp-pose-weld-design.md`
- Modify: `docs/superpowers/specs/2026-07-24-real-post-grasp-viewer-design.md`
- Modify: `robot_replay/README.md`
- Modify: `docs/superpowers/specs/2026-08-04-tcp-tip-flange-geometry-design.md` (status → Implemented)

**Interfaces:** none (docs only)

- [x] **Step 1: `real-world-proxy.md` EE section**

Replace tool radius **0.10 / Ø200** with **0.05 / Ø100**. Add tip/flange/+Z/bulk contract bullets. Add warning block:

```markdown
> **Warning — look-at vs logged TCP:** Gym, digital-twin, and generic
> `weld_direction` look-at welds do **not** yet consume a logged TCP SE(3).
> They use tip-out look-at (surface pole + constructed orientation). Only
> post-grasp replay (`real_post_grasp_plan` / `--grasp-after-settle`) uses
> full logged TCP pose. Do not assume look-at orientation matches recorded
> TCP quat.
```

Update coupling line to mention cylinder radius/half-height (not only `box_half_extents`). Fix checklist “Ø200” → “Ø100”.

- [x] **Step 2: `2026-08-04-true-tcp-pose-weld-design.md`**

Add the same warning under **Out of scope** / a new **Warnings** subsection. Keep post-grasp true-SE(3) as the implemented path.

- [x] **Step 3: `2026-07-24-real-post-grasp-viewer-design.md`**

At the top of each obsolete orientation table/row that says **+Z ∥ ŵ** / “do not use logged tcp quat”, add:

```markdown
> **Obsolete:** orientation superseded by
> `2026-08-04-true-tcp-pose-weld-design.md` (true TCP SE(3)). Do not implement
> look-at +Z∥ŵ for post-grasp replay.
```

Keep historical text; do not delete the whole doc.

- [x] **Step 4: `robot_replay/README.md`**

Replace:

- “optionally apply a post-grasp look-at weld”
- “Post-grasp snap (TCP-anchored surface apple; proxy +Z ∥ weld direction)”
- residual copy about forcing `|TCP−apple|=r` / look-at

With true TCP SE(3) + follow measured poses language matching `example_view_pre_grasp_settle.py`.

- [x] **Step 5: Mark tip/flange design Implemented**

In `2026-08-04-tcp-tip-flange-geometry-design.md`, set **Status** to `Implemented` and check success-criteria boxes that are done.

- [x] **Step 6: Commit** (if requested)

```bash
git add docs/real-world-proxy.md \
  docs/superpowers/specs/2026-08-04-true-tcp-pose-weld-design.md \
  docs/superpowers/specs/2026-07-24-real-post-grasp-viewer-design.md \
  docs/superpowers/specs/2026-08-04-tcp-tip-flange-geometry-design.md \
  robot_replay/README.md
git commit -m "$(cat <<'EOF'
Document tip/flange TCP contract and warn look-at is not logged SE(3).

EOF
)"
```

---

### Task 5: Final validation

**Files:** none new

- [x] **Step 1: Geometry + proxy regression suite**

```bash
uv run --env-file pytest.env pytest \
  apple_pick_sim/tests/test_ee_cylinder_geometry.py \
  apple_pick_sim/tests/test_real_world_proxy_fixture.py \
  apple_pick_sim/tests/test_real_post_grasp_plan.py \
  apple_pick_sim/tests/test_coupled_cable_scene.py \
  apple_pick_sim/tests/test_fr3_usd_import.py \
  -q --tb=line
```

Expected: all collected tests PASS (FR3 import may skip without usd-core).

- [x] **Step 2: Grep guard for stale user-facing look-at post-grasp copy**

```bash
rg -n "look-at weld|\+Z ∥ weld|proxy \+Z ∥|Ø200|0\.10 m \(100 mm\)" \
  robot_replay/README.md docs/real-world-proxy.md
```

Expected: no hits in those two files (obsolete 2026-07-24 historical doc may still contain the phrase **with** an Obsolete banner).

- [x] **Step 3: Spec self-check**

Confirm every success criterion in `2026-08-04-tcp-tip-flange-geometry-design.md` is checked or explicitly deferred (gym SE(3) remains deferred via warning).

---

## Spec coverage (plan self-review)

| Spec requirement | Task |
|------------------|------|
| Fix resolved USD tip=TCP, flange flush, Ø100×140 | Task 2 |
| Fix/quarantine authoring USD | Task 3 |
| Regression tip↔TCP (+ flange) test | Task 1–3 |
| Warning in true-tcp + real-world-proxy docs | Task 4 |
| Radius 0.05 in real-world-proxy | Task 4 |
| README true TCP / follow-data | Task 4 |
| Supersede 2026-07-24 orientation | Task 4 |
| Existing tip-out proxy tests green | Task 5 |
| Gym/DT true SE(3) out of scope | Warning only (Task 4); no code task |

**Placeholder scan:** none intentional.  
**Type consistency:** `EeCylinderLayout` + `ee_cylinder_layout_from_authored` / `scrape_ee_cylinder_authored` / `assert_tip_flange_tcp_contract` names stable across tasks.
