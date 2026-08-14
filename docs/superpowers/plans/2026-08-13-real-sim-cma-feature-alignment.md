# Real vs sim CMA feature alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make converted real bags and live sim-replay bags share one Sinkhorn `STATE_VECTOR` (world `ft_wrist`, two woody starts + `apple_pos`, scalar hold) and author Desk EE mass properties on `/fr3/ee`.

**Architecture:** Slice 0 authors USD `/fr3/ee` mass/COM/`I_ee` (TCP stays massless). Slice 1 rotates convert-time `ft_wrist` with `R(tcp)` (no second negate, no EMA). Slice 2 maps compiler Branch/Spur/Apple tags onto `CMA_WOODY_JUNCTIONS = (primary_spur, spur_stem)` and drops `woody_end` from the sys-ID bag. Slice 3 keeps scalar `hold_number` from `hold_index`. Implement in that order on `feature/real-replay-parallel-sysid`.

**Tech Stack:** Python, USDA `PhysicsMassAPI`, `real_to_batched_sysid.py`, `mmd_features.py`, pytest + `uv run --env-file pytest.env`.

**Spec:** `docs/superpowers/specs/2026-08-13-real-sim-cma-feature-alignment-design.md`

## Global Constraints

- Work on `feature/real-replay-parallel-sysid` (already a feature branch; do not edit `main`)
- TDD: failing test before production code
- Run tests: `uv run --env-file pytest.env python -m pytest -p no:launch_testing <path> -q`
- No sim F/T EMA / LPF; no torque slew on F/T obs; no second negate; no 0.1034/0.18 lever arm
- Score converted `ft_wrist` (baseline-corrected if present), not `ft_wrist_raw`
- Do not put COM/`I_ee` on `/ee/tcp`; TCP mass stays `0.001`
- Full 19D `action` stays in `STATE_VECTOR`
- Convert unified parquet only (woody/apple live there), not `*_robot.parquet`
- Live gym obs may still expose `woody_part_end_pos`; bags/collector/`STATE_VECTOR` must not

## File map

| Path | Responsibility |
| --- | --- |
| `apple_pick_sim/robot/fr3_robot/paths.py` | `EE_COM_IN_FLANGE_M`, `EE_COM_IN_EE_LOCAL_M`, `EE_INERTIA_DIAG_KGM2` |
| `apple_pick_sim/robot/fr3_robot/ee_cylinder_geometry.py` | RotX(180) COM helper + scrape ee mass/COM/inertia/tcp mass |
| `apple_pick_sim/robot/fr3_robot/__init__.py` | Re-export new constants |
| `assets/testfr3.usda`, `assets/testfr3_resolved.usda` | Author `/fr3/ee` mass 1.1 + COM + `I_ee` |
| `apple_pick_sim/system_id/mmd_features.py` | `CMA_WOODY_JUNCTIONS`; drop `woody_end` from `STATE_VECTOR` / collector; chord bending |
| `apple_pick_sim/system_id/real_to_batched_sysid.py` | World F/T rotate; compiler woody map; two-start bags |
| `apple_pick_sim/system_id/trajectory_store.py` | Optional `woody_part_end_pos` when **writing** rows |
| `apple_pick_sim/system_id/batched_trajectory_store.py` | **Read** path: `_stack_woody(WOODY_END_PREFIX)` must not require missing `woody_end__*` columns |
| `apple_pick_sim/system_id/real_pre_grasp_params.py` | **Do not change** length-9 compiler packing (plant rebuild). Bags only. |
| `apple_pick_gym/batched_envs/batched_sysid_collect.py` | Collect `junction_names = CMA_WOODY_JUNCTIONS` |
| `docs/real-world-proxy.md`, `docs/ROADMAP.md` | EE COM row; replace plumbing “slice 2 LPF” with this spec |

---

### Task 1: Slice 0 — USD EE mass / COM / `I_ee`

**Files:**
- Modify: `apple_pick_sim/robot/fr3_robot/paths.py`
- Modify: `apple_pick_sim/robot/fr3_robot/ee_cylinder_geometry.py`
- Modify: `apple_pick_sim/robot/fr3_robot/__init__.py`
- Modify: `apple_pick_sim/tests/test_ee_cylinder_geometry.py`
- Modify: `assets/testfr3.usda` (`/fr3/ee` mass block only)
- Modify: `assets/testfr3_resolved.usda` (`/fr3/ee` mass block only — file is huge; surgical replace)
- Modify: `docs/real-world-proxy.md` EE table + decision log

**Interfaces:**
- Consumes: Desk `ee_config` (`m_ee=1.1`, `F_x_Cee=(0,0,0.077)`, `I_ee` diagonal)
- Produces:
  - `EE_COM_IN_FLANGE_M: tuple[float, float, float] = (0.0, 0.0, 0.077)`
  - `EE_COM_IN_EE_LOCAL_M: tuple[float, float, float] = (0.0, 0.0, -0.077)`
  - `EE_INERTIA_DIAG_KGM2: tuple[float, float, float] = (0.0021521919406950474, 0.0021521919406950474, 0.0011912500485777855)`
  - `flange_com_to_ee_local(f_x_cee: tuple[float, float, float]) -> tuple[float, float, float]`
  - `scrape_ee_mass_properties(usd_path: Path) -> dict` with keys `ee_mass_kg`, `ee_com_xyz`, `ee_inertia_diag`, `tcp_mass_kg`

- [ ] **Step 1: Write the failing tests**

Add to `apple_pick_sim/tests/test_ee_cylinder_geometry.py`:

```python
from apple_pick_sim.robot.fr3_robot.ee_cylinder_geometry import (
    flange_com_to_ee_local,
    scrape_ee_mass_properties,
)
from apple_pick_sim.robot.fr3_robot.paths import (
    EE_COM_IN_EE_LOCAL_M,
    EE_COM_IN_FLANGE_M,
    EE_INERTIA_DIAG_KGM2,
    EE_MASS_KG,
)

def test_flange_com_to_ee_local_is_rotx_180():
    assert flange_com_to_ee_local((0.0, 0.0, 0.077)) == pytest.approx((0.0, 0.0, -0.077))
    assert flange_com_to_ee_local((0.01, 0.02, 0.03)) == pytest.approx((0.01, -0.02, -0.03))
    assert flange_com_to_ee_local(EE_COM_IN_FLANGE_M) == pytest.approx(EE_COM_IN_EE_LOCAL_M)


@pytest.mark.parametrize("usd_path", [AUTHORING_USD, TESTFR3_SCENE_USD])
def test_usd_ee_mass_com_inertia_and_massless_tcp(usd_path):
    props = scrape_ee_mass_properties(usd_path)
    assert props["ee_mass_kg"] == pytest.approx(EE_MASS_KG, abs=1e-6)
    assert props["ee_com_xyz"] == pytest.approx(EE_COM_IN_EE_LOCAL_M, abs=1e-6)
    assert props["ee_inertia_diag"] == pytest.approx(EE_INERTIA_DIAG_KGM2, rel=1e-6)
    assert props["tcp_mass_kg"] == pytest.approx(0.001, abs=1e-6)
```

`scrape_ee_mass_properties` must parse **only** `/fr3/ee` own attributes (text before the first nested `def`), then the nested `tcp` block — do not pick up link7 `centerOfMass`.

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_sim/tests/test_ee_cylinder_geometry.py::test_flange_com_to_ee_local_is_rotx_180 \
  apple_pick_sim/tests/test_ee_cylinder_geometry.py::test_usd_ee_mass_com_inertia_and_massless_tcp \
  -q
```

Expected: FAIL (`flange_com_to_ee_local` / `scrape_ee_mass_properties` not defined, or COM missing).

- [ ] **Step 3: Constants + helpers + USD**

`paths.py` (next to `EE_MASS_KG`):

```python
EE_COM_IN_FLANGE_M = (0.0, 0.0, 0.077)
EE_COM_IN_EE_LOCAL_M = (0.0, 0.0, -0.077)
EE_INERTIA_DIAG_KGM2 = (
    0.0021521919406950474,
    0.0021521919406950474,
    0.0011912500485777855,
)
```

`ee_cylinder_geometry.py`:

```python
def flange_com_to_ee_local(
    f_x_cee: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Map Franka flange F_x_Cee into USD ee local (fr3_joint8 RotX 180°)."""
    x, y, z = f_x_cee
    return (float(x), float(-y), float(-z))
```

Scrape: take the `ee` braced block, split own-attrs vs children at the first `\n        def `. Parse `float physics:mass`, `point3f physics:centerOfMass`, `float3 physics:diagonalInertia` from own-attrs; parse tcp `float physics:mass` from the `tcp` child. COM is in **meters**, not divided by ee scale `(0.2, 0.2, 0.18)`.

USD `/fr3/ee` (both files). Authoring currently `physics:mass = 1.5`; resolved already `1.1`. Insert on **ee**, not tcp:

```
        point3f physics:centerOfMass = (0, 0, -0.077)
        float3 physics:diagonalInertia = (0.0021521919406950474, 0.0021521919406950474, 0.0011912500485777855)
        quatf physics:principalAxes = (1, 0, 0, 0)
        float physics:mass = 1.1
```

Keep `tcp` `physics:mass = 0.001` with no COM/inertia. Export new constants from `fr3_robot/__init__.py`.

`docs/real-world-proxy.md`: add COM `(0,0,0.077)_F` / `(0,0,-0.077)_ee` and `I_ee` to the EE table; decision-log line dated 2026-08-13.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_sim/tests/test_ee_cylinder_geometry.py \
  apple_pick_sim/tests/test_real_world_proxy_fixture.py::test_fr3_ee_mass_matches_proxy_default \
  -q
```

Expected: PASS. Existing tip/flange/TCP tests still pass.

- [ ] **Step 5: Commit**

```bash
git add apple_pick_sim/robot/fr3_robot/paths.py \
  apple_pick_sim/robot/fr3_robot/ee_cylinder_geometry.py \
  apple_pick_sim/robot/fr3_robot/__init__.py \
  apple_pick_sim/tests/test_ee_cylinder_geometry.py \
  assets/testfr3.usda assets/testfr3_resolved.usda \
  docs/real-world-proxy.md
git commit -m "$(cat <<'EOF'
Author FR3 ee mass COM and inertia from recorded Desk load.

Keep /ee/tcp nearly massless so coupling wrenches stay at the tip.
EOF
)"
```

---

### Task 2: Slice 1 — convert F/T to world

**Files:**
- Modify: `apple_pick_sim/system_id/real_to_batched_sysid.py`
- Modify: `apple_pick_sim/tests/test_real_to_batched_sysid.py`

**Interfaces:**
- Consumes: `pose_4x4_to_pos_quat` in `apple_pick_sim/system_id/real_post_grasp_plan.py` (row-major 4×4, translation at 3,7,11)
- Produces: `world_wrench_from_ee_logged(ft_ee: np.ndarray, tcp_pose_4x4: Any) -> np.ndarray` shape `(6,)` float32

- [ ] **Step 1: Write the failing tests**

```python
def test_world_wrench_from_ee_logged_rotates_force_and_torque():
    from apple_pick_sim.system_id.real_to_batched_sysid import world_wrench_from_ee_logged

    # 90° about Z: e1 -> e2
    pose = [
        0.0, -1.0, 0.0, 0.0,
        1.0,  0.0, 0.0, 0.0,
        0.0,  0.0, 1.0, 0.0,
        0.0,  0.0, 0.0, 1.0,
    ]
    ft_ee = np.array([1.0, 0.0, 0.0, 0.0, 2.0, 0.0], dtype=np.float32)
    got = world_wrench_from_ee_logged(ft_ee, pose)
    # R @ e1 = e2; R @ (2 e2) = -2 e1. Do not expect τ → e3.
    np.testing.assert_allclose(got[:3], [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(got[3:], [-2.0, 0.0, 0.0], atol=1e-6)


def test_world_wrench_from_ee_logged_does_not_negate():
    from apple_pick_sim.system_id.real_to_batched_sysid import world_wrench_from_ee_logged

    pose = _identity_pose_4x4([0.1, 0.2, 0.3])
    ft_ee = np.array([1.0, -2.0, 3.0, 4.0, -5.0, 6.0], dtype=np.float32)
    got = world_wrench_from_ee_logged(ft_ee, pose)
    np.testing.assert_allclose(got, ft_ee, atol=1e-6)


def test_export_rotates_ft_wrist_and_requires_tcp_pose(tmp_path):
    # Extend _write_synthetic_real with tcp_pose_4x4 = RotZ(90°) and
    # ft_wrist = [1,0,0, 0,0,0], pull [0,-1,0].
    # After export_real_episode_to_batched_dataset, converted ft_wrist[:3]
    # equals R @ [1,0,0]. cos(F, pull) is computed on the converted bag.
    # A second call without tcp_pose_4x4 and pack_vic_pose True must raise.
```

Also: `ft_wrist_raw` if present is rotated the same way; converted `ft_wrist` must not equal the EE-frame input when `R != I`.

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_sim/tests/test_real_to_batched_sysid.py::test_world_wrench_from_ee_logged_rotates_force_and_torque \
  apple_pick_sim/tests/test_real_to_batched_sysid.py::test_world_wrench_from_ee_logged_does_not_negate \
  -q
```

Expected: FAIL (`world_wrench_from_ee_logged` missing).

- [ ] **Step 3: Implement rotate at convert**

```python
def world_wrench_from_ee_logged(ft_ee: Any, tcp_pose_4x4: Any) -> np.ndarray:
    ft = np.asarray(ft_ee, dtype=np.float64).reshape(6)
    R = np.asarray(tcp_pose_4x4, dtype=np.float64).reshape(4, 4)[:3, :3]
    out = np.empty(6, dtype=np.float32)
    out[:3] = (R @ ft[:3]).astype(np.float32)
    out[3:] = (R @ ft[3:]).astype(np.float32)
    return out
```

In `export_real_episode_to_batched_dataset` per-row loop, after reading `ft` / `raw_ft`: when `pack_vic_pose` is True, **raise** if `tcp_pose_4x4` is missing (today convert falls back to identity quat at lines ~651–654 — that must become a hard error for the CMA path). Then `ft = world_wrench_from_ee_logged(ft, tcp_pose)` and the same for `raw_ft`. Use logged `tcp_pose_4x4` (row-major numpy 4×4 from `ee_pos`/`ee_quat`). Do **not** parse SHM `O_T_EE` (column-major). Do **not** multiply by `-1`. Do not transport along 0.18 m.

- [ ] **Step 4: Run tests**

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_sim/tests/test_real_to_batched_sysid.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apple_pick_sim/system_id/real_to_batched_sysid.py \
  apple_pick_sim/tests/test_real_to_batched_sysid.py
git commit -m "$(cat <<'EOF'
Rotate converted real ft_wrist into the TCP world frame.

Match sim coupling wrenches without a second negate or lever-arm transport.
EOF
)"
```

---

### Task 3: Slice 2 — convert woody tags to two starts

**Files:**
- Modify: `apple_pick_sim/system_id/mmd_features.py` (add `CMA_WOODY_JUNCTIONS` only in this task if needed for import; or keep the constant in `real_to_batched_sysid.py` and re-export in Task 4 — **put it in `mmd_features.py`**)
- Modify: `apple_pick_sim/system_id/real_to_batched_sysid.py`
- Modify: `apple_pick_sim/tests/test_real_to_batched_sysid.py`

**Interfaces:**
- Consumes: compiler packing `starts=[Branch,Branch,Spur]`, `ends=[Spur,Apple,Apple]`
- Produces:
  - `CMA_WOODY_JUNCTIONS: tuple[str, str] = ("primary_spur", "spur_stem")` in `mmd_features.py`
  - `compiler_woody_to_cma_starts(start9: Any, end9: Any) -> dict[str, np.ndarray]` with those two keys, each shape `(3,)`

Replace `SIM_JUNCTION_NAMES` / `flat_woody_to_dicts` for **export bags only**. `rod_directions_from_woody` is **test-only** (not called from export). `real_pre_grasp_params.py` must keep reading compiler length-9 Branch/Spur/Apple for plant geometry.

- [ ] **Step 1: Write the failing tests**

Fix `_write_synthetic_real` woody to compiler topology:

```python
branch = [0.0, 1.0, 0.6]
spur = [0.0, 1.0, 0.5]
apple_pos = [0.0, 0.95, 0.38]
woody_start = branch + branch + spur
woody_end = spur + apple_pos + apple_pos
```

```python
def test_compiler_woody_to_cma_starts_maps_branch_and_spur():
    from apple_pick_sim.system_id.real_to_batched_sysid import compiler_woody_to_cma_starts

    branch = np.array([1.0, 2.0, 3.0])
    spur = np.array([4.0, 5.0, 6.0])
    apple = np.array([7.0, 8.0, 9.0])
    start9 = np.concatenate([branch, branch, spur])
    end9 = np.concatenate([spur, apple, apple])
    got = compiler_woody_to_cma_starts(start9, end9)
    assert set(got) == {"primary_spur", "spur_stem"}
    np.testing.assert_allclose(got["primary_spur"], branch)
    np.testing.assert_allclose(got["spur_stem"], spur)


def test_export_writes_two_woody_starts_and_no_ends(tmp_path):
    # After export, episode arrays junction_names == ["primary_spur", "spur_stem"]
    # woody_part_start_pos has those keys; woody_part_end_pos is missing or empty.
    # No stem_apple / support columns. apple_pos unchanged.
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_sim/tests/test_real_to_batched_sysid.py::test_compiler_woody_to_cma_starts_maps_branch_and_spur \
  -q
```

Expected: FAIL.

- [ ] **Step 3: Implement mapping + export**

```python
# mmd_features.py
CMA_WOODY_JUNCTIONS: tuple[str, str] = ("primary_spur", "spur_stem")
```

```python
def compiler_woody_to_cma_starts(start9: Any, end9: Any) -> dict[str, np.ndarray]:
    start = np.asarray(start9, dtype=np.float32).reshape(9)
    end = np.asarray(end9, dtype=np.float32).reshape(9)
    return {
        "primary_spur": start[0:3].copy(),
        "spur_stem": end[0:3].copy(),
    }
```

Export: `junction_names = list(CMA_WOODY_JUNCTIONS)`, `n_woody_parts = 2`. `obs["woody_part_start_pos"]` from `compiler_woody_to_cma_starts`. Omit `woody_part_end_pos` **if** `build_sysid_frame_row` already allows it (Task 5). If Task 3 runs first and the writer still requires ends, pass a dummy matching dict **only as a temporary** — do **not** ship that. **Order: implement Task 5 writer optionality in the same change set as this export omit**, or do Task 5 immediately after this task in the same session so convert never writes `woody_end__*` on main.

Practical sequencing: implement `build_sysid_frame_row` optional ends in this task (small, required for the export test).

In `build_sysid_frame_row`: if `woody_part_end_pos` missing, skip `woody_end__*` columns (only write `woody_start__*`). Keys of starts must be `CMA_WOODY_JUNCTIONS` for convert.

In `batched_trajectory_store.py` episode reader, `_stack_woody(WOODY_END_PREFIX)` today always reads `woody_end__{name}` for every `junction_names` entry — that will crash after convert omits those columns. If a column is absent, return `{}` or skip ends (do not require them for CMA arrays).

Update `episode_meta["junction_names"]` / `n_woody_parts` in convert metadata (today `SIM_JUNCTION_NAMES` at `build_episode_metadata_from_real`).

- [ ] **Step 4: Run tests**

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_sim/tests/test_real_to_batched_sysid.py \
  apple_pick_sim/tests/test_trajectory_store.py \
  apple_pick_sim/tests/test_batched_trajectory_store.py \
  -q
```

Expected: PASS (trajectory_store tests that still pass both start and end keep working).

- [ ] **Step 5: Commit**

```bash
git add apple_pick_sim/system_id/mmd_features.py \
  apple_pick_sim/system_id/real_to_batched_sysid.py \
  apple_pick_sim/system_id/trajectory_store.py \
  apple_pick_sim/tests/test_real_to_batched_sysid.py \
  apple_pick_sim/tests/test_trajectory_store.py
git commit -m "$(cat <<'EOF'
Map real Branch/Spur tags to two CMA woody starts.

Drop skip-level Branch→Apple and woody_end columns from converted bags.
EOF
)"
```

---

### Task 4: Slice 2 — `STATE_VECTOR`, bending, collector

**Files:**
- Modify: `apple_pick_sim/system_id/mmd_features.py`
- Modify: `apple_pick_sim/tests/test_mmd_features.py`

**Interfaces:**
- Consumes: `CMA_WOODY_JUNCTIONS`; arrays with `woody_part_start_pos` + `apple_pos`
- Produces: `build_bending_angles` chords:
  - `primary_spur`: `start[spur_stem] - start[primary_spur]`
  - `spur_stem`: `apple_pos - start[spur_stem]`
- `ReplayObservationCollector.record` requires `woody_start`, **not** `woody_end`
- `STATE_VECTOR_FIELDS` / `REQUIRED_ARRAY_KEYS` drop `woody_part_end_pos`
- `replay_obs_dict_from_sysid_numpy` omits `woody_end`

- [ ] **Step 1: Write the failing tests**

Replace `test_mmd_features.py` fixtures that feed `woody_part_end_pos` into `build_state_matrix` with two starts + `apple_pos`. Example bending test:

```python
def test_build_bending_angles_uses_spur_and_stem_chords():
    n = 2
    arrays = {
        "junction_names": ["primary_spur", "spur_stem"],
        "woody_part_start_pos": {
            "primary_spur": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
            "spur_stem": np.array([[0.0, 0.0, -0.1], [0.1, 0.0, 0.0]], dtype=np.float32),
        },
        "apple_pos": np.array([[0.0, 0.0, -0.2], [0.1, 0.0, -0.1]], dtype=np.float32),
    }
    ang = build_bending_angles(arrays, n_frames=n, junction_names=list(arrays["junction_names"]))
    assert ang.shape == (2, 2)
    assert ang[0, 0] == pytest.approx(0.0)
    assert ang[0, 1] == pytest.approx(0.0)
    # rest spur chord (0,0,-0.1); frame 1 (0.1,0,0) is 90° from -Z
    assert ang[1, 0] == pytest.approx(np.pi / 2, rel=1e-5)
    assert ang[1, 1] == pytest.approx(0.0)
```

Collector: `record({... no woody_end ...})` succeeds; missing `woody_start` still raises.

`build_state_matrix` width: `6+6+A+3+3+3*2+2` with `A = action_dim`.

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_sim/tests/test_mmd_features.py::test_build_bending_angles_uses_spur_and_stem_chords \
  -q
```

Expected: FAIL (still uses `woody_part_end_pos`).

- [ ] **Step 3: Implement**

`build_bending_angles`: implement the two-chord table when `junction_names == list(CMA_WOODY_JUNCTIONS)`. For other names (existing `joint_a` fixtures), use the same distal rule: chord `i` is `start[i+1] - start[i]` for `i < n-1` and `apple_pos - start[last]` for the last — **do not** keep a `woody_part_end_pos` path after `REQUIRED_ARRAY_KEYS` drops ends. Update `test_mmd_features.py` / wasserstein fixtures that asserted FIXED-joint end−start angles. Do not force every helper test onto CMA names if the generic last-chord=`apple_pos` rule already matches.

Collector: drop `_woody_end`; drop `woody_end` from the required obs keys; `to_arrays` omits `woody_part_end_pos`.

- [ ] **Step 4: Run tests**

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_sim/tests/test_mmd_features.py \
  apple_pick_sim/tests/test_wasserstein.py \
  -q
```

Expected: PASS after updating wasserstein fixtures to two starts + apple (no ends in `STATE_VECTOR`).

- [ ] **Step 5: Commit**

```bash
git add apple_pick_sim/system_id/mmd_features.py \
  apple_pick_sim/tests/test_mmd_features.py \
  apple_pick_sim/tests/test_wasserstein.py
git commit -m "$(cat <<'EOF'
Score woody starts and apple_pos chords, not FIXED-joint end anchors.

Align Sinkhorn bending with real Branch/Spur/Apple tags.
EOF
)"
```

---

### Task 5: Slice 2 — collect bags + gym replay adapter

**Files:**
- Modify: `apple_pick_gym/batched_envs/batched_sysid_collect.py` (metadata `junction_names`)
- Modify: `apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py` (`replay_obs_dict_from_sysid_numpy` call site if it still passes ends; woody MSE; prefer `true_params_for_structure` for plant rebuild)
- Modify: `apple_pick_gym/grid_viz_metrics.py`, `apple_pick_gym/grid_viz_table.py` (end-MSE)
- Modify: `apple_pick_sim/system_id/batched_trajectory_store.py` (reader)
- Modify: `apple_pick_sim/system_id/parquet_init.py` / `batched_digital_twin_init.py` only if collect-format tests fail after dropping ends
- Modify: `apple_pick_gym/batched_envs/obs_torch.py` only if collect export still flattens ends into bags — live obs may keep ends
- Tests: `apple_pick_gym/tests/test_batched_sysid_mmd_grid_helpers.py`, `test_batched_obs_torch.py`, `test_replay_env.py`, `apple_pick_gym/tests/test_batched_replay_export.py`, `test_batched_sysid_collect.py`, `apple_pick_sim/tests/test_sysid_dashboard_data.py` — update collector/state fixtures; do **not** delete live `woody_end` gym obs tests on `ApplePickCoupledEnv`

**Interfaces:**
- Collect parquet `junction_names == ["primary_spur", "spur_stem"]` even if `env.junction_names` is the full T-junction set (`primary_support_left/right`, `stem_apple`, …). There is no junction named `"support"`.
- Filter start columns written to those names; do not write `woody_end__*`
- Grid woody MSE: if ends absent, use starts + `apple_pos` (spec) — skip end-based MSE rather than resurrecting columns

- [ ] **Step 1: Write a failing collect-metadata unit test**

If collect helpers can be tested without GPU: assert a helper `cma_woody_junctions_from_env(names: list[str]) -> list[str]` returns `["primary_spur", "spur_stem"]` given the real T-junction set (`primary_support_left/right`, `stem_apple`, …), and raises if either required name is missing. There is no junction named `"support"`.

```python
def test_cma_woody_junctions_filters_support():
    from apple_pick_sim.system_id.mmd_features import cma_woody_junctions_from_env

    assert cma_woody_junctions_from_env(
        [
            "primary_support_left",
            "primary_support_right",
            "primary_spur",
            "spur_stem",
            "stem_apple",
        ]
    ) == ["primary_spur", "spur_stem"]
```

Wire collect metadata + `build_sysid_frame_row` obs to pass only those start dicts.

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_sim/tests/test_mmd_features.py::test_cma_woody_junctions_filters_support \
  -q
```

Expected: FAIL.

- [ ] **Step 3: Implement filter + update gym fixtures**

```python
def cma_woody_junctions_from_env(names: list[str]) -> list[str]:
    have = set(names)
    missing = [n for n in CMA_WOODY_JUNCTIONS if n not in have]
    if missing:
        raise ValueError(f"env junction_names missing {missing}; got {names}")
    return list(CMA_WOODY_JUNCTIONS)
```

Collect: `junction_names = cma_woody_junctions_from_env(env.junction_names)`; when recording, subset `woody_part_start_pos` to those keys; omit ends.

Update gym tests that construct `ReplayObservationCollector` / `build_state_matrix` to drop `woody_end`. Leave `test_apple_pick_coupled_env.py` live `woody_end` obs assertions. Live gym `obs_torch.py` / `ApplePickCoupledEnv` may keep ends.

Grid MSE helpers that iterate `woody_part_end_pos` (`batched_sysid_mmd_grid.py`, `grid_viz_metrics.py`, `grid_viz_table.py`): if key missing, compute segment MSE from the two starts + `apple_pos` or return empty end-MSE dict.

**CMA vs MMD-grid plant rebuild:** CMA uses `true_params_for_structure` (metadata `fruiting_system_params`) — dropping bag ends does **not** break CMA. `infer_base_params_for_structure` / `digital_twin_obs_from_batched_episode` / `parquet_init.py` still read `woody_part_end_pos` to infer rods (stem last chord uses `end[spur_stem]`; primary uses support **ends**). Do **not** change `infer_segment_geometry` unless a collect-format test fails. Prefer pointing MMD-grid plant rebuild at `true_params_for_structure` when metadata already has `fruiting_system_params` (sim collect already writes it). Do not change `real_pre_grasp_params.py` length-9 compiler packing.

Other bag readers that assume ends: `dashboard_data.py` (`woody_endpoint_series`), `example_gym_replay.py`, `test_sysid_dashboard_data.py`, `test_batched_digital_twin_init.py`, `test_batched_sysid_collect.py`, `test_batched_replay_export.py`. Make them tolerate missing ends or switch to starts + `apple_pos`.

- [ ] **Step 4: Run tests**

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_sim/tests/test_mmd_features.py \
  apple_pick_gym/tests/test_batched_sysid_mmd_grid_helpers.py \
  apple_pick_gym/tests/test_batched_replay_export.py \
  apple_pick_gym/tests/test_replay_env.py \
  apple_pick_gym/tests/test_batched_obs_torch.py \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apple_pick_sim/system_id/mmd_features.py \
  apple_pick_gym/batched_envs/batched_sysid_collect.py \
  apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py \
  apple_pick_gym/tests
git commit -m "$(cat <<'EOF'
Collect CMA woody bags as two starts, filtering support joints.

Keep live gym woody_end obs for debug; do not persist it in sys-ID parquet.
EOF
)"
```

---

### Task 6: Slice 3 — scalar hold_number

**Files:**
- Modify: `apple_pick_sim/tests/test_real_to_batched_sysid.py` (and `apple_pick_sim/system_id/real_to_batched_sysid.py` only if `_scalar_hold_number` fails the new cases)

**Interfaces:**
- Consumes: existing `_scalar_hold_number(value, *, hold_index=None) -> int`
- Produces: unchanged behavior, locked by tests — prefer `hold_index`; never write length-4 one-hot into converted parquet `hold_number`

- [ ] **Step 1: Write the failing tests**

```python
def test_scalar_hold_number_prefers_hold_index_over_onehot():
    from apple_pick_sim.system_id.real_to_batched_sysid import _scalar_hold_number

    assert _scalar_hold_number([0.0, 0.0, 1.0, 0.0], hold_index=2) == 2
    assert _scalar_hold_number([0.0, 1.0, 0.0, 0.0], hold_index=None) == 1
    assert _scalar_hold_number(None, hold_index=0) == 0
    assert _scalar_hold_number(None, hold_index=None) == -1


def test_export_hold_number_is_scalar(tmp_path):
    # _write_synthetic_real rows with hold_index=1 and hold_number=[0,1,0,0]
    # converted arrays["hold_number"] dtype int, shape (n,), values == 1
```

- [ ] **Step 2: Run tests**

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_sim/tests/test_real_to_batched_sysid.py::test_scalar_hold_number_prefers_hold_index_over_onehot \
  -q
```

Expected: may already PASS (implementation exists). If PASS, keep the tests as the lock. If FAIL, fix `_scalar_hold_number` only.

- [ ] **Step 3: Confirm scorer still one-hots**

Existing `test_mmd_features.py` hold-onehot tests must still pass with scalar `hold_number` in arrays.

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_sim/tests/test_mmd_features.py \
  apple_pick_sim/tests/test_real_to_batched_sysid.py \
  -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add apple_pick_sim/tests/test_real_to_batched_sysid.py \
  apple_pick_sim/system_id/real_to_batched_sysid.py
git commit -m "$(cat <<'EOF'
Lock converted hold_number to a scalar from hold_index.

Leave Sinkhorn hold one-hot to score time.
EOF
)"
```

---

### Task 7: ROADMAP — replace plumbing slice-2 LPF plan

**Files:**
- Modify: `docs/ROADMAP.md` Current focus (the paragraph that still says “F/T frame + LPF” and `action[0:7]`)
- Modify: `docs/superpowers/specs/2026-08-12-real-replay-cmaes-plumbing-design.md` Slice 2 pointer → this spec

- [ ] **Step 1: Edit ROADMAP Current focus**

Replace “Align sim `ft_wrist` … LPF … `action[0:7]`” with: implement `docs/superpowers/specs/2026-08-13-real-sim-cma-feature-alignment-design.md` slices 0–3 (USD COM → convert F/T rotate → two-start woody → scalar hold). No sim EMA. No pose-only action.

Checklist: tick slice 0–3 as they land; keep “trusted Cartesian ranking” / CMA as later.

- [ ] **Step 2: Point plumbing spec Slice 2 at the new spec**

One sentence: superseded by `2026-08-13-real-sim-cma-feature-alignment-design.md` (convert-time rotate, no LPF).

- [ ] **Step 3: Commit**

```bash
git add docs/ROADMAP.md \
  docs/superpowers/specs/2026-08-12-real-replay-cmaes-plumbing-design.md \
  docs/superpowers/specs/2026-08-13-real-sim-cma-feature-alignment-design.md
git commit -m "$(cat <<'EOF'
Point M4.0 current focus at real/sim CMA feature alignment.

Supersede the plumbing spec's score-time F/T+LPF slice.
EOF
)"
```

---

## Self-review (spec coverage)

| Spec requirement | Task |
| --- | --- |
| USD `/fr3/ee` mass 1.1, COM `(0,0,-0.077)`, `I_ee`, massless TCP | 1 |
| No COM on `/ee/tcp` | 1 |
| Convert `R(tcp) @` F and τ; no negate; raise if no `tcp_pose_4x4` | 2 |
| Score `ft_wrist` not raw; no sim EMA | 2 + Global Constraints |
| Compiler Branch/Spur/Apple → two starts + `apple_pos` | 3 |
| Drop `woody_end` from bag / collector / `STATE_VECTOR` | 3, 4, 5 |
| Bending from spur then stem chords | 4 |
| Collect `junction_names` two CMA junctions (no support) | 5 |
| Scalar `hold_number` from `hold_index` | 6 |
| ROADMAP / plumbing spec amend | 7 |
| Live gym `woody_end` obs allowed | 5 (do not delete coupled-env tests) |

---

## Code verification (2026-08-13)

Checked against this branch plus the pinned real logger (`Continuous_Force_RL` `apple_pullto_static.py` / `compile_static_sysid.py`, local interface `tmp/ad.py`). Plan snippets above already incorporate the fixes.

**Confirmed — do not reopen**

- Convert path, `build_sysid_frame_row`, collector, `STATE_VECTOR` still require `woody_end` today; slice 2 is a real gap.
- Logged `tcp_pose_4x4` is row-major (numpy 4×4 from `ee_pos`/`ee_quat`). SHM `O_T_EE` is column-major and is **unpacked before** logging; convert must use `tcp_pose_4x4`, which matches `pose_4x4_to_pos_quat`.
- Interface rotates **both** F and τ with `R.T` then negates. Slice 1 `R @` undoes the frame without a second negate.
- Real rows have `hold_index` (scalar) and `hold_number` (4-vector one-hot). `_scalar_hold_number` already prefers `hold_index`.
- CMA plant rebuild is `true_params_for_structure` (metadata JSON from pre-grasp), not bag woody ends.
- `env.junction_names` after reset: `primary_spur`, `spur_stem`, `stem_apple`, `primary_support_left`, `primary_support_right`. Filter must keep the two CMA names and ignore the rest.
- Resolved USD `/fr3/ee` mass is already 1.1; authoring USDA is still 1.5. Other links use `point3f physics:centerOfMass`. Newton reads `GetDiagonalInertiaAttr` + optional `principalAxes`; identity quat default is `Gf.Quatf(1,0,0,0)`.
- Unified parquet (not `*_robot.parquet`) is the convert input; compiler woody is length-9 Branch/Spur/Apple.

**Plan bugs that were wrong vs code (now patched in tasks)**

1. Task 2 RotZ unit test: `R @ (0,2,0) = (−2,0,0)`, not `(0,0,2)`.
2. Task 2 must **raise** if `pack_vic_pose` and `tcp_pose_4x4` missing (today identity-quat fallback).
3. Task 4 bending fixture: original `spur_stem[1] = (0.1,0,-0.1)` / `apple_pos[1] = (0.1,0,-0.2)` gave a 0° spur chord change, not the claimed 90°. Fixed fixture: `spur_stem[1] = (0.1,0,0)`, `apple_pos[1] = (0.1,0,-0.1)`. This keeps `spur_stem` chord (`apple_pos - start[spur_stem]`) constant at `(0,0,-0.1)` (0°, as asserted) while `primary_spur` chord (`start[spur_stem] - start[primary_spur]`) rotates from `(0,0,-0.1)` to `(0.1,0,0)` — a true 90°, matching `ang[1,0] == pi/2`.
4. Task 5 helper input is not `["support", …]`.
5. `rod_directions_from_woody` is test-only; do not treat it as export metadata.
6. `_write_synthetic_real` currently packs `spur_start + mid + spur_end` — switch to compiler `branch+branch+spur` / `spur+apple+apple` in Task 3. Keep `real_pre_grasp_params.py` on the compiler 9-vector.
7. Writer optionality is not enough: `batched_trajectory_store._stack_woody(WOODY_END_PREFIX)` will crash after convert omits columns.
8. Blast radius beyond the original Task 5 list: grid MSE, dashboard woody series, `parquet_init`, `digital_twin_obs_from_batched_episode`, replay-export tests. Live gym obs may keep ends.

**Do not mix** uncommitted `example_youngs_modulus_sys_id.py` GL/`--record-video` work into alignment commits.
