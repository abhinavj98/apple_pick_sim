# Stem harvest without penalty-weld damping

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stem harvest (arm `body_f` / gym `ft_wrist`) reports \(kC+\lambda\) only; VBD still damps the stem–apple penalty weld.

**Architecture:** Add `include_penalty_damping: bool = True` to `gather_joint_wrench_child_com_device`. When false, pass a cached zeros array instead of `solver.joint_penalty_kd` into the existing Newton gather kernel. Production stem harvest (device + CPU) and the harvest-parity helper pass `False`. Woody/debug gather keeps `True`. Do not patch Newton; do not zero fixture `stem_apple` \(k_d\).

**Tech Stack:** NVIDIA Warp, Newton `SolverVBD` gather kernel, pytest via `uv run --env-file pytest.env`.

**Spec:** `docs/superpowers/specs/2026-09-03-stem-harvest-no-penalty-damping-design.md`

## Global Constraints

- Do not edit `newton/` (submodule).
- Do not change fixture `joint_damping_ratio` / `stem_apple` \(k_d\) as the fix.
- Do not allocate a new zeros `wp.array` every coupled substep; cache on the solver.
- Do not include first-pull leftover-\(C\) work in this slice.
- TDD: failing test before production edits; `uv run --env-file pytest.env python -m pytest …` from repo root.
- Harvest sign, \(mg\), gain, and 40 N / 10 N·m caps stay as today.

## File map

| File | Role |
| --- | --- |
| `apple_pick_sim/vbd_fixed_joint_wrenches.py` | Flag + cached zeros; `fixed_joint_wrenches_child_com_vbd` forwards the flag |
| `apple_pick_sim/coupled_fruiting/proxy_coupling.py` | Device + CPU stem harvest pass `False` |
| `apple_pick_sim/tests/test_stem_harvest_penalty_damping.py` | Identity test \(F_{\mathrm{on}}-F_{\mathrm{off}}\approx -k_d\dot C\) |
| `apple_pick_sim/tests/test_coupled_fruiting_system.py` | `_stem_apple_wrench_from_scene` matches harvest (`False`) |
| `apple_pick_sim/tests/test_batched_stem_harvest.py` | Mock may assert `include_penalty_damping is False` |
| `docs/handbook-coupled-simulation.md` | H1 §6 one paragraph |
| `docs/WRENCH_READOUT.md` | Harvest vs debug gather |
| `docs/explicit-apple-load-tcp-harvest.md` | Gather flag on harvest path |
| `docs/damping-tuning.md` | One sentence: weld \(k_d\) stays in VBD, not TCP |

`batched_obs.py` keeps the default `True` (woody junction readout).

---

### Task 1: Failing gather identity test

**Files:**
- Create: `apple_pick_sim/tests/test_stem_harvest_penalty_damping.py`
- Modify: none yet

**Interfaces:**
- Consumes: `gather_joint_wrench_child_com_device` as it exists today (no flag yet)
- Produces: a test that must fail until Task 2 adds `include_penalty_damping`

- [ ] **Step 1: Write the failing test**

```python
"""Penalty-weld kd is omitted from stem harvest gather, not from VBD."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from conftest import (
    COUPLED_VBD_SCENE_KW,
    RANGES_FIXTURE,
    SUB_DT,
    build_vbd_only,
    requires_fr3,
)


def _import_cf():
    from apple_pick_sim import coupled_fruiting as cf

    return cf


def _import_fs():
    from apple_pick_sim import fruiting_system as fs

    return fs


def _stem_linear_kd(solver, joint_index: int) -> float:
    c0 = int(solver.joint_constraint_start.numpy()[int(joint_index)])
    return float(solver.joint_penalty_kd.numpy()[c0])


@requires_fr3
def test_gather_without_penalty_damping_drops_kd_cdot():
    """Same body_q; apple-shifted body_q_prev ⇒ F_on - F_off ≈ -kd Ċ (child)."""
    from apple_pick_sim.fruiting_system import GripperProxyConfig, load_ranges
    from apple_pick_sim.vbd_fixed_joint_wrenches import (
        gather_joint_wrench_child_com_device,
    )

    cf = _import_cf()
    fs = _import_fs()
    scene = build_vbd_only(
        cf,
        load_ranges(RANGES_FIXTURE),
        seed=21,
        gripper_proxy=GripperProxyConfig(fix_to_apple=True),
        **COUPLED_VBD_SCENE_KW,
    )
    cf.settle_vbd_substeps(scene, substeps=80, dt=SUB_DT)
    cable = scene.cable
    apple = int(cable.apple_body)
    stem_j = int(scene.stem_apple_joint_index)
    kd = _stem_linear_kd(cable.solver, stem_j)
    assert kd > 1.0, f"expected nontrivial stem_apple linear kd, got {kd}"

    q = cable.state_0.body_q.numpy().reshape(-1, 7).copy()
    q_prev = q.copy()
    delta = 1.0e-4
    q_prev[apple, 0] -= delta
    q_wp = wp.array(q, dtype=wp.transform, device=cable.solver.device)
    q_prev_wp = wp.array(q_prev, dtype=wp.transform, device=cable.solver.device)
    dt = float(SUB_DT)
    cdot = np.array([delta / dt, 0.0, 0.0], dtype=np.float64)

    f_on, _ = gather_joint_wrench_child_com_device(
        cable.model,
        cable.solver,
        body_q=q_wp,
        body_q_prev=q_prev_wp,
        joint_indices=[stem_j],
        dt=dt,
        include_penalty_damping=True,
    )
    f_off, _ = gather_joint_wrench_child_com_device(
        cable.model,
        cable.solver,
        body_q=q_wp,
        body_q_prev=q_prev_wp,
        joint_indices=[stem_j],
        dt=dt,
        include_penalty_damping=False,
    )
    diff = f_on.numpy()[0].astype(np.float64) - f_off.numpy()[0].astype(np.float64)
    np.testing.assert_allclose(diff, -kd * cdot, rtol=0.08, atol=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_stem_harvest_penalty_damping.py::test_gather_without_penalty_damping_drops_kd_cdot \
  -q
```

Expected: FAIL with `TypeError: ... unexpected keyword argument 'include_penalty_damping'` (flag missing). If the flag already exists and both gathers match, the test is wrong — stop.

- [ ] **Step 3: Commit the failing test only**

```bash
git add apple_pick_sim/tests/test_stem_harvest_penalty_damping.py
git commit -m "$(cat <<'EOF'
Add failing test for stem gather without penalty-weld kd.

EOF
)"
```

---

### Task 2: `include_penalty_damping` on device gather

**Files:**
- Modify: `apple_pick_sim/vbd_fixed_joint_wrenches.py` (`gather_joint_wrench_child_com_device`, `fixed_joint_wrenches_child_com_vbd`)

**Interfaces:**
- Consumes: Task 1 test
- Produces: `gather_joint_wrench_child_com_device(..., include_penalty_damping: bool = True)` and the same flag on `fixed_joint_wrenches_child_com_vbd`

- [ ] **Step 1: Implement the flag (minimal)**

In `gather_joint_wrench_child_com_device`, add `include_penalty_damping: bool = True`. Before `wp.launch`, select the kd buffer:

```python
kd_buf = solver.joint_penalty_kd
if not include_penalty_damping:
    zeros = getattr(solver, "_joint_penalty_kd_harvest_zeros", None)
    src = solver.joint_penalty_kd
    if (
        zeros is None
        or zeros.shape != src.shape
        or zeros.dtype != src.dtype
        or str(zeros.device) != str(src.device)
    ):
        zeros = wp.zeros_like(src)
        solver._joint_penalty_kd_harvest_zeros = zeros
    kd_buf = zeros
```

Pass `kd_buf` into the kernel where `solver.joint_penalty_kd` is passed today. Do not write `solver.joint_penalty_kd`.

On `fixed_joint_wrenches_child_com_vbd`, add `include_penalty_damping: bool = True` and gather via `gather_joint_wrench_child_com_device` (then copy `out_f`/`out_t` to the existing `FixedJointWrenchRecord` list). That keeps CPU harvest and NumPy helpers on one kernel. Default remains `True` so `test_wrench_equilibrium.py` is unchanged.

- [ ] **Step 2: Run the identity test (green)**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_stem_harvest_penalty_damping.py::test_gather_without_penalty_damping_drops_kd_cdot \
  -q
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add apple_pick_sim/vbd_fixed_joint_wrenches.py \
  apple_pick_sim/tests/test_stem_harvest_penalty_damping.py
git commit -m "$(cat <<'EOF'
Add include_penalty_damping=False to VBD joint gather.

EOF
)"
```

---

### Task 3: Stem harvest uses `False`; parity helpers match

**Files:**
- Modify: `apple_pick_sim/coupled_fruiting/proxy_coupling.py` (`harvest_batched_stem_tension` gather ~664, `harvest_stem_tension_for_tcp` gather ~972, `_harvest_stem_tension_for_tcp_cpu` ~846)
- Modify: `apple_pick_sim/tests/test_coupled_fruiting_system.py` (`_stem_apple_wrench_from_scene` ~967–985)
- Modify: `apple_pick_sim/tests/test_batched_stem_harvest.py` (optional assert on captured kwarg)
- Modify: `apple_pick_sim/tests/test_stem_harvest_penalty_damping.py` (add harvest-forwards test)

**Interfaces:**
- Consumes: `gather_joint_wrench_child_com_device(..., include_penalty_damping=False)`
- Produces: TCP `proxy_forces` / `ft_wrist` without \(k_d\dot C\)

- [ ] **Step 1: Write a failing harvest-forward test** in `test_stem_harvest_penalty_damping.py`

```python
@requires_fr3
def test_stem_harvest_gather_omits_penalty_damping():
    """harvest_stem_tension_for_tcp calls gather with include_penalty_damping=False."""
    from unittest.mock import patch

    from apple_pick_sim.coupled_fruiting.proxy_coupling import harvest_stem_tension_for_tcp

    captured: dict = {}

    def _fake_gather(*_a, **kwargs):
        captured["include_penalty_damping"] = kwargs.get("include_penalty_damping", True)
        n = 1
        dev = kwargs["body_q"].device
        return (
            wp.zeros(n, dtype=wp.vec3, device=dev),
            wp.zeros(n, dtype=wp.vec3, device=dev),
        )

    with patch(
        "apple_pick_sim.coupled_fruiting.proxy_coupling.gather_joint_wrench_child_com_device",
        side_effect=_fake_gather,
    ):
        # The harvest module imports gather inside the function; patch the
        # symbol used at call time (proxy_coupling's local import).
        pass
```

Do **not** use a broken stub. Harvest imports gather **inside** the function from `apple_pick_sim.vbd_fixed_joint_wrenches`. Patch that module:

```python
@requires_fr3
def test_stem_harvest_gather_omits_penalty_damping():
    from unittest.mock import patch

    import warp as wp

    from apple_pick_sim.coupled_fruiting.proxy_coupling import harvest_stem_tension_for_tcp

    captured: dict = {}

    def _fake_gather(*_a, **kwargs):
        captured["include_penalty_damping"] = kwargs.get("include_penalty_damping", True)
        dev = kwargs["body_q"].device
        return (
            wp.zeros(1, dtype=wp.vec3, device=dev),
            wp.zeros(1, dtype=wp.vec3, device=dev),
        )

    out = wp.zeros(8, dtype=wp.spatial_vector, device="cpu")
    bq = wp.zeros(4, dtype=wp.transform, device="cpu")
    with patch(
        "apple_pick_sim.vbd_fixed_joint_wrenches.gather_joint_wrench_child_com_device",
        side_effect=_fake_gather,
    ):
        harvest_stem_tension_for_tcp(
            cable_model=_CableModel(),  # control() stub as in test_batched_stem_harvest
            cable_solver=object(),
            body_q_post=bq,
            body_q_prev=bq,
            dt=SUB_DT,
            stem_apple_joint_index=0,
            tcp_body_index=0,
            out_robot_wrenches=out,
            explicit_apple_weight=False,
            explicit_apple_inertia=False,
        )
    assert captured["include_penalty_damping"] is False
```

Copy `_CableModel` from `test_batched_stem_harvest.py` (the class with `control()`). Run this test **before** wiring harvest: expected FAIL (`True` or missing key).

- [ ] **Step 2: Run it (red)**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_stem_harvest_penalty_damping.py::test_stem_harvest_gather_omits_penalty_damping \
  -q
```

Expected: FAIL (`assert True is False` or key missing).

- [ ] **Step 3: Wire harvest**

All three gather call sites pass `include_penalty_damping=False`.

`_harvest_stem_tension_for_tcp_cpu`: stop calling `fixed_joint_wrenches_child_com_vbd` without the flag. Either pass `include_penalty_damping=False` into `fixed_joint_wrenches_child_com_vbd` or call `gather_joint_wrench_child_com_device` with `False` and `.numpy()`. Prefer the shared helper with the flag so CPU/GPU stay identical.

`_stem_apple_wrench_from_scene` in `test_coupled_fruiting_system.py` must pass `include_penalty_damping=False` (same as harvest). Otherwise `test_fix_to_apple_tcp_harvest_matches_stem_apple_joint` will fail: TCP would drop \(k_d\dot C\) while the reference gather still includes it.

In `test_batched_stem_harvest.py` `_fake_gather`, record `kwargs.get("include_penalty_damping")` and assert `False` after `harvest_batched_stem_tension`.

- [ ] **Step 4: Run harvest + identity + existing stem tests**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_stem_harvest_penalty_damping.py \
  apple_pick_sim/tests/test_batched_stem_harvest.py \
  apple_pick_sim/tests/test_coupled_fruiting_system.py \
  apple_pick_sim/tests/test_explicit_apple_load.py \
  apple_pick_sim/tests/test_proxy_coupling.py \
  apple_pick_sim/tests/test_wrench_equilibrium.py \
  -q
```

Expected: PASS. `test_wrench_equilibrium` still uses default `True`.

- [ ] **Step 5: Commit**

```bash
git add apple_pick_sim/coupled_fruiting/proxy_coupling.py \
  apple_pick_sim/tests/test_stem_harvest_penalty_damping.py \
  apple_pick_sim/tests/test_coupled_fruiting_system.py \
  apple_pick_sim/tests/test_batched_stem_harvest.py
git commit -m "$(cat <<'EOF'
Omit penalty-weld kd from stem TCP harvest.

EOF
)"
```

---

### Task 4: Handbooks

**Files:**
- Modify: `docs/handbook-coupled-simulation.md` §6 (after the child-side sign paragraph)
- Modify: `docs/WRENCH_READOUT.md` (API section)
- Modify: `docs/explicit-apple-load-tcp-harvest.md` (Code table / Behavior)
- Modify: `docs/damping-tuning.md` (near joint \(k_d\) / stem_apple)

**Interfaces:** none

- [ ] **Step 1: Edit docs**

H1 §6, insert after the child-side / no-negation sentence:

> Stem harvest gather uses `include_penalty_damping=False`: the wrench written
> to `proxy_forces[tcp]` (and lagged into `body_f` / gym `ft_wrist`) is
> \(kC+\lambda\) plus explicit apple \(mg\) (and optional inertia), **not**
> the AVBD penalty-weld \(k_d\dot C\). VBD still uses `joint_penalty_kd` in
> the solve. Debug / woody `gather_joint_wrench_child_com_device` defaults
> keep damping on.

`WRENCH_READOUT.md`: note `include_penalty_damping` (default True for this
debug helper; harvest passes False).

`explicit-apple-load-tcp-harvest.md` Code table: harvest functions pass
`include_penalty_damping=False`.

`damping-tuning.md`: one sentence that fixture weld \(k_d\) remains the VBD
constraint damper and is not applied as TCP plant load.

Add the spec to H1 archive-spec list if that table is the house style for
landed slices.

- [ ] **Step 2: Commit**

```bash
git add docs/handbook-coupled-simulation.md \
  docs/WRENCH_READOUT.md \
  docs/explicit-apple-load-tcp-harvest.md \
  docs/damping-tuning.md
git commit -m "$(cat <<'EOF'
Document stem harvest omitting penalty-weld damping.

EOF
)"
```

---

## Self-review

1. **Spec coverage:** Flag + cached zeros, harvest (device+CPU) `False`, woody default `True`, harvest-parity helper, identity test, existing sign/mg tests, no Newton patch, first-pull out of scope — all tasked.
2. **Placeholders:** none.
3. **Types:** `include_penalty_damping: bool = True` is the same name in gather, NumPy helper, and harvest call sites.
