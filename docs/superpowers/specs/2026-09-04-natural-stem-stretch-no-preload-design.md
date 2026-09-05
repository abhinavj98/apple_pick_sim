# Natural stem stretch (no axial preload) — design

| Field | Value |
| ----- | ----- |
| **Date** | 2026-09-04 |
| **Status** | Approved for implementation |
| **Related** | Real replay pre-grasp mapping; settle → weld → post-grasp settle |

## Problem

Measurement error between catalog stem length and tracked spur→apple chord must not be absorbed into stem rest length. Building a preload-shortened rest length so tension at the chord equals \(mg\cos\theta\) makes rest length depend on mass, orientation, and Young’s modulus — different structures (and CMA \(E\) updates) get different stem lengths. That is the wrong place to put tag slop.

## Decision (approach A)

1. **Stem rest length** = catalog / GT segment length for every structure. Do **not** set `preload_chord_m` / `axial_preload_n` on real-replay stems.
2. **Apple radius** absorbs spur→CoM vs catalog-stem mismatch:  
   \(r = \|p_{\mathrm{apple}} - p_{\mathrm{spur\_end}}\| - L_{\mathrm{stem}}\), with density back-solved so apple mass stays fixed (logged `mass_kg` or catalog volume×density).
3. **Place** the radius-corrected apple at stem tip + \(r\) along the stem direction (existing fruiting build).
4. **Natural stretch** under gravity: \(\delta \approx mgL/EA\). Small; may depend on \(E\); rest length does not.

## Why settle is enough

Real replay already relaxes the cable under gravity:

1. **Pre-grasp free settle** (`settle_substeps`) — apple hangs on the stem alone; axial stretch develops before weld.
2. **Weld** (and logged post-grasp SE(3) teleport when used).
3. **Post-grasp settle** (`post_grasp_settle_substeps`) — with **`dynamic_apple=True`**, the apple stays VBD-dynamic under `fix_to_apple` (proxy prescribed; apple not co-teleported). The stem–apple chain can still extend under gravity/load sharing after the grasp teleport, so post-grasp settle is a real stretch/equilibrium solve, not just damping a kinematic jump.

Frame-0 taut + shared stem/robot load is therefore a **post-settle** equilibrium property, not a build-time rest-length trick. Brief pre-settle imbalance is acceptable.

## Out of scope

- Changing settle/post-grasp substep defaults.
- Replacing radius close with spur-length or woody-length absorption.
- Sim-sim fixtures that never used preload (unchanged).

## Code map (implementation targets)

| Area | Change |
| ---- | ------ |
| `apple_pick_sim/system_id/real_pre_grasp_params.py` | Stop writing stem `preload_chord_m` / `axial_preload_n`; keep radius solve + mass-invariant density |
| `apple_pick_sim/fruiting_system/params.py` | Keep `rest_length_for_axial_preload` helpers if still useful for tests/docs, but real path must not call them |
| `apple_pick_sim/tests/test_stem_axial_preload.py` | Drop or rewrite wiring tests that require real mapping to emit preload; keep pure math tests optional or delete with preload usage |
| `apple_pick_sim/tests/test_real_pre_grasp_params.py` | Keep radius-close + mass-invariant assertions; assert stem has no preload fields |

## Acceptance

- Mapped real stem: `length_m` = catalog; no preload pair on `rod_geometry["stem"]`.
- Solved apple radius closes \(\|spur\_end - apple\| = L_{\mathrm{stem}} + r\); mass unchanged vs catalog/logged.
- Changing stem \(E\) via `set_rod_youngs_modulus` does **not** change rest length when preload is absent.
- Existing real-replay path still: free settle → weld → post-grasp SE(3) → post-grasp settle.

## Validation

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_real_pre_grasp_params.py \
  apple_pick_sim/tests/test_stem_axial_preload.py -q
```
