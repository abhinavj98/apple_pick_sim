# Split flexural (bend) and Young's (axial) moduli

| Field | Value |
| ----- | ----- |
| **Status** | Draft — awaiting review before implementation |
| **Canonical living docs after impl:** | `docs/material-parameter-sampling.md`, `docs/damping-tuning.md`, `docs/handbook-youngs-cma.md` |
| **Date** | 2026-08-30 |
| **Roadmap** | User-requested material-schema change on `feature/real-replay-parallel-sysid`. **Not** M4.0 current-focus (torque-magnitude science gate). Do not treat this as closing Task 9. |
| **Related** | `docs/material-parameter-sampling.md`, `docs/damping-tuning.md`, `docs/handbook-youngs-cma.md`, `docs/superpowers/specs/2026-08-04-support-joint-kp-sysid-design.md` |

## Purpose

Stop using one `youngs_modulus_pa` for bending and a `vbd_stretch_force` budget for stretch. Store two material moduli:

- **`flexural_modulus_pa`** — today's bend \(E\) (literature / sys-ID wood–peduncle bending)
- **`youngs_modulus_pa`** — axial \(E\), **same numeric range as today's `youngs_modulus_pa`** (not converted from \(F_{\max}\))

Newton CABLE joints stay isotropic springs. This is **not** a full orthotropic constitutive model (no \(E_R\), \(G\), or separate torsion). It is a named split of the two existing CABLE knobs.

## Problem

- `youngs_modulus_pa` currently drives \(k_{\mathrm{bend}}=EI/L_{\mathrm{seg}}\) and, unless overridden, \(k_{\mathrm{stretch}}=EA/L_{\mathrm{seg}}\).
- The variance fixture overrides stretch with \(k=F_{\max}/(0.05 L_{\mathrm{seg}})\), so axial stiffness is a force budget, not a modulus.
- CMA / grid treat `youngs_modulus_pa` as **bend** \(E\) (`set_rod_youngs_modulus` + stretch-preserve). Renaming without retargeting apply-semantics would silently fit the wrong physics.

## Locked decisions (do not reopen)

| Topic | Choice |
| ----- | ------ |
| Bend modulus JSON/field | `flexural_modulus_pa` ← copy of today's `youngs_modulus_pa` bands |
| Axial modulus JSON/field | `youngs_modulus_pa` ← **same bands as flexural** (copy today's `youngs_modulus_pa`) |
| Stretch-force JSON | **Reject** `vbd_stretch_force` (and keep rejecting `vbd_stretch_fixed`) |
| Fixture conversion | No \(F_{\max}\to E\) formula. Every fixture that has `youngs_modulus_pa` today duplicates that band onto both keys. |
| Damping | One `damping_ratio` \(\zeta\) drives **both** bend and stretch |
| Primary moduli | **Fixed** (flexural and axial) from structure / fixture |
| CMA phenotype | 5D: support \(k_p\) + spur/stem flexural + spur/stem youngs |
| Cartesian grid | **Stays 3D**: \(k_p\) × flex-spur × flex-stem; axial \(E\) left at structure values |
| Old 3-vector CMA helper | Still accepted: length-3 log10 vector sets flexural only (axial `None` → unchanged) |
| Shared \(\zeta\) side effect | Variance fixture stretch \(\zeta\) was 1.0; becomes bend \(\zeta=0.3\) |

### Fixture conversion (one-time)

For every rod segment in every range JSON that currently has `youngs_modulus_pa`:

1. Rename/copy that band to `flexural_modulus_pa` (values unchanged).
2. Write the **same** `{min, max}` onto `youngs_modulus_pa`.
3. Delete `vbd_stretch_force` if present.

Variance fixture (and any other force-budget fixture) therefore does **not** preserve old stretch \(k=F_{\max}/(0.05 L_{\mathrm{seg}})\). Stretch becomes beam \(EA/L_{\mathrm{seg}}\) at wood/peduncle-scale \(E\). That is much stiffer than the 35 N / 750 N budgets. Primary flexural ~50 GPa is copied as axial too; retuning that 10× vs 5 GPa test expectation remains **out of scope**.

Keep `VBD_STRETCH_EXTENSION_FRACTION = 0.05` and `stretch_knobs_from_max_force` as a **unit-test helper** only; range JSON must not accept `vbd_stretch_force`. Do not use them to populate fixture \(E\).

## Constitutive mapping (sample time)

`RodParams` stores both moduli. `rod_params_from_material` takes both (plus shared \(\zeta\) and geometry):

\[
A=\pi r^{2},\quad I=\pi r^{4}/4,\quad L_{\mathrm{seg}}=L/N
\]

\[
k_{\mathrm{bend}}=\frac{E_{\mathrm{flex}} I}{L_{\mathrm{seg}}},\quad
c_{\mathrm{bend}}=2\zeta\sqrt{k_{\mathrm{bend}} J_{\mathrm{seg}}}
\]

\[
k_{\mathrm{stretch}}=\frac{E_{\mathrm{youngs}} A}{L_{\mathrm{seg}}},\quad
c_{\mathrm{stretch}}=2\zeta\sqrt{k_{\mathrm{stretch}} m_{\mathrm{seg}}}
\]

No stretch-stiffness override kwargs from range JSON. Optional explicit stretch kwargs may remain on the Python helper for unit tests only.

Tier constraint when primary and secondary are both enabled: **`primary.flexural_modulus_pa >= secondary.flexural_modulus_pa`**. No extra ordering on axial `youngs_modulus_pa`.

Setters (breaking semantics on the old name):

- `set_rod_flexural_modulus(params, segment, E_flex)` — re-derive **bend** knobs; freeze geometry, \(\zeta\), and axial knobs / `youngs_modulus_pa`.
- `set_rod_youngs_modulus(params, segment, E_youngs)` — re-derive **stretch** knobs; freeze geometry, \(\zeta\), and bend knobs / `flexural_modulus_pa`.

Today `set_rod_youngs_modulus` means bend-and-maybe-preserve-stretch. After this spec it means **axial only**. All current call sites that intend bend must switch to `set_rod_flexural_modulus`.

`set_rod_bend_stiffness` back-computes `flexural_modulus_pa` from \(k_{\mathrm{bend}} L_{\mathrm{seg}}/I\), not `youngs_modulus_pa`.

## Range JSON contract

Per enabled rod segment, required keys:

`num_segments`, `length`, `radius`, `density`, `flexural_modulus_pa`, `youngs_modulus_pa`, `damping_ratio`, plus existing angle keys.

Reject: `vbd_stretch_force`, `vbd_stretch_fixed`, legacy `bend_stiffness` / `stretch_stiffness` / `bend_damping`. Error text must name the replacement keys (`flexural_modulus_pa` + `youngs_modulus_pa`).

## Episode serialization

Bump `FRUITING_SYSTEM_PARAMS_SCHEMA` to `fruiting_system_params_v3`.

Write both moduli on each rod row. Fingerprint adds `*_flexural_modulus_pa` and keeps `*_youngs_modulus_pa` with the **new** (axial) meaning.

Read:

| On-disk schema | Bend \(E\) | Axial \(E\) |
| --- | --- | --- |
| v3 | `flexural_modulus_pa` | `youngs_modulus_pa` |
| v2 | `youngs_modulus_pa` (old meaning) | \(k_{\mathrm{stretch}} L_{\mathrm{seg}}/A\) from stored stretch stiffness |
| v1 | \(k_{\mathrm{bend}} L_{\mathrm{seg}}/I\) | \(k_{\mathrm{stretch}} L_{\mathrm{seg}}/A\) |

v1/v2 remain read-only. Missing v3 `flexural_modulus_pa` on a v3 row is an error (do not guess).

## CMA / grid phenotype

`SupportKpYoungsCandidate`:

```text
support_kp: float
spur: float          # flexural, same slot as today's spur E
stem: float          # flexural, same slot as today's stem E
spur_youngs: float | None = None
stem_youngs: float | None = None
```

`apply_to`:

1. `set_rod_flexural_modulus` on spur and stem when those rods exist.
2. If `spur_youngs` / `stem_youngs` is not `None`, `set_rod_youngs_modulus` on that rod.
3. Do not change primary (or secondary) moduli.
4. `support_kp` still applied only by fused replay support-penalty patching.

CMA log10 vector (length **5**):

\[
\mathbf{x}=\bigl(
\log_{10} k_p,\;
\log_{10} E_{\mathrm{flex,spur}},\;
\log_{10} E_{\mathrm{flex,stem}},\;
\log_{10} E_{\mathrm{youngs,spur}},\;
\log_{10} E_{\mathrm{youngs,stem}}
\bigr)
\]

Length-3 log10 vector remains valid and maps to `(kp, flex_spur, flex_stem)` with axial `None` (grid / old tests). Length other than 3 or 5 raises.

Cartesian grid: still `iter_support_kp_youngs_candidates(support_kp, spur, stem)` — 3-field candidates, axial `None`, structure axial \(E\) unchanged. Do **not** add a 5-way product.

### Bounds (axial copies flexural)

Append two axial slots that **duplicate** that run's spur/stem flexural box. Do not introduce a separate MPa-scale axial box.

Sim-sim default (today `[2,8,8]…[6,11,11]`):

| Index | Quantity | log10 lower | log10 upper |
| --- | --- | --- | --- |
| 0 | support \(k_p\) | 2 | 6 |
| 1 | spur flexural | 8 | 11 |
| 2 | stem flexural | 8 | 11 |
| 3 | spur youngs | 8 | 11 |
| 4 | stem youngs | 8 | 11 |

Real `vic_pose` (today `[2,8,6]…[6,11,8]`):

| Index | Quantity | log10 lower | log10 upper |
| --- | --- | --- | --- |
| 0 | support \(k_p\) | 2 | 6 |
| 1 | spur flexural | 8 | 11 |
| 2 | stem flexural | 6 | 8 |
| 3 | spur youngs | 8 | 11 |
| 4 | stem youngs | 6 | 8 |

Init mean = per-box midpoint (not GT), same policy as V.5.2.

`YoungsModulusCandidate` (legacy 3-E grid: primary/spur/stem) must call `set_rod_flexural_modulus`, not the new axial `set_rod_youngs_modulus`.

## Docs to update at implementation (not this spec)

- `docs/material-parameter-sampling.md` — two moduli; delete force-budget as the range contract; fix the wrong cantilever inverse \(E=3k L^{3}/(\pi r^{4})\) to \(E=k L^{3}/(3I)=4k L^{3}/(3\pi r^{4})\) while touching that section
- `docs/damping-tuning.md` — three-layer table: bend = flexural; stretch = youngs; joints unchanged
- `docs/handbook-youngs-cma.md` — 5D CMA, 3D grid, setter split
- `docs/real-world-proxy.md` — variance fixture stretch is beam \(EA/L\), not \(F_{\max}\)
- `docs/ROADMAP.md` — one line under known/follow-up that this schema landed; do not claim M4.0 science gate
- Fixture `_comment` fields

## Tests (behavior, TDD at impl)

`apple_pick_sim/tests/test_fruiting_system.py` and `test_real_world_proxy_fixture.py`:

- `rod_params_from_material` with distinct \(E_{\mathrm{flex}}\) and \(E_{\mathrm{youngs}}\) → \(k_{\mathrm{bend}}=E_{\mathrm{flex}}I/L_{\mathrm{seg}}\), \(k_{\mathrm{stretch}}=E_{\mathrm{youngs}}A/L_{\mathrm{seg}}\), shared \(\zeta\)
- `set_rod_flexural_modulus` does not change stretch knobs / axial \(E\)
- `set_rod_youngs_modulus` does not change bend knobs / flexural \(E\)
- `load_ranges` requires both modulus keys; rejects `vbd_stretch_force`
- Variance fixture: `flexural_modulus_pa` **and** `youngs_modulus_pa` equal today's `youngs_modulus_pa` bands; no `vbd_stretch_force`
- v3 round-trip; v2 episode: old `youngs_modulus_pa` → flexural, axial back-computed from stretch
- `stretch_knobs_from_max_force` still matches \(\delta=0.05 L_{\mathrm{seg}}\) if the helper is kept (not used for fixtures)

`apple_pick_gym` CMA/grid tests:

- `candidates_from_log10_vector` length 5; length 3 still sets flexural only
- `SupportKpYoungsCandidate.apply_to` changes spur/stem flexural; axial only when `*_youngs` set
- CLI/search-box vectors length 5 for CMA path; grid CLI still 3 axes

## Non-goals

- Orthotropic stiffness tensor, separate torsion \(GJ/L\), or Timoshenko shear
- Searching primary flexural or primary youngs
- 5D Cartesian grid
- Changing VIC, support \(k_p\) apply path, scoring, or real convert
- Retuning the 50 GPa primary flexural/youngs band
- Restoring stretch \(\zeta=1.0\) as a second JSON key
- Preserving old `vbd_stretch_force` stretch \(k\) (explicitly abandoned)

## Success criteria

- Variance (and other) fixtures: per segment, `flexural_modulus_pa` min/max equals `youngs_modulus_pa` min/max equals the pre-change `youngs_modulus_pa` band
- CMA 5-vector apply changes only spur/stem flexural + spur/stem youngs + (via existing replay) support \(k_p\)
- Grid 3-vector apply does not move axial \(E\)
- Existing v1/v2 fruiting-params metadata still deserializes
- `uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_fruiting_system.py apple_pick_sim/tests/test_real_world_proxy_fixture.py apple_pick_gym/tests/test_batched_sysid_cmaes_loop.py apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py -q` green
