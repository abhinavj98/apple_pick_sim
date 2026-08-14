# Support-joint \(k_p\) sys-ID (replace primary \(E\))

| Field | Value |
| ----- | ----- |
| **Status** | Implemented |
| **Canonical living doc:** | `docs/handbook-youngs-cma.md` |
| **Date** | 2026-08-04 |
| **Roadmap** | Extends V.5.2 Young's CMA/grid; primary branch compliance retarget |
| **Related** | `docs/youngs-modulus-sysid.md`, `docs/youngs-modulus-cmaes-implementation.md`, `docs/superpowers/specs/2026-07-16-youngs-modulus-cmaes-loop-design.md`, `docs/damping-tuning.md`, `docs/real-world-proxy.md` |

## Purpose

Retarget primary-branch sys-ID away from rod Young's modulus / bend–stretch
stiffness toward the **wire-tie FIXED joints** that pin the primary ends
(`joint_primary_support_left` / `right`).

On the T-junction primary, mid-span compliance under load is dominated by those
support mounts when the rod is stiff. Fitting primary \(E\) (or
`bend_stiffness`) with support \(k_p\) held fixed misattributes mount compliance
to the wood. This slice identifies **support \(k_p\)** jointly with spur and stem
\(E\), holding primary \(E\) and support \(\zeta\) fixed.

## Problem

- Current CMA/grid candidate is \(\log_{10}([E_{\mathrm{primary}}, E_{\mathrm{spur}}, E_{\mathrm{stem}}])\).
- Primary is world-supported at both ends via FIXED joints; fixture already
  exposes `sim_build.joint_*_kp_overrides["support"]` (default `10000`) and
  joint \(\zeta\) via `joint_damping_ratio`.
- Proxy tip spring constants (N/m) are **effective** mount+rod stiffness, not
  pure rod \(EI\).
- Under a stiff-primary theory, **linear** support compliance dominates mid-span
  translation; angular support \(k_p\) still affects orientation/moments. A
  shared numeric \(k_p\) on both slots is the minimal first search.

## Decisions (locked)

| Decision | Choice |
| -------- | ------ |
| Free primary material | **No** — primary \(E\) fixed from structure true/fixture params |
| Free support \(k_p\) | **Yes** — one scalar for left+right supports |
| Support DOFs | Shared numeric \(k_p\) → **angular and linear** penalty slots |
| Support \(\zeta\) | **Fixed** — not searched; taken from dataset `sim_config.joint_damping_ratio` at replay (collect/replay kd parity). Fallback 0.5 matches the proxy variance fixture. |
| Support \(k_d\) | \(k_{d,\mathrm{ang}} = \zeta\cdot 2\sqrt{k_p I}\), \(k_{d,\mathrm{lin}} = \zeta\cdot 2\sqrt{k_p m}\) with \(\zeta\) from dataset `joint_damping_ratio` (support role only; L/R may differ via child \(I\)/\(m\)) |
| Other FIXED joints | Unchanged fixture defaults / global `joint_damping_ratio` |
| Spur / stem \(E\) | Still free (same as V.5.2) |
| First delivery | Extend existing batched Young's **grid + CMA** (not real-robot-only first) |
| Approach | Replace 3-vector Young’s candidate with 3-vector `(support_kp, spur_E, stem_E)` |

## Parameter vector

CMA / continuous search:

\[
\mathbf{x} = \bigl(\log_{10}(k_p^{\mathrm{support}}),\;
\log_{10}(E_{\mathrm{spur}}),\;
\log_{10}(E_{\mathrm{stem}})\bigr)
\]

Cartesian grid: discrete `support_kp` values × spur \(E\) × stem \(E\) (drop
primary-\(E\) axes).

### Apply semantics (per candidate)

1. Start from structure base params (`true_params_for_structure` / recorded
   `fruiting_system_params`); **do not** change primary \(E\).
2. `set_rod_youngs_modulus` for **spur** and **stem** only.
3. Patch support joints (label match `"support"` / `primary_support`):
   - angular + linear \(k_p\) ← candidate scalar
   - angular + linear \(k_d\) ← critical-damping formula with dataset \(\zeta\)
4. Leave all non-support FIXED joint penalties at build/fixture values.

### Initialization and bounds

- Same V.5.2 policy: search-box midpoints (or explicit start mean), **not** GT.
- Spur/stem \(\log_{10} E\) bounds: reuse existing CMA search-box defaults unless
  retuned.
- Support \(k_p\): log10 search box centered on current fixture override
  (`10000`) with a wide band suitable for mount compliance (exact numeric
  bounds set in the implementation plan; must be documented in CLI help and
  `CMA_SEARCH_PARAMS`-style constants).

## Architecture

Reuse the V.5.2 library boundary; change the candidate phenotype only:

| Piece | Role |
| ----- | ---- |
| Candidate type | Evolve or replace `YoungsModulusCandidate` → e.g. `SupportKpYoungsCandidate(support_kp, spur, stem)` with log10 maps |
| Apply helper | New/extended apply: support \(k_p\)/\(k_d\) patch + spur/stem \(E\) |
| `batched_sysid_cmaes.py` | Orchestration, ask/tell, fused eval waves — dim 3 with new meaning |
| Grid CLI | Drop `--primary-*-values`; add `--support-kp-values` (or equivalent) |
| CMA CLI | Drop primary log10 dim; add `log10(support_kp)` bounds / start |
| Reports / gates | Primary-\(E\) GT error → support-\(k_p\) error vs recorded `sim_build` overrides when present; keep spur/stem \(E\) columns |
| Scoring / replay | Unchanged Sinkhorn features, fused multi-structure replay, instability policy |

```text
ask() per structure
  -> map to SupportKpYoungsCandidate
  -> fused replay (chunked)
  -> pooled Sinkhorn fitness
  -> tell()
final-mean wave + overlays
```

## Non-goals

- Free support \(\zeta\) (held fixed from `sim_build.joint_damping_ratio` for collect/replay parity; not searched).
- Separate free angular vs linear \(k_p\).
- Fitting primary \(E\), bend/stretch rod stiffness, density, geometry, or
  secondary \(E\).
- Changing Wasserstein / transition features or soft-disable policy.
- Real-data acceptance as the gate for this slice (sim-to-sim first, same as
  V.5.2).
- Updating DR fixture ranges from fitted support \(k_p\) (report only).
- Two-stage (support-then-\(E\)) optimization.

## Error handling and invariants

- Support label must match both left and right joints; fail closed if `"support"`
  matches nothing or ambiguously matches non-support joints under the existing
  substring rules.
- \(\zeta \ge 0\) from dataset (or fallback 0.5) and \(k_p > 0\); reject non-positive \(k_p\).
- \(I\) / \(m\) for \(k_d\) come from the child body at each support joint (same
  pattern as existing fixture \(\zeta \cdot 2\sqrt{k\cdot I}\) expansion in
  damping-tuning / CMA joint-kd hold).
- Shared numeric \(k_p\) on angular (N·m/rad) and linear (N/m) slots is an
  intentional pragmatic choice (already used in the variance fixture); do not
  claim dimensional equivalence beyond “one search knob.”
- All-invalid generation / scalar fallback behavior stays as in V.5.2 CMA.

## Tests (acceptance intent)

- Unit: candidate log10 maps; apply sets support angular+linear \(k_p\), sets
  support \(k_d\) for dataset \(\zeta\), leaves primary \(E\) and non-support joints
  unchanged; sets spur/stem \(E\).
- CLI: grid/CMA help and arg parsing expose support \(k_p\), not primary \(E\).
- Report: support \(k_p\) present; primary \(E\) not a fitted dimension.
- Existing fused replay / Sinkhorn path still runs on a tiny smoke
  (CPU or CUDA per roadmap validation norms).

## Verification (post-implement)

**Primary acceptance = sim-to-sim transfer in two complementary paths** (same
spirit as V.5.2 Young’s collect → grid / CMA). Unit/CLI tests are required but
not sufficient.

### Path 1 — Collect new GT, then Cartesian grid

1. Generate a fresh batched sys-ID dataset with known GT support \(k_p\) (via
   fixture / `sim_build` overrides) and known spur/stem \(E\), primary \(E\)
   held stiff and fixed.
2. Run the updated grid command over `support_kp` × spur \(E\) × stem \(E\).
3. Pass criterion: GT (or nearest grid node) ranks first / within the project’s
   existing ranking-gate majority policy on healthy samples; inspect overlay /
   report for support \(k_p\) recovery quality.

### Path 2 — PyCMA fit on the same (or held-out) collect

1. Run the updated CMA-ES entry on the collected dataset.
2. Score / report final-mean vs recorded GT for
   \((k_p^{\mathrm{support}}, E_{\mathrm{spur}}, E_{\mathrm{stem}})\); use
   overlays and fit-integrity artifacts analogous to
   `gate_youngs_modulus_cmaes.sh`.
3. Pass criterion: CMA integrity gate green; support \(k_p\) and spur/stem \(E\)
   recover within documented tolerances / qualitative “looks right” on overlays
   (exact numeric thresholds set in the implementation plan / gate scripts).

### Also required

- Focused pytest for candidate/apply + CLI parsing (no primary-\(E\) fit dim).
- README / ROADMAP commands updated to the collect → grid and collect → CMA
  recipes above.

## Open for implementation plan (not design blockers)

- Exact default `log10(support_kp)` search bounds and start mean.
- Whether to rename public types/CLIs in place or add parallel
  `support_kp_*` entry points and deprecate primary-\(E\) flags.
- Gate script renames vs in-place semantic update of Young’s gates.
