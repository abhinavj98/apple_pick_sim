# Real post-grasp settle → grasp → short settle viewer

| Field | Value |
| ----- | ----- |
| **Status** | Draft (brainstorming approved 2026-07-24) |
| **Date** | 2026-07-24 |
| **Scope** | Extend pre-grasp settle viewer: long settle → TCP-anchored grasp snap → short settle; FR3 milestone |
| **Implements first** | Slice A — proxy-only grasp (no FR3) |
| **Spec also covers** | Slice B — FR3 using the same grasp plan (implement later) |

Related: `docs/superpowers/specs/2026-07-24-real-pre-grasp-settle-viewer-design.md`,
`docs/real-sysid-pre-post-grasp-fixes.md`, `robot_replay/README.md`,
`docs/digital-twin.md`.

## Problem

The plant-only viewer rebuilds from `pre_grasp_geometry`, settles, and visualizes.
Real episodes also carry `post_grasp_geometry` (grasped TCP + apple). We need a
path that:

1. Settles the rebuilt plant under gravity (as today).
2. Applies a **grasp** that matches the real robot’s TCP pose, with the apple on
   the catalog surface relative to that TCP.
3. Runs a **short** post-grasp settle.
4. Later attaches an FR3 using the same grasp inputs (slice B).

Woody↔apple mismatch after the apple snap is intentional: it is a sys-ID /
PCD residual clue for CMA-ES.

## Pipeline

```text
pre_grasp → FruitingSystemParams + free proxy scene
         → long settle (visible; quiet every N)
         → GRASP (slice A):
              warn if ||tcp−apple_meas| − r| > 0.02 m
              proxy ← tcp_pose_4x4  (pos + quat)
              apple quat ← apple_pose_4x4
              apple_welded_pos ← tcp − r · normalize(tcp − apple_meas)
              warn if |apple_welded_pos − apple_meas| > 0.02 m
              woody bodies unchanged (post–long-settle)
              fix/weld proxy to apple
         → short settle (visible; quiet every N)
         → continue simulating
         → [slice B] FR3 at same grasp plan
```

## Grasp plan (locked math)

Let:

- \(\mathbf{p}_{\mathrm{tcp}}, R_{\mathrm{tcp}}\) from `post_grasp_geometry.tcp_pose_4x4`
  (translation must match `tcp_pos`)
- \(\mathbf{p}_{\mathrm{apple}}^{\mathrm{meas}}, R_{\mathrm{apple}}\) from
  `apple_pose_4x4` / `apple_pos`
- \(r\) = catalog apple radius from `pre_grasp_geometry.parts.apple.radius_m`
  (same radius used to build the sim apple)

```text
ŵ = normalize(p_tcp − p_apple_meas)     # apple → TCP  (weld / approach side)
p_apple_welded = p_tcp − r · ŵ          # force |TCP − apple| = r
```

| Quantity | Role |
| -------- | ---- |
| Proxy pose | Exactly measured TCP (`tcp_pose_4x4`) |
| Apple orientation | Measured `apple_pose_4x4` quat |
| Apple position | `p_apple_welded` (not raw measured COM) |
| Woody | Unchanged after long settle |

### Contract vs data bug

**Intended collection contract:** at post-grasp,
\(|\mathbf{p}_{\mathrm{tcp}}-\mathbf{p}_{\mathrm{apple}}|\approx r\).

On `s00-d00.parquet`, measured distance is **~21.9 mm** vs catalog **r = 40 mm**
(residual **~18.1 mm**). Treat as an upstream bug; do **not** silently trust
coincidence. Viewer **warns** when residual or apple shift exceeds **2 cm**, then
continues.

Both warning thresholds use the same tolerance **0.02 m**:

1. \(\big|\,|\mathbf{p}_{\mathrm{tcp}}-\mathbf{p}_{\mathrm{apple}}^{\mathrm{meas}}|-r\,\big| > 0.02\)
2. \(\lvert\mathbf{p}_{\mathrm{apple}}^{\mathrm{welded}}-\mathbf{p}_{\mathrm{apple}}^{\mathrm{meas}}\rvert > 0.02\)

(On the pure chord correction, (2) equals (1) in magnitude.)

## Approach

**Rebuild / apply-after-settle on one CLI** (chosen):

- Extend `robot_replay/example_view_pre_grasp_settle.py` with `--grasp-after-settle`.
- Library helper builds an immutable **grasp plan** from parquet metadata + \(r\).
- After long settle, apply apple `body_q` + welded proxy (rebuild welded cable
  scene and seed woody/`body_q` from the free settle snapshot where required by
  Newton FIXED-joint setup — prefer existing settle-then-weld / seed helpers).
- Default without the flag remains settle-only (backward compatible).

### Rejected for slice A

- Weld-at-build then settle (proxy mass affects free settle).
- In-place FIXED joint mid-run without a clear seed path.
- Teleporting woody to post-grasp chords (defeats the intentional residual cue).

## Architecture

### Library

New module `apple_pick_sim/system_id/real_post_grasp_plan.py`:

- Load `post_grasp_geometry` from dataset metadata (reuse
  `load_dataset_metadata`).
- Parse poses (`pose_4x4` → pos + quat `(x,y,z,w)` consistent with Newton).
- Compute `weld_direction`, `apple_welded_pos`, diagnostics.
- Emit warnings (stderr / `warnings.warn`) for threshold breaches; never raise
  for the 2 cm gates alone.
- Hard-fail on missing fields, non-finite values, or zero-length `tcp − apple`.

Invariant: \(\mathbf{p}_{\mathrm{apple}}^{\mathrm{welded}} + r\,\hat{w} = \mathbf{p}_{\mathrm{tcp}}\),
so a catalog-surface weld along \(\hat{w}\) lands on the measured TCP position.
Proxy orientation is taken from `tcp_pose_4x4` (not only the approach axis).

### CLI flags (slice A)

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--grasp-after-settle` | off | Enable grasp + short settle |
| `--settle-substeps` | 5000 | Long free settle |
| `--post-grasp-settle-substeps` | 500 | Short settle after grasp |
| `--settle-quiet-every` | 300 | Quiet cadence (both phases) |
| `--tcp-radius-warn-m` | 0.02 | Warn tolerance for \|d−r\| and apple shift |

Existing pre-grasp flags (`--parquet`, `--fixture`, `--strict`, `--dump-params`,
viewer) unchanged. Optional: extend `--dump-params` / a `--dump-grasp-plan` to
write the grasp plan JSON.

### Runtime phases

1. Build free `generate_coupled_cable_scene(..., fix_to_apple=False)` from pre-grasp.
2. Long settle (current visible settle loop + quiet).
3. If `--grasp-after-settle`: compute plan; apply apple pose; attach proxy at TCP
   welded to apple (`GripperProxyConfig(fix_to_apple=True, weld_direction=ŵ,
   weld_reference_pos/quat from plan)` as needed by build APIs).
4. Short settle + quiet.
5. Continue VBD simulation in the viewer.

## Slice B (spec only; implement after A)

- Same grasp plan object is the single source of truth.
- CLI growth: e.g. `--robot fr3` after grasp/short settle builds coupled FR3 and
  seeds from the grasped cable state (settle-then-weld / IK patterns already in
  `coupled_fruiting`).
- No new weld direction math; no trajectory replay in this milestone.

## Testing

- Unit tests for grasp-plan math (synthetic TCP/apple/r).
- Unit tests that residuals of 18 mm vs r=40 mm trigger warn helpers (capture
  warnings); residuals within 2 cm do not.
- Pose parse: `apple_pose_4x4` / `tcp_pose_4x4` → quat + translation consistency
  with `*_pos` fields.
- Optional headless smoke:
  `--grasp-after-settle --settle-substeps 50 --post-grasp-settle-substeps 20
  --viewer null --num-frames …`

## Non-goals

- Replaying episode `action` / pull trajectory.
- CMA-ES or writing full `batched_sysid_v1` datasets.
- Fixing upstream TCP–radius collection (document + warn only).
- Implementing FR3 in the first coding slice.

## Success criteria

- With `--grasp-after-settle`, viewer shows free settle, then grasped proxy at
  logged TCP with apple on the r-sphere along apple→TCP, then a short settle.
- Woody not teleported to post-grasp chords.
- Warnings fire appropriately on `s00-d00` (~18 mm residual) but run continues.
- Settle-only path unchanged when the new flag is off.
- Grasp plan reusable for slice B without changing formulas.

## Open implementation notes (not open design)

- Exact mechanism to “apply” welded proxy after free settle (in-place seed vs
  rebuild welded `CoupledCableScene` + copy woody `body_q`) — choose the
  smallest path that matches existing Newton/FIXED-joint constraints during
  implementation; behavior above is normative.
- Quat convention must match Newton `body_q` `(x,y,z,w)`.
