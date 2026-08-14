# Real post-grasp settle → grasp → short settle viewer

| Field | Value |
| ----- | ----- |
| **Status** | Superseded — slice A shipped, but the orientation contract was replaced by true TCP SE(3) |
| **Canonical living doc:** | `docs/handbook-real-replay.md` |
| **Date** | 2026-07-24 |
| **Scope** | Extend pre-grasp settle viewer: long settle → TCP-anchored grasp snap → short settle; FR3 milestone |
| **Implements first** | Slice A — proxy-only grasp (no FR3) |
| **Spec also covers** | Slice B — FR3 using the same grasp plan (implement later) |
| **Supersession** | Proxy orientation: use logged TCP quat (true SE(3)), not look-at +Z∥ŵ — see 2026-08-04 design |

Related: `docs/superpowers/specs/2026-07-24-real-pre-grasp-settle-viewer-design.md`,
`docs/real-sysid-pre-post-grasp-fixes.md`, `robot_replay/README.md`,
`docs/digital-twin.md`.

## Problem

The plant-only viewer rebuilds from `pre_grasp_geometry`, settles, and visualizes.
Real episodes also carry `post_grasp_geometry` (grasped TCP + apple). We need a
path that:

1. Settles the rebuilt plant under gravity (as today).
2. Applies a **grasp** with the apple on the catalog surface relative to the
   measured TCP **position**, and the gripper tool axis along the weld chord.
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
              warn if logged TCP +Z is far from ŵ (data bug; do not use logged quat)
              apple quat ← apple_pose_4x4
              apple_welded_pos ← tcp − r · ŵ
              warn if |apple_welded_pos − apple_meas| > 0.02 m
              proxy at surface: pos = apple_welded + r·ŵ (= tcp pos)
                         +Z ∥ ŵ  (look-at; tool-0 / approach axis)  [obsolete orientation]
              woody bodies unchanged (post–long-settle)
              fix/weld proxy to apple
         → short settle (visible; quiet every N)
         → continue simulating
         → [slice B] FR3 at same grasp plan
```

## Grasp plan (locked math)

Let:

- \(\mathbf{p}_{\mathrm{tcp}}\) from `tcp_pos` / `tcp_pose_4x4` translation
- \(\mathbf{p}_{\mathrm{apple}}^{\mathrm{meas}}, R_{\mathrm{apple}}\) from
  `apple_pose_4x4` / `apple_pos`
- \(r\) = catalog apple radius from `pre_grasp_geometry.parts.apple.radius_m`

```text
ŵ = normalize(p_tcp − p_apple_meas)     # apple → TCP  (weld / approach)
p_apple_welded = p_tcp − r · ŵ          # force |TCP − apple| = r
```

> **Obsolete:** orientation superseded by
> `2026-08-04-true-tcp-pose-weld-design.md` (true TCP SE(3)). Do not implement
> look-at +Z∥ŵ for post-grasp replay.

| Quantity | Role |
| -------- | ---- |
| Proxy **position** | Measured TCP position (via surface: \(\mathbf{p}_{\mathrm{apple}}^{\mathrm{welded}}+r\hat{w}\)) |
| Proxy **orientation** | **Look-at:** tool **+Z ∥ \(\hat{w}\)** (existing `weld_direction` FIXED build). **Do not** use logged `tcp_pose` rotation for the weld. |
| Apple orientation | Measured `apple_pose_4x4` quat |
| Apple position | `p_apple_welded` (not raw measured COM) |
| Woody | Unchanged after long settle |

### Orientation contract

> **Obsolete:** orientation superseded by
> `2026-08-04-true-tcp-pose-weld-design.md` (true TCP SE(3)). Do not implement
> look-at +Z∥ŵ for post-grasp replay.

**Intended real grasp:** tool approach (+Z) faces along the weld chord
\(\hat{w}\). Sim enforces that with look-at.

Logged `tcp_pose_4x4` rotation on `s00-d00` has **+Z ≈ world +Y** while
\(\hat{w}\approx(0.89,-0.39,0.25)\) (dot ≈ −0.39). Treat as a **parquet /
frame bug**; still use \(\mathbf{p}_{\mathrm{tcp}}\) for anchoring, but
**ignore** logged TCP quat for the FIXED joint. Optionally **warn** when
\(|\hat{w}\cdot(+Z_{\mathrm{tcp}})|\) is below a threshold (e.g. 0.9).

### Distance contract vs data bug

**Intended:** \(|\mathbf{p}_{\mathrm{tcp}}-\mathbf{p}_{\mathrm{apple}}|\approx r\).

On `s00-d00`, \(d\approx 21.9\,\mathrm{mm}\), \(r=40\,\mathrm{mm}\) (residual
\(\approx 18.1\,\mathrm{mm}\)). **Warn** if residual or apple shift exceeds
**0.02 m**, then continue.

1. \(\big|\,|\mathbf{p}_{\mathrm{tcp}}-\mathbf{p}_{\mathrm{apple}}^{\mathrm{meas}}|-r\,\big| > 0.02\)
2. \(\lvert\mathbf{p}_{\mathrm{apple}}^{\mathrm{welded}}-\mathbf{p}_{\mathrm{apple}}^{\mathrm{meas}}\rvert > 0.02\)

## Approach

**Rebuild / apply-after-settle on one CLI** (chosen):

- Extend `robot_replay/example_view_pre_grasp_settle.py` with `--grasp-after-settle`.
- Library helper builds an immutable **grasp plan** from parquet metadata + \(r\).
- After long settle, rebuild welded cable scene with
  `GripperProxyConfig(fix_to_apple=True, weld_direction=ŵ,
  weld_reference_pos/quat=apple welded)` — **stock look-at** (no custom FIXED
  offset from logged TCP quat).
- Seed woody from free settle; set apple to welded pose; quiet; short settle.
- Default without the flag remains settle-only.

### Rejected for slice A

- Weld-at-build then settle (proxy mass affects free settle).
- Baking logged TCP quat into the FIXED joint (inconsistent with +Z∥ŵ on bad dumps).
- Teleporting woody to post-grasp chords.

## Architecture

### Library

`apple_pick_sim/system_id/real_post_grasp_plan.py`:

- Load `post_grasp_geometry`; parse poses; compute \(\hat{w}\), welded apple,
  residuals; optional TCP-+Z alignment diagnostic.
- Warn on 2 cm gates / bad +Z∥ŵ; hard-fail on missing fields or zero chord.

Invariant: \(\mathbf{p}_{\mathrm{apple}}^{\mathrm{welded}} + r\,\hat{w} = \mathbf{p}_{\mathrm{tcp}}\).

### CLI flags (slice A)

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--grasp-after-settle` | off | Enable grasp + short settle |
| `--settle-substeps` | 5000 | Long free settle |
| `--post-grasp-settle-substeps` | 500 | Short settle after grasp |
| `--settle-quiet-every` | 300 | Quiet cadence (both phases) |
| `--tcp-radius-warn-m` | 0.02 | Warn tol for \|d−r\| and apple shift |

### Runtime phases

1. Free scene from pre-grasp.
2. Long settle.
3. Grasp: plan → welded rebuild with `weld_direction=ŵ` + apple references → seed.
4. Short settle.
5. Continue sim.

## Slice B (spec only)

Same grasp plan; FR3 later. Grasp arm seed joints from table `joint_pos` at the
post-grasp row (not inside `post_grasp_geometry`). No trajectory replay here.

## Testing

- Unit tests for plan math + warn thresholds.
- Optional warn when logged TCP +Z is misaligned with \(\hat{w}\).
- Headless smoke with `--grasp-after-settle` on `s00-d00`.

## Non-goals

- Trajectory replay; CMA-ES writer; fixing upstream collection (warn only);
  FR3 in first coding slice; custom FIXED offset from logged TCP quat.

## Success criteria

> **Obsolete:** orientation superseded by
> `2026-08-04-true-tcp-pose-weld-design.md` (true TCP SE(3)). Do not implement
> look-at +Z∥ŵ for post-grasp replay.

- Grasp places proxy at TCP **position** with **+Z ∥ \(\hat{w}\)**; apple on
  r-sphere; woody not teleported from post-long-settle.
- Warnings on `s00-d00` radius residual; run continues.
- Settle-only path unchanged without the flag.
