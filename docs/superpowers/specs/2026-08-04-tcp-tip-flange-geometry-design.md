# TCP tip / flange cylinder geometry (full hygiene)

| Field | Value |
| ----- | ----- |
| **Status** | Superseded — tool length is 180 mm (`EE_CYLINDER_HALF_HEIGHT=0.09`); radius/test notes remain useful |
| **Canonical living doc:** | `docs/handbook-coupled-simulation.md` |
| **Date** | 2026-08-04 |
| **Scope** | Lock tip-out TCP contract across VBD proxy, FR3 USD assets, and docs; regress tip↔TCP coincidence |
| **Related** | `2026-08-04-true-tcp-pose-weld-design.md` (post-grasp SE(3)); `docs/real-world-proxy.md` |
| **Amends** | Stale look-at / Ø200 language in README and older post-grasp docs |
| **Out of scope** | Gym / digital-twin consuming logged TCP quat (true SE(3)); mass reconciliation |

## Problem

Post-grasp replay and the VBD gripper proxy now follow tip-out TCP SE(3) with
cylinder bulk on the flange side (−Z). FR3 USD and several docs still disagree:

- `testfr3_resolved.usda`: TCP on +Z tip side, but mesh tip may not coincide with
  `/ee/tcp` after scales (~cm gap from authored mesh translate vs TCP `0.5`).
- `testfr3.usda` (authoring): TCP at large **−Z** — can regress resolved if copied.
- Docs/README: look-at **+Z ∥ ŵ**, surface snap, and tool radius **0.10 m (Ø200)**
  vs code/USD **r=0.05 (Ø100)**.

## Locked geometry contract

| Quantity | Rule |
| -------- | ---- |
| TCP / `/ee/tcp` / VBD proxy body origin | Center of **distal tip face** |
| World tip-out | Away from link7 (flange → tip). USD: `fr3_joint8` ~RotX(180) ⇒ tip-out = **ee −Z** |
| Recorded / VBD local **+Z** | Tip-out in the TCP frame (proxy bulk on −Z from tip) |
| Distal cylinder face | Flush with TCP (same point as tip-face center) |
| Proximal cylinder face | Meets **link7 visual mesh** end (~6 mm past `ee` origin / `fr3_joint8`, which is beyond the mesh) |
| Cylinder bulk | Between proximal (link7 mesh) and tip; length 0.14 m |
| Length / radius | **0.14 m** / **0.05 m** (Ø100); `hh = 0.07` |
| Post-grasp poses | Follow logged TCP + apple SE(3); **no** catalog-\(r\) surface snap |

VBD proxy (`gripper_proxy_cylinder_tcp_xform` at `(0,0,−hh)`) already matches
tip + bulk −Z. Look-at paths use tip-out (`z_axis = −approach` for exterior-pole
welds).

## Approach (chosen): full asset hygiene

1. **Fix `assets/testfr3_resolved.usda`** so mesh tip == `/ee/tcp` and proximal
   face is flange-flush; keep tip-out +Z and Ø100 × 140 mm.
2. **Fix or quarantine `assets/testfr3.usda`** so authoring cannot silently
   disagree (same tip/flange/TCP contract; remove −Z TCP).
3. **Regression guard:** test or scripted check that tip face ≈ TCP and
   length/radius match `EE_CYLINDER_*` / `GripperProxyConfig` defaults.
4. **Docs:** warnings + radius + README/supersession updates (below).

### Rejected / deferred

- Docs-only (leaves FR3 tip≠TCP).
- Resolved-only without authoring fix (regression risk).
- Gym / digital-twin true TCP SE(3) — **deferred**; document as warning only.

## Documentation changes

### Warning (both files)

Add an explicit **warning** to:

- `docs/superpowers/specs/2026-08-04-true-tcp-pose-weld-design.md`
- `docs/real-world-proxy.md`

Text intent:

> Gym, digital-twin, and generic `weld_direction` look-at welds do **not** yet
> consume a logged TCP SE(3). They use tip-out look-at (surface pole + constructed
> orientation). Only post-grasp replay (`real_post_grasp_plan` /
> `--grasp-after-settle`) uses full logged TCP pose. Do not assume look-at
> orientation matches recorded TCP quat.

### `docs/real-world-proxy.md`

- Tool radius **0.05 m (Ø100)** (fix Ø200).
- State tip = TCP, flange-side face flush with flange, +Z tip-out, bulk −Z.
- Note VBD proxy and FR3 USD must share this contract.
- Coupling wording: cylinder dims (not only `box_half_extents`).

### `robot_replay/README.md`

- Replace look-at / +Z ∥ weld / surface-snap language with true TCP SE(3) +
  follow-data.

### `docs/superpowers/specs/2026-07-24-real-post-grasp-viewer-design.md`

- Keep history; mark orientation tables obsolete; point at 2026-08-04 true-TCP
  design (no silent “+Z ∥ ŵ” as current).

## Success criteria

- [x] In resolved (and authoring) USD: distal mesh tip coincides with
      `/ee/tcp` within a tight tolerance after scales.
- [x] Proximal mesh face meets link7 visual mesh end (~6.2 mm past `ee` /
      `fr3_joint8`; joint is beyond the mesh).
- [x] Automated check (test or script) fails if tip↔TCP drifts.
- [x] `real-world-proxy.md` radius = 0.05; tip/flange/tip-out contract stated
      (including ee −Z / joint flip).
- [x] Warning present in both designated docs.
- [x] `robot_replay/README.md` no longer describes look-at post-grasp as current.
- [x] Existing tip-out proxy tests remain green.

## Validation

```bash
uv run --env-file pytest.env pytest \
  apple_pick_sim/tests/test_real_world_proxy_fixture.py \
  apple_pick_sim/tests/test_real_post_grasp_plan.py \
  apple_pick_sim/tests/test_coupled_cable_scene.py \
  -q
# plus any new USD tip↔TCP regression test once added
```
