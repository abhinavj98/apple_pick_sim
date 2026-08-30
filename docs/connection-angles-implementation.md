# Catalog connection angles (implementation)

How real-episode `manual_spur_angle_deg` / `manual_stem_angle_deg` become
rod directions. The checklist for a converted episode lives in
**H4** `docs/handbook-real-replay.md` (Checking connection angles).

## Behavior summary

When both catalog angles are present, plant rebuild uses them as ground
truth and **does not** use woody marker chords for spur/stem direction.

Proxy world: primary \(+\mathrm{X}\), robot reach \(+\mathrm{Y}\), hang
\(-\mathrm{Z}\). Fruiting→robot is \(-\mathrm{Y}\).

1. Spur rest is the horizontal T-junction, \(\hat{t}_{\mathrm{primary}} \times (-\hat{z})\)
   (\(+\mathrm{Y}\) when primary is \(+\mathrm{X}\)).
2. Clock the spur about the **primary** by \(-\theta_{\mathrm{spur}}\)
   (right-hand). \(\theta_{\mathrm{spur}}=90^\circ\) hangs to \(-\mathrm{Z}\).
3. Lean the stem about **fruiting→robot** (\(-\mathrm{Y}\)) by
   \(+\theta_{\mathrm{stem}}\). \(\theta_{\mathrm{stem}}=60^\circ\) after a
   90° hang yields \((\sin 60,\, 0,\, -\cos 60)\).

World \(Z\) is gravity. After a vertical hang, a rotation about world \(Z\)
cannot produce a 60° stem.

Owner: `rod_directions_from_manual_catalog_angles` in
`apple_pick_sim/system_id/real_pre_grasp_params.py`. Diagnostics
(`built_spur_stem_angle_deg`, `chord_spur_stem_angle_deg`) are printed by
`format_pre_grasp_diagnostics` from the settle viewer.

## Tests

- `test_rod_directions_from_manual_catalog_angles_clock_then_lean` — 90/60
  hang + lean, 60° between rods.
- `test_rod_directions_zero_stem_angle_stays_on_spur` — 0° stem stays on spur.
- `test_format_pre_grasp_diagnostics_reports_connection_angles` — printable
  checklist fields.
- `test_real_pre_grasp_params_smoke` — s09 parquet built angle matches catalog.

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_real_pre_grasp_params.py \
  -k "catalog_angles or connection_angles or smoke" -q
```
