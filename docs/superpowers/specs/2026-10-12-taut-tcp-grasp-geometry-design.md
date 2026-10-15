# Taut-at-grasp geometry from the logged TCP (`--taut-tcp`)

| Field | Value |
| ----- | ----- |
| **Date** | 2026-10-12 |
| **Status** | Implemented; s02 seed-56 gate pending |
| **Related** | `--apple-from-tcp` (median radius, 12.3 mm standoff); post-grasp pretension (2026-09-05) |

## Problem

The prescribed-TCP sweep (2026-10-09) was stopped for geometry bias. The plant was
built from the lengthened (metrology) T junction, about 16 mm above the during-run one, and
the apple radius was closed against the camera apple marker. That marker sits 16–45 mm
off the TCP-derived apple on s02–s04. Each tree has one plant, but every direction is
grasped at a different chain configuration (apple positions spread 4–37 mm while
segment lengths stay within about 1–3 mm), so a single settled hang cannot match all grasps.

## Decisions

- **Trusted:** the catalog rod lengths and radii, the manual catalog spur and stem angles (the stress-free
  rest shape), the during-run markers for T and the spur direction, and the logged robot TCP.
  The apple marker is not used.
- **Apple CoM at grasp:** `tcp + z_tool·(r + 2 mm)`.
- **Per tree, one radius:** for each direction, the spur tip = during-run T +
  catalog spur length along the T→spur-end marker direction, and the stem span =
  `stem_L + r_spur`. Solve `‖tcp + z(r+c) − tip‖ = span + r`. The tree radius is the
  **minimum** over directions, the largest radius that leaves no direction slack, so the
  other directions start stretched. The logged mass is kept and density is back-solved.
- **Base:** the mean during-run T junction, with no sag lift. The untuned primary is allowed
  to sag in the sim, which relieves stretch.
- **Build sequence:** build at rest, then free gravity settle, then weld seed, then **translate**
  the apple to the grasped CoM, keeping its settled rotation (the real apple is assumed only
  translated), then place the proxy at the logged TCP. The weld offset per world becomes
  `X_apple⁻¹ X_tcp`. After that, a plant-only welded settle (proxy prescribed) turns the gap into
  frame-0 preload. The junction anchor shift is not used.

## Code map

- `real_to_batched_sysid.export_real_tree_folder_to_batched_dataset(taut_tcp=True)`
  (CLI `--taut-tcp`): `_tcp_derived_apple_poses(radius_reduce="min",
  catalog_spur_tip=True)`, `TAUT_TCP_APPLE_STANDOFF_M`, and `prefer_lengthened_base=False`.
  Episode meta gets `grasp_transport="translate_only"` and the manifest gets
  `geometry_mode="taut_tcp"`.
- `batched_digital_twin_init.translate_apple_to_logged_grasp`. It rewrites, per world, the weld
  `joint_X_p`, the proxy rest, `scene.per_world_proxy_offsets`, and the harvest arrays. The VBD
  fixed joint's linear target is the `X_p` translation; its angular target is the rest relative
  rotation, set by `sync_weld_proxy_rest_from_apple_rest`.
- `real_batched_replay_build.make_real_replay_build_env_fn`. For translate-only metas it skips the
  anchor and the SE(3) teleport. It prints the per-env transport gap and the post-settle
  T junction minus the logged T.

## s02 numbers (convert)

r = 33.3 mm (per direction 33.3–36.1), ρ = 1423 kg/m³, standoff 2 mm.

## Gate

s02 seed 56 with the stopped run's flags (`--replay-mode tcp_prescribed --full-trajectory
--hold-aggregation none`, train 1,2,4,5,6,7, val 0,3). Compare against
`cma_s02_val03_seed56` on frame-0 force and torque, the holdout gate (direction 0 force
magnitude), and `match_metrics.json`.
