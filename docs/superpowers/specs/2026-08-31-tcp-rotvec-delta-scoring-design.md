# TCP rotvec + 30 Hz delta scoring

| Field | Value |
| ----- | ----- |
| **Status** | Implemented |
| **Date** | 2026-08-31 |
| **Roadmap** | M4.0 CMA scoring enrichment on `feature/real-replay-parallel-sysid` |
| **Related** | H3 `docs/handbook-sysid-scoring.md`; H5 `docs/handbook-youngs-cma.md` |

## Purpose

Add TCP orientation and 30 Hz transients to Sinkhorn `STATE_VECTOR` so CMA-ES can discriminate attitude mismatch and hold-frame dynamics, not only wrench / woody / bend levels.

## Locked decisions

| Topic | Choice |
| ----- | ------ |
| Bag storage | Keep `tcp_quat` xyzw in `batched_sysid_v1` parquet (unchanged) |
| Score encoding | Frame-0 relative 3D rotvec \(\phi_t = \log(q_0^{-1} q_t)\) after hemisphere align (\(q_t \cdot q_0 \ge 0\)) |
| Scored field name | `tcp_rotvec` (3 columns after `tcp_pos`) |
| Physical scale | `0.05` rad per component (same as bend angles) |
| Delta | Reuse existing `[s, \Delta s]` with `hold_aggregation=none`; consecutive stable 30 Hz hold frames |
| CMA default | `--hold-aggregation none --include-delta --categorical-weight 100`; `--no-include-delta` restores level bags |
| Resampling | None — bags already at `control_hz=30` |
| Not scored | `apple_quat`, raw xyzw, geodesic \(\Delta\phi\) special case |

## `STATE_VECTOR` width (J=2)

| Offset | Field | D |
| ------ | ----- | - |
| 0:6 | `ft_wrist` | 6 |
| 6:12 | `tcp_velocity` | 6 |
| 12:15 | `tcp_pos` | 3 |
| 15:18 | `tcp_rotvec` | 3 |
| 18:24 | `woody_part_start_pos` | 6 |
| 24:26 | `woody_bending_angles` | 2 |

General \(D_s = 18 + 4J\) (was \(15 + 4J\)). With `include_delta=True`, transition rows are width \(2 D_s\) plus hold/direction one-hots.

## Code touchpoints

- `apple_pick_sim/system_id/mmd_features.py` — `build_tcp_rotvec`, `STATE_VECTOR_FIELDS`, collector
- `apple_pick_gym/batched_envs/obs_torch.py` — batched replay download includes `tcp_quat`
- `apple_pick_gym/cma_generation_persist.py` — persist HTML slices width 26
- `apple_pick_gym/batched_envs/batched_sysid_cmaes.py` — `include_delta` default `True`

## Validation

```bash
uv run --env-file pytest.env pytest \
  apple_pick_sim/tests/test_mmd_features.py \
  apple_pick_sim/tests/test_mmd.py \
  apple_pick_sim/tests/test_wasserstein.py \
  apple_pick_gym/tests/test_cma_generation_persist.py \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py \
  apple_pick_gym/tests/test_batched_sysid_mmd_grid_helpers.py -q
```
