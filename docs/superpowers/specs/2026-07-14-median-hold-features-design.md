# Median hold features + named Sinkhorn gates

Date: 2026-07-14  
Branch: `feature/sysid-stable-collect`

## Gates

| Gate | Features |
|------|----------|
| `gate_median_hold` | Full-hold median hold→hold bags; per-direction Sinkhorn; paired per-hold median MSE; `--use-median` |
| `gate_hold_id` | + `hold_number` one-hot |
| `gate_pooled_dirs` | + pool bags across directions for Sinkhorn |

Pass bar: GT Sinkhorn `gt_rank <= 2`, GT not disqualified. Geometry: `total_movement_m=0.08`, `movement_per_step_m=0.01`, 5 structures × 5 directions × seeds 0,1,2.

## CLI

- `--use-median` / `--no-use-median` (default on)
- `--hold-id-onehot` / `--pool-directions`
- `--score-json-output` for gate harness

## Harness

```bash
bash scripts/gate_sysid_gt_sinkhorn.sh --gate gate_median_hold
bash scripts/gate_sysid_gt_sinkhorn.sh --gate gate_hold_id
bash scripts/gate_sysid_gt_sinkhorn.sh --gate gate_pooled_dirs
bash scripts/compare_sysid_gates.sh \
  --gate_median_hold <dir1> --gate_hold_id <dir2> --gate_pooled_dirs <dir3>
```
