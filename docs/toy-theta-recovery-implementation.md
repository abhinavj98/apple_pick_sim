# Toy theta recovery (FD gradient verification)

## Behavior summary

Toy (1) from [ROADMAP.md](ROADMAP.md): recover one scalar simulation parameter — `primary.bend_stiffness` \(k\) — from rollout features \(y(k)\) using forward finite-difference columns and Gauss–Newton with backtracking.

Two cable-only mega-plant modes:

| Mode | `fix_to_apple` | Features in loss / FD | Init |
|------|----------------|----------------------|------|
| Free proxy | `False` | 6D apple + proxy positions | Gravity sag only |
| Welded grasp | `True` | 6D stem–apple wrench block (`feat[6:12]`) | Settle-then-weld |

Each Gauss–Newton iteration rebuilds a two-column `MegaCoupledCableScene` (stiffness is baked into `joint_target_ke` at build time). Forward rollout uses `mega_vbd_substep`; the FD column uses `mega_fd_step` (one batched substep after `reset_perturbed_instances_to_nominal`). Position features require that extra substep — columns match immediately after reset.

Welded mode caches `weld_direction_in_apple_frame` from the settled free reference at \(k^*\) and reuses it for all evaluations.

## Code map

| Module | Role |
|--------|------|
| [`apple_pick_sim/identification/theta_recovery.py`](../apple_pick_sim/identification/theta_recovery.py) | `FeatureConfig`, `evaluate_at_k`, `recover_primary_bend_stiffness`, grid helper |
| [`apple_pick_sim/fruiting_system/params.py`](../apple_pick_sim/fruiting_system/params.py) | `set_rod_bend_stiffness` |
| [`apple_pick_sim/examples/toy_theta_recovery.py`](../apple_pick_sim/examples/toy_theta_recovery.py) | Headless CLI + matplotlib PNGs |

## Tests

- `apple_pick_sim/tests/test_toy_theta_recovery.py` — parametrized `fix_to_apple ∈ {False, True}`:
  - `test_theta_recovery_loss_decreases`
  - `test_theta_recovery_converges_within_tolerance` (≤ 10% relative error)
  - `test_theta_recovery_brute_force_grid_agrees`
  - `test_theta_recovery_welded_jacobian_wrench_rows_nonzero`

## How to verify

From repository root:

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest \
  ../apple_pick_sim/tests/test_toy_theta_recovery.py -q -p no:launch_testing

PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/toy_theta_recovery.py
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/toy_theta_recovery.py --fix-to-apple
```

Plots are written under `diagnostics/toy_theta_recovery/free/` and `.../welded/`.

## Success criteria

- `min(loss_hist) < loss_hist[0]` over GN iterations
- `|k_final - k*| / k* ≤ 0.10` with tuned defaults (`k0_scale` 0.95 free, 1.05 welded)
- Welded: `||J|| ≥ 0.05` on wrench slice before first GN step
