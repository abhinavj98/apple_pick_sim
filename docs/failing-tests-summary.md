# Failing tests summary (extended verification suite)

**Date:** 2026-06-04  
**Environment:** Linux, `uv` env under `newton/`, `PYTHONPATH` = repo root  
**Run:** 166 tests collected from 10 modules — **19 failed**, **147 passed** in ~14 min (842 s)

## How to reproduce

From the repository root:

```bash
cd /home/abhinav/codes/apple_pick_sim
export PYTHONPATH="$(pwd)"
unset PYTEST_DISABLE_PLUGIN_AUTOLOAD
uv run --directory newton python -m pytest \
  ../apple_pick_sim/tests/test_coupled_fruiting_system.py \
  ../apple_pick_sim/tests/test_settle_then_weld.py \
  ../apple_pick_sim/tests/test_explicit_apple_load.py \
  ../apple_pick_sim/tests/test_mega_fd.py \
  ../apple_pick_sim/tests/test_mega_fd_kinematics.py \
  ../apple_pick_sim/tests/test_mega_coupled_fruiting.py \
  ../apple_pick_sim/tests/test_toy_theta_recovery.py \
  ../apple_pick_sim/tests/test_proxy_coupling.py \
  ../apple_pick_sim/tests/test_coupling_stability.py \
  ../apple_pick_sim/tests/test_cuda_graph.py \
  -q -p no:launch_testing --tb=short
```

Full log from this run: `/tmp/failing_tests_run.txt` (if preserved).

## Executive summary

| Metric | Value |
|--------|-------|
| Failed | **19** |
| Passed | 147 |
| Modules with failures | 5 of 10 (`test_mega_fd*`, `test_proxy_coupling`, `test_cuda_graph` all green) |

**Top failure themes (by count):**

1. **Welded / stem–TCP wrench sign and magnitude (9 tests)** — Expected upward support or restoring lateral forces; measured **large negative** TCP or stem \(F_z\) (often kN-scale) or wrong force direction. Many runs show **TCP–proxy IK gap** at bootstrap (0.06–0.20 m) even when tests proceed.
2. **IK bootstrap / teleop reach (5 tests)** — `IKBootstrapConvergenceError` or `IKTeleopConvergenceError` when proxy pose after settle or scene layout is far from FR3 workspace from `COUPLED_ROBOT_BASE_POS` \((0.2, 0.2, -0.35)\).
3. **Harvest force cap / quiescence (2 tests)** — Quiescent stem harvest hits **1000 N** (likely a solver cap), above the **500 N** test threshold.
4. **Mega 1×1 vs instance-0 parity (1 test)** — Proxy position diverges after a few coupled substeps (especially **Z**).
5. **Theta recovery / FD identification (4 parametrized cases, 3 unique tests)** — Loss does not decrease or \(k\) estimate disagrees with brute-force grid when `fix_to_apple=True` (and flat loss when `False`).
6. **Stem quasi-static torque (1 test)** — Hanging-tree settle path: \(F_z\) OK, **\(|τ|\)** slightly above threshold.

**Self-collision:** Prior A/B showed `enable_self_collisions=False` is the intended default and is **not** the primary cause of these failures (see [Ruled out](#ruled-out-self-collision-default)).

**Already in worktree (context):** `conftest` `COUPLED_BASE_POS` / `COUPLED_ROBOT_BASE_POS` defaults, harvest-related seed tweaks, debug print removal in `settle_then_weld` — failures below are from a **fresh** run with those changes present.

---

## Failure table

| Test (nodeid) | Module purpose (1 line) | Failure symptom | Category | Root cause hypothesis | Notes |
|---------------|-------------------------|-----------------|----------|----------------------|-------|
| `test_coupled_fruiting_system.py::test_coupled_long_horizon_harvest_bounded` | 400 substeps, free proxy, direct-joint hold: harvest \(\|F\|\) stays bounded | `harvest \|F\| grew to 1000.00 N` (cap 500 N) | Harvest cap | Stem-harvest / coupling outputs peg at **1000 N** force cap under scene-merge layout; not a quiet quiescent hold | **seed=4**; IK bootstrap err **0.063 m**; not flaky |
| `test_coupled_fruiting_system.py::test_coupled_fr3_tcp_stem_load_at_hold_welded` | Welded hold: TCP stem harvest ≈ apple weight, upward \(F_z\) | `expected upward TCP support, Fz=-1039.73 N` | Welded stem Fz | Wrench sign/frame or double-count vs explicit weight; huge downward TCP force | **seed=45**; bootstrap **0 m** (perfect TCP–proxy) yet still wrong sign |
| `test_coupled_fruiting_system.py::test_welded_coupled_holding_hanging_tree_stem_reaction_upward` | Gravity hold, welded: stem reaction upward | `Fz=-6089.88 N` (need > 1 N) | Welded stem Fz | Same coupling/harvest path; stem gather not opposing gravity as tests expect | **seed=31**; bootstrap err **0.12 m** |
| `test_coupled_fruiting_system.py::test_welded_coupled_vertical_pull_produces_upward_stem_tension` | +Z teleop lift: stem tension upward | `Fz=-33850.91 N` while lifting | Welded stem Fz | Constraint impulse / harvest mapping produces downward stem wrench at lift | **seed=33**; apple **does** rise (\(dz > 0.01\) m) |
| `test_coupled_fruiting_system.py::test_coupled_stem_vertical_force_matches_apple_weight` | High-anchor VBD settle: stem \(F_z \approx mg\), small \(\|τ\|\) | `\|τ\|=0.0192 N·m` > thresh **0.0052 N·m** | Structural / scene (torque) | Quasi-static stem torque not negligible after settle (not coupled FR3) | **seed=31**; \(F_z\) assert passed; separate from mega merge |
| `test_coupled_fruiting_system.py::test_welded_coupled_lateral_drive_restoring_force[x_neg]` | Zero-g lateral teleop: stem force opposes displacement | `disp[0]=-0.0394`, `F[0]=-62084` — same sign, not restoring | Welded stem Fz | Lateral stem/TCP wrench dominated by spurious large forces; sign test fails | **seed=32** (`32+axis`); only **x_neg** failed in this run |
| `test_coupled_fruiting_system.py::test_fr3_ee_teleop_drives_mujoco_joint_targets` | Velocity IK teleop advances TCP target and `joint_target_pos` | `IKTeleopConvergenceError`: pos err **0.0058 m** > tol **0.0050 m** | IK bootstrap | Bootstrap leaves TCP **0.063 m** from proxy; small +X teleop step exceeds tight teleop tolerance | **seed=4** |
| `test_settle_then_weld.py::test_settle_then_weld_quiet_start_bounds_first_harvest_wrench` | Settle VBD → weld → one substep: first harvest wrench small | `IKBootstrapConvergenceError`: pos err **0.2698 m** at build | Settle→weld / IK bootstrap | After settle, proxy at **z≈1.28 m** unreachable from robot base; weld bootstrap fails before harvest check | **seed=0** |
| `test_settle_then_weld.py::test_seed_quiet_zeros_apple_and_proxy_twists` | `seed_fix_to_apple_from_settled` zeros twists | Bootstrap err **0.3714 m** at build | Settle→weld / IK bootstrap | Same: settled proxy pose outside FR3 reach | **seed=1** |
| `test_settle_then_weld.py::test_seed_bootstrap_clears_proxy_forces` | Seed clears `proxy_forces` / cache | Bootstrap err **0.4053 m** at build | Settle→weld / IK bootstrap | Same reachability failure on seed | **seed=3** |
| `test_explicit_apple_load.py::test_stem_harvest_explicit_adds_force_and_torque` | `explicit_apple_weight` delta on stem harvest matches analytic wrench | Torque delta ACTUAL kN-scale vs DESIRED **O(1e-3) N·m** | Welded stem Fz | Explicit torque term drowned by baseline harvest; wrong stem/TCP coupling | **seed=50**; bootstrap **0.20 m** |
| `test_explicit_apple_load.py::test_coupled_substep_default_includes_explicit_apple_weight` | Default coupled substep: TCP \(F_z \ge 0.5\,mg\) | `Fz=-10206.43 N` | Welded stem Fz | End-to-end harvest sign/magnitude broken despite `stem_harvest_explicit_apple_weight=True` | **seed=51** |
| `test_explicit_apple_load.py::test_settle_weld_hold_explicit_support_matches_mg` | Settle→weld→hold: explicit support ≈ \(mg\) | `IKBootstrapConvergenceError` after settle: err **0.4950 m**, proxy **(-0.135, 0.179, 1.282)** | Settle→weld / IK bootstrap | Re-bootstrap at fixed origin cannot reach settled proxy | **seed=52** |
| `test_mega_coupled_fruiting.py::test_mega_instance0_parity_vs_1x1` | Mega instance 0 proxy matches 1×1 coupled scene | Proxy **Y,Z** mismatch (e.g. Z **1.327** vs **0.445**) | Mega parity | Scene merge / instance layout / bootstrap differs between mega and single build | **SEED=11**; mega bootstrap **0.19 m**, single **0.066 m** |
| `test_toy_theta_recovery.py::test_theta_recovery_loss_decreases[False]` | FD recovery: loss decreases (free proxy) | `min(loss_hist)=0` not `< loss_hist[0]=0` | Theta recovery | Flat loss landscape — gradient step does not move loss | **SEED=7**, `fix_to_apple=False` |
| `test_toy_theta_recovery.py::test_theta_recovery_loss_decreases[True]` | FD recovery: loss decreases (welded) | Loss stuck at **0.365** | Theta recovery | Welded feature path insensitive or wrong wrench feature vs \(k\) | **SEED=7**, `fix_to_apple=True` |
| `test_toy_theta_recovery.py::test_theta_recovery_converges_within_tolerance[True]` | Relative error in \(k\) ≤ 10% | `rel_err=0.126` > 0.1 | Theta recovery | Poor convergence when welded (warmup 300 substeps) | **SEED=7**; `[False]` passed |
| `test_toy_theta_recovery.py::test_theta_recovery_brute_force_grid_agrees[True]` | Recovered \(k\) within 15% of grid best | \(\|k_{final}-k_{grid}\|/k_* \approx 0.25\) | Theta recovery | Optimizer lands on wrong minimum when welded | **SEED=7**; `[False]` passed |
| `test_coupling_stability.py::test_stem_coupling_quiescent_forces_bounded` | 60 substeps free proxy: \(\|F\| < 500\) N | `\|F\|=1000.00 N` | Harvest cap | Same 1000 N peg as long-horizon harvest test | **seed=20**; bootstrap **0.074 m** |

---

## By category

### Harvest cap (2)

- `test_coupled_long_horizon_harvest_bounded` — 400 substeps, **seed 4**
- `test_stem_coupling_quiescent_forces_bounded` — 60 substeps, **seed 20**

Both report **exactly 1000 N**, suggesting an internal force cap in stem harvest or coupling, not gradual instability.

### Welded stem / TCP \(F_z\) and sign (7)

- `test_coupled_fr3_tcp_stem_load_at_hold_welded` (seed 45)
- `test_welded_coupled_holding_hanging_tree_stem_reaction_upward` (seed 31)
- `test_welded_coupled_vertical_pull_produces_upward_stem_tension` (seed 33)
- `test_welded_coupled_lateral_drive_restoring_force[x_neg]` (seed 32)
- `test_stem_harvest_explicit_adds_force_and_torque` (seed 50)
- `test_coupled_substep_default_includes_explicit_apple_weight` (seed 51)

Common pattern: large **negative** world \(F_z\) or lateral forces with kN magnitudes; tests expect support opposite gravity or motion.

### Settle→weld / IK bootstrap (4)

- `test_settle_then_weld_quiet_start_bounds_first_harvest_wrench` (seed 0)
- `test_seed_quiet_zeros_apple_and_proxy_twists` (seed 1)
- `test_seed_bootstrap_clears_proxy_forces` (seed 3)
- `test_settle_weld_hold_explicit_support_matches_mg` (seed 52) — fails in `seed_fix_to_apple_from_settled` re-bootstrap

Settled cable places proxy **~1.2–1.3 m** high; FR3 from `(0.2, 0.2, -0.35)` cannot close within **0.25 m** bootstrap tolerance.

### IK teleop (1)

- `test_fr3_ee_teleop_drives_mujoco_joint_targets` (seed 4) — marginal teleop tolerance vs bootstrap gap

### Mega parity (1)

- `test_mega_instance0_parity_vs_1x1` (seed 11)

### Theta recovery (4 failing parametrizations)

- `test_theta_recovery_loss_decreases[False|True]`
- `test_theta_recovery_converges_within_tolerance[True]`
- `test_theta_recovery_brute_force_grid_agrees[True]`

Free-proxy `[False]` converge/grid tests **passed**; welded `[True]` path is the weak branch.

### Quasi-static stem torque (1)

- `test_coupled_stem_vertical_force_matches_apple_weight` (seed 31) — **not** coupled FR3; high `base_pos=(0,0,4)` fruiting scene

---

## Ruled out: self-collision default

Investigation concluded **`enable_self_collisions=False`** is correct for coupled cable scenes and is **not** the primary driver of the 19 failures. Failures align with **scene merge / robot base placement**, **stem-harvest wrench mapping**, and **settle→weld IK reachability**, not intra-chain self-collision toggles.

---

## Recommended fix order (maintainers)

1. **Stem harvest / TCP wrench convention** — Fix sign and magnitude for welded `fix_to_apple` (and explicit apple weight delta) so \(F_z\) is upward \(\mathcal{O}(mg)\) and lateral restoring signs match displacement. Unblocks the largest cluster (7 tests) and stabilizes explicit-load tests.
2. **Scene merge + robot base vs proxy placement** — Align `COUPLED_BASE_POS`, `COUPLED_ROBOT_BASE_POS`, and post-settle proxy poses so bootstrap error stays well below 0.25 m (and teleop 0.005 m where needed). Unblocks settle→weld (4) and reduces IK noise on force tests.
3. **Harvest 1000 N cap** — Identify whether 1000 N is intentional; either lower quiescent coupling or raise test cap with documented physics bound. Two tests.
4. **Mega instance-0 parity** — After (1–2), diff mega vs 1×1 build/bootstrap/substep for instance 0 (seed 11).
5. **Theta recovery (welded)** — Revisit feature slice / warmup / wrench rows once coupled harvest is trustworthy; free-proxy path already mostly green.
6. **Stem torque threshold** — Revisit `test_coupled_stem_vertical_force_matches_apple_weight` tolerance or settle length if still needed after coupling fixes (lower priority; isolated scene).

---

## Passing modules (no action from this run)

- `test_mega_fd.py`, `test_mega_fd_kinematics.py`, `test_proxy_coupling.py`, `test_cuda_graph.py` — all tests passed in this suite.
