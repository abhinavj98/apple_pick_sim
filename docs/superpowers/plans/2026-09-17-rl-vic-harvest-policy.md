# RL infrastructure for a learned VIC apple-picking policy — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax. **Every task ends with a human-viewable artifact under `tmp/rl_vic_viz/` and requires maintainer approval before the next task starts.**

**Goal:** batched RL infrastructure that trains a recurrent variable-impedance policy to pick apples in the CMA-calibrated simulator, gated on reward and success-rate learning curves.

**Spec:** `docs/superpowers/specs/2026-09-17-rl-vic-harvest-policy-design.md` — read it first; it owns every design decision and its rationale.

**Worktree / branch:** `/home/abhinav/codes/apple_pick_sim-rl-vic` on `feature/rl-vic`.

## Global constraints

- TDD per `.cursor/rules/test-driven-development.mdc`: failing test, minimal green, refactor, commit.
- Run everything with `uv run --env-file pytest.env` from the worktree root.
- New simulation math in `apple_pick_sim/`; RL/gym code in `apple_pick_gym/`.
- GPU hot paths use Warp kernels, not per-substep `.numpy()` round-trips.
- **Do not edit `newton/`** (vendored submodule).
- **Never run more than one `ApplePickVicHarvestEnv`/`BatchedHeterogeneousCoupledSim` build in the same pytest process.** Confirmed in Task 6: a 7-test file that built one env per test hit the documented, pre-existing, open `docs/in-process-rebuild-heap-corruption.md` bug (exact match: `TypeError: 'function' object is not subscriptable` inside `SolverVBD._solve_rigid_body_iteration` -> `wp.launch` -> `pack_arg`). The identical first test passed cleanly in 60s when run alone. Every test in `test_apple_pick_vic_harvest_env.py` must therefore be run as its own process (`pytest ...::test_name` invoked separately, e.g. via a small shell loop), never as a batch `pytest test_apple_pick_vic_harvest_env.py`.
- **Never scale a per-world MuJoCo parameter by `num_envs`.** `njmax` and `nconmax` are per-world; mujoco_warp allocates `(nworld, njmax)` and derives the global contact pool itself via `_resolve_batch_size` (`io.py:955`). Multiplying them by the world count applies it twice and makes allocation O(N²). See Task 0a.
- **Do not add an EMA/LPF to the sys-ID scoring path.** The observation EMA (Task 4) lives strictly in the gym observation path; the "No sim EMA/LPF" rule in H3 still governs `batched_sysid_v1` feature bags.
- Do not reimplement anisotropic VIC gains — `main` already has
  `compute_vic_spatial_wrench_aniso` and the `vic_kp_lin_wp` / `vic_kd_ang_wp`
  buffers. Consume them through the 19-D `vic_pose` action.

## Measured baseline (do not re-derive)

RTX 4090, `gym_defaults()`, 60 Hz control. **After Task 0a** (constraint-arena fix): **N=512 → 1545.6 env-steps/s, 331.3 ms/step, 2.27 GB.** Before it: 863.2 env-steps/s, 593.2 ms, 7.35 GB. N=1024 is **unverified** — the arena fix removes its OOM but the build then dies silently mid-settle. Default to **N=512** for training runs and **N=2–4 on CPU** for tests.

---

### Task 0a: Fix the MuJoCo constraint-arena sizing — **DONE**

**This is a required fix, not an optimization.** No other task in this plan may be benchmarked, tuned, or accepted until it lands, because every throughput and memory number downstream depends on it. It is also a hard blocker for the maintainer's stated goal of scaling to **thousands of environments**.

**Files:** modify `apple_pick_sim/coupled_fruiting/batched_build.py`; test `apple_pick_sim/tests/test_mujoco_constraint_arena.py`.

`batched_build.py:375-376` passes `njmax = nconmax = max(200, 80 * num_envs)`, but both are **per-world** in mujoco_warp (`io.py:930`; arrays are `(nworld, njmax)` at `io.py:996`; `nconmax` is multiplied by `nworld` internally at `io.py:955`). The world count is therefore applied twice, making allocation **O(N²)**.

Verified at N=512: fixing it gives **+79% throughput (863 → 1546 env-steps/s) and −69% GPU memory (7.35 → 2.27 GB)**.

**Why this blocks scaling.** Constraint-arena cost for ~9 `(nworld, njmax)` arrays (`efc.J` adds comparable again on the buggy side):

| N | `njmax = 80·N` | `njmax = 200` | ratio |
| --- | --- | --- | --- |
| 512 | 0.7 GB | 3.5 MB | 205× |
| 1024 | 2.8 GB | 7.0 MB | 410× |
| 2048 | 11.2 GB | 14.1 MB | 819× |
| **4096** | **45.0 GB** | **28.1 MB** | **1638×** |

At N=4096 the current code demands ~45 GB of constraint arena alone on a 24 GB card — for a workload whose measured peak is **9 rows per world**. Measured marginal sim memory once fixed is ~1.2 MB/env, so N=4096 extrapolates to roughly **5 GB total**.

**Audit note:** `batched_build.py:375-376` are the *only* quadratic allocations in the build. Other `* num_envs` expressions (`batched_obs.py:298`, `fruiting_system/build.py:1238`, `broadcast_device.py:181`) are legitimately linear — a per-env quantity times a per-env constant.

- [x] **Step 1:** Failing test — `apple_pick_sim/tests/test_mujoco_constraint_arena.py`, 3 cases (njmax constant across N=2/8/32; per-world nconmax constant; global footprint linear not quadratic in N). Confirmed all 3 fail against the buggy code: `njmax_by_n = {2: 200, 8: 640, 32: 2560}`; footprint ratio 64× instead of 8× for an 8× env-count increase.
- [x] **Step 2:** Confirmed failure (above).
- [x] **Step 3:** Dropped the `* num_envs`. Module-level constants `_ROBOT_MUJOCO_NJMAX_PER_WORLD = _ROBOT_MUJOCO_NCONMAX_PER_WORLD = 200` in `batched_build.py`, overridable via `mujoco_solver_kwargs["njmax"/"nconmax"]`.
- [x] **Step 4:** Confirmed pass — all 3 tests green.
- [x] **Step 5: Safety validation — DONE.** Three fully independent builds (separate OS processes — required, see methodology note below — N=64, 300 diverse random 6D-twist steps, seed=0): `fixed_run1` (njmax=200, peak nefc=14), `fixed_run2` (njmax=200, peak nefc=13), `old_run` (pre-fix formula, njmax=5120 at N=64, peak nefc=14). **Peak nefc across all three independent builds: 13–14**, matching the theoretical worst case (7 permanent dry-friction rows + up to 7 momentary joint-limit rows, 0 contacts — the arm's MuJoCo world has no ground plane and gravity=0). **njmax=200 never came close to being exceeded; no constraints were dropped.**

  **Methodology correction (load-bearing):** the first parity/determinism attempts built and stepped the sim **twice in the same process**, which is exactly the reproduction pattern in `docs/in-process-rebuild-heap-corruption.md` (open, pre-existing, documented bug) — one attempt crashed with SIGSEGV (exit 139) the moment the second in-process build started. **Every subsequent build here uses one fresh OS process per config**, trajectory saved to disk, compared in a separate sim-free process.

  **Full-trajectory bit-parity turned out not to be a usable check on this codebase, and this is itself a real finding worth keeping:** comparing `fixed_run1` vs `fixed_run2` (byte-identical config, njmax=200 both times, separate processes) shows the same large divergence (max abs diff 1.93, 58/64 envs differing by >0.05, only 1/64 matching to <1e-3) as `fixed_run1` vs `old_run` (max abs diff 1.97, 60/64 envs differing, 0/64 matching to <1e-3). The divergence is present from the very first recorded step and does not grow over 300 steps — consistent with per-env IK bootstrap landing on a different local solution (e.g. elbow-up vs elbow-down) at build time across separate processes, not chaotic drift from stepping. **Since the determinism control (same code, same config, twice) shows the identical statistical signature as the arena comparison, the divergence is not caused by the njmax/nconmax fix** — the per-env IK bootstrap is simply not run-to-run reproducible, independent of this change (consistent with the `IKBootstrapConvergenceWarning`s already observed in every build here, some exceeding the 0.05 m tolerance by 2–3×; ties to the IK-convergence risk already flagged for Task 6).

  **The safety argument therefore rests on the measured nefc-vs-clamp margin (≥14× headroom, reproduced across 3 independent builds), not on trajectory identity**, since trajectory identity is not achievable on this codebase for reasons unrelated to this change.

  **New follow-up flagged (not blocking, not previously in this plan):** per-env IK bootstrap placement (`_bootstrap_tcp_per_env`) is not run-to-run reproducible across separate builds of an identical config — most envs land in a qualitatively different arm configuration each build. This matters for anything assuming deterministic per-env grasp placement from a fixed seed, including Task 6's "per-env grasp fixed at build, diversity from N" design. Worth a dedicated investigation later.
- [x] **Step 6:** Artifact — `tmp/rl_vic_viz/task0a_arena.png`: before/after throughput+memory (N=512) and the nefc-vs-clamp distribution across 4 independent builds (N=8 and 3×N=64), all comfortably under the njmax=200 clamp on a log scale.
- [x] **Step 7:** Commit.

---

### Task 0b: ~~Blocking decision — CMA phenotype → sampling distribution~~ **RESOLVED**

**No longer blocking.** The maintainer decided plant DR samples **uniformly from the current ranges fixture** (`fruiting_system_ranges_real_world_proxy_variance.json`, already the `default_ranges_fixture_path()` default), superseding the earlier CMA-centred direction. No distribution needs designing and nothing gates Task 3. Kept as a numbered heading so task references elsewhere stay valid.

---

### Task 0c: Characterise the scaling ceiling — **REQUIRED before any run above N=512**

The maintainer intends to train at **thousands of environments**. Task 0a removes the quadratic memory wall, but it does **not** clear the path to that scale: with the arena fixed, **N=1024 still dies** — silently, mid-settle, with no Python traceback. That is consistent with the SIGSEGV / heap-corruption failure mode this repo already documents in `ApplePickBatchedBaseEnv.close()` (the reason CMA uses process-isolated evaluation waves). It is a separate, unresolved bug.

Training at N≤512 does not need this task. Training above it does.

- [ ] **Step 1:** Reproduce the N=1024 crash under `faulthandler` (and `cuda-gdb` or `compute-sanitizer` if needed) to get an actual fault location. Distinguish a genuine crash from the known `close()`-path corruption by forcing `wp.synchronize()` before teardown.
- [ ] **Step 2:** Fix or document the root cause.
- [ ] **Step 3:** Throughput sweep at N = 512 / 1024 / 2048 / 4096, recording env-steps/s, GPU memory, and **build time** (per-env IK bootstrap is the component that grows with N and may dominate at scale).
- [ ] **Step 4:** Record the measured ceiling and the throughput curve in the design spec, replacing the current "N=512 supported, above unverified" note.
- [ ] **Step 5:** Artifact — throughput and memory versus N, with the crash boundary marked. `tmp/rl_vic_viz/task0c_scaling.png`.

**Do not assume throughput keeps scaling past 512.** Exactly one clamped datapoint exists (512 → 1546 env-steps/s). The previous apparent saturation at 256–512 turned out to be the Task 0a bug, so the honest position is that the curve above 512 is unmeasured, not that it is linear.

---

### Task 1: 13-D delta-pose action → 19-D `vic_pose` packing — **DONE**

**Files:** created `apple_pick_gym/batched_envs/harvest_action.py`; test `apple_pick_gym/tests/test_harvest_action.py`.

Pure-torch, no sim. Ported the damping law (not the file) from `feature/rl-gym`'s `harvest_gains.py::derive_critical_damping`.

- [x] **Step 1:** Failing tests (13 cases) — delta integration accumulates onto a target pose; quaternion stays normalized under repeated small rotations (200-step norm check + an exact quarter-turn check); `Kd = 2ζ√Kp` (incl. scalar and per-env-tensor `ζ`, and negative-stiffness clamping); bounds clamp (delta norm-clip, Kp/ζ range-clamp); wrong action width raises; output is exactly 19 wide with `quat` in `wxyz` order (H2 §3 external contract) verified via an asymmetric quaternion round-tripping unreordered; full split→integrate→pack pipeline.
- [x] **Step 2:** Confirmed failure (`ModuleNotFoundError`).
- [x] **Step 3:** Implemented `HarvestActionBounds` (13-D layout: `[dp(3), drot(3), Klin(3), Kang(3), zeta(1)]`), `split_harvest_action` (reuses the existing `clip_action_tensor` norm-clip — see `batched_action_twists.py` — rather than reimplementing it), `integrate_delta_pose(target, delta)` (world-frame incremental rotation via left-multiply `normalize(delta_q ⊗ quat)`, matching the existing single-env `keyboard.py::integrate_tcp_target` convention, adapted to batched torch + `wxyz`), `pack_vic_pose_action(target, linear_k, angular_k, zeta) -> (N,19)`.
- [x] **Step 4:** Confirmed pass — 13/13 (one test had an arithmetic mistake in its own expected value, fixed; implementation was correct).
- [x] **Step 5:** Artifact — `tmp/rl_vic_viz/task1_delta_integration.png`: position drift under a repeating commanded pattern, `‖quat‖` flat at 1.0 (±1e-4) over 120 steps, and delta norms correctly saturating at the configured clamps.
- [x] **Step 6:** Commit.

---

### Task 2: Per-env arm domain randomization — **DONE**

**Files:** modified `apple_pick_sim/robot/fr3_robot/setup.py`; created `apple_pick_sim/robot/fr3_robot/arm_domain_randomization.py`; test `apple_pick_sim/tests/test_arm_domain_randomization.py`.

Generalized the existing broadcast loops (`setup.py`'s `_set_fr3_joint_armature` / `_set_vic_passive_joint_damping` / `_set_fr3_joint_friction`) from one `values` vector to per-world `(num_worlds, num_arm_dofs)` values via a new shared `_broadcast_or_per_world_dof_values` helper — existing scalar/vector callers are untouched (verified: the 2D-array test fails on the pre-edit code, passes after; 33 tests across `test_batched_vic.py`/`test_vic_joint_torques.py`/`test_fr3_v21_props.py`/`test_fr3_usd_import.py` still pass). The per-world MuJoCo sync is verified end to end — `notify_model_changed(JOINT_DOF_PROPERTIES)` → `update_dof_properties_kernel` at `dim=(nworld, nv)` → `dof_armature[world, mjc_dof]`.

- [x] **Step 1:** Failing tests (9 cases) — pure-numpy sampling (shapes, range bounds, seed reproducibility, per-env distinctness) plus sim-integration (per-world 2D setter behavior, MuJoCo-side sync distinct per world, re-randomization, link-mass/inertia and EE-payload scaling).
- [x] **Step 2:** Confirmed failure (`ModuleNotFoundError`, then one test's own bug — see below).
- [x] **Step 3:** Implemented per-world setters (additive-only generalization) plus `ArmDomainRandomization{Ranges,Sample}` and `sample_arm_domain_randomization` / `apply_joint_dynamics_dr` / `apply_link_mass_inertia_dr` / `apply_ee_payload_dr` in the new module. **Controller gains (`kp_null`/`kd_null`) are sampled but explicitly NOT wired per-env** — `vic_joint_torques_batched.py`'s null-space law takes them as batch-uniform Python floats today; the underlying torch broadcast would work with a `(N,1)` tensor with no other math changes, but threading it through `scene.vic_jt_kp_null`/`kd_null` and consumers is a contained follow-up, out of scope for this task's file list (only `setup.py` + the new module).
- [x] **Step 4:** Confirmed pass — 9/9 (one test had its own bug: called a setter without the `dofs_per_world` kwarg the real batched caller always passes, fixed in the test, not the implementation).
- [x] **Step 5:** Artifact — `tmp/rl_vic_viz/task2_arm_dr.png`: per-env armature/friction/damping histograms (N=16) plus a same-commanded-twist, two-extreme-envs response plot in the full coupled sim, shown as **delta-from-start** (isolating the DR effect from each env's own distinct IK start pose — see Task 0a's IK-non-reproducibility finding). Low armature/friction (×0.3) produces a much larger, oscillatory joint-2 response (~0.35 rad) than high (×3.0, ~0.06 rad, monotonic) — exactly the expected physical direction.
- [x] **Step 6:** Commit.

**Note for the implementer:** arm DR is reassignable at any time plus a notify, so it can be re-randomized **per episode** on reset (`apply_joint_dynamics_dr` re-verified idempotent/reassignable in the test suite). Plant DR cannot (baked at build). Task 6 should call `sample_arm_domain_randomization` + the three `apply_*` functions from its `reset()` path.

---

### Task 3: Plant DR — uniform from fixture, extended to **all ten CMA search knobs** — **DONE**

**Files:** created `apple_pick_sim/fixtures/fruiting_system_ranges_rl_harvest_variance.json` (new RL-specific fixture, decision below); modified `apple_pick_sim/fruiting_system/params.py` (`sim_build.support_dr` schema, additive); created `apple_pick_gym/batched_envs/support_joint_dr.py`; tests in `apple_pick_sim/tests/test_fruiting_system.py`, `apple_pick_sim/tests/test_rl_harvest_fixture.py`, `apple_pick_gym/tests/test_support_joint_dr.py`.

**Decision: new fixture, not widening the shared one.** The shared `fruiting_system_ranges_real_world_proxy_variance.json` is read by sys-ID collect/replay/CMA/benchmarks; widening its degenerate `damping_ratio` ranges in place would silently change sampled physics for every one of those consumers. `fruiting_system_ranges_rl_harvest_variance.json` forks it with (a) spur/stem `damping_ratio` widened `[0.3,0.3] -> [0.05,0.95]` and (b) `sim_build.support_dr` added. Verified byte-identical-unaffected: `test_shared_fixture_is_byte_unaffected`.

**Schema extension (`sim_build.support_dr`, additive):** new `RangeF`/`SupportJointDRRanges` dataclasses in `params.py`, wired into `_validate_sim_build`/`parse_sim_build` behind a new allowed key. Existing consumers reading `sim_build` without this key are unaffected (11 pre-existing `sim_build`/`parse_sim_build` tests still pass unchanged).

**Range provenance:**

| Knob | Range | Source |
| --- | --- | --- |
| 6-7: spur/stem `damping_ratio` | `[0.05, 0.95]` | CMA's own `[0,1]` linear phenotype dim for these, edges trimmed |
| 8: support `kp` | `[100, 1e6]` | Literally CMA's `DEFAULT_SUPPORT_KP_CMA_LOG10_LOWER/UPPER = [2,6]` |
| 9: support `roll_kp` | `[0.075, 7.5]` | This fixture's own choice — one decade either side of CMA's `DEFAULT_SUPPORT_ROLL_KP_LOG10_MIDPOINT = log10(0.75)`; no explicit CMA box exists for this dim |
| 10: support `zeta` | `[0.05, 0.95]` | CMA's own `[0,1]` linear dim, edges trimmed |

**Sampling distribution matters, found mid-task:** `kp`/`roll_kp` span 4 and 2 decades respectively and are searched by CMA on a **log10** scale. An early draft sampled them linear-uniform, which put >98% of draws in the top decade of `kp`'s range (measured: 1.15% of 2000 draws below the log-midpoint, vs the ~50% a log-uniform draw gives). Fixed to log-uniform for `kp`/`roll_kp`; `zeta` stays linear-uniform (matches its own linear `[0,1]` CMA encoding). Regression-guarded by `test_kp_and_roll_kp_are_log_uniform_matching_cma_encoding`.

**The application side already existed and was reused, not rebuilt** — `apply_per_env_support_joint_penalties` / `apply_per_env_support_roll_penalties` in `support_joint_penalties.py`, built for CMA.

- [x] **Step 1:** Failing tests — 8 schema tests (accept/reject `support_dr`, missing/unknown keys, inverted/negative ranges); 6 sampler tests (shapes, range bounds, log-uniform, seed reproducibility, per-env distinctness); 1 integration test (per-env `kp` reaches `scene.cable.solver`'s actual joint penalty array, topology stays uniform); 9 fixture tests (non-degenerate damping ratios, all-knobs-vary, reproducibility, topology, density-in-range, shared-fixture-unaffected).
- [x] **Step 2:** Confirmed failure at each stage (`ValueError: sim_build has unknown keys: ['support_dr']`; `ModuleNotFoundError`; log-uniform test failed at 1.15% vs the ~50% expected before the fix).
- [x] **Step 3a:** Widened `damping_ratio` in the **new** fixture only (not the shared one).
- [x] **Step 3b:** Schema extension landed as described above.
- [x] **Step 3c:** `sample_support_joint_dr` + `apply_support_joint_dr` wired, mirroring `stress_plant_rebuild_loop.py:412`'s call pattern.
- [x] **Step 4:** Confirmed pass — 159 tests across the three files, 0 regressions (also reran the pre-existing full `test_fruiting_system.py` suite: 144 passed).
- [x] **Step 5:** Artifact — `tmp/rl_vic_viz/task3_plant_dr.png`: all ten knobs histogrammed against declared min/max at N=200, none degenerate, knobs 8-9 now correctly log-uniform after the fix.

  **Extremes sanity check (required by this task, not skipped):** four corners tested (loosest/stiffest/stiff+low-damping/loose+high-damping) via `apply_support_joint_dr` + a post-override VBD re-settle. First attempt read exactly `0.0 m/s` residual speed for **every** corner — traced to `settle_quiet_every`'s periodic velocity-zeroing landing on the final sampled substep (a real artifact, not a real result; documented in the build's own warning text). Disabling it (`settle_quiet_every=None`) gave real signal: **no NaN/divergence in any corner** (a genuine positive), but **all four showed nonzero "residual_motion" after 400 re-settle substeps**, worst in the stiff-support+low-damping corner (4.18 m/s) — exactly the resonance-risk combination this task's own warning anticipated. Ran that worst corner again at 3000 substeps: speed **decreased** to 1.25 m/s, confirming convergent (damped, not runaway) dynamics — the residual is a re-settle-budget artifact of a large sudden stiffness change, not instability. **Follow-up for Task 6:** applying `apply_support_joint_dr` post-build (the CMA pattern this reuses) leaves a real, non-instantaneous transient; either budget a re-settle pass after applying it, or accept a short in-episode settling transient at reset.
- [x] **Step 6:** Commit.

**Range-setting note (from the original plan, upheld):** the CMA search box was the literal source for knobs 8 and 6/7/10; knob 9 (`roll_kp`) had no explicit CMA box, so this fixture's own decade-either-side choice is documented as such, not misattributed to CMA.
---

### Task 4: Sensor-realistic `ft_wrist` (EMA + bias/noise/drift) — **DONE**

**Files:** created `apple_pick_gym/batched_envs/sensor_realism.py`; test `apple_pick_gym/tests/test_sensor_realism.py`.

Pure-torch, no sim dependency (like Task 1). `FtSensorModel` holds `(N,6)` EMA state, per-env bias, and drift, all **episode-scoped** — `reset()` reseeds every one of them (matching "bias is resampled per env per episode" / "drift is a slow per-episode walk"), with a `env_mask` for future per-env resets. The EMA is seeded with the first post-reset sample rather than 0, so there is no artificial warm-up ramp.

- [x] **Step 1:** Failing tests (8 cases) — analytic `alpha` at fc=10Hz/60Hz matches ≈0.65; DC passes unchanged; Nyquist-frequency attenuation matches the discrete EMA transfer function `|H(pi)| = a/(2-a)` to <5%; bias constant within an episode, varies per env, resamples on `reset()`; no warm-up transient; drift accumulates within an episode and resets toward zero; masked reset only touches selected envs.
- [x] **Step 2:** Confirmed failure (`ModuleNotFoundError`).
- [x] **Step 3:** Implemented `FtSensorConfig`/`FtSensorModel` as described above.
- [x] **Step 4:** Confirmed pass — 8/8 on the first implementation attempt.
- [x] **Step 5:** Artifact — `tmp/rl_vic_viz/task4_ft_sensor.png`: a synthetic ramp+25Hz-chatter pull through EMA-only vs the full model (bias+noise+drift visible as an offset/jitter band around the EMA-only trace), plus a simulated sine-sweep frequency response measured against the analytic `|H(f)|` curve across 0.5-30Hz — **max abs error 0.0346** (one sweep point near 15Hz shows a small window-edge measurement artifact; everywhere else the fit is near-exact).
- [x] **Step 6:** Commit.

**Invariant restated for whoever wires this into Task 6:** this EMA is strictly a gym-observation-path model of what a real-time controller would see. It is not trying to match the real dataset's `ft_wrist_lpf` (a *zero-phase*, offline `filtfilt`, used only for CMA scoring — no online policy could reproduce that). The "No sim EMA/LPF" rule for `batched_sysid_v1` feature bags (H3) is unaffected; this module must never be wired into the sys-ID scoring path.

---

### Task 5: Observation flattening (actor + privileged critic) — **DONE**

**Files:** created `apple_pick_gym/batched_envs/harvest_obs.py`; test `apple_pick_gym/tests/test_harvest_obs.py`.

skrl memories are flat, and the v3 obs dict is nested (`woody_part_start_pos: dict[str, (N,3)]`). Flattens deterministically **by sorted junction name** so the layout is stable across runs and topologies with the same junction set. `flatten_actor_obs` never receives privileged inputs at all — privileged data cannot leak into the actor vector by construction, not merely by convention.

**Layout (4-junction example, see the artifact for the full table):** actor **71-D** (`tcp_pos`(3) + `tcp_quat`(4) + `tcp_velocity`(6) + `ft_wrist`(6) + `apple_pos`(3) + `apple_quat`(4) + `robot_joint_q`(7) + junction `woody_part_start_pos`/`end_pos`(3 each) + `last_action`(13) + `step_frac`(1)); critic **130-D** = actor's 71 as an exact prefix + 59 privileged (spur/stem flexural+axial modulus, damping ratios, primary density, support `kp`/`roll_kp`/`zeta`, the 3×7-D arm-DR joint arrays + 4 arm-DR scale factors, plus every junction's wrench).

- [x] **Step 1:** Failing tests (9 cases) — actor width/order matches the documented layout and is invariant to junction dict-insertion order (only sorted-name order matters); flattening is deterministic; known fields land at their documented slices; critic layout is the actor layout as an exact `(start,width)`-preserving prefix; `flatten_critic_obs`'s output has `flatten_actor_obs`'s output as an exact tensor prefix; no privileged/force field name appears in the actor layout; mutating privileged inputs after the fact does not change the actor output (pins the by-construction guarantee, not just a convention).
- [x] **Step 2:** Confirmed failure (`ModuleNotFoundError`).
- [x] **Step 3:** Implemented `ObsLayout`/`ObsLayoutEntry`, `actor_obs_layout`, `critic_obs_layout`, `flatten_actor_obs`, `flatten_critic_obs`.
- [x] **Step 4:** Confirmed pass — 9/9 on the first implementation attempt.
- [x] **Step 5:** Artifact — `tmp/rl_vic_viz/task5_obs_layout.md`: full field/start/width/slice table for both layouts, privileged fields marked.
- [x] **Step 6:** Commit.

**Amendment (post-Task 7, during resumed verification):** the maintainer
reversed the actor field list — `apple_pos`/`apple_quat`/
`woody_part_start_pos`/`woody_part_end_pos` were sys-ID's vision-tracked
geometry, not something the pick policy should rely on (proprioception + F/T
only). Removed from `_ACTOR_FIXED_FIELDS`/the junction loops in
`harvest_obs.py` and from `apple_pick_vic_harvest_env.py`'s
`_harvest_observation_space`/`_gather_obs`; the four fields are now
produced via `info` (`_make_info`) instead, unchanged in content, just not
flattened into the policy's input. **New actor width: 40-D** (`tcp_pos`(3)
+ `tcp_quat`(4) + `tcp_velocity`(6) + `ft_wrist`(6) + `robot_joint_q`(7) +
`last_action`(13) + `step_frac`(1), no junction-keyed content); critic
width follows the same composition (actor's new 40 as the prefix + the
same 59 privileged fields = 99-D for a 4-junction topology).
`docs/superpowers/specs/2026-09-17-rl-vic-harvest-policy-design.md`'s
Observation pipeline section and `test_harvest_obs.py` updated to match.

---

### Task 6: `ApplePickVicHarvestEnv` — **DONE**

**Files:** created `apple_pick_gym/batched_envs/apple_pick_vic_harvest_env.py`; test `apple_pick_gym/tests/test_apple_pick_vic_harvest_env.py`.

Builds on `ApplePickBatchedBaseEnv` with `ControllerConfig(mode="vic_pose", action_dim=19)`. Wires together Tasks 1-5: `harvest_action.py` for the 13-D delta-pose split/integrate/pack, `support_joint_dr.py` + the RL harvest fixture for build-time plant DR, `arm_domain_randomization.py` for build-time (link mass/EE payload) and per-reset (joint dynamics) arm DR, `sensor_realism.py` for observed `ft_wrist`. Defaults `ranges_path` to `fruiting_system_ranges_rl_harvest_variance.json` (Task 3's fixture) rather than the shared default, so this env covers all ten CMA knobs out of the box. Per-env grasp direction reuses the existing sys-ID Fibonacci-hemisphere sampler rather than inventing new geometry.

**Two real bugs found and fixed during verification (both in this task's new code, not pre-existing):**

1. **Stale `tcp_pose` read in `reset()`.** The original `reset()` called `restore_episode_snapshot()` then read `bufs.tcp_pose` directly to seed `_target_pose` — but `restore_episode_snapshot()` does not itself refresh `obs_bufs`, so the read saw the *previous* episode's stale value. Caught by `test_target_pose_initializes_to_current_tcp_pose_on_reset` failing with `Mismatched elements: 6/6 (100%)`, up to 0.63 absolute difference — far too large to be numerical noise. Fixed by calling `self._sim.gather_obs()` (a cheap buffer-only refresh) immediately after the restore, before reading `tcp_pose`.
2. **Double-gather with stale sensor state.** The original `reset()` called `super().reset()` (which gathers once) and then gathered again after resetting DR/sensor state — wasteful, and the first gather computed `ft_wrist` through the *pre-reset* sensor bias/EMA. Fixed by not calling `super().reset()` at all: `reset()` now replicates the base class's restore-then-gather sequence itself, with the buffer refresh, `_target_pose` seeding, `_ft_sensor.reset()`, and arm-DR resample all happening in the correct order before the single final `self._gather_obs()` this method returns.

**A third issue was a test bug, not an implementation bug:** `test_sensor_model_state_resets` initially asserted bias differs across resets while constructing the env with `FtSensorConfig()`'s own default `bias_std=0.0` (deliberately "quiet unless configured" — see Task 4) — bias was trivially `0 == 0` every time. Fixed in the test by passing an explicit nonzero `bias_std`.

- [x] **Step 1:** Failing tests (7 cases) — 13-D action space; `vic_pose`/19-D controller config; `_target_pose` initializes to current TCP pose on reset; actions reach the sim as 19-D; `info` carries `woody_part_force`/`target_junction_force` but `obs` does not; arm joint-dynamics DR resamples on reset; sensor model state resets.
- [x] **Step 2:** Confirmed failure (`ModuleNotFoundError`, all 7).
- [x] **Step 3:** Implemented as described above.
- [x] **Step 4:** Confirmed pass — **all 7 acceptance criteria verified correct**, each via at least one clean isolated pass, after fixing the two real bugs and one test bug above.

  **Methodology note (load-bearing, matches Task 0a's finding):** this test file **must never be run as a batch** (`pytest test_apple_pick_vic_harvest_env.py`) — doing so hits the documented, pre-existing, open `docs/in-process-rebuild-heap-corruption.md` bug (confirmed exact match: `TypeError: 'function' object is not subscriptable` in `SolverVBD._solve_rigid_body_iteration` → `wp.launch` → `pack_arg`). Every test must run as its own process. Even in isolation, ~1 in 3 builds hit a further, intermittent crash from the same bug category (`Fatal Python error: Aborted` in a Warp array's `__del__` during `vbd_substep`; silent SIGSEGV with no traceback) — always cleared on retry with identical code/config, confirming genuine intermittency rather than a deterministic logic error. This is now the **fourth** distinct surface symptom of this bug personally observed across this plan (Task 0a: one; Task 6: three), all in Newton/Warp internals never touched by this task's code. Added as a global constraint in this plan.
- [x] **Step 5: IK convergence gate — measured, not skipped.** At the sys-ID default full-hemisphere grasp cone (`max_polar_angle_rad=pi/2`), measured convergence at N=32 was **only 56% (14/32 envs missed the 0.05 m tolerance, up to 0.128 m error)** — confirming the background investigation's earlier N=512 finding (errors up to 0.158 m). Per this task's own instruction ("must be fixed... before training on grasp diversity"), narrowed the default cone to **`pi/6` (30 degrees around straight-down)**: convergence improved to **81% (6/32 missed, up to 0.126 m)**. **The residual ~19% is not eliminated by cone angle alone** — consistent with the separately-documented finding (Task 0a) that per-env IK bootstrap placement is not perfectly reproducible even for a fixed, reachable target. Rejection-sampling failed grasps at build time would close this further but needs a build-path change (`_bootstrap_tcp_per_env`) outside this task's file list — tracked as a follow-up, not silently deferred. At N in the hundreds, ~19% lost envs is a real but survivable training cost, not a blocker; the narrower default is adopted.
- [x] **Step 6:** Artifact — `tmp/rl_vic_viz/task6_env_rollout.png`: TCP vs commanded target (showing the intended compliant lag under plant resistance, not a tracking bug), `ft_wrist` raw vs observed, `spur_stem` junction force ramping past 600 N under a sustained pull, the IK error histogram (3 distinct failure clusters, matching 26/32 converged), and mean TCP displacement across all 32 envs.
- [x] **Step 7:** Commit.

**Global constraint added to this plan** (see the top): never run more than one `ApplePickVicHarvestEnv`/`BatchedHeterogeneousCoupledSim` build in the same pytest process.
---

### Task 7: Reward, success freeze, fixed-length episodes — **DONE**

**Files:** ported `harvest_reward.py` from `feature/rl-gym` (math unchanged, `f_threshold_n` updated 30.0 -> **5.0**); created `apple_pick_gym/batched_envs/harvest_episode.py`; also wired both into `apple_pick_vic_harvest_env.py`'s `compute_reward`/`compute_terminated` (previously Task 6's stubs) and `_actions_tensor` (frozen-env action hold), since leaving them stubbed would have left Task 6's env not actually usable. Tests: `apple_pick_gym/tests/test_harvest_reward.py`, `apple_pick_gym/tests/test_harvest_episode.py`, plus one env-level wiring smoke test added to `test_apple_pick_vic_harvest_env.py`.

`F_thresh = 5 N` (maintainer's explicitly provisional placeholder), success = sustained for `K` consecutive steps with no safety-cap violation. **All envs truncate together**; a succeeded env freezes (action held via `FreezeMask.apply_to_action` in `_actions_tensor`, reward masked to 0) but the episode does not end for the trainer — this keeps LSTM hidden-state resets batch-uniform. `compute_reward` and `compute_terminated` share one `success`/`safety_violation` computation (cached in `self._pending_terminated`) rather than each recomputing it, since the base env's `step()` always calls `compute_reward` first and calling `SuccessStreakTracker.update()` twice per step would double-increment the streak.

- [x] **Step 1:** Failing tests (19 across both modules) — success requires `K` *consecutive* steps and a gap resets the counter (not just pauses it); per-env independence; a frozen env accrues no further reward and holds its last action; freeze is sticky across updates and clears on `reset()`; safety-cap violation on force *or* torque; terminal reward gives the success bonus with no penalty, or the failure penalty with **no bonus even if a success streak completed on the same step** (a safety violation is defined as a failure regardless).
- [x] **Step 2:** Confirmed failure (`ModuleNotFoundError`, all 19).
- [x] **Step 3:** Implemented `HarvestRewardConfig`/reward math (ported), `EpisodeConfig`/`SuccessStreakTracker`/`FreezeMask`/`check_safety_violation`/`compute_terminal_reward` (new), then wired both into the env (`info["ft_wrist"]` added as the raw/privileged channel `compute_pullout_penalty` needs, distinct from `obs["ft_wrist"]`).
- [x] **Step 4:** Confirmed pass — 19/19 module tests on the first implementation attempt; the env-level wiring smoke test (reward no longer the Task 6 stub, `truncated` uniform every step, tracker/mask shapes correct) passed on retry (first attempt hit the same pre-existing intermittent Newton/Warp crash documented in Tasks 0a/6; cleared on an identical rerun). A regression check on `test_actions_reach_sim_as_19d_vic_pose` (touched by the `_actions_tensor` freeze-mask addition) also passed.

  **Safety-cap values used:** `EpisodeConfig` defaults to 40 N / 10 N*m, matching H2's documented `DEFAULT_STEM_FORCE_CAP_N`/`DEFAULT_STEM_TORQUE_CAP_NM` (the stem-harvest transfer cap already in this codebase), not the design spec's alternative suggestion of reusing `batched_stability_monitor.py`'s separate 100 N / 40 N*m caps -- kept in scope to avoid pulling in another module. Checked against both `info["target_junction_force"]` (uncapped, so this is a real check) and `info["ft_wrist"]` (already hard-capped at the same values by the stem-harvest transfer, so this half is currently a no-op safety net, kept for robustness if the caps ever diverge).
- [x] **Step 5:** Artifact — `tmp/rl_vic_viz/task7_reward.png`: reward-term decomposition for one env (dense terms continue evolving after freeze since freezing holds the *action*, not the physics; total step reward correctly flatlines at 0 once frozen), the success-streak counter for all 4 envs, and the freeze-mask timeline. **The rollout organically captured a genuine streak-reset event** (one env's streak climbed to 44, then a real dip below the force threshold reset it to 0, then it climbed again) -- live confirmation of the "gap resets the counter" behavior in the actual environment, not just a synthetic unit test.
- [x] **Step 6:** Commit.

---

### Task 8: skrl integration — recurrent PPO, privileged critic

**Files:** add an `rl` extra to `pyproject.toml`; create `apple_pick_gym/rl/` (skrl env wrapper, LSTM actor, privileged LSTM critic, trainer entry point); tests alongside.

**Dependency check (done 2026-09-17):** `uv pip install --dry-run "skrl>=1.4"` resolves to **skrl 2.1.0**, pulling `tensorboard` 2.21 (reuse it for the Task 9 learning curves rather than adding another logger). **Caution:** most skrl recurrent-PPO examples online target the 1.x API; check the 2.x docs for the installed version instead of copying a 1.x recipe.

- [ ] **Step 1:** Failing tests — the wrapper exposes skrl's expected API over the batched env; actor and critic hidden state have the right shapes and **reset on episode boundaries**; a checkpoint round-trips *both* networks' hidden-state specs and the Task 5 obs layout.
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Implement. Keep the privileged critic path strictly separate from the actor's observation.
- [ ] **Step 4:** Confirm pass.
- [ ] **Step 5:** Artifact — a short smoke run (small N, few hundred steps) producing a loss curve, confirming gradients flow through both recurrent towers. `tmp/rl_vic_viz/task8_smoke.png`.
- [ ] **Step 6:** Commit.

---

### Task 9: Training entry point, logging, and the learning-curve gate

**Files:** create `apple_pick_gym/rl/train_vic_harvest.py`; a CLI test.

- [ ] **Step 1:** Failing CLI test — `--num-envs`, `--total-steps`, `--checkpoint-dir`, `--seed` parse and dry-run.
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Implement, logging episode reward and success rate per iteration, with periodic checkpointing and resume.
- [ ] **Step 4:** Confirm pass.
- [ ] **Step 5:** **Capstone artifact** — a real training run at N=512 producing reward and success-rate learning curves. This is the phase's exit criterion. `tmp/rl_vic_viz/task9_learning_curves.png`.
- [ ] **Step 6:** Write `docs/handbook-rl-policy.md` (H6) covering the action/observation contracts, DR ranges, reward, and how to launch and resume training.
- [ ] **Step 7:** Run the full new test suite end to end.
- [ ] **Step 8:** Commit; update `docs/ROADMAP.md` [M5].

---

## Follow-ups (explicitly not in this plan)

- CUDA graph capture of `coupled_substep` — the largest throughput lever; the code is already written to be capture-safe (`scene.py:592-594`) but no capture exists.
- Investigate the **silent N=1024 build crash** remaining after Task 0a (reproduce under `faulthandler` / `cuda-gdb`; suspect the documented `close()`-path heap corruption).
- Real per-env `reset_idx` on `BatchedHeterogeneousCoupledSim`.
- Reward revisit: combined force + torque success criterion (maintainer flagged).
- Per-env plant base pose relative to the robot.
- Reach-and-grasp before harvest; real stem-detach physics.
- 30 Hz policy rate or action-repeat for deployment parity with the real rig.
