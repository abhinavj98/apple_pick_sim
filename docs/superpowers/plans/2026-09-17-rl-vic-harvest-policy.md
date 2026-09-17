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

### Task 1: 13-D delta-pose action → 19-D `vic_pose` packing

**Files:** create `apple_pick_gym/batched_envs/harvest_action.py`; test `apple_pick_gym/tests/test_harvest_action.py`.

Pure-torch, no sim. Port `derive_critical_damping` from `feature/rl-gym`'s `harvest_gains.py`.

- [ ] **Step 1:** Failing tests — delta integration accumulates onto a target pose; quaternion stays normalized under repeated small rotations; `Kd = 2ζ√Kp`; bounds clamp; output is exactly 19 wide with `quat` in `wxyz` order (H2 §3 external contract).
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Implement `HarvestActionBounds`, `integrate_delta_pose(target, delta, bounds)`, `pack_vic_pose_action(target, Kp, zeta) -> (N,19)`.
- [ ] **Step 4:** Confirm pass.
- [ ] **Step 5:** Artifact — plot a commanded delta sequence and the integrated target pose trajectory to `tmp/rl_vic_viz/task1_delta_integration.png`.
- [ ] **Step 6:** Commit.

---

### Task 2: Per-env arm domain randomization

**Files:** modify `apple_pick_sim/robot/fr3_robot/setup.py`; create `apple_pick_sim/robot/fr3_robot/arm_domain_randomization.py`; test `apple_pick_sim/tests/test_arm_domain_randomization.py`.

Generalize the existing broadcast loops (`setup.py:346` and siblings) from one `values` vector to per-world values. The per-world sync is already verified end to end — `notify_model_changed(JOINT_DOF_PROPERTIES)` → `update_dof_properties_kernel` at `dim=(nworld, nv)` → `dof_armature[world, mjc_dof]`.

- [ ] **Step 1:** Failing test — assign distinct armature/friction/damping per world, notify, then assert `mjw_model.dof_armature[w]` differs per world and matches what was written. Cover link mass/inertia via `BODY_INERTIAL_PROPERTIES`.
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Implement per-world setters (keep the existing broadcast signatures working — additive only) plus an `ArmDomainRandomization` sampler with bounded ranges for joint dynamics, link mass/inertia, EE payload/TCP geometry (**tight band — this is sys-ID-calibrated**), and controller gains.
- [ ] **Step 4:** Confirm pass.
- [ ] **Step 5:** Artifact — per-world armature/friction/damping histograms plus a plot showing two envs with different arm DR responding differently to the same commanded pose step. Save to `tmp/rl_vic_viz/task2_arm_dr.png`.
- [ ] **Step 6:** Commit.

**Note for the implementer:** arm DR is reassignable at any time plus a notify, so it can be re-randomized **per episode** on reset. Plant DR cannot (baked at build). Expose a `resample()` entry point for the reset path in Task 6.

---

### Task 3: Plant DR — uniform from fixture, extended to **all ten CMA search knobs**

**Files:** modify `apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json` (or a new RL-specific fixture); modify `apple_pick_sim/fruiting_system/params.py` (`sim_build` schema); create `apple_pick_gym/batched_envs/support_joint_dr.py`; tests alongside.

Plant DR samples **uniformly from the ranges fixture**. Rod-level params already flow per-env through `sample_heterogeneous_params_list` (`params.py:1122`). The requirement here is that DR covers **every knob the CMA search varies** — a parameter worth identifying is one the policy should be robust to.

**Current coverage — only 5 of 10 vary:**

| # | CMA knob | Fixture today | Status |
| --- | --- | --- | --- |
| 1–2 | spur/stem flexural modulus | `flexural_modulus_pa` 25–100 MPa | varies |
| 3–4 | spur/stem axial modulus | `youngs_modulus_pa` 25–100 MPa | varies |
| 5 | primary density | `primary.density` 600–900 | varies |
| 6–7 | spur/stem `damping_ratio` | `min == max == 0.3` | **degenerate** |
| 8 | `support_kp` | `sim_build.joint_{angular,linear}_kp_overrides.support = 10000.0` | **scalar, no range** |
| 9 | `support_roll_kp` | `sim_build.joint_roll_kp_overrides.support = 0.75` | **scalar** |
| 10 | `support_joint_zeta` | `sim_build.joint_damping_ratio = 0.3` | **scalar** |

**The application side already exists** — do not rebuild it. `apply_per_env_support_joint_penalties` (accepts `zeta_per_env`, covering knobs 8 and 10) and `apply_per_env_support_roll_penalties` (knob 9) in `apple_pick_gym/batched_envs/support_joint_penalties.py` already take per-env values; they were built for CMA. Only the *sampling* and *wiring* are missing.

- [ ] **Step 1:** Failing tests — (a) sampled spur/stem `damping_ratio` is non-degenerate across envs; (b) per-env support `k_p`, roll `k_p` and ζ are distinct across envs and reach the solver arrays; (c) sampling is reproducible for a fixed seed; (d) segment topology stays identical across envs. Assert apple density lands near ~800 kg/m³ rather than assuming geometric self-consistency (chord closure is *enforced*, so it proves nothing).
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3a:** Widen the two degenerate `damping_ratio` ranges in the fixture.
- [ ] **Step 3b:** **Schema extension** — `sim_build` currently holds scalars and is validated against `_SIM_BUILD_ALLOWED_KEYS` (`params.py`). Add a ranges-shaped block for the support joint (e.g. `support_joint: {kp: {min,max}, roll_kp: {min,max}, zeta: {min,max}}`) and keep the existing scalar form working for every current consumer — sys-ID collect, replay and CMA all read `sim_build` and **must not change behaviour**.
- [ ] **Step 3c:** Sample the three support values per env and call the existing `apply_per_env_support_*` helpers after build (see `apple_pick_sim/examples/stress_plant_rebuild_loop.py:412` for the call pattern).
- [ ] **Step 4:** Confirm pass.
- [ ] **Step 5:** Artifact — per-parameter sampled distributions for **all ten knobs** against their declared min/max, showing none is degenerate. `tmp/rl_vic_viz/task3_plant_dr.png`.
- [ ] **Step 6:** Commit.

**Range-setting note:** the CMA search box is the natural source for sensible min/max on knobs 6–10, since it is where those parameters were searched. Using the search bounds is *not* the same as the superseded "centre on the CMA fit" idea — it borrows the plausible interval, then samples it uniformly.

**Do not widen blindly:** knobs 8–10 are support-joint stiffness/damping. Very low `support_kp` makes the whole structure flop; very high values make it rigid and remove the compliance the task depends on. Sanity-check the extremes of each range in a short rollout before training on them.

---

### Task 4: Sensor-realistic `ft_wrist` (EMA + bias/noise/drift)

**Files:** create `apple_pick_gym/batched_envs/sensor_realism.py`; test alongside.

- [ ] **Step 1:** Failing tests — EMA with `a = 1 - exp(-2π·fc/f_control)` (≈0.65 at fc=10 Hz, 60 Hz control) attenuates a high-frequency input by the expected factor and passes DC unchanged; bias is constant within an episode and varies per env; state resets on `reset()`.
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Implement a batched `FtSensorModel` holding `(N,6)` EMA state, per-env bias, per-step noise, slow drift.
- [ ] **Step 4:** Confirm pass.
- [ ] **Step 5:** Artifact — raw vs EMA-filtered vs noisy `ft_wrist` over a real pull, plus the measured frequency response against the analytic EMA curve. `tmp/rl_vic_viz/task4_ft_sensor.png`.
- [ ] **Step 6:** Commit.

---

### Task 5: Observation flattening (actor + privileged critic)

**Files:** create `apple_pick_gym/batched_envs/harvest_obs.py`; test alongside.

skrl memories are flat, and the v3 obs dict is nested (`woody_part_start_pos: dict[str, (N,3)]`). Flatten deterministically **by sorted junction name** so the layout is stable across runs and topologies.

- [ ] **Step 1:** Failing tests — actor vector has the documented fixed width and order; the same obs dict always flattens identically; the privileged vector is a strict superset; **no privileged field leaks into the actor vector** (assert explicitly by name).
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Implement `flatten_actor_obs` / `flatten_critic_obs` plus a recorded layout descriptor for checkpoint compatibility.
- [ ] **Step 4:** Confirm pass.
- [ ] **Step 5:** Artifact — a printed layout table (field, slice, width) to `tmp/rl_vic_viz/task5_obs_layout.md`.
- [ ] **Step 6:** Commit.

---

### Task 6: `ApplePickVicHarvestEnv`

**Files:** create `apple_pick_gym/batched_envs/apple_pick_vic_harvest_env.py`; test alongside. Port the obs/`info` layout from `feature/rl-gym`'s `ApplePickHarvestEnv` as reference.

Builds on `ApplePickBatchedBaseEnv` with `ControllerConfig(mode="vic_pose", action_dim=19)`. Per-env grasps via `per_env_grippers` (`weld_direction` + `weld_proxy_offset_in_apple_frame`); per-env plant params from Task 3.

- [ ] **Step 1:** Failing tests — 13-D action space; `_target_pose` initializes to current TCP pose on reset; actions reach the sim as 19-D `vic_pose`; `info` carries `woody_part_force` and `target_junction_force` but `obs` does not; arm DR resamples on reset; sensor model state resets.
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Implement, wiring Tasks 1–5 together.
- [ ] **Step 4:** Confirm pass (CPU, N=2).
- [ ] **Step 5: IK convergence gate (acceptance criterion, not optional).** Per-env grasps are placed by `_bootstrap_tcp_per_env` (`batched_heterogeneous_build.py:981`), which runs IK to each env's sampled grasp pose against `IK_TELEOP_POS_TOL_M = 0.005` m. A background investigation observed many `IKBootstrapConvergenceWarning`s at N=512 with position errors up to **0.158 m** — 30× tolerance. An env whose IK misses starts with the arm welded at the wrong pose, which silently corrupts exactly the grasp diversity this design depends on.

  Measure and report the **IK convergence rate across the sampled grasp distribution**, and the error distribution of the failures. If a meaningful fraction misses, that bounds achievable grasp diversity and must be fixed (more `ik_bootstrap_iterations`, more seed restarts via `_IK_BOOTSTRAP_JOINT_Q_SEED_FRACS`, or rejection-sampling grasp poses to the reachable set) **before** training on grasp diversity. Do not proceed to Task 7 with an unmeasured IK failure rate.
- [ ] **Step 6:** Artifact — rollout under a scripted pull showing TCP pose, commanded vs achieved target, `ft_wrist` raw vs observed, and `spur_stem` junction force, **plus the IK convergence histogram**. `tmp/rl_vic_viz/task6_env_rollout.png`.
- [ ] **Step 7:** Commit.

---

### Task 7: Reward, success freeze, fixed-length episodes

**Files:** port `harvest_reward.py` from `feature/rl-gym`; create `apple_pick_gym/batched_envs/harvest_episode.py`; tests alongside.

`F_thresh = 5 N`, success = sustained for `K` consecutive steps with no safety-cap violation. **All envs truncate together**; a succeeded env freezes (action held, reward masked) but the episode does not end for the trainer — this keeps LSTM hidden-state resets batch-uniform.

- [ ] **Step 1:** Failing tests — success requires `K` *consecutive* steps (a gap resets the counter); a frozen env accrues no further reward and holds its last action; `truncated` is uniform across the batch; safety-cap violation terminates with the failure penalty and no bonus.
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Implement the reward config, success-streak tracker, and freeze mask.
- [ ] **Step 4:** Confirm pass.
- [ ] **Step 5:** Artifact — reward-term decomposition over an episode plus a success/freeze timeline across envs. `tmp/rl_vic_viz/task7_reward.png`.
- [ ] **Step 6:** Commit.

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
