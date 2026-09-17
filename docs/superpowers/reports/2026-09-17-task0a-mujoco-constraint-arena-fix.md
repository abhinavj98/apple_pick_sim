# Task 0a report: MuJoCo constraint-arena sizing fix

**Status:** DONE.

## Change

`apple_pick_sim/coupled_fruiting/batched_build.py::build_replicated_robot_model`:
dropped `njmax = nconmax = max(200, 80 * num_envs)` (per-world sizes multiplied
by world count, causing O(N^2) allocation) in favor of module-level per-world
constants `_ROBOT_MUJOCO_NJMAX_PER_WORLD = _ROBOT_MUJOCO_NCONMAX_PER_WORLD = 200`,
overridable via `mujoco_solver_kwargs["njmax"/"nconmax"]` if a caller ever needs to.

## TDD

- Failing test written first: `apple_pick_sim/tests/test_mujoco_constraint_arena.py`
  (3 cases: njmax constant across N, per-world nconmax constant across N,
  global footprint linear not quadratic in N). Confirmed all 3 fail against
  the buggy code (njmax_by_n = {2: 200, 8: 640, 32: 2560}; footprint ratio
  64x instead of 8x for an 8x env-count increase).
- Fix applied; all 3 tests pass.

## Safety validation

- [x] Peak `nefc` under a realistic workload (N=64, 300 steps, diverse random
  6D twists including a +Z bias to load the stem junction) vs the N=8
  scripted-pull probe used earlier — see "Corrected safety-validation
  results" below.
- [x] Rollout parity: same action sequence run under the fixed arena
  (njmax=200) and the old formula (njmax=80*N) — see below; turned out not
  to be a valid check on this codebase for unrelated reasons (see
  "Methodology correction").

## Regression check

Targeted subset covering `build_replicated_robot_model` callers:
`test_mujoco_constraint_arena`, `test_batched_heterogeneous_build`,
`test_batched_heterogeneous_coupled_sim`, `test_batched_heterogeneous_config`,
`test_controller_config_actions`, `test_episode_state_snapshot` — **82 passed,
1 failed** (`test_defaults_preset_constructs_and_validates`, asserting
`settle_substeps == 2000` against a current default of `6000`). Confirmed
**pre-existing**: fails identically with the Task 0a change reverted
(`git stash`). Unrelated config/test drift, out of scope here.

## Known confound (not caused by this change)

An earlier full `apple_pick_sim/tests/` run crashed (SIGSEGV-style faulthandler
dump, no traceback) during `test_use_settle_cache_false_ignores_disk`. This
matches `docs/in-process-rebuild-heap-corruption.md` exactly: an open,
documented, intermittent bug from rebuilding many heavy scenes in one process
(likely upstream OpenUSD `LoadUsdPhysicsFromRange`), unrelated to njmax/nconmax.
Evidence this is not caused by the Task 0a change:
- The same test passes cleanly in isolation on the *unmodified* (pre-fix) code.
- The Task 0a change touches only which two integers are passed to
  `SolverMuJoCo(...)` at construction; it does not touch scene rebuild,
  teardown, or Warp module coloring — the mechanisms the corruption doc
  implicates.
- The doc explicitly states "a single passing/failing run proves nothing" for
  this bug and that it is mitigated in production via process-isolated CMA
  evaluation waves, not by any per-run test discipline.

## Methodology correction (important)

The first attempt at rollout parity (and a follow-up determinism check) built
and stepped `ApplePickBatchedVicEnv` **twice in the same Python process**.
This is precisely the reproduction pattern documented in
`docs/in-process-rebuild-heap-corruption.md` ("Rebuilding
`BatchedHeterogeneousCoupledSim` many times in a single process and stepping
each scene" -> host-side memory corruption, SIGSEGV or silently wrong
results). The determinism check crashed with SIGSEGV (exit 139) exactly when
the second in-process build started. The first parity attempt's huge
divergence (1.96 m abs, 113581x rel) must therefore be treated as an artifact
of this unrelated, pre-existing, documented bug -- not evidence about the
njmax/nconmax fix.

**Corrected methodology:** each config (fixed x2 for determinism, old-formula
x1 for parity) is built and stepped in its own fresh OS process
(`uv run ... python run_single_arena_config.py <out.npz> <njmax> <steps> <n>`),
saving the TCP-pose trajectory to disk. Comparison happens in a separate,
sim-free process that only loads the `.npz` files. No process ever builds the
sim more than once.

## Corrected safety-validation results

Three fully independent builds (separate OS processes, N=64, 300 diverse
random 6D-twist steps each, seed=0 for the action sequence):

| Run | njmax (per-world) | peak `nefc` | peak `nacon` |
| --- | --- | --- | --- |
| fixed_run1 | 200 | 14 | 0 |
| fixed_run2 | 200 | 13 | 0 |
| old_run (pre-fix formula, N=64 -> 80*64=5120) | 5120 | 14 | 0 |

**Peak nefc across all three independent builds: 13-14**, matching the
theoretical worst case (7 permanent dry-friction rows + up to 7 momentary
joint-limit rows, 0 contacts since the arm's MuJoCo world has no ground
plane and gravity=0). **njmax=200 never came close to being exceeded in any
run.** No constraints were dropped by the fix.

**Full-trajectory bit-parity is not a usable check on this codebase**, and
this is a real, separate finding: comparing TCP-pose trajectories between
`fixed_run1` and `fixed_run2` (byte-identical config, njmax=200 both times)
shows the SAME large divergence (max abs diff 1.93, 58/64 envs differing by
>0.05, only 1/64 matching to <1e-3) as comparing `fixed_run1` to `old_run`
(max abs diff 1.97, 60/64 envs differing, 0/64 matching to <1e-3). The
divergence is present from the very first recorded step and does not grow
over the 300-step rollout (consistent with a different per-env IK
solution/local-minimum being reached at build time -- e.g. elbow-up vs
elbow-down -- rather than chaotic accumulation from stepping). **Since the
determinism control (same code, same config, twice) shows the identical
statistical signature as the arena comparison, the trajectory divergence is
not caused by the njmax/nconmax fix.** The per-env IK bootstrap is simply
not run-to-run reproducible across separate builds, independent of this
change. (Consistent with the `IKBootstrapConvergenceWarning`s already
observed in every one of these builds, some exceeding the 0.05 m tolerance
by 2-3x -- this ties to the IK-convergence risk already flagged for Task 6.)

**Conclusion:** the arena fix is safe. The safety argument rests on the
measured/theoretical nefc-vs-clamp margin (>=14x headroom, reproduced across
3 independent builds), not on trajectory identity, because trajectory
identity is not achievable on this codebase for reasons unrelated to this
change.

## Artifact

`tmp/rl_vic_viz/task0a_arena.png` — before/after throughput+memory (N=512)
and the nefc-vs-clamp distribution across 4 independent builds, all
comfortably under the njmax=200 clamp on a log scale.

## Follow-up flagged (not blocking Task 0a, not in original plan)

Per-env IK bootstrap placement (`_bootstrap_tcp_per_env`) is not
run-to-run reproducible across separate process builds of the identical
config -- most envs land in a qualitatively different arm configuration each
build. This matters for anything assuming deterministic per-env grasp
placement from a fixed seed (e.g. Task 6's "per-env grasp fixed at build,
diversity from N" design, and reproducing/debugging any specific training
run). Worth a dedicated investigation later; out of scope here.
