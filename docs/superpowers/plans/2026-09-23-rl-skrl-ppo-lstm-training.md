# skrl PPO + LSTM training pipeline for the VIC harvest policy — implementation plan

> Expands Tasks 8–9 of `docs/superpowers/plans/2026-09-17-rl-vic-harvest-policy.md` (design spec: `docs/superpowers/specs/2026-09-17-rl-vic-harvest-policy-design.md`).
> **Worktree / branch:** `.claude/worktrees/rl-skrl-ppo` on `feature/rl-skrl-ppo`, forked from `fix/harvest-vic-forces`.

## Context

The harvest env stack is done and the screened worlds look stable. It lives on `fix/harvest-vic-forces`, not `main`, and covers:
- `ApplePickVicHarvestEnv`: 13-D delta-pose VIC action, a 40-D actor obs, arm and plant DR, the F/T sensor EMA, reward, and freeze logic.
- 2000 screened worlds (`harvest_worlds_v2`) with settled snapshots, including a single all-2000 snapshot committed at `apple_pick_gym/world_sets/snapshots/harvest_worlds_v2_all2000/shard_00*` (also in `~/.cache/apple_pick_sim/world_sets/harvest_worlds_v2_all2000/`).

Tasks 8–9 of `docs/superpowers/plans/2026-09-17-rl-vic-harvest-policy.md` are still open: the skrl recurrent-PPO integration and the training run with its learning-curve gate. `apple_pick_gym/rl/` does not exist yet. This plan expands those two tasks into executable steps against skrl **2.1.0**, which is already installed through the `rl` extra. I read its source rather than relying on 1.x examples.

Maintainer decisions this plan follows:
- LSTM actor with a privileged LSTM critic.
- Plant DR is baked into the build and snapshot; arm joint-dynamics DR is resampled per reset.
- Train on screened worlds.
- **Train on the single N=2000 build.**
- The **exit gate is beating a scripted baseline** on held-out worlds, with no worse safety-violation rate.

## Worktree

The RL groundwork is on `fix/harvest-vic-forces`, so branch from there. Branching from `main` would lose it.
```bash
git -C /home/abhinav/codes/apple_pick_sim worktree add .claude/worktrees/rl-skrl-ppo -b feature/rl-skrl-ppo fix/harvest-vic-forces
# EnterWorktree(path=".claude/worktrees/rl-skrl-ppo")
git submodule update --init --recursive
uv sync --extra gym --extra vic --extra dev --extra rl
```
Snapshots live in `~/.cache`, so the new worktree shares them.

## Problems found in skrl 2.1 that the design must handle

1. **Terminated must fire once.** `PPO_RNN.record_transition` zeroes an env's LSTM state on `terminated | truncated` (ppo_rnn.py:367-374). The env's `compute_terminated` returns `success_achieved | safety_violation` on every step. Frozen envs keep holding force, so the success streak stays true. The fix is to emit `terminated` only on the freeze edge: `newly_done & ~was_frozen`.
2. **GAE does not cut at truncation.** `compute_gae` only masks on `terminated` (ppo_rnn.py:45-54). At the synchronized time limit, the wrapper auto-resets, so the last step would bootstrap from the next episode's reset value. `step_frac` is in both actor and critic inputs, so the time limit is part of the MDP. The wrapper therefore reports **`terminated = True` at the time-limit step** (with `truncated` also set), and we keep `time_limit_bootstrap=False`.
3. **The model resets hidden state inside a sequence.** During the update, skrl passes `rnn` initial states plus `terminated`/`truncated` for each sampled sequence (ppo_rnn.py:470-495). The model's `compute` must split the sequence at `terminated | truncated` and zero the state after each boundary, as in skrl's LSTM model example. Sequences are per-env contiguous slices (`all_sequence_indexes`, memories/torch/base.py:63), so `rollouts % sequence_length == 0` is required.
4. **Auto-reset is the wrapper's job.** `SequentialTrainer` never calls `reset()` when `num_envs > 1`, and it calls `env.state()` right after `step()`/`reset()`. The wrapper must cache the critic state that goes with the obs it returns, including the post-reset state.
5. **Frozen and invalid samples still enter the PPO loss.** After the edge, frozen envs keep stepping with reward 0, and skrl normalizes advantages across the batch. Actions in frozen envs are overridden by the freeze mask, so their log-prob gradient is zero-mean noise, not bias. We accept that initially and give the critic a `frozen` flag so V≈0 there. We monitor the frozen fraction, and fall back to a small masked `PPO_RNN` subclass if it hurts (see Risks).

## Module layout (`apple_pick_gym/rl/`)

| File | Contents |
| --- | --- |
| `action_scaling.py` | `HarvestActionScaler(bounds: HarvestActionBounds)`: maps `[-1,1]^13` to env units. Pose deltas and ζ map affinely; K_lin and K_ang map log-affinely. Has `to_env`/`to_policy`. Pure torch. |
| `privileged_state.py` | `build_privileged(env) -> dict` in `harvest_obs._PRIVILEGED_FIELDS` order. Moduli, kp and roll_kp are fed as **log10**. It reads `env._sim.per_env_params`, `_last_support_dr_sample`, `_arm_build_dr_sample`, `_last_arm_dr_sample`. Static parts are computed once per build; arm joint DR is refreshed on each reset. |
| `skrl_wrapper.py` | `HarvestSkrlWrapper(skrl.envs.wrappers.torch.base.Wrapper)`. Exposes `observation_space=Box(40)`, `state_space=Box(99+2)` (critic features plus `frozen`, `invalid`), and `action_space=Box(-1,1,13)`. `step`: scale the action, call `env.step`, flatten with `flatten_actor_obs`/`flatten_critic_obs`, and apply the time-limit terminated rule. On truncation it runs a whole-batch `env.reset()` and returns the reset obs. It also caches `state()` and emits episode stats through `info` for `track_data`. |
| `models.py` | `LstmGaussianActor(GaussianMixin, Model)`: obs(40) → MLP 256 → LSTM(256, 1 layer) → MLP [256,128] → mean(13). Log std is a state-independent parameter (init −1.0, clip [−5, 0.5]) with `clip_actions=True`. `LstmValueCritic(DeterministicMixin, Model)`: state(101) → MLP 256 → LSTM(256) → MLP [256,128] → 1. Each has `get_specification()` returning `{"rnn": {"sequence_length": L, "sizes": [(1,N,256),(1,N,256)]}}` and the terminated-aware sequence `compute`. The two networks are separate (not shared) because the inputs differ. |
| `config.py` | `TrainConfig` frozen dataclass plus JSON load/save. Holds PPO hyperparameters, model sizes, world-set/snapshot paths, logging, and the checkpoint cadence. It follows the repo convention of dataclasses plus argparse, with no hydra. |
| `checkpoint.py` | Wraps skrl `agent.write_checkpoint`/`load`, which cover policy, value, optimizer, and all three preprocessors. Adds a sidecar `meta.json` with the actor and critic `ObsLayout` tables, action bounds, the RNN spec, world-set fingerprint, global timestep, the wandb run id, and the git SHA. Load refuses on a layout or bound mismatch. |
| `baselines.py` | `ZeroPolicy`, `RandomPolicy`, `ScriptedPullPolicy`. The scripted one retreats along each env's grasp axis (−weld direction) at a constant rate with mid K and ζ≈0.9, which mirrors the screening pull. |
| `train_vic_harvest.py` | CLI entry point: build the env from the world set plus snapshot, wrap it, run `PPO_RNN` with `SequentialTrainer`, and support `--resume`. |
| `eval_vic_harvest.py` | CLI: roll out the deterministic policy mean or a baseline on a world set for K episodes. Writes `metrics.json`: success rate, safety rate, time-to-success, peak junction/wrist force, reward decomposition. |
| `run_training_campaign.py` | Supervisor. Runs training as a **fresh subprocess per segment** and resumes from the latest checkpoint after a SIGSEGV/abort. This matches the `--retries` pattern in `example_build_world_snapshots.py` for the known Warp heap-corruption crash. Periodic held-out eval also runs in its own process. |

Reuse, don't rebuild:
- `harvest_obs.flatten_actor_obs` / `flatten_critic_obs` / `actor_obs_layout` / `critic_obs_layout`
- `harvest_action.HarvestActionBounds`
- `world_set.load_world_set` / `world_specs_to_env_kwargs`
- `ApplePickVicHarvestEnv(episode_snapshot_path=...)`
- `harvest_logging.HarvestMetricsLogger` / `WandbSink` for extra wandb panels
- `harvest_video.HarvestVideoRecorder` for eval videos
- `example_build_world_snapshots.py` for the held-out snapshot

## PPO / LSTM starting hyperparameters (in `TrainConfig`)

| Knob | Value | Why |
| --- | --- | --- |
| N | 2000 (all-2000 snapshot) | Maintainer decision |
| control rate / episode | 60 Hz / 500 steps (8.3 s) | Existing env |
| rollouts | 64 | 128k samples per update at N=2000 |
| sequence_length (BPTT) | 32 (~0.5 s) | Hidden state carries across sequences via stored initial states, so this only truncates gradients. Must divide rollouts. |
| learning_epochs / mini_batches | 5 / 8 | ~16k-sample minibatches (500 sequences × 32) |
| lr | 3e-4 with `KLAdaptiveLR(kl_threshold=0.01)` | Standard for continuous-control PPO |
| discount / GAE λ | 0.99 / 0.95 | Horizon ~100 steps, well inside 500 |
| ratio_clip / value_clip / grad_norm_clip | 0.2 / 0.2 / 1.0 | |
| entropy_loss_scale | 0.0 (0.001 if std collapses) | |
| preprocessors | `RunningStandardScaler` on observations, states and values | Privileged inputs are already log10-transformed |
| time_limit_bootstrap | False | Time is observed; see problem 2 |
| mixed_precision | False | Keep the LSTM stable first |

Budget arithmetic: throughput at N=2000 is measured in Task 2. If it scales like N=512 (~1.5k env-steps/s), one 128k-sample rollout takes ~85 s, 10M steps take ~2 h, and 50M take ~9 h. The capstone budget gets fixed after Task 2.

## Tasks

TDD throughout: write a failing test, confirm it fails, implement, then commit. Run with `uv run --env-file pytest.env`. Any test that builds a sim is marked `slow` and **runs in its own pytest process**, never batched in-process, because of `docs/in-process-rebuild-heap-corruption.md`. Each task ends with an artifact under `tmp/rl_vic_viz/` for maintainer review.

**Task 0: Worktree** (commands above). Smoke-check: `uv run python -c "import skrl; print(skrl.__version__)"` should print 2.1.x.

**Task 1: Env fixes** (`apple_pick_vic_harvest_env.py`, `test_apple_pick_vic_harvest_env.py`)
- `compute_terminated` emits only the freeze edge. Test: an env that keeps holding force after success reports `terminated` exactly once.
- Add `privileged_state()` delegating to `rl/privileged_state.py`, plus `info["episode"]["frozen"]` / `info["invalid_env"]`, which already exist and are kept.
- Update the one existing wiring test that asserted the old behavior.

**Task 2: N=2000 throughput and stability gate** (script `apple_pick_gym/rl/bench_harvest_throughput.py`)
- In fresh processes, build from the all-2000 snapshot and run 1000 random-action steps, 3 trials.
- Record build time, env-steps/s, ms/step, peak GPU memory, and invalid count.
- If N=2000 crashes repeatedly or needs more than ~20 GB, stop and report back instead of falling back silently.
- Artifact: `task2_throughput.md`.

**Task 3: Held-out world set**
- The accepted worlds not in v2 are 157 in `tmp/harvest_worlds/v2_pass2_all.jsonl` and 55 in `pass2_all.jsonl`, 212 total.
- Save them as `apple_pick_gym/world_sets/harvest_worlds_v2_holdout.jsonl` with a coverage report (`world_set_coverage`).
- Build the snapshot with `example_build_world_snapshots.py --shard-size 212 --retries 4` into `~/.cache/.../harvest_worlds_v2_holdout/`, and add a README row.
- Test: no world_id overlap with v2.

**Task 4: Action scaling + privileged state** (pure torch, fast tests)
- Round-trip `to_env(to_policy(a)) == a`; ±1 maps exactly to the bounds; K is log-affine (0 maps to √(Kmin·Kmax)).
- Privileged dict has the right field order and widths, is log10 where specified, and has no NaN. It uses a fake env object with DR samples, so no sim is built.

**Task 5: LSTM models** (`test_rl_models.py`, CPU, fast)
- Spec shapes are correct.
- **Sequence/step equivalence:** running L single steps with resets at terminated indices gives the same outputs and final state as one sequence-mode forward with the `terminated` tensor.
- Gradients reach both LSTMs' weights.
- The actor never receives `states` (its forward ignores the key; asserted by passing garbage states).

**Task 6: Wrapper** (`test_rl_skrl_wrapper.py`)
- Fast tests use a `FakeHarvestEnv` with the same dict obs and info contract, no sim. They check:
  - skrl API and spaces
  - scaled actions reach the env in env units
  - `state()` pairs with the obs just returned, including after auto-reset
  - the time-limit step reports terminated & truncated and the next obs is the reset obs
  - the freeze edge passes through once
- One `slow` integration test builds a real env with N=4 worlds from `harvest_worlds_v1` (hold settle, no snapshot) and runs 20 wrapped steps.

**Task 7: Pipeline proof on a memory task** (`test_rl_ppo_rnn_memory.py`, slow, CPU, a few minutes)
- A toy batched env where each episode draws a hidden gain g. g appears in the obs only at t=0, and reward at t>0 is −|a − g|.
- A memoryless policy's ceiling is analytically known. PPO_RNN with our models and wrapper conventions must clearly beat it within a fixed step budget.
- This proves BPTT, hidden-state reset and the GAE handling end to end without the sim.

**Task 8: Checkpointing + training CLI** (`test_train_vic_harvest_cli.py`)
- `--config`, `--world-set`, `--snapshot`, `--num-envs`, `--total-steps`, `--checkpoint-dir`, `--seed`, `--resume {latest|path}`, `--dry-run`, and `--wandb/--no-wandb` (TensorBoard always on).
- Dry-run builds agent and memory against the fake env.
- Checkpoint round-trip restores weights, optimizer, scalers and `meta.json`, and refuses a mismatched obs layout.
- The wandb run id persists across resumes (`resume="allow"`).
- Logged every update, besides skrl's defaults:
  - success rate over valid envs, safety rate, invalid fraction, frozen fraction
  - mean time-to-success
  - reward terms (progress / pull-out / collateral / terminal)
  - mean and peak target-junction and wrist force
  - policy std, per-dim K/ζ usage histograms, grad norms, env and update timing

**Task 9: Baselines + eval CLI** (`test_eval_vic_harvest_cli.py`)
- Eval runs deterministic rollouts (policy mean, hidden state carried through the episode).
- Baselines: zero, random, scripted pull.
- Run the three baselines once on the held-out snapshot and on the train snapshot. That calibrates reward scale and confirms the task is solvable: if the scripted pull succeeds 0% of the time, stop and revisit `f_threshold_n` (10 N in code, "provisional") with the maintainer before training.
- Artifact: `task9_baselines.md`.

**Task 10: Smoke training run**
- N=2000, ~2–3M steps via the campaign driver.
- Check: finite losses, KL near target, std not collapsing, value explained-variance rising, reward and success above zero-action. One checkpoint resume is exercised mid-run.
- Artifact: `task10_smoke.png` (losses, KL, std, reward, success).

**Task 11: Capstone run + gate**
- Train within the budget from Task 2, with held-out eval every M updates in a separate process.
- **Gate:**
  - held-out success rate is rising and ends above the scripted-pull baseline, as well as the zero and random baselines;
  - held-out safety-violation rate is ≤ the scripted baseline's.
- Artifact: `task11_learning_curves.png`, with baseline reference lines and the train vs held-out curves, plus eval videos of a few held-out worlds.

**Task 12: Docs**
- Write `docs/handbook-rl-policy.md` (H6) covering the action/obs/state contracts, the terminated/truncation semantics, hyperparameters, and launch/resume/eval commands.
- Mark Tasks 8–9 in the existing plan doc as done (pointing here), and update `docs/ROADMAP.md` [M5] and `docs/CODEBASE_GUIDE.md`.

## Risks and fallbacks

- **Mid-run Warp crash (SIGSEGV):** the supervisor resumes in a fresh process from the latest checkpoint (checkpoint every ~10 updates). Crash count gets logged.
- **N=2000 too slow or unstable:** Task 2 is a stop-and-report gate. The fallback is the 4×500 shards, which already exist; that choice goes back to the maintainer.
- **Frozen-sample noise hurts learning** (frozen fraction > ~50% with a flat success curve): add `MaskedPPO_RNN`. It stores a `valid` tensor (not frozen before this step, not invalid) and masks the policy loss, entropy and advantage normalization.
- **Reward too sparse or easily gamed:** baselines in Task 9 expose this before any long run. Reward-weight changes go through the maintainer; `f_threshold_n` is explicitly provisional.
- **Plant DR is fixed per build (2000 worlds):** diversity comes from N. Arm joint DR and the F/T sensor model still resample every reset.

## Verification

- Fast suite: `uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_rl_*.py -q -m "not slow"`.
- Slow tests one per process, for example: `for t in $(pytest --collect-only -q -m slow apple_pick_gym/tests/test_rl_*.py | grep ::); do uv run --env-file pytest.env python -m pytest "$t" -q || echo FAIL $t; done`.
- Existing harvest tests stay green: `test_harvest_*`, `test_world_set.py`, and env tests run one per process.
- End to end: `uv run python -m apple_pick_gym.rl.run_training_campaign --config <cfg.json> --total-steps 2e6`, then `uv run python -m apple_pick_gym.rl.eval_vic_harvest --checkpoint <dir>/latest --snapshot ~/.cache/apple_pick_sim/world_sets/harvest_worlds_v2_holdout/...` compared against the Task 9 baseline JSON.
- Commit per task on `feature/rl-skrl-ppo`, push the branch, and open a draft PR into `fix/harvest-vic-forces` (or `main` once that branch merges).
