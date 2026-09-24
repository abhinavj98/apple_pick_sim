# RL harvest policy handbook (H6)

This is the living reference for **[M5]**: learning a variable-impedance (VIC) policy
that detaches the apple at the **spur–stem junction** while loading every other
junction as little as possible, in the sys-ID-calibrated simulator. It owns the task
definition (detach envelope, reward, episode semantics), the action / observation /
critic-state contracts, domain randomization, and the skrl training stack under
`apple_pick_gym/rl/`. Sequencing and status belong in `docs/ROADMAP.md`.

## Document status

| Field | Value |
| ----- | ----- |
| Last reviewed | 2026-09-24 |
| Code owners | `apple_pick_gym/batched_envs/apple_pick_vic_harvest_env.py`; `harvest_detach.py`; `harvest_outcome.py`; `harvest_reward.py`; `harvest_privileged.py`; `apple_pick_gym/rl/` |
| Status | Living handbook -- infrastructure proven on CPU (surrogate); real-sim training needs CUDA |
| Related handbooks | H1 `docs/handbook-coupled-simulation.md` (two-model coupling); H2 `docs/handbook-variable-impedance.md` (`vic_pose`); H5 `docs/handbook-youngs-cma.md` (the calibrated plant) |
| Archive | Design: `docs/superpowers/specs/2026-09-17-rl-vic-harvest-policy-design.md`; plans: `docs/superpowers/plans/2026-09-17-rl-vic-harvest-policy.md`, `docs/superpowers/plans/2026-09-23-rl-skrl-ppo-lstm-training.md` |

## 1. The idea in one paragraph

Sys-ID (H3–H5) tuned the plant so its compliance matches the real rig. The policy now
starts from a settled, grasped apple (post-grasp, one screened world per env) and
commands a bounded pose delta plus per-axis stiffness and a damping ratio each 60 Hz
step, through the same `vic_pose` interface the real rig uses. The fruit detaches when
the spur–stem junction's combined force and torque cross an elliptical failure envelope.
The policy has to get there using only proprioception and a realistic wrist F/T signal,
and do it without over-loading the spur, primary, supports or the apple–stem junction.
The critic sees everything, including every DR draw. Because the plant's compliance can't
be identified from one frame, the actor is recurrent (LSTM), so it can infer the plant
from interaction history.

## 2. Task definition

### 2.1 Detach envelope (success)

\[
\left(\frac{\lVert F\rVert}{F_{\max}}\right)^2 + \left(\frac{\lVert\tau\rVert}{\tau_{\max}}\right)^2 \ge 1,
\qquad F_{\max} = 20\ \mathrm{N},\ \tau_{\max} = 0.05\ \mathrm{N\,m}
\]

This is an elliptical combined-loading criterion, the biomechanical analogue of von Mises
or Tsai–Wu. It captures the **twist-and-pull synergy**: torque lowers the junction's pull
resistance and vice versa. A pull at 0.7 F_max plus a twist at 0.75 τ_max already fails.

- `harvest_detach.detach_index(wrench)` gives the left-hand side;
  `detach_utilization = sqrt(index)` is the radial fraction of the envelope, which is
  linear in load.
- **Which torque.** `SolverVBD`'s fixed-joint readout reports torque about the *child
  body's COM*, which is `couple + (x_anchor − x_com) × F`. The lever term is not load
  on the junction. With τ_max = 0.05 N·m it is not negligible either: 2.5 mm of lever
  at F_max already reaches τ_max. The env therefore publishes
  `info["target_junction_wrench"]`, the spur–stem wrench with torque shifted to the
  joint anchor (`junction_wrench_at_anchor`, verified against the AVBD kernel, which
  computes `t_lin = r × f` with `r = x_c − com`). The raw readout remains in
  `info["target_junction_force"]`. At rest the child COM sits ~1.2 mm from the anchor.
- **[D1] Which wrench.** The rigid spur-stem fixed joint's constraint readout carries a
  +-0.03 N*m step-to-step solver-noise floor, with no matching motion: the apple rotates < 0.001 rad
  per step. The envelope therefore reads the **stem-root elastic wrench**: the first soft stem cable
  joint's wrench, moved to the junction by statics (`M_J = M_A + (A - J) x F`). It has the same
  mean and ~65x less noise. `info["junction_readout_wrench"]` keeps the readout, and
  `wrench_source="junction_readout"` reverts. Rationale:
  `docs/superpowers/decisions/2026-09-24-rl-open-decisions.md` (D1).
- `info["detach_index"]` is published every step.
- Success needs the index ≥ 1 for `EpisodeConfig.success_streak_steps` consecutive
  steps (default **3** = 50 ms). The streak only rejects single-step solver spikes; the
  envelope is a physical failure criterion, not a hold.
- The fruit does **not** physically detach in the sim. Detachment is a force proxy, and
  the env freezes on success (§4).

Measured at rest on screened v1 worlds (CPU build): spur–stem |F| ≈ 3.7–5.2 N (apple +
stem weight), |τ| ≈ 0.002–0.03 N·m, index ≈ 0.05–0.5. Torque dominates the rest index
and fluctuates step to step with solver noise.

### 2.2 Reward

Every step, for envs not frozen before the step (`harvest_outcome.evaluate_harvest_step`,
shared by the real env and the surrogate):

| Term | Formula | Default weight |
| --- | --- | --- |
| progress | `delta` (**default**): `u_t − u_{t−1}`; `absolute`: `u_t`, with `u = clip(detach_utilization, 0, 1)` | `w_progress = 1.0` |
| pull-out | `−relu(F_wrist · ee_z)` (raw wrist force along the grasp axis) | `w_pullout = 0.5` |
| collateral | `−Σ_{j ≠ spur_stem} relu(‖F_j‖ − ‖F_j‖_rest)`; the rest baseline is recorded at reset (`info["collateral_baseline_norm"]`) | `w_collateral = 0.1` |
| slack | `−1` per live (not-yet-frozen) step: a time cost, so detaching sooner pays; ≤ −5 over a 500-step episode | `w_slack = 0.01` |
| terminal | `+success_bonus` on the success edge; `failure_penalty` on a safety violation (no bonus even if the streak also completes) | `+10 / −20` |

Safety is the anchor-frame target wrench or the raw wrist wrench exceeding 40 N / 10 N·m.

**Why `delta` is the default (finding, 2026-09-24).** With `absolute` progress, hovering
just under the envelope out-earns detaching. Success freezes the env, so reward after it
is 0, and recurrent PPO cuts the return at the freeze edge. On the surrogate, `absolute`
reached 0.68 success, then *declined* to ~0.37 while the return kept rising. `delta`
reached 1.00 success in the same budget (`tmp/rl_vic_viz/surrogate_learning_curves.png`).
`delta` is potential-style shaping: an episode's progress terms telescope to
`u_last − u_0`, so the success bonus and the slack cost are what drive the policy to
detach, and to detach quickly. With slack on, the CPU smoke goes from 0.02 to 1.00 success
in 10 episodes while mean steps-to-success fall from 78 to 48.

Reward reads privileged signals (uncapped junction wrenches, raw F/T). That is legitimate
because reward is not part of the deployed policy. F_max = 20 N is below the 40 N
stem-harvest cap, so the policy's `ft_wrist` stays informative up to detachment.

## 3. Action

The policy acts in `[−1, 1]^13`. `rl/action_scaling.HarvestActionScaler` maps that box to
the env's 13-D action `[dp(3), drot(3), K_lin(3), K_ang(3), ζ]` (`harvest_action.py`):

- pose deltas: affine, `u · bound`, with defaults ±0.02 m and ±0.1 rad per step
  (norm-clipped); 0 means hold;
- stiffness: **log**-affine between `k_min` and `k_max` (20–200 N/m, 2–40 N·m/rad);
  0 maps to the geometric mean;
- ζ: affine on [0.3, 2.0]; damping is derived, `D = 2ζ√K`.

The env integrates the target (`target ← target ⊕ delta`) and packs the 19-D `vic_pose`
action (H2 §3).

**Target leash** (`HarvestActionBounds.max_target_pos_offset_m` / `_rot_offset_rad`,
training defaults 0.15 m / 0.5 rad): the integrated target is projected back within a
leash of the TCP (`leash_target_pose`). Because the VIC wrench is `K·(target − x)`, the
leash bounds the commanded wrench (200 N/m × 0.15 m = 30 N, below the 40 N safety cap).
Without it, a random-walk policy winds the setpoint up without bound (±0.02 m/step × 500
steps ≈ 0.45 m). The env default is `None` (off) for back-compatibility with screening
tools.

Frozen envs hold their target (zero delta) with their last gains.

## 4. Episode semantics and the skrl contract

`BatchedHeterogeneousCoupledSim` resets **the whole batch only**. Consequences, each
enforced by a test:

| Rule | Why | Where |
| --- | --- | --- |
| All envs truncate together at `max_episode_steps` (default 500 = 8.3 s) | no per-env `reset_idx` | base env |
| A succeeded or violating env **freezes**: action held, reward exactly 0 | it can't reset alone | `FreezeMask` |
| `terminated` fires **once**, on the freeze edge (`FreezeMask.update` returns the edge) | `PPO_RNN` zeroes LSTM state and cuts GAE on *every* `terminated` | `harvest_outcome` |
| Invalid envs (failed IK grasp) freeze at reset and never emit an edge | they carry no signal | env `_detect_invalid_envs` |
| The wrapper auto-resets the batch at the time limit and returns the new episode's first obs; `state()` pairs with it | skrl's trainer never calls `reset()` for `num_envs > 1` and calls `state()` right after `step()` | `HarvestSkrlWrapper` |
| The time-limit step is reported `terminated = truncated = True`; `time_limit_bootstrap=False` | skrl's GAE cuts only at `terminated`, and `step_frac` is observed, so time is part of the MDP; otherwise the last step would bootstrap from the next episode's reset value | wrapper |
| LSTM sequences split at `terminated \| truncated` *inside* a stored sequence and zero state there, exactly as in the rollout | step/sequence equivalence | `models._RecurrentTower` |
| Non-finite env rows are zeroed, get reward 0, and are counted in `Step / nonfinite envs` | one blown-up world must not NaN the batch | wrapper |

Frozen envs still produce samples (with reward 0). Their actions are overridden, so the
policy-gradient contribution is zero-mean noise, and the critic sees `frozen` so it can
learn V ≈ 0 there. If frozen samples start to hurt, the fallback is a masked `PPO_RNN`
subclass (plan, *Risks*).

## 5. Observations and critic state

**Actor (40-D, `harvest_obs.flatten_actor_obs`):** `tcp_pos(3)`, `tcp_quat(4, xyzw)`,
`tcp_velocity(6)`, `ft_wrist(6, sensor model)`, `robot_joint_q(7)`, `last_action(13)`,
`step_frac(1)`. There is no vision geometry: the pick policy uses F/T and proprioception
only.

**Critic (`rl/critic_state.critic_state_layout`, 130-D for the five-junction plant):**

1. the actor vector (exact prefix);
2. `_PRIVILEGED_FIELDS` (35): log10 spur/stem flexural and axial moduli, spur/stem damping
   ratios, primary density, log10 support kp / roll_kp, support ζ, per-reset arm
   armature / friction / damping (7 each), and build-time arm link-mass / inertia and EE
   mass / inertia scales;
3. every junction's raw wrench (6 each, junction names sorted);
4. plant geometry: spur/stem length and radius, apple radius and density, weld (grasp)
   axis (9);
5. raw wrist wrench (6), anchor-frame target wrench (6), detach index, success-streak
   fraction, `frozen`, `invalid`.

The actor model reads only `inputs["observations"]` and the critic only
`inputs["states"]`; tests assert that each ignores the other. Checkpoints store both
layouts and refuse to load into a build whose layout differs (§8).

## 6. Domain randomization

| Axis | What | When | Where it comes from | In critic |
| --- | --- | --- | --- | --- |
| Plant materials | spur/stem flexural + axial E, damping ratios, primary density, geometry (real-data fixture, CMA μ ± 1σ) | per env, **baked at build** | world set (`harvest_worlds_v2`) | yes |
| Support joints | kp, roll_kp (log-uniform), ζ | per env, baked into the build *and* the settled snapshot | world set | yes |
| Grasp | weld (approach/pull) direction, 30° cone around straight down | per env, at build | world set | yes (geometry) |
| Arm, build-time | link mass/inertia ×[0.9, 1.1], EE payload ×[0.95, 1.05] | per env, once after build | world set | yes |
| Arm, per episode | joint armature ×[0.7, 1.3], Coulomb friction ×[0.5, 1.5], viscous damping ×[0.7, 1.3] | **every reset** | `_resample_joint_dynamics_dr` | yes |
| F/T sensor | 10 Hz causal EMA + per-episode bias (0.5 N / 0.05 N·m, estimate) + per-step noise **measured on the real rig** (0.12 N; Tx/Ty 0.04–0.05, Tz 0.005 N·m) + bounded drift (estimate) | **every reset** (bias, drift) / step (noise) | `FtSensorConfig.rl_training()` (training default; the env default is noise-free) | raw F/T yes |

**Verified (2026-09-24).** `test_harvest_env_rl_contract.py` (real build) checks that
support DR is per env and in the build, that arm joint DR and the F/T bias change on every
reset and `privileged_fields()` follows the new draw, and that build-time fields stay
fixed. `test_harvest_privileged.py` checks the layout and log10 transforms.
`test_support_joint_dr.py` / `test_sensor_realism.py` check samplers and filters.

Not randomized per episode yet (known gaps):

- Null-space `kp_null` / `kd_null` are sampled but not wired per env (batch-uniform floats
  in `vic_joint_torques_batched`).
- Link mass and payload are per world, not re-drawn per reset (no reset-to-nominal
  entry point).
- Plant base pose relative to the robot is batch-scalar.
- Latency and action delay are not modelled.

With N = 2000 worlds, plant/grasp diversity comes from N.

## 7. Models and PPO

`rl/models.py`, skrl 2.1 `PPO_RNN`. The actor and critic are separate networks (their
inputs differ):

- **actor:** obs → MLP 256 → LSTM 256 → MLP [256, 128] → mean(13). Log std is a
  state-independent parameter (init −0.7, clipped to [−5, 0.5]), and actions are clipped
  to the box.
- **critic:** state → the same tower → V.

The plan's starting hyperparameters (`rl/config.py`):

| Knob | Default |
| --- | --- |
| rollouts / BPTT | 64 / 32 |
| epochs / mini-batches | 5 / 8 |
| lr | 3e-4 with KL-adaptive schedule, target KL 0.01 |
| discount γ / GAE λ | 0.99 / 0.95 |
| clip (ratio / value / grad-norm) | 0.2 / 0.2 / 1.0 |
| entropy | 0 |
| scaling | running standard scaler on obs, state and value |

`TrainConfig.validate()` enforces the sequence and minibatch constraints `PPO_RNN`
imposes: rollouts must be a multiple of the BPTT length, and minibatches must hold whole
sequences.

## 8. Commands

Environment: `uv sync --extra gym --extra vic --extra dev --extra rl`.

```bash
# CPU smoke: PPO learns the surrogate (success 0 -> ~1 in ~1-2 min)
uv run python -m apple_pick_gym.rl.train_vic_harvest --config apple_pick_gym/rl/configs/surrogate_smoke.json

# Real env wiring on CPU (2 worlds, 2 updates; the arm does not move on CPU -- plumbing only)
uv run python -m apple_pick_gym.rl.train_vic_harvest --config apple_pick_gym/rl/configs/sim_wiring_cpu.json --allow-cpu-sim

# GPU: real env wiring (64 v2 worlds, hold settle), then the plan's Task 10 smoke (all 2000 worlds, ~3M samples)
uv run python -m apple_pick_gym.rl.train_vic_harvest --config apple_pick_gym/rl/configs/sim_wiring_gpu.json
uv run python -m apple_pick_gym.rl.train_vic_harvest --config apple_pick_gym/rl/configs/sim_smoke_gpu.json --wandb

# Resume after a crash (fresh process; newest checkpoint; wandb run id is kept)
uv run python -m apple_pick_gym.rl.train_vic_harvest --config <cfg.json> --resume latest

# Baselines and checkpoint evaluation -> metrics JSON
uv run python -m apple_pick_gym.rl.eval_vic_harvest --config <cfg.json> --baseline scripted_pull --episodes 3 --out <dir>/pull.json
uv run python -m apple_pick_gym.rl.eval_vic_harvest --checkpoint <run>/checkpoints/ckpt_<t> --episodes 3 --out <dir>/policy.json
```

Outputs under `--run-dir`:

- `config.json`;
- `metrics.jsonl`, with one `"kind": "update"` row per PPO update (losses, std, lr,
  timing, averaged tracked signals) and one `"kind": "episode"` row per episode (success
  and safety over valid envs, return, peak detach index / target / collateral / wrist
  force, reward-term sums, K/ζ usage, steps to success);
- TensorBoard events (skrl), and wandb with `--wandb` (TensorBoard synced);
- `checkpoints/ckpt_<timestep>/{agent.pt, meta.json}`. `agent.pt` holds the policy,
  value, optimizer and the three scalers. `meta.json` holds the actor/critic layouts,
  action bounds, RNN spec, timestep, updates, wandb id, git SHA and config.

Baselines (`rl/baselines.py`):

- `zero`: hold;
- `random`;
- `scripted_pull`: retreat along **+weld**, i.e. away from the plant, at 2 mm/step,
  K = 150, ζ = 0.9. Grasps approach from below, so pulling along −weld pushes the apple
  up and buckles the stem;
- `scripted_twist_pull`.

The Task 11 gate is to beat `scripted_pull` on held-out worlds without a worse safety
rate.

## 9. The surrogate env

`rl/surrogate_env.SurrogateHarvestEnv` is an analytic spring plant with the real env's
RL contract: the same obs dict, `info` keys, action bounds and leash, the DR split and
privileged layout, and the same `evaluate_harvest_step`. It runs thousands of env-steps
per second on a CPU.

Its structure mirrors the task:

- pulling along the weld axis loads the stem in tension;
- pushing barely loads it;
- twisting loads spur–stem torque, and the other junctions carry the pull load (serial
  chain) but almost none of the twist.

So twist-and-pull detaches with less collateral than a straight pull (tested). It is for
proving infrastructure. **Its numbers mean nothing about the plant.** It integrates
damping implicitly: D = 2ζ√K is sized for unit inertia and diverged explicitly at ζ = 2
(regression-tested).

## 10. Known limitations and open issues

- **The real env needs CUDA.** On a CPU Warp device the batched FR3 arm runs on
  Newton's MuJoCo-CPU backend, which holds a single `mj_data` and does not integrate
  replicated (`separate_worlds`) arms. The TCP stays exactly fixed whatever the action,
  even at N = 1. The env warns, and the CLI refuses `--env sim` on CPU without
  `--allow-cpu-sim`. Existing CPU env tests checked wiring only, which is why this went
  unnoticed. A Newton-side fix is out of scope (vendored).
- **Real-env throughput and stability at N = 2000 are unmeasured** (plan Task 2). N = 512
  was 1.5k env-steps/s on an RTX 4090. N = 1024 had a silent build crash (design spec).
- **Held-out world set not built** (plan Task 3).
- **Reward is untuned.** `delta` progress and a 0.01 slack cost are the defaults. Weights, streak length and F_max/τ_max
  are the maintainer's to tune. At rest the torque term already uses 20–70% of the
  envelope index on some worlds, and it is noisy step to step.
- **Scripted pull direction verified only by geometry** (TCP below apple, weld ≈ −z).
  The real-sim pull could not be run on CPU.
- **In-process rebuild heap corruption** (`docs/in-process-rebuild-heap-corruption.md`):
  one sim build per process. Long runs should resume in a fresh process from
  `--resume latest`. The campaign supervisor (plan) is not written yet.

## 11. Validation

```bash
# fast RL + harvest suite (no sim builds)
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_rl_*.py apple_pick_gym/tests/test_harvest_*.py \
  apple_pick_gym/tests/test_sensor_realism.py apple_pick_gym/tests/test_train_vic_harvest_cli.py \
  apple_pick_gym/tests/test_eval_vic_harvest_cli.py -q -m "not slow"
# CPU learning smoke (~2 min)
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_rl_smoke_training.py -q
# real-env builds: one test per process
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_harvest_env_rl_contract.py -q
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_rl_sim_wiring.py -q
```
