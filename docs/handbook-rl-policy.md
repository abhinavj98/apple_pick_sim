# RL harvest policy handbook (H6)

This is the living reference for **[M5]**: learning a variable-impedance (VIC) policy
that detaches the apple at the **spur–stem junction** while loading every other
junction as little as possible, in the sys-ID-calibrated simulator. It owns the task
definition (detach criterion, reward, episode semantics), the action / observation /
critic-state contracts, domain randomization, and the skrl training stack under
`apple_pick_gym/rl/`. Sequencing and status belong in `docs/ROADMAP.md`.

## Document status

| Field | Value |
| ----- | ----- |
| Last reviewed | 2026-10-02 |
| Code owners | `apple_pick_gym/batched_envs/apple_pick_vic_harvest_env.py`; `harvest_detach.py`; `harvest_outcome.py`; `harvest_reward.py`; `harvest_obs.py`; `harvest_privileged.py`; `apple_pick_gym/rl/` |
| Status | Living handbook -- **task v2 (D17)** implemented and CPU-proven on the surrogate; the v3 world set (equator grasps) is being screened; v2 GPU training not yet run |
| Related handbooks | H1 `docs/handbook-coupled-simulation.md` (two-model coupling); H2 `docs/handbook-variable-impedance.md` (`vic_pose`); H5 `docs/handbook-youngs-cma.md` (the calibrated plant) |
| Archive | Design: `docs/superpowers/specs/2026-09-17-rl-vic-harvest-policy-design.md`; plans: `docs/superpowers/plans/2026-09-17-rl-vic-harvest-policy.md`, `docs/superpowers/plans/2026-09-23-rl-skrl-ppo-lstm-training.md`; decisions D1-D17: `docs/superpowers/decisions/2026-09-24-rl-open-decisions.md` |

**Task versions.** v1 (D1-D16) is the infrastructure build: grasps from below, an elliptical
force/torque envelope, absolute world-frame observations and actions, an 8-term reward. Policies
trained on it were tested on the real FR3 and did not pick the way the maintainer wants
(human-like bend/twist, then pull). **v2 (D17, 2026-10-02)** is the redesign this handbook now
describes; v1 settings stay reachable through config switches for reproducing old runs
(`detach_criterion="envelope"`, `action_frame="world"`, the v1 reward weights).

## 1. The idea in one paragraph

Sys-ID (H3–H5) tuned the plant so its compliance matches the real rig. The policy starts from a
settled apple grasped **around its equator** (one screened world per env) and commands a bounded
pose delta plus per-axis stiffness and a damping ratio each 60 Hz step, **in the TCP frame**,
through the `vic_pose` interface. The fruit detaches when the stem's tension reaches the force a
human-measured bend-and-pull curve (Li et al. 2016) needs at the stem's current bending angle:
bending the stem first cuts the pull needed from ~26 N to ~4 N. The policy sees only start-relative
proprioception and a realistic wrist F/T signal in the TCP frame, and is paid to detach quickly
while loading the rest of the tree little. The critic sees everything, including every DR draw.
The actor is recurrent (LSTM), so it can infer the plant from interaction history.

## 2. Task definition

### 2.1 Detach criterion (success) -- bend-and-pull (D17)

Li et al. 2016 (*Characterizing apple picking patterns for robotic harvesting*, Comput. Electron.
Agric. 127:633) measured the stem tension that detaches a Gala apple at a given stem bending angle
(Eq 3, pulling against the stem's natural orientation; Eq 4, along it). v2 uses their average,
held flat past its minimum (the quadratics were fitted at 10-30 deg and turn back up beyond):

\[
F_d(\theta_b) = s\,\big(0.015\,\theta^2 - 1.155\,\theta + 26.09\big)\ \mathrm{N},\quad
\theta = \min(\theta_b, 38.5^\circ),\qquad \text{detach when } F_t \ge F_d(\theta_b)
\]

| theta_b | 0 deg | 20 deg | >= 38.5 deg |
| --- | --- | --- | --- |
| F_d (s = 1) | 26.1 N | 9.0 N | 3.86 N |

- `s ~ U(0.85, 1.15)` per env per episode (`DetachEnvelopeConfig.detach_scale_range`); the critic
  sees it (`detach_scale`).
- **theta_b** is the angle of the **stem chord** (spur–stem junction -> stem–apple attachment) from
  its settled direction at reset (`harvest_detach.chord_angle_deg`).
- **F_t** is the stem tension: the D1 stem-root elastic force along `-chord`
  (`harvest_detach.stem_tension`). GPU check (16 v2 worlds): 3.6 N at rest (u0 = 0.14), rising to
  ~27 N in a straight pull.
- `u = F_t / F_d` is published as `info["detach_utilization"]` and `info["detach_index"]`; success is
  `u >= 1` for `success_streak_steps` (3) consecutive steps, then the env freezes. The fruit does not
  physically detach in the sim.
- At full bend `F_d` (3.3-4.4 N) is about the rest tension, so a ~40 deg bend alone detaches -- the
  paper's pendulum tests detach at 45-50 deg too.
- The env owns the criterion; the reward reads `info["detach_utilization"]`. `info` also carries
  `theta_b_deg`, `stem_tension`, `detach_force`, `detach_scale`.

**v1 envelope (legacy, `criterion="envelope"`).** `(|F|/20 N)^2 + (|tau|/0.05 N m)^2 >= 1` on the
anchor-frame junction wrench (D1). Its 0.05 N m total moment is dominated by bending: ~1.1 N of
sideways force at the apple already reached it. The offline tools (`detach_sweep`,
`diagnose_detach`) still evaluate it.

### 2.2 Reward (v2: four terms)

Every step, for envs not frozen before the step (`harvest_outcome.evaluate_harvest_step`):

| Term | Formula | v2 weight |
| --- | --- | --- |
| progress | `w (u_t − u_{t−1})`, `u = clip(detach_utilization, 0, 1)`; telescopes to `w (u_end − u_0)` | `w_progress = 10` |
| collateral | `−w min(Σ_{j ≠ spur_stem} relu(‖F_j‖ − ‖F_j‖_rest), clip)` | `w_collateral = 0.02`, `collateral_clip_n = 50` |
| slack | `−w` per live step | `w_slack = 0.01` |
| success | `+bonus` on the success edge | `success_bonus = 60` |

- **Off in v2** (zero weight / caps at 1e12 in `sim_train_gpu_v2.json`): pull-out, the wrist
  soft cap (D11), the peak-collateral charge (D13) and the safety trip (-40 and its freeze).
  Training no longer penalises force; the rig's own limits must. Peaks over 40 N are reported:
  `Episode / peak wrist|target force > 40 N (frac)`.
- The D12 blow-up guard stays (> 200 N or non-finite: freeze, reward 0, invalid).
- **Why the collateral clip.** Scale does not matter (skrl normalises advantages; values pass a
  running scaler), heavy tails do: one overloaded episode dominates the value loss and the scaler.
  The weighted term is bounded at -1 per step; the raw term (peak metrics) is not clipped.
- **Bounds** (250 steps): progress in `[-10 u0, 10 (1 - u0)]`; slack in `[-2.5, 0]`; collateral >= -250.
  Idle scores -2.5; every success scores ~+30 or more. Bend vs pull is decided by force x time in the
  reward, and by the physics (26 N vs ~4 N needed).
- Peak collateral and the safety rate remain logged metrics.

The v1 reward history (D9 rebalance, D11, D13, D13c) is in the decision log.

## 3. Action (v2: tool frame)

The policy acts in `[−1, 1]^13`. `rl/action_scaling.HarvestActionScaler` maps that box to the env's
13-D action `[dp(3), drot(3), K_lin(3), K_ang(3), ζ]` (`harvest_action.py`):

- pose deltas: affine, `u · bound`, D8b cap 0.4 m/s and 0.3 rad/s at 60 Hz (6.7 mm, 5 mrad per
  step), norm-clipped; 0 means hold. With `action_frame="tool"` the delta is **in the current TCP
  frame** (`tool_delta_to_world`, norm-preserving): a twist about the grip axis is one dimension;
- stiffness: log-affine, K_lin 20-**500** N/m (`EnvConfig.k_lin_max`), K_ang 2-40 N·m/rad. With
  `action_frame="tool"` the per-axis gains act **along the TCP axes** (`K_world = R diag(k) Rᵀ`,
  `ControllerConfig.vic_gain_frame="tool"`, kernel `compute_vic_spatial_wrench_aniso_tool`);
- ζ: affine on [0.3, 2.0]; damping is derived, `D = 2ζ√K`.

The env integrates the target (`target ← target ⊕ delta`) and packs the 19-D `vic_pose` action
(H2 §3). The TCP +z points exactly at the apple centre (GPU-checked), so `-z` of the tool is the
pull-out direction.

**Target leash** (0.15 m / 0.5 rad of the TCP): a windup guard only. At K_lin = 500 it allows up to
75 N commanded; v2 relies on the rig for force limits (§2.2).

Frozen envs hold their target (zero delta) with their last gains.

## 4. Episode semantics and the skrl contract

`BatchedHeterogeneousCoupledSim` resets **the whole batch only**. Consequences, each
enforced by a test:

| Rule | Why | Where |
| --- | --- | --- |
| All envs truncate together at `max_episode_steps` (v2 configs: 250 = 4.2 s) | no per-env `reset_idx` | base env |
| A succeeded or violating env **freezes**: action held, reward exactly 0 | it can't reset alone | `FreezeMask` |
| `terminated` fires **once**, on the freeze edge (`FreezeMask.update` returns the edge) | `PPO_RNN` zeroes LSTM state and cuts GAE on *every* `terminated` | `harvest_outcome` |
| Invalid envs (failed IK grasp) freeze at reset and never emit an edge | they carry no signal | env `_detect_invalid_envs` |
| The wrapper auto-resets the batch at the time limit and returns the new episode's first obs; `state()` pairs with it | skrl's trainer never calls `reset()` for `num_envs > 1` and calls `state()` right after `step()` | `HarvestSkrlWrapper` |
| The time-limit step is reported `terminated = truncated = True`; `time_limit_bootstrap=False` | skrl's GAE cuts only at `terminated`, and `step_frac` is observed (by the critic only in v2), so time is part of the MDP; otherwise the last step would bootstrap from the next episode's reset value | wrapper |
| LSTM sequences split at `terminated \| truncated` *inside* a stored sequence and zero state there, exactly as in the rollout | step/sequence equivalence | `models._RecurrentTower` |
| Non-finite env rows are zeroed, get reward 0, and are counted in `Step / nonfinite envs` | one blown-up world must not NaN the batch | wrapper |

Frozen envs still produce samples (with reward 0). Their actions are overridden, so the
policy-gradient contribution is zero-mean noise, and the critic sees `frozen` so it can
learn V ≈ 0 there. If frozen samples start to hurt, the fallback is a masked `PPO_RNN`
subclass (plan, *Risks*).

## 5. Observations and critic state

**Actor, v2 (`action_frame="tool"`, 40-D, `harvest_obs._ACTOR_TOOL_FIELDS`):** everything is
relative to the episode start and expressed in the **current TCP frame**, so the policy cannot
memorise where the apple sits in the world:

| field | width | meaning |
| --- | --- | --- |
| `d_pos_tool` | 3 | TCP displacement since the episode start |
| `d_rot_tool` | 3 | rotation since the start, body-frame rotation vector |
| `gravity_tool` | 3 | world down: encodes the grasp direction relative to gravity |
| `twist_tool` | 6 | TCP linear + angular velocity |
| `ft_tool` | 6 | wrist F/T; the sensor model (bias, drift, EMA, noise) runs in this frame |
| `target_offset_tool` | 6 | VIC setpoint minus TCP (position, orientation-error rotvec) |
| `last_action` | 13 | previous action, env units |

No joint angles, no absolute pose, **no clock**. The rig tares F/T before grasping, like the sim.
The world-frame fields stay in the obs dict for logging and screening (`obs["ft_wrist"]` is the
sensor reading rotated back). **v1 actor (`action_frame="world"`):** absolute `tcp_pos`,
`tcp_quat`, `tcp_velocity`, `ft_wrist`, `robot_joint_q`, `last_action`, `step_frac`.

**Critic (`rl/critic_state.critic_state_layout(junction_names, frame)`):**

1. the actor vector (exact prefix);
2. `_PRIVILEGED_FIELDS`: log10 spur/stem moduli, damping ratios, primary density, log10 support kp /
   roll_kp, support ζ, per-reset arm armature / friction / damping, build-time arm scales, and (D17)
   the episode's `detach_scale` and `realized_k_scale` (6);
3. every junction's raw wrench (sorted junction names);
4. plant geometry: spur/stem length and radius, apple radius and density, the **realized** grasp
   axis (`weld_direction` = TCP -z at reset), and `grasp_roll`;
5. raw wrist wrench, target wrench, detach index, success-streak fraction, `frozen`, `invalid`;
6. (tool layout) `step_frac`. skrl's GAE cuts only at `terminated` and the wrapper reports the time
   limit as `terminated = truncated = True` with `time_limit_bootstrap=False`, so the time limit is
   part of the MDP and the critic must see time even though the actor does not.

**The stored weld direction is not the realized grasp (finding, 2026-10-02).** On 16 v2 worlds the
TCP +z points at the apple centre, but 26 deg (median) and up to 68 deg away from the world set's
sampled `weld_direction`. Cause: the build applies `weld_direction` in the apple body frame, which is
not world-aligned (the realized axis equals `R_apple · weld` exactly). The critic and the baselines
use the realized axis.

The actor model reads only `inputs["observations"]` and the critic only `inputs["states"]`.
Checkpoints store both layouts (the wrapper's actual actor layout) and refuse a mismatch.

## 6. Domain randomization

| Axis | What | When | Where it comes from | In critic |
| --- | --- | --- | --- | --- |
| Plant materials | spur/stem flexural + axial E, damping ratios, primary density, geometry (real-data fixture, CMA μ ± 1σ) | per env, **baked at build** | world set | yes |
| Support joints | kp, roll_kp (log-uniform), ζ | per env, baked into the build *and* the settled snapshot | world set | yes |
| Grasp direction | v3: **60 deg cone about world -y** (equator, robot side), i.i.d. area-uniform per world (`--grasp-sampling random`); v1/v2: 30 deg about straight down on a fixed Fibonacci lattice (the v2 set has 192 distinct grasps) | per env, at build | world set (`--grasp-pole`, `--grasp-max-polar-deg`) | realized axis |
| Grasp roll (D17) | uniform ±90 deg about the approach axis (a parallel gripper repeats every 180 deg), on its own RNG stream (D17j) | per env, at build | world set (`weld_roll_rad`, `--grasp-roll-max-deg`) | yes |
| Arm, build-time | link mass/inertia ×[0.9, 1.1], EE payload ×[0.95, 1.05] | per env, once after build | world set | yes |
| Arm, per episode | joint armature ×[0.7, 1.3], Coulomb friction ×[0.5, 1.5], viscous damping ×[0.7, 1.3] | **every reset** | `_resample_joint_dynamics_dr` | yes |
| Realized stiffness (D17) | applied K = U(0.8, 1.2) × commanded, per axis (damping stays commanded) | **every reset** | `realized_k_scale_range` | yes |
| Detach force (D17) | `F_d` × U(0.85, 1.15) | **every reset** | `detach_scale_range` | yes |
| F/T sensor | 8.2 Hz causal EMA + per-episode bias + measured rig noise + bounded drift, in the TCP frame (v2) | every reset (bias, drift) / step (noise) | `FtSensorConfig.rl_training()` | raw F/T yes |

**Post-grasp hold (D17).** The snapshot is settled with the arm holding 500 N/m / 30 N·m/rad (was
100 / 10; damping ratios kept). The rig's dataset gains are 100 N/m / 30 N·m/rad, so the linear hold
is 5x the rig's.

Not randomized yet: action latency; null-space `kp_null` / `kd_null` per env; the plant base pose
(batch-scalar -- less critical now that the actor's observations are start-relative and in the TCP
frame).

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
| discount γ / GAE λ | 0.99 / 0.95 (v2: γ = 0.995, a ~200-step horizon on 250-step episodes) |
| clip (ratio / value / grad-norm) | 0.2 / 0.2 / 1.0 |
| entropy | 0 |
| scaling | running standard scaler on obs, state and value |

`TrainConfig.validate()` enforces the sequence and minibatch constraints `PPO_RNN`
imposes: rollouts must be a multiple of the BPTT length, and minibatches must hold whole
sequences.

## 8. Commands

Environment: `uv sync --extra gym --extra vic --extra dev --extra rl`.

```bash
# CPU smokes: PPO learns the v1 surrogate and the v2 bend-pull surrogate (~1 min, ~40 s)
uv run python -m apple_pick_gym.rl.train_vic_harvest --config apple_pick_gym/rl/configs/surrogate_smoke.json
uv run python -m apple_pick_gym.rl.train_vic_harvest --config apple_pick_gym/rl/configs/surrogate_smoke_v2.json

# GPU v2 training (needs the v3 world set + snapshot; see apple_pick_gym/world_sets/README.md)
uv run python -m apple_pick_gym.rl.train_vic_harvest --config apple_pick_gym/rl/configs/sim_train_gpu_v2.json --wandb

# Resume after a crash (fresh process; newest checkpoint; wandb run id is kept)
uv run python -m apple_pick_gym.rl.train_vic_harvest --config <cfg.json> --resume latest

# Baselines and checkpoint evaluation -> metrics JSON
uv run python -m apple_pick_gym.rl.eval_vic_harvest --config <cfg.json> --baseline scripted_bend_pull --episodes 1 --out <dir>/bend_pull.json
uv run python -m apple_pick_gym.rl.eval_vic_harvest --checkpoint <run>/checkpoints/ckpt_<t> --episodes 1 --out <dir>/policy.json

# v3 world set: screen candidates with the equator cone (one batch per process), then rescreen
uv run python apple_pick_gym/batched_examples/example_screen_harvest_worlds.py sample --num-envs 128 --seed <s> \
    --grasp-pole 0 -1 0 --grasp-max-polar-deg 60 --grasp-roll-max-deg 90 --grasp-frame world --grasp-sampling random \
    --out tmp/harvest_worlds_v3/pass1.jsonl
```

Outputs under `--run-dir`: `config.json`; `metrics.jsonl` (one `"update"` row per PPO update, one
`"episode"` row per episode); TensorBoard events (skrl), and wandb with `--wandb`;
`checkpoints/ckpt_<timestep>/{agent.pt, meta.json}` (policy, value, optimizer, the three scalers;
layouts, action bounds, RNN spec, timestep, updates, wandb id, git SHA and config).

Baselines (`rl/baselines.py`):

- `zero`: hold; `random`;
- `scripted_pull`: retreat along the grasp axis at 2 mm/step, K = 150, ζ = 0.9. In the tool frame
  that is `-z` of the TCP. **With an equator grasp it is horizontal, so it already swings the stem**
  (surrogate: theta_b ~39 deg at detach);
- `scripted_bend_pull` (D17): move the apple 25 mm horizontally to bend the stem, then pull at 40 deg
  from straight down along the bend; `scripted_down_pull`: the same script with no bend and a
  straight-down pull, the axial no-bend reference. Both use only `gravity_tool` from the actor obs;
- `scripted_twist_pull` (v1).

**Gate (`rl/gate.py`).** v2 (pass `--bend-pull`): success >= `scripted_pull`'s; collateral per
successful pick <= the scripted bend-pull's (ratio 1.0); success >= random's; collateral strictly
below random's. The safety clause is dropped (no trip in v2). Without `--bend-pull` the v1 gate
(D2/D2a/D6: success and safety vs the pull, collateral <= 0.5x the pull's, the random clauses)
applies.

```bash
uv run python -m apple_pick_gym.rl.gate --policy eval/policy.json --scripted-pull eval/pull.json \
    --bend-pull eval/bend_pull.json --random eval/random.json --out eval/gate.json
```

## 9. The surrogate envs

`rl/surrogate_env.py` holds two analytic plants with the real env's RL contract (obs dict incl. the
tool-frame fields, `info` keys, action bounds and leash, DR split and privileged layout, the shared
`evaluate_harvest_step`). They run thousands of env-steps per second on a CPU and prove
infrastructure; **their numbers mean nothing about the plant.**

- `SurrogateHarvestEnv` (v1, envelope criterion): springs around a grasp from below; twist-and-pull
  reaches the envelope with less collateral than a straight pull.
- `SurrogateBendPullEnv` (D17, `detach_criterion="bend_pull"`): grasps from the equator cone with the
  TCP +z into the apple; the stem runs from a fixed junction to its attachment on the apple, with an
  axial spring on its stretch and a soft lateral spring on its swing. Bending lowers `F_d`, so
  bend-then-pull detaches at < 0.5x the tension of a straight-down pull, with less collateral
  (tested). PPO on `surrogate_smoke_v2.json` goes 0.84 -> 1.00 success and 46 -> 16 steps per pick in
  ~35 s, detaching at theta_b ~45-50 deg with ~6.5 N of tension.

Both support the tool-frame action and gains, realized-K DR and the TCP-frame sensor model. They
integrate damping implicitly (in the gain frame).

## 10. Known limitations and open issues

- **Grasp frame (fixed, D17i).** The batched build used to apply the world-frame `weld_direction` in
  the apple body frame (rotated a median 118 deg from world), so side grasps landed at the wrong
  azimuth. `weld_direction_frame="world"` (`WorldSpec.weld_frame`, `--grasp-frame world`) references
  the settled apple; realized vs sampled is now 0.6 deg median. v1/v2 world sets keep the legacy
  frame (`"apple"`), so their snapshots still pair.
- **v3 screening pilot with the legacy frame** (2026-10-02, superseded by the fix). Pilot: 48% of candidates pass
  pass 1 (v2: ~91%), with a strong elevation bias: grasps above the equator pass 21%, below it 74%.
  About 27% fail at rest at every elevation (failed grasp: TCP 10-49 cm off, 30-200 N at the wrist);
  the stiffer hold raises that modestly (14/32 vs 10/32 at the old 100/10 hold on the same worlds).
  Above-equator grasps additionally fail the hold / pull probes. Decision pending (maintainer):
  accept the bias, stratify, or narrow the cone. Details in the decision log (D17).
- **Rig-side changes needed for deployment:** per-axis stiffness along the TCP axes (the rig's
  `compute_pose_task_wrench` is base-frame per-axis today); observations rotated into the TCP frame
  with the start pose stored at grasp; force limits from the rig, since v2 training does not
  penalise force.
- **The real env needs CUDA.** On a CPU Warp device the batched FR3 arm does not integrate (Newton's
  MuJoCo-CPU backend, `separate_worlds`); CPU is wiring only.
- **Action latency is not modelled.**
- **In-process rebuild heap corruption** (`docs/in-process-rebuild-heap-corruption.md`): one sim
  build per process; long runs resume in a fresh process from `--resume latest`.

## 11. Validation

```bash
# fast RL + harvest suite (no sim builds)
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_rl_*.py apple_pick_gym/tests/test_harvest_*.py \
  apple_pick_gym/tests/test_sensor_realism.py apple_pick_gym/tests/test_world_set.py apple_pick_gym/tests/test_train_vic_harvest_cli.py \
  apple_pick_gym/tests/test_eval_vic_harvest_cli.py -q -m "not slow"
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_vic_wrench_tool_frame.py -q
# CPU learning smokes (v1 ~1 min, v2 ~40 s)
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_rl_smoke_training.py -q
# real-env builds: one test per process
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_harvest_env_rl_contract.py -q
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_harvest_env_bend_pull.py -q
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_harvest_env_tool_frame.py -q
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_rl_sim_wiring.py -q
```
