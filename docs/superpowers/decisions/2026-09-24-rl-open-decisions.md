# [M5] open decisions taken by the cloud session (2026-09-24), for maintainer review

The maintainer handed over on 2026-09-24 with "make the open decisions and mark the commits; we'll
discuss tomorrow". Each decision below has an ID. Every commit implementing one is tagged `[Dn]` in
its subject line (`git log --oneline --grep '\[D'`). Each entry lists the choice, the evidence,
the alternatives, and how to revert it. Nothing here changes the reward weights or PPO
hyperparameters beyond what is listed.

| ID | Decision | Status |
| --- | --- | --- |
| D1 | Detach signal: the envelope reads the stem-root *elastic* wrench, not the rigid-junction readout | done; GPU-confirmed |
| D2 | Task 11 gate: success AND safety AND collateral vs scripted pull, AND >= random | done |
| D2a | Gate also requires peak collateral strictly below random's | done |
| D6 | Gate compares collateral per *successful* pick | done |
| D7 | Per-env F_max / tau_max draws | parked (plumbing, off) |
| D8 | Cap VIC target speed to the real rig (2 mm/step, 0.01 rad/step at 60 Hz) | done |
| D9 | Reward rebalance so a pick beats standing still | done |
| D10 | LR floor 1e-4 for the KL-adaptive schedule | superseded by D10b |
| D10b | KL-adaptive LR bounded to [3e-5, base LR] | done |
| D11 | Smooth wrist-force cost above a 25 N soft cap | done |
| D12 | Solver blow-up guard (> 200 N or non-finite): freeze, no penalty, invalid | done |
| D3 | F/T sensor model matched to the real rig: noise and online EMA corner (8.2 Hz) | done |
| D4 | F/T observation frame for deployment (sim is world frame; rig is mixed) | flagged, no code change |
| D5 | Random still reaches the envelope through force (0.81): keep leash / K range / F_max, rely on the D2 gate | decided, no code change |

## Morning brief (for the 2026-09-25 discussion)

**State.** The RL infrastructure is in place.
- On CPU, the surrogate env learns: success goes from 0.02 to 1.00 in ~10 episodes.
- The real env is wired end to end on GPU: N=2000 at ~5200 env-steps/s.
- D1 is GPU-confirmed.
- No learned policy has been trained on the real env yet. Rewards and PPO hyperparameters are
  untuned, as agreed.

**To discuss (not decided).**
1. **Torsion is unmodelled.** Newton's cable joint has one angular stiffness for bend and twist,
   identified from pulls only. Twisting therefore barely loads the junction (D1 (b)), and
   twist-and-pull cannot emerge as a strategy. Fixing this needs a separate torsional stiffness
   plus twist data.
2. **Grasp pre-load.** The grasp leaves 16-45% of tau_max on the junction at rest (D1 (a)). Is
   that acceptable, or should the grasp be relaxed or the pre-load subtracted?
3. **Random succeeds at 0.81 through force** (D5). Keep the action bounds (my choice) or shrink the
   leash / K range?
4. **Collateral per successful pick** (D6, done). Open question: should failed episodes' collateral
   also be bounded? Today only the safety rate covers them.
5. **F/T frames on the rig** (D4): the tare mixes the K and O frames.
6. **Real data**: s05-d05 and s05-d07 look like duplicates.

7. **Pre-existing CPU test failure (not from this branch).**
   `apple_pick_gym/tests/test_apple_pick_coupled_env.py::test_vic_env_tcp_moves_under_velocity_command`
   fails on CPU: the TCP moves dx = -0.024 m instead of > +0.05 m. It fails the same way at the
   merge-base c41070e (dx = -0.023 m), before any M5 work. It is likely related to the MuJoCo-CPU
   arm path (H6 §10). I left it alone; it is out of M5 scope and tests are never skipped.

**Suggested next GPU steps.**
- Post-D1 baselines on one held-out snapshot: `eval_vic_harvest --baseline {random,scripted_pull}`.
- Then the first real-env training run (`configs/sim_smoke_gpu.json`), watching success,
  collateral and the D2/D2a gate.

## D1 -- The detach envelope reads the stem-root elastic wrench

**Problem (GPU, local session).** Under the total-moment envelope, a random policy and a
do-nothing-then-jiggle policy "detach" at 100%. Torque dominates the index, and the zero policy
alone crosses the envelope 8 times in 128 steps.

**Diagnosis (CPU, real env, arm frozen, zero action).**
1. Torque jumps of 0.033 N*m (step-to-step p99) occur while the apple moves 0.0003 mm and rotates
   0.0007 rad per step (p99).
   - Through the model's stem stiffness (0.017-0.025 N*m/rad per segment) that rotation explains
     ~1e-5 N*m.
   - So neither torsional nor bending stiffness is the cause, tuned or not.
2. The same readout without the AVBD penalty-damping term is identical, so damping is not the cause.
3. The rest *level* (~0.01-0.02 N*m) is real. The grasp leaves the very soft stem deformed by
   ~0.47-0.65 rad per cable joint, and stiffness x deformation gives 0.008-0.017 N*m.
- Conclusion: the ~0.02 N*m level is real stem elasticity. The +-0.03 N*m jitter comes from reading
  a stiff penalty constraint (the rigid spur-stem fixed joint) whose sub-micron violations are
  multiplied by a huge stiffness.

**Choice.** `DetachEnvelopeConfig.wrench_source = "stem_elastic"` (default).
- The spur-stem wrench is taken from the first soft stem cable joint's wrench on its child segment,
  moved to the junction anchor by statics: `M_J = M_A + (A - J) x F`, lever ~2.3 mm.
- The ~1e-4 N weight of the one stem segment in between is ignored.
- `info["junction_readout_wrench"]` keeps the rigid-junction readout.
- F_max = 20 N, tau_max = 0.05 N*m, the total-moment envelope and the 3-step streak are unchanged.
  The noise is removed at the source, so no filter or threshold change is needed.

**Evidence (CPU prototype, 2 worlds x 50 steps).**
| | readout | stem elastic |
| --- | --- | --- |
| mean torque vector, world 0 | (0.0020, 0.0019, -0.0085) | (0.0016, 0.0017, -0.0079) |
| mean torque vector, world 1 | (0.0010, 0.0176, -0.0119) | (0.0017, 0.0187, -0.0123) |
| torque std | 0.0097 / 0.0082 | 0.00006 / 0.00025 |
| d tau p99 | 0.032 | 0.0005 |
| mean force | 4.29 / 5.23 N | 4.09 / 4.71 N |

The slow contract test asserts: force within 15% of the readout, mean torque within 0.01 N*m, and
max step change below 0.2x the readout's.

**Consequences worth discussing.**
- (a) The rest pre-load from the grasp already uses 16-45% of tau_max per world. It is physically
  real in this model, but it depends on how the grasp twists the stem.
- (b) The stem is so soft in rotation (twist shares the bend stiffness, which was only identified
  from pulls) that twisting the apple barely loads the junction. The envelope is in practice
  force-driven plus bending, and twist-and-pull cannot emerge until torsion is modelled and
  identified. That needs a separate torsional stiffness per stem segment (Newton's cable joint has
  one angular stiffness) plus twist data.
- (c) GPU confirmation (below): the torque loophole is closed, but random still reaches the
  envelope through *force* (see D5).

**GPU confirmation (local session, RTX 4090, 298ff94, `sim_wiring_gpu`, seed 12345, `detach_sweep`).**
Success under the default rule (total moment, raw, 3-step streak). The step-to-step torque p99 is
shown for the new signal and for the rigid readout.

| policy | success | success, tau_max 0.3 | success, force only | d tau p99, stem elastic | d tau p99, readout |
| --- | --- | --- | --- | --- | --- |
| zero | 0.00 | 0.00 | 0.00 | 0.0007 | 0.0276 |
| random | 0.81 | 0.69 | 0.67 | 0.0107 | 0.0933 |
| scripted_pull | 1.00 | 1.00 | 1.00 | 0.0133 | 0.1108 |
| scripted_twist_pull | 1.00 | 1.00 | 1.00 | 0.0057 | 0.0526 |

- Zero no longer detaches at any tau_max. The at-rest noise drops ~40x.
- The scripted policies succeed under every rule, so tau_max = 0.05 stays.
- Random still succeeds at 0.67 even with a force-only envelope, so it gets there by force.

**Alternatives.**
- Raise tau_max above the noise (~0.15-0.2): hides the jitter and makes the envelope effectively
  force-only.
- EMA the readout and lengthen the streak: keeps a noisy, non-physical signal.
- Force-only envelope: drops the torque term that the maintainer wants.

**Revert.** Set `wrench_source="junction_readout"` (EnvConfig / DetachEnvelopeConfig).

## D2 -- Task 11 exit gate includes collateral

**Choice.** `apple_pick_gym/rl/gate.py`. A learned policy passes only if, on the same held-out
snapshot and seed, all of these hold:
- success >= scripted pull;
- safety <= scripted pull;
- peak collateral <= 0.5 x scripted pull's;
- success >= random.

**Evidence.**
- On GPU (sim_wiring_gpu, 64 worlds), scripted pull detaches at 19 N junction force with 45 N peak
  collateral. The plan's gate ("beat scripted pull on success, no worse safety") would accept a
  policy that reproduces that.
- Random reached 1.00 success under the total-moment envelope, so beating random guards against
  an envelope a policy can trip by accident.

**Alternatives.**
- Success-only gate (the plan).
- A weighted score.
- Comparing against scripted twist-pull.

The 0.5 ratio is a judgement call. It demands a clear improvement in the thing the task is about.

**Revert.** Use `--collateral-ratio 1e9` and omit `--random`, or ignore the tool.

### D2a -- Collateral must also be strictly below random's

**Finding (local GPU, `eval_vic_harvest`, sim_wiring_gpu, seed 12345, 1 episode, pre-D1 e2b3fb5).**

| baseline | peak collateral | success | safety |
| --- | --- | --- | --- |
| random | 15.2 N | 1.00 | 0 |
| scripted_pull | 45.0 N | -- | -- |

- Random's collateral is 0.34x the scripted pull's, so random itself passed the D2 collateral
  clause. With equal success, random matched itself on every criterion and passed the whole gate.

**Choice.** When `--random` is given, add `collateral_vs_random`: the policy's peak collateral must
be strictly below random's. Random can no longer pass its own gate, and a learned policy must load
the tree less than untargeted flailing does.

**Caveats.**
- The data is one episode, pre-D1.
- Under D1, random detaches later (0.81), so its collateral will likely rise. The next
  `eval_vic_harvest --baseline random` on current code will update it.
- `peak_collateral_n_mean` averages over failed episodes too. A random run that often fails
  without pulling hard looks gentler than it is. If that bites, compare collateral on successful
  episodes only.

**Revert.** Drop the `collateral_vs_random` line in `gate.py`.

## D3 -- F/T sensor noise from real data

**Choice.** `FtSensorConfig.rl_training().noise_std = (0.12, 0.11, 0.12, 0.04, 0.05, 0.005)`, measured
on the real rig's quiet unloaded holds (s02, 32 segments, ~60 Hz block mean).
- Bias and drift remain estimates: they can't be measured from the sys-ID data.
- Torque bias was raised to 0.05 N*m (Tz 0.005).

**Evidence.** The local GPU session's Q3 table. The previous guess had Tx/Ty noise 10x too low.

**Where the noise is added (maintainer note: real F/T is EMA'd and low-passed during conversion).**
- 60 Hz block-averaging barely reduced the measured noise. White 1 kHz noise would drop ~4x, so
  the measured column was already low-passed.
- The measured std is therefore post-filter noise, which is where the sim adds it (after its
  causal EMA). The calibration is consistent.
- **Online filter (resolved).** From `real_robot_exps` code (local session): the rig reads Franka's
  external-wrench *estimate* `K_F_ext_hat_K` at 1 kHz, subtracts a bias, and EMAs it online with
  `ft_ema_alpha` = 0.05 (config since 2026-08-17; the s02 data is 2026-08-20). That gives
  fc = -ln(0.95) * 1000 / 2pi ~ 8.2 Hz.
  - `rl_training()` now uses `cutoff_hz=8.16` (was 10).
  - The collected `ft_wrist_raw` is that EMA'd signal, so the measured noise is post-filter, as
    modelled.
- The offline zero-phase `filtfilt` used for sys-ID scoring is non-causal. It is deliberately not
  modelled in the policy's observation.

**Revert.** `git revert be9465b`.

## D4 -- F/T frame convention (flagged for the maintainer; no code change)

**Finding (local session, from code).**
- The sim observes `ft_wrist` as the world-frame TCP coupling wrench.
- The sys-ID converter rotates real F/T into that frame (`R(tcp) @ F/T`, ROADMAP slice 1).
- The live rig path is not in that frame:
  - it reads `K_F_ext_hat_K`, which is stiffness-frame;
  - one read path only negates, another rotates base -> body;
  - dataset metadata says "force in EE frame, torque in base frame";
  - the per-episode bias tare averages `O_F_ext_hat_K` (base frame) and subtracts it from the
    K-frame signal. That is a frame mismatch whenever the K and O frames differ.

**Decision.**
- Keep the sim contract (world frame).
- Deployment must apply the same transform as the converter before feeding the policy.
- The rig-side tare frame mismatch is for the maintainer to confirm and fix on the rig. It also
  bears on the M4.0 torque-magnitude gate: a wrong torque frame or tare changes torque
  magnitudes.

## D5 -- Random reaches the envelope through force; keep the action bounds, gate on collateral

**Finding (GPU sweep under D1).**
- Random succeeds 0.81 under the default rule and 0.67 under a force-only envelope.
- Its random walk against the leash commands up to ~30 N: a 0.15 m position leash x K_lin up to
  200 N/m. That exceeds F_max = 20 N.

**Choice.** No change to the leash, the K range or F_max.
- In this task, pulling hard enough is easy. The hard part, which the policy must learn, is
  loading the spur-stem junction *without* loading the rest of the tree.
- The D2 gate already demands success >= random AND peak collateral <= 0.5x the scripted pull's.
  The reward already charges collateral.
- 30 N at 0.15 m is within what the real arm does in a pick, so shrinking the bounds would
  restrict the policy to make a baseline look worse.

**Alternatives (maintainer's call).**
- Shrink `max_target_pos_offset_m`.
- Lower `k_lin_max`.
- Raise F_max from its measured value.
- Each makes random fail more often, but none makes the task more like the real one.

**Open.** The sweep does not record collateral, so random's collateral versus the scripted
pull's is unmeasured. It comes out of the first `eval_vic_harvest --random` run, with no extra
GPU run.

**Revert.** Nothing to revert.

## D6 -- The gate compares collateral per successful pick

**Problem.** `peak_collateral_n_mean` averages over every episode, including failed ones that never
pulled. A policy that often fails, such as random at 0.81 under D1, looks gentler than it is, and
the D2a clause "strictly below random" becomes too easy or too hard for the wrong reason.

**Choice.**
- The wrapper logs `Episode / peak collateral N, successful (mean)`: the mean over valid envs that
  succeeded, NaN if none did.
- `eval_vic_harvest` reports it as `peak_collateral_n_success_mean`, with each batch weighted by its
  count of successful valid envs.
- `gate.py` uses it for both collateral clauses when every metrics JSON has it. Otherwise it falls
  back to the all-episode mean. The result records `collateral_metric`.
- A baseline with no successful pick sets no collateral bar. A policy with none fails its
  collateral clauses.

**Alternatives.**
- Keep the all-episode mean. It is biased toward policies that fail often.
- Bound both means. That is stricter, but it penalises failed-but-gentle attempts that the safety
  rate already covers.

**Revert.** Drop `_COLL_SUCCESS` in `gate.py`: the gate then uses the all-episode mean again. The
extra logged metric does no harm.

## Rest load check (maintainer: "verify zero-action torques are way below the max applied")

**Result: OK.**
- The grasp-only hold (zero baseline, GPU, N=2000, 500 steps, post-D1) peaks at a detach index
  of 0.125, i.e. ~1/8 of the envelope (utilization ~0.35).
- The pulling baselines reach it: junction force 18-20 N, torque p99 ~0.07 N*m.
- Rest torque at reset (CPU, 8 worlds, `rl/diagnose_rest_load.py`): median 0.0135 N*m (27% of
  tau_max), junction force 4.0 N.

**Statics (open, parked as small).**
- The hanging stem + apple weigh 2.6 N and put 0.006 N*m of gravity moment on the junction.
- The weld to the gripper carries 0.18 N. The direct weld-joint readout equals the wrist
  wrench, with its sign flipped.
- That leaves ~1 N (median) of junction force unexplained. The likely sources are
  apple/stem contacts or penalty bias in the readout.
- The maintainer judged the rest load acceptable, so it is not chased further.
- After 5 hold steps the picture is the same (steady, not a reset transient).
  - Torque balances within 0.006 N*m (median): gravity 0.006 N*m plus a couple of 0.014 N*m
    that the grasp applies through the weld. So the rest torque is the apple's weight plus a
    small twist or bend held in by the grasp.
  - Force is still ~0.9 N short.

## D7 -- Per-env F_max / tau_max (parked)

- Plumbing only (0a541b2): `envelope_thresholds`, per-env thresholds in `detach_index` and
  progress, and the surrogate env.
- Not wired into the real env, the critic or the config. Off by default.
- Revisit if the F_max / tau_max estimates stay uncertain once a policy learns.

## D8 -- Cap the VIC target speed to the real rig

**Finding.**
- The action allowed a 2 cm target step and a 0.1 rad target rotation per step at 60 Hz, i.e.
  1.2 m/s and 6 rad/s.
- The real sys-ID pulls (72 runs, 228k samples at 1 kHz) had TCP speed:

| TCP speed statistic | value |
| --- | --- |
| median | 0.014 m/s |
| p90 | 0.03 m/s |
| per-run peak, median | 0.055 m/s |
| per-run peak, max | 0.21 m/s |
| angular, p90 | 0.064 rad/s |
| angular, max | 0.55 rad/s |

- Random detached 94% of the time in ~75 steps by yanking the target around. Fast yanks are
  outside what the sim was identified on.

**Choice.** `EnvConfig.linear_delta_m = 0.002` and `angular_delta_rad = 0.01`, i.e. 0.12 m/s and
0.6 rad/s. That is below the rig's per-run peak speeds and leaves both scripted baselines
unchanged (they already run at exactly these rates). It is a hard bound, not reward shaping. The
current reward still favours speed (a per-step slack cost and a per-step collateral cost), and
the cap bounds that incentive without retuning weights.

**New metrics.** `Episode / peak TCP speed m/s (mean)` and `Episode / mean TCP speed m/s (mean)`,
also in `eval_vic_harvest` as `peak_tcp_speed_mps_mean` and `mean_tcp_speed_mps_mean`. A learned
policy should beat random on time to detach, junction force at detach, collateral and speed, even
if its success rate is a little lower.

**Consequence.** The random baseline now moves ~10x slower, so its success rate should drop. The
baselines need a re-run under D8.

**Surrogate opt-out.** `configs/surrogate_smoke.json` keeps the old 0.02 m / 0.1 rad bounds. Its
analytic plant is not rig-calibrated, and under the cap its 96-step episodes never reached the
envelope (the CPU learning smoke stalled at return -1.05). It checks the learning pipeline, not
realism.

**Revert.** Set `linear_delta_m=0.02` and `angular_delta_rad=0.1` in the config.

## First real training on the D1 signal (pre-D8 reference)

Run: `sim_smoke_gpu.json` @ f5a46a4, N=2000, 1536 steps (3 episodes), 24 updates. The run was
clean: 0 nonfinite envs, no NaN, 5.4 GB.

| episode end | success | safety | return | steps to success | junction F | collateral / success | peak wrist F |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t=500 | 0.920 | 0.080 | -166.7 | 85.8 | 18.6 N | 41.0 N | 20.9 N |
| t=1000 | 0.848 | 0.152 | -58.6 | 68.6 | 18.5 N | 29.6 N | 26.2 N |
| t=1500 | 0.902 | 0.098 | -36.8 | 50.4 | 19.7 N | 30.4 N | 24.4 N |
| eval (ckpt 1536) | 0.905 | 0.095 | -16.9 | 43.3 | 19.5 N | 26.8 N | 22.7 N |

**Gate vs the pre-D8 baselines.**
- Passes only collateral_vs_random: 26.8 < 42.0 N.
- Fails success (0.905 vs 0.998), safety (0.095 vs 0.002), collateral (26.8 vs 22.2 N) and
  beats_random (0.905 vs 0.941).

**Read.**
- It learns on the real signal. Detaches take half the steps, collateral per success drops by a
  third, and the pull-out penalty mostly disappears.
- Safety is the problem: 8-15%, with the wrist force rising. That fits fast target yanks
  under the old 1.2 m/s bound, which D8 now caps.
- The KL-adaptive schedule had already cut the LR to 9e-5 by update 24. Watch that in the long
  run; the update row has no KL key yet.

## Baselines under D8 (sim_train_gpu.json, N=2000, 1 episode, d5b9e93)

| baseline | success | safety | steps | junction F | collateral / success | peak TCP v | mean TCP v |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero | 0.000 | 0.000 | 500 | 3.8 N | -- | 0.001 m/s | 0.000 m/s |
| random | 0.053 | 0.013 | 364 | 9.1 N | 35.8 N | 0.060 m/s | 0.014 m/s |
| scripted_pull | 0.998 | 0.002 | 31 | 18.6 N | 44.4 N | 0.102 m/s | 0.072 m/s |
| scripted_twist_pull | 0.999 | 0.001 | 52 | 17.8 N | 42.4 N | 0.064 m/s | 0.042 m/s |

**Read.**
- D8 removes the yank loophole: random falls from 0.941 to 0.053, and the scripted pulls are
  unchanged.
- Gate bars under D8:
  - success >= 0.998 and safety <= 0.002 (scripted pull);
  - collateral per success <= 22.2 N (0.5x scripted pull);
  - collateral < 35.8 N (random). That bar comes from only ~5% of envs, so it is noisy, but the
    0.5x-scripted clause is the binding one anyway.

## D9 -- Reward rebalance: a pick must beat standing still

**Finding (D8 long run, stopped after 3 episodes).**
- Return rose while junction force fell and success stayed ~2%: PPO was learning not to pull.
- Per-episode term sums (GPU eval_d8, old weights):

| policy | progress | pull-out | collateral | slack | terminal | return |
| --- | --- | --- | --- | --- | --- | --- |
| zero | 0 | -26 | -6 | -5 | 0 | -37 |
| scripted_pull | 0.68 | -81 | -56 | -0.3 | +10 | -126 |
| pre-D8 learned | 0.67 | -2 | -22 | -0.4 | +7 | -17 |

So the reward ranked zero above scripted_pull. Three causes:
- Pull-out charged every newton along the grip axis, even the 0.1 N at rest (-26 over 500 steps).
- Collateral summed every step with weight 0.1.
- The one-off +10 bonus and ~0.7 of progress were too small to pay for either.

**Choice (training config `EnvConfig`; the library `HarvestRewardConfig` defaults are unchanged).**
- `pullout_threshold_n = 10`: pull-out counts only beyond a grip-capacity stand-in.
- `w_collateral = 0.02`.
- `w_progress = 10`: delta progress telescopes to 10 * (u_end - u_0), so it cannot be farmed.
- `success_bonus = 20`, `failure_penalty = -40`.
- `w_slack` and `w_pullout` unchanged.

Re-weighting the same sums gives: clean learned pick ~ +16, scripted_pull ~ +10, zero ~ -6, and
random far below. A test pins that ordering.

**Open.** 10 N is a stand-in, not a measured grip capacity. The maintainer should set it from the
gripper.

**Revert.** Set the old values in the config: (1, 0.5 @ 0 N, 0.1, 10, -20).

## D10 -- LR floor for the KL-adaptive schedule

`PPOConfig.kl_adaptive_min_lr = 1e-4`. The KL-adaptive schedule took the LR from 3e-4 to 4e-5
within 18 GPU updates (KL 0.014 > 0.01 target), and skrl's default floor is 1e-6. That makes
escaping any plateau slow. With the floor, the schedule still adapts above it. KL per update is
now logged (`Policy / KL (mean)`).

## D9 long run: per-junction statics, torque, and why it degraded

**Per-junction peak forces (GPU, N=2000).**

| policy | primary_spur | spur_stem | stem_apple | each support | collateral / success |
| --- | --- | --- | --- | --- | --- |
| scripted_pull | 20.8 N | 18.8 N | 17.9 N | 13 N | 44.4 N |
| D9 ckpt_2560 | 19.6 N | 17.7 N | 17.1 N | 12.5 N | 42.2 N |

Series statics: `stem_apple` and `primary_spur` each carry about the target force. So collateral
tracks the detach *force*. It can only fall if the policy detaches with less force, i.e. with
torque. The torque at the target is ~0.03 N*m for every policy (the grasp preload): nobody bends
or twists yet. The maintainer keeps collateral as the objective.

**Degradation.**
- Success peaked at EP6: 0.986, safety 0.4%, 97 steps.
- By EP10 it had fallen to 0.879 success, 6.1% safety, 177 steps. Collateral stayed flat.
- The LR had climbed to 5e-4, above the 3e-4 base: skrl's KLAdaptiveLR multiplies the LR by 1.5
  whenever KL < target/2, with a default max_lr of 0.01.

## D10b -- Bound the KL-adaptive LR to [3e-5, base LR]

- `kl_adaptive_min_lr = 3e-5`. Pinned at 1e-4, the KL ran at ~3.5x target.
- `kl_adaptive_max_lr = None` means the ceiling is the base `learning_rate`.
- The schedule can now only slow learning below the tuned base, never push it above.

## New episode metrics for torque at detach

- `Episode / detach {force N, torque N*m, torsion N*m, bending N*m, force share, torque share} (mean)`:
  target junction values at the detach step, averaged over successful envs. The shares are
  (F/F_max)^2 and (tau/tau_max)^2.
- `Episode / peak target torque N*m (mean)`.
- In the eval JSON: `detach_*_mean` and `peak_target_torque_nm_mean`.

## Torque at detach: twist does not load the junction in this model (maintainer decision needed)

Eval, GPU, N=2000, 984979c, sim_train_gpu.json:

| policy | success | detach F | detach tau | torsion | bending | force share | torque share | collateral / success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scripted_pull | 0.998 | 18.5 N | 0.027 N*m | 0.009 | 0.024 | 0.87 | 0.32 | 44.4 N |
| scripted_twist_pull | 0.999 | 17.7 N | 0.026 N*m | 0.009 | 0.023 | 0.80 | 0.31 | 42.4 N |
| D9 ckpt_3840 (best) | 0.935 | 17.7 N | 0.026 N*m | 0.009 | 0.023 | 0.80 | 0.31 | 41.7 N |
| D9 ckpt_5120 (last) | 0.860 | 17.1 N | 0.027 N*m | 0.010 | 0.024 | 0.75 | 0.34 | 38.9 N |

**Gates (vs eval_d8).** Both checkpoints pass only beats_random (1 of 5 criteria).

**Read.**
- Deliberate twisting gives the same junction torque as a straight pull, ~0.026 N*m, which is
  the grasp preload.
- The stem is a soft cable with one shared bend/twist stiffness, identified from pulls only.
  Rotating the apple just rotates the stem, and almost no moment reaches the spur-stem junction.
- So every policy detaches by force, ~80% of the envelope. Series statics then put collateral
  at ~2.2x the detach force, ~40 N.
- In this model, collateral cannot drop much below that however long we train. The D2
  collateral target (<= 22 N) is out of reach. So is a lower-force twist-and-pull, which is
  the behaviour the maintainer wants to find.

**Options for the maintainer.** This is sim modelling, not RL.
1. Give the stem's first segments (the abscission zone) a separate, stiffer torsional and
   bending stiffness, so twist and bend load the junction. That needs a joint model with
   separate twist stiffness, and twist data to identify it.
2. Keep the model. Accept the force-detach floor, re-base the collateral gate on it (e.g. <=
   scripted_pull's), and judge the policy on time, safety and force.

**Also.** `collateral_vs_random` compares against random's collateral from its ~5% successful
envs, a biased subset. Treat that clause as advisory.

## Correction: a torque-dominant detach IS reachable (ep250 run, episodes 1-4)

The section above concluded that torque is not a usable lever. The ep250 run (984979c, 250-step
episodes, D10b LR bounds) proved that wrong for **bending**:

| t | success | safety | steps | collateral / success | detach F | detach tau | force share | torque share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 250 | 0.055 | 0.001 | 222 | 34.4 N | 15.8 N | 0.035 | 0.64 | 0.51 |
| 500 | 0.032 | 0.015 | 209 | 18.8 N | 12.3 N | 0.042 | 0.45 | 0.74 |
| 750 | 0.280 | 0.081 | 186 | 20.1 N | 12.2 N | 0.042 | 0.42 | 0.72 |
| 1000 | 0.343 | 0.305 | 135 | 11.4 N | 10.5 N | 0.047 | 0.33 | 0.90 |

- Collateral per success falls to 11 N, below the D2 22 N target, via a torque-dominated
  detach.
- The scripted twist does not do this; whatever the policy does (likely bending the stem) does.
- Safety violations climb to 30%. New metrics show which cap trips: `Episode / safety {target
  force, target torque, wrist force, wrist torque} (frac)`, plus peak wrist torque. The eval JSON
  has `safety_*_frac` and `peak_wrist_torque_nm_mean`.

## D11 -- Smooth wrist-force cost above a 25 N soft cap

**Finding.** Side eval of ep250 ckpt_1600 at 3a6e225, N=2000.

| safety cap | fraction of envs tripping it |
| --- | --- |
| wrist force | 10.1% |
| target force | 3.3% |
| wrist torque | 0.05% |
| target torque | 0 |
| total | 13.5% |

- Peak wrist force 23.4 N, peak wrist torque 3.7 N*m, peak target force 11.3 N.
- At detach: F 10.4 N, tau 0.046 N*m (bending 0.040, torsion 0.019), torque share 0.85.
  Success 0.41, collateral per success 10.2 N.

**Read.**
- The low-collateral strategy bends the stem by pushing laterally at the gripper.
- The wrist sees ~2x the junction force. With 0.2 N of weld force at rest, the difference most
  likely goes through contact: the apple pressed against the branch.
- Pull-out charges only the grip axis, so lateral pushing was free up to the 40 N cliff.

**Choice (training config).** A new dense term, `wrist = relu(|F_wrist| - 25 N)`, weighted
`w_wrist = 0.5` per step. That is the same slope as pull-out, but on total force. It is a smooth
cost starting 15 N below the safety cliff.
- The library `HarvestRewardConfig` default is `w_wrist = 0` (off).
- It does not affect the D9 ordering: the scripted pull's wrist force peaks at ~15 N.
- Logged as `Episode / reward wrist (sum)` and `reward_wrist_sum`.

**Open (maintainer).** The ~12 N the wrist carries beyond the junction is probably apple-branch
contact. Pressing the fruit into the tree bruises it in reality, and its realism depends on the
contact model. Worth checking contact forces before trusting the bend strategy.

**Revert.** `w_wrist = 0`.

## Contact check: the bend detach does not press the fruit into the tree

`rl/diagnose_contacts.py` (de605e2), GPU, N=2000, 1 episode. Episode peaks, median / p90:

| policy, envs | fruit-woody contact | fruit-gripper contact | wrist force |
| --- | --- | --- | --- |
| ep250 ckpt_1600 (bend), successful (519) | 0 / 0 N | 0 / 0 N | 16.2 / 31.5 N |
| ep250 ckpt_1600 (bend), all | 0 / 6.6 N | 0 / 0 N | 23.7 / 39.8 N |
| scripted_pull, successful (1987) | 0 / 0 N | 0 / 0 N | 20.3 / 23.4 N |

- Every world has 8-10 fruit-woody contact pairs (not filtered), but they carry ~no force in
  successful picks.
- The bend strategy does not rely on pressing the apple into the branch.
- The wrist-junction gap (16 N vs 10.4 N at detach) fits statics: the stem + apple weight
  (~2.6 N), plus arm dynamics and peaks that are not simultaneous.

**Gate of the pre-D11 bend checkpoint (vs eval_d8): 3 of 5.**
- Pass: collateral 10.2 <= 22.2 N; beats_random; collateral_vs_random.
- Fail: success 0.41 and safety 0.13. D11 targets exactly these.

**Flag: isolated solver blow-ups.** In both policies a few single worlds show absurd values:
fruit-woody up to 1923 N, fruit-proxy 842 N, wrist 183 N. The medians are unaffected, but such
worlds can count as safety failures and inject huge rewards. The non-finite guard only catches
NaN/inf. Candidate follow-up: flag worlds whose plant wrench jumps beyond a physical bound as
invalid for the rest of the episode.

## D12 -- Solver blow-up guard

**Finding.** Trip-jump eval of D11 ckpt_1600 (ba83dd8, N=2000):

| metric | value |
| --- | --- |
| target force at trip, median | 41.0 N |
| target force one step before, median | 6.5 N |
| trips with a >5x one-step jump | 45% |
| total safety (eval) | 5.5% (vs 24.8% in training rollouts: exploration noise) |

The local agent read the jumps as numeric. I don't think they are:
- The trip force is barely over the cap (41 vs 40 N). A 6.5 -> 41 N rise in one 60 Hz step
  fits a physical snap (the bent stem going taut), and the policy should learn to avoid it.
- The real numerics are the rare absurd values (contact 1.9 kN, wrist 183 N in single worlds).

**Choice.**
- `EpisodeConfig.blowup_force_n = 200` (5x the cap). A target-junction or wrist force above it,
  or a non-finite one, marks a blow-up in `evaluate_harvest_step`.
- The world is frozen (terminated edge) with reward 0: no failure penalty, no success, no dense
  term from the bad readout.
- Episode stats count the world as invalid, so it is excluded from success/safety rates.
  `Episode / blowup fraction` (eval: `blowup_fraction`) reports how often it happens.
- The 40 N safety cap is unchanged: real overshoots and snaps stay failures.

**Revert.** `blowup_force_n = None`.
