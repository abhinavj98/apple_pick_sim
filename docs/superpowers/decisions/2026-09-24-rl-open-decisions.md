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
| D13 | Charge -0.5 x episode peak collateral at the success edge | done |
| D14 | Hold a blown-up world's last good obs/state (keeps the obs scaler clean) | done |
| D15 | Fix skrl PPO_RNN storing h_{t+1} for row t (rnn-state dict aliasing) | done, GPU-confirmed |
| D16 | tanh-squash the actor mean into the action box | superseded by D16b |
| D16b | Actor mean bounded at +-1.5 (1.5 tanh(x/1.5)) | done |
| D8b | Max target speed 0.4 m/s, 0.3 rad/s (maintainer) | done |
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

## D13 -- Charge the episode's peak collateral at the success edge

**Finding (D11 run, ep250_d11 at d08e65d).**

| episode | success | safety | collateral / success | torque share at detach | detach F |
| --- | --- | --- | --- | --- | --- |
| EP5 | 0.35 | 25% | 10.4 N | 0.92 | 10.4 N |
| EP10 | 0.43 | 16% | 15.1 N | 0.84 | 11.4 N |
| EP15 | 0.60 | 12% | 27.2 N | 0.62 | 14.2 N |

- The policy drifts from the bend detach back toward pulling.
- D11 made the lateral push expensive, while collateral is only charged per step at 0.02. That
  term mostly measures time, not the pick's load.
- So the objective the maintainer cares about, collateral per successful pick, was barely in
  the reward.

**Choice.** `w_peak_collateral = 0.5`. At the success edge only, the terminal reward becomes
`success_bonus - 0.5 x (episode peak collateral, N above rest)`.
- Pull (~44 N): 20 - 22 = -2 at success, ~ +5 with progress (still > zero's -6).
- Bend (~10 N): 20 - 5 = +15.
- The running peak is tracked like `progress_prev`: `peak_collateral_prev` in
  `evaluate_harvest_step`, reset per episode in both envs.
- The library `HarvestRewardConfig` default is off.

**Revert.** `w_peak_collateral = 0`.

## D14 -- KL spikes: the obs scaler, not the LSTM. Hold blown-up worlds' last good rows

**Finding (D11 run, local agent).** KL(mean) per update spiked to 0.19-0.33, and once 9.09, with
the LR pinned at the 3e-5 floor and std flat. A real policy change that large is impossible at
that LR, so the recomputed log-probs disagreed with the rollout's.

**Tests (`test_rl_logprob_consistency.py`).**
- Recurrent path: 13-step episodes against 8-step BPTT sequences, so auto-resets land mid-sequence
  at varying offsets. Recomputing the whole stored rollout with the update's sequence sampling,
  stored LSTM states and terminated/truncated resets (scalers frozen) matches the stored log-probs
  to float32 precision (max 1e-3 on ~-7). The LSTM bookkeeping is correct.
- Scaler: skrl's `RunningStandardScaler` updates its statistics inside update epoch 0
  (`train=not epoch`), then renormalises every row. One blow-up row (1900 N) in a batch moves
  every other row's normalised input by > 0.5 std. That changes all log-probs, hence the KL spikes.

**Choice.** Once a world is flagged as a blow-up (D12), the wrapper emits that world's **last
good** actor obs and critic state until the next reset. The world is already frozen with zero
reward, so this only keeps its garbage out of the scaler statistics and the PPO batch. The D11
run predates D12/D14, which explains its spikes.

**Revert.** Drop the `_held` block in `HarvestSkrlWrapper._refresh`.

## D11 run: died of NaN; divergence guards

**Outcome.** The D11 run (d08e65d, pre-D12/D14) went NaN at update t=12544: all losses, KL and std
NaN, then non-finite envs 1498 -> 1997. Before that its KL had reached 17-31. This is the endpoint
of the scaler contamination that D14 removes.

**Pre-NaN checkpoints (eval at 4096c9e).**

| ckpt | success | safety | steps | collateral / success | torque share | gate |
| --- | --- | --- | --- | --- | --- | --- |
| 8000 | 0.896 | 0.024 | 63 | 32.8 N | 0.56 | 2/5 |
| 11200 | 0.851 | 0.039 | 61 | 26.8 N | 0.65 | 2/5 |

**Guards added.**
- The wrapper sanitises actions: non-finite becomes 0 or +-1, clamped to the box, and
  `Step / nonfinite actions` counts them. A NaN policy output can no longer poison the sim.
- `run_training` checks the policy/value weights after every update. If any is non-finite, it
  writes a `{"kind": "diverged"}` metrics row and raises `TrainingDiverged`, naming the last
  healthy checkpoint. Nothing is saved from the poisoned update on.

## D13 run: target behaviour reached; a stored-data mismatch in the real env (open)

**D13 checkpoints (eval, gate vs eval_d8: 3/5 each).**

| ckpt | success | safety | steps | collateral / success | torque share | wrist F |
| --- | --- | --- | --- | --- | --- | --- |
| 4800 | 0.257 | 0.065 | 95 | 10.5 N | 0.90 | 11.9 N |
| 6400 | 0.258 | 0.068 | 81 | 11.0 N | 0.91 | 11.3 N |

- Both collateral clauses and beats_random pass. Success and safety fail; nearly all safety
  failures are wrist force.
- The behaviour is the target one: a bend detach at ~10.5 N collateral per success. The
  remaining gap is reliability.

**Divergence.** From t~6100 the KL climbed 173 -> 1954 -> 8.6e8 with large positive policy loss.
The run was stopped at ~6600, while the deterministic policy was still intact.

**Diagnostic (`ppo.debug_kl`, resumed from ckpt_4800, 16 updates).**
- Scaler shift is at most 0.017 std: ruled out.
- Pre-update KL (no gradient step, scalers frozen) is 0.07-0.34 on 3 of 4 rollouts that contain
  a batch reset, and 73 on one that does not.
- The surrogate does not reproduce it, and neither does the consistency test (auto-reset rows
  recompute exactly). So a few real-env rows differ between rollout and recompute.

**Next.**
- `debug_kl` now dumps the worst rows (step, env, position in the BPTT sequence, ended flags
  around it, obs magnitude and argmax dim, stored LSTM-state norm) to `debug_kl_<t>.json` when
  the pre-update KL > 0.05.
- skrl's `kl_threshold` early stop is on at 0.05 (`PPOConfig.kl_threshold`) as a safety net.

## D15 -- skrl PPO_RNN stored the wrong LSTM state: the cause of the KL spikes and divergences

**Evidence.**
- GPU row dump (`debug_kl`, real env): the stored-vs-recomputed log-prob gap (no gradient step)
  sits on BPTT sequence starts (`pos_in_seq` 0) and decays over the next 1-4 steps of the same
  env. That is the signature of a wrong initial hidden state.
- CPU surrogate, bigger net: the per-position gap is ~20x larger at position 0 and grows with
  every update.
- The stored h[t+1] does not equal one LSTM step from (h[t], obs[t]): error 0.18-0.37.
- One step from (stored h[t-1], obs[t]) reproduces stored h[t] **exactly**.
- `_rnn_initial_states is _rnn_final_states` is True.

**Cause (skrl 2.1).**
- `record_transition` ends with `self._rnn_initial_states = self._rnn_final_states`, making them
  one dict.
- The next `act()` writes `self._rnn_final_states["policy"] = outputs["rnn"]`. Through the
  alias, that also replaces the initial state that `record_transition` stores right after.
- The rollout acted on h_t but memory held h_{t+1}. The update restarted every sequence one step
  ahead.
- The error is invisible at first: the fresh policy barely uses h, because the mean head is
  initialised x0.01. It grows as the policy learns to use its memory. That gave the KL spikes, the
  LR pinned at its floor, and the D11/D13 divergences.

**Fix.** `HarvestPPO_RNN.act` un-aliases the dict before calling `super().act`.
- `test_rl_logprob_consistency.py::test_stored_rnn_state_is_the_state_the_policy_acted_on` pins it:
  stored h follows the LSTM recurrence to < 1e-5, and log-probs match after several updates.
- The first consistency test is back to 1e-4 tolerance. Its earlier 1e-3 mismatch was at a
  sequence start: this bug.
- Per-position gap after the fix: exactly 0.

**Consequence.** Every recurrent run so far trained on corrupted sequence starts. Results from D9
onward should be re-established on the fixed code; the D13 reward settings are the starting point.

**D15 GPU confirmation.** Debug side run at 4c7acee: the pre-update KL (scalers frozen) is ~1e-10
on all 16 updates, with 0 dumps. The stored-data mismatch is gone.

## D16 -- tanh-squash the actor mean

**Finding.** With exact stored data (D15), the post-update KL still spiked (8.1, 7080) at LR <=
3e-4. That size cannot come from a real policy change of that step size.

**Mechanism.**
- The actor's mean was unbounded, while actions are clipped to [-1, 1] and the log-prob is taken
  at the clipped action (skrl `clip_actions`).
- When the mean drifts outside the box, the clipped action sits deep in the Gaussian tail, where
  log p is hypersensitive to the mean: a few nats per dimension for tiny steps.
- The KL estimator mean(exp(r) - 1 - r) explodes on those rows. It also gives huge,
  meaningless policy ratios there.

**Choice.** `mean = tanh(mean_head(features))`.
- The mean head's x0.01 init keeps it in tanh's linear range at the start. The Gaussian std and
  the clipping are unchanged.
- Tests: the mean stays inside the box even for a pre-activation of 50, and the log-prob of a
  boundary action moves < 0.05 for a small parameter step.
- New per-update metric `Policy / actions at box edge (frac)`: the share of stored actions at
  |a| >= 0.999.

**Consequence.** Checkpoints saved before D16 must be evaluated at a pre-D16 commit, because the
same weights now give different means. The running D15 run is pre-D16.

**Revert.** Drop the `torch.tanh`.

## D15 run (4c7acee): the memory fix lifts success past the D13 plateau

| episode | success | safety | steps | collateral / success | torque share | wrist F | KL | LR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.354 | 0.281 | 123 | 9.9 N | 0.91 | 25.6 N | 0.010 | 3e-4 |
| 10 | 0.346 | 0.162 | 129 | 11.9 N | 0.90 | 18.8 N | 0.015 | 3e-4 |
| 15 | 0.342 | 0.141 | 122 | 10.1 N | 0.92 | 17.4 N | 0.016 | 1.3e-4 |
| 20 | 0.430 | 0.107 | 103 | 12.0 N | 0.89 | 16.4 N | 0.085 | 3e-5 |
| 25 | 0.554 | 0.102 | 84 | 12.2 N | 0.89 | 18.0 N | 0.079 | 3e-5 |

- D13 at EP20/25 had success 0.24/0.23. The D15 run keeps climbing while collateral stays ~12 N
  with a bend detach. Safety is down to ~10%, mostly wrist force.
- No NaN, no divergence, no KL > 0.5. KL creeps to ~0.08 with the LR at its floor, which D16
  (tanh mean) targets.

**D15 run final.** 17,500 steps in 2 h (273 updates), no NaN/divergence. Isolated KL spikes: 99.9,
27.2, 16.8. Eval at 4c7acee, gate vs eval_d8:

| ckpt | success | safety | steps | collateral / success | detach F | torque share | wrist F | peak TCP v | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14400 (best) | 0.709 | 0.107 | 62 | 14.8 N | 11.8 N | 0.88 | 19.6 N | 0.186 m/s | 3/5 |
| 16000 (last) | 0.697 | 0.113 | 58 | 15.2 N | 12.0 N | 0.87 | 19.7 N | 0.187 m/s | 3/5 |

- Success is ~2.7x the D13 plateau (0.26), with a bend detach at ~15 N collateral per success
  (scripted pull: 44 N).
- The gate fails only success (>= 0.998) and safety (<= 0.002). About 90% of safety trips are
  the 40 N wrist-force cap.

**Open (maintainer).** The peak TCP speed of 0.19 m/s exceeds the 0.12 m/s D8 target-speed cap,
because the cap is on the VIC target and the TCP can overshoot, likely at the detach snap. It is
within the rig's per-run peak maximum (0.21) but well above its median peak (0.055).

## D16b -- Bound the mean at +-1.5, not +-1

**Finding (D16 run, plain tanh).** Numerics were clean: no NaN, KL <= 0.015, LR at base, actions
at the box edge ~4.4%. But the policy drifted passive:

| episode | D16 success | D16 safety | D15 success (same reward) |
| --- | --- | --- | --- |
| 5 | 0.29 | 21.7% | 0.35 |
| 10 | 0.24 | 8.0% | 0.35 |
| 15 | 0.15 | 4.4% | 0.34 |

D15 went on to 0.71.

**Read.**
- The only difference from D15 is the tanh. Under D8 the useful pulls are full-rate: a 2 mm/step
  target move is action 1.0.
- With a bound of exactly 1, commanding that needs a saturated tanh, where the gradient vanishes.
  The policy retreats to smaller, safer actions.

**Choice.** `mean = 1.5 * tanh(x / 1.5)` (`RecurrentNetConfig.mean_bound = 1.5`).
- Edge actions stay easy to sample: a mean of 1 has a gradient of 0.56.
- The clipped action stays within ~1.1 std (std ~0.45) of the mean, never deep in the tail
  that caused the KL spikes.
- The D16 run was stopped; relaunched as D16b.

**Revert.** `mean_bound` large (e.g. 1e3) ~ unbounded.

**D16b outcome (stopped at EP30 per the rule).**

| episode | success | safety | collateral / success | torque share | KL |
| --- | --- | --- | --- | --- | --- |
| 20 | 0.245 | 5.6% | 14.2 N | 0.86 | 0.018 |
| 25 | 0.286 | 6.6% | 10.5 N | 0.91 | 0.016 |
| 30 | 0.287 | 6.1% | 11.5 N | 0.92 | 0.016 |

- D15 was at 0.607 by EP30.
- D16b has the cleanest numerics of any run (no KL > 5) and the lowest safety (~6%), but success
  plateaued at 0.25-0.29.
- Bounding the mean (at 1 or 1.5) trades the tail KL spikes for a policy that won't commit to
  full-rate pulls.
- Next: the fallback `sim_train_gpu_ep250_unbounded.json` (D15 actor + kl_threshold 0.05 + guards),
  a fresh 2 h run. If it reproduces D15 (~0.7), the unbounded mean becomes the default again.

**D16b checkpoint eval (ckpt_6400 at 096a5ff; gate 3/5).**

| policy | success | safety | collateral / success | peak TCP speed |
| --- | --- | --- | --- | --- |
| D16b ckpt_6400 | 0.279 | 3.0% | 10.4 N | 0.052 m/s |
| D15 ckpt_14400 | 0.709 | 10.7% | 14.8 N | 0.186 m/s |

- The rig's per-run peak TCP speed has a median of 0.055 m/s. D16b moves like the rig; D15 moves
  ~3x faster.
- The bounded mean is the most rig-like and safest policy; the unbounded one picks 2.5x more
  often.
- This is a real trade-off for the maintainer: success vs rig-realistic motion and safety.

## D8b -- Max target speed 0.4 m/s and 0.3 rad/s (maintainer)

- The maintainer set the cap to 0.4 m/s linear and 0.3 rad/s angular: at 60 Hz,
  `linear_delta_m = 0.4/60 = 6.67 mm/step` and `angular_delta_rad = 0.3/60 = 5 mrad/step`.
- Relative to D8 that is 3.3x faster linear and 2x slower angular.
- `scripted_pull` (2 mm/step) is unchanged. `scripted_twist_pull`'s 10 mrad/step twist is clamped
  to 5 mrad/step. Random is faster.
- The baselines are re-run for the gate (`runs/eval_d8b`).
- With full-scale actions no longer the useful pull rate (0.12 m/s is now ~0.3 of the range), the
  bounded-mean actors may no longer be handicapped.
- Three variants, same reward: `sim_train_gpu_d8b_{unbounded,tanh1,tanh15}.json`.

**D8b plan change (maintainer): three seeds of the bounded actor, not three variants.**

- Run `mean_bound = 1.5` (D16b actor) under the D8b cap with seeds 0/1/2:
  `sim_train_gpu_d8b_tanh15_s{0,1,2}.json`, each with wandb and a video every 5 episodes.
- Seeds measure run-to-run spread, which none of the single-run comparisons so far (D15 vs D16b)
  could separate from the actor change.
- Video: `TrainConfig.video_every` (default 0 = off) records one randomly picked env for a whole
  episode every N episodes (`HarvestVideoRecorder`, CUDA only). Clips go to
  `<run_dir>/videos/` and to wandb as `Video / episode`.

**D8b baselines (GPU, `runs/eval_d8b`, N=2000, 1 episode).**

| policy | success | safety | collateral / success | steps | peak TCP |
| --- | --- | --- | --- | --- | --- |
| zero | 0.000 | 0.0% | -- | -- | 0.001 m/s |
| random | 0.431 | 10.9% | 39.5 N | 150 | 0.155 m/s |
| scripted_pull | 0.998 | 0.2% | 44.5 N | 31 | 0.102 m/s |
| scripted_twist_pull | 0.999 | 0.1% | 42.4 N | 53 | 0.058 m/s |

- At the faster cap, random succeeds again (0.43, vs 0.05 under D8). The gate now needs success
  >= 0.431 and collateral < 39.5 N.
- The pulls succeed almost always, but at ~42-44 N collateral. The learned policy's value is the
  low-collateral bend detach (~10-15 N in D15/D16b), not success rate.
- wandb 0.30 has no `wandb.util.generate_id`. The local agent fixed the run-id minting (014ae74).
  Seeds live: s0 `26yqs6lv`, s1 `l478x3im`, s2 `sc4364a7` (project `pruning-rl/apple_pick_vic_harvest`).

**D8b parallel seeds: throughput and segmenting.**

- All three seeds together run at 3.24 steps/s: 1.3x a single run (2.43 steps/s). Each seed runs
  at 1.08 steps/s. The GPU is saturated (98% util, 12.6 / 24.6 GB). The first clip was written
  for every seed.
- One 2 h run then covers only ~31 of the ~70 planned episodes per seed.
- Decision: keep the three seeds parallel and continue in <= 2 h segments (`--resume latest
  --max-updates 112`; the new flag ends a segment on a checkpoint). This keeps the 2 h rule and
  gives ~55 episodes per seed over two segments. A third segment is decided from the rows.

**D8b seeds, EP1-EP5 (early read, no action).**

| seed | success EP1 -> EP5 | safety EP5 | collateral / success EP5 | torque share EP5 |
| --- | --- | --- | --- | --- |
| s0 | 0.95 -> 0.94 | 4.7% | 39.3 N | 0.39 |
| s1 | 0.77 -> 0.68 | 19.8% (wrist F) | 34.2 N | 0.53 |
| s2 | 0.67 -> 0.92 | 3.5% | 39.9 N | 0.37 |

- At the 0.4 m/s cap, the untrained policy already detaches 67-95% of the time by pulling
  (~42 N, torque share 0.33). Under D8 it started at ~5% and bend-dominated.
- The runs start in the pull basin. The only gain left is un-learning it for a lower-collateral
  bend detach.
- At the terminal, a 42 N pull scores 20 - 0.5x42 = -1, and a ~12 N bend scores +14. That is a
  ~15-point pull toward bend, set against the wrist soft cap that bend pays (s1 shows exactly
  this trade).
- No reward change mid-run. Decision point is after segment 2. If no seed falls below ~25 N
  collateral, consider a stronger peak-collateral weight (D13 0.5 -> 1.0).

**D8b seeds, EP10-15: a reliable pull at ~37-38 N; D13b prepared, not launched.**

| seed | success EP15 | safety EP15 | collateral / success | torque share | peak TCP |
| --- | --- | --- | --- | --- | --- |
| s0 | 0.982 | 1.8% | 38.3 N | 0.47 | 0.36 m/s |
| s1 | 0.685 | 5.0% | 37.9 N | 0.54 | 0.17 m/s |
| s2 | 0.967 | 2.4% | 36.7 N | 0.45 | 0.32 m/s |

- The seed spread is small in collateral: all three sit at 37-38 N.
- s0 and s2 beat scripted_pull on safety and slightly on collateral (44.4 N), but are far from the
  D15/D16b bend detach (~10-15 N). Numerics are clean.
- Speed climbs toward the 0.4 m/s cap. At this cap a fast pull succeeds almost always, so the
  pull basin wins. Under D8, pulls failed and bend was the only way.
- Segment 2 runs as planned, same reward. The maintainer expects longer training to find the harder
  behaviour, and D15 only broke out after EP15-25.
- **D13b (ready if no seed goes below ~25 N after segment 2):** `w_peak_collateral` 0.5 -> 1.0,
  `sim_train_gpu_d8b_tanh15_pc1_s{0,1,2}.json`.
  - Terminal success value: a 38 N pull +20-38 = -18 (still +22 over failure, -40); a 12 N bend
    +8.
  - The bend-over-pull margin rises from ~13 to ~26 points. A success is still always better
    than failing below the 60 N collateral mark.

**Segment 1 end (EP25) and wandb verified.**

- wandb now holds continuous scalar history from timestep 64 for all three seeds (backfill + live).
- EP25:
  - s0: success 0.955, collateral 34.8 N, TCP 0.40 m/s, LR at the 3e-5 floor.
  - s1: 0.728, 36.6 N.
  - s2: 0.974, 35.5 N.
- Collateral creeps down (~42 -> 35 N over 25 episodes, torque share ~0.5). This is a slow drift
  toward bend, not a switch.
- Watch in segment 2: s0's LR at the floor slows further change.
- The D13b trigger stays: no seed below ~25 N by the segment 2 end.

**D13b launched: peak-collateral weight 1.0, 3 fresh seeds (after segment 2 + gate).**

| seed | EP30 | EP35 |
| --- | --- | --- |
| s0 | succ 0.944, collateral 33.1 N | succ 0.973, collateral 35.5 N, safety 2.7% |
| s1 | succ 0.685, collateral 35.4 N | succ 0.739, collateral 36.5 N |
| s2 | succ 0.984, collateral 36.3 N | succ 0.985, collateral 36.6 N, safety 1.5% |

- Collateral per success has plateaued at 35-37 N, with torque share flat at ~0.5. Success,
  safety and speed keep improving: the policies optimise reliability and speed, not collateral.
- The D13b trigger is met.
- Launch fresh rather than fine-tuning from the D8b checkpoints: those have a converged pull and
  s0's LR is at the floor.
- Fresh runs are directly comparable to the D8b seeds; the only change is the weight.
- Weight 1.0, not 2.0: at 2.0 a 38 N pull scores 20-76 = -56, worse than failing (-40). That
  invites the passive drift seen in D16.

**D8b seeds final (212 updates, ckpt_13568; best == last; gate 2/5 each vs `runs/eval_d8b`).**

| seed | success | safety | collateral / success mean, median | p10 / p90 | < 15 N | < 22 N | torque share | peak TCP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s0 | 0.972 | 2.8% | 30.7, 32.2 N | 11.5 / 45.5 N | 16% | 26% | 0.56 | 0.36 m/s |
| s1 | 0.760 | 4.7% | 31.0, 34.1 N | 10.0 / 45.5 N | 21% | 27% | 0.64 | 0.25 m/s |
| s2 | 0.996 | 0.4% | 34.8, 35.7 N | 20.9 / 46.8 N | 4% | 12% | 0.50 | 0.38 m/s |

- All three pass beats_random and collateral_vs_random. They fail success / safety / collateral,
  because scripted_pull is near-perfect under D8b (0.998 / 0.2%) and the collateral bar is 22.2 N.
- Success-conditioned collateral is **bimodal**: a bend mode (~10 N) and a pull mode (~35-46 N).
  The mean hides it. s0 and s1 already put 16-21% of successes below 15 N.
- s2 is the most reliable and safe learned policy so far, but it is almost all pull.
- D13b (weight 1.0, fresh pc1 seeds) now aims to move mass into a bend mode the policy can already
  reach. The signal is the < 15 N / < 22 N fractions and the median, not the mean.

**Correction: best != last.** The selection rule picked earlier checkpoints: s0 ckpt_12800, s1
ckpt_11200, s2 ckpt_8000. All are also 2/5.

- The last checkpoints (13568) are better on reliability (s2 0.996 / 0.4% vs 0.987 / 0.7%) and on
  the low-force share (< 15 N: s0 0.159 vs 0.107, s1 0.211 vs 0.164).
- The bend mode grew during segment 2, so longer training helps.
- The D8b seeds use the last checkpoints as their reference.
- A D8b segment 3 (to 17.5k steps) stays an option after D13b, if D13b does not beat the growth
  rate.

**D13b EP5 (early).** Weight 1.0 works on collateral but costs success early.

| seed | success | safety | collateral / success median | < 15 N | torque share | TCP |
| --- | --- | --- | --- | --- | --- | --- |
| s0 | 0.28 | 14.8% | 29.5 N | 21% | 0.61 | 0.13 m/s |
| s1 | 0.14 | 17.6% | 9.9 N | 81% | 0.90 | 0.13 m/s |
| s2 | 0.09 | 10.5% | 12.6 N | 56% | 0.83 | 0.13 m/s |

- Successes move into the bend basin, but success collapses. Safety rises, so these are failing
  bends, not passive drift. This is the same signature as D15's early phase (which went
  0.35 -> 0.71).
- Keep running. The EP15 check stands: stop only on success < 0.5 with LOW safety, or no success
  recovery by EP25.
