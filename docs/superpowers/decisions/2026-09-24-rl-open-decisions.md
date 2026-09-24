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

**Revert.** Set `linear_delta_m=0.02` and `angular_delta_rad=0.1` in the config.
