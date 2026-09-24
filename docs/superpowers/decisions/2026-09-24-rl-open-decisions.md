# [M5] open decisions taken by the cloud session (2026-09-24), for maintainer review

The maintainer handed over on 2026-09-24 with "make the open decisions and mark the commits; we'll
discuss tomorrow". Each decision below has an ID. Every commit implementing one is tagged `[Dn]` in
its subject line (`git log --oneline --grep '\[D'`). Each entry lists the choice, the evidence,
the alternatives, and how to revert it. Nothing here changes the reward weights or PPO
hyperparameters beyond what is listed.

| ID | Decision | Status |
| --- | --- | --- |
| D1 | Detach signal: the envelope reads the stem-root *elastic* wrench, not the rigid-junction readout | done on CPU; GPU sweep requested |
| D2 | Task 11 gate: success AND safety AND collateral vs scripted pull, AND >= random | done |
| D3 | F/T sensor model matched to the real rig: noise and online EMA corner (8.2 Hz) | done |
| D4 | F/T observation frame for deployment (sim is world frame; rig is mixed) | flagged, no code change |

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
- (c) The GPU sweep (`detach_sweep`) should confirm that zero and random no longer detach while the
  scripted pull still does.

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
