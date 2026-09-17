# RL infrastructure for a learned VIC apple-picking policy (design)

## Document status

| Field | Value |
| ----- | ----- |
| **Status** | Draft — decisions agreed with maintainer in the 2026-09-17 design session |
| **Date** | 2026-09-17 |
| **Owner** | Abhinav |
| **Branch** | `feature/rl-vic` (worktree `../apple_pick_sim-rl-vic`) off `main` |
| **Related** | `docs/VISION.md` ([M5] pick policy), `docs/ROADMAP.md`, H1 `docs/handbook-coupled-simulation.md`, H2 `docs/handbook-variable-impedance.md`, H3 `docs/handbook-sysid-scoring.md`, H5 `docs/handbook-youngs-cma.md`, `docs/gym-observation-contract.md` |
| **Supersedes** | `docs/superpowers/specs/2026-07-28-rl-gym-harvest-env-design.md` (partially — see Relationship to `feature/rl-gym`) |

## Why this exists

System identification is treated as working. The next milestone is [M5]: learn a
variable-impedance policy that picks apples in the calibrated simulator. This
spec defines the **infrastructure** — action space, observation pipeline,
domain randomization, reward, episode structure, and trainer integration — not
the final reward shaping or a trained policy.

## Relationship to `feature/rl-gym`

The `feature/rl-gym` branch (worktree `../apple_pick_sim-rl-gym`, 15 commits,
branched at `21456b2`) is **128 commits behind `main`**. Its Tasks 1–4 added an
anisotropic per-env VIC wrench path to `apple_pick_sim/coupled_fruiting/`.
`main` has since grown an equivalent path independently for `vic_pose`:
`compute_vic_spatial_wrench_aniso` (`vic_wrench.py:101`),
`_compute_vic_wrenches_batched_aniso_kernel`, and the `vic_kp_lin_wp` /
`vic_kd_ang_wp` scene buffers. **Those tasks are superseded.**

What is worth porting from that branch, at the gym layer only:

| From `feature/rl-gym` | Disposition |
| --- | --- |
| `harvest_reward.py` (progress / pull-out / collateral, pure tensor, tested) | **Port** — reward math is independent of the controller path |
| `harvest_gains.py` `derive_critical_damping` | **Port** — `D = 2ζ√K` is still the damping law |
| `ApplePickHarvestEnv` obs/`info` layout | **Port as reference**, rewrite against the new action space |
| Tasks 1–4 anisotropic VIC kernels | **Drop** — `main` already has this |
| `HarvestActionBounds` 12-D twist layout | **Drop** — replaced by the 13-D delta-pose action |

## Measured baseline (RTX 4090, 2026-09-17)

`ApplePickBatchedVicEnv`, `gym_defaults()`, 60 Hz control (30 substeps/step),
zero-action warmup then 30 measured steps:

| N | build (s) | step (ms) | env-steps/s | GPU used (GB) |
| --- | --- | --- | --- | --- |
| 4 | 52.7 | 292.7 | 13.7 | 2.50 |
| 16 | 52.8 | 296.7 | 53.9 | 2.50 |
| 64 | 52.7 | 297.7 | 215.0 | 2.53 |
| 256 | 57.6 | 374.7 | 683.1 | 3.72 |
| 512 | 66.7 | 593.2 | 863.2 | 7.35 |
| 1024 | — | OOM at build | — | — |

**These numbers are all distorted by a constraint-arena bug — see below.** With
the arena corrected, N=512 measures **331.3 ms / 1545.6 env-steps/s / 2.27 GB**.

### The constraint-arena bug (fix before trusting any of the above)

`batched_build.py:375` passes `njmax = nconmax = max(200, 80 * num_envs)`. Both
are **per-world** parameters in mujoco_warp, not global budgets:

- `io.py:930` — *"njmax: Number of constraints to allocate **per world**.
  Constraint arrays are batched by world."*
- `io.py:996` — `efc.J_rownnz = wp.zeros((nworld, njmax), dtype=int)`
- `io.py:955` — `naconmax = _resolve_batch_size(naconmax, nconmax, nworld, 0)`,
  so `nconmax` is multiplied by `nworld` internally too

Multiplying a per-world size by the world count makes allocation **O(N²)** —
`80·N²` entries across 9+ `(nworld, njmax)` arrays plus `efc.J` at
`(nworld, njmax_pad, nv_pad)`. The N=1024 OOM was an allocation of exactly
335,544,320 bytes = `1024 × 81920 × 4`, one `(nworld, njmax)` int array.

It is not only a memory bug: the constraint solver iterates the oversized arena,
so it costs throughput too. Measured at N=512 with the arena clamped to a
per-world 200:

| N=512 | `njmax = 80·N` | `njmax = 200` | change |
| --- | --- | --- | --- |
| step | 593.2 ms | **331.3 ms** | −44% |
| env-steps/s | 863.2 | **1545.6** | **+79%** |
| GPU used | 7.35 GB | **2.27 GB** | −69% |

For a 7-DOF gravity-compensated arm whose plant lives in a *separate* VBD model,
200 constraints per world is already generous — mujoco_warp's own
`_default_njmax` (`io.py:739`) evaluates to ~53 for a model with no height
fields, flex or SDF.

**Safety condition:** the fix is behaviour-preserving *only if* no world ever
needs more than the clamp. If a world exceeds `njmax`, constraints are dropped
and the physics changes silently. Validate actual per-world `nefc` against the
clamp before adopting it.

### Operating point

**Recommended: N=512 with the arena fixed — 1545.6 env-steps/s** (10M env-steps
in ~1.8 h, versus ~3.2 h on the buggy path). Reset
(`restore_episode_snapshot`) is ~2–7 ms and effectively independent of N. Build
is ~63 s.

**N=1024 remains unverified.** The arena fix removes the quadratic OOM there
(GPU sat at ~1.7 GB where the old path demanded >22 GB), but the build then dies
*silently* mid-settle with no Python exception — consistent with the SIGSEGV /
heap-corruption failure mode this repo already documents in
`ApplePickBatchedBaseEnv.close()`, which is why CMA uses process-isolated
evaluation waves. Treat 512 as the supported ceiling until someone reproduces
1024 under `faulthandler` or `cuda-gdb`.

Consequences that shape this design:

1. **Small batches are almost pure waste** — N=4 and N=64 cost nearly the same
   wall-clock per step. Run at least a few hundred envs.
2. **The apparent saturation at N=256–512 was the constraint-arena bug, not a
   real limit.** Do not plan around it.
3. **N=1024 OOMs at build.** `batched_build.py:375` sizes the MuJoCo constraint
   arena as `njmax = nconmax = max(200, 80 * num_envs)` — 81,920 slots at
   N=1024. For a 7-DOF gravity-compensated arm whose plant lives in a separate
   VBD model, 80 constraints per env is far more than joint limits and contacts
   require. **Retuning that coefficient is the first lever**, before accepting
   any env-count ceiling as fundamental.
4. **CUDA graph capture is the next untapped speedup after the arena fix.** `scene.py:592-594`
   caches values explicitly "to avoid host sync during CUDA graph capture", but
   no capture exists in the step path. Since the loop is launch-overhead bound,
   capture is the highest-leverage optimization. Out of scope here; recorded as
   a follow-up.

## Action space — 13-D delta-pose + stiffness

The policy emits a **bounded delta** around a per-env target pose it maintains,
plus per-axis stiffness and a damping ratio:

```text
action[0:3]   dp        bounded pose delta      [m per step]
action[3:6]   drot      bounded rotation delta  [rad per step]
action[6:12]  Kp        [lin(3), ang(3)]
action[12]    zeta      -> Kd = 2*zeta*sqrt(Kp)
```

The env keeps `_target_pose: (N, 7)` state, integrates `target <- target (+) delta`
each step, derives `Kd`, and **packs the result into the existing 19-D
`vic_pose` action** (`[pos(3), quat_wxyz(4), Kp(6), Kd(6)]`, H2 §3). Nothing in
`apple_pick_sim/coupled_fruiting/` changes: the controller already accepts this.

Rationale: absolute world poses are badly conditioned for exploration, while the
underlying interface stays exactly the real rig's pose-PD contract, so the
learned policy is deployable without a conversion layer. `_target_pose` is
initialized to the current TCP pose on reset.

**Damping is derived, not commanded**, so the policy cannot pick a K/D pair that
destabilizes the impedance controller. `zeta` is a single scalar action dim
rather than 6, keeping the K/D relationship coupled per axis.

## Observation pipeline

**Actor observation — sensor-realistic only.** Flattened to a fixed-order
`(N, D)` tensor (skrl memories are flat; the nested
`woody_part_start_pos: dict[str, (N,3)]` layout from the v3 contract must be
flattened deterministically by junction name):

| Source | Notes |
| --- | --- |
| `tcp_pos`, `tcp_quat`, `tcp_velocity` | Existing v3 keys |
| `ft_wrist` | **EMA-filtered + bias/noise/drift** — see Sensor realism |
| `apple_pos`, `apple_quat` | Existing v3 keys |
| `robot_joint_q` | Existing v3 key |
| `woody_part_start_pos` / `woody_part_end_pos` | Vision-tracked junction geometry |
| `last_action` | 13-D, keeps commanded impedance Markov |
| `step_frac` | `step_count / max_episode_steps` |

**Critic observation — privileged.** Actor observation plus per-env ground truth:
spur/stem flexural and axial `E`, damping ratios, support `k_p`, primary rod
density, the sampled arm-DR values, and **all** junction wrenches from
`info["woody_part_force"]`.

Both actor and critic are **recurrent (LSTM)**. The plant's compliance is not
identifiable from a single frame, so the actor needs memory to infer it from
interaction history.

### Sensor realism on `ft_wrist`

Sim `ft_wrist` is raw; the real rig's F/T is filtered. The observation path
applies, per env:

```text
ema[n] = a * raw[n] + (1 - a) * ema[n-1]     a = 1 - exp(-2*pi*fc/f_control)
obs    = ema + bias + noise + drift
```

At `f_control = 60` Hz and `fc = 10` Hz, `a ≈ 0.65`. `bias` is resampled per
env per episode; `noise` is per-step; `drift` is a slow per-episode walk.

**Two invariants that must not be confused:**

- The real dataset's `ft_wrist_lpf` is **zero-phase** (`filtfilt`, offline, for
  CMA scoring). No online policy can reproduce it. The EMA models what a
  *real-time* controller sees and is deliberately not trying to match it.
- The **"No sim EMA/LPF" rule still holds for the sys-ID scoring path** (H3).
  This EMA lives strictly in the gym observation path and must never reach the
  `batched_sysid_v1` feature bags.

## Domain randomization

### Plant — uniform from the ranges fixture

DR samples **uniformly from the current ranges fixture**,
`apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json`
— already what `default_ranges_fixture_path()` (`params.py:793`) returns, so
this is the existing gym default.

**No new sampler is needed.** `sample_heterogeneous_params_list`
(`params.py:1122`) already fixes segment topology from `topology_seed` via
`_fix_topology` and draws every other parameter per environment from the fixture
ranges. Plant DR is therefore *already implemented*; the work is choosing the
fixture and confirming the path, not building a distribution.

**Superseded:** an earlier draft of this spec centred plant DR on the calibrated
CMA phenotype. That is no longer the direction. The trade-off, recorded once and
not relitigated: uniform fixture sampling covers regions that system
identification may have ruled out, buying broader robustness at the cost of
spending some capacity on less plausible plants.

**Coverage requirement: all ten CMA search knobs.** DR must vary every parameter
the CMA search identifies — spur/stem flexural modulus, spur/stem axial modulus,
spur/stem damping ratio, support `k_p`, support roll `k_p`, support joint ζ, and
primary density. A parameter worth identifying is one the policy should be
robust to. **Only 5 of 10 vary today:** the two rod `damping_ratio` ranges are
pinned (`min == max == 0.3`), and the three support-joint knobs live in the
fixture's `sim_build` block, which holds scalars rather than min/max ranges.
Closing this needs a schema extension, not just wider numbers — see Task 3.

The per-env *application* path already exists and must be reused, not rebuilt:
`apply_per_env_support_joint_penalties` (with `zeta_per_env`) and
`apply_per_env_support_roll_penalties` in
`apple_pick_gym/batched_envs/support_joint_penalties.py` were written for CMA and
already accept per-env values.

**Constraint (unchanged):** apple radius and density are *derived* to close the
measured chord, so they are not free parameters. Geometry DR must respect that
closure; validate density against ~800 kg/m³ rather than assuming
self-consistency.

Plant material parameters are **baked at build** (CMA rebuilds a fused world
every generation for exactly this reason), so plant DR is per-env, fixed across
resets.

### Arm — all four axes, within ranges

| Axis | Mechanism | Feasibility |
| --- | --- | --- |
| Joint dynamics (armature, Coulomb friction, viscous damping) | `_set_fr3_joint_armature` / `_set_fr3_joint_friction` / `_set_vic_passive_joint_damping` + `notify_model_changed(JOINT_DOF_PROPERTIES)` | **Confirmed per-env** |
| Link mass / inertia | `_set_body_inertial_full` + `BODY_INERTIAL_PROPERTIES` | Confirmed per-env |
| EE payload + TCP geometry | `/fr3/ee` mass, COM, `I_ee` | Per-env; **randomizes a sys-ID-calibrated quantity** — keep the band tight |
| Controller + sensing | `kp_null` / `kd_null`, 200 N·m/s torque slew, F/T bias/noise/drift, latency | Gym/controller layer, not model arrays |

**Why arm DR is cheap here.** The setters in
`apple_pick_sim/robot/fr3_robot/setup.py` already walk the replicated model
per world:

```python
for start in range(0, n, stride):
    arr[start : start + n_arm] = values   # same `values` broadcast to every world
```

Per-env DR generalizes `values` from one vector to a per-world one. The stride
is already computed. The sync is verified per-world end to end:
`notify_model_changed(JOINT_DOF_PROPERTIES)` → `_update_joint_dof_properties`
(`solver_mujoco.py:6679`) launches `update_dof_properties_kernel` at
`dim=(nworld, nv)`, and that kernel (`kernels.py:2011`) writes
`dof_armature[world, mjc_dof] = joint_armature[newton_dof]` into **`wp.array2d`
per-world** mjw fields.

**Asymmetry worth exploiting:** arm DR is reassignable at any time plus a
notify, so it *can* be re-randomized every episode. Plant DR cannot. This gives
per-episode variation even though grasps and plant params are frozen per env.

### Grasp — per-env, fixed at build

Grasp diversity comes from **N**, not from resampling: each env draws its own
grasp at build and keeps it. Build cost is ~constant in N (52–58 s), so this is
nearly free.

| Axis | Mechanism | In scope |
| --- | --- | --- |
| Approach / pull direction | `GripperProxyConfig.weld_direction`, per env via `per_env_grippers` | **Yes** |
| Gripper roll about apple | `weld_proxy_offset_in_apple_frame` (full SE(3) in apple frame) | **Yes** |
| Apple / spur geometry | Already per-env: `sample_heterogeneous_params_list` fixes only segment topology via `_fix_topology` | **Yes** |
| Plant base pose vs robot | `fruiting_base_pos` / `robot_base_pos` are **batch-scalar** (`batched_heterogeneous_build.py:478`) | **Deferred** — needs new build plumbing, defeats `reuse_replicated_mujoco`, forces per-env IK, and dilutes coverage |

Every axis added is spent against N permanently, since each env is one draw from
the joint distribution for the whole run. That is the main argument for
deferring the base-pose axis.

## Reward and termination

```text
r_t = w1*r_progress - w2*r_pullout - w3*r_collateral - w4*r_safety + b_success
```

Ported from `harvest_reward.py`. `F_thresh = 5 N` **for now** — the maintainer
has flagged the reward for a later revisit, likely as a combined force+torque
criterion.

**Success:** `‖F_spur_stem[:3]‖ ≥ F_thresh` sustained for `K` consecutive steps
with no safety-cap violation.

**Note on the privileged-reward gap.** The reward reads the *uncapped* woody
`anchor_force` (`batched_obs.py:399`), a different path from the 40 N / 10 N·m
capped stem harvest that feeds `ft_wrist`. **At `F_thresh = 5 N` this is
harmless** — 5 N is far below the cap, so `ft_wrist` is fully informative across
the success-relevant range. The gap only bites if the threshold is later raised
toward 40 N; revisit then.

## Episode structure and the reset constraint

`BatchedHeterogeneousCoupledSim` supports **whole-batch reset only**
(`restore_episode_snapshot`); there is no `reset_idx`. Rather than fight this:

**Fixed-length episodes with frozen-on-success envs.** All envs truncate
together at `max_episode_steps`. An env that satisfies the success condition is
frozen (action held, reward accrual masked) but the episode does not end for the
trainer.

This matters specifically because the policy is recurrent: synchronized episode
boundaries make **LSTM hidden-state resets trivial and batch-uniform**. Ragged
per-env termination against a sim that cannot reset individual envs would
desynchronize hidden-state handling and PPO bootstrapping. The freeze mask stays
gym-layer-local, so a future real `reset_idx` is a single-seam change.

## Trainer

**skrl**, matching the base env's documented "SKRL-native" conventions and the
existing M2.2c roadmap item. Requires a new `rl` extra in `pyproject.toml`.

Integration work:

- A flat `(N, D)` observation encoder plus a separate `(N, D_priv)` critic
  encoder (skrl memories are flat; the v3 obs dict is nested).
- Recurrent PPO with LSTM actor and privileged LSTM critic.
- Checkpoint/resume must persist **both** networks' hidden-state specs.
- Logging: reward and success-rate learning curves (the phase's exit criterion).

## Exit criterion

Learning curves on **episode reward** and **success rate**. Sim-internal, no
real-robot dependency — this phase is self-contained.

## Decision log

| Decision | Choice | Rationale |
| --- | --- | --- |
| Action space | 13-D delta-pose + Kp + ζ, packed to 19-D `vic_pose` | Well-conditioned exploration over the real rig's actual interface |
| Success signal | Force proxy, `F_thresh = 5 N` | No new physics; explicitly provisional |
| Episode scope | Post-grasp, per-env grasp fixed at build | Build cost constant in N; diversity from N |
| Branch | New branch off `main` | `feature/rl-gym` Tasks 1–4 are superseded by `main`'s aniso path |
| Trainer | skrl | Base env already shaped for it |
| Actor / critic | LSTM actor, privileged LSTM critic | Compliance is not single-frame identifiable |
| Reward privilege | Uncapped junction force, gap documented | Train-time privilege is legitimate; moot at 5 N |
| Plant DR | Uniform from the fixture, covering **all ten CMA knobs** | Rod params already sampled; damping ratios are degenerate and support-joint knobs need a `sim_build` schema extension |
| Arm DR | All four axes, bounded ranges | Plant-only DR does not transfer |
| F/T realism | EMA + bias/noise/drift | An EMA *is* a causal first-order low-pass; no filter design needed |
| Control rate | 60 Hz | Maintainer decision; deployment to a 30 Hz rig needs decimation |
| Grasp base-pose DR | Deferred | Batch-scalar today; dilutes coverage |

## Follow-ups (not built here)

- CUDA graph capture of the coupled substep loop — the largest throughput lever.
- Investigate the **silent N=1024 build crash** that remains after the arena fix.
- **Per-env IK bootstrap placement is not run-to-run reproducible.** Found
  while validating Task 0a: two fully independent builds of the identical
  config (separate OS processes) land most envs (58-60 of 64) in
  qualitatively different arm configurations, diverging by up to ~2 in TCP
  pose from the very first step. Consistent with `IKBootstrapConvergenceWarning`s
  observed in every build (some exceeding the 0.05 m tolerance by 2-3x) —
  the IK solve likely has multiple local minima (e.g. elbow-up vs
  elbow-down) and GPU floating-point reduction order tips which one it
  lands in. Matters for anything assuming deterministic per-env grasp
  placement from a fixed seed, including this plan's own "per-env grasp
  fixed at build, diversity from N" design (Grasp DR section above) and
  reproducing/debugging any specific training run.
- Real per-env `reset_idx` on `BatchedHeterogeneousCoupledSim`.
- Reward revisit: combined force + torque success criterion.
- Per-env plant base pose vs robot.
- Reach-and-grasp phase before harvest.
- Real stem-detach physics to replace the force proxy.
- 30 Hz policy rate or action-repeat for deployment parity.
