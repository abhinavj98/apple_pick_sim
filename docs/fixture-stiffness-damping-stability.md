# Fixture stiffness/damping domain randomization: stability and sys-ID targets

## Status

**Design analysis — not yet implemented.** This document records the reasoning behind a
proposed change to how `apple_pick_sim/fixtures/*.json` range JSON is sampled
(`apple_pick_sim.fruiting_system.params.sample_params`). It exists to fix a known
instability source in domain randomization (DR) and to define the correct sys-ID
targets for the upcoming M3 CEM calibration work. Feeds **[V].2.1.2** in
`docs/ROADMAP.md` ("Fixture stability + real-world likeness").

No code has been changed yet. Implementing this is a **TDD** task per
`.cursor/rules/test-driven-development.mdc`: write failing tests against the derived
sampling formulas before touching `sample_params`.

## 1. Problem statement

`apple_pick_sim/fixtures/*.json` specifies independent `[min, max]` ranges for each rod
segment's `bend_stiffness`, `bend_damping`, `radius`, `length`, `density`, and
`num_segments`. `sample_params` (`apple_pick_sim/fruiting_system/params.py:318-463`)
draws each of these independently via `rng.uniform`/`rng.integers`. That independent,
per-axis sampling is what produces unstable draws (apple drift/sag, non-convergent
settle, blown-up velocities — see `apple_pick_sim/diagnostics/sweep_zero_vic_stability.py`
and `apple_pick_sim/coupled_fruiting/settle_quasi_static.py`).

The root cause: these quantities are **not physically independent**. Whether a given
`bend_damping` is "enough" depends on `bend_stiffness` and on the segment's mass/inertia
(set by `radius`, `length`, `density`, `num_segments`). Sampling them from independent
boxes covers a hyperrectangle in parameter space, but the physically stable region is a
curved manifold that cuts diagonally through that box — many corners of the box are
guaranteed-unstable regardless of how "reasonable" each individual range looks in
isolation.

## 2. Rod-level params vs. joint-level params

`RodParams` (`apple_pick_sim/fruiting_system/params.py:29-39`) is **one scalar tuple per
rod segment** (`primary`/`secondary`/`spur`/`stem`). `build.py` passes that tuple
straight into `newton.ModelBuilder.add_rod`
(`apple_pick_sim/fruiting_system/build.py:205-217`, `:337-349`, `:378-390`), which
creates `num_segments` capsule bodies and `num_segments - 1` internal cable joints, and
broadcasts the *same* `bend_stiffness`/`bend_damping`/`radius`/`density` to **every**
joint in that segment (`newton/newton/_src/sim/builder.py:7234-7439`, per-joint units:
torque/radian for bend stiffness, N·m·s/rad for bend damping). There is no per-joint
variation within one sampled segment — homogeneity within a segment is structural, not
sampled.

This means `num_segments` is itself a hidden multiplier on stability, not just a
topology knob: more segments in series over the same physical `length` requires each
joint to be *stiffer* to represent the same overall compliance (springs in series), and
—critically—it also changes each joint's *effective inertia* far more steeply (see §4).

## 3. Deriving `bend_damping` from a target damping ratio

Model each joint as a torsional spring-damper acting on its segment's rotational
inertia. For one rod segment (`radius` \(r\), `density` \(\rho\), `length`, split into
`num_segments` \(N\)):

- Segment length: \(L_{seg} = \text{length}/N\)
- Segment mass: \(m_{seg} = \rho \cdot \pi r^2 \cdot L_{seg}\)
- Effective bending inertia about the joint (cantilevered slender-segment
  approximation): \(I_{eff} \approx \tfrac13 m_{seg} L_{seg}^2 = \tfrac{\pi}{3}\rho r^2
  L_{seg}^3\)
- Natural frequency: \(\omega_n = \sqrt{k_{bend}/I_{eff}}\)
- Damping ratio: \(\zeta = \dfrac{c_{bend}}{2\sqrt{k_{bend}\cdot I_{eff}}}\)

Instead of sampling `bend_damping` independently, sample a dimensionless **damping
ratio** \(\zeta\) (e.g. `{"min": 0.5, "max": 1.5}`) and derive:

$$c_{bend} = \zeta \cdot 2\sqrt{k_{bend}\cdot I_{eff}} = \zeta \cdot 2\sqrt{\tfrac{\pi}{3}\,k_{bend}\,\rho\, r^2\, L_{seg}^3}$$

using that same env's already-sampled `bend_stiffness`, `radius`, `density`, `length`,
`num_segments`. Every draw then lands at a controlled, geometry-consistent damping
ratio instead of an absolute number that means something different for every geometry
combination.

## 4. Deriving `bend_stiffness` from Young's modulus, not sampling it directly

Bending stiffness is itself not a free material parameter — it decomposes as \(k_{bend}
\propto E \cdot I_{area}\), where \(E\) (Young's modulus) is an intrinsic material
property and \(I_{area} = \pi r^4/4\) is the cross-section's second moment of area (for
a solid circular rod). The per-joint discretized bending stiffness the solver consumes
is:

$$k_{bend,joint} = \frac{E\, I_{area}}{L_{seg}} = \frac{E\cdot \pi r^4/4 \cdot N}{\text{length}}$$

Combining with §3's \(I_{eff}\):

$$\omega_n = \sqrt{\frac{3E}{4\rho}}\cdot \frac{r\cdot N^2}{\text{length}^2}$$

Practical implication: **sample `E` (and derive `bend_stiffness` per env from that env's
own geometry), not `bend_stiffness` directly.** A stiffness range tuned for one geometry
silently becomes wrong the moment `radius`/`length` are domain-randomized independently
— the same physical material needs different `bend_stiffness` at different radii (by
\(r^4\)!). Sampling `E` keeps material and geometry properly decoupled, and it directly
matches the M3 CEM sys-ID target (θ becomes lower-dimensional and geometry-invariant —
see §6).

**Units caveat (already flagged in `docs/real-world-proxy.md`):** the current proxy
stiffness table (210–736 "N/m") is a **cantilever tip force/deflection** measurement
(\(k_{cantilever} = 3EI_{area}/L^3\)), not the solver's per-joint torque/radian
quantity. Converting requires knowing the bench geometry used for that measurement:

$$E = \frac{3\,k_{cantilever}\cdot L^3}{\pi r^4}$$

## 5. The `ω_n · dt` numerical guard (independent of physical correctness)

Even with `bend_damping` correctly derived from a target \(\zeta\), and even with `E`
sys-ID'd and `ρ` taken from a defensible source, a draw can still be **numerically**
unstable: if the fixed simulation substep `dt` isn't small relative to the joint's
oscillation period \(T = 2\pi/\omega_n\), the discrete-time integrator can't represent
the continuous-time system the \(\zeta\) formula assumed. This is a sampling/resolution
problem, not a damping problem — no choice of \(\zeta\) fixes it.

Two concrete mechanisms:

- **Aliasing:** rule of thumb, want \(T/dt \gtrsim 10\text{–}20\) samples per period,
  i.e. \(\omega_n \cdot dt \lesssim 0.3\text{–}0.6\).
- **Iterative solver non-convergence:** `make_fruiting_solver_vbd`
  (`apple_pick_sim/fruiting_system/build.py:868-879`) runs a fixed `iterations: 50`
  budget per substep. A joint whose local stiffness is very high relative to `dt` can
  fail to converge within that budget regardless of its analytical damping.

**Why `num_segments` matters more than expected:** from the combined formula in §4,
\(\omega_n \propto N^2\) — natural frequency scales with the *square* of segment count,
independent of whether `E`/`ρ` are "correct." Fixtures already randomize
`num_segments` (e.g. `{"min": 2, "max": 6}`), so a max-segment draw is `9×` faster in
\(\omega_n\) than a min-segment draw from the same fixture, purely from discretization.
This axis needs to be included in the guard, not just `E`/`ρ`/`radius`/`length`.

**Application, given shared `sim_dt` across a batch:** `sim_dt` is one scalar for the
whole batch in `example_batched_heterogeneous_coupled_fruiting.py`
(`self.sim_dt = (1.0/60.0)/self.sim_substeps`), so the guard must act at **sampling
time** per env (reject/resample or clamp the offending axis — most naturally
`num_segments` given its \(N^2\) leverage), not at integration time. It should also be
checked against a fixture's *worst corner* (max `num_segments`, max `E`, min `ρ`, min
`radius`) at fixture-authoring time, not only per-draw, since a fixture whose individual
ranges each look reasonable can still have an unresolvable worst corner.

## 6. Sys-ID targets: `E`, `ζ`, `ρ` — not raw `bend_stiffness`/`bend_damping`

The existing two-phase excitation protocol in `docs/system_identification.md` already
separates exactly the right quantities:

- **§2.1 quasi-static stepped mapping** (suppresses velocity/inertia effects) →
  identifies **`E`** directly: fit the force/deflection curve to a `K`, then
  \(E = 3K L^3/(\pi r^4)\) using measured `radius`/`length`.
- **§2.2 log chirps** → the resonance response separates **`ρ`** and **`ζ`** via two
  independent features of the same signal:
  - **Oscillation frequency** (peak location / peak-to-peak timing in a ringdown) →
    \(\omega_n\) → (with `K` already known from §2.1) → \(\rho = K/(\omega_n^2 \cdot
    \text{geometry factor})\).
  - **Amplitude decay rate** (log decrement \(\delta = \ln(x_i/x_{i+1})\), or frequency-domain
    peak width / quality factor) → \(\zeta = \delta/\sqrt{4\pi^2+\delta^2}\), independent
    of `ρ`'s magnitude.

  The transient right after each "fast move" in §2.1 (currently discarded — only the
  settled hold value is used for the stiffness fit) is itself a free-decay ringdown and
  may already contain enough information for a first-pass `(ρ, ζ)` estimate without a
  separate chirp.

Treat these closed-form per-segment SDOF estimates as an initializer (\(\mu_0\), tight
\(\Sigma_0\)) for the full multi-body CEM/MMD fit in §4 of `system_identification.md`,
not as final ground truth — the real system is a coupled multi-mode chain.

### If density can't be measured directly

- **Don't fix `ρ` to a single point.** Real branches vary in density (species,
  moisture, age); collapsing to one value understates real branch-to-branch variance
  and narrows the training distribution incorrectly.
- **Source a range from literature**, matching the correct regime: real wood-species
  density tables (green/living-wood figures, not seasoned-lumber tables) if the
  physical rig is literal wood; the material spec sheet if the bench proxy is a
  mechanical rig with rigid links (per `docs/real-world-proxy.md`'s description of
  "rigid links... ball-socket... joints" — confirm which applies before picking a
  source).
- **Sanity-check regime consistency** between the sys-ID'd `E` and the assumed `ρ`
  range: `E` and `ρ` are physically linked through moisture content in real wood
  (wetter → lower `E`, higher `ρ`; drier → higher `E`, lower `ρ`), so an independently
  chosen `ρ` range could combine with the sys-ID'd `E` to produce an unphysical
  combination. If possible, measure `ρ` on the *same* specimens used for the `E`
  sys-ID to preserve this correlation instead of treating them as independent axes.
- **Don't collapse `E` to a bare point estimate either.** Once `ρ` is fixed/literature-
  sourced, `ζ` becomes the only remaining domain-randomized material axis unless `E`'s
  own sys-ID uncertainty (CEM confidence interval, or the existing Low/Med/High tier
  spread) is preserved as a genuine DR range.
- **Document as a placeholder** in the fixture `_comment` field (matching the existing
  convention, e.g. `fruiting_system_ranges_real_world_proxy.json`'s `_comment`), so it's
  revisited if CEM/MMD replay validation later shows a systematic mismatch consistent
  with a wrong density assumption.

## 7. Recommended sampling scheme (summary)

Replace independent sampling of `(bend_stiffness, bend_damping, radius, length,
density)` with:

1. Sample geometry as today: `radius`, `length`, `num_segments`, `density` (or use a
   literature-informed range per §6 if unmeasured).
2. Sample `E` (material stiffness; ideally sys-ID'd tiers with preserved uncertainty,
   §6) instead of `bend_stiffness` directly. Derive `bend_stiffness = E · π r^4/4 · N /
   length` per env from that env's own sampled geometry.
3. Sample `ζ` (damping ratio) broadly (§3) instead of `bend_damping` directly. Derive
   `bend_damping = ζ · 2√(bend_stiffness · I_eff)` per env.
4. Apply the `ω_n · dt` guard (§5) against the full per-env combination — including
   `num_segments`'s \(N^2\) leverage — both at fixture-authoring time (worst-corner
   check) and at per-env sampling time (reject/resample) as defense in depth.
5. Validate empirically with the existing settle-stability sweep tooling
   (`apple_pick_sim/diagnostics/sweep_zero_vic_stability.py`,
   `apple_pick_sim/coupled_fruiting/settle_quasi_static.py::settle_stability_reports_from_cable`)
   before locking in new fixture ranges, since all closed-form formulas above are
   single-joint/SDOF approximations of a coupled multi-body VBD system.

## 8. Open items / follow-up

- [ ] Decide fixture JSON schema change: add `damping_ratio` (replacing `bend_damping`)
  and either `youngs_modulus` (replacing `bend_stiffness`) or keep `bend_stiffness` as a
  derived/output-only field for back-compat, with a `damping_mode`/`stiffness_mode` flag
  for migration. Update `_validate_ranges` (`apple_pick_sim/fruiting_system/params.py:710-812`)
  accordingly.
- [ ] Implement derivation in `sample_params` (`apple_pick_sim/fruiting_system/params.py:318-463`),
  test-first per TDD.
- [ ] Implement the `ω_n · dt` guard as a fixture-validation check and/or a
  resample-on-violation path in `sample_params` / `sample_heterogeneous_params_list`.
- [ ] Back-convert the existing proxy stiffness table (210–736 N/m) to an `E` range
  using the bench geometry actually used for that measurement (§4 cantilever formula).
- [ ] Source a literature-appropriate `ρ` range per §6, confirm wood-vs-rig-material
  regime, and add the placeholder note to affected fixture `_comment` fields.
- [ ] Extend `docs/system_identification.md` §2.1/§2.2 post-processing to report `E`,
  `ρ`, `ζ` explicitly (not raw `K`, `B`) so CEM's θ vector in §4 matches this doc's
  targets.
- [ ] Re-run the settle-stability sweep across proposed new ranges before replacing any
  existing fixture (`apple_pick_sim/fixtures/*.json`).
