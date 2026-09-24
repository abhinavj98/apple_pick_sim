# Project vision

## Document status


| Field             | Value          |
| ----------------- | -------------- |
| **Last reviewed** | 2026-09-24     |
| **Owner**         | Abhinav        |
| **Related**       | `./ROADMAP.md` |


**How to use this file:** keep it short (about one to three screens). Detailed sequencing lives in `docs/ROADMAP.md`. Update **Last reviewed** when scope, success criteria, or non-goals change.

**For agents:** this file is the source of truth for intent, boundaries, and success criteria. If details are missing, infer from `ROADMAP.md` and code; do not contradict this document without maintainer input. When unsure, follow **Ambiguity defaults** at the bottom.

## One-line mission

Build an apple-picking simulator whose parameters are grounded in and refined against real-world data, to improve sim-to-real transfer for manipulation of compliant plant tissue and fruit.

## Problem statement

- **Context:** Highly stiff contact and articulated plant models were historically unstable in simulation. The Newton physics engine’s AVBD-style formulation enables stable simulation of stiff, coupled systems relevant to trees, stems, and fruit.
- **Pain:** Matching contact forces, dynamics, and compliance for sim-to-real transfer remains difficult without data and a clear calibration loop.
- **Opportunity:** Combine rich simulation (Newton), policy learning, and real trajectories so the simulator becomes a testbed for deformable and articulated manipulation under procedural scene variation.



## Target outcomes (vision-level)

1. **Visual and structural variance:** Read properties of the fruiting system from configuration or data files, then procedurally vary geometry and layout so policies and estimators see diverse but plausible canopies and fruit.
2. **Manipulation stack:** Use Newton with appropriate solvers for plant and fruit physics, and integrate MuJoCo-based control and contact where needed for a Franka FR3 arm interacting with the scene.
3. **Learning infrastructure:** Build reinforcement-learning tooling so a policy can be trained in simulation; exploration and reward design should support objectives such as maximizing Fisher information (informative trajectories for identification and downstream transfer).
4. **Replayable observation data:** Define the smallest real-world observable bundle needed to initialize and replay sys-ID episodes without privileged simulator arrays, then use sim-to-sim tests to quantify the drift introduced by partial state information.
5. **Digital-twin scene reconstruction:** Use calibrated geometry observations and named fixture catalogs to rebuild fruiting-system topology, base poses, apple/stem frames, and grasp transforms before tuning dynamics.
6. **Real-world data:** Collect trajectories with the same (or closely matched) policy and sensing assumptions used in simulation, so datasets align across the sim–real gap.
7. **Calibration loop:** First verify parameter recovery and held-out improvement in sim-to-sim experiments, then use real-world observations together with gradients, sensitivity information, or black-box objectives to update physical and scene parameters where agreement matters for manipulation.
8. **Manipulation Policy:** Then learn a final apple-picking policy via RL in this fine-tuned simulation. Concretely ([M5], H6 `docs/handbook-rl-policy.md`): a recurrent **variable-impedance (VIC)** policy that, from a grasped apple and using only proprioception and a realistic wrist F/T signal, loads the **spur–stem junction** past its combined force–torque **detach envelope** \((F/F_{max})^2 + (\tau/\tau_{max})^2 \ge 1\) (\(F_{max} = 20\) N, \(\tau_{max} = 0.05\) N·m) while adding as little load as possible to every other junction (spur, primary, supports, apple–stem) — exploiting the twist-and-pull synergy a human picker uses. It is trained with domain randomization over the calibrated plant, the arm and the F/T sensor so it transfers to the real rig's `vic_pose` interface without a conversion layer.



## Success criteria (measurable where possible)


| Criterion                     | How we know                                                                                                                     | Notes                                                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Procedural fruiting variance  | Automated script or test asserts distinct geometry or labels across seeds; runs in CI or locally                                | Same generator API, different seeds → different valid scenes                                                       |
| Simulated pick / interaction  | Regression test or recorded metric on a canonical scenario (e.g. stem load, slip, or task success)                              | Golden scenarios may start simple and grow with milestones                                                         |
| Policy and data alignment     | Documented policy interface; real logs can be ingested next to sim rollouts without ad hoc rewrites                             | Formats and observation spaces stay versioned                                                                      |
| Observation-only replay       | With privileged snapshots withheld, sim-to-sim replay initialized from observations has bounded drift against privileged replay | Drift metrics include TCP/apple pose, woody marker positions, and F/T error over the same recorded action sequence |
| Digital-twin reconstruction   | A named fixture built from calibration observations recreates geometry/topology well enough for replay and parameter tuning     | Start with sim-to-sim ground truth fixtures before real-world reconstruction                                       |
| Sim-to-sim calibration        | Recovered material parameters or predicted behavior improve on held-out simulated structures or trajectories              | Required before real-data collection becomes the acceptance dependency                                             |
| Real-data calibration         | Quantitative comparison (e.g. force, pose, or event error) drops on a held-out real segment after calibration             | Ultimate sim-to-real criterion after M4 data collection; exact metric chosen in roadmap                            |
| Learned pick policy ([M5])    | On held-out screened worlds, the learned VIC policy's detach success rate (envelope reached at the spur–stem junction) beats a scripted pull, with no worse safety-violation rate and lower collateral junction load; learning curves (reward, success) rise | Sim-internal; infrastructure first proven on a CPU surrogate (learning smoke), then on the CUDA sim (H6)            |




## In scope

- Newton-based dynamics for plant and fruit where the project already relies on Newton; project-local orchestration and scenarios under `apple_pick_sim/`.
- Procedural or data-driven variation of fruiting-system assets from structured inputs.
- Integration paths for arm simulation and control (e.g. MuJoCo + Franka FR3) that match documented milestones.
- RL training harnesses, logging, and evaluation hooks tied to the simulator (Gymnasium adapter `apple_pick_gym/` — see `docs/ROADMAP.md` [M2]).
- Real-data collection protocols, observation contracts, fixture catalogs, and file formats that pair with simulation and calibration.



## Out of scope (non-goals)

Explicit boundaries so work does not expand by default.

- Production deployment, certification, or safety case for physical robots (research and simulation first).
- Full-farm logistics, economics, or long-horizon fleet scheduling.
- Replacing Newton as the primary tree and fruit dynamics backend unless a milestone explicitly migrates physics.
- Open-ended “any manipulator / any crop” generalization without a scoped milestone.



## Constraints and assumptions

- **Technical:** The `newton/` submodule remains the vendored physics engine; prefer new simulation logic in `apple_pick_sim/`. Python environment and runs follow project `uv` conventions (see `README.md` and `.cursor/rules/`).
- **Dependencies:** Upstream Newton APIs and licenses apply; do not assume unavailable proprietary assets unless provided.
- **Performance / quality:** Tests and CI-facing paths should be deterministic where practical (seeded randomness, no undeclared network dependencies). Real-time visualization is desirable but not a substitute for reproducible metrics.
- **Arm vs plant gravity (sim-to-real):** The coupled stack models a **gravity-compensated arm with zero payload** (Model A zero-g) and transfers **variable apple/plant load** through the lagged TCP wrench path (Model B gravity + stem harvest). RL policies are trained to be **robust to domain-randomized fruit**, not to balance link gravity. See H1 `docs/handbook-coupled-simulation.md`.



## Guiding principles (architecture and process)

- Prefer **deterministic** simulations and tests unless a document explicitly opts into nondeterminism.
- Keep **simulation-specific** code and scenarios under `apple_pick_sim/`; avoid drive-by edits in `newton/` unless the task is to patch or sync the submodule.
- Use **test-driven development**: failing test first, smallest change to green, then refactor.
- Prefer **small, reversible** changes over speculative frameworks.
- When vision, roadmap, and code disagree, **surface the conflict** to the maintainer instead of silently rewriting intent.



## Key terms (glossary)


| Term                                  | Definition                                                                                                                                                                                                     |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Newton**                            | The physics engine used as a submodule (`newton/`) for this project’s dynamics.                                                                                                                                |
| **AVBD**                              | A solver / formulation class in Newton suited to stiff multibody and contact-heavy models; referenced when discussing stable stiff simulation.                                                                 |
| **Fruiting system**                   | The branch, stem, leaf, and fruit arrangement treated as one configurable scene or asset family.                                                                                                               |
| **Sim-to-real**                       | Closing the gap between simulated and physical behavior (forces, timing, contacts, sensing).                                                                                                                   |
| **Zero-payload gravity compensation** | Arm feedforward cancels link gravity only; fruit mass is treated as an external EE load (sim: Model A zero-g + stem harvest; real: no apple term in gravity comp).                                             |
| **Detach envelope**                   | The elliptical combined-loading failure criterion \((F/F_{max})^2 + (\tau/\tau_{max})^2 \ge 1\) at the spur–stem junction that defines fruit detachment (success) for the pick policy; torque taken at the joint anchor. See H6.                                  |
| **VIC**                               | Variable-impedance control: the policy commands a pose target plus stiffness (and a damping ratio, \(D = 2\zeta\sqrt{K}\)) rather than torques or positions; the same `vic_pose` interface as the real rig (H2).                                          |
| **Domain randomization (DR)**         | Training over randomized plant materials/geometry, support joints, grasp, arm dynamics and F/T sensor bias/noise/drift so the policy is robust to what calibration leaves uncertain; the critic sees every draw (privileged), the actor does not. |
| **Fisher information**                | In this vision, a quantitative notion of how informative trajectories are for estimating parameters or reducing uncertainty; used to shape learning objectives, not as a one-line substitute for task success. |




## For agents: ambiguity defaults

When this document and the codebase disagree, **stop and surface the conflict** in your summary (do not silently “fix” the vision).

If something is unspecified:

1. Prefer **tests and existing patterns** in this repository over inventing new conventions.
2. Prefer **small, reversible changes** over large speculative frameworks.
3. Prefer **project-local code** (`apple_pick_sim/`) over edits to vendored submodules unless the task explicitly requires upstream changes.
4. Follow `.cursor/rules/` and **TDD** (tests first) for implementation work.

**Next doc to read:** `docs/ROADMAP.md` for phased work and current focus, then `docs/CODEBASE_GUIDE.md` for a map of the codebase and documentation set.