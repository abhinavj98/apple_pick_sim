# Project vision

## Document status


| Field             | Value          |
| ----------------- | -------------- |
| **Last reviewed** | 2026-07-17     |
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
8. **Manipulation Policy:** Then learn a final apple-picking policy via RL in this fine-tuned simulation



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
- **Arm vs plant gravity (sim-to-real):** The coupled stack models a **gravity-compensated arm with zero payload** (Model A zero-g) and transfers **variable apple/plant load** through the lagged TCP wrench path (Model B gravity + stem harvest). RL policies are trained to be **robust to domain-randomized fruit**, not to balance link gravity. See `docs/mujoco-vbd-coupling-architecture.md` §2.5 and `docs/vectorized-coupled-fruiting.md` § Sim-to-real and RL training contract.



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
| **Fisher information**                | In this vision, a quantitative notion of how informative trajectories are for estimating parameters or reducing uncertainty; used to shape learning objectives, not as a one-line substitute for task success. |




## For agents: ambiguity defaults

When this document and the codebase disagree, **stop and surface the conflict** in your summary (do not silently “fix” the vision).

If something is unspecified:

1. Prefer **tests and existing patterns** in this repository over inventing new conventions.
2. Prefer **small, reversible changes** over large speculative frameworks.
3. Prefer **project-local code** (`apple_pick_sim/`) over edits to vendored submodules unless the task explicitly requires upstream changes.
4. Follow `.cursor/rules/` and **TDD** (tests first) for implementation work.

**Next doc to read:** `docs/ROADMAP.md` for phased work and current focus, then `docs/CODEBASE_GUIDE.md` for a map of the codebase and documentation set.