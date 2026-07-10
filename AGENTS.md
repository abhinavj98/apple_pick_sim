# Agent instructions

Read these **before** substantial implementation work. They apply to automated agents and humans using agent-assisted workflows.

## Source of truth (read in this order)

1. **`docs/VISION.md`** — Intent, scope, non-goals, success criteria, ambiguity defaults.
2. **`docs/ROADMAP.md`** — Current focus, milestones, ordered next slices, validation commands, when to stop and ask.
3. **`docs/CODEBASE_GUIDE.md`** — Map of the codebase and the full `docs/` set; read this before searching for "which doc covers X."
4. **Post-grasp VIC (when relevant):** **`docs/variable-impedance-teleop.md`** — dynamic arm, total TCP wrench, FD modes for \(\pi_{\mathrm{exp}}\).
5. **`.cursor/rules/`** — Persistent project rules (environment, TDD, tooling, Newton layout, **GPU/Warp parallelism**). Obey them unless the maintainer overrides them for a specific task.

If **vision**, **roadmap**, and **code** disagree, **stop** and report the conflict instead of silently choosing a direction.

## Repository layout

| Path | Role |
|------|------|
| `apple_pick_sim/` | Project-local simulation code; runnable examples in `examples/`. |
| `apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py` | Canonical batched heterogeneous coupled simulation demo (V.3.2). |
| `apple_pick_gym/` | Gymnasium adapter; `batched_envs/` + `batched_examples/` for parallel collect / MMD grid. |
| `newton/` | Upstream Newton **git submodule**; treat as vendored unless the task is explicitly to patch or sync it. |
| `docs/` | Project vision and roadmap (this repo’s planning docs, not `newton/docs/`). |

## Setup and runs

See **`README.md`** for clone/submodule steps and example commands. Prefer **`uv`** for installs and execution, as described in `.cursor/rules/uv-package-manager.mdc`.

## Parallel feature work (worktrees)

For isolated feature or fix work, agents should follow `.cursor/rules/worktree-feature-dev.mdc`: create or reuse a sibling worktree (`../apple_pick_sim-<slug>`), run submodule + `uv sync` there, and ask you to open that folder in a **new Cursor window**. A `sessionStart` hook (`.cursor/hooks/worktree-reminder.sh`) reinforces this when the workspace is on `main`/`master`.

## Execution expectations

- Keep simulation **hot paths on GPU** with **NVIDIA Warp** where appropriate; see `.cursor/rules/gpu-warp-parallelism.mdc` and `docs/gpu-coupling-optimization.md`.
- Follow **TDD** (tests first) per `.cursor/rules/test-driven-development.mdc`.
- Prefer completing the **next slice** in `docs/ROADMAP.md` under **Current focus**; do not expand scope into the backlog unless instructed.
- After changes, run the **validation commands** listed in `docs/ROADMAP.md` (update that section if the canonical commands change).

# Follow these rules unless overriden

`.cursor/rules/` contains rules that should be followed unless the user explicitly overrides them for a specific task. Rules are organized by category and can be enabled/disabled individually. Common rules include: 

`.cursor/rules/task-decomposition.mdc` - breaking down tasks into smaller steps
`.cursor/rules/test-driven-development.mdc` - writing tests before implementation
`.cursor/rules/code-generation.mdc` - guidelines for code generation
`.cursor/rules/code-review.mdc` - code review checklist
`.cursor/rules/error-handling.mdc` - error handling best practices
`.cursor/rules/security.mdc` - security best practices
`.cursor/rules/performance-optimization.mdc` - performance optimization
`.cursor/rules/documentation.mdc` - documentation guidelines
`.cursor/rules/testing-strategy.mdc` - testing strategy
