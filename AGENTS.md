# Agent instructions

Read these **before** substantial implementation work. They apply to automated agents and humans using agent-assisted workflows.

## Source of truth (read in this order)

1. **`docs/VISION.md`** — Intent, scope, non-goals, success criteria, ambiguity defaults.
2. **`docs/ROADMAP.md`** — Current focus, milestones, ordered next slices, validation commands, when to stop and ask.
3. **Post-grasp VIC (when relevant):** **`docs/variable-impedance-teleop.md`** — dynamic arm, total TCP wrench, FD modes for \(\pi_{\mathrm{exp}}\).
4. **`.cursor/rules/`** — Persistent project rules (environment, TDD, tooling, Newton layout, **GPU/Warp parallelism**). Obey them unless the maintainer overrides them for a specific task.

If **vision**, **roadmap**, and **code** disagree, **stop** and report the conflict instead of silently choosing a direction.

## Repository layout

| Path | Role |
|------|------|
| `apple_pick_sim/` | Project-local simulation code; runnable examples in `examples/`. |
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
