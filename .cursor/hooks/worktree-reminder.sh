#!/usr/bin/env bash
# sessionStart: remind agents about worktree workflow when on main/master.
set -euo pipefail

cat >/dev/null

root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
branch="$(git -C "$root" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
worktrees="$(git -C "$root" worktree list 2>/dev/null || echo "(none)")"

python3 - "$branch" "$root" "$worktrees" <<'PY'
import json
import sys

branch, root, worktrees = sys.argv[1:4]

if branch in ("main", "master"):
    context = (
        f"WORKTREE CHECKLIST: Current branch is '{branch}' at {root}. "
        "Before production code changes, create or reuse an isolated git worktree "
        "(see .cursor/rules/worktree-feature-dev.mdc). "
        f"Existing worktrees:\n{worktrees}"
    )
else:
    context = (
        f"Git branch: {branch} at {root}. Keep feature changes on this branch. "
        f"Existing worktrees:\n{worktrees}"
    )

print(
    json.dumps(
        {
            "additional_context": context,
            "env": {
                "APPLE_PICK_SIM_GIT_ROOT": root,
                "APPLE_PICK_SIM_GIT_BRANCH": branch,
            },
        }
    )
)
PY
