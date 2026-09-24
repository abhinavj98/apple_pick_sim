"""The repo's Claude Code PreToolUse hook: ``rm`` and friends only inside ``/tmp``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_HOOK = Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "block_rm_outside_tmp.py"


def _decision(command: str, cwd: str = "/home/user/apple_pick_sim") -> str:
    payload = {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": cwd}
    out = subprocess.run(
        [sys.executable, str(_HOOK)], input=json.dumps(payload), capture_output=True, text=True, check=True
    ).stdout
    if not out.strip():
        return "allow"
    return json.loads(out)["hookSpecificOutput"]["permissionDecision"]


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "git status && pytest -q",
        "echo rm is fine as a word",
        "grep -rn 'rm -rf' docs/",
        "rm /tmp/foo.txt",
        "rm -rf /tmp/claude-0/scratch /tmp/other",
        "cd /tmp && rm -rf ./build",
        "rm -f /tmp/a/*.log",
        "find /tmp/claude-0 -name '*.pyc' -delete",
        "rmdir /tmp/empty",
        "git rm --cached file.txt",
        "rm -rf ../x",
        "rm -f /tmp/x.log 2>/dev/null || true",
        "cat > notes.md <<'EOF'\nrun rm -rf build to clean\nEOF\ngit add notes.md",
        "git clean -n",
    ],
)
def test_allowed(command: str) -> None:
    cwd = "/tmp/work" if command == "rm -rf ../x" else "/home/user/apple_pick_sim"
    assert _decision(command, cwd) == "allow"


@pytest.mark.parametrize(
    "command",
    [
        "rm file.txt",
        "rm -rf runs/",
        "rm -rf /",
        "rm -rf /tmp",
        "rm -rf /tmp/../home/user",
        "rm -rf ~/x",
        "rm -rf $HOME/x",
        "rm -rf \"$TMPDIR/x\"",
        "rm -rf $(pwd)/x",
        "ls && rm -rf build",
        "true; /bin/rm -r build",
        "echo hi | xargs rm",
        "sudo rm -rf /var/log/x",
        "bash -c 'rm -rf build'",
        "find . -name '*.pyc' -delete",
        "find /tmp -exec rm {} \;",
        "find /home -exec rm {} \;",
        "unlink foo",
        "rmdir build",
        "env X=1 rm build",
        "cd /home/user && rm -rf x",
        "rm -rf /tmp/x build",
        "bash <<'EOF'\nrm -rf build\nEOF",
        "git clean -fdx",
        "ls \\\n && rm -r build",
    ],
)
def test_denied(command: str) -> None:
    assert _decision(command) == "deny"


def test_non_bash_payload_passes() -> None:
    payload = {"tool_name": "Read", "tool_input": {"file_path": "/etc/passwd"}}
    out = subprocess.run(
        [sys.executable, str(_HOOK)], input=json.dumps(payload), capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == ""
