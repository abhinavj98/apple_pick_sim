#!/usr/bin/env python3
"""Claude Code PreToolUse hook (Bash): file deletion is only allowed strictly inside /tmp.

Maintainer rule (2026-09-24): "rm commands are off limits ... only allowed in tmp directory.
Make it a rule that is always enforced."

Denied unless every target resolves strictly inside ``/tmp`` (``/tmp`` itself is denied):

- ``rm`` / ``rmdir`` / ``unlink`` / ``shred`` -- anywhere in a chain, pipeline, subshell,
  ``bash -c '...'``, ``eval``, a heredoc fed to a shell, or behind ``sudo`` / ``env`` / ``nohup`` ...;
- ``find ... -delete`` and ``find ... -exec rm ...`` (checked against find's start paths);
- ``git clean -f`` (checked against the working directory).

Always denied, because the targets cannot be known before the command runs:
``xargs rm``, and target paths containing ``$``, backticks or command substitution.

Paths are resolved against the hook's ``cwd`` (tracking literal ``cd``) and through symlinks.
Scope: shell-level deletion. Deletion from inside a program (``python -c "shutil.rmtree(...)"``)
is not parsed. If the command cannot be parsed and mentions a deletion command, it is denied.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys

ALLOWED_ROOT = "/tmp"
RM_FAMILY = {"rm", "rmdir", "unlink", "shred"}
SHELLS = {"bash", "sh", "zsh", "dash", "ksh"}
# Wrappers that run their argument vector as a command: (name, number of option args to skip)
WRAPPERS = {"sudo", "env", "command", "builtin", "exec", "nohup", "nice", "time", "stdbuf", "timeout", "doas"}
KEYWORDS = {"if", "then", "else", "elif", "fi", "do", "done", "while", "until", "for", "in", "!", "{", "}"}
REDIRECTS = {">", ">>", "<", "<<", "<<<", ">&", "<&", "&>", "&>>", ">|", "<>", "<<-"}
_RM_WORD = re.compile(r"(^|[\s;&|(`/'\"])(rm|rmdir|unlink|shred|-delete|clean)(\s|$|[;&|)`'\"])")
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
_SHELL_BEFORE_HEREDOC = re.compile(r"(^|[\s;&|(])(sudo\s+)?(\S*/)?(bash|sh|zsh|dash|ksh)(\s+-\S+)*\s*$")


class Denied(Exception):
    pass


def _inside_root(path: str, cwd: str | None) -> bool:
    if "$" in path or "`" in path:
        return False
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        if cwd is None:
            return False
        path = os.path.join(cwd, path)
    resolved = os.path.realpath(path)
    root = os.path.realpath(ALLOWED_ROOT)
    return resolved.startswith(root + os.sep) and resolved != root


def _require_inside(paths: list[str], cwd: str | None, what: str) -> None:
    for p in paths:
        if not _inside_root(p, cwd):
            raise Denied(f"{what}: target {p!r} is not strictly inside {ALLOWED_ROOT} (cwd={cwd})")


def _split_heredocs(command: str) -> tuple[str, list[str]]:
    """Drop heredoc bodies from ``command``; return bodies that are fed to a shell."""
    out, shell_bodies = [], []
    lines = command.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        for m in _HEREDOC.finditer(line):
            if line[max(0, m.start() - 1) : m.start()] == "<":  # "<<<" here-string
                continue
            body = []
            while i < len(lines) and lines[i].strip() != m.group(2):
                body.append(lines[i])
                i += 1
            i += 1  # the delimiter line
            if _SHELL_BEFORE_HEREDOC.search(line[: m.start()]):
                shell_bodies.append("\n".join(body))
    return "\n".join(out), shell_bodies


def _tokens(command: str) -> list[str]:
    command = command.replace("\\\n", " ").replace("\n", " ; ")
    lex = shlex.shlex(command, posix=True, punctuation_chars=";&|()<>")
    lex.whitespace_split = True
    lex.commenters = "#"
    return list(lex)


def _is_separator(tok: str) -> bool:
    return bool(tok) and set(tok) <= set(";&|()") and tok not in ("&>",)


def _segments(tokens: list[str]) -> list[list[str]]:
    segs, cur = [], []
    skip_next = False
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok in REDIRECTS:
            if cur and cur[-1].isdigit():
                cur.pop()
            skip_next = True
            continue
        if _is_separator(tok):
            if cur:
                segs.append(cur)
            cur = []
            continue
        cur.append(tok)
    if cur:
        segs.append(cur)
    return segs


def _strip_wrappers(argv: list[str]) -> list[str]:
    while argv:
        head = os.path.basename(argv[0])
        if head in KEYWORDS or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", argv[0]):
            argv = argv[1:]
            continue
        if head in WRAPPERS:
            argv = argv[1:]
            # options of the wrapper, env assignments, a timeout duration / nice level
            while argv and (argv[0].startswith("-") or "=" in argv[0] or re.fullmatch(r"[0-9.]+[smhd]?", argv[0])):
                argv = argv[1:]
            continue
        break
    return argv


def _positional(args: list[str]) -> list[str]:
    out, only_paths = [], False
    for a in args:
        if only_paths:
            out.append(a)
        elif a == "--":
            only_paths = True
        elif not a.startswith("-"):
            out.append(a)
    return out


def _check_find(args: list[str], cwd: str | None) -> None:
    starts = []
    for a in args:
        if a.startswith("-") or a in ("(", "!", "\\("):
            break
        starts.append(a)
    deletes = "-delete" in args
    for i, a in enumerate(args):
        if a in ("-exec", "-execdir", "-ok", "-okdir") and i + 1 < len(args):
            sub = args[i + 1 :]
            head = os.path.basename(sub[0])
            if head in RM_FAMILY or (head in SHELLS and _RM_WORD.search(" ".join(sub[1:]))):
                deletes = True
    if deletes:
        _require_inside(starts or ["."], cwd, "find with deletion")


def check(command: str, cwd: str | None, depth: int = 0) -> None:
    if depth > 5:
        raise Denied("command nesting too deep to verify")
    command, shell_bodies = _split_heredocs(command)
    for body in shell_bodies:
        check(body, cwd, depth + 1)
    for seg in _segments(_tokens(command)):
        argv = _strip_wrappers(seg)
        if not argv:
            continue
        head, args = os.path.basename(argv[0]), argv[1:]
        if head == "cd":
            target = _positional(args)
            if not target:
                cwd = os.path.expanduser("~")
            elif "$" in target[0] or "`" in target[0] or target[0] == "-":
                cwd = None
            else:
                base = os.path.expanduser(target[0])
                cwd = os.path.realpath(base if os.path.isabs(base) or cwd is None else os.path.join(cwd, base))
        elif head in RM_FAMILY:
            _require_inside(_positional(args), cwd, head)
        elif head == "xargs":
            rest = _strip_wrappers([a for a in args if not a.startswith("-")])
            if rest and os.path.basename(rest[0]) in RM_FAMILY:
                raise Denied("xargs rm: targets come from stdin and cannot be verified")
        elif head == "find":
            _check_find(args, cwd)
        elif head in SHELLS and "-c" in args:
            i = args.index("-c")
            if i + 1 < len(args):
                check(args[i + 1], cwd, depth + 1)
        elif head == "eval":
            check(" ".join(args), cwd, depth + 1)
        elif head == "git" and "clean" in args:
            flags = [a for a in args[args.index("clean") + 1 :] if a.startswith("-")]
            if any(("f" in f and not f.startswith("--")) or f == "--force" for f in flags):
                _require_inside(["."], cwd, "git clean")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    if payload.get("tool_name") != "Bash":
        return 0
    command = (payload.get("tool_input") or {}).get("command") or ""
    cwd = payload.get("cwd") or os.getcwd()
    reason = None
    try:
        check(command, cwd)
    except Denied as exc:
        reason = str(exc)
    except Exception as exc:  # unparseable: fail closed only if it looks like a deletion
        if _RM_WORD.search(command):
            reason = f"could not parse the command to verify its deletion targets ({exc})"
    if reason is None:
        return 0
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"Repo rule: deleting files is only allowed strictly inside {ALLOWED_ROOT}. {reason}. "
                        "Do not work around this; ask the maintainer."
                    ),
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
