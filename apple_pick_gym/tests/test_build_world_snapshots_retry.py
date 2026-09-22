"""Retry driver for example_build_world_snapshots.py: fresh-process retries on crash.

A SIGSEGV/process-abort from the documented intermittent Warp/Newton bug is not a
catchable Python exception, so retrying inside one process cannot work -- each attempt
must be a fresh OS process (matching this repo's existing process-isolated CMA
evaluation-wave mitigation). These tests stub the subprocess call; no sim build runs.
"""

from __future__ import annotations

from unittest.mock import patch

from apple_pick_gym.batched_examples.example_build_world_snapshots import (
    make_parser,
    run_with_retries,
)


def test_retries_flag_parses_with_default_zero():
    a = make_parser().parse_args(["--world-set", "w.jsonl", "--shard-size", "500", "--shard-index", "0", "--out-dir", "d"])
    assert a.retries == 0


def _fake_run(returncodes):
    """subprocess.run stand-in: returns object with .returncode from `returncodes` in order."""
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        rc = returncodes[len(calls) - 1]
        return type("R", (), {"returncode": rc})()

    run.calls = calls
    return run


def test_succeeds_on_first_attempt_when_worker_exits_zero():
    fake = _fake_run([0])
    with patch("subprocess.run", fake):
        ok, attempts = run_with_retries(["--world-set", "w.jsonl", "--out-dir", "d"], max_retries=2)
    assert ok is True and attempts == 1
    assert len(fake.calls) == 1


def test_retries_on_nonzero_exit_and_stops_at_first_success():
    fake = _fake_run([1, -11, 0])  # first two attempts "crash" (1 = error, -11 = SIGSEGV), third succeeds
    with patch("subprocess.run", fake):
        ok, attempts = run_with_retries(["--world-set", "w.jsonl", "--out-dir", "d"], max_retries=5)
    assert ok is True and attempts == 3
    assert len(fake.calls) == 3


def test_gives_up_after_max_retries_exhausted():
    fake = _fake_run([1, 1, 1])
    with patch("subprocess.run", fake):
        ok, attempts = run_with_retries(["--world-set", "w.jsonl", "--out-dir", "d"], max_retries=2)
    assert ok is False and attempts == 3  # 1 initial + 2 retries
    assert len(fake.calls) == 3


def test_worker_invocation_carries_the_hidden_worker_flag_and_original_args():
    fake = _fake_run([0])
    with patch("subprocess.run", fake):
        run_with_retries(["--world-set", "w.jsonl", "--shard-size", "2000", "--out-dir", "d"], max_retries=0)
    cmd = fake.calls[0]
    assert "--world-set" in cmd and "w.jsonl" in cmd
    assert "--shard-size" in cmd and "2000" in cmd
    assert "--_worker" in cmd
    assert "--retries" not in cmd  # the worker subprocess must not itself try to retry
