"""CLI contract tests for the sys-ID example."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_example_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "example_gym_sysid.py"
    )
    spec = importlib.util.spec_from_file_location("example_gym_sysid_under_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_save_snapshot_defaults_to_false_and_is_opt_in(monkeypatch):
    module = _load_example_module()

    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)

    parser = module._make_parser()

    default_args = parser.parse_args([])
    opt_in_args = parser.parse_args(["--save-snapshot"])

    assert default_args.save_snapshot is False
    assert opt_in_args.save_snapshot is True
