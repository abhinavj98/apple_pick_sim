"""CLI contract tests for Young's-modulus E-grid examples."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


def _load_module(filename: str, name: str):
    path = Path(__file__).resolve().parents[1] / "batched_examples" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_keyboard_cli_defaults_and_help(monkeypatch):
    module = _load_module(
        "example_batched_youngs_modulus_keyboard.py",
        "example_batched_youngs_modulus_keyboard_under_test",
    )
    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)
    parser = module._make_parser()
    args = parser.parse_args([])
    assert args.log10_e_primary == "8.0,8.5"
    assert args.log10_e_spur == "7.5"
    assert args.log10_e_stem == "7.0"
    assert args.seed == 42
    help_text = parser.format_help()
    assert "--log10-e-primary" in help_text


def test_candidates_from_log10_cli_cartesian():
    from apple_pick_gym.batched_examples._youngs_e_grid_cli import candidates_from_log10_cli

    cands = candidates_from_log10_cli(
        log10_e_primary="8.0,8.5",
        log10_e_spur="7.5",
        log10_e_stem="7.0",
    )
    assert len(cands) == 2
    assert cands[0].primary == pytest.approx(1.0e8)
    assert cands[1].primary == pytest.approx(10**8.5)
