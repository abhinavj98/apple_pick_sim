"""CLI tests for the sys-ID dataset dashboard example."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_example_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "dashboard_sysid_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("dashboard_sysid_dataset_under_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parser_defaults_to_localhost_dashboard():
    module = _load_example_module()

    args = module._make_parser().parse_args(["--dataset", "/tmp/sysid_dataset"])

    assert args.dataset == "/tmp/sysid_dataset"
    assert args.host == "127.0.0.1"
    assert args.port == 8050
    assert args.debug is False
    assert args.open_browser is False


def test_parser_rejects_missing_dataset_argument():
    module = _load_example_module()

    try:
        module._make_parser().parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parser accepted missing --dataset")


def test_parser_description_mentions_sysid_dashboard():
    module = _load_example_module()

    parser = module._make_parser()

    assert isinstance(parser, argparse.ArgumentParser)
    assert "sys-ID dataset dashboard" in parser.description
