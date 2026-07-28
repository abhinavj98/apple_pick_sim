"""Tests for robot-base RGB axes logging used by robot_replay viewer example."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_example_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "robot_replay"
        / "example_view_pre_grasp_settle.py"
    )
    spec = importlib.util.spec_from_file_location("example_view_pre_grasp_settle", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _MockViewer:
    def __init__(self) -> None:
        self.device = "cpu"
        self.calls: list[tuple] = []

    def log_arrows(self, name, starts, ends, colors, **kw) -> None:
        self.calls.append(("arrows", name, starts, ends, colors))


@pytest.fixture(scope="module")
def example_mod():
    return _load_example_module()


def test_log_robot_base_axes_draws_rgb_at_origin(example_mod):
    viewer = _MockViewer()
    origin = (0.1, -0.2, 0.3)
    length = 0.25

    example_mod.log_robot_base_axes(viewer, origin, length)

    names = [c[1] for c in viewer.calls]
    assert names == [
        "/debug/robot_base_axis_x",
        "/debug/robot_base_axis_y",
        "/debug/robot_base_axis_z",
    ]
    assert all(c[0] == "arrows" for c in viewer.calls)

    expected_ends = [
        (0.1 + length, -0.2, 0.3),
        (0.1, -0.2 + length, 0.3),
        (0.1, -0.2, 0.3 + length),
    ]
    expected_colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    for call, end, color in zip(viewer.calls, expected_ends, expected_colors, strict=True):
        _, _, starts, ends, colors = call
        assert np.allclose(starts.numpy()[0], origin, atol=1e-6)
        assert np.allclose(ends.numpy()[0], end, atol=1e-6)
        assert np.allclose(colors.numpy()[0], color, atol=1e-6)


def test_log_robot_base_axes_noop_when_origin_none(example_mod):
    viewer = _MockViewer()
    example_mod.log_robot_base_axes(viewer, None, 0.3)
    assert viewer.calls == []


def test_parser_has_robot_base_axes_flag(example_mod):
    parser = example_mod._make_parser()
    args = parser.parse_args(
        ["--parquet", "robot_replay/s00-d00.parquet", "--robot-base-axes"]
    )
    assert args.robot_base_axes is True
    assert args.robot_base_axes_length == pytest.approx(0.3)
