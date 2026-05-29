"""Unit tests for TCP coupling force arrow debug drawing."""

from __future__ import annotations

import numpy as np
import pytest

from apple_pick_sim.tcp_force_viz import arrow_length_m, force_arrow_segment_arrays


def test_force_arrow_segment_arrays_unit_force_default_scale():
    starts, ends, colors = force_arrow_segment_arrays(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        scale_per_newton=0.02,
    )
    assert starts.shape == (1, 3)
    assert ends.shape == (1, 3)
    np.testing.assert_allclose(starts[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(ends[0], [0.02, 0.0, 0.0])
    assert colors.shape == (1, 3)


def test_force_arrow_segment_arrays_translated_origin():
    starts, ends, _colors = force_arrow_segment_arrays(
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 10.0),
        scale_per_newton=0.01,
    )
    np.testing.assert_allclose(starts[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(ends[0], [1.0, 2.0, 3.1])


def test_force_arrow_segment_arrays_below_threshold_returns_none():
    out = force_arrow_segment_arrays(
        (0.0, 0.0, 0.0),
        (1e-9, 0.0, 0.0),
        force_threshold=1e-6,
    )
    assert out is None


def test_force_arrow_segment_arrays_rejects_nonpositive_scale():
    with pytest.raises(ValueError, match="scale_per_newton"):
        force_arrow_segment_arrays((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), scale_per_newton=0.0)


def test_arrow_length_m_gain_and_clamp():
    assert arrow_length_m(10.0, scale_per_newton=0.02, gain=2.0) == pytest.approx(0.4)
    assert arrow_length_m(1.0, scale_per_newton=0.01, min_length=0.05) == pytest.approx(0.05)
    assert arrow_length_m(100.0, scale_per_newton=0.02, max_length=1.0) == pytest.approx(1.0)


def test_force_arrow_segment_arrays_respects_min_length():
    _starts, ends, _colors = force_arrow_segment_arrays(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        scale_per_newton=0.001,
        min_length=0.1,
    )
    np.testing.assert_allclose(ends[0], [0.0, 0.0, 0.1])


def test_log_tcp_force_arrow_noop_without_log_lines():
    class _Viewer:
        pass

    from apple_pick_sim.tcp_force_viz import log_tcp_force_arrow

    log_tcp_force_arrow(
        _Viewer(),
        "/debug/tcp_force",
        (0.0, 0.0, 0.0),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )


def test_log_coupled_scene_tcp_force_noop_when_proxy_forces_missing():
    class _Scene:
        proxy_forces = None
        tcp_body_index = 0
        robot_state_0 = None

    class _Viewer:
        pass

    from apple_pick_sim.tcp_force_viz import log_coupled_scene_tcp_force

    log_coupled_scene_tcp_force(_Viewer(), _Scene())
