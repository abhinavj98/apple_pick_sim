"""Unit tests for world / body coordinate frame debug drawing."""

from __future__ import annotations

import numpy as np

from apple_pick_sim.global_frame_viz import axis_segment_arrays


def test_axis_segment_arrays_identity_at_origin():
    starts, ends, colors = axis_segment_arrays(
        (0.0, 0.0, 0.0),
        quat_wxyz=None,
        length=1.0,
    )
    np.testing.assert_allclose(starts, np.zeros((3, 3)))
    np.testing.assert_allclose(ends[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(ends[1], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(ends[2], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(colors[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(colors[1], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(colors[2], [0.0, 0.0, 1.0])


def test_axis_segment_arrays_translated_origin():
    starts, ends, _colors = axis_segment_arrays(
        (2.0, -1.0, 0.5),
        quat_wxyz=None,
        length=0.5,
    )
    np.testing.assert_allclose(starts, np.full((3, 3), [2.0, -1.0, 0.5]))
    np.testing.assert_allclose(ends[0], [2.5, -1.0, 0.5])


def test_axis_segment_arrays_z90_rotates_x_axis_to_world_y():
    import math

    half = math.sqrt(0.5)
    quat = (0.0, 0.0, half, half)  # +90° about world Z (wxyz)
    _starts, ends, _colors = axis_segment_arrays((0.0, 0.0, 0.0), quat_wxyz=quat, length=1.0)
    np.testing.assert_allclose(ends[0], [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(ends[1], [-1.0, 0.0, 0.0], atol=1e-6)


def test_axis_segment_arrays_rejects_nonpositive_length():
    import pytest

    with pytest.raises(ValueError, match="length"):
        axis_segment_arrays((0.0, 0.0, 0.0), length=0.0)


def test_log_coordinate_frame_noop_without_log_lines():
    class _Viewer:
        pass

    # Must not raise when viewer lacks log_lines (headless / null viewers).
    from apple_pick_sim.global_frame_viz import log_coordinate_frame

    log_coordinate_frame(_Viewer(), "/debug/test", (0.0, 0.0, 0.0))
