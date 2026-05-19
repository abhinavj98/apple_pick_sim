"""Tests for staggered coupling force debug readouts."""

from __future__ import annotations

import numpy as np
import pytest


def _import_debug():
    import apple_pick_sim.coupling_force_debug as cfd

    return cfd


def test_wrench_magnitudes_from_spatial_vector():
    cfd = _import_debug()
    w = np.array([3.0, 4.0, 0.0, 0.0, 0.0, 12.0], dtype=np.float64)
    fmag, tmag = cfd.wrench_magnitudes(w)
    assert fmag == pytest.approx(5.0)
    assert tmag == pytest.approx(12.0)


def test_read_tcp_wrench_from_flat_body_array():
    cfd = _import_debug()
    wrenches = np.zeros((4, 6), dtype=np.float32)
    wrenches[2] = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0]
    w = cfd.read_tcp_wrench(wrenches, tcp_body_index=2)
    np.testing.assert_allclose(w, [1.0, 0.0, 0.0, 0.0, 2.0, 0.0])


def test_recorder_stores_applied_and_harvested():
    cfd = _import_debug()
    rec = cfd.CouplingForceDebugRecorder()
    applied = np.array([3.0, 0.0, 4.0, 0.0, 0.0, 0.0], dtype=np.float64)
    harvested = np.array([0.0, 6.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    rec.record_applied(applied)
    rec.record_harvested(harvested)
    assert rec.applied_force_mag == pytest.approx(5.0)
    assert rec.harvested_force_mag == pytest.approx(6.0)
    assert rec.harvested_torque_mag == pytest.approx(1.0)


class _ScalarViewer:
    def __init__(self) -> None:
        self.logged: list[tuple[str, float]] = []

    def log_scalar(self, name: str, value: float, *, clear: bool = False, smoothing: int = 1):
        del clear, smoothing
        self.logged.append((name, float(value)))


def test_log_to_viewer_emits_both_force_series():
    cfd = _import_debug()
    rec = cfd.CouplingForceDebugRecorder()
    rec.record_applied(np.array([3.0, 4.0, 0.0, 0.0, 0.0, 0.0]))
    rec.record_harvested(np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]))
    viewer = _ScalarViewer()
    rec.log_to_viewer(viewer)
    names = [n for n, _ in viewer.logged]
    assert any("MuJoCo" in n and "|F|" in n for n in names)
    assert any("VBD harvest" in n and "|F|" in n for n in names)
