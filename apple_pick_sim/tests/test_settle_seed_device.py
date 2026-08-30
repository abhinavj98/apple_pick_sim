"""Tests for device-side settle seeding helpers."""

from __future__ import annotations

import numpy as np
import warp as wp

from apple_pick_sim.coupled_fruiting import settle_seed_device as seed_mod
from apple_pick_sim.coupled_fruiting.settle_seed_device import capture_body_q_numpy

_POSES = np.array(
    [
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        [1.5, -2.5, 0.75, 0.0, 0.70710678, 0.0, 0.70710678],
    ],
    dtype=np.float32,
)


def _body_q(poses: np.ndarray) -> wp.array:
    return wp.array(poses, dtype=wp.transform, device="cpu")


def test_capture_body_q_numpy_reads_poses():
    out = capture_body_q_numpy(_body_q(_POSES))

    assert out.shape == (2, 7)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, _POSES, rtol=0, atol=1e-6)


def test_capture_body_q_numpy_does_not_alias_warp_storage():
    """A retained view would be freed with the scene and dangle on the next rebuild."""
    body_q = _body_q(_POSES)
    out = capture_body_q_numpy(body_q)

    body_q.assign(np.zeros_like(_POSES))

    np.testing.assert_allclose(out, _POSES, rtol=0, atol=1e-6)


def test_capture_body_q_numpy_does_not_route_through_torch(monkeypatch):
    """``wp.to_torch`` hands Warp-owned storage to torch for a one-shot host read.

    The rebuild path calls this on a VBD ``body_q`` immediately before discarding the
    scene that owns it, which is the suspected source of the host-heap corruption that
    surfaces as a SIGSEGV in ``warp.context.pack_arg`` on the next scene.
    """

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("capture_body_q_numpy must not call wp.to_torch")

    monkeypatch.setattr(seed_mod.wp, "to_torch", _forbidden)

    out = capture_body_q_numpy(_body_q(_POSES))

    np.testing.assert_allclose(out, _POSES, rtol=0, atol=1e-6)
