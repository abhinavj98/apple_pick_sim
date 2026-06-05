"""Apply lagged spatial wrenches to MuJoCo robot ``body_f``."""

from __future__ import annotations

from typing import Any

import warp as wp


def _apply_spatial_wrench_to_body_f(state: Any, tcp_body_index: int, wrenches_spatial: wp.array) -> None:
    """Write lagged coupling wrench into ``state.body_f[tcp]``; clear all other body_f slots."""
    state.body_f.zero_()
    dev = state.body_f.device
    wp.launch(
        _write_tcp_spatial_wrench_kernel,
        dim=1,
        inputs=[state.body_f, int(tcp_body_index), wrenches_spatial],
        device=dev,
    )


def _add_tcp_spatial_wrench_inplace(
    wrenches_spatial: wp.array,
    tcp_body_index: int,
    delta: wp.spatial_vector,
) -> None:
    """Add ``delta`` to ``wrenches_spatial[tcp]`` in place."""
    dev = wrenches_spatial.device
    wp.launch(
        _add_tcp_spatial_wrench_kernel,
        dim=1,
        inputs=[wrenches_spatial, int(tcp_body_index), delta],
        device=dev,
    )


@wp.kernel
def _add_tcp_spatial_wrench_kernel(
    wrenches: wp.array(dtype=wp.spatial_vector),
    tcp_index: int,
    delta: wp.spatial_vector,
):
    wrenches[tcp_index] = wrenches[tcp_index] + delta


@wp.kernel
def _write_tcp_spatial_wrench_kernel(
    body_f: wp.array(dtype=wp.spatial_vector),
    tcp_index: int,
    wrenches: wp.array(dtype=wp.spatial_vector),
):
    body_f[tcp_index] = wrenches[tcp_index]
