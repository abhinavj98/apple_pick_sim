"""CUDA graph helpers for example simulation loops."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import warp as wp


def can_capture_graph(device: str | wp.Device) -> bool:
    """True when Warp CUDA graphs are supported on ``device``."""
    dev = wp.get_device(device) if isinstance(device, str) else device
    return bool(dev.is_cuda and wp.is_mempool_enabled(dev))


def capture_substep_loop(
    fn: Callable[[], None],
    *,
    device: str | wp.Device,
    warmup: int = 2,
) -> Any | None:
    """Warm up ``fn`` then capture it into a CUDA graph, or return ``None`` if unsupported."""
    if not can_capture_graph(device):
        return None
    dev = wp.get_device(device) if isinstance(device, str) else device
    for _ in range(warmup):
        fn()
    wp.synchronize_device(dev)
    with wp.ScopedCapture(device=dev) as cap:
        fn()
    return cap.graph
