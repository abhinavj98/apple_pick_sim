"""Default Warp device selection for apple_pick_sim builders and examples."""

from __future__ import annotations

import os

import warp as wp

_CUDA_DEVICE = "cuda:0"
_CPU_DEVICE = "cpu"


def default_sim_device() -> str:
    """Return the preferred simulation device: ``cuda:0`` when CUDA is available, else ``cpu``."""
    override = os.environ.get("APPLE_PICK_SIM_DEVICE")
    if override:
        return override
    if wp.is_cuda_available():
        return _CUDA_DEVICE
    return _CPU_DEVICE


def resolve_sim_device(device: str | None = None) -> str:
    """Use ``device`` when set; otherwise :func:`default_sim_device`."""
    if device is not None:
        return device
    return default_sim_device()
