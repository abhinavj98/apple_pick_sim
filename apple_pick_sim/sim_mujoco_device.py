"""MuJoCo CPU vs Warp backend selection for coupled scenes."""

from __future__ import annotations


def resolve_mujoco_use_cpu(device: str, mujoco_use_cpu: bool | None = None) -> bool:
    """Return whether ``SolverMuJoCo`` should use the CPU backend.

    When ``mujoco_use_cpu`` is ``None``: CPU Warp device → MuJoCo CPU; CUDA device →
    MuJoCo Warp (``use_mujoco_cpu=False``). MuJoCo Warp on a CPU Warp device fails at
    runtime (see ``diagnostics/benchmark_coupling.py``).
    """
    if mujoco_use_cpu is not None:
        if not mujoco_use_cpu and "cuda" not in device:
            raise ValueError(
                f"mujoco_use_cpu=False requires a CUDA Warp device, got {device!r}"
            )
        return mujoco_use_cpu
    return "cuda" not in device
