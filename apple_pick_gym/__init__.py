"""Gymnasium environments wrapping :mod:`apple_pick_sim`.

This package is intentionally separate from :mod:`apple_pick_sim` so simulation code
does not import Gymnasium (see M2.1 constraints in docs/ROADMAP.md).
"""

from __future__ import annotations

from importlib import import_module


def _register_envs() -> None:
    try:
        from gymnasium.envs.registration import register
    except Exception:
        # Gymnasium is optional at import time; tests/install paths should provide it.
        return

    # Avoid double-registration if imported multiple times.
    try:
        import gymnasium as gym

        if "ApplePickCoupled-v0" in gym.envs.registry:
            return
    except Exception:
        # If gym registry isn't accessible, attempt a best-effort register anyway.
        pass

    register(
        id="ApplePickCoupled-v0",
        entry_point="apple_pick_gym.envs:ApplePickCoupledEnv",
    )


# Ensure envs are registered on import.
_register_envs()

# Re-export env class for convenience.
ApplePickCoupledEnv = import_module("apple_pick_gym.envs").ApplePickCoupledEnv

__all__ = ["ApplePickCoupledEnv"]

