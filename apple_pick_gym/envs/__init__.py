"""Env registry for :mod:`apple_pick_gym`."""

from __future__ import annotations

from apple_pick_gym.envs.apple_pick_base_env import ApplePickBaseEnv
from apple_pick_gym.envs.apple_pick_coupled_env import ApplePickCoupledEnv
from apple_pick_gym.envs.apple_pick_replay_env import ApplePickReplayEnv
from apple_pick_gym.envs.apple_pick_sysid_env import ApplePickSysIdEnv
from apple_pick_gym.envs.apple_pick_vic_env import ApplePickVicEnv

__all__ = [
    "ApplePickBaseEnv",
    "ApplePickCoupledEnv",
    "ApplePickReplayEnv",
    "ApplePickSysIdEnv",
    "ApplePickVicEnv",
]

