"""Screen candidate harvest worlds for build/settle/interaction stability.

The RL pipeline trains on a curated set of worlds that survive this probe (see
``world_set.py``). Per world:

1. **Rest** -- the env's build-time invalid mask (failed IK grasp / loaded rest).
2. **Hold** (episode 0) -- all-zeros action for ``hold_steps``: the minimum gains
   the policy can command. The TCP must not drift and the wrist must stay quiet.
3. **Pulls** (episodes 1..N) -- sys-ID-style scripted pulls at the real rig's gains:
   the target ramps ``pull_amplitude_m`` along one of +-x/+-y/+-z, holds, and
   returns. The real rig reads ~4-11 N on 3 cm pulls; the caps below are generous
   and exist to catch numerical blow-ups, not to grade realism.

Success/safety freezing must be disabled on the env being screened
(:func:`screening_episode_config`) so every world follows the full script.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch

# Order chosen so pull_axis_indices spreads each env over mixed signs/axes.
PULL_AXES = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)


@dataclasses.dataclass(frozen=True)
class ScreeningConfig:
    hold_steps: int = 120
    # Pull episode: rest, ramp out, hold, ramp back, settle (60 Hz steps).
    pull_rest_steps: int = 30
    pull_ramp_steps: int = 60
    pull_hold_steps: int = 30
    pull_settle_steps: int = 30
    pull_amplitude_m: float = 0.03
    num_pull_episodes: int = 3
    # Real rig pose-PD gains (dataset controller_gains: Kp 100/30, Kd 17.5/9.6 -> zeta ~0.9).
    pull_linear_k: float = 100.0
    pull_angular_k: float = 30.0
    pull_zeta: float = 0.9
    # Pass/fail caps.
    max_hold_wrist_n: float = 20.0
    max_hold_drift_m: float = 0.02
    max_pull_wrist_n: float = 40.0
    max_pull_track_err_m: float = 0.06
    max_pull_junction_n: float = 100.0
    max_pull_end_wrist_excess_n: float = 5.0

    @property
    def pull_episode_steps(self) -> int:
        return (
            self.pull_rest_steps
            + 2 * self.pull_ramp_steps
            + self.pull_hold_steps
            + self.pull_settle_steps
        )

    @property
    def total_steps(self) -> int:
        return self.hold_steps + self.num_pull_episodes * self.pull_episode_steps


def screening_episode_config(cfg: ScreeningConfig) -> Any:
    """An EpisodeConfig that never freezes (no success streak, no safety cap)."""
    from apple_pick_gym.batched_envs.harvest_episode import EpisodeConfig

    return EpisodeConfig(
        success_streak_steps=10**9, safety_force_cap_n=1e12, safety_torque_cap_nm=1e12
    )


def pull_axis_indices(*, num_envs: int, num_episodes: int) -> np.ndarray:
    """``(num_episodes, num_envs)`` indices into ``PULL_AXES``; env ``i`` episode ``k``
    pulls along axis ``(i + 2k) % 6`` so two adjacent envs cover all six directions."""
    k = np.arange(num_episodes)[:, None]
    i = np.arange(num_envs)[None, :]
    return (i + 2 * k) % len(PULL_AXES)


def pull_delta_schedule(cfg: ScreeningConfig) -> np.ndarray:
    """Per-step scalar target displacement along the pull axis for one pull episode."""
    step = cfg.pull_amplitude_m / cfg.pull_ramp_steps
    return np.concatenate(
        [
            np.zeros(cfg.pull_rest_steps),
            np.full(cfg.pull_ramp_steps, step),
            np.zeros(cfg.pull_hold_steps),
            np.full(cfg.pull_ramp_steps, -step),
            np.zeros(cfg.pull_settle_steps),
        ]
    )


def probe_actions(axes: torch.Tensor, *, delta_m: float, cfg: ScreeningConfig, device: Any) -> torch.Tensor:
    """13-D harvest actions: ``delta_m`` along each env's axis at the real rig gains."""
    n = axes.shape[0]
    a = torch.zeros((n, 13), dtype=torch.float32, device=device)
    a[:, :3] = axes.to(device=device, dtype=torch.float32) * float(delta_m)
    a[:, 6:9] = cfg.pull_linear_k
    a[:, 9:12] = cfg.pull_angular_k
    a[:, 12] = cfg.pull_zeta
    return a


_CRITERIA = (
    # (metric, cap attribute or None for boolean-fail metrics)
    ("rest_invalid", None),
    ("nonfinite", None),
    ("hold_max_wrist_n", "max_hold_wrist_n"),
    ("hold_tcp_drift_m", "max_hold_drift_m"),
    ("pull_max_wrist_n", "max_pull_wrist_n"),
    ("pull_max_track_err_m", "max_pull_track_err_m"),
    ("pull_max_junction_n", "max_pull_junction_n"),
    ("pull_end_wrist_excess_n", "max_pull_end_wrist_excess_n"),
)


def evaluate_screening(
    metrics: dict[str, np.ndarray], cfg: ScreeningConfig
) -> tuple[np.ndarray, list[list[str]]]:
    """Per-world pass flag and human-readable failure reasons. NaN metrics fail."""
    n = len(next(iter(metrics.values())))
    reasons: list[list[str]] = [[] for _ in range(n)]
    for key, cap_attr in _CRITERIA:
        v = np.asarray(metrics[key])
        if cap_attr is None:
            bad = v.astype(bool)
            for i in np.nonzero(bad)[0]:
                reasons[i].append(key)
        else:
            cap = float(getattr(cfg, cap_attr))
            bad = ~(v <= cap)  # NaN -> bad
            for i in np.nonzero(bad)[0]:
                reasons[i].append(f"{key}={float(v[i]):.4g}>{cap:g}")
    passed = np.array([not r for r in reasons], dtype=bool)
    return passed, reasons


def run_screening(env: Any, cfg: ScreeningConfig) -> dict[str, np.ndarray]:
    """Drive ``env`` through the hold + pull episodes; return per-env metrics (numpy)."""
    n, dev = env.num_envs, env.device

    def norm3(x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(x[:, :3], dim=-1)

    nonfinite = torch.zeros(n, dtype=torch.bool, device=dev)

    def track(info: dict[str, Any], obs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal nonfinite
        wrist = norm3(info["ft_wrist"])
        junction = torch.stack([norm3(v) for v in info["woody_part_force"].values()], dim=-1).amax(dim=-1)
        err = torch.linalg.norm(obs["tcp_pos"] - info["target_pose"][:, :3], dim=-1)
        nonfinite |= ~(torch.isfinite(wrist) & torch.isfinite(junction) & torch.isfinite(err))
        return wrist, junction, err

    # -- episode 0: zero-action hold -------------------------------------------------------
    obs, info = env.reset()
    rest_invalid = info["invalid_env"].clone()
    tcp0 = obs["tcp_pos"].clone()
    hold_wrist = torch.zeros(n, device=dev)
    zero = torch.zeros((n, 13), dtype=torch.float32, device=dev)
    for _ in range(cfg.hold_steps):
        obs, _r, _te, _tr, info = env.step(zero)
        wrist, _j, _e = track(info, obs)
        hold_wrist = torch.maximum(hold_wrist, torch.nan_to_num(wrist, nan=float("inf")))
    hold_drift = torch.linalg.norm(obs["tcp_pos"] - tcp0, dim=-1)

    # -- pull episodes --------------------------------------------------------------------
    axes_idx = pull_axis_indices(num_envs=n, num_episodes=cfg.num_pull_episodes)
    schedule = pull_delta_schedule(cfg)
    pull_wrist = torch.zeros(n, device=dev)
    pull_junction = torch.zeros(n, device=dev)
    pull_err = torch.zeros(n, device=dev)
    end_excess = torch.zeros(n, device=dev)
    for k in range(cfg.num_pull_episodes):
        obs, info = env.reset()
        axes = torch.as_tensor(PULL_AXES[axes_idx[k]], dtype=torch.float32, device=dev)
        rest_wrist = None
        for t, d in enumerate(schedule):
            obs, _r, _te, _tr, info = env.step(probe_actions(axes, delta_m=float(d), cfg=cfg, device=dev))
            wrist, junction, err = track(info, obs)
            if t == cfg.pull_rest_steps - 1:
                rest_wrist = wrist.clone()
            inf = float("inf")
            pull_wrist = torch.maximum(pull_wrist, torch.nan_to_num(wrist, nan=inf))
            pull_junction = torch.maximum(pull_junction, torch.nan_to_num(junction, nan=inf))
            pull_err = torch.maximum(pull_err, torch.nan_to_num(err, nan=inf))
        end_excess = torch.maximum(end_excess, torch.nan_to_num(wrist - rest_wrist, nan=float("inf")))

    to_np = lambda x: x.detach().float().cpu().numpy() if x.dtype != torch.bool else x.cpu().numpy()
    return {
        "rest_invalid": to_np(rest_invalid),
        "nonfinite": to_np(nonfinite),
        "hold_max_wrist_n": to_np(hold_wrist),
        "hold_tcp_drift_m": to_np(hold_drift),
        "pull_max_wrist_n": to_np(pull_wrist),
        "pull_max_track_err_m": to_np(pull_err),
        "pull_max_junction_n": to_np(pull_junction),
        "pull_end_wrist_excess_n": to_np(end_excess),
    }
