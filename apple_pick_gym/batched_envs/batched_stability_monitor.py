"""Vectorized online stability monitor for batched gym observation dicts."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import torch

from apple_pick_sim.coupled_fruiting.scene import (
    DEFAULT_STEM_FORCE_CAP_N,
    DEFAULT_STEM_TORQUE_CAP_NM,
)


@dataclasses.dataclass(frozen=True)
class StabilityThresholds:
    max_force_n: float = DEFAULT_STEM_FORCE_CAP_N
    max_torque_nm: float = DEFAULT_STEM_TORQUE_CAP_NM
    max_tcp_speed_mps: float = 0.5
    max_apple_speed_mps: float = 0.5


@dataclasses.dataclass(frozen=True)
class BatchedStabilityReport:
    step_idx: int
    unstable: torch.Tensor
    reasons: list[list[str]]


@dataclasses.dataclass(frozen=True)
class PluginCheckResult:
    unstable: torch.Tensor
    reasons: list[str | None]


class StabilityCheckPlugin(Protocol):
    name: str
    required_obs_keys: frozenset[str]

    def check(self, obs: Mapping[str, Any], *, step_idx: int) -> PluginCheckResult: ...


def _as_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def _nan_or_inf_mask(x: torch.Tensor) -> torch.Tensor:
    bad = torch.isnan(x) | torch.isinf(x)
    if bad.ndim == 1:
        return bad
    return bad.any(dim=-1)


def _merge_reason_lists(
    num_envs: int,
    *reason_lists: list[list[str]],
) -> list[list[str]]:
    merged: list[list[str]] = [[] for _ in range(num_envs)]
    for reasons in reason_lists:
        for i in range(num_envs):
            if reasons[i]:
                merged[i].extend(reasons[i])
    return merged


def ik_bootstrap_unstable_mask(env: Any, num_envs: int) -> torch.Tensor:
    """Return per-env bool mask: True where IK bootstrap missed tolerance at build time."""
    n = int(num_envs)
    sim = getattr(env, "_sim", None)
    build_result = getattr(sim, "build_result", None) if sim is not None else None
    ik_results = getattr(build_result, "ik_envelope_results", None) if build_result is not None else None
    if not ik_results:
        return torch.zeros(n, dtype=torch.bool)
    out = torch.zeros(n, dtype=torch.bool)
    for i, row in enumerate(ik_results):
        if i >= n:
            break
        inside = bool(row[2]) if len(row) >= 3 else True
        out[i] = not inside
    return out


class BatchedStabilityMonitor:
    """Report-only batched stability checks over gym obs dicts."""

    def __init__(
        self,
        num_envs: int,
        *,
        known_obs_keys: set[str],
        thresholds: StabilityThresholds | None = None,
        plugins: Sequence[StabilityCheckPlugin] = (),
        initial_unstable: torch.Tensor | None = None,
        initial_reason: str = "ik_bootstrap_not_converged",
    ) -> None:
        self._num_envs = int(num_envs)
        if self._num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {self._num_envs}")
        self._known_obs_keys = set(known_obs_keys)
        self._thresholds = thresholds or StabilityThresholds()
        self._plugins = tuple(plugins)
        self._prev_apple_pos: torch.Tensor | None = None
        self._initial_reason = str(initial_reason)
        if initial_unstable is None:
            self._initial_unstable: torch.Tensor | None = None
        else:
            init = _as_tensor(initial_unstable).to(dtype=torch.bool).reshape(-1)
            if int(init.numel()) != self._num_envs:
                raise ValueError(
                    f"initial_unstable length {int(init.numel())} != num_envs {self._num_envs}"
                )
            self._initial_unstable = init

        for plugin in self._plugins:
            missing = set(plugin.required_obs_keys) - self._known_obs_keys
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise ValueError(
                    f"plugin {plugin.name!r} requires missing obs keys: {{{missing_str}}}"
                )

    def check(self, obs: Mapping[str, Any], *, step_idx: int) -> BatchedStabilityReport:
        n = self._num_envs
        device = _as_tensor(obs["ft_wrist"]).device
        unstable = torch.zeros(n, dtype=torch.bool, device=device)
        reason_lists: list[list[str]] = [[] for _ in range(n)]

        core_unstable, core_reasons = self._core_checks(obs)
        unstable = unstable | core_unstable
        reason_lists = _merge_reason_lists(n, reason_lists, core_reasons)

        for plugin in self._plugins:
            result = plugin.check(obs, step_idx=int(step_idx))
            plugin_unstable = result.unstable.to(device=device, dtype=torch.bool).reshape(-1)
            if int(plugin_unstable.numel()) != n:
                raise ValueError(
                    f"plugin {plugin.name!r} returned unstable length "
                    f"{int(plugin_unstable.numel())}, expected {n}"
                )
            unstable = unstable | plugin_unstable
            plugin_reasons: list[list[str]] = [[] for _ in range(n)]
            for i, reason in enumerate(result.reasons):
                if reason is not None:
                    plugin_reasons[i].append(str(reason))
            reason_lists = _merge_reason_lists(n, reason_lists, plugin_reasons)

        if self._initial_unstable is not None:
            init_mask = self._initial_unstable.to(device=device, dtype=torch.bool)
            unstable = unstable | init_mask
            for i in range(n):
                if bool(init_mask[i].item()):
                    reason_lists[i].append(self._initial_reason)

        return BatchedStabilityReport(
            step_idx=int(step_idx),
            unstable=unstable,
            reasons=reason_lists,
        )

    def _core_checks(self, obs: Mapping[str, Any]) -> tuple[torch.Tensor, list[list[str]]]:
        n = self._num_envs
        device = _as_tensor(obs["ft_wrist"]).device
        unstable = torch.zeros(n, dtype=torch.bool, device=device)
        reason_lists: list[list[str]] = [[] for _ in range(n)]

        def _flag(mask: torch.Tensor, reason: str) -> None:
            nonlocal unstable
            mask = mask.to(device=device, dtype=torch.bool).reshape(-1)
            if int(mask.numel()) != n:
                raise ValueError(f"mask for {reason!r} has length {int(mask.numel())}, expected {n}")
            unstable = unstable | mask
            for i in range(n):
                if bool(mask[i].item()):
                    reason_lists[i].append(reason)

        ft_wrist = _as_tensor(obs["ft_wrist"]).reshape(n, 6)
        tcp_velocity = _as_tensor(obs["tcp_velocity"]).reshape(n, 6)
        apple_pos = _as_tensor(obs["apple_pos"]).reshape(n, 3)

        _flag(_nan_or_inf_mask(ft_wrist), "nan_or_inf:ft_wrist")
        _flag(_nan_or_inf_mask(tcp_velocity), "nan_or_inf:tcp_velocity")
        _flag(_nan_or_inf_mask(apple_pos), "nan_or_inf:apple_pos")

        woody = obs.get("woody_part_info")
        if isinstance(woody, Mapping):
            for junction_name, part in woody.items():
                anchor_force = _as_tensor(part["anchor_force"]).reshape(n, -1)
                _flag(
                    _nan_or_inf_mask(anchor_force),
                    f"nan_or_inf:woody_part_info.{junction_name}.anchor_force",
                )

        force_norm = torch.linalg.norm(ft_wrist[:, :3], dim=-1)
        torque_norm = torch.linalg.norm(ft_wrist[:, 3:], dim=-1)
        _flag(force_norm > self._thresholds.max_force_n, "force_cap_exceeded")
        _flag(torque_norm > self._thresholds.max_torque_nm, "torque_cap_exceeded")

        tcp_speed = torch.linalg.norm(tcp_velocity[:, :3], dim=-1)
        _flag(tcp_speed > self._thresholds.max_tcp_speed_mps, "tcp_speed_exceeded")

        if self._prev_apple_pos is not None:
            apple_speed = torch.linalg.norm(apple_pos - self._prev_apple_pos.to(device=device), dim=-1)
            _flag(apple_speed > self._thresholds.max_apple_speed_mps, "apple_speed_exceeded")

        self._prev_apple_pos = apple_pos.detach().clone()

        return unstable, reason_lists
