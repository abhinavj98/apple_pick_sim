"""Debug/training metrics for :class:`ApplePickVicHarvestEnv` and an optional wandb sink.

``HarvestMetricsLogger`` turns each ``env.step()`` result into a flat ``dict[str, float]``:
batch mean/min/max for every signal, plus full per-env traces for the first
``trace_env_ids``. It is pure torch (no wandb import), so the random-action example now and
the RL trainer later share one implementation. ``WandbSink`` is the thin, lazily-imported
wandb wrapper.

Reads only what the env already exposes: ``obs`` (policy view), ``info`` (privileged
forces, ``reward_terms``, ``episode``, ``target_pose``) and the action that was sent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from apple_pick_gym.batched_envs.harvest_action import (
    HarvestActionBounds,
    derive_critical_damping,
    split_harvest_action,
)

_AXES = ("x", "y", "z")


def _norm3(wrench: torch.Tensor, start: int = 0) -> torch.Tensor:
    return torch.linalg.norm(wrench[:, start : start + 3], dim=-1)


def _add_stats(out: dict[str, float], key: str, x: torch.Tensor) -> None:
    """Batch mean/min/max of a per-env ``(N,)`` tensor under ``key``."""
    x = x.detach().float().reshape(-1)
    out[f"{key}/mean"] = float(x.mean())
    out[f"{key}/min"] = float(x.min())
    out[f"{key}/max"] = float(x.max())


def _add_traces(
    out: dict[str, float], key: str, x: torch.Tensor, trace_env_ids: Sequence[int]
) -> None:
    """Per-env values of ``x`` (``(N,)``) for the traced envs under ``env{i}/key``."""
    x = x.detach().float().reshape(-1)
    for i in trace_env_ids:
        out[f"env{i}/{key}"] = float(x[i])


def _quat_angle(q_a_wxyz: torch.Tensor, q_b_wxyz: torch.Tensor) -> torch.Tensor:
    """Angle (rad) of the relative rotation between two unit quaternions, ``(N,)``."""
    dot = torch.abs(torch.sum(q_a_wxyz * q_b_wxyz, dim=-1)).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


class HarvestMetricsLogger:
    """Stateful per-step metric builder; call :meth:`on_reset` after each ``env.reset()``."""

    def __init__(
        self,
        *,
        num_envs: int,
        action_bounds: HarvestActionBounds,
        trace_env_ids: Sequence[int] | None = None,
        trace_envs: int = 4,
    ) -> None:
        self.num_envs = int(num_envs)
        self.action_bounds = action_bounds
        ids = list(trace_env_ids) if trace_env_ids is not None else list(range(min(trace_envs, num_envs)))
        if any(not (0 <= i < self.num_envs) for i in ids):
            raise ValueError(f"trace_env_ids {ids} out of range for num_envs={self.num_envs}")
        self.trace_env_ids = ids
        self._apple_ref: torch.Tensor | None = None
        self._ret = torch.zeros(self.num_envs)
        self._len = torch.zeros(self.num_envs)
        self._outcome = ["running"] * self.num_envs
        self._steps_to_success = [float("nan")] * self.num_envs

    def on_reset(self, info: dict[str, Any]) -> dict[str, float]:
        """Record the reset apple position; return the previous episode's summary metrics."""
        summary = self._episode_summary()
        self._apple_ref = info["apple_pos"].detach().float().cpu().clone()
        self._ret.zero_()
        self._len.zero_()
        self._outcome = ["running"] * self.num_envs
        self._steps_to_success = [float("nan")] * self.num_envs
        return summary

    def flush_summary(self) -> dict[str, float]:
        """Summary of the in-progress episode (unfinished envs count as truncated)."""
        return self._episode_summary()

    def _update_episode(self, info: dict[str, Any], truncated: torch.Tensor) -> None:
        total = info["reward_terms"]["total"].detach().float().cpu()
        ep = info["episode"]
        succ = ep["success_achieved"].detach().cpu()
        safe = (ep["safety_junction"] | ep["safety_wrist"]).detach().cpu()
        trunc = bool(truncated.any())
        running = torch.tensor([o == "running" for o in self._outcome])
        self._ret += total * running
        self._len += running.float()
        for i in range(self.num_envs):
            if self._outcome[i] != "running":
                continue
            if bool(safe[i]):
                self._outcome[i] = "safety"
            elif bool(succ[i]):
                self._outcome[i] = "success"
                self._steps_to_success[i] = float(self._len[i])
            elif trunc:
                self._outcome[i] = "truncated"

    def _episode_summary(self) -> dict[str, float]:
        if not bool((self._len > 0).any()):
            return {}
        n = float(self.num_envs)
        outcomes = ["truncated" if o == "running" else o for o in self._outcome]
        out = {
            "episode/return_mean": float(self._ret.mean()),
            "episode/return_min": float(self._ret.min()),
            "episode/return_max": float(self._ret.max()),
            "episode/len_mean": float(self._len.mean()),
            "episode/success_rate": outcomes.count("success") / n,
            "episode/safety_rate": outcomes.count("safety") / n,
            "episode/truncated_rate": outcomes.count("truncated") / n,
        }
        done = [s for s in self._steps_to_success if s == s]
        if done:
            out["episode/steps_to_success_mean"] = sum(done) / len(done)
        return out

    def step_metrics(
        self,
        obs: dict[str, Any],
        info: dict[str, Any],
        actions: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> dict[str, float]:
        """Metrics for one ``env.step``. ``actions`` is the raw ``(N,13)`` action sent."""
        out: dict[str, float] = {}
        tr = self.trace_env_ids
        dev = obs["ft_wrist"].device

        # -- reward decomposition -------------------------------------------------------
        rt = info["reward_terms"]
        for name in ("total", "dense", "terminal"):
            _add_stats(out, f"reward/{name}", rt[name])
            _add_traces(out, f"reward/{name}", rt[name], tr)
        for name, x in rt["raw"].items():
            _add_stats(out, f"reward/raw/{name}", x)
            _add_traces(out, f"reward/raw/{name}", x, tr)
        for name, x in rt["weighted"].items():
            _add_stats(out, f"reward/weighted/{name}", x)
            _add_traces(out, f"reward/weighted/{name}", x, tr)

        # -- forces: every junction + target -------------------------------------------
        for name, wrench in sorted(info["woody_part_force"].items()):
            f, t = _norm3(wrench), _norm3(wrench, 3)
            _add_stats(out, f"force/junction/{name}/F", f)
            _add_stats(out, f"force/junction/{name}/T", t)
            _add_traces(out, f"force/junction/{name}/F", f, tr)
            _add_traces(out, f"force/junction/{name}/T", t, tr)
        tgt = info["target_junction_force"]
        _add_stats(out, "force/target/F", _norm3(tgt))
        _add_stats(out, "force/target/T", _norm3(tgt, 3))
        _add_traces(out, "force/target/F", _norm3(tgt), tr)

        # -- wrist: sim raw vs policy-observed -------------------------------------------
        raw, seen = info["ft_wrist"], obs["ft_wrist"]
        _add_stats(out, "wrist/raw/F", _norm3(raw))
        _add_stats(out, "wrist/raw/T", _norm3(raw, 3))
        _add_stats(out, "wrist/obs/F", _norm3(seen))
        _add_stats(out, "wrist/obs/T", _norm3(seen, 3))
        _add_stats(out, "wrist/obs_minus_raw/F", torch.linalg.norm((seen - raw)[:, :3], dim=-1))
        for i in tr:
            for k, ax in enumerate(_AXES):
                out[f"env{i}/wrist/raw/F{ax}"] = float(raw[i, k])
                out[f"env{i}/wrist/obs/F{ax}"] = float(seen[i, k])
        _add_traces(out, "wrist/raw/F", _norm3(raw), tr)
        _add_traces(out, "wrist/obs/F", _norm3(seen), tr)

        # -- action decode ---------------------------------------------------------------
        act = actions.detach().to(dev, torch.float32)
        split = split_harvest_action(act, self.action_bounds)
        _add_stats(out, "action/dp_norm", _norm3(split.delta))
        _add_stats(out, "action/drot_norm", _norm3(split.delta, 3))
        _add_stats(out, "action/zeta", split.zeta.squeeze(-1))
        _add_stats(out, "action/K_lin", split.linear_k.mean(dim=-1))
        _add_stats(out, "action/K_ang", split.angular_k.mean(dim=-1))
        d_lin = derive_critical_damping(split.linear_k, split.zeta)
        d_ang = derive_critical_damping(split.angular_k, split.zeta)
        _add_stats(out, "action/D_lin", d_lin.mean(dim=-1))
        _add_stats(out, "action/D_ang", d_ang.mean(dim=-1))
        for k, ax in enumerate(_AXES):
            _add_stats(out, f"action/K_lin_{ax}", split.linear_k[:, k])
            _add_stats(out, f"action/K_ang_{ax}", split.angular_k[:, k])
        _add_traces(out, "action/zeta", split.zeta.squeeze(-1), tr)
        _add_traces(out, "action/K_lin", split.linear_k.mean(dim=-1), tr)

        # -- pose tracking + motion ------------------------------------------------------
        target = info["target_pose"].to(dev)
        pos_err = torch.linalg.norm(obs["tcp_pos"] - target[:, :3], dim=-1)
        tcp_wxyz = obs["tcp_quat"][:, [3, 0, 1, 2]]  # obs is xyzw, target is wxyz
        rot_err = _quat_angle(tcp_wxyz, target[:, 3:7])
        _add_stats(out, "tracking/pos_err_m", pos_err)
        _add_stats(out, "tracking/rot_err_rad", rot_err)
        _add_traces(out, "tracking/pos_err_m", pos_err, tr)
        tcp_speed = torch.linalg.norm(obs["tcp_velocity"][:, :3], dim=-1)
        _add_stats(out, "state/tcp_speed", tcp_speed)
        if self._apple_ref is not None:
            apple_disp = torch.linalg.norm(
                info["apple_pos"].detach().float().cpu() - self._apple_ref, dim=-1
            )
            _add_stats(out, "state/apple_disp_m", apple_disp)
            _add_traces(out, "state/apple_disp_m", apple_disp, tr)

        # -- episode / termination -------------------------------------------------------
        ep = info["episode"]
        n = float(self.num_envs)
        out["episode/frozen_frac"] = float(ep["frozen"].sum()) / n
        out["episode/success_frac"] = float(ep["success_achieved"].sum()) / n
        out["episode/safety_junction_frac"] = float(ep["safety_junction"].sum()) / n
        out["episode/safety_wrist_frac"] = float(ep["safety_wrist"].sum()) / n
        out["episode/success_streak_max"] = float(ep["success_streak"].max())
        out["episode/terminated_frac"] = float(terminated.sum()) / n
        _add_traces(out, "episode/success_streak", ep["success_streak"], tr)

        self._update_episode(info, truncated)
        return out


def dr_table_rows(env: Any) -> list[dict[str, float]]:
    """One row per env of the domain-randomized quantities (for a wandb table)."""
    rows: list[dict[str, float]] = [{"env": i} for i in range(env.num_envs)]
    sup = getattr(env, "_last_support_dr_sample", None)
    if sup is not None:
        for i, row in enumerate(rows):
            row["support_kp"] = float(sup.kp[i])
            row["support_roll_kp"] = float(sup.roll_kp[i])
            row["support_zeta"] = float(sup.zeta[i])
    for i, params in enumerate(env._sim.per_env_params):
        rows[i]["apple_radius"] = float(params.apple_radius)
        rows[i]["apple_density"] = float(params.apple_density)
    return rows


class WandbSink:
    """Thin wandb wrapper; ``wandb`` is imported lazily so the module works without it."""

    def __init__(
        self,
        *,
        project: str,
        run_name: str | None = None,
        mode: str = "online",
        config: dict[str, Any] | None = None,
    ) -> None:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging requested but wandb is not installed "
                "(uv sync --extra gym, or pass --no-wandb)."
            ) from e
        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name, mode=mode, config=config or {})

    def log(self, metrics: dict[str, float], step: int) -> None:
        if metrics:
            self._wandb.log(metrics, step=step)

    def log_table(self, name: str, rows: list[dict[str, float]], step: int = 0) -> None:
        if not rows:
            return
        columns = list(rows[0].keys())
        table = self._wandb.Table(columns=columns, data=[[r.get(c) for c in columns] for r in rows])
        self._wandb.log({name: table}, step=step)

    def log_video(self, key: str, path: Any, step: int, fps: float) -> None:
        self._wandb.log({key: self._wandb.Video(str(path), fps=int(round(fps)), format="mp4")}, step=step)

    def finish(self) -> None:
        self._run.finish()
