"""skrl 2.1 ``Wrapper`` over a harvest env (``ApplePickVicHarvestEnv`` or the surrogate).

Contract with skrl's ``PPO_RNN`` / trainer (read from the installed 2.1 source):

- **Spaces.** ``observation_space`` is the flat 40-D actor vector
  (``harvest_obs.flatten_actor_obs``), ``state_space`` the privileged critic vector
  (``critic_state.critic_state_layout``), ``action_space`` the box ``[-1, 1]^13`` that
  :class:`HarvestActionScaler` maps to env units.
- **Auto-reset is the wrapper's job.** skrl's trainer never calls ``reset()`` for
  ``num_envs > 1`` and calls ``state()`` right after ``step()``. Harvest envs truncate the
  whole batch on the same step (whole-batch reset only), so on that step the wrapper
  resets the env and returns the *new* episode's first observation, and ``state()`` pairs
  with it.
- **Time-limit semantics.** skrl's GAE only cuts at ``terminated``. The episode clock is
  observed (``step_frac`` in the actor and critic), so the time limit is part of the MDP:
  the time-limit step is reported ``terminated = truncated = True`` and the run uses
  ``time_limit_bootstrap=False``. Without this the last step would bootstrap from the next
  episode's reset value.
- **terminated** otherwise comes from the env and fires once per env, on the freeze edge.
- **Non-finite guard.** An env row whose observation, critic state or reward is not
  finite (a world that blew up) is zeroed and gets reward 0, and the count is logged as
  ``Step / nonfinite envs`` -- one bad world must not turn the whole PPO batch into NaN.
  A persistently non-zero count is a simulator problem to investigate, not to train on.
- **Logging.** Per-episode summaries (success / safety rate over valid envs, return,
  peak loads, reward-term sums, impedance usage) are emitted once per episode, and a few
  per-step signals every step, as scalar tensors in ``info["log"]`` (point the trainer's
  ``environment_info`` at ``"log"``).
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
import torch

from skrl.envs.wrappers.torch.base import Wrapper

from apple_pick_gym.batched_envs.harvest_obs import actor_obs_layout, flatten_actor_obs
from apple_pick_gym.batched_envs.harvest_privileged import PLANT_GEOMETRY_FIELDS
from apple_pick_gym.rl.action_scaling import HarvestActionScaler
from apple_pick_gym.rl.critic_state import build_critic_state, critic_state_layout


def _box(width: int) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(width,), dtype=np.float32)


class _EpisodeStats:
    """Per-env running episode statistics, summarized over valid envs at the time limit."""

    def __init__(self, num_envs: int, device: torch.device) -> None:
        self.n, self.device = int(num_envs), device
        self.reset(torch.zeros(self.n, dtype=torch.bool, device=device))

    def reset(self, invalid: torch.Tensor) -> None:
        z = lambda: torch.zeros(self.n, device=self.device)
        self.invalid = invalid.clone()
        self.ret, self.steps = z(), 0
        self.peak_idx, self.peak_force, self.peak_coll, self.peak_wrist = z(), z(), z(), z()
        self.terms = {k: z() for k in ("progress", "pullout", "collateral", "slack", "terminal")}
        self.success = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.safety = torch.zeros_like(self.success)
        self.steps_to_success = torch.full((self.n,), float("nan"), device=self.device)
        self.k_lin, self.k_ang, self.zeta, self.live_steps = z(), z(), z(), z()
        self.peak_speed, self.sum_speed = z(), z()
        self.peak_junction: dict[str, torch.Tensor] = {}

    def update(
        self,
        reward: torch.Tensor,
        info: dict[str, Any],
        terminated: torch.Tensor,
        env_action: torch.Tensor,
        tcp_speed: torch.Tensor | None = None,
    ) -> None:
        self.steps += 1
        ep, rt = info["episode"], info["reward_terms"]
        live = ~ep["frozen"] | terminated  # not frozen before this step
        self.ret += reward.reshape(-1)
        m = lambda cur, new: torch.where(live, torch.maximum(cur, torch.nan_to_num(new, nan=0.0, posinf=0.0)), cur)
        self.peak_idx = m(self.peak_idx, info["detach_index"])
        self.peak_force = m(self.peak_force, torch.linalg.norm(info["target_junction_wrench"][:, :3], dim=-1))
        self.peak_coll = m(self.peak_coll, rt["raw"]["collateral"])
        self.peak_wrist = m(self.peak_wrist, torch.linalg.norm(info["ft_wrist"][:, :3], dim=-1))
        for k in ("progress", "pullout", "collateral", "slack"):
            self.terms[k] += torch.where(live, rt["weighted"][k], torch.zeros_like(rt["weighted"][k]))
        self.terms["terminal"] += torch.where(live, rt["terminal"], torch.zeros_like(rt["terminal"]))
        safe = ep["safety_junction"] | ep["safety_wrist"]
        won = terminated & ep["success_achieved"] & ~safe
        self.steps_to_success = torch.where(won & ~self.success, torch.full_like(self.steps_to_success, self.steps), self.steps_to_success)
        self.success |= won
        self.safety |= terminated & safe
        lf = live.float()
        self.k_lin += lf * env_action[:, 6:9].mean(-1)
        self.k_ang += lf * env_action[:, 9:12].mean(-1)
        self.zeta += lf * env_action[:, 12]
        self.live_steps += lf
        for name, wr in info["woody_part_force"].items():
            cur = self.peak_junction.get(name, torch.zeros(self.n, device=self.device))
            self.peak_junction[name] = m(cur, torch.linalg.norm(wr[:, :3].to(cur), dim=-1))
        if tcp_speed is not None:
            sp = torch.nan_to_num(tcp_speed.reshape(-1).to(self.peak_speed), nan=0.0, posinf=0.0)
            self.peak_speed = m(self.peak_speed, sp)
            self.sum_speed += lf * sp

    def summary(self) -> dict[str, torch.Tensor]:
        valid = ~self.invalid
        s = lambda x: torch.tensor(0.0, device=self.device) if not bool(valid.any()) else x[valid].float().mean()
        per_live = lambda x: x / self.live_steps.clamp_min(1.0)
        won = self.success & valid
        stt = self.steps_to_success[won].mean() if bool(won.any()) else torch.tensor(float(self.steps), device=self.device)
        # [D6] load on the tree per successful pick; NaN when no valid env succeeded
        coll_won = self.peak_coll[won].mean() if bool(won.any()) else torch.tensor(float("nan"), device=self.device)
        return {
            "Episode / success rate": s(self.success),
            "Episode / safety rate": s(self.safety),
            "Episode / invalid fraction": self.invalid.float().mean(),
            "Episode / return (mean)": s(self.ret),
            "Episode / peak detach index (mean)": s(self.peak_idx),
            "Episode / peak target force N (mean)": s(self.peak_force),
            "Episode / peak collateral N (mean)": s(self.peak_coll),
            "Episode / peak collateral N, successful (mean)": coll_won,
            "Episode / peak wrist force N (mean)": s(self.peak_wrist),
            "Episode / steps to success (mean)": stt,
            "Episode / reward progress (sum)": s(self.terms["progress"]),
            "Episode / reward pullout (sum)": s(self.terms["pullout"]),
            "Episode / reward collateral (sum)": s(self.terms["collateral"]),
            "Episode / reward slack (sum)": s(self.terms["slack"]),
            "Episode / reward terminal (sum)": s(self.terms["terminal"]),
            "Episode / K_lin used (mean)": s(per_live(self.k_lin)),
            "Episode / K_ang used (mean)": s(per_live(self.k_ang)),
            "Episode / zeta used (mean)": s(per_live(self.zeta)),
            "Episode / peak TCP speed m/s (mean)": s(self.peak_speed),
            "Episode / mean TCP speed m/s (mean)": s(per_live(self.sum_speed)),
            **{f"Episode / peak force {k} N (mean)": s(v) for k, v in self.peak_junction.items()},
        }


class HarvestSkrlWrapper(Wrapper):
    """Flat obs / privileged state / normalized actions / auto-reset for skrl ``PPO_RNN``."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self._scaler = HarvestActionScaler(env.action_bounds)
        self._junctions = list(env.junction_names)
        self._layout = critic_state_layout(self._junctions)
        self._obs_space = _box(actor_obs_layout().total_width)
        self._state_space = _box(self._layout.total_width)
        self._action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)
        self._stats = _EpisodeStats(env.num_envs, self.device)
        self._obs: torch.Tensor | None = None
        self._state: torch.Tensor | None = None
        self._privileged: dict[str, torch.Tensor] | None = None
        self._geometry: dict[str, torch.Tensor] | None = None

    # skrl reads these as properties
    @property
    def observation_space(self) -> gymnasium.Space:
        return self._obs_space

    @property
    def state_space(self) -> gymnasium.Space:
        return self._state_space

    @property
    def action_space(self) -> gymnasium.Space:
        return self._action_space

    @property
    def num_envs(self) -> int:
        return int(self._env.num_envs)

    @property
    def num_agents(self) -> int:
        return 1

    @property
    def critic_layout(self):
        return self._layout

    @property
    def action_scaler(self) -> HarvestActionScaler:
        return self._scaler

    def _refresh(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        self._obs = flatten_actor_obs(obs).to(self.device, torch.float32)
        self._state = build_critic_state(
            obs,
            info,
            privileged=self._privileged,
            geometry=self._geometry,
            junction_names=self._junctions,
            success_streak_steps=int(self._env.episode_config.success_streak_steps),
            invalid=self._env.invalid_env_mask,
        ).to(self.device, torch.float32)
        self._nonfinite = ~(torch.isfinite(self._obs).all(-1) & torch.isfinite(self._state).all(-1))
        self._obs = torch.nan_to_num(self._obs, nan=0.0, posinf=0.0, neginf=0.0)
        self._state = torch.nan_to_num(self._state, nan=0.0, posinf=0.0, neginf=0.0)

    def _on_reset(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        # Build-time DR (plant, support, geometry) is fixed; arm joint DR resamples each reset.
        self._privileged = self._env.privileged_fields()
        if self._geometry is None:
            geo = self._env.plant_geometry()
            self._geometry = {k: geo[k] for k, _ in PLANT_GEOMETRY_FIELDS}
        self._stats.reset(self._env.invalid_env_mask)
        self._refresh(obs, info)

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        obs, info = self._env.reset()
        self._on_reset(obs, info)
        return self._obs, info

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        env_action = self._scaler.to_env(actions.to(self.device, torch.float32))
        obs, reward, terminated, truncated, info = self._env.step(env_action)
        terminated = terminated.reshape(-1, 1).bool()
        truncated = truncated.reshape(-1, 1).bool()
        reward = reward.reshape(-1, 1).to(torch.float32)
        bad_reward = ~torch.isfinite(reward).all(-1)
        reward = torch.where(bad_reward.unsqueeze(-1), torch.zeros_like(reward), reward)
        tcp_speed = torch.linalg.norm(obs["tcp_velocity"][:, :3].to(self.device, torch.float32), dim=-1)
        self._stats.update(reward, info, terminated.flatten(), env_action, tcp_speed=tcp_speed)
        log = {
            "Step / detach index (mean)": torch.nan_to_num(info["detach_index"].float()).mean(),
            "Step / frozen fraction": info["episode"]["frozen"].float().mean(),
        }
        if bool(truncated.any()):
            if not bool(truncated.all()):
                raise RuntimeError("harvest envs truncate the whole batch together; got a partial truncation")
            terminated = terminated | truncated
            log.update(self._stats.summary())
            info = dict(info)
            obs, reset_info = self._env.reset()
            self._on_reset(obs, reset_info)
        else:
            self._refresh(obs, info)
            bad_reward = bad_reward | self._nonfinite
            reward = torch.where(bad_reward.unsqueeze(-1), torch.zeros_like(reward), reward)
        log["Step / nonfinite envs"] = bad_reward.float().sum()
        info["log"] = log
        return self._obs, reward, terminated, truncated, info

    def state(self) -> torch.Tensor:
        return self._state

    def render(self, *args, **kwargs) -> Any:
        return None

    def close(self) -> None:
        self._env.close()
