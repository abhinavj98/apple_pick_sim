"""Training configuration: frozen dataclasses + JSON (the repo's dataclass + argparse style).

``TrainConfig`` nests an :class:`EnvConfig` (which harvest env, how many worlds, reward /
envelope / DR knobs), a :class:`PPOConfig` (skrl ``PPO_RNN`` hyperparameters) and one
:class:`RecurrentNetConfig` each for the actor and the critic. A JSON file may give any
subset of fields; unknown keys are an error (typos must not silently fall back to
defaults).

Defaults are the starting point of ``docs/superpowers/plans/2026-09-23-rl-skrl-ppo-lstm-training.md``;
reward weights and PPO knobs are expected to be tuned once the infrastructure is proven.
"""

from __future__ import annotations

import dataclasses
import json
import typing
from pathlib import Path
from typing import Any, Literal

from apple_pick_gym.rl.models import RecurrentNetConfig

_ALL2000 = "apple_pick_gym/world_sets/snapshots/harvest_worlds_v2_all2000"


@dataclasses.dataclass(frozen=True)
class EnvConfig:
    """Which harvest env to build and its reward / envelope / DR knobs."""

    kind: Literal["sim", "surrogate"] = "sim"
    # sim: world set + (optional) settled snapshot for exactly those worlds, in order
    world_set: str | None = f"{_ALL2000}/shard_00.jsonl"
    snapshot: str | None = f"{_ALL2000}/shard_00_snapshot.npz"
    num_envs: int = 2000
    max_episode_steps: int = 500
    device: str = "auto"  # "auto" -> cuda:0 if available, else cpu
    # detach envelope (spur-stem): (F/f_max)^2 + (tau/tau_max)^2 >= 1
    f_max_n: float = 20.0
    tau_max_nm: float = 0.05
    success_streak_steps: int = 3
    safety_force_cap_n: float = 40.0
    safety_torque_cap_nm: float = 10.0
    # reward weights
    w_progress: float = 1.0
    w_pullout: float = 0.5
    w_collateral: float = 0.1
    success_bonus: float = 10.0
    failure_penalty: float = -20.0
    # VIC target leash: bounds the commanded wrench to K_max * offset (200 N/m * 0.15 m = 30 N)
    max_target_pos_offset_m: float | None = 0.15
    max_target_rot_offset_rad: float | None = 0.5
    # F/T sensor DR (bias / noise / drift, FtSensorConfig.rl_training)
    sensor_dr: bool = True
    # surrogate only: fraction of envs flagged invalid (frozen from reset)
    surrogate_invalid_fraction: float = 0.0


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    """skrl ``PPO_RNN`` knobs (names follow ``skrl.agents.torch.ppo.PPO_CFG``)."""

    rollouts: int = 64
    learning_epochs: int = 5
    mini_batches: int = 8
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    kl_adaptive_lr_threshold: float | None = 0.01  # KLAdaptiveLR target KL; None = fixed lr
    ratio_clip: float = 0.2
    value_clip: float = 0.2
    grad_norm_clip: float = 1.0
    entropy_loss_scale: float = 0.0
    value_loss_scale: float = 1.0
    time_limit_bootstrap: bool = False  # the time limit is observed -> terminated, no bootstrap
    running_standard_scaler: bool = True  # observations, states and values


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    env: EnvConfig = dataclasses.field(default_factory=EnvConfig)
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    actor: RecurrentNetConfig = dataclasses.field(default_factory=RecurrentNetConfig)
    critic: RecurrentNetConfig = dataclasses.field(default_factory=RecurrentNetConfig)
    timesteps: int = 20_000  # vectorized env steps (x num_envs = samples)
    seed: int = 0
    checkpoint_every_updates: int = 10
    run_dir: str = "runs/vic_harvest"
    wandb: bool = False
    wandb_project: str = "apple_pick_vic_harvest"

    def validate(self) -> None:
        seq = self.actor.sequence_length
        if self.critic.sequence_length != seq:
            raise ValueError(
                f"actor and critic sequence_length differ ({seq} vs {self.critic.sequence_length}); "
                "skrl PPO_RNN samples one sequence length for both"
            )
        if self.ppo.rollouts % seq:
            raise ValueError(f"ppo.rollouts ({self.ppo.rollouts}) must be a multiple of sequence_length ({seq})")
        n_seq = self.env.num_envs * self.ppo.rollouts // seq
        if n_seq % self.ppo.mini_batches:
            raise ValueError(
                f"{n_seq} stored sequences (num_envs * rollouts / sequence_length) do not split into "
                f"mini_batches={self.ppo.mini_batches} whole-sequence minibatches"
            )
        if self.env.num_envs < 2:
            raise ValueError("num_envs must be >= 2 (vectorized training)")

    # ------------------------------------------------------------------ JSON
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save_json(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        return _build(cls, data, "")

    @classmethod
    def load_json(cls, path: Path | str) -> TrainConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))


def _build(cls: type, data: dict[str, Any], where: str) -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown config keys in {where or 'root'}: {sorted(unknown)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value, hint = data[f.name], hints[f.name]
        if dataclasses.is_dataclass(hint):
            value = _build(hint, value, f"{where}{f.name}.")
        elif isinstance(value, list):
            value = tuple(value)
        kwargs[f.name] = value
    return cls(**kwargs)
