"""Build the harvest env + skrl ``PPO_RNN`` agent from a :class:`TrainConfig` and train.

The loop is skrl's ``SequentialTrainer`` loop (act -> step -> record -> post-interaction),
written out so it can (a) resume from a global timestep, (b) write this repo's
checkpoints with their compatibility sidecar, and (c) keep a JSON-lines history
(``metrics.jsonl``) next to skrl's TensorBoard events -- one ``"kind": "update"`` row per
PPO update (losses, std, lr, timing, and the tracked signals averaged over that rollout)
and one ``"kind": "episode"`` row per finished episode (``Episode / ...`` stats from
:class:`HarvestSkrlWrapper`).
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from apple_pick_gym.rl.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from apple_pick_gym.rl.config import EnvConfig, TrainConfig
from apple_pick_gym.rl.models import LstmGaussianActor, LstmValueCritic
from apple_pick_gym.rl.skrl_wrapper import HarvestSkrlWrapper


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def build_env(env_cfg: EnvConfig, *, seed: int = 0):
    """The raw harvest env (``ApplePickVicHarvestEnv`` or :class:`SurrogateHarvestEnv`)."""
    from apple_pick_gym.batched_envs.harvest_action import HarvestActionBounds
    from apple_pick_gym.batched_envs.harvest_detach import DetachEnvelopeConfig
    from apple_pick_gym.batched_envs.harvest_episode import EpisodeConfig
    from apple_pick_gym.batched_envs.harvest_reward import HarvestRewardConfig
    from apple_pick_gym.batched_envs.sensor_realism import FtSensorConfig

    device = resolve_device(env_cfg.device)
    bounds = HarvestActionBounds(
        max_target_pos_offset_m=env_cfg.max_target_pos_offset_m,
        max_target_rot_offset_rad=env_cfg.max_target_rot_offset_rad,
    )
    reward = HarvestRewardConfig(
        detach=DetachEnvelopeConfig(f_max_n=env_cfg.f_max_n, tau_max_nm=env_cfg.tau_max_nm),
        w_progress=env_cfg.w_progress,
        w_pullout=env_cfg.w_pullout,
        w_collateral=env_cfg.w_collateral,
        success_bonus=env_cfg.success_bonus,
        failure_penalty=env_cfg.failure_penalty,
    )
    episode = EpisodeConfig(
        success_streak_steps=env_cfg.success_streak_steps,
        safety_force_cap_n=env_cfg.safety_force_cap_n,
        safety_torque_cap_nm=env_cfg.safety_torque_cap_nm,
    )
    sensor = FtSensorConfig.rl_training() if env_cfg.sensor_dr else FtSensorConfig()
    common = dict(
        max_episode_steps=env_cfg.max_episode_steps,
        action_bounds=bounds,
        reward_config=reward,
        episode_config=episode,
        ft_sensor_config=sensor,
    )
    if env_cfg.kind == "surrogate":
        from apple_pick_gym.rl.surrogate_env import SurrogateHarvestEnv

        return SurrogateHarvestEnv(
            num_envs=env_cfg.num_envs,
            device=device,
            invalid_fraction=env_cfg.surrogate_invalid_fraction,
            seed=seed,
            **common,
        )
    if env_cfg.kind != "sim":
        raise ValueError(f"unknown env kind {env_cfg.kind!r}")
    from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import ApplePickVicHarvestEnv
    from apple_pick_gym.batched_envs.world_set import load_world_set, world_specs_to_env_kwargs

    if env_cfg.world_set is None:
        raise ValueError("env.kind='sim' needs env.world_set")
    specs = load_world_set(Path(env_cfg.world_set))
    if len(specs) < env_cfg.num_envs:
        raise ValueError(f"world set {env_cfg.world_set} has {len(specs)} worlds < num_envs={env_cfg.num_envs}")
    if env_cfg.snapshot is not None and len(specs) != env_cfg.num_envs:
        raise ValueError(
            f"a snapshot is tied to its exact worlds: num_envs must equal the world set size ({len(specs)}); "
            "pass snapshot=null to take a prefix and hold-settle instead"
        )
    return ApplePickVicHarvestEnv(
        device=device,
        dr_seed=seed,
        episode_snapshot_path=env_cfg.snapshot,
        **common,
        **world_specs_to_env_kwargs(specs[: env_cfg.num_envs]),
    )


def _ppo_rnn_class():
    from skrl.agents.torch.ppo import PPO_RNN

    class HarvestPPO_RNN(PPO_RNN):
        """``PPO_RNN`` that also hands every TensorBoard write to a history callback."""

        history_callback = None

        def write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
            if self.history_callback is not None:
                row = {}
                for k, v in self.tracking_data.items():
                    if not len(v):
                        continue
                    agg = np.min if k.endswith("(min)") else np.max if k.endswith("(max)") else np.mean
                    row[k] = float(agg(v))
                self.history_callback(timestep, row)
            super().write_tracking_data(timestep=timestep, timesteps=timesteps)

    return HarvestPPO_RNN


def build_agent(wrapper: HarvestSkrlWrapper, cfg: TrainConfig, *, run_dir: Path, wandb_run_id: str | None = None):
    from skrl.memories.torch import RandomMemory
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.resources.schedulers.torch import KLAdaptiveLR

    device = wrapper.device
    n = wrapper.num_envs
    spaces = dict(
        observation_space=wrapper.observation_space,
        state_space=wrapper.state_space,
        action_space=wrapper.action_space,
        device=device,
    )
    models = {
        "policy": LstmGaussianActor(**spaces, num_envs=n, cfg=cfg.actor),
        "value": LstmValueCritic(**spaces, num_envs=n, cfg=cfg.critic),
    }
    memory = RandomMemory(memory_size=cfg.ppo.rollouts, num_envs=n, device=device)
    p = cfg.ppo
    agent_cfg: dict[str, Any] = dict(
        rollouts=p.rollouts,
        learning_epochs=p.learning_epochs,
        mini_batches=p.mini_batches,
        discount_factor=p.discount_factor,
        gae_lambda=p.gae_lambda,
        learning_rate=p.learning_rate,
        ratio_clip=p.ratio_clip,
        value_clip=p.value_clip,
        grad_norm_clip=p.grad_norm_clip,
        entropy_loss_scale=p.entropy_loss_scale,
        value_loss_scale=p.value_loss_scale,
        time_limit_bootstrap=p.time_limit_bootstrap,
        experiment=dict(
            directory=str(run_dir.parent),
            experiment_name=run_dir.name,
            write_interval=p.rollouts,  # one TensorBoard / history row per PPO update
            checkpoint_interval=0,  # this repo's checkpoints (with meta.json) instead
            wandb=cfg.wandb,
            wandb_kwargs=dict(project=cfg.wandb_project, id=wandb_run_id, resume="allow", dir=str(run_dir)),
        ),
    )
    if p.kl_adaptive_lr_threshold is not None:
        agent_cfg["learning_rate_scheduler"] = KLAdaptiveLR
        agent_cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": p.kl_adaptive_lr_threshold}
    if p.running_standard_scaler:
        agent_cfg["observation_preprocessor"] = RunningStandardScaler
        agent_cfg["observation_preprocessor_kwargs"] = {"size": wrapper.observation_space, "device": device}
        agent_cfg["state_preprocessor"] = RunningStandardScaler
        agent_cfg["state_preprocessor_kwargs"] = {"size": wrapper.state_space, "device": device}
        agent_cfg["value_preprocessor"] = RunningStandardScaler
        agent_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    return _ppo_rnn_class()(
        models=models,
        memory=memory,
        observation_space=wrapper.observation_space,
        state_space=wrapper.state_space,
        action_space=wrapper.action_space,
        device=device,
        cfg=agent_cfg,
    )


def build_training(cfg: TrainConfig, *, wandb_run_id: str | None = None):
    """``(wrapper, agent)`` for ``cfg`` (agent not yet ``init``-ed)."""
    from skrl.utils import set_seed

    cfg.validate()
    set_seed(cfg.seed)
    wrapper = HarvestSkrlWrapper(build_env(cfg.env, seed=cfg.seed))
    agent = build_agent(wrapper, cfg, run_dir=Path(cfg.run_dir), wandb_run_id=wandb_run_id)
    return wrapper, agent


@dataclasses.dataclass
class TrainResult:
    start_timestep: int
    timestep: int
    updates: int
    run_dir: Path
    last_checkpoint: Path | None
    episodes: list[dict[str, float]]


def run_training(cfg: TrainConfig, *, resume: str | None = None, max_updates: int | None = None) -> TrainResult:
    """Train until ``cfg.timesteps`` (or ``max_updates`` more updates). ``resume``: ``"latest"`` or a checkpoint dir."""
    from skrl.trainers.torch import SequentialTrainerCfg

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = None
    if resume == "latest":
        ckpt_path = latest_checkpoint(run_dir)
        if ckpt_path is None:
            raise FileNotFoundError(f"--resume latest: no checkpoint under {run_dir}/checkpoints")
    elif resume:
        ckpt_path = Path(resume)
    wandb_run_id = None
    if ckpt_path is not None:
        wandb_run_id = json.loads((ckpt_path / "meta.json").read_text()).get("wandb_run_id")
    if cfg.wandb and wandb_run_id is None:
        import wandb

        wandb_run_id = wandb.util.generate_id()

    wrapper, agent = build_training(cfg, wandb_run_id=wandb_run_id)
    cfg.save_json(run_dir / "config.json")
    metrics = (run_dir / "metrics.jsonl").open("a")
    episodes: list[dict[str, float]] = []

    def _history(timestep: int, row: dict[str, float]) -> None:
        if any(k.startswith("Loss /") for k in row):
            metrics.write(json.dumps({"kind": "update", "timestep": int(timestep), **row}) + "\n")
            metrics.flush()

    agent.history_callback = _history
    agent.init(trainer_cfg=SequentialTrainerCfg(timesteps=cfg.timesteps))
    start, updates = 0, 0
    if ckpt_path is not None:
        meta = load_checkpoint(ckpt_path, agent, wrapper, cfg)
        start, updates = int(meta["timestep"]), int(meta["updates"])
    agent.enable_training_mode(True)
    agent.enable_models_training_mode(False)

    last_ckpt = ckpt_path
    obs, _ = wrapper.reset()
    states = wrapper.state()
    t_env = t_act = 0.0
    timestep = start
    stop_at = cfg.timesteps if max_updates is None else min(cfg.timesteps, start + max_updates * cfg.ppo.rollouts)
    try:
        for timestep in range(start, stop_at):
            agent.pre_interaction(timestep=timestep, timesteps=cfg.timesteps)
            with torch.no_grad():
                t0 = time.perf_counter()
                actions, _ = agent.act(obs, states, timestep=timestep, timesteps=cfg.timesteps)
                t1 = time.perf_counter()
                next_obs, rewards, terminated, truncated, infos = wrapper.step(actions)
                next_states = wrapper.state()
                t_act += t1 - t0
                t_env += time.perf_counter() - t1
                agent.record_transition(
                    observations=obs,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_obs,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=cfg.timesteps,
                )
                log = {k: float(v) for k, v in infos.get("log", {}).items()}
                for k, v in log.items():
                    agent.track_data(k, v)
                if "Episode / success rate" in log:
                    row = {"kind": "episode", "timestep": timestep + 1, **{k: v for k, v in log.items() if k.startswith("Episode /")}}
                    episodes.append(row)
                    metrics.write(json.dumps(row) + "\n")
                    metrics.flush()
            if (timestep + 1) % cfg.ppo.rollouts == 0:
                agent.track_data("Stats / env step time per rollout (s)", t_env)
                agent.track_data("Stats / policy act time per rollout (s)", t_act)
                t_env = t_act = 0.0
            agent.post_interaction(timestep=timestep, timesteps=cfg.timesteps)
            obs, states = next_obs, next_states
            if (timestep + 1) % cfg.ppo.rollouts == 0:
                updates += 1
                if updates % cfg.checkpoint_every_updates == 0 or timestep + 1 == stop_at:
                    last_ckpt = save_checkpoint(
                        run_dir / "checkpoints", agent, wrapper, cfg,
                        timestep=timestep + 1, updates=updates, wandb_run_id=wandb_run_id,
                    )
        timestep = stop_at
    finally:
        metrics.close()
        wrapper.close()
    return TrainResult(
        start_timestep=start, timestep=timestep, updates=updates, run_dir=run_dir, last_checkpoint=last_ckpt, episodes=episodes
    )
