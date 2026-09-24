"""Train the VIC harvest policy with skrl recurrent PPO.

Examples (from the repo root)::

    # real sim, all 2000 screened worlds from the committed settled snapshot (CUDA)
    uv run python -m apple_pick_gym.rl.train_vic_harvest --run-dir runs/vic_harvest/exp1 --wandb

    # resume the same run after a crash (fresh process; picks the newest checkpoint)
    uv run python -m apple_pick_gym.rl.train_vic_harvest --run-dir runs/vic_harvest/exp1 --resume latest

    # CPU infrastructure smoke on the analytic surrogate env (minutes, reward should rise)
    uv run python -m apple_pick_gym.rl.train_vic_harvest --config apple_pick_gym/rl/configs/surrogate_smoke.json

    # one rollout + one PPO update, then exit (wiring check)
    uv run python -m apple_pick_gym.rl.train_vic_harvest --env surrogate --dry-run

A ``--config`` JSON gives any subset of :class:`TrainConfig`; command-line flags override it.
Outputs under ``--run-dir``: ``config.json``, ``metrics.jsonl``, TensorBoard events and
``checkpoints/ckpt_<timestep>/{agent.pt, meta.json}``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from apple_pick_gym.rl.config import TrainConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="TrainConfig JSON (any subset of fields)")
    p.add_argument("--run-dir", help="output directory (config, metrics, TensorBoard, checkpoints)")
    p.add_argument("--resume", help="'latest' or a checkpoint directory")
    p.add_argument("--dry-run", action="store_true", help="one rollout + one PPO update, then exit")
    p.add_argument(
        "--max-updates", type=int,
        help="stop after this many more PPO updates, with a checkpoint (a wall-capped segment; continue with --resume latest)",
    )
    g = p.add_argument_group("environment")
    g.add_argument("--env", choices=("sim", "surrogate"), help="real ApplePickVicHarvestEnv or the analytic surrogate")
    g.add_argument("--world-set", help="sim: world-set JSONL")
    g.add_argument("--snapshot", help="sim: settled snapshot .npz for exactly that world set ('none' to hold-settle)")
    g.add_argument("--num-envs", type=int)
    g.add_argument("--max-episode-steps", type=int)
    g.add_argument("--device", help="'auto', 'cpu' or 'cuda:0'")
    g.add_argument(
        "--allow-cpu-sim",
        action="store_true",
        help="allow --env sim on CPU (wiring checks only: the batched arm does not move on MuJoCo-CPU)",
    )
    g = p.add_argument_group("training")
    g.add_argument("--timesteps", type=int, help="vectorized env steps to train for (x num_envs = samples)")
    g.add_argument("--seed", type=int)
    g.add_argument("--rollouts", type=int)
    g.add_argument("--mini-batches", type=int)
    g.add_argument("--learning-epochs", type=int)
    g.add_argument("--learning-rate", type=float)
    g.add_argument("--sequence-length", type=int, help="BPTT length (actor and critic)")
    g.add_argument("--hidden", type=int, help="LSTM and MLP width (actor and critic)")
    g.add_argument("--checkpoint-every-updates", type=int)
    w = g.add_mutually_exclusive_group()
    w.add_argument("--wandb", dest="wandb", action="store_true", default=None)
    w.add_argument("--no-wandb", dest="wandb", action="store_false")
    g.add_argument("--wandb-project")
    g.add_argument("--video-every", type=int, help="record one env every N episodes (0 = off; CUDA)")
    return p


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig.load_json(args.config) if args.config else TrainConfig()
    env = {
        "kind": args.env,
        "world_set": args.world_set,
        "snapshot": None if args.snapshot == "none" else args.snapshot,
        "num_envs": args.num_envs,
        "max_episode_steps": args.max_episode_steps,
        "device": args.device,
    }
    if args.snapshot is None:
        env.pop("snapshot")
    env = {k: v for k, v in env.items() if v is not None or k == "snapshot"}
    ppo = {k: v for k, v in {
        "rollouts": args.rollouts,
        "mini_batches": args.mini_batches,
        "learning_epochs": args.learning_epochs,
        "learning_rate": args.learning_rate,
    }.items() if v is not None}
    net = {}
    if args.sequence_length is not None:
        net["sequence_length"] = args.sequence_length
    if args.hidden is not None:
        net.update(pre_mlp=(args.hidden,), lstm_hidden=args.hidden, post_mlp=(args.hidden,))
    top = {k: v for k, v in {
        "run_dir": args.run_dir,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "checkpoint_every_updates": args.checkpoint_every_updates,
        "wandb": args.wandb,
        "wandb_project": args.wandb_project,
        "video_every": args.video_every,
    }.items() if v is not None}
    return dataclasses.replace(
        cfg,
        env=dataclasses.replace(cfg.env, **env),
        ppo=dataclasses.replace(cfg.ppo, **ppo),
        actor=dataclasses.replace(cfg.actor, **net),
        critic=dataclasses.replace(cfg.critic, **net),
        **top,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    cfg.validate()
    from apple_pick_gym.rl.trainer import resolve_device, run_training

    if cfg.env.kind == "sim" and resolve_device(cfg.env.device) == "cpu" and not args.allow_cpu_sim:
        raise SystemExit(
            "--env sim needs CUDA: on a CPU Warp device the batched FR3 arm runs on Newton's MuJoCo-CPU "
            "backend, which does not integrate replicated arms (the TCP never moves). Use --env surrogate "
            "for CPU training, or --allow-cpu-sim for a wiring-only check."
        )
    result = run_training(cfg, resume=args.resume, max_updates=1 if args.dry_run else args.max_updates)
    print(
        f"trained timesteps {result.start_timestep} -> {result.timestep} ({result.updates} updates); "
        f"last checkpoint: {result.last_checkpoint}; run dir: {result.run_dir}"
    )
    if result.episodes:
        last = result.episodes[-1]
        print(
            f"last episode: success {last['Episode / success rate']:.3f}, "
            f"return {last['Episode / return (mean)']:.2f}, safety {last['Episode / safety rate']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
