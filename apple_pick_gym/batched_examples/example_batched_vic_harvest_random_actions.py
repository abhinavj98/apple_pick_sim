"""Random-action smoke test for :class:`ApplePickVicHarvestEnv`.

Builds the batched harvest env with multiple parallel worlds, then steps it
with uniformly-random 13-D delta-pose actions (independently sampled per
env, per step) for a fixed number of steps. This is a manual/visual
end-to-end health check -- not a pytest assertion suite -- meant to confirm:

- the env builds and steps without crashing across N>1 parallel worlds,
- ``obs`` only exposes proprioception + F/T keys (the vision-tracked
  geometry fields live in ``info`` now, not ``obs`` -- see
  ``apple_pick_vic_harvest_env.py::_harvest_observation_space``),
- reward/termination signals look sane under a non-scripted (random) policy.

Run from repo root (one process per invocation -- see
``docs/in-process-rebuild-heap-corruption.md``; never loop-rebuild the env
in-process)::

    uv run python apple_pick_gym/batched_examples/example_batched_vic_harvest_random_actions.py \\
        --viewer gl --num-envs 4 --num-steps 200 --seed 0

Headless (no window)::

    uv run python apple_pick_gym/batched_examples/example_batched_vic_harvest_random_actions.py \\
        --viewer null --num-envs 4 --num-steps 50

wandb (reward terms, every junction's force, wrist raw vs policy-observed, action/tracking
stats, termination fractions, per-env DR table; per-env traces for the first ``--trace-envs``
envs). Online by default (``wandb login`` first); ``--wandb-mode offline`` needs no account and
``--no-wandb`` turns it off::

    uv run python apple_pick_gym/batched_examples/example_batched_vic_harvest_random_actions.py \\
        --viewer null --num-envs 8 --num-steps 300 --wandb-mode offline
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import newton
import newton.examples
import torch

from apple_pick_gym.batched_envs.harvest_logging import (
    HarvestMetricsLogger,
    WandbSink,
    dr_table_rows,
)
from apple_pick_gym.batched_envs.harvest_video import HarvestVideoRecorder


def _make_parser() -> argparse.ArgumentParser:
    p = newton.examples.create_parser()
    p.add_argument("--num-envs", type=int, default=4, help="Parallel batched worlds.")
    p.add_argument("--num-steps", type=int, default=50, help="Random-action steps to run.")
    p.add_argument("--seed", type=int, default=0, help="Topology + DR + action-sampling seed.")
    p.add_argument(
        "--use-settle-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse settle snapshots across runs (default: off).",
    )
    p.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log debug metrics to wandb (default: on).",
    )
    p.add_argument("--wandb-project", type=str, default="ApplePick")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="online needs `wandb login`; offline writes a local run dir to sync later.",
    )
    p.add_argument(
        "--trace-envs",
        type=int,
        default=4,
        help="Log full per-env traces for the first K envs (batch stats are always logged).",
    )
    p.add_argument(
        "--action-scale",
        type=float,
        default=1.0,
        help="Scale the sampled pose-delta dims (dp, drot) by this factor; K and zeta keep "
        "their full random range.",
    )
    p.add_argument(
        "--zero-actions",
        action="store_true",
        help="Send an all-zeros 13-D action every step (no pose delta; gains/zeta clamp to their "
        "lower bounds) instead of random actions -- the no-action baseline.",
    )
    p.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record an MP4 of one randomly picked env every --record-every episodes "
        "(own headless GL viewer; independent of --viewer).",
    )
    p.add_argument("--record-every", type=int, default=5, help="Record every K-th episode (and episode 0).")
    p.add_argument("--video-dir", type=str, default="tmp/harvest_videos")
    return p


def _make_env(args: argparse.Namespace):
    from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import ApplePickVicHarvestEnv

    return ApplePickVicHarvestEnv(
        num_envs=int(args.num_envs),
        device=args.device,
        topology_seed=int(args.seed),
        dr_seed=int(args.seed),
        use_settle_cache=bool(args.use_settle_cache),
    )


def _sample_random_actions(env, generator: torch.Generator, scale: float = 1.0) -> torch.Tensor:
    low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=env.device)
    high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=env.device)
    unit = torch.rand((env.num_envs, low.shape[0]), generator=generator, device="cpu").to(env.device)
    actions = low + unit * (high - low)
    actions[:, :6] *= float(scale)
    return actions


def _render_frame(viewer: object, env, sim_time: float) -> None:
    scene = env._sim.scene
    if scene.last_vbd_contacts is not None:
        contacts = scene.last_vbd_contacts
    else:
        contacts = scene.cable.model.collide(
            scene.cable.state_0,
            collision_pipeline=scene.cable_collision_pipeline,
        )
    viewer.begin_frame(sim_time)
    viewer.log_state(scene.cable.state_0)
    viewer.log_contacts(contacts, scene.cable.state_0)
    viewer.end_frame()


def main() -> None:
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null"])
            print("No DISPLAY/WAYLAND_DISPLAY: using --viewer null.")

    viewer, args = newton.examples.init(parser=_make_parser())
    env = _make_env(args)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))

    sink = None
    if args.wandb:
        sink = WandbSink(
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                "num_envs": int(args.num_envs),
                "num_steps": int(args.num_steps),
                "seed": int(args.seed),
                "policy": "zero" if args.zero_actions else "uniform_random",
                "action_scale": float(args.action_scale),
                "record_video": bool(args.record_video),
            },
        )
    metrics = HarvestMetricsLogger(
        num_envs=env.num_envs,
        action_bounds=env._action_bounds,
        trace_envs=int(args.trace_envs),
    )
    video = (
        HarvestVideoRecorder(args.video_dir, record_every=int(args.record_every), seed=int(args.seed))
        if args.record_video
        else None
    )
    episode_idx = 0

    def _start_clip() -> None:
        if video is not None and video.should_record(episode_idx):
            env_idx = video.start_episode(env, episode_idx)
            video.capture(env, 0.0)
            print(f"recording episode {episode_idx} (env {env_idx})")

    def _finish_clip(step: int) -> None:
        if video is None or not video.recording:
            return
        env_idx = video.env_idx
        path = video.end_episode()
        if path is None:
            return
        print(f"wrote {path}")
        if sink is not None:
            sink.log_video("video/episode", path, step, fps=video.fps)
            sink.log({"video/env_idx": float(env_idx)}, step=step)

    try:
        obs, info = env.reset(seed=int(args.seed))
        metrics.on_reset(info)
        if sink is not None:
            sink.log_table("dr/per_env", dr_table_rows(env))
        sim = env._sim
        graphical = isinstance(viewer, newton.viewer.ViewerGL)
        viewer.set_model(sim.scene.cable.model)
        if graphical and env.num_envs > 1:
            viewer.set_world_offsets(tuple(sim.config.runtime.env_spacing))
        if hasattr(viewer, "hide_loading_splash"):
            viewer.hide_loading_splash()
        frame_dt = float(sim.frame_dt)
        sim_time = 0.0

        print(f"num_envs={env.num_envs} device={env.device} obs_layout={info['obs_layout']}")
        print(f"obs keys: {sorted(obs.keys())}")
        assert set(obs.keys()) == {
            "tcp_pos",
            "tcp_quat",
            "tcp_velocity",
            "ft_wrist",
            "robot_joint_q",
            "last_action",
            "step_frac",
        }, "obs leaked a non-proprioceptive/F-T field (or is missing one)"
        assert "apple_pos" in info and "woody_part_start_pos" in info, (
            "expected vision-tracked geometry to still be present in info"
        )
        _render_frame(viewer, env, sim_time)
        _start_clip()

        total_reward = torch.zeros(env.num_envs, device=env.device)
        num_terminated = 0
        num_truncated = 0
        num_resets = 0
        steps_run = 0
        num_steps = int(args.num_steps)
        print_every = max(1, num_steps // 10) if num_steps >= 20 else 1

        for step in range(num_steps):
            # Only a real window can be closed; the null viewer's is_running() turns False
            # after Newton's --num-frames (default 100), which would cut --num-steps short.
            if graphical and not viewer.is_running():
                break
            if args.zero_actions:
                actions = torch.zeros((env.num_envs, 13), dtype=torch.float32, device=env.device)
            else:
                actions = _sample_random_actions(env, generator, float(args.action_scale))
            obs, reward, terminated, truncated, info = env.step(actions)
            steps_run += 1
            if sink is not None:
                sink.log(
                    metrics.step_metrics(obs, info, actions, terminated, truncated), step=steps_run
                )
            else:
                metrics.step_metrics(obs, info, actions, terminated, truncated)
            sim_time += frame_dt
            _render_frame(viewer, env, sim_time)
            if video is not None and video.recording:
                video.capture(env, sim_time)

            reward = reward.reshape(env.num_envs)
            total_reward += reward
            num_terminated += int(terminated.sum().item())
            num_truncated += int(truncated.sum().item())

            if step % print_every == 0 or step == num_steps - 1:
                # Stats over valid envs only (failed-grasp envs are flagged at build).
                valid = ~info["invalid_env"]
                if not bool(valid.any()):
                    valid = torch.ones_like(valid)
                target_force = info["target_junction_force"][valid, :3].norm(dim=-1)
                wrist_force = obs["ft_wrist"][valid, :3].norm(dim=-1)
                print(
                    f"step {step:4d}  reward mean={reward[valid].mean().item(): .4f}  "
                    f"ft_wrist |F| mean={wrist_force.mean().item():.3f} max={wrist_force.max().item():.3f} N  "
                    f"target_junction |F| mean={target_force.mean().item():.3f} max={target_force.max().item():.3f} N  "
                    f"frozen={int(env._freeze_mask.done_mask.sum().item())}/{env.num_envs} "
                    f"invalid={int(info['invalid_env'].sum().item())} "
                    f"truncated={int(truncated.sum().item())}"
                )

            # Terminated envs are frozen in place by the env (held action, zeroed
            # reward); the sim can only reset the whole batch, so only do that at
            # truncation or once every env has finished.
            if bool(truncated.any()) or bool(env._freeze_mask.done_mask.all()):
                num_resets += 1
                _finish_clip(steps_run)
                obs, info = env.reset()
                episode_idx += 1
                summary = metrics.on_reset(info)
                if sink is not None:
                    sink.log(summary, step=steps_run)
                sim_time = 0.0
                _render_frame(viewer, env, sim_time)
                _start_clip()

            if graphical:
                time.sleep(max(0.0, frame_dt))

        _finish_clip(steps_run)
        if sink is not None:
            sink.log(metrics.flush_summary(), step=steps_run)
        print(
            f"\nDone: {steps_run} steps x {env.num_envs} envs. "
            f"cumulative reward mean={total_reward.mean().item():.4f}  "
            f"total terminated events={num_terminated}  total truncated events={num_truncated}  "
            f"whole-batch resets triggered={num_resets}"
        )
    finally:
        if video is not None:
            video.close()
        env.close()
        if sink is not None:
            sink.finish()


if __name__ == "__main__":
    main()
