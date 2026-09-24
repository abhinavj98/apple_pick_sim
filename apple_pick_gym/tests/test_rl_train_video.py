"""Training-loop video hook: record one env every ``video_every`` episodes and hand the clip to a logger."""

from __future__ import annotations

from pathlib import Path

import pytest

from apple_pick_gym.rl.config import TrainConfig
from apple_pick_gym.rl.trainer import TrainVideo


class FakeRecorder:
    def __init__(self, record_every: int) -> None:
        self.record_every = record_every
        self.events: list[tuple] = []
        self._on = False

    def should_record(self, ep: int) -> bool:
        return ep == 0 or ep % self.record_every == 0

    def start_episode(self, env, ep: int) -> int:
        self.events.append(("start", ep))
        self._on = True
        return 3

    def capture(self, env, sim_time: float) -> None:
        if self._on:
            self.events.append(("cap", round(sim_time, 6)))

    def end_episode(self):
        on, self._on = self._on, False
        self.events.append(("end",))
        return Path("clip.mp4") if on else None

    def close(self) -> None:
        self.events.append(("close",))


def test_records_selected_episodes_and_logs_clips():
    rec = FakeRecorder(record_every=2)
    logged = []
    tv = TrainVideo(rec, episode_steps=3, control_hz=10.0, log=lambda path, ep, t: logged.append((path, ep, t)))
    for t in range(12):  # 4 episodes
        tv.before_step(env=None, timestep=t)
        tv.after_step(timestep=t)
    tv.close()
    starts = [e[1] for e in rec.events if e[0] == "start"]
    assert starts == [0, 2]
    caps = [e[1] for e in rec.events if e[0] == "cap"]
    assert caps == [0.0, 0.1, 0.2, 0.0, 0.1, 0.2]
    assert logged == [(Path("clip.mp4"), 0, 3), (Path("clip.mp4"), 2, 9)]
    assert rec.events[-1] == ("close",)


def test_resume_counts_episodes_from_the_resume_reset():
    # run_training resets the env at the resume timestep, so episodes restart there
    rec = FakeRecorder(record_every=1)
    logged = []
    tv = TrainVideo(rec, episode_steps=4, control_hz=10.0, start_timestep=6, log=lambda *a: logged.append(a))
    for t in range(6, 14):
        tv.before_step(env=None, timestep=t)
        tv.after_step(timestep=t)
    assert [e for e in rec.events if e[0] == "start"] == [("start", 1), ("start", 2)]
    assert [a[2] for a in logged] == [10, 14]


def test_video_every_config_default_off_and_validated():
    assert TrainConfig().video_every == 0
    with pytest.raises(ValueError, match="video_every"):
        TrainConfig(video_every=-1).validate()


def test_cli_video_every_overrides_config():
    from apple_pick_gym.rl.train_vic_harvest import build_parser, config_from_args

    cfg_path = "apple_pick_gym/rl/configs/sim_train_gpu_d8b_tanh15_s1.json"
    args = build_parser().parse_args(["--config", cfg_path])
    assert config_from_args(args).video_every == 5
    args = build_parser().parse_args(["--config", cfg_path, "--video-every", "0"])
    cfg = config_from_args(args)
    assert cfg.video_every == 0 and cfg.seed == 1


def test_cli_max_updates_is_passed_to_run_training(monkeypatch):
    import apple_pick_gym.rl.train_vic_harvest as cli
    import apple_pick_gym.rl.trainer as trainer

    seen = {}

    class R:
        start_timestep = timestep = updates = 0
        last_checkpoint = run_dir = None
        episodes = []

    def fake(cfg, *, resume=None, max_updates=None):
        seen["max_updates"] = max_updates
        return R()

    monkeypatch.setattr(trainer, "run_training", fake)
    args = ["--config", "apple_pick_gym/rl/configs/surrogate_smoke.json"]
    cli.main(args + ["--max-updates", "7"])
    assert seen["max_updates"] == 7
    cli.main(args + ["--max-updates", "7", "--dry-run"])
    assert seen["max_updates"] == 1
    cli.main(args)
    assert seen["max_updates"] is None
