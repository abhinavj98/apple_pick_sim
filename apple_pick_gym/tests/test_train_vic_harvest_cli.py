"""train_vic_harvest CLI: config + flag overrides, dry run, resume (surrogate env, CPU)."""

from __future__ import annotations

import json

import pytest

from apple_pick_gym.rl import train_vic_harvest as cli

_SMALL = [
    "--env", "surrogate", "--num-envs", "8", "--max-episode-steps", "16", "--device", "cpu",
    "--rollouts", "16", "--mini-batches", "2", "--learning-epochs", "1", "--sequence-length", "8",
    "--hidden", "16",
]


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    assert "--resume" in capsys.readouterr().out


def test_flags_override_the_json_config(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"env": {"kind": "surrogate", "num_envs": 4}, "seed": 7}))
    cfg = cli.config_from_args(cli.build_parser().parse_args(["--config", str(cfg_path), "--num-envs", "8", "--run-dir", str(tmp_path / "r")]))
    assert cfg.env.num_envs == 8 and cfg.env.kind == "surrogate" and cfg.seed == 7
    assert cfg.run_dir == str(tmp_path / "r")


def test_dry_run_does_one_update_and_writes_a_checkpoint(tmp_path):
    rc = cli.main([*_SMALL, "--timesteps", "1000", "--run-dir", str(tmp_path / "run"), "--dry-run"])
    assert rc == 0
    ckpts = sorted((tmp_path / "run" / "checkpoints").iterdir())
    assert [p.name for p in ckpts] == ["ckpt_000000016"]
    assert json.loads((tmp_path / "run" / "config.json").read_text())["env"]["kind"] == "surrogate"


def test_train_then_resume_latest(tmp_path):
    run = str(tmp_path / "run")
    assert cli.main([*_SMALL, "--timesteps", "32", "--run-dir", run]) == 0
    assert cli.main([*_SMALL, "--timesteps", "48", "--run-dir", run, "--resume", "latest"]) == 0
    names = [p.name for p in sorted((tmp_path / "run" / "checkpoints").iterdir())]
    assert names[-1] == "ckpt_000000048"


def test_sim_on_cpu_is_refused_without_the_wiring_flag(tmp_path):
    with pytest.raises(SystemExit, match="CUDA"):
        cli.main(["--env", "sim", "--device", "cpu", "--run-dir", str(tmp_path / "r"), "--dry-run"])
