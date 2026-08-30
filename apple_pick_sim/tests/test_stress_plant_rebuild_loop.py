"""Tests for stress_plant_rebuild_loop (plant-only rebuild stress harness)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
)

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STRESS = _EXAMPLES_DIR / "stress_plant_rebuild_loop.py"

if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from stress_plant_rebuild_loop import (  # noqa: E402
    DEFAULT_LOG10_LOWER,
    DEFAULT_LOG10_UPPER,
    assert_cloned_obs_matches_sim,
    build_stress_config,
    clone_step_obs_cpu,
    make_parser,
    read_rss_mib,
    resolve_replay_episode_sources,
    run_rebuild_cycles,
    sample_random_candidates,
    teardown_env,
    teardown_sim,
    validate_cli_args,
)


def test_parser_defaults_reuse_mujoco_on():
    args = make_parser().parse_args([])
    assert args.num_envs == 100
    assert args.cycles == 10
    assert args.mode == "rebuild"
    assert args.reuse_replicated_mujoco is True
    assert args.settle_substeps == 2000
    assert args.post_grasp_settle_substeps == 500
    assert args.params_seed == 0
    assert args.replay_steps == 0
    assert args.structure_idx == 0
    assert args.resets_per_wave == 100
    assert list(args.log10_lower) == list(DEFAULT_LOG10_LOWER)
    assert list(args.log10_upper) == list(DEFAULT_LOG10_UPPER)


def test_teardown_sim_synchronizes_before_collect(monkeypatch):
    order: list[str] = []
    import stress_plant_rebuild_loop as stress

    monkeypatch.setattr(stress.wp, "synchronize", lambda: order.append("sync"))
    monkeypatch.setattr(stress.gc, "collect", lambda: order.append("gc") or 0)
    teardown_sim(object())
    assert order == ["sync", "gc"]


def test_teardown_env_close_then_sync(monkeypatch):
    order: list[str] = []
    import stress_plant_rebuild_loop as stress

    monkeypatch.setattr(stress.wp, "synchronize", lambda: order.append("sync"))
    monkeypatch.setattr(stress.gc, "collect", lambda: order.append("gc") or 0)

    class _Env:
        def close(self):
            order.append("close")

    teardown_env(_Env())
    assert order == ["close", "sync", "gc"]


def test_validate_cli_replay_reset_requires_dataset():
    args = make_parser().parse_args(["--mode", "replay-reset"])
    with pytest.raises(SystemExit, match="--dataset"):
        validate_cli_args(args)


def test_validate_cli_rebuild_replay_requires_dataset_and_resets():
    args = make_parser().parse_args(["--mode", "rebuild-replay"])
    with pytest.raises(SystemExit, match="--dataset"):
        validate_cli_args(args)
    args = make_parser().parse_args(
        ["--mode", "rebuild-replay", "--dataset", "tmp/foo", "--resets-per-wave", "0"]
    )
    with pytest.raises(SystemExit, match="--resets-per-wave"):
        validate_cli_args(args)


def test_validate_cli_rejects_negative_post_grasp_settle():
    args = make_parser().parse_args(["--post-grasp-settle-substeps", "-1"])
    with pytest.raises(SystemExit, match="--post-grasp-settle-substeps"):
        validate_cli_args(args)


def test_resolve_replay_episode_sources_cycles_directions():
    sources = resolve_replay_episode_sources(
        structure_idx=3,
        num_envs=5,
        direction_indices=(1, 2),
    )
    assert len(sources) == 5
    assert [(s.structure_idx, s.direction_idx) for s in sources] == [
        (3, 1),
        (3, 2),
        (3, 1),
        (3, 2),
        (3, 1),
    ]


def test_sample_random_candidates_stay_in_log10_bounds():
    import random

    rng = random.Random(7)
    candidates = sample_random_candidates(
        rng=rng,
        num_envs=20,
        log10_lower=DEFAULT_LOG10_LOWER,
        log10_upper=DEFAULT_LOG10_UPPER,
    )
    assert len(candidates) == 20
    for candidate in candidates:
        assert 10.0 ** DEFAULT_LOG10_LOWER[0] <= candidate.support_kp <= 10.0 ** DEFAULT_LOG10_UPPER[0]
        assert 10.0 ** DEFAULT_LOG10_LOWER[1] <= candidate.spur_pa <= 10.0 ** DEFAULT_LOG10_UPPER[1]
        assert 10.0 ** DEFAULT_LOG10_LOWER[2] <= candidate.stem_pa <= 10.0 ** DEFAULT_LOG10_UPPER[2]


def test_build_stress_config_forwards_reuse_and_settle():
    cfg = build_stress_config(
        num_envs=4,
        device="cpu",
        settle_substeps=50,
        post_grasp_settle_substeps=500,
        reuse_mujoco=True,
        topology_seed=7,
    )
    assert cfg.runtime.num_envs == 4
    assert cfg.runtime.device == "cpu"
    assert cfg.scene.settle_substeps == 50
    assert cfg.scene.post_grasp_settle_substeps == 500
    assert cfg.scene.settle_quiet_every == 50
    assert cfg.robot.reuse_replicated_mujoco is True
    assert cfg.robot.fix_to_apple is True
    assert cfg.domain_randomization.topology_seed == 7


def test_read_rss_mib_positive():
    assert read_rss_mib() > 0.0


def _fake_last_obs(*, num_envs: int = 2):
    import torch

    n = int(num_envs)
    return {
        "woody_part_info": {
            "j0": {
                "anchors_pos": torch.arange(n * 6, dtype=torch.float32).reshape(n, 6),
            }
        },
        "ft_wrist": torch.arange(n * 6, dtype=torch.float32).reshape(n, 6) + 1.0,
        "tcp_velocity": torch.zeros(n, 6, dtype=torch.float32),
        "tcp_pos": torch.zeros(n, 3, dtype=torch.float32),
        "apple_pos": torch.zeros(n, 3, dtype=torch.float32),
        "apple_quat": torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]] * n, dtype=torch.float32
        ),
    }


def test_clone_step_obs_cpu_does_not_alias_torch_last_obs():
    from types import SimpleNamespace

    obs = _fake_last_obs(num_envs=2)
    env = SimpleNamespace(_last_obs=obs, junction_names=["j0"])
    cloned = clone_step_obs_cpu(env)
    ft_before = cloned["ft_wrist"].copy()
    quat_before = cloned["apple_quat"].copy()
    obs["ft_wrist"].fill_(9.0)
    obs["apple_quat"].fill_(0.5)
    np.testing.assert_allclose(cloned["ft_wrist"], ft_before)
    np.testing.assert_allclose(cloned["apple_quat"], quat_before)
    assert not np.allclose(cloned["ft_wrist"], 9.0)


def test_assert_cloned_obs_matches_sim_body_q_and_cache():
    from types import SimpleNamespace

    class _HostArray:
        def __init__(self, values) -> None:
            self._values = np.asarray(values, dtype=np.float32)

        def numpy(self):
            return self._values

    body_q = np.zeros((4, 7), dtype=np.float32)
    body_q[:, 6] = 1.0
    body_q[1, 3:7] = [0.1, 0.2, 0.3, 0.9]
    body_q[3, 3:7] = [0.0, 0.0, 0.0, 1.0]
    cache = np.arange(24, dtype=np.float32).reshape(4, 6)
    env = SimpleNamespace(
        num_envs=2,
        _sim=SimpleNamespace(
            layout=SimpleNamespace(
                apple_body_indices=(1, 3),
                tcp_body_indices=(0, 2),
            ),
            scene=SimpleNamespace(
                cable=SimpleNamespace(
                    state_0=SimpleNamespace(body_q=_HostArray(body_q))
                ),
                coupling_forces_cache=_HostArray(cache),
            ),
        ),
    )
    cloned = {
        "apple_quat": np.stack([body_q[1, 3:7], body_q[3, 3:7]]),
        "ft_wrist": np.stack([cache[0], cache[2]]),
    }
    assert_cloned_obs_matches_sim(env, cloned)

    bad = dict(cloned)
    bad["apple_quat"] = cloned["apple_quat"] + 0.25
    with pytest.raises(AssertionError, match="apple_quat"):
        assert_cloned_obs_matches_sim(env, bad)

    bad_ft = dict(cloned)
    bad_ft["ft_wrist"] = cloned["ft_wrist"] + 1.0
    with pytest.raises(AssertionError, match="ft_wrist"):
        assert_cloned_obs_matches_sim(env, bad_ft)


@pytest.mark.slow
def test_stress_rebuild_two_cycles_cpu_subprocess():
    cmd = [
        sys.executable,
        str(_STRESS),
        "--num-envs",
        "2",
        "--cycles",
        "2",
        "--settle-substeps",
        "20",
        "--post-grasp-settle-substeps",
        "0",
        "--topology-seed",
        "42",
        "--device",
        "cpu",
        "--reuse-replicated-mujoco",
    ]
    proc = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "cycle=0" in proc.stdout
    assert "cycle=1" in proc.stdout


@pytest.mark.slow
def test_run_rebuild_cycles_mujoco_cache_hits_on_second_cycle():
    from apple_pick_sim.coupled_fruiting.replicated_robot_cache import (
        clear_process_replicated_robot_cache,
    )
    from apple_pick_sim.fruiting_system import default_ranges_fixture_path, load_ranges

    clear_process_replicated_robot_cache()
    ranges = load_ranges(default_ranges_fixture_path())
    cfg = build_stress_config(
        num_envs=2,
        device="cpu",
        settle_substeps=20,
        post_grasp_settle_substeps=0,
        reuse_mujoco=True,
        topology_seed=42,
    )
    results = run_rebuild_cycles(
        config=cfg,
        ranges=ranges,
        cycles=2,
        params_seed=42,
    )
    assert len(results) == 2
    assert results[0].mujoco_misses >= 1
    assert results[1].mujoco_hits >= 1
    assert results[1].mujoco_misses == results[0].mujoco_misses


@pytest.mark.slow
def test_stress_replay_reset_two_cycles_cpu_subprocess():
    dataset = _REPO_ROOT / "tmp" / "real_batched_s09_k_frame"
    if not dataset.is_dir():
        pytest.skip("requires tmp/real_batched_s09_k_frame dataset")
    cmd = [
        sys.executable,
        str(_STRESS),
        "--mode",
        "replay-reset",
        "--dataset",
        str(dataset),
        "--num-envs",
        "2",
        "--cycles",
        "2",
        "--settle-substeps",
        "20",
        "--post-grasp-settle-substeps",
        "0",
        "--topology-seed",
        "42",
        "--device",
        "cpu",
        "--reuse-replicated-mujoco",
        "--direction-indices",
        "0",
        "1",
    ]
    proc = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "build_s=" in proc.stdout
    assert "wave=0 reset=0" in proc.stdout
    assert "wave=0 reset=1" in proc.stdout
    assert "reset_s=" in proc.stdout


@pytest.mark.slow
def test_stress_rebuild_replay_two_waves_cpu_subprocess():
    dataset = _REPO_ROOT / "tmp" / "real_batched_s09_k_frame"
    if not dataset.is_dir():
        pytest.skip("requires tmp/real_batched_s09_k_frame dataset")
    cmd = [
        sys.executable,
        str(_STRESS),
        "--mode",
        "rebuild-replay",
        "--dataset",
        str(dataset),
        "--num-envs",
        "2",
        "--cycles",
        "2",
        "--resets-per-wave",
        "2",
        "--settle-substeps",
        "20",
        "--post-grasp-settle-substeps",
        "0",
        "--topology-seed",
        "42",
        "--device",
        "cpu",
        "--reuse-replicated-mujoco",
        "--direction-indices",
        "0",
        "1",
    ]
    proc = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert proc.stdout.count("wave=") >= 2
    assert proc.stdout.count("build_s=") >= 2
    assert "reset_s=" in proc.stdout
