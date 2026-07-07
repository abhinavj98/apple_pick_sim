"""CLI contract tests for the batched sys-ID MMD grid replay example."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import subprocess
import sys
from pathlib import Path

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
)


def _load_example_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "batched_examples"
        / "example_batched_sysid_mmd_grid.py"
    )
    spec = importlib.util.spec_from_file_location(
        "example_batched_sysid_mmd_grid_under_test", path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_help_smoke_via_subprocess():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "apple_pick_gym/batched_examples/example_batched_sysid_mmd_grid.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--dataset" in proc.stdout
    assert "--replay-only" in proc.stdout
    assert "--primary-bend-stiffness-values" in proc.stdout


def test_parser_defaults_and_grid_args(monkeypatch):
    module = _load_example_module()

    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)

    parser = module._make_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "/tmp/batched_sysid",
            "--primary-bend-stiffness-values",
            "1,2",
            "--secondary-bend-stiffness-values",
            "10",
            "--spur-bend-stiffness-values",
            "100",
            "--stem-bend-stiffness-values",
            "1000,2000",
        ]
    )

    assert args.dataset == "/tmp/batched_sysid"
    assert args.structure_indices is None
    assert args.max_envs_per_batch == 0
    assert args.max_candidates == 0
    assert args.seed == 0
    assert args.replay_only is False
    assert args.primary_bend_stiffness_values == (1.0, 2.0)
    assert args.secondary_bend_stiffness_values == (10.0,)
    assert args.spur_bend_stiffness_values == (100.0,)
    assert args.stem_bend_stiffness_values == (1000.0, 2000.0)


def test_parser_accepts_structure_indices_and_batch_limits(monkeypatch):
    module = _load_example_module()

    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)

    parser = module._make_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "/tmp/batched_sysid",
            "--structure-indices",
            "0,2,5",
            "--max-envs-per-batch",
            "16",
            "--max-candidates",
            "8",
            "--seed",
            "7",
            "--replay-only",
            "--primary-bend-stiffness-values",
            "1",
            "--secondary-bend-stiffness-values",
            "2",
            "--spur-bend-stiffness-values",
            "3",
            "--stem-bend-stiffness-values",
            "4",
        ]
    )

    assert args.structure_indices == (0, 2, 5)
    assert args.max_envs_per_batch == 16
    assert args.max_candidates == 8
    assert args.seed == 7
    assert args.replay_only is True


def test_chunk_candidates_no_chunking_when_limit_zero():
    module = _load_example_module()
    from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import BendStiffnessCandidate

    candidates = [
        BendStiffnessCandidate(1.0, 2.0, 3.0, 4.0),
        BendStiffnessCandidate(5.0, 6.0, 7.0, 8.0),
    ]

    chunks = module.chunk_candidates(
        candidates,
        max_envs_per_batch=0,
        num_directions=3,
    )

    assert chunks == [candidates]


def test_chunk_candidates_respects_env_budget():
    module = _load_example_module()
    from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import BendStiffnessCandidate

    candidates = [
        BendStiffnessCandidate(float(i), 1.0, 1.0, 1.0) for i in range(5)
    ]

    chunks = module.chunk_candidates(
        candidates,
        max_envs_per_batch=6,
        num_directions=2,
    )

    assert [len(chunk) for chunk in chunks] == [3, 2]


def test_sim_config_stays_in_module_constants():
    module = _load_example_module()

    cfg = module.build_sim_config(num_envs=8)
    gym_cfg = BatchedHeterogeneousCoupledSimConfig.gym_defaults(num_envs=8)
    assert cfg == dataclasses.replace(
        gym_cfg,
        runtime=dataclasses.replace(gym_cfg.runtime, control_hz=module.CONTROL_HZ),
        scene=dataclasses.replace(gym_cfg.scene, settle_substeps=module.SETTLE_SUBSTEPS),
        controller=dataclasses.replace(
            gym_cfg.controller,
            vic_gains=module.VIC_GAINS,
        ),
    )
    assert cfg.controller.mode == "vic"
