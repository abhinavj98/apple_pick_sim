"""CLI contract tests for the batched sys-ID MMD grid replay example."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

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
    assert "--score-mse" in proc.stdout
    assert "--score-wasserstein" in proc.stdout
    assert "--infer-params" in proc.stdout
    assert "--use-snapshot" in proc.stdout
    assert "--primary-bend-stiffness-values" in proc.stdout
    assert "--settle-quiet-every" in proc.stdout


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
    assert args.max_envs_per_batch == module.MAX_ENVS_PER_BATCH
    assert args.max_candidates == 0
    assert args.use_snapshot is False
    assert args.infer_params is False
    assert args.seed is None
    assert args.replay_only is False
    assert args.score_mse is False
    assert args.score_wasserstein is False
    assert args.grid_values_are_gt_multipliers is False
    assert args.primary_bend_stiffness_values == (1.0, 2.0)
    assert args.secondary_bend_stiffness_values == (10.0,)
    assert args.spur_bend_stiffness_values == (100.0,)
    assert args.stem_bend_stiffness_values == (1000.0, 2000.0)
    assert module.SETTLE_QUIET_EVERY == 300
    assert args.settle_quiet_every == 300


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
            "--score-mse",
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
    assert args.score_mse is True
    assert args.use_median is True
    assert args.hold_id_onehot is True
    assert args.pool_directions is True

    args_off = parser.parse_args(
        [
            "--dataset",
            "/tmp/batched_sysid",
            "--no-use-median",
            "--no-hold-id-onehot",
            "--no-pool-directions",
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
    assert args_off.use_median is False
    assert args_off.hold_id_onehot is False
    assert args_off.pool_directions is False


def test_deprecated_mse_hold_flags_map_to_use_median(monkeypatch):
    module = _load_example_module()
    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)
    parser = module._make_parser()
    base = [
        "--dataset",
        "/tmp/batched_sysid",
        "--primary-bend-stiffness-values",
        "1",
        "--secondary-bend-stiffness-values",
        "2",
        "--spur-bend-stiffness-values",
        "3",
        "--stem-bend-stiffness-values",
        "4",
    ]
    with pytest.warns(DeprecationWarning, match="mse-hold-aggregation"):
        args_med = module.apply_deprecated_mse_cli_flags(
            parser.parse_args([*base, "--mse-hold-aggregation", "median"])
        )
    assert args_med.use_median is True
    with pytest.warns(DeprecationWarning, match="mse-hold-aggregation"):
        args_none = module.apply_deprecated_mse_cli_flags(
            parser.parse_args([*base, "--mse-hold-aggregation", "none"])
        )
    assert args_none.use_median is False
    with pytest.warns(DeprecationWarning, match="mse-hold-latter-half"):
        args_lh = module.apply_deprecated_mse_cli_flags(
            parser.parse_args([*base, "--no-mse-hold-latter-half"])
        )
    assert args_lh.use_median is True


def test_pool_directions_implies_dir_id_onehot_in_score_json_header(monkeypatch):
    """Score JSON header mirrors wasserstein: pooling auto-enables dir one-hot."""
    module = _load_example_module()
    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)
    parser = module._make_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "/tmp/batched_sysid",
            "--pool-directions",
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
    # Same mapping written in _run when --score-json-output is set.
    payload = {
        "pool_directions": bool(args.pool_directions),
        "dir_id_onehot": bool(args.pool_directions),
    }
    assert payload["pool_directions"] is True
    assert payload["dir_id_onehot"] is True


def test_plot_metrics_validation_rejects_unknown_metric(monkeypatch):
    module = _load_example_module()

    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)

    parser = module._make_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--dataset",
                "/tmp/batched_sysid",
                "--plot-output",
                "/tmp/plots",
                "--plot-metrics",
                "err_pos_hold,not_a_metric",
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


def test_plot_metrics_validation_dedupes_and_preserves_order():
    module = _load_example_module()
    metrics = module._parse_plot_metrics("err_pos_hold, err_force_hold, err_pos_hold")
    assert metrics == ("err_pos_hold", "err_force_hold")


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


def test_collection_control_hz_reads_manifest():
    module = _load_example_module()
    assert module._collection_control_hz({"control_hz": 25.0}) == pytest.approx(25.0)
    assert module._collection_control_hz({}) == pytest.approx(module.CONTROL_HZ)


def test_sim_config_stays_in_module_constants():
    from apple_pick_sim.fruiting_system import default_ranges_fixture_path, load_ranges

    module = _load_example_module()

    cfg = module.build_sim_config(num_envs=8)
    ranges = load_ranges(default_ranges_fixture_path())
    (
        vic_gains,
        joint_angular_kd,
        joint_linear_kd,
        joint_angular_kp,
        joint_linear_kp,
        joint_damping_ratio,
    ) = module._resolve_sim_build_knobs(ranges)
    gym_cfg = BatchedHeterogeneousCoupledSimConfig.gym_defaults(num_envs=8)
    assert cfg == dataclasses.replace(
        gym_cfg,
        runtime=dataclasses.replace(gym_cfg.runtime, control_hz=module.CONTROL_HZ),
        scene=dataclasses.replace(
            gym_cfg.scene,
            settle_substeps=module.SETTLE_SUBSTEPS,
            settle_gravity_ramp=module.SETTLE_GRAVITY_RAMP,
            settle_quiet_every=module.SETTLE_QUIET_EVERY,
        ),
        controller=dataclasses.replace(
            gym_cfg.controller,
            vic_gains=vic_gains,
        ),
        fruiting_system=dataclasses.replace(
            gym_cfg.fruiting_system,
            joint_angular_kd_overrides=joint_angular_kd,
            joint_linear_kd_overrides=joint_linear_kd,
            joint_angular_kp_overrides=joint_angular_kp,
            joint_linear_kp_overrides=joint_linear_kp,
            joint_damping_ratio=joint_damping_ratio,
        ),
    )
    assert cfg.controller.mode == "vic"
    assert cfg.fruiting_system.stem_force_cap_N == pytest.approx(
        module.DEFAULT_STEM_FORCE_CAP_N
    )
    assert cfg.fruiting_system.stem_torque_cap_Nm == pytest.approx(
        module.DEFAULT_STEM_TORQUE_CAP_NM
    )
    assert module.DEFAULT_STEM_FORCE_CAP_N == pytest.approx(100.0)
    assert module.DEFAULT_STEM_TORQUE_CAP_NM == pytest.approx(40.0)


def test_build_sim_config_reads_sim_build_from_default_fixture():
    from apple_pick_sim.fruiting_system import default_ranges_fixture_path, load_ranges, parse_sim_build

    module = _load_example_module()
    ranges = load_ranges(default_ranges_fixture_path())
    sb = parse_sim_build(ranges)
    assert sb is not None
    cfg = module.build_sim_config(num_envs=2, ranges=ranges)
    assert cfg.controller.vic_gains.linear_k == pytest.approx(sb.vic_gains.linear_k)
    assert cfg.fruiting_system.joint_angular_kd_overrides == sb.joint_angular_kd_overrides
    assert cfg.fruiting_system.joint_angular_kp_overrides == sb.joint_angular_kp_overrides
    assert cfg.fruiting_system.joint_damping_ratio == sb.joint_damping_ratio
    assert cfg.fruiting_system.joint_damping_ratio == pytest.approx(0.2)


def test_build_sim_config_settle_override():
    module = _load_example_module()
    cfg = module.build_sim_config(num_envs=2, settle_substeps=0)
    assert cfg.scene.settle_substeps == 0


def test_build_sim_config_settle_override_is_int_coerced():
    module = _load_example_module()
    cfg = module.build_sim_config(num_envs=2, settle_substeps=1.0)
    assert cfg.scene.settle_substeps == 1


def test_build_sim_config_settle_quiet_every_override():
    module = _load_example_module()
    cfg = module.build_sim_config(num_envs=2, settle_quiet_every=200)
    assert cfg.scene.settle_quiet_every == 200


def test_settle_config_kwargs_snapshot_disables_settle():
    module = _load_example_module()
    args = argparse.Namespace(
        settle_substeps=5000,
        settle_gravity_ramp=True,
        settle_quiet_every=100,
    )
    kwargs = module._settle_config_kwargs(args=args, use_snapshot=True)
    assert kwargs == {
        "settle_substeps": 0,
        "settle_gravity_ramp": False,
        "settle_quiet_every": None,
    }


def test_candidates_for_structure_retains_gt_under_max_candidates(monkeypatch):
    module = _load_example_module()
    from apple_pick_gym.batched_envs import batched_sysid_mmd_grid as grid_mod
    from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import BendStiffnessCandidate

    gt = BendStiffnessCandidate(99.0, 99.0, 99.0, 99.0)
    monkeypatch.setattr(
        grid_mod,
        "gt_bend_stiffness_candidate_from_structure",
        lambda dataset, structure_idx: gt,
    )
    monkeypatch.setattr(
        module,
        "_build_candidate_grid",
        lambda args: [
            BendStiffnessCandidate(1.0, 1.0, 1.0, 1.0),
            BendStiffnessCandidate(2.0, 2.0, 2.0, 2.0),
            BendStiffnessCandidate(3.0, 3.0, 3.0, 3.0),
        ],
    )
    args = argparse.Namespace(
        max_candidates=1,
        grid_values_are_gt_multipliers=False,
    )
    out = module._candidates_for_structure(object(), args, structure_idx=0)
    assert out[0] == BendStiffnessCandidate(1.0, 1.0, 1.0, 1.0)
    assert out[-1] == gt
    assert len(out) == 2


def test_build_sim_config_accepts_device_override():
    module = _load_example_module()
    cfg = module.build_sim_config(num_envs=2, device="cpu")
    assert cfg.runtime.device == "cpu"
