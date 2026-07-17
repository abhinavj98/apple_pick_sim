"""Integration tests for batched sys-ID structure replay orchestration."""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from apple_pick_gym.batched_envs import ApplePickBatchedSysIdEnv
from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes
from apple_pick_gym.batched_envs.batched_sysid_collect import (
    collect_batched_quasi_static_dataset,
    sample_and_broadcast_structure_params,
)
from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
    gt_bend_stiffness_candidate_from_structure,
    replay_batched_sysid_structure,
    strip_pre_weld_rows,
)
from apple_pick_gym.batched_envs.batched_sysid_world_info import weld_direction_for_world
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ControllerConfig,
    FruitingSystemConfig,
    ObsConfig,
    RobotConfig,
    SceneSettleCollisionConfig,
)
from apple_pick_sim.fruiting_system.params import GripperProxyConfig
from apple_pick_sim.fruiting_system import fruiting_params_from_json
from apple_pick_sim.system_id import BatchedSysIdDataset, QuasiStaticStepConfig
from apple_pick_sim.system_id.batched_digital_twin_init import (
    gripper_proxy_from_episode_metadata,
    infer_base_params_for_structure,
    true_params_for_structure,
)
from apple_pick_sim.tests.conftest import RANGES_FIXTURE, fr3_assets_available

_SEED = 42


def _maybe_import_gymnasium():
    try:
        import gymnasium as gym  # noqa: F401

        return True
    except Exception:
        return False


gymnasium_available = pytest.mark.skipif(
    not _maybe_import_gymnasium(),
    reason="gymnasium not installed",
)

requires_fr3 = pytest.mark.skipif(
    not fr3_assets_available(),
    reason="Requires bundled assets/fr3 and usd-core",
)


def _test_sim_config(*, num_envs: int) -> BatchedHeterogeneousCoupledSimConfig:
    return dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=num_envs),
        robot=RobotConfig(
            kind="fr3",
            step_mode="coupled",
            fix_to_apple=True,
            skip_ik_bootstrap=False,
            defer_template_robot_bootstrap=False,
            force_batched_layout=True,
        ),
        scene=SceneSettleCollisionConfig(settle_substeps=8),
        controller=ControllerConfig(mode="vic", linear_speed=1.0, angular_speed=1.0),
        domain_randomization=dataclasses.replace(
            BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=num_envs).domain_randomization,
            topology_seed=_SEED,
        ),
        obs=ObsConfig(allocate_buffers=True),
    )


@pytest.fixture(scope="module")
def tiny_batched_dataset(tmp_path_factory) -> BatchedSysIdDataset:
    if not _maybe_import_gymnasium() or not fr3_assets_available():
        pytest.skip("requires gymnasium and FR3 assets")
    num_structures = 1
    num_directions = 2
    num_envs = num_structures * num_directions
    config = QuasiStaticStepConfig(
        movement_per_step_m=0.02,
        total_movement_m=0.04,
        move_speed_mps=0.2,
        hold_duration_s=0.1,
        skip_return=True,
    )
    per_env_params = sample_and_broadcast_structure_params(
        RANGES_FIXTURE,
        topology_seed=_SEED,
        num_structures=num_structures,
        num_directions=num_directions,
    )
    output_dir = tmp_path_factory.mktemp("batched_sysid_replay")
    env = ApplePickBatchedSysIdEnv(
        num_envs=num_envs,
        max_episode_steps=120,
        ranges_path=RANGES_FIXTURE,
        topology_seed=_SEED,
        use_settle_cache=False,
        sim_config=_test_sim_config(num_envs=num_envs),
        per_env_params=per_env_params,
        control_hz=float(config.control_hz),
    )
    try:
        collect_batched_quasi_static_dataset(
            env,
            num_structures=num_structures,
            num_directions=num_directions,
            config=config,
            output_dir=output_dir,
            seed=_SEED,
            ranges_path=RANGES_FIXTURE,
            max_steps=20,
        )
    finally:
        env.close()
    return BatchedSysIdDataset(output_dir)


@pytest.fixture(scope="module")
def tiny_two_structure_dataset(tmp_path_factory) -> BatchedSysIdDataset:
    if not _maybe_import_gymnasium() or not fr3_assets_available():
        pytest.skip("requires gymnasium and FR3 assets")
    num_structures = 2
    num_directions = 2
    num_envs = num_structures * num_directions
    config = QuasiStaticStepConfig(
        movement_per_step_m=0.02,
        total_movement_m=0.04,
        move_speed_mps=0.2,
        hold_duration_s=0.1,
        skip_return=True,
    )
    per_env_params = sample_and_broadcast_structure_params(
        RANGES_FIXTURE,
        topology_seed=_SEED,
        num_structures=num_structures,
        num_directions=num_directions,
    )
    output_dir = tmp_path_factory.mktemp("batched_sysid_two_structure_parity")
    env = ApplePickBatchedSysIdEnv(
        num_envs=num_envs,
        max_episode_steps=120,
        ranges_path=RANGES_FIXTURE,
        topology_seed=_SEED,
        use_settle_cache=False,
        sim_config=_test_sim_config(num_envs=num_envs),
        per_env_params=per_env_params,
        control_hz=float(config.control_hz),
    )
    try:
        collect_batched_quasi_static_dataset(
            env,
            num_structures=num_structures,
            num_directions=num_directions,
            config=config,
            output_dir=output_dir,
            seed=_SEED,
            ranges_path=RANGES_FIXTURE,
            max_steps=20,
        )
    finally:
        env.close()
    return BatchedSysIdDataset(output_dir)


def _build_env_fn(
    *,
    ranges_path: str,
    topology_seed: int,
    sim_config_fn: Callable[..., BatchedHeterogeneousCoupledSimConfig],
) -> Callable[..., Any]:
    def build_env_fn(
        *,
        num_envs: int,
        per_env_params: list[Any],
        max_episode_steps: int,
        gripper: GripperProxyConfig | None = None,
        per_env_grippers: list[GripperProxyConfig] | None = None,
    ) -> ApplePickBatchedSysIdEnv:
        if gripper is not None and per_env_grippers is not None:
            raise ValueError(
                "scalar gripper and per_env_grippers cannot both be provided"
            )
        sim_config = sim_config_fn(num_envs=num_envs)
        if gripper is not None:
            sim_config = dataclasses.replace(
                sim_config,
                robot=dataclasses.replace(sim_config.robot, gripper=gripper),
            )
        return ApplePickBatchedSysIdEnv(
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            ranges_path=ranges_path,
            topology_seed=topology_seed,
            use_settle_cache=False,
            sim_config=sim_config,
            per_env_params=per_env_params,
            per_env_grippers=per_env_grippers,
        )

    return build_env_fn


@pytest.mark.slow
@gymnasium_available
@requires_fr3
def test_two_structure_youngs_grid_fused_matches_independent(
    tiny_two_structure_dataset: BatchedSysIdDataset,
):
    dataset = tiny_two_structure_dataset
    scoring = cmaes.YoungsModulusScoringConfig(n_directions=2)
    structures: list[
        tuple[int, tuple[cmaes.YoungsModulusCandidate, ...]]
    ] = []
    for structure_idx in (0, 1):
        gt = cmaes.gt_youngs_modulus_candidate_from_structure(dataset, structure_idx)
        structures.append(
            (
                structure_idx,
                (
                    cmaes.YoungsModulusCandidate(
                        primary=gt.primary * 2.0,
                        spur=gt.spur,
                        stem=gt.stem,
                    ),
                    gt,
                ),
            )
        )

    build_env_fn = _build_env_fn(
        ranges_path=RANGES_FIXTURE,
        topology_seed=_SEED,
        sim_config_fn=_test_sim_config,
    )
    independent_by_structure = {
        structure_idx: cmaes.evaluate_youngs_modulus_candidates(
            dataset=dataset,
            structure_idx=structure_idx,
            candidates=candidates,
            num_directions=2,
            build_env_fn=build_env_fn,
            scoring=scoring,
            seed=_SEED,
            replay_sim_config=_test_sim_config(num_envs=1),
        )
        for structure_idx, candidates in structures
    }
    fused_batch = cmaes.evaluate_youngs_modulus_structures(
        dataset=dataset,
        structures=structures,
        num_directions=2,
        build_env_fn=build_env_fn,
        scoring=scoring,
        seed=_SEED,
        replay_sim_config=_test_sim_config(num_envs=1),
    )

    assert fused_batch.errors == {}
    assert fused_batch.retried_structures == ()
    assert fused_batch.replay_diagnostics is not None
    assert fused_batch.replay_diagnostics.chunk_env_counts == (8,)
    for structure_idx in (0, 1):
        independent = independent_by_structure[structure_idx]
        fused = fused_batch.evaluations[structure_idx]
        assert [score.disqualified for score in fused.scores] == [
            score.disqualified for score in independent.scores
        ]
        assert [score.disqualification_reason for score in fused.scores] == [
            score.disqualification_reason for score in independent.scores
        ]
        assert [score.rank for score in fused.scores] == [
            score.rank for score in independent.scores
        ]
        for fused_score, independent_score in zip(
            fused.scores, independent.scores, strict=True
        ):
            if math.isfinite(independent_score.aggregate_sinkhorn):
                assert fused_score.aggregate_sinkhorn == pytest.approx(
                    independent_score.aggregate_sinkhorn,
                    rel=1e-5,
                    abs=1e-6,
                )
            assert fused_score.per_direction_sinkhorn == pytest.approx(
                independent_score.per_direction_sinkhorn,
                rel=1e-5,
                abs=1e-6,
            )


def _weld_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float64).reshape(3)
    vb = np.asarray(b, dtype=np.float64).reshape(3)
    va /= np.linalg.norm(va)
    vb /= np.linalg.norm(vb)
    return float(np.degrees(np.arccos(np.clip(float(np.dot(va, vb)), -1.0, 1.0))))


@gymnasium_available
@requires_fr3
def test_replay_batched_structure_produces_frames(tiny_batched_dataset: BatchedSysIdDataset):
    num_directions = 2
    structure_idx = 0
    candidates = [gt_bend_stiffness_candidate_from_structure(tiny_batched_dataset, structure_idx)]
    num_envs = len(candidates) * num_directions

    arrays_dir0 = strip_pre_weld_rows(
        dict(tiny_batched_dataset.load_episode_obs_arrays(structure_idx, 0))
    )
    n_frames = int(arrays_dir0["action"].shape[0])
    assert n_frames > 0

    collectors = replay_batched_sysid_structure(
        dataset=tiny_batched_dataset,
        structure_idx=structure_idx,
        candidates=candidates,
        num_directions=num_directions,
        seed=_SEED,
        build_env_fn=_build_env_fn(
            ranges_path=RANGES_FIXTURE,
            topology_seed=_SEED,
            sim_config_fn=_test_sim_config,
        ),
        replay_sim_config=_test_sim_config(num_envs=1),
    )

    for env_idx in range(num_envs):
        assert collectors.n_rows(env_idx) == n_frames

    replay_arrays = collectors.to_arrays(0)
    assert replay_arrays["ft_wrist"].shape[0] > 0


@gymnasium_available
@requires_fr3
def test_replay_build_applies_recorded_weld_direction(tmp_path):
    """Recorded weld metadata must match built proxy direction (was ~16 deg without fix)."""
    num_directions = 1
    num_envs = 1
    config = QuasiStaticStepConfig(
        movement_per_step_m=0.02,
        total_movement_m=0.04,
        move_speed_mps=0.2,
        hold_duration_s=0.1,
        skip_return=True,
    )
    per_env_params = sample_and_broadcast_structure_params(
        RANGES_FIXTURE,
        topology_seed=_SEED,
        num_structures=1,
        num_directions=num_directions,
    )
    output_dir = tmp_path / "weld_replay"
    collect_env = ApplePickBatchedSysIdEnv(
        num_envs=num_envs,
        max_episode_steps=120,
        ranges_path=RANGES_FIXTURE,
        topology_seed=_SEED,
        use_settle_cache=False,
        sim_config=_test_sim_config(num_envs=num_envs),
        per_env_params=per_env_params,
        control_hz=float(config.control_hz),
    )
    try:
        collect_batched_quasi_static_dataset(
            collect_env,
            num_structures=1,
            num_directions=num_directions,
            config=config,
            output_dir=output_dir,
            seed=_SEED,
            ranges_path=RANGES_FIXTURE,
            max_steps=20,
        )
    finally:
        collect_env.close()

    dataset = BatchedSysIdDataset(output_dir)
    structure_idx = 0
    meta = dataset.load_episode_metadata(structure_idx, 0)
    recorded_weld = np.asarray(meta["weld_direction"], dtype=np.float64)
    arrays = dataset.load_episode_obs_arrays(structure_idx, 0)
    n_frames = int(arrays["action"].shape[0])
    params = per_env_params[0]
    inferred_params = infer_base_params_for_structure(dataset, structure_idx)

    build_fn = _build_env_fn(
        ranges_path=RANGES_FIXTURE,
        topology_seed=_SEED,
        sim_config_fn=_test_sim_config,
    )
    good_gripper = gripper_proxy_from_episode_metadata(meta)

    env_without = build_fn(
        num_envs=1,
        per_env_params=[inferred_params],
        max_episode_steps=n_frames,
    )
    env_with = build_fn(
        num_envs=1,
        per_env_params=[params],
        max_episode_steps=n_frames,
        gripper=good_gripper,
    )
    try:
        env_without.reset(seed=_SEED)
        wrong_weld = weld_direction_for_world(
            env_without._sim.scene,
            env_without._sim.layout,
            0,
        )
        wrong_angle = _weld_angle_deg(wrong_weld, recorded_weld)

        env_with.reset(seed=_SEED)
        live_weld = weld_direction_for_world(
            env_with._sim.scene,
            env_with._sim.layout,
            0,
        )
        fixed_angle = _weld_angle_deg(live_weld, recorded_weld)
    finally:
        env_without.close()
        env_with.close()

    assert wrong_angle > 5.0, f"expected large mismatch without fix, got {wrong_angle:.2f} deg"
    assert fixed_angle < 1.0, f"recorded weld mismatch {fixed_angle:.2f} deg"


@gymnasium_available
@requires_fr3
def test_replay_structure_uses_true_params_geometry(tiny_batched_dataset: BatchedSysIdDataset):
    structure_idx = 0
    true_params = true_params_for_structure(tiny_batched_dataset, structure_idx)
    inferred_params = infer_base_params_for_structure(tiny_batched_dataset, structure_idx)
    meta = tiny_batched_dataset.load_episode_metadata(structure_idx, 0)
    recorded_params = fruiting_params_from_json(str(meta["fruiting_system_params"]))
    arrays = tiny_batched_dataset.load_episode_obs_arrays(structure_idx, 0)
    n_frames = int(arrays["action"].shape[0])

    build_fn = _build_env_fn(
        ranges_path=RANGES_FIXTURE,
        topology_seed=_SEED,
        sim_config_fn=_test_sim_config,
    )
    env = build_fn(
        num_envs=1,
        per_env_params=[true_params],
        max_episode_steps=n_frames,
        gripper=gripper_proxy_from_episode_metadata(meta),
    )
    try:
        env.reset(seed=_SEED)
        built_params = env._sim._per_env_params[0]
        assert built_params.primary is not None
        assert recorded_params.primary is not None
        assert inferred_params.primary is not None

        assert built_params.primary.length == pytest.approx(recorded_params.primary.length)
        assert built_params.primary.direction == pytest.approx(
            recorded_params.primary.direction
        )
        assert built_params.primary.bend_stiffness == pytest.approx(
            recorded_params.primary.bend_stiffness
        )

        length_delta = abs(inferred_params.primary.length - recorded_params.primary.length)
        direction_delta = 1.0 - abs(
            float(
                np.dot(
                    np.asarray(inferred_params.primary.direction),
                    np.asarray(recorded_params.primary.direction),
                )
            )
        )
        assert length_delta > 1e-6 or direction_delta > 1e-6
    finally:
        env.close()


@gymnasium_available
@requires_fr3
def test_replay_uses_manifest_seed_when_seed_is_none(tiny_batched_dataset: BatchedSysIdDataset):
    num_directions = 2
    structure_idx = 0
    candidates = [gt_bend_stiffness_candidate_from_structure(tiny_batched_dataset, structure_idx)]
    manifest_seed = int(tiny_batched_dataset.manifest["collection"]["seed"])
    assert manifest_seed == _SEED

    build_fn = _build_env_fn(
        ranges_path=RANGES_FIXTURE,
        topology_seed=_SEED,
        sim_config_fn=_test_sim_config,
    )
    seen_seeds: list[int] = []
    original_reset = ApplePickBatchedSysIdEnv.reset

    def tracking_reset(self, *, seed=None, options=None):
        if seed is not None:
            seen_seeds.append(int(seed))
        return original_reset(self, seed=seed, options=options)

    ApplePickBatchedSysIdEnv.reset = tracking_reset  # type: ignore[method-assign]
    try:
        replay_batched_sysid_structure(
            dataset=tiny_batched_dataset,
            structure_idx=structure_idx,
            candidates=candidates,
            num_directions=num_directions,
            seed=None,
            build_env_fn=build_fn,
            replay_sim_config=_test_sim_config(num_envs=1),
        )
    finally:
        ApplePickBatchedSysIdEnv.reset = original_reset  # type: ignore[method-assign]

    assert seen_seeds == [manifest_seed]


@gymnasium_available
@requires_fr3
def test_replay_warns_on_manifest_sim_config_mismatch(tiny_batched_dataset: BatchedSysIdDataset):
    num_directions = 2
    structure_idx = 0
    candidates = [gt_bend_stiffness_candidate_from_structure(tiny_batched_dataset, structure_idx)]
    mismatched = dataclasses.replace(
        _test_sim_config(num_envs=1),
        fruiting_system=FruitingSystemConfig(
            joint_angular_kd_overrides={
                "support": 1.0,
                "primary_spur": 1.0,
                "stem_apple": 99.0,
            }
        ),
    )
    def _mismatched_sim_config_fn(*, num_envs: int, **_kwargs):
        base = _test_sim_config(num_envs=int(num_envs))
        return dataclasses.replace(base, fruiting_system=mismatched.fruiting_system)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        replay_batched_sysid_structure(
            dataset=tiny_batched_dataset,
            structure_idx=structure_idx,
            candidates=candidates,
            num_directions=num_directions,
            seed=_SEED,
            build_env_fn=_build_env_fn(
                ranges_path=RANGES_FIXTURE,
                topology_seed=_SEED,
                sim_config_fn=_mismatched_sim_config_fn,
            ),
            replay_sim_config=mismatched,
        )

    assert any("manifest sim_config mismatch" in str(w.message) for w in caught)
    assert any("stem_apple" in str(w.message) for w in caught)


@gymnasium_available
@requires_fr3
def test_replay_silent_when_manifest_sim_config_matches(tiny_batched_dataset: BatchedSysIdDataset):
    num_directions = 2
    structure_idx = 0
    candidates = [gt_bend_stiffness_candidate_from_structure(tiny_batched_dataset, structure_idx)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        replay_batched_sysid_structure(
            dataset=tiny_batched_dataset,
            structure_idx=structure_idx,
            candidates=candidates,
            num_directions=num_directions,
            seed=_SEED,
            build_env_fn=_build_env_fn(
                ranges_path=RANGES_FIXTURE,
                topology_seed=_SEED,
                sim_config_fn=_test_sim_config,
            ),
            replay_sim_config=_test_sim_config(num_envs=1),
        )

    mismatch_warnings = [
        w for w in caught if "manifest sim_config mismatch" in str(w.message)
    ]
    assert mismatch_warnings == []
