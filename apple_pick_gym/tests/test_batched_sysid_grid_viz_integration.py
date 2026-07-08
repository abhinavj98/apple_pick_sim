from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import numpy as np
import pytest

from apple_pick_gym.batched_envs.batched_sysid_collect import collect_batched_quasi_static_dataset
from apple_pick_gym.batched_envs.batched_sysid_collect import sample_and_broadcast_structure_params
from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
    BendStiffnessCandidate,
    direction_episodes_from_collectors,
    gt_bend_stiffness_candidate_from_structure,
    replay_candidates_for_structure,
    load_recorded_episodes_for_structure,
)
from apple_pick_gym.grid_viz_report import summarize_structure
from apple_pick_gym.grid_viz_table import build_grid_viz_rows
from apple_pick_sim.system_id import BatchedSysIdDataset, QuasiStaticStepConfig
from apple_pick_sim.tests.conftest import RANGES_FIXTURE, fr3_assets_available


def _maybe_import_gymnasium():
    try:
        import gymnasium as gym  # noqa: F401

        return True
    except Exception:
        return False


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


gymnasium_available = pytest.mark.skipif(
    not _maybe_import_gymnasium(),
    reason="gymnasium not installed",
)
torch_available = pytest.mark.skipif(
    not _torch_available(),
    reason="PyTorch required for batched VIC replay",
)
requires_fr3 = pytest.mark.skipif(
    not fr3_assets_available(),
    reason="Requires bundled assets/fr3 and usd-core",
)


@gymnasium_available
@torch_available
@requires_fr3
def test_grid_viz_pipeline_prefers_gt_over_far_candidate(tmp_path: Path):
    # Tiny deterministic dataset.
    num_structures = 1
    num_directions = 1
    num_envs = num_structures * num_directions
    seed = 123
    config = QuasiStaticStepConfig(
        movement_per_step_m=0.02,
        total_movement_m=0.04,
        move_speed_mps=0.2,
        hold_duration_s=0.1,
        skip_return=True,
    )

    from apple_pick_gym.batched_envs import ApplePickBatchedSysIdEnv
    from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
        BatchedHeterogeneousCoupledSimConfig,
        ControllerConfig,
        ObsConfig,
        RobotConfig,
        SceneSettleCollisionConfig,
    )

    sim_config = dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=num_envs),
        robot=RobotConfig(
            kind="fr3",
            step_mode="coupled",
            fix_to_apple=True,
            skip_ik_bootstrap=False,
            defer_template_robot_bootstrap=False,
            force_batched_layout=True,
        ),
        scene=SceneSettleCollisionConfig(settle_substeps=200),
        controller=ControllerConfig(mode="vic", linear_speed=1.0, angular_speed=1.0),
        obs=ObsConfig(allocate_buffers=True),
    )

    per_env_params = sample_and_broadcast_structure_params(
        RANGES_FIXTURE,
        topology_seed=seed,
        num_structures=num_structures,
        num_directions=num_directions,
    )

    dataset_dir = tmp_path / "batched_gt"
    env = ApplePickBatchedSysIdEnv(
        num_envs=num_envs,
        max_episode_steps=120,
        ranges_path=RANGES_FIXTURE,
        topology_seed=seed,
        use_settle_cache=False,
        sim_config=sim_config,
        per_env_params=per_env_params,
        control_hz=float(config.control_hz),
    )
    try:
        collect_batched_quasi_static_dataset(
            env,
            num_structures=num_structures,
            num_directions=num_directions,
            config=config,
            output_dir=dataset_dir,
            seed=seed,
            ranges_path=RANGES_FIXTURE,
            max_steps=30,
        )
    finally:
        env.close()

    dataset = BatchedSysIdDataset(dataset_dir)
    structure_idx = 0
    gt = gt_bend_stiffness_candidate_from_structure(dataset, structure_idx)

    # Construct a "far" candidate by scaling primary stiffness by 10×.
    far = BendStiffnessCandidate(
        primary=float(gt.primary) * 10.0 if float(gt.primary) != 0.0 else 1.0,
        secondary=float(gt.secondary),
        spur=float(gt.spur),
        stem=float(gt.stem),
    )
    candidates = [gt, far]
    num_candidates = len(candidates)

    recorded_eps = load_recorded_episodes_for_structure(
        dataset,
        structure_idx=structure_idx,
        num_directions=num_directions,
    )

    collectors = replay_candidates_for_structure(
        dataset=dataset,
        structure_idx=structure_idx,
        candidates=candidates,
        num_directions=num_directions,
        seed=None,
        build_env_fn=lambda **kw: _build_env_for_replay(
            kw=kw,
            base_sim_config=sim_config,
            control_hz=float(config.control_hz),
            topology_seed=seed,
        ),
    )

    replay_eps_by_candidate = []
    for cand_idx in range(num_candidates):
        replay_eps_by_candidate.append(
            direction_episodes_from_collectors(
                collectors,
                candidate_index=cand_idx,
                num_directions=num_directions,
            )
        )

    rows = build_grid_viz_rows(
        structure_idx=structure_idx,
        candidates=candidates,
        gt_candidate=gt,
        recorded_eps=recorded_eps,
        replay_eps_by_candidate=replay_eps_by_candidate,
        hold_phase_value=1,
        pos_weights=(1.0, 1.0),
        dist_keys=("primary", "spur", "stem"),
    )

    rep = summarize_structure(
        structure_idx=structure_idx,
        rows=rows,
        metrics=("err_pos_hold", "err_force_hold", "err_torque_hold"),
    )

    # Primary acceptance: GT should win on pose error during hold for this sim-to-sim setup.
    pose = next(s for s in rep.summaries if s.metric == "err_pos_hold")
    assert pose.best_is_gt is True

    # Also expect the far candidate to be non-zero distance.
    assert float(rows[1].dist_log_gt) > 0.0


def _build_env_for_replay(
    *,
    kw: dict,
    base_sim_config,
    control_hz: float,
    topology_seed: int,
):
    """Match the `example_batched_sysid_mmd_grid.py` build_env_fn signature."""
    from apple_pick_gym.batched_envs import ApplePickBatchedSysIdEnv

    gripper = kw.pop("gripper", None)
    num_envs = int(kw.get("num_envs", 1))
    sim_cfg = base_sim_config
    if gripper is not None:
        sim_cfg = dataclasses.replace(
            sim_cfg,
            robot=dataclasses.replace(sim_cfg.robot, gripper=gripper),
        )
    # Ensure runtime.num_envs matches the replay request.
    sim_cfg = dataclasses.replace(
        sim_cfg,
        runtime=dataclasses.replace(sim_cfg.runtime, num_envs=int(num_envs)),
    )
    return ApplePickBatchedSysIdEnv(
        ranges_path=RANGES_FIXTURE,
        topology_seed=int(topology_seed),
        use_settle_cache=False,
        sim_config=sim_cfg,
        control_hz=float(control_hz),
        **kw,
    )

