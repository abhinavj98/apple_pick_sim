"""Physics gate: export real parquet → replay_batched_sysid_structure moves TCP.

Uses short free settle and post_grasp_settle_substeps=0 for CI speed. Full
settle-viewer defaults live on ``robot_replay/example_replay_real_batched.py``.

Rebuild uses converted episode metadata (fruiting_base_pos + oracle params) and
fixture ``sim_build`` support-joint knobs on ``gym_defaults`` (not CPU
``test_minimal``), matching ``example_view_pre_grasp_settle`` pose + stable settle.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apple_pick_gym.batched_envs import ApplePickBatchedSysIdEnv
from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
    gt_bend_stiffness_candidate_from_structure,
    replay_batched_sysid_structure,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ObsConfig,
)
from apple_pick_sim.fruiting_system.params import GripperProxyConfig, load_ranges, parse_sim_build
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import ImpedanceGains
from apple_pick_sim.system_id import BatchedSysIdDataset
from apple_pick_sim.tests.conftest import fr3_assets_available

_SEED = 0
_MAX_REPLAY_FRAMES = 24
_SETTLE_SUBSTEPS = 80
_POST_GRASP_SETTLE_SUBSTEPS = 0
_VARIANCE = Path(
    "apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json"
)
_REAL_SRC = Path("robot_replay/s02-d00_action.parquet")


def _maybe_import_gymnasium() -> bool:
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


def _fruiting_base_pos_from_meta(meta: dict[str, Any]) -> tuple[float, float, float]:
    arr = np.asarray(meta["fruiting_base_pos"], dtype=np.float64).reshape(3)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _test_sim_config(
    *,
    num_envs: int,
    fruiting_base_pos: tuple[float, float, float],
    ranges: dict,
    bootstrap_joint_q: tuple[float, ...] | None = None,
) -> BatchedHeterogeneousCoupledSimConfig:
    gym_cfg = BatchedHeterogeneousCoupledSimConfig.gym_defaults(num_envs=num_envs)
    sb = parse_sim_build(ranges)
    fruiting = gym_cfg.fruiting_system
    controller = dataclasses.replace(
        gym_cfg.controller, mode="vic", linear_speed=1.0, angular_speed=1.0
    )
    if sb is not None:
        fruiting = dataclasses.replace(
            fruiting,
            joint_angular_kd_overrides=dict(sb.joint_angular_kd_overrides),
            joint_linear_kd_overrides=dict(sb.joint_linear_kd_overrides),
            joint_angular_kp_overrides=dict(sb.joint_angular_kp_overrides),
            joint_linear_kp_overrides=dict(sb.joint_linear_kp_overrides),
            joint_damping_ratio=sb.joint_damping_ratio,
        )
        controller = dataclasses.replace(
            controller,
            vic_gains=ImpedanceGains(
                linear_k=sb.vic_gains.linear_k,
                linear_d=sb.vic_gains.linear_d,
                angular_k=sb.vic_gains.angular_k,
                angular_d=sb.vic_gains.angular_d,
            ),
        )
    return dataclasses.replace(
        gym_cfg,
        robot=dataclasses.replace(
            gym_cfg.robot,
            kind="fr3",
            step_mode="coupled",
            fix_to_apple=True,
            skip_ik_bootstrap=True,
            defer_template_robot_bootstrap=True,
            force_batched_layout=True,
            robot_base_pos=(0.0, 0.0, 0.0),
            per_env_ik=False,
            bootstrap_joint_q=bootstrap_joint_q,
        ),
        scene=dataclasses.replace(
            gym_cfg.scene,
            settle_substeps=_SETTLE_SUBSTEPS,
            post_grasp_settle_substeps=_POST_GRASP_SETTLE_SUBSTEPS,
            fruiting_base_pos=fruiting_base_pos,
        ),
        controller=controller,
        fruiting_system=fruiting,
        domain_randomization=dataclasses.replace(
            gym_cfg.domain_randomization,
            topology_seed=_SEED,
        ),
        obs=ObsConfig(allocate_buffers=True),
    )


def _build_env_fn(
    *,
    ranges_path: Path,
    ranges: dict,
    fruiting_base_pos: tuple[float, float, float],
    bootstrap_joint_q: tuple[float, ...] | None = None,
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
        sim_config = _test_sim_config(
            num_envs=num_envs,
            fruiting_base_pos=fruiting_base_pos,
            ranges=ranges,
            bootstrap_joint_q=bootstrap_joint_q,
        )
        if gripper is not None:
            sim_config = dataclasses.replace(
                sim_config,
                robot=dataclasses.replace(sim_config.robot, gripper=gripper),
            )
        return ApplePickBatchedSysIdEnv(
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            ranges_path=ranges_path,
            topology_seed=_SEED,
            use_settle_cache=False,
            sim_config=sim_config,
            per_env_params=per_env_params,
            per_env_grippers=per_env_grippers,
        )

    return build_env_fn


@pytest.mark.slow
@gymnasium_available
@requires_fr3
def test_real_exported_s02_replay_moves_tcp(tmp_path: Path):
    """Exported real episode must drive FR3: TCP not stationary under recorded actions."""
    if not _REAL_SRC.is_file():
        pytest.skip(f"missing {_REAL_SRC}")
    if not _VARIANCE.is_file():
        pytest.skip(f"missing {_VARIANCE}")

    from apple_pick_sim.system_id.real_to_batched_sysid import (
        export_real_episode_to_batched_dataset,
    )

    out = tmp_path / "real_batched"
    export_real_episode_to_batched_dataset(
        _REAL_SRC,
        fixture_path=_VARIANCE,
        output_dir=out,
        overwrite=True,
    )
    dataset = BatchedSysIdDataset(out)
    ranges = load_ranges(_VARIANCE)
    meta = dataset.load_episode_metadata(0, 0)
    fruiting_base_pos = _fruiting_base_pos_from_meta(meta)
    bootstrap_joint_q = tuple(
        float(x) for x in np.asarray(meta["initial_robot_joint_q"], dtype=np.float64).reshape(-1)
    )

    structure_idx = 0
    candidates = [gt_bend_stiffness_candidate_from_structure(dataset, structure_idx)]

    def on_step(*, frame_idx: int, env: Any) -> bool:
        del env
        return int(frame_idx) + 1 < _MAX_REPLAY_FRAMES

    collectors = replay_batched_sysid_structure(
        dataset=dataset,
        structure_idx=structure_idx,
        candidates=candidates,
        num_directions=1,
        seed=_SEED,
        build_env_fn=_build_env_fn(
            ranges_path=_VARIANCE,
            ranges=ranges,
            fruiting_base_pos=fruiting_base_pos,
            bootstrap_joint_q=bootstrap_joint_q,
        ),
        replay_sim_config=_test_sim_config(
            num_envs=1,
            fruiting_base_pos=fruiting_base_pos,
            ranges=ranges,
            bootstrap_joint_q=bootstrap_joint_q,
        ),
        on_step=on_step,
        use_oracle_params=True,
    )
    tcp = np.asarray(collectors.to_arrays(0)["tcp_pos"], dtype=np.float64)
    assert tcp.ndim == 2 and tcp.shape[1] == 3
    assert tcp.shape[0] >= 2
    motion_m = float(np.linalg.norm(tcp[-1] - tcp[0]))
    assert motion_m > 1e-4, f"TCP stationary during real replay (motion={motion_m} m)"
