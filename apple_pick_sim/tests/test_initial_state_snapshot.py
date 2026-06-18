"""Tests for post-warmup initial-state snapshot save/load and replay injection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apple_pick_sim.system_id import EpisodeMeta, TrajectoryWriter
from apple_pick_sim.system_id.trajectory_store import (
    INITIAL_STATE_KEYS,
    TrajectoryDataset,
    grasp_snapshot_from_env,
    target_tf_from_array,
    target_tf_to_array,
)
from apple_pick_sim.tests.conftest import fr3_assets_available


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


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


torch_available = pytest.mark.skipif(
    not _torch_available(),
    reason="PyTorch required for VIC joint torques (uv sync --extra vic)",
)


def _synthetic_snapshot() -> dict[str, np.ndarray]:
    return {
        "robot_body_q": np.arange(14, dtype=np.float32).reshape(2, 7),
        "robot_body_qd": np.arange(12, dtype=np.float32).reshape(2, 6),
        "robot_joint_q": np.arange(7, dtype=np.float32),
        "robot_joint_qd": np.arange(7, dtype=np.float32),
        "cable_body_q": np.arange(21, dtype=np.float32).reshape(3, 7),
        "cable_body_qd": np.arange(18, dtype=np.float32).reshape(3, 6),
        "vic_target_tf": np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }


def test_save_load_initial_state(tmp_path: Path):
    writer = TrajectoryWriter(episode_id="snap-ep-1")
    snapshot = _synthetic_snapshot()
    writer.save_initial_state(tmp_path, snapshot)
    _record_minimal_episode(writer, tmp_path)

    dataset = TrajectoryDataset(tmp_path)
    loaded = dataset.load_initial_state("snap-ep-1")
    assert loaded is not None
    for key in INITIAL_STATE_KEYS:
        np.testing.assert_allclose(loaded[key], snapshot[key])


def test_load_initial_state_missing_returns_none(tmp_path: Path):
    writer = TrajectoryWriter(episode_id="snap-ep-1")
    _record_minimal_episode(writer, tmp_path)
    dataset = TrajectoryDataset(tmp_path)
    assert dataset.load_initial_state("snap-ep-1") is None
    assert dataset.load_initial_state("missing") is None


def test_weld_stored_in_snapshot():
    weld = (0.1, 0.2, 0.97)
    snapshot = _synthetic_snapshot()
    snapshot["weld_direction"] = np.asarray(weld, dtype=np.float32)
    np.testing.assert_allclose(snapshot["weld_direction"], np.asarray(weld, dtype=np.float32))


def test_grasp_snapshot_from_env_stores_weld_direction():
    class _FakeController:
        target_tf = None

    class _FakeBodyArray:
        def numpy(self):
            return np.zeros((1, 7), dtype=np.float32)

    class _FakeState:
        body_q = _FakeBodyArray()
        body_qd = _FakeBodyArray()

    class _FakeJointArray:
        def numpy(self):
            return np.zeros(7, dtype=np.float32)

    class _FakeCable:
        state_0 = _FakeState()

    class _FakeScene:
        robot_state_0 = type(
            "RS",
            (),
            {
                "body_q": _FakeBodyArray(),
                "body_qd": _FakeBodyArray(),
                "joint_q": _FakeJointArray(),
                "joint_qd": _FakeJointArray(),
            },
        )()
        cable = _FakeCable()

    import warp as wp

    env = type(
        "Env",
        (),
        {
            "_scene": _FakeScene(),
            "_controller": type(
                "C",
                (),
                {
                    "target_tf": wp.transform(
                        wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0)
                    )
                },
            )(),
        },
    )()
    weld = (-0.935, -0.333, 0.123)
    snapshot = grasp_snapshot_from_env(env, weld_direction=weld)
    np.testing.assert_allclose(snapshot["weld_direction"], np.asarray(weld, dtype=np.float32))


def test_target_tf_roundtrip():
    import warp as wp

    target = wp.transform(wp.vec3(0.1, 0.2, 0.3), wp.quat(0.0, 0.0, 0.0, 1.0))
    arr = target_tf_to_array(target)
    restored = target_tf_from_array(arr)
    pos0 = wp.transform_get_translation(target)
    pos1 = wp.transform_get_translation(restored)
    np.testing.assert_allclose([pos0[0], pos0[1], pos0[2]], [pos1[0], pos1[1], pos1[2]], atol=1e-6)


def _record_minimal_episode(writer: TrajectoryWriter, tmp_path: Path) -> None:
    junction_names = ["joint_0", "joint_1"]
    writer.record_step(
        step_idx=0,
        sim_time=0.0,
        phase="hold",
        dir_idx=0,
        amplitude_m=0.0,
        action=np.zeros(6, dtype=np.float32),
        obs={
            "excitation_type": 0,
            "excitation_direction": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "tcp_velocity": np.zeros(6, dtype=np.float32),
            "woody_part_start_pos": {
                name: np.zeros(3, dtype=np.float32) for name in junction_names
            },
            "woody_part_end_pos": {
                name: np.zeros(3, dtype=np.float32) for name in junction_names
            },
            "ft_wrist": np.zeros(6, dtype=np.float32),
            "tcp_pos": np.zeros(3, dtype=np.float32),
            "apple_pos": np.zeros(3, dtype=np.float32),
            "woody_part_force": np.zeros(12, dtype=np.float32),
        },
    )
    writer.save(
        tmp_path,
        EpisodeMeta(
            episode_id=writer.episode_id,
            weld_direction=(0.0, 0.0, 1.0),
            excitation_type="quasi_static",
            n_woody_parts=2,
            junction_names=junction_names,
            params_fingerprint=json.dumps({"stem_bend_stiffness": 30.0}),
            control_hz=60.0,
            seed=3,
            n_directions=1,
            skip_return=True,
        ),
    )


@gymnasium_available
@torch_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_replay_reset_matches_collection(tmp_path: Path):
    from apple_pick_gym.envs import ApplePickReplayEnv, ApplePickSysIdEnv

    seed = 0
    warmup = 60
    step_action = np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0], dtype=np.float32)

    collect_env = ApplePickSysIdEnv(
        max_episode_steps=8,
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=warmup,
        robot_facing_weld=True,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    collect_obs, collect_info = collect_env.reset(seed=seed)
    assert collect_env._scene is not None
    reset_weld = collect_info.get("weld_direction")

    writer = TrajectoryWriter(episode_id="snap-replay-ep")
    snapshot = grasp_snapshot_from_env(
        collect_env,
        obs=collect_obs,
        weld_direction=reset_weld,
    )
    writer.save_initial_state(tmp_path, snapshot)

    collect_obs_after, _, _, _, collect_step_info = collect_env.step(step_action)

    collect_obs_at_reset = {
        "apple_pos": np.asarray(collect_obs["apple_pos"], dtype=np.float64),
        "tcp_pos": np.asarray(collect_obs["tcp_pos"], dtype=np.float64),
        "ft_wrist": np.asarray(collect_obs["ft_wrist"], dtype=np.float64),
    }

    writer.record_step(
        step_idx=0,
        sim_time=1.0 / 60.0,
        phase="move_out",
        dir_idx=0,
        amplitude_m=0.05,
        action=step_action,
        obs=collect_obs_after,
    )
    writer.save(
        tmp_path,
        EpisodeMeta(
            episode_id=writer.episode_id,
            weld_direction=tuple(float(x) for x in collect_info.get("weld_direction", [0, 0, 1])),
            excitation_type="quasi_static",
            n_woody_parts=int(collect_info.get("n_woody_parts", 0)),
            junction_names=list(collect_env.unwrapped.junction_names),
            params_fingerprint=json.dumps(collect_info.get("params_fingerprint", {}), sort_keys=True),
            control_hz=60.0,
            seed=seed,
            n_directions=1,
            skip_return=True,
        ),
    )
    collect_env.close()

    replay_env = ApplePickReplayEnv(
        max_episode_steps=8,
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=warmup,
        robot_facing_weld=True,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    replay_env.load_dataset(tmp_path, episode_id=writer.episode_id)
    replay_obs, replay_info = replay_env.reset(seed=seed)

    assert replay_info.get("initial_state_restored") is False

    apple_err_mm = (
        1000.0
        * float(
            np.linalg.norm(
                np.asarray(replay_obs["apple_pos"], dtype=np.float64)
                - collect_obs_at_reset["apple_pos"]
            )
        )
    )
    tcp_err_mm = (
        1000.0
        * float(
            np.linalg.norm(
                np.asarray(replay_obs["tcp_pos"], dtype=np.float64)
                - collect_obs_at_reset["tcp_pos"]
            )
        )
    )
    ft_norm = float(np.linalg.norm(np.asarray(replay_obs["ft_wrist"], dtype=np.float64)[:3]))

    assert apple_err_mm < 1.0, f"apple_pos error {apple_err_mm:.2f} mm"
    assert tcp_err_mm < 1.0, f"tcp_pos error {tcp_err_mm:.2f} mm"
    assert ft_norm < 0.1, f"|F| at reset {ft_norm:.3f} N"

    replay_step_obs, _, _, _, replay_step_info = replay_env.step(step_action)

    step_ft_err = float(
        np.linalg.norm(
            np.asarray(replay_step_obs["ft_wrist"], dtype=np.float64)[:3]
            - np.asarray(collect_obs_after["ft_wrist"], dtype=np.float64)[:3]
        )
    )
    step_tcp_err_mm = (
        1000.0
        * float(
            np.linalg.norm(
                np.asarray(replay_step_obs["tcp_pos"], dtype=np.float64)
                - np.asarray(collect_obs_after["tcp_pos"], dtype=np.float64)
            )
        )
    )
    assert step_ft_err < 5.0, f"step-1 |ΔF| {step_ft_err:.2f} N"
    assert step_tcp_err_mm < 2.0, f"step-1 |Δtcp| {step_tcp_err_mm:.2f} mm"
    assert "weld_direction" in collect_step_info
    assert "weld_direction" in replay_step_info

    replay_env.close()


@gymnasium_available
@torch_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_replay_snapshot_restore_reseeds_vic_defaults(tmp_path: Path):
    from apple_pick_gym.envs import ApplePickReplayEnv, ApplePickSysIdEnv

    seed = 0
    warmup = 60

    collect_env = ApplePickSysIdEnv(
        max_episode_steps=8,
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=warmup,
        robot_facing_weld=True,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    collect_obs, collect_info = collect_env.reset(seed=seed)
    writer = TrajectoryWriter(episode_id="snap-vic-default-ep")
    writer.save_initial_state(
        tmp_path,
        grasp_snapshot_from_env(
            collect_env,
            obs=collect_obs,
            weld_direction=collect_info.get("weld_direction"),
        ),
    )
    step_action = np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0], dtype=np.float32)
    collect_obs_after, _, _, _, _ = collect_env.step(step_action)
    writer.record_step(
        step_idx=0,
        sim_time=1.0 / 60.0,
        phase="move_out",
        dir_idx=0,
        amplitude_m=0.05,
        action=step_action,
        obs=collect_obs_after,
    )
    writer.save(
        tmp_path,
        EpisodeMeta(
            episode_id=writer.episode_id,
            weld_direction=tuple(float(x) for x in collect_info.get("weld_direction", [0, 0, 1])),
            excitation_type="quasi_static",
            n_woody_parts=int(collect_info.get("n_woody_parts", 0)),
            junction_names=list(collect_env.unwrapped.junction_names),
            params_fingerprint=json.dumps(collect_info.get("params_fingerprint", {}), sort_keys=True),
            control_hz=60.0,
            seed=seed,
            n_directions=1,
            skip_return=True,
        ),
    )
    collect_env.close()

    replay_env = ApplePickReplayEnv(
        max_episode_steps=8,
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=0,
        robot_facing_weld=True,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    replay_env.load_dataset(tmp_path, episode_id=writer.episode_id)
    replay_obs, replay_info = replay_env.reset(seed=seed)

    assert replay_info.get("initial_state_restored") is True
    np.testing.assert_allclose(replay_obs["tcp_pos"], collect_obs["tcp_pos"], atol=1e-6)

    scene = replay_env._scene
    assert scene is not None
    state_q = scene.robot_state_0.joint_q.numpy().reshape(-1)[:7]
    model_q = scene.robot_model.joint_q.numpy().reshape(-1)[:7]
    control_target = scene.robot_control.joint_target_pos.numpy().reshape(-1)[:7]
    vic_default = scene.vic_jt_default_dof_pos.numpy().reshape(-1)[:7]
    expected_default = state_q.copy()
    expected_default[6] = 0.0

    np.testing.assert_allclose(model_q, state_q, atol=1e-6)
    np.testing.assert_allclose(control_target, state_q, atol=1e-6)
    np.testing.assert_allclose(vic_default, expected_default, atol=1e-6)

    replay_env.close()


@gymnasium_available
@torch_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_replay_snapshot_restore_preserves_cable_previous_pose(tmp_path: Path):
    from apple_pick_gym.envs import ApplePickReplayEnv, ApplePickSysIdEnv

    seed = 0
    warmup = 60

    collect_env = ApplePickSysIdEnv(
        max_episode_steps=8,
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=warmup,
        robot_facing_weld=True,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    collect_obs, collect_info = collect_env.reset(seed=seed)
    collect_prev = collect_env._scene.cable.state_1.body_q.numpy().copy()
    writer = TrajectoryWriter(episode_id="snap-cable-prev-ep")
    writer.save_initial_state(
        tmp_path,
        grasp_snapshot_from_env(
            collect_env,
            obs=collect_obs,
            weld_direction=collect_info.get("weld_direction"),
        ),
    )
    step_action = np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0], dtype=np.float32)
    collect_obs_after, _, _, _, _ = collect_env.step(step_action)
    writer.record_step(
        step_idx=0,
        sim_time=1.0 / 60.0,
        phase="move_out",
        dir_idx=0,
        amplitude_m=0.05,
        action=step_action,
        obs=collect_obs_after,
    )
    writer.save(
        tmp_path,
        EpisodeMeta(
            episode_id=writer.episode_id,
            weld_direction=tuple(float(x) for x in collect_info.get("weld_direction", [0, 0, 1])),
            excitation_type="quasi_static",
            n_woody_parts=int(collect_info.get("n_woody_parts", 0)),
            junction_names=list(collect_env.unwrapped.junction_names),
            params_fingerprint=json.dumps(collect_info.get("params_fingerprint", {}), sort_keys=True),
            control_hz=60.0,
            seed=seed,
            n_directions=1,
            skip_return=True,
        ),
    )
    collect_env.close()

    replay_env = ApplePickReplayEnv(
        max_episode_steps=8,
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=0,
        robot_facing_weld=True,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    replay_env.load_dataset(tmp_path, episode_id=writer.episode_id)
    _, replay_info = replay_env.reset(seed=seed)

    assert replay_info.get("initial_state_restored") is True
    replay_prev = replay_env._scene.cable.state_1.body_q.numpy()
    np.testing.assert_allclose(replay_prev, collect_prev, atol=1e-6)

    replay_env.close()


@gymnasium_available
@torch_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_replay_snapshot_rebuilds_settled_weld_reference(tmp_path: Path):
    from apple_pick_gym.envs import ApplePickReplayEnv, ApplePickSysIdEnv

    seed = 0
    warmup = 60

    collect_env = ApplePickSysIdEnv(
        max_episode_steps=8,
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=warmup,
        robot_facing_weld=True,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    collect_obs, collect_info = collect_env.reset(seed=seed)
    collect_offset = collect_env._scene.cable.gripper_proxy_offset_in_apple_frame
    assert collect_offset is not None

    writer = TrajectoryWriter(episode_id="snap-weld-reference-ep")
    writer.save_initial_state(
        tmp_path,
        grasp_snapshot_from_env(
            collect_env,
            obs=collect_obs,
            weld_direction=collect_info.get("weld_direction"),
        ),
    )
    step_action = np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0], dtype=np.float32)
    collect_obs_after, _, _, _, _ = collect_env.step(step_action)
    writer.record_step(
        step_idx=0,
        sim_time=1.0 / 60.0,
        phase="move_out",
        dir_idx=0,
        amplitude_m=0.05,
        action=step_action,
        obs=collect_obs_after,
    )
    writer.save(
        tmp_path,
        EpisodeMeta(
            episode_id=writer.episode_id,
            weld_direction=tuple(float(x) for x in collect_info.get("weld_direction", [0, 0, 1])),
            excitation_type="quasi_static",
            n_woody_parts=int(collect_info.get("n_woody_parts", 0)),
            junction_names=list(collect_env.unwrapped.junction_names),
            params_fingerprint=json.dumps(collect_info.get("params_fingerprint", {}), sort_keys=True),
            control_hz=60.0,
            seed=seed,
            n_directions=1,
            skip_return=True,
        ),
    )
    collect_env.close()

    replay_env = ApplePickReplayEnv(
        max_episode_steps=8,
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=0,
        robot_facing_weld=True,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    replay_env.load_dataset(tmp_path, episode_id=writer.episode_id)
    _, replay_info = replay_env.reset(seed=seed)

    assert replay_info.get("initial_state_restored") is True
    replay_offset = replay_env._scene.cable.gripper_proxy_offset_in_apple_frame
    assert replay_offset is not None
    np.testing.assert_allclose(replay_offset, collect_offset, atol=1e-6)

    replay_env.close()
