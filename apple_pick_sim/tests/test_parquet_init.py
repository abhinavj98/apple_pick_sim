"""Tests for observation-only Parquet replay initialization."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from apple_pick_sim.system_id import EpisodeMeta, TrajectoryDataset, TrajectoryWriter
from apple_pick_sim.tests.conftest import RANGES_FIXTURE


class _FakeArray:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def numpy(self):
        return self.value.copy()

    def assign(self, value):
        self.value = np.asarray(value, dtype=np.float32).copy()


def _write_topological_episode(
    tmp_path: Path,
    *,
    fixture_path: Path,
    fruiting_system_params: str | None = None,
) -> str:
    episode_id = "obs-init-no-apple"
    junction_names = [
        "primary_secondary",
        "secondary_spur",
        "spur_stem",
        "stem_apple",
    ]
    woody_start = {
        "primary_secondary": np.array([0.0, 0.2, 1.2], dtype=np.float32),
        "secondary_spur": np.array([0.0, 0.2, 1.05], dtype=np.float32),
        "spur_stem": np.array([0.0, 0.2, 0.9], dtype=np.float32),
        "stem_apple": np.array([0.0, 0.2, 0.75], dtype=np.float32),
    }
    woody_end = {
        "primary_secondary": np.array([0.0, 0.2, 1.15], dtype=np.float32),
        "secondary_spur": np.array([0.0, 0.2, 1.0], dtype=np.float32),
        "spur_stem": np.array([0.0, 0.2, 0.85], dtype=np.float32),
        "stem_apple": np.array([0.0, 0.2, 0.7], dtype=np.float32),
    }

    writer = TrajectoryWriter(episode_id=episode_id)
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
            "woody_part_start_pos": woody_start,
            "woody_part_end_pos": woody_end,
            "ft_wrist": np.zeros(6, dtype=np.float32),
            "tcp_pos": np.zeros(3, dtype=np.float32),
            "apple_pos": np.zeros(3, dtype=np.float32),
            "tcp_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "apple_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot_joint_q": np.zeros(7, dtype=np.float32),
            "woody_part_force": np.zeros(24, dtype=np.float32),
        },
    )
    writer.save(
        tmp_path,
        EpisodeMeta(
            episode_id=episode_id,
            weld_direction=(0.0, 0.0, 1.0),
            excitation_type="quasi_static",
            n_woody_parts=len(junction_names),
            junction_names=junction_names,
            params_fingerprint=json.dumps({"fixture": "no-apple"}),
            fruiting_system_params=fruiting_system_params,
            control_hz=60.0,
            n_directions=1,
            fixture_path=str(fixture_path),
            fruiting_base_pos=(0.0, 0.2, 1.3),
            apple_radius=None,
            rod_radii={"primary": 0.012, "secondary": 0.01, "spur": 0.008, "stem": 0.004},
            weld_reference_pos=(0.0, 0.2, 0.75),
            weld_reference_quat=(0.0, 0.0, 0.0, 1.0),
            skip_return=True,
        ),
    )
    return episode_id


def test_observation_reset_options_prefers_serialized_fruiting_params(
    tmp_path: Path, monkeypatch
):
    """Exact sampled params in metadata should bypass seed/range inference."""
    import apple_pick_sim.fruiting_system as fs
    import apple_pick_sim.system_id.parquet_init as parquet_init

    exact = fs.sample_params(fs.load_ranges(RANGES_FIXTURE), seed=7)
    episode_id = _write_topological_episode(
        tmp_path,
        fixture_path=RANGES_FIXTURE,
        fruiting_system_params=fs.fruiting_params_to_json(exact),
    )

    def _unexpected_inference(*_args, **_kwargs):
        raise AssertionError("serialized params should be preferred")

    monkeypatch.setattr(parquet_init, "infer_params_from_obs", _unexpected_inference)

    options = parquet_init.observation_reset_options_from_parquet(
        TrajectoryDataset(tmp_path),
        episode_id,
        base_ranges_path=RANGES_FIXTURE,
    )

    assert dataclasses.asdict(options["params"]) == dataclasses.asdict(exact)


def test_observation_reset_options_skip_params_without_apple_body(tmp_path: Path, monkeypatch):
    """Replay defaults fix to apple, so apple-less inferred params must not be injected."""
    import apple_pick_sim.system_id.parquet_init as parquet_init
    from apple_pick_sim.digital_twin.from_obs import params_from_ranges_median
    from apple_pick_sim.fruiting_system.params import load_ranges

    appleless_params = params_from_ranges_median(load_ranges(RANGES_FIXTURE))
    appleless_params.apple_radius = None
    appleless_params.apple_density = None
    monkeypatch.setattr(
        parquet_init,
        "infer_params_from_obs",
        lambda obs, base_ranges_path: appleless_params,
    )
    episode_id = _write_topological_episode(tmp_path, fixture_path=RANGES_FIXTURE)

    options = parquet_init.observation_reset_options_from_parquet(
        TrajectoryDataset(tmp_path),
        episode_id,
        base_ranges_path=RANGES_FIXTURE,
    )

    assert "params" not in options
    assert options["ranges_path"] == RANGES_FIXTURE
    assert options["fruiting_base_pos"] == (0.0, 0.2, 1.3)
    assert options["weld_direction"] == (0.0, 0.0, 1.0)
    assert options["weld_reference_pos"] == (0.0, 0.2, 0.75)
    assert options["weld_reference_quat"] == (0.0, 0.0, 0.0, 1.0)


def test_initialize_env_from_parquet_syncs_robot_buffers_and_vic_targets(
    tmp_path: Path, monkeypatch
):
    """Frame-0 robot observations should leave MuJoCo/VIC buffers consistent."""
    import apple_pick_sim.system_id.parquet_init as parquet_init
    from apple_pick_sim.system_id.trajectory_store import target_tf_from_array, target_tf_to_array

    episode_id = "obs-init-recorded-tcp"
    recorded_tcp_pos = np.array([0.12, -0.03, 0.88], dtype=np.float32)
    recorded_tcp_quat = np.array([0.0, 0.2, 0.0, 0.98], dtype=np.float32)
    recorded_joint_q = np.linspace(0.1, 0.7, 7, dtype=np.float32)
    writer = TrajectoryWriter(episode_id=episode_id)
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
            "woody_part_start_pos": {"joint_0": np.zeros(3, dtype=np.float32)},
            "woody_part_end_pos": {"joint_0": np.zeros(3, dtype=np.float32)},
            "ft_wrist": np.zeros(6, dtype=np.float32),
            "tcp_pos": recorded_tcp_pos,
            "apple_pos": np.zeros(3, dtype=np.float32),
            "tcp_quat": recorded_tcp_quat,
            "apple_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot_joint_q": recorded_joint_q,
            "woody_part_force": np.zeros(6, dtype=np.float32),
        },
    )
    writer.save(
        tmp_path,
        EpisodeMeta(
            episode_id=episode_id,
            weld_direction=(0.0, 0.0, 1.0),
            excitation_type="quasi_static",
            n_woody_parts=1,
            junction_names=["joint_0"],
            params_fingerprint=json.dumps({"fixture": "fake"}),
            control_hz=60.0,
            n_directions=1,
            fruiting_base_pos=(0.0, 0.2, 1.3),
            skip_return=True,
        ),
    )

    fk_target = target_tf_from_array(
        np.array([9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )
    controller = SimpleNamespace(
        target_tf=fk_target,
        sync_target_from_state=lambda _state, _tcp_body_index: None,
    )
    scene = SimpleNamespace(
        robot_state_0=SimpleNamespace(
            joint_q=_FakeArray(np.zeros(7, dtype=np.float32)),
            joint_qd=_FakeArray(np.ones(7, dtype=np.float32)),
        ),
        robot_model=SimpleNamespace(
            joint_q=_FakeArray(np.zeros(7, dtype=np.float32)),
            joint_qd=_FakeArray(np.ones(7, dtype=np.float32)),
        ),
        robot_state_1=SimpleNamespace(),
        mj_solver=SimpleNamespace(),
        robot_control=SimpleNamespace(),
        tcp_body_index=0,
        vic_target_tf=fk_target,
        vic_target_twist=object(),
        vic_jt_default_dof_pos=_FakeArray(np.ones(7, dtype=np.float32)),
    )
    env = SimpleNamespace(_scene=scene, _controller=controller)
    import newton

    monkeypatch.setattr(newton, "eval_fk", lambda *_args, **_kwargs: None)
    buffer_calls = []
    hold_calls = []
    twist_sentinel = object()
    monkeypatch.setattr(
        parquet_init,
        "init_robot_mujoco_step_buffers",
        lambda called_scene: buffer_calls.append(called_scene),
        raising=False,
    )
    monkeypatch.setattr(
        parquet_init,
        "fr3_robot",
        SimpleNamespace(
            EEVelocity=lambda: twist_sentinel,
            hold_mujoco_actuator_targets_at_state=lambda *args: hold_calls.append(args),
        ),
        raising=False,
    )

    parquet_init.initialize_env_from_parquet(env, TrajectoryDataset(tmp_path), episode_id)

    expected_tf = np.concatenate([recorded_tcp_pos, recorded_tcp_quat]).astype(np.float32)
    np.testing.assert_allclose(target_tf_to_array(scene.vic_target_tf), expected_tf, atol=1e-6)
    np.testing.assert_allclose(target_tf_to_array(controller.target_tf), expected_tf, atol=1e-6)
    np.testing.assert_allclose(scene.robot_state_0.joint_q.numpy(), recorded_joint_q)
    np.testing.assert_allclose(scene.robot_model.joint_q.numpy(), recorded_joint_q)
    assert buffer_calls == [scene]
    assert hold_calls == [(scene.robot_model, scene.robot_state_0, scene.robot_control)]
    expected_default_q = recorded_joint_q.copy()
    expected_default_q[6] = 0.0
    np.testing.assert_allclose(scene.vic_jt_default_dof_pos.numpy(), expected_default_q)
    assert scene.vic_target_twist is twist_sentinel


def test_initialize_env_from_parquet_prefers_metadata_reset_state(
    tmp_path: Path, monkeypatch
):
    """Reset metadata must win over frame 0, which is already after action 0."""
    import apple_pick_sim.system_id.parquet_init as parquet_init
    from apple_pick_sim.system_id.trajectory_store import target_tf_from_array, target_tf_to_array

    episode_id = "obs-init-reset-metadata"
    reset_tcp_pos = np.array([0.12, -0.03, 0.88], dtype=np.float32)
    reset_tcp_quat = np.array([0.0, 0.2, 0.0, 0.98], dtype=np.float32)
    reset_joint_q = np.linspace(0.1, 0.7, 7, dtype=np.float32)
    post_step_tcp_pos = np.array([0.99, 0.98, 0.97], dtype=np.float32)
    post_step_tcp_quat = np.array([0.3, 0.0, 0.0, 0.95], dtype=np.float32)
    post_step_joint_q = np.linspace(1.1, 1.7, 7, dtype=np.float32)

    writer = TrajectoryWriter(episode_id=episode_id)
    writer.record_step(
        step_idx=0,
        sim_time=1.0 / 60.0,
        phase="move_out",
        dir_idx=0,
        amplitude_m=0.01,
        action=np.ones(6, dtype=np.float32),
        obs={
            "excitation_type": 0,
            "excitation_direction": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "tcp_velocity": np.zeros(6, dtype=np.float32),
            "woody_part_start_pos": {"joint_0": np.zeros(3, dtype=np.float32)},
            "woody_part_end_pos": {"joint_0": np.zeros(3, dtype=np.float32)},
            "ft_wrist": np.zeros(6, dtype=np.float32),
            "tcp_pos": post_step_tcp_pos,
            "apple_pos": np.zeros(3, dtype=np.float32),
            "tcp_quat": post_step_tcp_quat,
            "apple_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot_joint_q": post_step_joint_q,
            "woody_part_force": np.zeros(6, dtype=np.float32),
        },
    )
    writer.save(
        tmp_path,
        EpisodeMeta(
            episode_id=episode_id,
            weld_direction=(0.0, 0.0, 1.0),
            excitation_type="quasi_static",
            n_woody_parts=1,
            junction_names=["joint_0"],
            params_fingerprint=json.dumps({"fixture": "fake"}),
            control_hz=60.0,
            n_directions=1,
            fruiting_base_pos=(0.0, 0.2, 1.3),
            initial_tcp_pos=tuple(float(x) for x in reset_tcp_pos),
            initial_tcp_quat=tuple(float(x) for x in reset_tcp_quat),
            initial_robot_joint_q=tuple(float(x) for x in reset_joint_q),
            skip_return=True,
        ),
    )

    fk_target = target_tf_from_array(
        np.array([9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )
    controller = SimpleNamespace(
        target_tf=fk_target,
        sync_target_from_state=lambda _state, _tcp_body_index: None,
    )
    scene = SimpleNamespace(
        robot_state_0=SimpleNamespace(
            joint_q=_FakeArray(np.zeros(7, dtype=np.float32)),
            joint_qd=_FakeArray(np.ones(7, dtype=np.float32)),
        ),
        robot_model=SimpleNamespace(
            joint_q=_FakeArray(np.zeros(7, dtype=np.float32)),
            joint_qd=_FakeArray(np.ones(7, dtype=np.float32)),
        ),
        robot_state_1=SimpleNamespace(),
        mj_solver=SimpleNamespace(),
        robot_control=SimpleNamespace(),
        tcp_body_index=0,
        vic_target_tf=fk_target,
        vic_target_twist=object(),
        vic_jt_default_dof_pos=_FakeArray(np.ones(7, dtype=np.float32)),
    )
    env = SimpleNamespace(_scene=scene, _controller=controller)
    import newton

    monkeypatch.setattr(newton, "eval_fk", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        parquet_init,
        "init_robot_mujoco_step_buffers",
        lambda _called_scene: None,
        raising=False,
    )
    monkeypatch.setattr(
        parquet_init,
        "fr3_robot",
        SimpleNamespace(
            EEVelocity=lambda: object(),
            hold_mujoco_actuator_targets_at_state=lambda *_args: None,
        ),
        raising=False,
    )

    parquet_init.initialize_env_from_parquet(env, TrajectoryDataset(tmp_path), episode_id)

    expected_tf = np.concatenate([reset_tcp_pos, reset_tcp_quat]).astype(np.float32)
    np.testing.assert_allclose(target_tf_to_array(scene.vic_target_tf), expected_tf, atol=1e-6)
    np.testing.assert_allclose(scene.robot_state_0.joint_q.numpy(), reset_joint_q)
    np.testing.assert_allclose(scene.robot_model.joint_q.numpy(), reset_joint_q)
