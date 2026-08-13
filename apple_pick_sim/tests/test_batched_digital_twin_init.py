"""Tests for batched digital-twin init helpers."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from apple_pick_gym.batched_envs import ApplePickBatchedSysIdEnv
from apple_pick_gym.batched_envs.batched_sysid_collect import (
    collect_batched_quasi_static_dataset,
    sample_and_broadcast_structure_params,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ControllerConfig,
    ObsConfig,
    RobotConfig,
    SceneSettleCollisionConfig,
)
from apple_pick_sim.fruiting_system.params import FruitingSystemParams
from apple_pick_sim.system_id import BatchedSysIdDataset, QuasiStaticStepConfig
from apple_pick_sim.fruiting_system.params import GripperProxyConfig
from apple_pick_sim.system_id.batched_trajectory_store import (
    FIRST_TRAJECTORY_STEP_IDX,
    frame_index_for_step,
)
from apple_pick_sim.fruiting_system import fruiting_params_from_json
from apple_pick_sim.system_id.batched_digital_twin_init import (
    ReplayEpisodeSource,
    apply_logged_post_grasp_se3_to_cable,
    digital_twin_obs_from_batched_episode,
    gripper_proxy_for_real_batched_replay,
    gripper_proxy_from_episode_metadata,
    infer_base_params_for_structure,
    initialize_batched_env_from_episode_sources,
    initialize_batched_env_from_dataset,
    true_params_for_structure,
)
from apple_pick_sim.system_id.real_post_grasp_plan import proxy_offset_from_apple_and_tcp
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


@dataclasses.dataclass(frozen=True)
class ExpectedSource:
    structure_idx: int
    direction_idx: int


class _FakeArray:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)
        self.assign_calls = 0

    def numpy(self):
        return self.value.copy()

    def assign(self, value):
        self.assign_calls += 1
        self.value = np.asarray(value, dtype=np.float32).copy()


class _FakeLayout:
    @staticmethod
    def joint_q_slice(env_idx: int) -> slice:
        return slice(7 * env_idx, 7 * (env_idx + 1))

    @staticmethod
    def joint_qd_slice(env_idx: int) -> slice:
        return slice(7 * env_idx, 7 * (env_idx + 1))


def _mock_episode_sources_env():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    robot_state = SimpleNamespace(
        joint_q=_FakeArray(np.full(14, -1.0)),
        joint_qd=_FakeArray(np.full(14, -1.0)),
    )
    robot_model = SimpleNamespace(
        joint_q=_FakeArray(np.full(14, -2.0)),
        joint_qd=_FakeArray(np.full(14, -2.0)),
    )
    vic = SimpleNamespace(
        _target_pos_wp=_FakeArray(np.full((2, 3), -3.0)),
        _target_rot_wp=_FakeArray(np.full((2, 4), -4.0)),
        _sync_target_tf_from_device=MagicMock(),
        stage_targets_to_scene=MagicMock(),
    )
    scene = SimpleNamespace(
        robot_state_0=robot_state,
        robot_model=robot_model,
        robot_control=object(),
        vic_controller=vic,
        vic_jt_default_dof_pos=_FakeArray(np.full(7, -5.0)),
        vic_jt_default_dof_pos_batched=_FakeArray(np.full((2, 7), -6.0)),
    )
    env = SimpleNamespace(
        num_envs=2,
        _sim=SimpleNamespace(scene=scene, layout=_FakeLayout()),
        set_excitation_context=MagicMock(),
    )
    return env


def _mock_episode_sources_dataset():
    from unittest.mock import MagicMock

    dataset = MagicMock()

    def arrays(structure_idx: int, direction_idx: int):
        sentinel = 10 * structure_idx + direction_idx
        return {
            "excitation_direction": np.asarray([[sentinel, 1.0, 0.0]], dtype=np.float32),
            "robot_joint_q": np.full((1, 7), sentinel + 100, dtype=np.float32),
            "tcp_pos": np.full((1, 3), sentinel + 200, dtype=np.float32),
            "tcp_quat": np.full((1, 4), sentinel + 300, dtype=np.float32),
        }

    dataset.load_episode_obs_arrays.side_effect = arrays
    dataset.load_episode_metadata.side_effect = lambda structure_idx, direction_idx: {}
    return dataset


def test_initialize_batched_env_from_episode_sources_routes_each_world(monkeypatch):
    import newton

    monkeypatch.setattr(newton, "eval_fk", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "apple_pick_sim.system_id.batched_digital_twin_init.init_robot_mujoco_step_buffers",
        lambda scene: None,
    )
    monkeypatch.setattr(
        "apple_pick_sim.system_id.batched_digital_twin_init.fr3_robot."
        "hold_mujoco_actuator_targets_at_state",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "apple_pick_sim.system_id.batched_digital_twin_init.fr3_robot.EEVelocity",
        lambda: object(),
    )
    env = _mock_episode_sources_env()
    dataset = _mock_episode_sources_dataset()
    sources = (
        ReplayEpisodeSource(structure_idx=0, direction_idx=2),
        ReplayEpisodeSource(structure_idx=3, direction_idx=0),
    )

    initialize_batched_env_from_episode_sources(env, dataset, sources)

    for target in (env._sim.scene.robot_state_0, env._sim.scene.robot_model):
        np.testing.assert_allclose(target.joint_q.value.reshape(2, 7), [[102] * 7, [130] * 7])
        np.testing.assert_allclose(target.joint_qd.value, 0.0)
        assert target.joint_q.assign_calls == 1
        assert target.joint_qd.assign_calls == 1
    np.testing.assert_allclose(
        env._sim.scene.vic_jt_default_dof_pos_batched.value,
        [[102, 102, 102, 102, 102, 102, 0], [130, 130, 130, 130, 130, 130, 0]],
    )
    assert env._sim.scene.vic_jt_default_dof_pos_batched.assign_calls == 1
    assert env._sim.scene.vic_jt_default_dof_pos.assign_calls == 0
    np.testing.assert_allclose(env._sim.scene.vic_controller._target_pos_wp.value, [[202] * 3, [230] * 3])
    np.testing.assert_allclose(
        env._sim.scene.vic_controller._target_rot_wp.value, [[302] * 4, [330] * 4]
    )
    assert [call.args[0] for call in env.set_excitation_context.call_args_list] == [0, 1]
    np.testing.assert_allclose(
        [call.args[1].direction for call in env.set_excitation_context.call_args_list],
        [
            np.asarray([2.0, 1.0, 0.0]) / np.sqrt(5.0),
            np.asarray([30.0, 1.0, 0.0]) / np.sqrt(901.0),
        ],
    )
    assert dataset.load_episode_obs_arrays.call_args_list == [
        ((0, 2),),
        ((3, 0),),
    ]
    assert dataset.load_episode_metadata.call_args_list == [
        ((0, 2),),
        ((3, 0),),
    ]


def test_initialize_batched_env_from_episode_sources_rejects_wrong_source_count():
    env = _mock_episode_sources_env()

    with pytest.raises(ValueError, match="sources length"):
        initialize_batched_env_from_episode_sources(
            env,
            _mock_episode_sources_dataset(),
            (ReplayEpisodeSource(0, 0),),
        )


def test_initialize_batched_env_from_dataset_delegates_with_physical_direction_ids(monkeypatch):
    from types import SimpleNamespace

    captured = {}

    def capture_sources(env, dataset, sources):
        captured["env"] = env
        captured["dataset"] = dataset
        captured["sources"] = tuple(sources)

    monkeypatch.setattr(
        "apple_pick_sim.system_id.batched_digital_twin_init."
        "initialize_batched_env_from_episode_sources",
        capture_sources,
    )
    env = SimpleNamespace(num_envs=4)
    dataset = object()

    initialize_batched_env_from_dataset(
        env,
        dataset,
        structure_idx=3,
        num_directions=2,
        direction_indices=(2, 7),
    )

    assert captured["env"] is env
    assert captured["dataset"] is dataset
    assert captured["sources"] == (
        ReplayEpisodeSource(3, 2),
        ReplayEpisodeSource(3, 7),
        ReplayEpisodeSource(3, 2),
        ReplayEpisodeSource(3, 7),
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
    num_directions = 1
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
    output_dir = tmp_path_factory.mktemp("batched_digital_twin")
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


def test_gripper_proxy_from_episode_metadata_sets_weld_fields():
    meta = {
        "weld_direction": [0.1, 0.2, 0.97],
        "weld_reference_pos": [1.0, 2.0, 3.0],
        "weld_reference_quat": [0.0, 0.0, 0.0, 1.0],
    }
    base = GripperProxyConfig(mass=0.5)

    proxy = gripper_proxy_from_episode_metadata(meta, base=base)

    assert proxy.fix_to_apple is True
    assert proxy.robot_facing_weld is False
    assert proxy.mass == pytest.approx(0.5)
    assert proxy.weld_direction == pytest.approx((0.1, 0.2, 0.97))
    assert proxy.weld_reference_pos == pytest.approx((1.0, 2.0, 3.0))
    assert proxy.weld_reference_quat == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_gripper_proxy_from_episode_metadata_robot_facing_when_no_weld_direction():
    meta: dict = {}
    proxy = gripper_proxy_from_episode_metadata(meta)
    assert proxy.fix_to_apple is True
    assert proxy.robot_facing_weld is True
    assert proxy.weld_direction is None


def test_gripper_proxy_for_real_batched_replay_sets_true_tcp_offset():
    meta = {
        "weld_direction": [0.0, -1.0, 0.0],
        "initial_apple_pos": [0.1, 0.2, 0.3],
        "initial_apple_quat": [0.0, 0.0, 0.0, 1.0],
        "initial_tcp_pos": [0.1, 0.15, 0.3],
        "initial_tcp_quat": [0.0, 0.0, 0.0, 1.0],
        "weld_reference_pos": [0.1, 0.2, 0.3],
        "weld_reference_quat": [0.0, 0.0, 0.0, 1.0],
    }
    proxy = gripper_proxy_for_real_batched_replay(meta)
    assert proxy.fix_to_apple is True
    assert proxy.weld_reference_pos == pytest.approx((0.1, 0.2, 0.3))
    assert proxy.weld_reference_quat == pytest.approx((0.0, 0.0, 0.0, 1.0))
    expected = proxy_offset_from_apple_and_tcp(
        apple_pos=(0.1, 0.2, 0.3),
        apple_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        tcp_pos=(0.1, 0.15, 0.3),
        tcp_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    assert proxy.weld_proxy_offset_in_apple_frame == pytest.approx(expected)


def test_gripper_proxy_for_real_batched_replay_requires_tcp_and_apple():
    with pytest.raises(ValueError, match="initial_tcp"):
        gripper_proxy_for_real_batched_replay(
            {
                "initial_apple_pos": [0.0, 0.0, 0.0],
                "initial_apple_quat": [0.0, 0.0, 0.0, 1.0],
            }
        )


def test_apply_logged_post_grasp_se3_to_cable_sets_apple_and_proxy():
    """Minimal cable stub: apple + proxy bodies; apply logged SE(3) from meta."""
    apple_pos = (0.5, 0.6, 0.7)
    apple_quat = (0.0, 0.0, 0.0, 1.0)
    tcp_pos = (0.5, 0.55, 0.7)
    tcp_quat = (0.0, 0.0, 0.0, 1.0)
    offset = proxy_offset_from_apple_and_tcp(
        apple_pos=apple_pos,
        apple_quat_xyzw=apple_quat,
        tcp_pos=tcp_pos,
        tcp_quat_xyzw=tcp_quat,
    )
    bq = np.zeros((2, 7), dtype=np.float32)
    bq[0] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    bq[1] = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    bqd = np.ones((2, 6), dtype=np.float32)

    class _Arr:
        def __init__(self, value):
            self._v = np.asarray(value, dtype=np.float32)

        def numpy(self):
            return self._v.copy()

        def assign(self, value):
            self._v = np.asarray(value, dtype=np.float32).reshape(self._v.shape).copy()

    class _State:
        def __init__(self):
            self.body_q = _Arr(bq.copy())
            self.body_qd = _Arr(bqd.copy())

    class _Cable:
        def __init__(self):
            self.apple_body = 0
            self.gripper_proxy_body = 1
            self.gripper_proxy_offset_in_apple_frame = offset
            self.state_0 = _State()
            self.state_1 = _State()
            self.model = type("_M", (), {"body_count": 2})()

    cable = _Cable()
    sync_calls: list[object] = []

    def _fake_sync(c):
        sync_calls.append(c)

    import apple_pick_sim.system_id.batched_digital_twin_init as mod

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mod, "sync_model_body_q_rest_from_state", _fake_sync)
    monkeypatch.setattr(mod, "align_proxy_body_q_prev_for_vbd", lambda *_a, **_k: None)
    try:
        apply_logged_post_grasp_se3_to_cable(
            cable,
            {
                "initial_apple_pos": list(apple_pos),
                "initial_apple_quat": list(apple_quat),
                "initial_tcp_pos": list(tcp_pos),
                "initial_tcp_quat": list(tcp_quat),
            },
        )
    finally:
        monkeypatch.undo()

    out = cable.state_0.body_q.numpy().reshape(-1, 7)
    np.testing.assert_allclose(out[0, :3], apple_pos, atol=1e-6)
    assert abs(float(np.dot(out[0, 3:7], apple_quat))) > 1.0 - 1e-5
    np.testing.assert_allclose(out[1, :3], tcp_pos, atol=1e-5)
    assert abs(float(np.dot(out[1, 3:7], tcp_quat))) > 1.0 - 1e-4
    assert sync_calls


def test_apply_logged_post_grasp_se3_writes_every_layout_world():
    from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
    from apple_pick_sim.system_id.real_post_grasp_plan import proxy_offset_from_apple_and_tcp

    apple_pos = (0.5, 0.6, 0.7)
    apple_quat = (0.0, 0.0, 0.0, 1.0)
    tcp_pos = (0.5, 0.55, 0.7)
    tcp_quat = (0.0, 0.0, 0.0, 1.0)
    offset = proxy_offset_from_apple_and_tcp(
        apple_pos=apple_pos,
        apple_quat_xyzw=apple_quat,
        tcp_pos=tcp_pos,
        tcp_quat_xyzw=tcp_quat,
    )
    bq = np.zeros((4, 7), dtype=np.float32)
    bq[0] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    bq[1] = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    bq[2] = [2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0]
    bq[3] = [3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    bqd = np.ones((4, 6), dtype=np.float32)

    class _Arr:
        def __init__(self, value):
            self._v = np.asarray(value, dtype=np.float32)

        def numpy(self):
            return self._v.copy()

        def assign(self, value):
            self._v = np.asarray(value, dtype=np.float32).reshape(self._v.shape).copy()

    class _State:
        def __init__(self):
            self.body_q = _Arr(bq.copy())
            self.body_qd = _Arr(bqd.copy())

    class _Cable:
        def __init__(self):
            self.apple_body = 0
            self.gripper_proxy_body = 1
            self.gripper_proxy_offset_in_apple_frame = offset
            self.state_0 = _State()
            self.state_1 = _State()
            self.model = type("_M", (), {"body_count": 4})()

    cable = _Cable()
    meta = {
        "initial_apple_pos": list(apple_pos),
        "initial_apple_quat": list(apple_quat),
        "initial_tcp_pos": list(tcp_pos),
        "initial_tcp_quat": list(tcp_quat),
    }
    layout = BatchedEnvLayout(
        num_envs=2,
        bodies_per_world=2,
        robot_bodies_per_world=1,
        joints_per_world=1,
        joint_coord_count_per_world=1,
        joint_dof_count_per_world=1,
        template_tcp_body=0,
        template_proxy_body=1,
        template_apple_body=0,
        tcp_body_indices=(0, 0),
        proxy_body_indices=(1, 3),
        apple_body_indices=(0, 2),
    )

    import apple_pick_sim.system_id.batched_digital_twin_init as mod

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mod, "sync_model_body_q_rest_from_state", lambda *_a, **_k: None)
    monkeypatch.setattr(mod, "align_proxy_body_q_prev_for_vbd", lambda *_a, **_k: None)
    try:
        apply_logged_post_grasp_se3_to_cable(cable, meta, layout=layout)
    finally:
        monkeypatch.undo()

    out = cable.state_0.body_q.numpy().reshape(-1, 7)
    for apple_id in (0, 2):
        np.testing.assert_allclose(out[apple_id, :3], apple_pos, atol=1e-6)
    for proxy_id in (1, 3):
        np.testing.assert_allclose(out[proxy_id, :3], tcp_pos, atol=1e-5)


@gymnasium_available
@requires_fr3
def test_digital_twin_obs_from_batched_episode_has_junction_names(
    tiny_batched_dataset: BatchedSysIdDataset,
):
    obs = digital_twin_obs_from_batched_episode(tiny_batched_dataset, 0, 0)
    assert obs.junction_names
    assert len(obs.woody_part_start_pos) == 3 * len(obs.junction_names)
    assert len(obs.woody_part_end_pos) == 3 * len(obs.junction_names)
    assert obs.fruiting_base_pos is not None
    assert obs.weld_direction is not None


@gymnasium_available
@requires_fr3
def test_infer_base_params_for_structure_returns_fruiting_params(
    tiny_batched_dataset: BatchedSysIdDataset,
):
    params = infer_base_params_for_structure(tiny_batched_dataset, 0)
    assert isinstance(params, FruitingSystemParams)
    assert params.primary is not None
    assert params.primary.bend_stiffness > 0


@gymnasium_available
@requires_fr3
def test_true_params_for_structure_returns_exact_sampled_params(
    tiny_batched_dataset: BatchedSysIdDataset,
):
    true_params = true_params_for_structure(tiny_batched_dataset, 0)
    meta = tiny_batched_dataset.load_episode_metadata(0, 0)
    from_metadata = fruiting_params_from_json(str(meta["fruiting_system_params"]))
    inferred = infer_base_params_for_structure(tiny_batched_dataset, 0)

    assert true_params.primary is not None
    assert from_metadata.primary is not None
    assert inferred.primary is not None

    assert true_params.primary.length == pytest.approx(from_metadata.primary.length)
    assert true_params.primary.direction == pytest.approx(from_metadata.primary.direction)
    assert true_params.primary.bend_stiffness == pytest.approx(
        from_metadata.primary.bend_stiffness
    )

    length_delta = abs(inferred.primary.length - true_params.primary.length)
    direction_delta = 1.0 - abs(
        float(np.dot(np.asarray(inferred.primary.direction), np.asarray(true_params.primary.direction)))
    )
    assert length_delta > 1e-6 or direction_delta > 1e-6


@gymnasium_available
@requires_fr3
def test_initialize_batched_env_from_dataset_sets_joint_q_and_tcp(
    tiny_batched_dataset: BatchedSysIdDataset,
):
    structure_idx = 0
    num_directions = 1
    params = infer_base_params_for_structure(tiny_batched_dataset, structure_idx)
    arrays = tiny_batched_dataset.load_episode_obs_arrays(structure_idx, 0)
    first_trajectory_frame = frame_index_for_step(arrays, FIRST_TRAJECTORY_STEP_IDX)
    recorded_joint_q = np.asarray(
        arrays["robot_joint_q"][first_trajectory_frame], dtype=np.float32
    ).reshape(-1)
    recorded_tcp_pos = np.asarray(arrays["tcp_pos"][first_trajectory_frame], dtype=np.float32).reshape(3)

    env = ApplePickBatchedSysIdEnv(
        num_envs=1,
        max_episode_steps=10,
        ranges_path=RANGES_FIXTURE,
        topology_seed=_SEED,
        use_settle_cache=False,
        sim_config=_test_sim_config(num_envs=1),
        per_env_params=[params],
    )
    try:
        env.reset(seed=_SEED)
        initialize_batched_env_from_dataset(
            env,
            tiny_batched_dataset,
            structure_idx=structure_idx,
            num_directions=num_directions,
        )
        env._gather_obs()
        exported = env.sysid_numpy_obs(0)
        np.testing.assert_allclose(
            exported["robot_joint_q"],
            recorded_joint_q,
            rtol=0,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            exported["tcp_pos"],
            recorded_tcp_pos,
            rtol=0,
            atol=1e-3,
        )
    finally:
        env.close()


def test_initialize_batched_env_from_dataset_raises_without_joint_q():
    from unittest.mock import MagicMock

    env = MagicMock()
    env.num_envs = 1
    env._sim.layout = MagicMock()
    dataset = MagicMock()
    dataset.load_episode_obs_arrays.return_value = {
        "excitation_direction": np.zeros((1, 3), dtype=np.float32),
    }
    dataset.load_episode_metadata.return_value = {}

    with pytest.raises(ValueError, match="missing initial robot_joint_q"):
        initialize_batched_env_from_dataset(
            env,
            dataset,
            structure_idx=0,
            num_directions=1,
        )
