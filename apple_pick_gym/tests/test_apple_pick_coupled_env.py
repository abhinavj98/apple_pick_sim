"""M2.1 Gymnasium environment tests (placeholder obs + parity)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


def _maybe_import_gymnasium():
    try:
        import gymnasium as gym  # noqa: F401
        from gymnasium.utils.env_checker import check_env  # noqa: F401

        return True
    except Exception:
        return False


gymnasium_available = pytest.mark.skipif(
    not _maybe_import_gymnasium(),
    reason="gymnasium not installed (expected to be provided by newton[dev])",
)


def _import_constants():
    # Reuse direct-path timing constants to avoid drift.
    from apple_pick_sim.tests.conftest import FRAME_DT, SUBSTEPS_PER_FRAME, SUB_DT, RANGES_FIXTURE, requires_fr3

    return FRAME_DT, SUBSTEPS_PER_FRAME, SUB_DT, RANGES_FIXTURE, requires_fr3


@dataclass(frozen=True)
class _StepMetrics:
    joint_q: np.ndarray
    tcp_body_q: np.ndarray
    tcp_wrench: np.ndarray
    params_fp: dict
    fruiting_link_forces: dict[str, dict[str, np.ndarray]]


def _fruiting_link_forces_from_scene(scene, sub_dt: float) -> dict[str, dict[str, np.ndarray]]:
    import apple_pick_sim.fruiting_system as fs

    cable = scene.cable
    measured = fs.measure_fruiting_forces(
        cable, cable.state_0.body_q, cable.state_1.body_q, dt=float(sub_dt)
    )
    out: dict[str, dict[str, np.ndarray]] = {}
    for rec in measured["fixed_joints"]:
        key = rec.label.removeprefix("joint_") or rec.label
        out[key] = {
            "force_world": np.asarray(rec.force_world, dtype=np.float64),
            "torque_at_child_com_world": np.asarray(rec.torque_at_child_com_world, dtype=np.float64),
        }
    return out


def _extract_metrics(scene, sub_dt: float) -> _StepMetrics:
    import apple_pick_sim.fruiting_system as fs

    tcp = scene.tcp_body_index
    joint_q = scene.robot_state_0.joint_q.numpy().reshape(-1).copy()
    tcp_body_q = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp].copy()
    tcp_wrench = scene.proxy_forces.numpy().reshape(-1, 6)[tcp].copy()
    params_fp = fs.params_fingerprint(scene.cable.params)
    return _StepMetrics(
        joint_q=joint_q,
        tcp_body_q=tcp_body_q,
        tcp_wrench=tcp_wrench,
        params_fp=params_fp,
        fruiting_link_forces=_fruiting_link_forces_from_scene(scene, sub_dt),
    )


@gymnasium_available
def test_env_observation_contract():
    import gymnasium as gym
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    env = gym.make("ApplePickCoupled-v0", render_mode=None, max_episode_steps=2)
    obs, info = env.reset(seed=0)
    unwrapped = env.unwrapped

    expected_keys = {
        "woody_part_start_pos",
        "woody_part_end_pos",
        "woody_part_force",
        "apple_pos",
        "tcp_force",
        "tcp_velocity",
    }
    assert isinstance(obs, dict)
    assert set(obs.keys()) == expected_keys
    assert env.observation_space.contains(obs)
    assert obs["apple_pos"].shape == (3,)
    assert obs["tcp_force"].shape == (6,)
    assert obs["tcp_velocity"].shape == (6,)
    n = int(info["n_woody_parts"])
    assert n > 0
    assert isinstance(obs["woody_part_start_pos"], dict)
    assert isinstance(obs["woody_part_end_pos"], dict)
    assert set(obs["woody_part_start_pos"].keys()) == set(unwrapped.junction_names)
    assert set(obs["woody_part_end_pos"].keys()) == set(unwrapped.junction_names)
    for name in unwrapped.junction_names:
        assert obs["woody_part_start_pos"][name].shape == (3,)
        assert obs["woody_part_end_pos"][name].shape == (3,)
    assert obs["woody_part_force"].shape == (n * 6,)
    assert obs["apple_pos"].dtype == np.float32
    assert "params_fingerprint" in info

    apple_key = next(k for k in info["fruiting_link_forces"] if k.endswith("_apple"))
    apple_end = obs["woody_part_end_pos"][apple_key]
    assert not np.allclose(apple_end, obs["apple_pos"], atol=1e-5)

    obs2, reward, terminated, truncated, info2 = env.step(12)
    assert env.observation_space.contains(obs2)
    assert isinstance(reward, float)
    assert terminated is False
    assert isinstance(truncated, (bool, np.bool_))
    assert info2["n_woody_parts"] == n
    assert "end_effector_wrench" in info2
    assert "fruiting_link_forces" in info2

    env.close()


@gymnasium_available
def test_info_exposes_fruiting_link_and_ee_forces():
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    import gymnasium as gym

    env = gym.make("ApplePickCoupled-v0", render_mode=None, max_episode_steps=2)
    _, info = env.reset(seed=0)

    ee = info["end_effector_wrench"]
    assert ee.shape == (6,)
    assert ee.dtype == np.float32

    links = info["fruiting_link_forces"]
    assert isinstance(links, dict)
    assert len(links) > 0
    for label, entry in links.items():
        assert isinstance(label, str)
        assert entry["force_world"].shape == (3,)
        assert entry["torque_at_child_com_world"].shape == (3,)
        assert entry["force_world"].dtype == np.float32

    _, _, _, _, info2 = env.step(12)
    links2 = info2["fruiting_link_forces"]
    apple_key = next(k for k in links2 if k.endswith("_apple"))
    stem_force = links2[apple_key]["force_world"]
    assert float(np.linalg.norm(stem_force)) > 1.0

    assert info2["end_effector_wrench"].shape == (6,)
    assert len(info2["fruiting_link_forces"]) == len(links)

    env.close()


@gymnasium_available
def test_env_parity_against_direct_coupled_sim():
    FRAME_DT, SUBSTEPS_PER_FRAME, SUB_DT, RANGES_FIXTURE, requires_fr3 = _import_constants()

    if getattr(requires_fr3, "args", None):  # pytest mark object
        # If FR3 assets are missing, skip like the existing direct-path tests do.
        requires_fr3.args[0]  # touch for type check
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    import gymnasium as gym
    import apple_pick_sim.coupled_fruiting as cf
    import apple_pick_sim.fruiting_system as fs
    from apple_pick_sim.robot import fr3_robot
    from apple_pick_sim.tests.conftest import RANGES_FIXTURE, build_coupled_fr3

    ranges = fs.load_ranges(RANGES_FIXTURE)
    seed = 13
    mujoco_solver_kwargs = {"disable_contacts": True}

    # --- Direct reference rollout ---
    direct_scene = build_coupled_fr3(
        cf,
        ranges,
        seed,
        mujoco_solver_kwargs=mujoco_solver_kwargs,
    )
    direct_scene.robot_kinematic_mode = True
    direct_ctrl = fr3_robot.Fr3EEDirectJointController(direct_scene.robot_model, direct_scene.tcp_body_index)
    direct_ctrl.sync_target_from_state(direct_scene.robot_state_0)

    # --- Gym rollout (same seed + action schedule) ---
    env = gym.make(
        "ApplePickCoupled-v0",
        render_mode=None,
        max_episode_steps=10,
        mujoco_solver_kwargs=mujoco_solver_kwargs,
        enable_self_collisions=False,
    )
    env.reset(seed=seed, options={"ranges_path": str(RANGES_FIXTURE)})

    # Deterministic action schedule (includes noop and several axes).
    action_schedule = [12, 0, 0, 2, 10, 12, 4, 6]

    for a in action_schedule:
        # Step direct scene using the env's mapping.
        vel = env.unwrapped._action_to_velocity(a)  # intentional: parity contract surface
        direct_scene.update_fr3_ee_teleop_direct(FRAME_DT, direct_ctrl, velocity=vel)
        for _ in range(SUBSTEPS_PER_FRAME):
            direct_scene.coupled_substep(SUB_DT)

        # Step gym env once (one frame).
        env.step(a)

    got = env.unwrapped._scene
    assert got is not None

    m_direct = _extract_metrics(direct_scene, SUB_DT)
    m_gym = _extract_metrics(got, SUB_DT)

    np.testing.assert_allclose(m_gym.joint_q, m_direct.joint_q, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(m_gym.tcp_body_q, m_direct.tcp_body_q, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(m_gym.tcp_wrench, m_direct.tcp_wrench, rtol=1e-5, atol=1e-4)
    assert m_gym.params_fp == m_direct.params_fp
    assert m_gym.fruiting_link_forces.keys() == m_direct.fruiting_link_forces.keys()
    for key in m_direct.fruiting_link_forces:
        np.testing.assert_allclose(
            m_gym.fruiting_link_forces[key]["force_world"],
            m_direct.fruiting_link_forces[key]["force_world"],
            rtol=1e-5,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            m_gym.fruiting_link_forces[key]["torque_at_child_com_world"],
            m_direct.fruiting_link_forces[key]["torque_at_child_com_world"],
            rtol=1e-5,
            atol=1e-4,
        )

    # ``info`` reflects state *after* this step; compare to a fresh scene readout, not pre-step metrics.
    _, _, _, _, info = env.step(12)
    m_gym_after = _extract_metrics(got, SUB_DT)
    np.testing.assert_allclose(info["end_effector_wrench"], m_gym_after.tcp_wrench, rtol=1e-5, atol=1e-4)
    for key in m_gym_after.fruiting_link_forces:
        np.testing.assert_allclose(
            info["fruiting_link_forces"][key]["force_world"],
            m_gym_after.fruiting_link_forces[key]["force_world"],
            rtol=1e-5,
            atol=1e-4,
        )

    env.close()


def test_action_to_velocity_matches_fr3_keyboard_world_frame():
    """Discrete actions must use the same world-frame axes and speeds as keyboard teleop."""
    from apple_pick_gym.envs import ApplePickCoupledEnv
    from apple_pick_sim.robot import fr3_robot

    env = ApplePickCoupledEnv()
    lin = 0.2
    ang = 1.0

    # Gym action 0:+X must match keyboard ``i`` at default linear_speed.
    assert env._action_to_velocity(0) == fr3_robot.EEVelocity(linear=(+lin, 0.0, 0.0))
    assert env._action_to_velocity(4) == fr3_robot.read_keyboard_ee_velocity(
        type("_V", (), {"is_key_down": lambda _s, k: k == "r"})(),
        linear_speed=lin,
        angular_speed=ang,
        poll_events=False,
    )
    assert env._action_to_velocity(12) == fr3_robot.EEVelocity()
    assert env._action_to_velocity(10) == fr3_robot.EEVelocity(angular=(0.0, 0.0, +ang))


@gymnasium_available
def test_reset_same_seed_same_params_fingerprint():
    import gymnasium as gym
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    env = gym.make("ApplePickCoupled-v0", render_mode=None, max_episode_steps=2)
    _, info_a = env.reset(seed=7)
    _, info_b = env.reset(seed=7)
    assert info_a["params_fingerprint"] == info_b["params_fingerprint"]
    env.close()


@gymnasium_available
def test_step_before_reset_raises():
    import gymnasium as gym
    from gymnasium.error import ResetNeeded
    from apple_pick_gym.envs import ApplePickCoupledEnv
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    raw = ApplePickCoupledEnv(max_episode_steps=2)
    with pytest.raises(RuntimeError, match="reset"):
        raw.step(0)

    wrapped = gym.make("ApplePickCoupled-v0", render_mode=None, max_episode_steps=2)
    with pytest.raises(ResetNeeded, match="reset"):
        wrapped.step(0)
    wrapped.close()


@gymnasium_available
def test_invalid_action_rejected():
    from apple_pick_gym.envs import ApplePickCoupledEnv

    env = ApplePickCoupledEnv()
    with pytest.raises(ValueError, match="Invalid action"):
        env._action_to_velocity(13)


@gymnasium_available
def test_truncation_after_max_episode_steps():
    import gymnasium as gym
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    env = gym.make("ApplePickCoupled-v0", render_mode=None, max_episode_steps=2)
    env.reset(seed=0)
    _, _, _, truncated, _ = env.step(12)
    assert truncated is False
    _, _, _, truncated, _ = env.step(12)
    assert truncated is True
    env.close()


@gymnasium_available
def test_apple_pick_base_env_is_abstract():
    from apple_pick_gym.envs import ApplePickBaseEnv

    with pytest.raises(TypeError):
        ApplePickBaseEnv()


@gymnasium_available
def test_replay_env_observation_contract():
    from apple_pick_gym.envs import ApplePickReplayEnv
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    env = ApplePickReplayEnv(max_episode_steps=2)
    obs, info = env.reset(seed=0)

    expected_keys = {
        "ft_wrist",
        "woody_start",
        "woody_end",
        "tcp_velocity",
        "tcp_pos",
        "apple_pos",
    }
    assert isinstance(obs, dict)
    assert set(obs.keys()) == expected_keys
    assert env.observation_space.contains(obs)
    assert obs["ft_wrist"].shape == (6,)
    assert obs["tcp_velocity"].shape == (6,)
    n = int(info["n_woody_parts"])
    assert n > 0
    assert obs["woody_start"].shape == (n * 3,)
    assert obs["woody_end"].shape == (n * 3,)
    env.close()


@gymnasium_available
def test_reset_params_injection_round_trip():
    import apple_pick_sim.fruiting_system as fs
    from apple_pick_gym.envs import ApplePickReplayEnv
    from apple_pick_sim.tests.conftest import RANGES_FIXTURE, fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    ranges = fs.load_ranges(RANGES_FIXTURE)
    seed = 11
    params = fs.sample_params(ranges, seed)
    params = fs.perturb_rod_stiffness(params, "stem", bend_delta=12.5)

    env = ApplePickReplayEnv(max_episode_steps=2)
    _, info = env.reset(
        seed=seed,
        options={"ranges_path": str(RANGES_FIXTURE), "params": params},
    )
    assert info["params_fingerprint"] == fs.params_fingerprint(params)
    env.close()


@gymnasium_available
def test_vic_env_dynamic_arm_configured():
    from apple_pick_gym.envs import ApplePickVicEnv
    from apple_pick_sim.robot.fr3_robot import Fr3EEImpedanceController
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    env = ApplePickVicEnv(
        max_episode_steps=2,
        fix_to_apple=False,
        fix_to_apple_warmup_substeps=0,
    )
    env.reset(seed=0)
    unwrapped = env.unwrapped
    assert unwrapped._scene.robot_kinematic_mode is False
    assert isinstance(unwrapped._controller, Fr3EEImpedanceController)
    assert unwrapped._scene.vic_controller is unwrapped._controller
    assert unwrapped._scene.vic_use_joint_torques is True
    env.close()


@gymnasium_available
def test_vic_env_ft_wrist_is_lagged_plant_not_fresh_harvest():
    from apple_pick_gym.envs import ApplePickVicEnv
    from apple_pick_sim.coupling_force_debug import read_tcp_wrench
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    env = ApplePickVicEnv(
        max_episode_steps=2,
        fix_to_apple=False,
        fix_to_apple_warmup_substeps=0,
    )
    obs, _ = env.reset(seed=0)
    scene = env.unwrapped._scene
    tcp = int(scene.tcp_body_index)
    cache = read_tcp_wrench(scene.coupling_forces_cache, tcp).astype(np.float32)
    np.testing.assert_allclose(obs["ft_wrist"], cache, rtol=1e-5, atol=1e-4)

    obs2, *_ = env.step(0)  # +X motion so plant harvest can diverge from cache
    cache2 = read_tcp_wrench(scene.coupling_forces_cache, tcp).astype(np.float32)
    np.testing.assert_allclose(obs2["ft_wrist"], cache2, rtol=1e-5, atol=1e-4)
    # F/T proxy is lagged applied plant load; tcp_force is the fresh VBD harvest.
    assert obs2["ft_wrist"].shape == (6,)
    assert obs2["tcp_force"].shape == (6,)
    env.close()


@gymnasium_available
def test_vic_env_log_ft_wrist_arrow():
    from newton.viewer import ViewerNull

    from apple_pick_gym.envs import ApplePickVicEnv

    class _ArrowProbe(ViewerNull):
        def __init__(self):
            super().__init__(num_frames=1)
            self.arrow_calls: list[str] = []
            self.line_calls: list[str] = []

        def log_arrows(self, name, starts, ends, colors, hidden=False):
            del starts, ends, colors, hidden
            self.arrow_calls.append(name)

        def log_lines(self, name, starts, ends, colors, hidden=False):
            del starts, ends, colors, hidden
            self.line_calls.append(name)

    class _BodyQ:
        @staticmethod
        def numpy():
            return np.array([[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    scene = type(
        "Scene",
        (),
        {"tcp_body_index": 0, "robot_state_0": type("S", (), {"body_q": _BodyQ()})()},
    )()
    viewer = _ArrowProbe()
    obs = {"ft_wrist": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)}
    ApplePickVicEnv.log_ft_wrist_arrow(viewer, obs, scene=scene)
    assert viewer.arrow_calls == ["/gym/ft_wrist"]

    viewer2 = _ArrowProbe()
    ApplePickVicEnv.log_ft_wrist_arrow(viewer2, {"ft_wrist": np.zeros(6, np.float32)}, scene=scene)
    assert viewer2.arrow_calls == ["/gym/ft_wrist"]
    assert viewer2.line_calls == []


@gymnasium_available
def test_vic_env_log_junction_force_arrows():
    from newton.viewer import ViewerNull

    from apple_pick_gym.envs import ApplePickVicEnv

    class _ArrowProbe(ViewerNull):
        def __init__(self):
            super().__init__(num_frames=1)
            self.arrow_calls: list[str] = []
            self.line_calls: list[str] = []

        def log_arrows(self, name, starts, ends, colors, hidden=False):
            del starts, ends, colors, hidden
            self.arrow_calls.append(name)

        def log_lines(self, name, starts, ends, colors, hidden=False):
            del starts, ends, colors, hidden
            self.line_calls.append(name)

    class _Cable:
        fruiting_fixed_joints = [(0, "joint_primary_secondary"), (1, "joint_stem_apple")]

    scene = type("Scene", (), {"cable": _Cable()})()
    viewer = _ArrowProbe()
    obs = {
        "woody_part_start_pos": {
            "primary_secondary": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "stem_apple": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        },
        "woody_part_end_pos": {
            "primary_secondary": np.array([0.0, 0.0, 0.1], dtype=np.float32),
            "stem_apple": np.array([1.0, 0.0, 0.1], dtype=np.float32),
        },
        "woody_part_force": np.array(
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
    }
    ApplePickVicEnv.log_junction_force_arrows(viewer, obs, scene=scene)
    assert viewer.arrow_calls == [
        "/gym/junction_forces/primary_secondary",
        "/gym/junction_forces/stem_apple",
    ]

    viewer2 = _ArrowProbe()
    ApplePickVicEnv.log_junction_force_arrows(
        viewer2,
        {
            **obs,
            "woody_part_force": np.zeros(12, dtype=np.float32),
        },
        scene=scene,
    )
    assert viewer2.arrow_calls == [
        "/gym/junction_forces/primary_secondary",
        "/gym/junction_forces/stem_apple",
    ]
    assert viewer2.line_calls == []


@gymnasium_available
def test_vic_env_log_woody_part_markers():
    from newton.viewer import ViewerNull

    from apple_pick_gym.envs import ApplePickVicEnv

    class _LogPointsProbe(ViewerNull):
        def __init__(self):
            super().__init__(num_frames=1)
            self.calls: list[tuple[str, int | None]] = []

        def log_points(self, name, points, radii=None, colors=None, hidden=False):
            self.calls.append((name, None if points is None else len(points)))

    viewer = _LogPointsProbe()
    obs = {
        "woody_part_start_pos": {
            "primary_secondary": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "stem_apple": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        },
        "woody_part_end_pos": {
            "primary_secondary": np.array([0.0, 0.0, 0.1], dtype=np.float32),
            "stem_apple": np.array([1.0, 0.0, 0.1], dtype=np.float32),
        },
    }
    ApplePickVicEnv.log_woody_part_markers(viewer, obs, radius=0.01)

    assert viewer.calls == [("/gym/woody_parts", 2), ("/gym/primary_base", None)]

    viewer3 = _LogPointsProbe()

    class _BodyQ:
        @staticmethod
        def numpy():
            return np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    class _Cable:
        primary_bodies = [0]
        state_0 = type("S", (), {"body_q": _BodyQ()})()

    ApplePickVicEnv.log_woody_part_markers(viewer3, obs, scene=type("Scene", (), {"cable": _Cable()})())
    assert viewer3.calls == [("/gym/woody_parts", 2), ("/gym/primary_base", 1)]

    viewer2 = _LogPointsProbe()
    ApplePickVicEnv.log_woody_part_markers(
        viewer2,
        {
            "woody_part_start_pos": {},
            "woody_part_end_pos": {},
        },
    )
    assert viewer2.calls == [("/gym/woody_parts", None), ("/gym/primary_base", None)]


@gymnasium_available
def test_vic_env_observation_contract():
    import gymnasium as gym
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    env = gym.make(
        "ApplePickVic-v0",
        render_mode=None,
        max_episode_steps=2,
        fix_to_apple=False,
        fix_to_apple_warmup_substeps=0,
    )
    obs, info = env.reset(seed=0)

    expected_keys = {
        "woody_part_start_pos",
        "woody_part_end_pos",
        "woody_part_force",
        "apple_pos",
        "tcp_force",
        "tcp_velocity",
        "ft_wrist",
    }
    assert isinstance(obs, dict)
    assert set(obs.keys()) == expected_keys
    assert env.observation_space.contains(obs)
    assert env.action_space.n == 13
    assert obs["ft_wrist"].shape == (6,)

    obs2, reward, terminated, truncated, _ = env.step(12)
    assert env.observation_space.contains(obs2)
    assert isinstance(reward, float)
    assert terminated is False
    assert isinstance(truncated, (bool, np.bool_))
    env.close()


@gymnasium_available
def test_vic_env_tcp_moves_under_velocity_command():
    from apple_pick_gym.envs import ApplePickVicEnv
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    env = ApplePickVicEnv(
        max_episode_steps=60,
        fix_to_apple=False,
        fix_to_apple_warmup_substeps=0,
    )
    env.reset(seed=0)
    scene = env.unwrapped._scene
    tcp = int(scene.tcp_body_index)
    x0 = float(scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp, 0])

    for _ in range(60):
        env.step(0)  # Discrete(13): +X linear (0.2 m/s); VIC ramps slower than kinematic

    x1 = float(scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp, 0])
    assert x1 - x0 > 0.05, f"expected VIC env to advance TCP +X, got dx={x1 - x0:.6f} m"
    env.close()


@gymnasium_available
def test_check_env_smoke():
    _, _, _, _, _ = _import_constants()
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env

    env = gym.make("ApplePickCoupled-v0", render_mode=None, max_episode_steps=3)
    check_env(env, skip_render_check=True)
    env.close()

