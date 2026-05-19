"""Tests for FR3 end-effector velocity integration and keyboard teleop."""

from __future__ import annotations

import unittest

import warp as wp

from apple_pick_sim import fr3_robot


class _MockViewer:
    def __init__(self, keys: set[str] | None = None) -> None:
        self._keys = {k.lower() for k in (keys or set())}

    def is_key_down(self, key: str) -> bool:
        return key.lower() in self._keys


class TestIntegrateTcpTarget(unittest.TestCase):
    def test_translation_only(self):
        tf0 = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_identity())
        tf1 = fr3_robot.integrate_tcp_target(
            tf0,
            linear_vel=wp.vec3(0.1, -0.2, 0.0),
            angular_vel=wp.vec3(0.0, 0.0, 0.0),
            dt=2.0,
        )
        p = wp.transform_get_translation(tf1)
        self.assertAlmostEqual(p[0], 1.2, places=5)
        self.assertAlmostEqual(p[1], 1.6, places=5)
        self.assertAlmostEqual(p[2], 3.0, places=5)

    def test_rotation_about_z(self):
        import numpy as np

        tf0 = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        tf1 = fr3_robot.integrate_tcp_target(
            tf0,
            linear_vel=wp.vec3(0.0, 0.0, 0.0),
            angular_vel=wp.vec3(0.0, 0.0, 1.0),
            dt=0.5,
        )
        q = np.asarray(wp.transform_get_rotation(tf1), dtype=np.float64)
        # 1.0 rad/s × 0.5 s = 0.5 rad about +Z (warp quat: x, y, z, w)
        angle = 2.0 * np.arctan2(q[2], q[3])
        self.assertAlmostEqual(angle, 0.5, places=3)

    def test_zero_velocity_is_identity(self):
        tf0 = wp.transform(wp.vec3(0.5, 0.0, 1.0), wp.quat(0.0, 0.0, 0.7071, 0.7071))
        tf1 = fr3_robot.integrate_tcp_target(
            tf0,
            linear_vel=wp.vec3(0.0, 0.0, 0.0),
            angular_vel=wp.vec3(0.0, 0.0, 0.0),
            dt=0.1,
        )
        p0 = wp.transform_get_translation(tf0)
        p1 = wp.transform_get_translation(tf1)
        self.assertAlmostEqual(p0[0], p1[0], places=5)
        self.assertAlmostEqual(p0[1], p1[1], places=5)
        self.assertAlmostEqual(p0[2], p1[2], places=5)


class TestSyncMujocoActuatorTargets(unittest.TestCase):
    def test_idle_command_zeros_target_vel(self):
        import numpy as np

        class _State:
            joint_q = type("A", (), {})()

        class _Control:
            def __init__(self) -> None:
                self.joint_target_pos = _Arr()
                self.joint_target_vel = _Arr()

        class _Arr:
            def __init__(self) -> None:
                self._data = np.zeros(3, dtype=np.float32)

            def assign(self, x) -> None:
                self._data = np.asarray(x, dtype=np.float32).reshape(-1)

        class _Model:
            joint_dof_count = 3

        state = _State()
        state.joint_q.numpy = lambda: np.array([0.0, 0.1, 0.0], dtype=np.float32)
        control = _Control()
        fr3_robot.sync_mujoco_actuator_targets_from_joint_q(
            _Model(),
            state,
            control,
            target_joint_q=np.array([0.5, 0.1, 0.0], dtype=np.float32),
            frame_dt=1.0 / 60.0,
            command_velocity=fr3_robot.EEVelocity(),
        )
        np.testing.assert_allclose(control.joint_target_vel._data, 0.0, atol=1e-6)
        np.testing.assert_allclose(control.joint_target_pos._data, [0.5, 0.1, 0.0], atol=1e-6)

    def test_active_command_uses_velocity_feedforward(self):
        import numpy as np

        class _State:
            joint_q = type("A", (), {})()

        class _Control:
            def __init__(self) -> None:
                self.joint_target_pos = _Arr()
                self.joint_target_vel = _Arr()

        class _Arr:
            def __init__(self) -> None:
                self._data = np.zeros(3, dtype=np.float32)

            def assign(self, x) -> None:
                self._data = np.asarray(x, dtype=np.float32).reshape(-1)

        class _Model:
            joint_dof_count = 3

        state = _State()
        state.joint_q.numpy = lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32)
        control = _Control()
        dt = 1.0 / 60.0
        fr3_robot.sync_mujoco_actuator_targets_from_joint_q(
            _Model(),
            state,
            control,
            target_joint_q=np.array([0.06, 0.0, 0.0], dtype=np.float32),
            frame_dt=dt,
            command_velocity=fr3_robot.EEVelocity(linear=(1.0, 0.0, 0.0)),
        )
        np.testing.assert_allclose(control.joint_target_vel._data, [3.6, 0.0, 0.0], rtol=1e-5)


class TestKeyboardVelocity(unittest.TestCase):
    def test_read_keyboard_world_linear(self):
        v = fr3_robot.read_keyboard_ee_velocity(
            _MockViewer({"i", "r"}),
            linear_speed=1.0,
            angular_speed=1.0,
            poll_events=False,
        )
        self.assertAlmostEqual(v.linear[0], 1.0)
        self.assertAlmostEqual(v.linear[2], 1.0)
        self.assertAlmostEqual(v.angular[0], 0.0)

    def test_read_keyboard_no_viewer(self):
        v = fr3_robot.read_keyboard_ee_velocity(None, linear_speed=0.2, angular_speed=0.5)
        self.assertEqual(v.linear, (0.0, 0.0, 0.0))
        self.assertEqual(v.angular, (0.0, 0.0, 0.0))


def _usd_available() -> bool:
    try:
        import pxr  # noqa: F401
    except ImportError:
        return False
    return fr3_robot.fr3_assets_available()


@unittest.skipUnless(_usd_available(), "Requires usd-core and bundled assets/fr3")
class TestFr3EEVelocityController(unittest.TestCase):
    def test_step_updates_joint_q(self):
        import newton

        model, tcp_idx, _ = fr3_robot.build_fr3_robot_model_from_usd(device="cpu")
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        ctrl = fr3_robot.Fr3EEVelocityController(model, tcp_idx)
        ctrl.sync_target_from_state(state)
        jq_before = model.joint_q.numpy().copy()
        ctrl.step(
            1.0 / 60.0,
            viewer=_MockViewer({"i"}),
        )
        ctrl.apply_to_model_and_state(state)
        jq_after = model.joint_q.numpy()
        self.assertFalse((jq_before == jq_after).all())

    def test_seed_ik_from_state_copies_simulated_q(self):
        import numpy as np
        import newton

        model, tcp_idx, _ = fr3_robot.build_fr3_robot_model_from_usd(device="cpu")
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        ctrl = fr3_robot.Fr3EEVelocityController(model, tcp_idx)
        ctrl.sync_target_from_state(state)

        sim_q = state.joint_q.numpy().copy()
        sim_q[0] += 0.15
        state.joint_q.assign(sim_q)
        model.joint_q.assign(np.zeros_like(sim_q))

        ctrl.seed_ik_from_state(state)
        np.testing.assert_allclose(
            ctrl.joint_q.numpy().reshape(-1),
            sim_q.reshape(-1),
            rtol=0,
            atol=1e-6,
        )

    def test_solve_ik_accepts_state_and_seeds(self):
        import numpy as np
        import newton

        model, tcp_idx, _ = fr3_robot.build_fr3_robot_model_from_usd(device="cpu")
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        ctrl = fr3_robot.Fr3EEVelocityController(model, tcp_idx)
        ctrl.sync_target_from_state(state)

        sim_q = state.joint_q.numpy().copy()
        sim_q[0] += 0.12
        state.joint_q.assign(sim_q)
        model.joint_q.assign(np.zeros_like(sim_q))

        ctrl.seed_ik_from_state(state)
        np.testing.assert_allclose(
            ctrl.joint_q.numpy().reshape(-1),
            sim_q.reshape(-1),
            rtol=0,
            atol=1e-6,
        )
        ctrl.joint_q.assign(np.zeros_like(sim_q))
        ctrl.solve_ik(state)
        assert np.isfinite(ctrl.joint_q.numpy()).all()


if __name__ == "__main__":
    unittest.main()
