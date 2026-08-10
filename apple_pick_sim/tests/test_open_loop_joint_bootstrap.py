"""Open-loop FR3 joint bootstrap (skip IK) for real replay."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from apple_pick_sim.coupled_fruiting.settle_then_weld import (
    _bootstrap_tcp_at_fixed_origin,
    apply_open_loop_fr3_joint_q,
)


def test_apply_open_loop_fr3_joint_q_writes_arm_coords_and_evals_fk(monkeypatch):
    q_rec = np.array([0.1, 0.2, 0.3, -1.0, 0.0, 1.5, -0.5], dtype=np.float32)
    jq = np.zeros(9, dtype=np.float32)
    jqd = np.ones(9, dtype=np.float32)

    class _Arr:
        def __init__(self, data):
            self._data = np.asarray(data, dtype=np.float32).copy()

        def numpy(self):
            return self._data.copy()

        def assign(self, values):
            self._data = np.asarray(values, dtype=np.float32).reshape(self._data.shape).copy()

    model = SimpleNamespace(
        joint_q=_Arr(jq),
        joint_qd=_Arr(jqd),
        joint_coord_count=9,
        joint_dof_count=9,
        world_count=1,
    )
    state = SimpleNamespace(joint_q=_Arr(jq), joint_qd=_Arr(jqd))
    scene = SimpleNamespace(
        robot_model=model,
        robot_state_0=state,
        robot_control=object(),
        mj_solver=object(),
        layout=None,
        ik_template_robot_model=None,
        proxy_forces=None,
        coupling_forces_cache=None,
        fr3_root_world_pos=(0.0, 0.0, 0.0),
    )

    fk = MagicMock()
    monkeypatch.setattr("newton.eval_fk", fk)
    init_buf = MagicMock()
    monkeypatch.setattr(
        "apple_pick_sim.coupled_fruiting.settle_then_weld.init_robot_mujoco_step_buffers",
        init_buf,
    )
    init_act = MagicMock()
    monkeypatch.setattr(
        "apple_pick_sim.robot.fr3_robot.init_mujoco_actuator_targets_from_model",
        init_act,
    )

    apply_open_loop_fr3_joint_q(scene, q_rec)

    got = model.joint_q.numpy()
    np.testing.assert_allclose(got[:7], q_rec, atol=1e-6)
    np.testing.assert_allclose(got[7:], 0.0, atol=1e-6)
    np.testing.assert_allclose(model.joint_qd.numpy(), 0.0, atol=1e-6)
    fk.assert_called_once()
    init_buf.assert_called_once_with(scene)
    init_act.assert_called_once()


def test_bootstrap_tcp_at_fixed_origin_skips_ik_when_joint_q_given(monkeypatch):
    called = {"ik": 0, "open": 0}

    def _fake_ik(*_a, **_k):
        called["ik"] += 1

    def _fake_open(scene, joint_q):
        called["open"] += 1
        assert list(joint_q) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    monkeypatch.setattr(
        "apple_pick_sim.coupled_fruiting.bootstrap.bootstrap_articulated_tcp_from_proxy",
        _fake_ik,
    )
    monkeypatch.setattr(
        "apple_pick_sim.coupled_fruiting.settle_then_weld.apply_open_loop_fr3_joint_q",
        _fake_open,
    )

    scene = SimpleNamespace(
        robot_model=object(),
        robot_state_0=object(),
        mj_solver=object(),
        layout=None,
        ik_template_robot_model=None,
        fr3_root_world_pos=(0.0, 0.0, 0.0),
    )
    _bootstrap_tcp_at_fixed_origin(
        scene,
        bootstrap_joint_q=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
    )
    assert called["open"] == 1
    assert called["ik"] == 0


def test_apply_open_loop_rejects_empty_joint_q():
    scene = SimpleNamespace(robot_model=object(), robot_state_0=object(), mj_solver=object())
    with pytest.raises(ValueError, match="bootstrap_joint_q"):
        apply_open_loop_fr3_joint_q(scene, [])
