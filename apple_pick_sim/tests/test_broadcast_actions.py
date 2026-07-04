"""Tests for batched action broadcast helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from conftest import COUPLED_SCENE_KW, RANGES_FIXTURE
from apple_pick_sim.coupled_fruiting import build_batched_coupled_fruiting_placeholder
from apple_pick_sim.coupled_fruiting.broadcast_actions import (
    _broadcast_joint_q_host,
    broadcast_joint_q_from_world0,
)
from apple_pick_sim.fruiting_system import load_ranges


def _two_env_placeholder_scene():
    return build_batched_coupled_fruiting_placeholder(
        load_ranges(RANGES_FIXTURE),
        42,
        num_envs=2,
        device="cpu",
        **COUPLED_SCENE_KW,
    )


def test_broadcast_joint_q_device_path_matches_host_reference():
    scene = _two_env_placeholder_scene()
    layout = scene.layout
    assert layout is not None
    jq = scene.robot_model.joint_q.numpy().copy()
    jqd = scene.robot_model.joint_qd.numpy().copy()
    w0_slice = layout.joint_q_slice(0)
    w0_dof = layout.joint_qd_slice(0)
    jq[w0_slice] += 0.1
    jqd[w0_dof] += 0.05
    scene.robot_model.joint_q.assign(jq)
    scene.robot_model.joint_qd.assign(jqd)
    scene.robot_state_0.joint_q.assign(jq)
    scene.robot_state_0.joint_qd.assign(jqd)

    ref_scene = _two_env_placeholder_scene()
    ref_jq = ref_scene.robot_model.joint_q.numpy().copy()
    ref_jqd = ref_scene.robot_model.joint_qd.numpy().copy()
    ref_jq[w0_slice] += 0.1
    ref_jqd[w0_dof] += 0.05
    ref_scene.robot_model.joint_q.assign(ref_jq)
    ref_scene.robot_model.joint_qd.assign(ref_jqd)
    ref_scene.robot_state_0.joint_q.assign(ref_jq)
    ref_scene.robot_state_0.joint_qd.assign(ref_jqd)
    _broadcast_joint_q_host(ref_scene, layout)

    broadcast_joint_q_from_world0(scene, layout)
    jq_out = scene.robot_model.joint_q.numpy()
    jqd_out = scene.robot_model.joint_qd.numpy()
    ref_jq_out = ref_scene.robot_model.joint_q.numpy()
    ref_jqd_out = ref_scene.robot_model.joint_qd.numpy()
    np.testing.assert_allclose(jq_out, ref_jq_out)
    np.testing.assert_allclose(jqd_out, ref_jqd_out)
