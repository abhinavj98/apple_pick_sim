"""Tests for GPU broadcast kernels."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import warp as wp

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from conftest import RANGES_FIXTURE, build_two_env_fr3_batched, requires_fr3
from apple_pick_sim.coupled_fruiting.broadcast_actions import (
    _broadcast_joint_q_host,
    broadcast_joint_q_from_world0,
)
from apple_pick_sim.coupled_fruiting.broadcast_device import (
    broadcast_joint_q_from_world0_device,
    broadcast_robot_state_from_template_device,
)
from apple_pick_sim.coupled_fruiting.batched_build import (
    _broadcast_robot_state_from_template_host,
)
from apple_pick_sim.fruiting_system import load_ranges


def _two_env_fr3_scene(*, device: str = "cpu"):
    return build_two_env_fr3_batched(load_ranges(RANGES_FIXTURE), 42, device=device)


def _perturb_world0_joint_q(scene):
    layout = scene.layout
    assert layout is not None
    jq = scene.robot_model.joint_q.numpy().copy()
    jqd = scene.robot_model.joint_qd.numpy().copy()
    w0_q = layout.joint_q_slice(0)
    w0_qd = layout.joint_qd_slice(0)
    jq[w0_q] += 0.1
    jqd[w0_qd] += 0.05
    scene.robot_model.joint_q.assign(jq)
    scene.robot_model.joint_qd.assign(jqd)
    scene.robot_state_0.joint_q.assign(jq)
    scene.robot_state_0.joint_qd.assign(jqd)
    return layout, w0_q, w0_qd


@requires_fr3
def test_broadcast_joint_q_device_matches_host():
    scene = _two_env_fr3_scene(device="cpu")
    layout, w0_q, w0_qd = _perturb_world0_joint_q(scene)

    host_scene = _two_env_fr3_scene(device="cpu")
    _perturb_world0_joint_q(host_scene)
    _broadcast_joint_q_host(host_scene, host_scene.layout)
    host_jq = host_scene.robot_model.joint_q.numpy()
    host_jqd = host_scene.robot_model.joint_qd.numpy()

    broadcast_joint_q_from_world0_device(scene, layout)
    wp.synchronize()
    dev_jq = scene.robot_model.joint_q.numpy()
    dev_jqd = scene.robot_model.joint_qd.numpy()

    for w in range(1, layout.num_envs):
        np.testing.assert_allclose(dev_jq[layout.joint_q_slice(w)], host_jq[w0_q])
        np.testing.assert_allclose(dev_jqd[layout.joint_qd_slice(w)], host_jqd[w0_qd])


@requires_fr3
def test_broadcast_joint_q_public_api_copies_world0():
    scene = _two_env_fr3_scene(device="cpu")
    layout, w0_q, w0_qd = _perturb_world0_joint_q(scene)
    broadcast_joint_q_from_world0(scene, layout)
    jq_out = scene.robot_model.joint_q.numpy()
    for w in range(1, layout.num_envs):
        np.testing.assert_allclose(jq_out[layout.joint_q_slice(w)], jq_out[w0_q])


@requires_fr3
def test_broadcast_robot_template_device_matches_host():
    scene = _two_env_fr3_scene(device="cpu")
    layout = scene.layout
    assert layout is not None
    model = scene.robot_model
    tpl_jc = int(layout.joint_coord_count_per_world)
    tpl_dof = int(layout.joint_dof_count_per_world)

    tpl_jq = model.joint_q.numpy()[:tpl_jc].copy()
    tpl_jqd = model.joint_qd.numpy()[:tpl_dof].copy()
    tpl_jq += 0.33
    tpl_jqd += 0.07

    class _TemplateModel:
        joint_coord_count = tpl_jc
        joint_dof_count = tpl_dof
        device = model.device
        joint_q = wp.array(tpl_jq, dtype=float, device=model.device)
        joint_qd = wp.array(tpl_jqd, dtype=float, device=model.device)

    host_model = scene.robot_model
    host_jq = host_model.joint_q.numpy().copy()
    host_jq[tpl_jc:] = 99.0
    host_model.joint_q.assign(host_jq)
    _broadcast_robot_state_from_template_host(_TemplateModel(), host_model)
    host_out = host_model.joint_q.numpy()

    dev_model = _two_env_fr3_scene(device="cpu").robot_model
    dev_jq = dev_model.joint_q.numpy().copy()
    dev_jq[tpl_jc:] = 99.0
    dev_model.joint_q.assign(dev_jq)
    broadcast_robot_state_from_template_device(_TemplateModel(), dev_model)
    wp.synchronize()
    dev_out = dev_model.joint_q.numpy()

    np.testing.assert_allclose(dev_out, host_out)
