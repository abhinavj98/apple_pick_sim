"""Tests for batched Newton/MuJoCo robot status helpers."""

from __future__ import annotations

import numpy as np

from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.coupled_fruiting.batched_robot_status import (
    batched_robot_diagnostics,
    format_batched_robot_status_line,
)
from apple_pick_sim.coupled_fruiting.builders import build_batched_coupled_fruiting_placeholder
from apple_pick_sim.fruiting_system import load_ranges
from apple_pick_sim.tests.conftest import COUPLED_SCENE_KW


def _two_env_placeholder_scene():
    ranges = load_ranges(
        "apple_pick_sim/fixtures/fruiting_system_ranges_example_variance_soft.json"
    )
    return build_batched_coupled_fruiting_placeholder(
        ranges,
        42,
        num_envs=2,
        device="cpu",
        **COUPLED_SCENE_KW,
    )


def test_batched_robot_diagnostics_per_world_joint_q():
    scene = _two_env_placeholder_scene()
    layout = scene.layout
    assert layout is not None

    jq = scene.robot_model.joint_q.numpy().copy()
    jq[layout.joint_q_slice(1)][0] += 0.25
    scene.robot_model.joint_q.assign(jq)
    scene.robot_state_0.joint_q.assign(jq)

    d0 = batched_robot_diagnostics(scene, layout, 0)
    d1 = batched_robot_diagnostics(scene, layout, 1)
    assert d0["newton_joint_q"].shape == (7,)
    assert d1["newton_joint_q"].shape == (7,)
    assert float(d0["newton_joint_q"][0]) != float(d1["newton_joint_q"][0])
    assert d0["mujoco_qpos"] is not None
    assert d1["mujoco_qpos"] is not None
    assert d0["joint_target_pos"] is not None


def test_format_status_line_includes_base_and_mujoco():
    layout = BatchedEnvLayout(
        num_envs=1,
        bodies_per_world=1,
        robot_bodies_per_world=1,
        joints_per_world=1,
        joint_coord_count_per_world=7,
        joint_dof_count_per_world=6,
        template_tcp_body=0,
        template_proxy_body=0,
        template_apple_body=None,
        tcp_body_indices=(0,),
        proxy_body_indices=(0,),
        apple_body_indices=(-1,),
    )
    diag = {
        "world": 0,
        "base_pos": np.array([1.0, 2.0, 3.0]),
        "base_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "newton_joint_q": np.zeros(7),
        "state_joint_q": np.zeros(7),
        "newton_joint_qd": np.zeros(6),
        "mujoco_qpos": np.zeros(7),
        "joint_target_pos": np.zeros(6),
        "model_state_joint_q_max_abs": 0.0,
        "newton_mujoco_qpos_max_abs": 0.0,
    }
    line = format_batched_robot_status_line(diag)
    assert "env0:" in line
    assert "base=(1.0000, 2.0000, 3.0000)" in line
    assert "newton_q=" in line
    assert "mj_qpos=" in line
