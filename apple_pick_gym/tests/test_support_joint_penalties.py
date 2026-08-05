"""Tests for per-env support joint kp/kd applicator (sys-ID)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_SIM_TESTS_DIR = _TESTS_DIR.parent.parent / "apple_pick_sim" / "tests"
if str(_SIM_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_TESTS_DIR))

from conftest import COUPLED_SCENE_KW, requires_fr3  # noqa: E402
from apple_pick_gym.batched_envs.support_joint_penalties import (  # noqa: E402
    SUPPORT_JOINT_ZETA_FALLBACK,
    apply_per_env_support_joint_penalties,
    support_joint_zeta_from_dataset,
)
from apple_pick_sim.coupled_fruiting import CoupledFruitingScene  # noqa: E402
from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout  # noqa: E402
from apple_pick_sim.coupled_fruiting.batched_build import (  # noqa: E402
    build_heterogeneous_coupled_cable_scene,
)
from apple_pick_sim.fruiting_system import (  # noqa: E402
    GripperProxyConfig,
    load_ranges,
    sample_heterogeneous_params_list,
)
from apple_pick_sim.fruiting_system.joint_kd_scaling import (  # noqa: E402
    joint_kd_from_damping_ratio,
)
from apple_pick_sim.robot import fr3_robot  # noqa: E402

_NUM_ENVS = 2
T_JUNCTION_RANGES_FIXTURE = (
    _SIM_TESTS_DIR.parent / "fixtures" / "fruiting_system_ranges_real_world_proxy_variance.json"
)


def _gripper_free() -> GripperProxyConfig:
    return GripperProxyConfig(
        mass=fr3_robot.EE_MASS_KG,
        fix_to_apple=False,
        robot_facing_weld=False,
    )


def _build_support_scene() -> CoupledFruitingScene:
    ranges = load_ranges(T_JUNCTION_RANGES_FIXTURE)
    params_list = sample_heterogeneous_params_list(
        ranges, topology_seed=7, num_envs=_NUM_ENVS
    )
    cable, _offsets = build_heterogeneous_coupled_cable_scene(
        params_list,
        env_spacing=(2.5, 2.5, 0.0),
        device="cpu",
        enable_self_collisions=False,
        base_pos=COUPLED_SCENE_KW["base_pos"],
        robot_base_pos=COUPLED_SCENE_KW["robot_base_pos"],
        gripper_proxy=_gripper_free(),
    )
    layout = BatchedEnvLayout.from_cable_only(cable, cable.model)
    return CoupledFruitingScene(
        cable=cable,
        cable_collision_pipeline=None,
        vbd_only=True,
        layout=layout,
    )


def _joints_per_world(cable) -> int:
    jws = cable.model.joint_world_start.numpy()
    return int(jws[1] - jws[0])


def _template_joint_by_label(fruiting_fixed_joints, label_substr: str) -> int:
    matches = [j for j, lab in fruiting_fixed_joints if label_substr in lab]
    assert len(matches) == 1, f"expected one joint for {label_substr!r}, got {matches}"
    return matches[0]


def _angular_kd_at_joint(solver, global_joint_index: int) -> float:
    import newton

    jc_start = solver.joint_constraint_start.numpy()
    kd = solver.joint_penalty_kd.numpy()
    c0 = int(jc_start[global_joint_index])
    return float(kd[c0 + newton.solvers.SolverVBD.JointSlot.ANGULAR])


def _linear_kd_at_joint(solver, global_joint_index: int) -> float:
    import newton

    jc_start = solver.joint_constraint_start.numpy()
    kd = solver.joint_penalty_kd.numpy()
    c0 = int(jc_start[global_joint_index])
    return float(kd[c0 + newton.solvers.SolverVBD.JointSlot.LINEAR])


def _angular_kp_at_joint(solver, global_joint_index: int) -> float:
    import newton

    jc_start = solver.joint_constraint_start.numpy()
    k = solver.joint_penalty_k.numpy()
    c0 = int(jc_start[global_joint_index])
    return float(k[c0 + newton.solvers.SolverVBD.JointSlot.ANGULAR])


def _linear_kp_at_joint(solver, global_joint_index: int) -> float:
    import newton

    jc_start = solver.joint_constraint_start.numpy()
    k = solver.joint_penalty_k.numpy()
    c0 = int(jc_start[global_joint_index])
    return float(k[c0 + newton.solvers.SolverVBD.JointSlot.LINEAR])


def test_apply_per_env_support_joint_penalties_sets_kp_and_critical_kd():
    scene = _build_support_scene()
    cable = scene.cable
    layout = scene.layout
    num_envs = int(layout.num_envs)
    joints_per_world = int(layout.joints_per_world)
    j_support = _template_joint_by_label(cable.fruiting_fixed_joints, "primary_support_left")
    j_spur_stem = _template_joint_by_label(cable.fruiting_fixed_joints, "spur_stem")

    spur_stem_kd_before = [
        _angular_kd_at_joint(cable.solver, w * joints_per_world + j_spur_stem)
        for w in range(num_envs)
    ]

    support_kp = [1.0e3, 2.0e4]
    zeta = 1.0
    apply_per_env_support_joint_penalties(
        scene,
        support_kp,
        num_envs=num_envs,
        joints_per_world=joints_per_world,
        zeta=zeta,
    )

    model = cable.model
    body_mass = model.body_mass.numpy()
    body_inertia = model.body_inertia.numpy()
    joint_child = model.joint_child.numpy()
    bodies_per_world = int(layout.bodies_per_world)

    for w, kp in enumerate(support_kp):
        global_joint = w * joints_per_world + j_support
        assert _angular_kp_at_joint(cable.solver, global_joint) == pytest.approx(kp)
        assert _linear_kp_at_joint(cable.solver, global_joint) == pytest.approx(kp)

        ang_kd, lin_kd = joint_kd_from_damping_ratio(
            zeta=zeta,
            roles=("support",),
            fruiting_fixed_joints=cable.fruiting_fixed_joints,
            body_mass=body_mass,
            body_inertia=body_inertia,
            joint_child=joint_child,
            angular_kp_by_role={"support": kp},
            linear_kp_by_role={"support": kp},
            body_offset=w * bodies_per_world,
        )
        assert _angular_kd_at_joint(cable.solver, global_joint) == pytest.approx(
            ang_kd["support"]
        )
        assert _linear_kd_at_joint(cable.solver, global_joint) == pytest.approx(
            lin_kd["support"]
        )
        assert ang_kd["support"] == pytest.approx(
            zeta * 2.0 * math.sqrt(kp * _inertia_max(body_inertia, child_body(w, bodies_per_world, cable, j_support)))
        )
        assert lin_kd["support"] == pytest.approx(
            zeta * 2.0 * math.sqrt(kp * _child_mass(body_mass, child_body(w, bodies_per_world, cable, j_support)))
        )

        spur_kd_after = _angular_kd_at_joint(
            cable.solver, w * joints_per_world + j_spur_stem
        )
        assert spur_kd_after == pytest.approx(spur_stem_kd_before[w])


def _inertia_max(body_inertia: np.ndarray, child: int) -> float:
    mat = body_inertia[child]
    sym = 0.5 * (mat + mat.T)
    return float(np.max(np.linalg.eigvalsh(sym)))


def _child_mass(body_mass: np.ndarray, child: int) -> float:
    return float(body_mass[child])


def child_body(world: int, bodies_per_world: int, cable, template_joint: int) -> int:
    joint_child = cable.model.joint_child.numpy()
    child_local = int(joint_child[template_joint])
    return world * bodies_per_world + child_local


def test_apply_rejects_nonpositive_support_kp():
    scene = _build_support_scene()
    layout = scene.layout
    with pytest.raises(ValueError, match="support_kp"):
        apply_per_env_support_joint_penalties(
            scene,
            [0.0],
            num_envs=int(layout.num_envs),
            joints_per_world=int(layout.joints_per_world),
        )


def test_apply_rejects_wrong_support_kp_length():
    scene = _build_support_scene()
    layout = scene.layout
    with pytest.raises(ValueError, match="support_kp"):
        apply_per_env_support_joint_penalties(
            scene,
            [1.0e3],
            num_envs=int(layout.num_envs),
            joints_per_world=int(layout.joints_per_world),
        )


def test_support_joint_zeta_from_dataset_reads_sim_config():
    dataset = type(
        "D",
        (),
        {
            "manifest": {
                "collection": {
                    "sim_config": {"joint_damping_ratio": 0.5},
                }
            },
            "dataset_dir": Path("/tmp/fake_support_kp_dataset"),
        },
    )()
    assert support_joint_zeta_from_dataset(dataset) == pytest.approx(0.5)


def test_support_joint_zeta_from_dataset_falls_back_when_missing():
    dataset = type(
        "D",
        (),
        {
            "manifest": {"collection": {}},
            "dataset_dir": Path("/tmp/fake_support_kp_dataset"),
        },
    )()
    assert support_joint_zeta_from_dataset(dataset) == pytest.approx(
        SUPPORT_JOINT_ZETA_FALLBACK
    )


def test_support_joint_zeta_from_dataset_rejects_negative():
    dataset = type(
        "D",
        (),
        {
            "manifest": {
                "collection": {"sim_config": {"joint_damping_ratio": -0.1}}
            },
            "dataset_dir": Path("/tmp/fake"),
        },
    )()
    with pytest.raises(ValueError, match="joint_damping_ratio"):
        support_joint_zeta_from_dataset(dataset)
