"""Batched stem harvest and heterogeneous runtime coupling arrays."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import warp as wp

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from conftest import COUPLED_SCENE_KW, RANGES_FIXTURE, SUB_DT, requires_fr3
from apple_pick_sim.coupled_fruiting import (
    seed_fix_to_apple_from_settled,
    quiet_all_cable_bodies,
    settle_vbd_substeps,
)
from apple_pick_sim.coupled_fruiting.builders import build_heterogeneous_coupled_fruiting_fr3
from apple_pick_sim.coupled_fruiting.proxy_coupling import (
    harvest_batched_stem_tension,
    harvest_stem_tension_for_tcp,
    prepare_batched_stem_harvest_arrays,
    welded_co_teleport_arrays_for_layout,
)
from apple_pick_sim.fruiting_system import (
    GripperProxyConfig,
    load_ranges,
    sample_heterogeneous_params_list,
)
from apple_pick_sim.robot import fr3_robot

_NUM_ENVS = 2
_SETTLE_SUBSTEPS = 50


@pytest.fixture
def ranges():
    return load_ranges(RANGES_FIXTURE)


def _gripper_free() -> GripperProxyConfig:
    return GripperProxyConfig(
        mass=fr3_robot.EE_MASS_KG,
        fix_to_apple=False,
        robot_facing_weld=False,
    )


def _gripper_welded() -> GripperProxyConfig:
    return GripperProxyConfig(
        mass=fr3_robot.EE_MASS_KG,
        fix_to_apple=True,
        robot_facing_weld=True,
    )


def _make_hetero_welded_scene(ranges, seed: int, *, settle_substeps: int = _SETTLE_SUBSTEPS):
    params_list = sample_heterogeneous_params_list(
        ranges, topology_seed=seed, num_envs=_NUM_ENVS
    )
    build_kw = dict(
        ranges=ranges,
        params_list=params_list,
        device="cpu",
        env_spacing=(2.5, 2.5, 0.0),
        **COUPLED_SCENE_KW,
    )
    settled = build_heterogeneous_coupled_fruiting_fr3(
        vbd_only=True,
        gripper_proxy=_gripper_free(),
        **build_kw,
    )
    settle_vbd_substeps(settled, substeps=settle_substeps, dt=SUB_DT)
    quiet_all_cable_bodies(settled.cable)
    welded = build_heterogeneous_coupled_fruiting_fr3(
        gripper_proxy=_gripper_welded(),
        skip_ik_bootstrap=True,
        defer_template_robot_bootstrap=True,
        **build_kw,
    )
    seed_fix_to_apple_from_settled(
        welded_scene=welded,
        settled_scene=settled,
        quiet_apple_proxy=True,
        per_env_ik=True,
        per_world_proxy_offsets=welded.per_world_proxy_offsets,
    )
    prepare_batched_stem_harvest_arrays(welded, welded.layout)
    return welded, params_list


def test_harvest_batched_stem_tension_no_numpy_on_joint_indices():
    """Batched stem harvest must pass device joint indices without .numpy()."""
    n = 2
    dev = "cpu"
    stem_joint_indices_wp = wp.array([10, 19], dtype=wp.int32, device=dev)
    tcp_indices_wp = wp.array([7, 15], dtype=wp.int32, device=dev)
    apple_indices_wp = wp.array([5, 15], dtype=wp.int32, device=dev)
    grasp_offsets_wp = wp.array(
        [wp.transform_identity(), wp.transform_identity()],
        dtype=wp.transform,
        device=dev,
    )
    apple_masses_wp = wp.array([0.1, 0.2], dtype=float, device=dev)
    use_grasp_offset_wp = wp.array([0, 0], dtype=int, device=dev)
    out_robot_wrenches = wp.zeros(16, dtype=wp.spatial_vector, device=dev)
    out_f = wp.zeros(n, dtype=wp.vec3, device=dev)
    out_t = wp.zeros(n, dtype=wp.vec3, device=dev)
    body_q = wp.zeros(20, dtype=wp.transform, device=dev)
    captured: dict[str, object] = {}

    def _fake_gather(*_args, **kwargs):
        captured["joint_indices"] = kwargs.get("joint_indices")
        captured["out_f"] = kwargs.get("out_f")
        captured["out_t"] = kwargs.get("out_t")
        return out_f, out_t

    class _CableModel:
        def control(self, clone_variables: bool = False):
            del clone_variables
            return object()

    with patch(
        "apple_pick_sim.vbd_fixed_joint_wrenches.gather_joint_wrench_child_com_device",
        side_effect=_fake_gather,
    ):
        harvest_batched_stem_tension(
            stem_joint_indices_wp=stem_joint_indices_wp,
            tcp_indices_wp=tcp_indices_wp,
            apple_indices_wp=apple_indices_wp,
            grasp_offsets_wp=grasp_offsets_wp,
            apple_masses_wp=apple_masses_wp,
            use_grasp_offset_wp=use_grasp_offset_wp,
            cable_model=_CableModel(),
            cable_solver=object(),
            body_q_post=body_q,
            body_q_prev=body_q,
            dt=0.01,
            out_robot_wrenches=out_robot_wrenches,
            out_f=out_f,
            out_t=out_t,
            device=dev,
        )

    assert isinstance(captured["joint_indices"], wp.array)
    assert captured["out_f"] is out_f
    assert captured["out_t"] is out_t


def test_prepare_batched_stem_harvest_arrays_allocates_wrench_scratch():
    """prepare_batched_stem_harvest_arrays pre-allocates wrench scratch buffers."""
    layout = SimpleNamespace(
        num_envs=2,
        joint_index=lambda w, j: w * 9 + j,
        tcp_body_indices=(7, 16),
        apple_body_indices=(2, 4),
    )
    scene = SimpleNamespace(
        stem_apple_joint_index=3,
        cable=SimpleNamespace(
            model=SimpleNamespace(
                device="cpu",
                body_mass=wp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=float, device="cpu"),
            ),
            gripper_proxy_offset_in_apple_frame=None,
        ),
        per_world_proxy_offsets=None,
    )
    prepare_batched_stem_harvest_arrays(scene, layout)
    assert scene.stem_harvest_wrench_f_scratch is not None
    assert scene.stem_harvest_wrench_t_scratch is not None
    assert int(scene.stem_harvest_wrench_f_scratch.shape[0]) == 2
    assert int(scene.stem_harvest_wrench_t_scratch.shape[0]) == 2


@requires_fr3
@pytest.mark.slow
def test_batched_stem_harvest_output_matches_serial_loop(ranges):
    """Batched stem harvest matches per-env serial calls on the same post-step state."""
    scene, _params = _make_hetero_welded_scene(ranges, seed=31)
    layout = scene.layout
    assert layout is not None
    assert scene.stem_harvest_joint_indices_wp is not None

    from conftest import run_coupled_substeps_direct_hold

    run_coupled_substeps_direct_hold(scene, fr3_robot, 8, sub_dt=SUB_DT)

    cable = scene.cable
    dt = SUB_DT
    out_batched = wp.zeros(scene.robot_model.body_count, dtype=wp.spatial_vector, device="cpu")
    harvest_batched_stem_tension(
        stem_joint_indices_wp=scene.stem_harvest_joint_indices_wp,
        tcp_indices_wp=scene.stem_harvest_tcp_indices_wp,
        apple_indices_wp=scene.stem_harvest_apple_indices_wp,
        grasp_offsets_wp=scene.stem_harvest_grasp_offsets_wp,
        apple_masses_wp=scene.stem_harvest_apple_masses_wp,
        use_grasp_offset_wp=scene.stem_harvest_use_grasp_offset_wp,
        cable_model=cable.model,
        cable_solver=cable.solver,
        body_q_post=cable.state_0.body_q,
        body_q_prev=cable.state_1.body_q,
        dt=dt,
        out_robot_wrenches=out_batched,
        coupling_gain=scene.stem_coupling_gain,
        force_cap_N=scene.stem_force_cap_N,
        torque_cap_Nm=scene.stem_torque_cap_Nm,
        explicit_apple_weight=scene.stem_harvest_explicit_apple_weight,
        gravity=scene.gravity_vec,
        robot_body_q=scene.robot_state_0.body_q,
        device="cpu",
    )

    out_serial = wp.zeros_like(out_batched)
    offset_default = cable.gripper_proxy_offset_in_apple_frame
    per_offsets = scene.per_world_proxy_offsets
    masses = scene.stem_harvest_apple_masses_wp.numpy()
    for w in range(layout.num_envs):
        grasp = per_offsets[w] if per_offsets and per_offsets[w] is not None else offset_default
        harvest_stem_tension_for_tcp(
            cable_model=cable.model,
            cable_solver=cable.solver,
            body_q_post=cable.state_0.body_q,
            body_q_prev=cable.state_1.body_q,
            dt=dt,
            stem_apple_joint_index=int(layout.joint_index(w, scene.stem_apple_joint_index)),
            tcp_body_index=int(layout.tcp_body_indices[w]),
            out_robot_wrenches=out_serial,
            coupling_gain=scene.stem_coupling_gain,
            force_cap_N=scene.stem_force_cap_N,
            torque_cap_Nm=scene.stem_torque_cap_Nm,
            explicit_apple_weight=scene.stem_harvest_explicit_apple_weight,
            apple_body_index=int(layout.apple_body_indices[w]),
            apple_mass_kg=float(masses[w]),
            gravity=scene.gravity_vec,
            robot_body_q=scene.robot_state_0.body_q,
            grasp_offset_in_apple_frame=grasp,
            clear_wrenches=(w == 0),
        )

    w_batched = out_batched.numpy().reshape(-1, 6)
    w_serial = out_serial.numpy().reshape(-1, 6)
    for w in range(layout.num_envs):
        tcp = int(layout.tcp_body_indices[w])
        np.testing.assert_allclose(
            w_batched[tcp],
            w_serial[tcp],
            rtol=0.02,
            atol=0.5,
            err_msg=f"world {w} batched vs serial stem harvest mismatch",
        )


@requires_fr3
@pytest.mark.slow
def test_harvest_fallback_raises_when_no_wp_arrays(ranges):
    """Batched multi-env stem harvest must not silently fall back to a CPU loop."""
    scene, _params = _make_hetero_welded_scene(ranges, seed=35)
    assert scene.stem_harvest_joint_indices_wp is not None
    scene.stem_harvest_joint_indices_wp = None

    with pytest.raises(RuntimeError, match="prepare_batched_stem_harvest_arrays"):
        scene.coupled_substep(SUB_DT)


@requires_fr3
@pytest.mark.slow
def test_batched_stem_harvest_no_python_loop(ranges):
    """coupled_substep uses batched stem harvest, not per-env serial calls."""
    scene, _params = _make_hetero_welded_scene(ranges, seed=32)
    assert scene.stem_harvest_joint_indices_wp is not None

    from conftest import run_coupled_substeps_direct_hold

    calls: list[int] = []

    def _track(*args, **kwargs):
        calls.append(1)
        raise AssertionError("serial harvest_stem_tension_for_tcp should not be called")

    with patch(
        "apple_pick_sim.coupled_fruiting.scene.harvest_stem_tension_for_tcp",
        side_effect=_track,
    ):
        run_coupled_substeps_direct_hold(scene, fr3_robot, 4, sub_dt=SUB_DT)

    assert len(calls) == 0


def test_heterogeneous_per_env_grasp_offset_in_runtime_mirror(ranges):
    """welded_co_teleport_arrays_for_layout uses per-env grasp offsets when provided."""
    import dataclasses

    from apple_pick_sim.coupled_fruiting.batched_build import build_heterogeneous_coupled_cable_scene

    params_list = sample_heterogeneous_params_list(ranges, topology_seed=33, num_envs=_NUM_ENVS)
    cable, offsets = build_heterogeneous_coupled_cable_scene(
        params_list,
        env_spacing=(2.5, 2.5, 0.0),
        device="cpu",
        enable_self_collisions=False,
        base_pos=COUPLED_SCENE_KW["base_pos"],
        robot_base_pos=COUPLED_SCENE_KW["robot_base_pos"],
        gripper_proxy=_gripper_welded(),
    )

    @dataclasses.dataclass
    class _LayoutStub:
        num_envs: int
        apple_body_indices: tuple[int, ...]

    layout = _LayoutStub(num_envs=_NUM_ENVS, apple_body_indices=(0, 1))
    _apple_ids, _pos_off, grasp_same = welded_co_teleport_arrays_for_layout(
        layout, cable, device="cpu"
    )
    _apple_ids2, _pos_off2, grasp_per = welded_co_teleport_arrays_for_layout(
        layout, cable, device="cpu", per_world_proxy_offsets=offsets
    )
    g_same = grasp_same.numpy()
    g_per = grasp_per.numpy()
    if offsets[0] != offsets[1]:
        assert not np.allclose(g_per[0], g_per[1])


@requires_fr3
@pytest.mark.slow
def test_heterogeneous_per_env_apple_mass_in_stem_harvest(ranges):
    """Different per-env apple masses produce different stem harvest wrenches."""
    scene, params = _make_hetero_welded_scene(ranges, seed=34)
    if params[0].apple_radius == params[1].apple_radius:
        pytest.skip("need different apple radii for this seed")
    layout = scene.layout
    assert layout is not None
    masses = scene.stem_harvest_apple_masses_wp.numpy()
    assert masses[0] != masses[1]

    from conftest import run_coupled_substeps_direct_hold

    run_coupled_substeps_direct_hold(scene, fr3_robot, 8, sub_dt=SUB_DT)

    w = scene.proxy_forces.numpy().reshape(-1, 6)
    tcp0 = int(layout.tcp_body_indices[0])
    tcp1 = int(layout.tcp_body_indices[1])
    assert not np.allclose(w[tcp0], w[tcp1], atol=1e-3)
