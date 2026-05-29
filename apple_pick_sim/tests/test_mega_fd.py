"""Tests for mega plant FD: offset copy, per-step reset, Jacobian vs sequential gold."""

from __future__ import annotations

import numpy as np
import pytest

from apple_pick_sim.tests.conftest import COUPLED_BASE_POS, NO_SELF_COLLISION_KW, RANGES_FIXTURE

WRENCH_SLICE = slice(6, 12)

SEED = 7
EPS = 0.02
DT = 1.0 / 1800.0
ATOL = 1e-4
RTOL = 1e-3
INSTANCE_SPACING = (0.0, 1.5, 0.0)


def _import_fs():
    import apple_pick_sim.fruiting_system as fs

    return fs


def _import_mega_fd():
    from apple_pick_sim.fruiting_system import mega_fd

    return mega_fd


@pytest.fixture(scope="module", autouse=True)
def _warp_init():
    import warp as wp

    wp.init()


def _fd_kw():
    return {
        "base_pos": COUPLED_BASE_POS,
        "instance_spacing": INSTANCE_SPACING,
        **NO_SELF_COLLISION_KW,
    }


def _build_mega_fd(fs, *, params_list=None, epsilon=EPS):
    ranges = fs.load_ranges(RANGES_FIXTURE)
    if params_list is not None:
        return fs.MegaCoupledCableScene.build(params_list, **_fd_kw())
    return fs.generate_mega_coupled_cable_scene(
        ranges, seed=SEED, stiffness_epsilon=epsilon, **_fd_kw()
    )


def _collision_pipeline(fs, model):
    return fs.example_collision_pipeline(model)


def _local_features(mega, i: int) -> np.ndarray:
    return _import_mega_fd().default_mega_fd_features(mega, i)


def _world_primary_pos(mega, i: int) -> np.ndarray:
    bid = mega.instance(i).primary_bodies[0]
    return mega.state_0.body_q.numpy().reshape(-1, 7)[bid, :3]


def _instance_spacing_vector(mega) -> np.ndarray:
    p0 = mega.instance(0).base_pos
    p1 = mega.instance(1).base_pos
    return np.array(p1, dtype=np.float64) - np.array(p0, dtype=np.float64)


def _sequential_scenes(fs, params_nom, params_pert):
    from apple_pick_sim.fruiting_system.coupled import _build_coupled_cable_scene
    from apple_pick_sim.sim_device import resolve_sim_device

    device = resolve_sim_device(None)
    proxy_cfg = fs.GripperProxyConfig()
    kw = dict(
        base_pos=COUPLED_BASE_POS,
        device=device,
        enable_self_collisions=False,
        gripper_proxy=proxy_cfg,
    )
    scene_nom = _build_coupled_cable_scene(params_nom, **kw)
    scene_pert = _build_coupled_cable_scene(params_pert, **kw)
    return scene_nom, scene_pert


def _sequential_one_step(fs, params_nom, params_pert, *, dt=DT):
    mfd = _import_mega_fd()
    scene_nom, scene_pert = _sequential_scenes(fs, params_nom, params_pert)
    mfd.copy_coupled_scene_from_nominal(scene_nom, scene_pert)
    pipe_nom = _collision_pipeline(fs, scene_nom.model)
    pipe_pert = _collision_pipeline(fs, scene_pert.model)
    mfd.coupled_vbd_substep(scene_nom, dt, collision_pipeline=pipe_nom)
    mfd.coupled_vbd_substep(scene_pert, dt, collision_pipeline=pipe_pert)
    y_nom = _local_features_coupled(scene_nom)
    y_pert = _local_features_coupled(scene_pert)
    return y_nom, y_pert


def _local_features_coupled(scene) -> np.ndarray:
    base = np.array(COUPLED_BASE_POS, dtype=np.float64)
    bq = scene.state_0.body_q.numpy().reshape(-1, 7)
    parts: list[np.ndarray] = []
    if scene.apple_body is not None:
        parts.append(bq[scene.apple_body, :3] - base)
    parts.append(bq[scene.gripper_proxy_body, :3] - base)
    return np.concatenate(parts, dtype=np.float64)


def _sequential_jacobian_column(fs, params_nom, params_pert, *, epsilon=EPS, dt=DT):
    y_nom, y_pert = _sequential_one_step(fs, params_nom, params_pert, dt=dt)
    return (y_pert - y_nom) / epsilon


def _build_two_instance_mega(fs, params_nom, params_pert):
    return fs.MegaCoupledCableScene.build([params_nom, params_pert], **_fd_kw())


# --- A: Positions and offset-aware copy ---


def test_copy_preserves_instance_spacing():
    fs = _import_fs()
    mfd = _import_mega_fd()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=SEED)
    mega = _build_two_instance_mega(fs, nominal, fs.copy_fruiting_params(nominal))
    mfd.copy_mega_instance_state(mega, 0, 1)
    np.testing.assert_allclose(
        _world_primary_pos(mega, 1) - _world_primary_pos(mega, 0),
        _instance_spacing_vector(mega),
        atol=ATOL,
        rtol=RTOL,
    )


def test_copy_all_chain_bodies_offset_consistent():
    fs = _import_fs()
    mfd = _import_mega_fd()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=SEED)
    mega = _build_two_instance_mega(fs, nominal, fs.copy_fruiting_params(nominal))
    spacing = _instance_spacing_vector(mega)
    mfd.copy_mega_instance_state(mega, 0, 1)
    bq = mega.state_0.body_q.numpy().reshape(-1, 7)
    inst0 = mega.instance(0)
    inst1 = mega.instance(1)
    for bid0, bid1 in zip(inst0.chain_bodies, inst1.chain_bodies, strict=True):
        np.testing.assert_allclose(bq[bid1, :3] - bq[bid0, :3], spacing, atol=ATOL, rtol=RTOL)


def test_copy_idempotent():
    fs = _import_fs()
    mfd = _import_mega_fd()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=SEED)
    mega = _build_two_instance_mega(fs, nominal, fs.copy_fruiting_params(nominal))
    mfd.copy_mega_instance_state(mega, 0, 1)
    bq1 = mega.state_0.body_q.numpy().copy()
    mfd.copy_mega_instance_state(mega, 0, 1)
    bq2 = mega.state_0.body_q.numpy()
    np.testing.assert_allclose(bq1, bq2, atol=0.0, rtol=0.0)


def test_sync_at_build_local_features_match():
    fs = _import_fs()
    mega = _build_mega_fd(fs)
    y0 = _local_features(mega, 0)
    for k in range(1, mega.num_instances):
        np.testing.assert_allclose(_local_features(mega, k), y0, atol=ATOL, rtol=RTOL)


def test_copy_does_not_move_nominal():
    fs = _import_fs()
    mfd = _import_mega_fd()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=SEED)
    mega = _build_two_instance_mega(fs, nominal, fs.copy_fruiting_params(nominal))
    before = mega.state_0.body_q.numpy().copy()
    mfd.copy_mega_instance_state(mega, 0, 1)
    after = mega.state_0.body_q.numpy()
    inst0 = mega.instance(0)
    for bid in inst0.chain_bodies:
        np.testing.assert_allclose(after.reshape(-1, 7)[bid], before.reshape(-1, 7)[bid])


def test_reset_copies_joint_q_and_body_q_prev_blocks():
    """After an artificial perturbation on column 1, reset restores full joint/body buffers."""
    fs = _import_fs()
    mfd = _import_mega_fd()
    from apple_pick_sim.fruiting_system.mega_fd import _instance_slices

    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=SEED)
    mega = _build_two_instance_mega(fs, nominal, fs.copy_fruiting_params(nominal))
    sl0 = _instance_slices(mega, 0)
    sl1 = _instance_slices(mega, 1)
    jq = mega.state_0.joint_q.numpy().copy()
    jq[sl1.joint_coord] += 0.05
    mega.state_0.joint_q.assign(jq)
    bqp = mega.solver.body_q_prev.numpy().reshape(-1, 7).copy()
    bid1 = mega.instance(1).primary_bodies[0]
    bqp[bid1, :3] += np.array([0.1, 0.0, 0.0])
    mega.solver.body_q_prev.assign(bqp.ravel())
    mfd.reset_perturbed_instances_to_nominal(mega)
    jq_after = mega.state_0.joint_q.numpy()
    np.testing.assert_allclose(jq_after[sl1.joint_coord], jq_after[sl0.joint_coord], atol=ATOL)
    bqp_after = mega.solver.body_q_prev.numpy().reshape(-1, 7)
    bid0 = mega.instance(0).primary_bodies[0]
    spacing = _instance_spacing_vector(mega)
    np.testing.assert_allclose(bqp_after[bid1, :3], bqp_after[bid0, :3] + spacing, atol=ATOL)


def test_body_q_prev_aligned_after_copy():
    fs = _import_fs()
    mfd = _import_mega_fd()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=SEED)
    mega = _build_two_instance_mega(fs, nominal, fs.copy_fruiting_params(nominal))
    mfd.copy_mega_instance_state(mega, 0, 1)
    bq = mega.state_0.body_q.numpy().reshape(-1, 7)
    bqp = mega.solver.body_q_prev.numpy().reshape(-1, 7)
    for bid in mfd.instance_body_ids(mega.instance(1)):
        np.testing.assert_allclose(bqp[bid], bq[bid], atol=ATOL, rtol=RTOL)


# --- B: Reset behavior ---


def test_step_without_reset_diverges_features():
    fs = _import_fs()
    mfd = _import_mega_fd()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=SEED)
    segs = fs.enabled_rod_segments(nominal)
    pert = fs.perturb_rod_stiffness(nominal, segs[0], bend_delta=EPS, stretch_delta=0.0)
    mega = _build_two_instance_mega(fs, nominal, pert)
    mfd.sync_all_instances_from_nominal(mega)
    pipe = _collision_pipeline(fs, mega.model)
    mfd.mega_vbd_substep(mega, DT, collision_pipeline=pipe)
    assert not np.allclose(_local_features(mega, 1), _local_features(mega, 0), atol=1e-8)


def test_reset_restores_local_features():
    fs = _import_fs()
    mfd = _import_mega_fd()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=SEED)
    segs = fs.enabled_rod_segments(nominal)
    pert = fs.perturb_rod_stiffness(nominal, segs[0], bend_delta=EPS, stretch_delta=0.0)
    mega = _build_two_instance_mega(fs, nominal, pert)
    mfd.sync_all_instances_from_nominal(mega)
    pipe = _collision_pipeline(fs, mega.model)
    mfd.mega_vbd_substep(mega, DT, collision_pipeline=pipe)
    mfd.reset_perturbed_instances_to_nominal(mega)
    np.testing.assert_allclose(
        _local_features(mega, 1), _local_features(mega, 0), atol=ATOL, rtol=RTOL
    )


def test_reset_preserves_world_spacing():
    fs = _import_fs()
    mfd = _import_mega_fd()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=SEED)
    segs = fs.enabled_rod_segments(nominal)
    pert = fs.perturb_rod_stiffness(nominal, segs[0], bend_delta=EPS, stretch_delta=0.0)
    mega = _build_two_instance_mega(fs, nominal, pert)
    pipe = _collision_pipeline(fs, mega.model)
    mfd.mega_vbd_substep(mega, DT, collision_pipeline=pipe)
    mfd.reset_perturbed_instances_to_nominal(mega)
    np.testing.assert_allclose(
        _world_primary_pos(mega, 1) - _world_primary_pos(mega, 0),
        _instance_spacing_vector(mega),
        atol=ATOL,
        rtol=RTOL,
    )


def test_reset_all_perturbed_columns():
    fs = _import_fs()
    mfd = _import_mega_fd()
    mega = _build_mega_fd(fs)
    pipe = _collision_pipeline(fs, mega.model)
    mfd.mega_vbd_substep(mega, DT, collision_pipeline=pipe)
    mfd.reset_perturbed_instances_to_nominal(mega)
    y0 = _local_features(mega, 0)
    for k in range(1, mega.num_instances):
        np.testing.assert_allclose(_local_features(mega, k), y0, atol=ATOL, rtol=RTOL)


def test_reset_leaves_nominal_untouched():
    fs = _import_fs()
    mfd = _import_mega_fd()
    mega = _build_mega_fd(fs)
    pipe = _collision_pipeline(fs, mega.model)
    mfd.mega_vbd_substep(mega, DT, collision_pipeline=pipe)
    before = mega.state_0.body_q.numpy().copy()
    mfd.reset_perturbed_instances_to_nominal(mega)
    after = mega.state_0.body_q.numpy()
    inst0 = mega.instance(0)
    for bid in mfd.instance_body_ids(inst0):
        np.testing.assert_allclose(
            after.reshape(-1, 7)[bid], before.reshape(-1, 7)[bid], atol=0.0, rtol=0.0
        )


def test_multi_step_reset_prevents_drift():
    fs = _import_fs()
    mfd = _import_mega_fd()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=SEED)
    segs = fs.enabled_rod_segments(nominal)
    pert = fs.perturb_rod_stiffness(nominal, segs[0], bend_delta=EPS, stretch_delta=0.0)
    mega = _build_two_instance_mega(fs, nominal, pert)
    pipe = _collision_pipeline(fs, mega.model)
    for _ in range(3):
        np.testing.assert_allclose(
            _local_features(mega, 1), _local_features(mega, 0), atol=ATOL, rtol=RTOL
        )
        result = mfd.mega_fd_step(mega, EPS, dt=DT, collision_pipeline=pipe)
    assert np.all(np.isfinite(result.jacobian))


def test_multi_step_without_reset_accumulates_drift():
    fs = _import_fs()
    mfd = _import_mega_fd()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=SEED)
    segs = fs.enabled_rod_segments(nominal)
    pert = fs.perturb_rod_stiffness(nominal, segs[0], bend_delta=EPS, stretch_delta=0.0)
    mega = _build_two_instance_mega(fs, nominal, pert)
    mfd.sync_all_instances_from_nominal(mega)
    pipe = _collision_pipeline(fs, mega.model)
    mfd.mega_vbd_substep(mega, DT, collision_pipeline=pipe)
    drift1 = np.linalg.norm(_local_features(mega, 1) - _local_features(mega, 0))
    mfd.mega_vbd_substep(mega, DT, collision_pipeline=pipe)
    mfd.mega_vbd_substep(mega, DT, collision_pipeline=pipe)
    drift3 = np.linalg.norm(_local_features(mega, 1) - _local_features(mega, 0))
    assert drift3 > drift1 + 1e-10


def test_mega_fd_step_calls_reset():
    fs = _import_fs()
    mfd = _import_mega_fd()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=SEED)
    segs = fs.enabled_rod_segments(nominal)
    pert = fs.perturb_rod_stiffness(nominal, segs[0], bend_delta=EPS, stretch_delta=0.0)
    mega = _build_two_instance_mega(fs, nominal, pert)
    pipe = _collision_pipeline(fs, mega.model)
    mfd.mega_fd_step(mega, EPS, dt=DT, collision_pipeline=pipe)
    np.testing.assert_allclose(
        _local_features(mega, 1), _local_features(mega, 0), atol=ATOL, rtol=RTOL
    )


# --- C: Jacobian vs sequential gold ---


def test_jacobian_shape():
    fs = _import_fs()
    mfd = _import_mega_fd()
    mega = _build_mega_fd(fs)
    pipe = _collision_pipeline(fs, mega.model)
    result = mfd.mega_fd_step(mega, EPS, dt=DT, collision_pipeline=pipe)
    n = mega.num_instances
    f = result.features.shape[1]
    assert result.features.shape == (n, f)
    assert result.jacobian.shape == (f, n - 1)


def test_jacobian_column_matches_sequential_2instance():
    fs = _import_fs()
    mfd = _import_mega_fd()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=SEED)
    segs = fs.enabled_rod_segments(nominal)
    pert = fs.perturb_rod_stiffness(nominal, segs[0], bend_delta=EPS, stretch_delta=0.0)
    mega = _build_two_instance_mega(fs, nominal, pert)
    pipe = _collision_pipeline(fs, mega.model)
    result = mfd.mega_fd_step(mega, EPS, dt=DT, collision_pipeline=pipe)
    gold = _sequential_jacobian_column(fs, nominal, pert)
    np.testing.assert_allclose(result.jacobian[:, 0], gold, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("segment", ["primary", "secondary", "spur", "stem"])
def test_jacobian_each_column_matches_sequential(segment):
    fs = _import_fs()
    mfd = _import_mega_fd()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=SEED)
    if getattr(nominal, segment) is None:
        pytest.skip(f"segment {segment} disabled in fixture sample")
    pert = fs.perturb_rod_stiffness(nominal, segment, bend_delta=EPS, stretch_delta=0.0)
    mega = _build_two_instance_mega(fs, nominal, pert)
    pipe = _collision_pipeline(fs, mega.model)
    result = mfd.mega_fd_step(mega, EPS, dt=DT, collision_pipeline=pipe)
    gold = _sequential_jacobian_column(fs, nominal, pert)
    np.testing.assert_allclose(result.jacobian[:, 0], gold, atol=ATOL, rtol=RTOL)


def test_jacobian_full_mega_all_columns():
    fs = _import_fs()
    mfd = _import_mega_fd()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=SEED)
    cols = fs.fd_stiffness_param_columns(nominal, EPS)
    mega = fs.MegaCoupledCableScene.build(cols, **_fd_kw())
    pipe = _collision_pipeline(fs, mega.model)
    result = mfd.mega_fd_step(mega, EPS, dt=DT, collision_pipeline=pipe)
    for i in range(1, mega.num_instances):
        gold = _sequential_jacobian_column(fs, cols[0], cols[i])
        np.testing.assert_allclose(
            result.jacobian[:, i - 1], gold, atol=ATOL, rtol=RTOL
        )


def test_jacobian_second_step_from_reset_state():
    fs = _import_fs()
    mfd = _import_mega_fd()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=SEED)
    segs = fs.enabled_rod_segments(nominal)
    pert = fs.perturb_rod_stiffness(nominal, segs[0], bend_delta=EPS, stretch_delta=0.0)
    mega = _build_two_instance_mega(fs, nominal, pert)
    pipe = _collision_pipeline(fs, mega.model)
    mfd.mega_fd_step(mega, EPS, dt=DT, collision_pipeline=pipe)
    result2 = mfd.mega_fd_step(mega, EPS, dt=DT, collision_pipeline=pipe)
    scene_nom, scene_pert = _sequential_scenes(fs, nominal, pert)
    _copy_mega_nominal_to_coupled(mfd, mega, scene_nom)
    _copy_mega_nominal_to_coupled(mfd, mega, scene_pert)
    mfd.copy_coupled_scene_from_nominal(scene_nom, scene_pert)
    pipe_nom = _collision_pipeline(fs, scene_nom.model)
    pipe_pert = _collision_pipeline(fs, scene_pert.model)
    mfd.coupled_vbd_substep(scene_nom, DT, collision_pipeline=pipe_nom)
    mfd.coupled_vbd_substep(scene_pert, DT, collision_pipeline=pipe_pert)
    gold = (_local_features_coupled(scene_pert) - _local_features_coupled(scene_nom)) / EPS
    np.testing.assert_allclose(result2.jacobian[:, 0], gold, atol=ATOL, rtol=RTOL)


def _copy_mega_nominal_to_coupled(mfd, mega, scene) -> None:
    """Copy mega instance 0 bodies onto a standalone coupled scene (same base_pos)."""
    inst = mega.instance(0)
    n = scene.model.body_count
    nom_bq = mega.state_0.body_q.numpy().reshape(-1, 7)
    nom_bqd = mega.state_0.body_qd.numpy().reshape(-1, 6)
    for state in (scene.state_0, scene.state_1):
        bq = state.body_q.numpy().reshape(-1, 7).copy()
        bqd = state.body_qd.numpy().reshape(-1, 6).copy()
        for k in range(min(n, len(inst.chain_bodies))):
            bid_mega = inst.chain_bodies[k]
            bq[k] = nom_bq[bid_mega]
            bqd[k] = nom_bqd[bid_mega]
        if scene.apple_body is not None and inst.apple_body is not None:
            bq[scene.apple_body] = nom_bq[inst.apple_body]
            bqd[scene.apple_body] = nom_bqd[inst.apple_body]
        bq[scene.gripper_proxy_body] = nom_bq[inst.gripper_proxy_body]
        bqd[scene.gripper_proxy_body] = nom_bqd[inst.gripper_proxy_body]
        state.body_q.assign(bq.ravel())
        state.body_qd.assign(bqd.ravel())
    bodies = mfd.instance_body_ids_from_coupled(scene)
    from apple_pick_sim.coupled_fruiting.proxy_coupling import align_proxy_body_q_prev_for_vbd

    align_proxy_body_q_prev_for_vbd(scene, bodies)


def test_jacobian_custom_features():
    fs = _import_fs()
    mfd = _import_mega_fd()

    def proxy_only(mega, i):
        bid = mega.instance(i).gripper_proxy_body
        return mega.state_0.body_q.numpy().reshape(-1, 7)[bid, :3].copy()

    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=SEED)
    segs = fs.enabled_rod_segments(nominal)
    pert = fs.perturb_rod_stiffness(nominal, segs[0], bend_delta=EPS, stretch_delta=0.0)
    mega = _build_two_instance_mega(fs, nominal, pert)
    pipe = _collision_pipeline(fs, mega.model)
    result = mfd.mega_fd_step(
        mega, EPS, dt=DT, collision_pipeline=pipe, extract_features=proxy_only
    )
    assert result.features.shape == (2, 3)
    assert result.jacobian.shape == (3, 1)
    assert np.all(np.isfinite(result.jacobian))


# --- D: FIM (J.T @ sigma_inv @ J) ---


def test_fim_step_none_without_sigma_inv():
    fs = _import_fs()
    mfd = _import_mega_fd()
    mega = _build_mega_fd(fs)
    pipe = _collision_pipeline(fs, mega.model)
    result = mfd.mega_fd_step(mega, EPS, dt=DT, collision_pipeline=pipe)
    assert result.fim_step is None


def test_fim_step_identity_sigma():
    fs = _import_fs()
    mfd = _import_mega_fd()
    mega = _build_mega_fd(fs)
    pipe = _collision_pipeline(fs, mega.model)
    n_params = mega.num_instances - 1
    sigma_inv = np.eye(result_dim := _local_features(mega, 0).size)
    result = mfd.mega_fd_step(
        mega, EPS, dt=DT, collision_pipeline=pipe, sigma_inv=sigma_inv
    )
    assert result.fim_step is not None
    assert result.fim_step.shape == (n_params, n_params)
    assert np.all(np.isfinite(result.fim_step))
    np.testing.assert_allclose(result.fim_step, result.fim_step.T, atol=ATOL, rtol=RTOL)
    assert np.trace(result.fim_step) > 0.0


def test_fim_equals_jt_sigma_inv_j():
    fs = _import_fs()
    mfd = _import_mega_fd()
    mega = _build_mega_fd(fs)
    pipe = _collision_pipeline(fs, mega.model)
    dim = _local_features(mega, 0).size
    sigma_inv = np.diag([2.0, 1.5, 1.0, 0.5, 1.25, 3.0][:dim])
    result = mfd.mega_fd_step(
        mega, EPS, dt=DT, collision_pipeline=pipe, sigma_inv=sigma_inv
    )
    expected = result.jacobian.T @ sigma_inv @ result.jacobian
    np.testing.assert_allclose(result.fim_step, expected, atol=1e-12, rtol=1e-12)


def test_fim_scales_linearly_with_sigma_inv():
    fs = _import_fs()
    mfd = _import_mega_fd()
    mega = _build_mega_fd(fs)
    pipe = _collision_pipeline(fs, mega.model)
    dim = _local_features(mega, 0).size
    sigma_inv = np.eye(dim)
    result = mfd.mega_fd_step(
        mega, EPS, dt=DT, collision_pipeline=pipe, sigma_inv=sigma_inv
    )
    fim_unit = result.fim_step
    assert fim_unit is not None
    scaled_sigma = 2.5 * sigma_inv
    fim_scaled = result.jacobian.T @ scaled_sigma @ result.jacobian
    np.testing.assert_allclose(fim_scaled, 2.5 * fim_unit, atol=1e-12, rtol=1e-12)

    mega2 = _build_mega_fd(fs)
    pipe2 = _collision_pipeline(fs, mega2.model)
    result2 = mfd.mega_fd_step(
        mega2, EPS, dt=DT, collision_pipeline=pipe2, sigma_inv=scaled_sigma
    )
    np.testing.assert_allclose(result2.fim_step, fim_scaled, atol=ATOL, rtol=RTOL)


def test_mega_fd_step_invalid_epsilon_raises():
    fs = _import_fs()
    mfd = _import_mega_fd()
    mega = _build_mega_fd(fs)
    with pytest.raises(ValueError, match="epsilon"):
        mfd.mega_fd_step(mega, 0.0, dt=DT)


# --- E: Feature layout, stem joint wiring, wrench convention ---


def _build_mega_with_proxy(fs, *, fix_to_apple: bool):
    ranges = fs.load_ranges(RANGES_FIXTURE)
    params = fs.sample_params(ranges, seed=SEED)
    return fs.MegaCoupledCableScene.build(
        [params],
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=fix_to_apple),
        **_fd_kw(),
    )


@pytest.mark.parametrize(
    ("fix_to_apple", "expected_dim"),
    [(False, 6), (True, 12)],
)
def test_default_mega_fd_feature_dim_depends_on_fix_to_apple(fix_to_apple, expected_dim):
    """Free proxy: apple+proxy positions; welded: adds 6-D stem–apple wrench block."""
    fs = _import_fs()
    mfd = _import_mega_fd()
    mega = _build_mega_with_proxy(fs, fix_to_apple=fix_to_apple)
    feat = mfd.default_mega_fd_features(mega, 0, dt=DT)
    assert feat.shape == (expected_dim,)
    assert np.all(np.isfinite(feat))


def test_mega_fd_stem_apple_joint_matches_coupled_stem_finder():
    """``default_mega_fd_features`` resolves the stem→apple joint, not the proxy weld."""
    fs = _import_fs()
    mfd = _import_mega_fd()
    mega = _build_mega_with_proxy(fs, fix_to_apple=True)
    cable = mega.as_single_instance_coupled(0)
    from apple_pick_sim.coupled_fruiting.stem import _find_stem_apple_joint

    stem_j = _find_stem_apple_joint(cable)
    assert stem_j is not None
    inst = mega.instance(0)
    assert inst.gripper_proxy_apple_joint is not None
    assert stem_j != inst.gripper_proxy_apple_joint

    jchild = mega.model.joint_child.numpy()
    resolved = None
    for j_idx, _label in inst.fruiting_fixed_joints:
        if int(jchild[j_idx]) == inst.apple_body:
            resolved = j_idx
            break
    assert resolved == stem_j


def test_mega_fix_to_apple_prescribes_apple_and_proxy_inv_mass():
    """Welded mega model zeros VBD integration on apple and proxy (teleport path)."""
    fs = _import_fs()
    mega = _build_mega_with_proxy(fs, fix_to_apple=True)
    inv = mega.model.body_inv_mass.numpy()
    inst = mega.instance(0)
    assert inst.apple_body is not None
    assert inv[inst.apple_body] == 0.0
    assert inv[inst.gripper_proxy_body] == 0.0


def test_mega_fd_wrench_block_matches_measure_fruiting_forces_convention():
    """Post-step ``state_0`` + pre-step ``state_1.body_q`` matches coupled wrench gather."""
    fs = _import_fs()
    mfd = _import_mega_fd()
    mega = _build_mega_with_proxy(fs, fix_to_apple=True)
    pipe = _collision_pipeline(fs, mega.model)
    mfd.mega_vbd_substep(mega, DT, collision_pipeline=pipe)

    cable = mega.as_single_instance_coupled(0)
    out = fs.measure_fruiting_forces(
        cable,
        cable.state_0.body_q,
        cable.state_1.body_q,
        dt=DT,
    )
    from apple_pick_sim.coupled_fruiting.stem import _find_stem_apple_joint

    stem_j = _find_stem_apple_joint(cable)
    stem_rec = next(r for r in out["fixed_joints"] if r.joint_index == stem_j)
    gather = np.concatenate(
        [stem_rec.force_world, stem_rec.torque_at_child_com_world], dtype=np.float64
    )
    feat = mfd.default_mega_fd_features(mega, 0, dt=DT)
    np.testing.assert_allclose(
        feat[WRENCH_SLICE],
        gather,
        rtol=1e-5,
        atol=1e-4,
        err_msg="mega_fd wrench block must match measure_fruiting_forces on coupled view",
    )


def test_jacobian_shape_includes_wrench_block_when_welded():
    """Welded mega FD step yields ``(12, num_instances - 1)`` Jacobian."""
    fs = _import_fs()
    mfd = _import_mega_fd()
    mega = _build_mega_with_proxy(fs, fix_to_apple=True)
    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=SEED)
    segs = fs.enabled_rod_segments(nominal)
    cols = fs.fd_stiffness_param_columns(nominal, EPS)
    mega_fd = fs.MegaCoupledCableScene.build(
        cols,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True),
        **_fd_kw(),
    )
    assert mfd.default_mega_fd_features(mega, 0).size == 12
    pipe = _collision_pipeline(fs, mega_fd.model)
    result = mfd.mega_fd_step(mega_fd, EPS, dt=DT, collision_pipeline=pipe)
    n = mega_fd.num_instances
    assert result.features.shape == (n, 12)
    assert result.jacobian.shape == (12, n - 1)
    assert len(segs) == n - 1
