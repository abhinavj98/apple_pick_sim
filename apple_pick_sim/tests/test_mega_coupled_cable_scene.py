"""Tests for multi-instance mega VBD cable model (``MegaCoupledCableScene``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apple_pick_sim.tests.conftest import COUPLED_BASE_POS, NO_SELF_COLLISION_KW, RANGES_FIXTURE


def _import_fs():
    import apple_pick_sim.fruiting_system as fs

    return fs


@pytest.fixture(scope="module", autouse=True)
def _warp_init():
    import warp as wp

    wp.init()


def test_fd_stiffness_param_columns_length_and_nominal_first():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=3)
    eps = 0.01
    cols = fs.fd_stiffness_param_columns(nominal, eps)
    segs = fs.enabled_rod_segments(nominal)
    assert len(cols) == 1 + len(segs)
    assert cols[0] == nominal
    for i, seg in enumerate(segs, start=1):
        rod = getattr(cols[i], seg)
        assert rod is not None
        nom_rod = getattr(nominal, seg)
        assert nom_rod is not None
        assert rod.bend_stiffness == pytest.approx(nom_rod.bend_stiffness + eps)
        assert rod.length == nom_rod.length


def test_mega_body_count_scales_with_instances():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    seed = 5
    single = fs.generate_coupled_cable_scene(
        ranges, seed=seed, base_pos=COUPLED_BASE_POS, **NO_SELF_COLLISION_KW
    )
    mega = fs.generate_mega_coupled_cable_scene(
        ranges,
        seed=seed,
        stiffness_epsilon=0.02,
        base_pos=COUPLED_BASE_POS,
        **NO_SELF_COLLISION_KW,
    )
    n = mega.num_instances
    assert n == 1 + len(fs.enabled_rod_segments(single.params))
    assert mega.model.body_count == n * single.model.body_count


def test_mega_instances_have_distinct_stiffness_fingerprints():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    mega = fs.generate_mega_coupled_cable_scene(
        ranges,
        seed=0,
        stiffness_epsilon=0.05,
        base_pos=COUPLED_BASE_POS,
        **NO_SELF_COLLISION_KW,
    )
    fps = [fs.params_fingerprint(inst.params) for inst in mega.instances]
    assert len({str(fp) for fp in fps}) == len(fps)


def _cable_joint_bend_ke_values(mega) -> dict[int, set[float]]:
    """Per-instance set of cable-joint bend ``target_ke`` (N·m/rad) on the built ``Model``."""
    ke = mega.model.joint_target_ke.numpy()
    jds = mega.model.joint_qd_start.numpy()
    out: dict[int, set[float]] = {}
    for inst in mega.instances:
        bends: set[float] = set()
        for j in inst.cable_joint_indices:
            d = int(jds[int(j)])
            bends.add(float(ke[d + 1]))
        out[inst.index] = bends
    return out


def test_mega_model_joint_bend_ke_matches_fd_columns():
    """FD columns differ in ``Model.joint_target_ke``, not only in ``FruitingInstanceLayout.params``."""
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    nominal = fs.sample_params(ranges, seed=0)
    eps = 0.05
    cols = fs.fd_stiffness_param_columns(nominal, eps)
    mega = fs.MegaCoupledCableScene.build(
        cols,
        base_pos=COUPLED_BASE_POS,
        **NO_SELF_COLLISION_KW,
    )
    bend_by_inst = _cable_joint_bend_ke_values(mega)
    nom_bends = bend_by_inst[0]
    segs = fs.enabled_rod_segments(nominal)
    for i, seg in enumerate(segs, start=1):
        inst = mega.instance(i)
        rod = getattr(inst.params, seg)
        assert rod is not None
        assert any(
            b == pytest.approx(rod.bend_stiffness, rel=0.0, abs=1e-4)
            for b in bend_by_inst[i]
        ), f"instance {i}: no joint bend_ke matches {seg} bend_stiffness"
        assert bend_by_inst[i] != nom_bends, (
            f"instance {i} cable bend_ke identical to nominal; FD column should perturb {seg!r}"
        )


def test_mega_instance_positions_offset_along_spacing():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    spacing = (0.0, 2.0, 0.0)
    nominal = fs.sample_params(ranges, seed=1)
    mega = fs.MegaCoupledCableScene.build(
        [nominal, fs.copy_fruiting_params(nominal)],
        base_pos=COUPLED_BASE_POS,
        instance_spacing=spacing,
        **NO_SELF_COLLISION_KW,
    )
    assert mega.num_instances == 2
    bq = mega.state_0.body_q.numpy().reshape(-1, 7)
    p0 = bq[mega.instance(0).primary_bodies[0], :3]
    p1 = bq[mega.instance(1).primary_bodies[0], :3]
    np.testing.assert_allclose(p1 - p0, spacing, atol=1e-4)


def test_mega_instance_indices_are_contiguous_from_zero():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    mega = fs.generate_mega_coupled_cable_scene(
        ranges,
        seed=1,
        stiffness_epsilon=0.01,
        base_pos=COUPLED_BASE_POS,
        **NO_SELF_COLLISION_KW,
    )
    indices = [inst.index for inst in mega.instances]
    assert indices == list(range(len(indices)))


def test_mega_vbd_substep_finite():
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    mega = fs.generate_mega_coupled_cable_scene(
        ranges,
        seed=2,
        stiffness_epsilon=0.01,
        base_pos=COUPLED_BASE_POS,
        **NO_SELF_COLLISION_KW,
    )
    pipe = fs.example_collision_pipeline(mega.model)
    dt = 1.0 / 1800.0
    contacts = pipe.contacts()
    for _ in range(5):
        pipe.collide(mega.state_0, contacts)
        mega.solver.step(
            mega.state_0, mega.state_1, mega.control, contacts, dt
        )
        mega.state_0, mega.state_1 = mega.state_1, mega.state_0
    bq = mega.state_0.body_q.numpy()
    assert np.all(np.isfinite(bq))
