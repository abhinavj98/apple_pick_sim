"""Post-settle branch path length vs nominal spur+stem+apple rest length."""

from __future__ import annotations

import numpy as np
import pytest

from apple_pick_sim.coupled_fruiting.settle_quasi_static import (
    branch_path_length_m,
    count_apples_outside_envelope,
    count_non_quasi_static_envs,
    nominal_spur_stem_apple_length_m,
    path_length_within_nominal,
    per_env_settle_quasi_static_reports,
    per_env_settle_stability_reports,
    print_envelope_coverage_report,
    print_settle_stability_report,
)
from apple_pick_sim.fruiting_system.params import (
    FruitingSystemParams,
    RodParams,
    rod_params_from_vbd_targets,
)


def _params(*, spur_len: float, stem_len: float, apple_r: float) -> FruitingSystemParams:
    rod_kw = dict(
        num_segments=4,
        radius=0.01,
        bend_stiffness=1.0,
        bend_damping=0.1,
        stretch_stiffness=1.0e6,
        density=300.0,
        direction=(0.0, 0.0, -1.0),
    )
    return FruitingSystemParams(
        primary=rod_params_from_vbd_targets(length=0.3, **rod_kw),
        secondary=None,
        spur=rod_params_from_vbd_targets(length=spur_len, **rod_kw),
        stem=rod_params_from_vbd_targets(length=stem_len, **rod_kw),
        apple_radius=apple_r,
        apple_density=700.0,
    )


def test_nominal_spur_stem_apple_length_sums_segments():
    params = _params(spur_len=0.08, stem_len=0.05, apple_r=0.06)
    assert nominal_spur_stem_apple_length_m(params) == pytest.approx(0.19)


def test_path_length_within_nominal_allows_small_slack():
    assert path_length_within_nominal(0.2111, 0.2110)
    assert not path_length_within_nominal(0.30, 0.19)


def test_branch_path_length_straight_chain():
    body_q = np.zeros((4, 7), dtype=np.float64)
    body_q[0, :3] = (0.0, 0.0, 1.0)
    body_q[1, :3] = (0.0, 0.0, 0.5)
    body_q[2, :3] = (0.0, 0.0, 0.0)
    body_q[3, :3] = (0.0, 0.0, -0.06)
    length = branch_path_length_m(
        body_q,
        spur_bodies=[0, 1],
        stem_bodies=[2],
        apple_body=3,
    )
    assert length == pytest.approx(1.06)


def test_per_env_reports_flags_stretched_branch():
    params = _params(spur_len=0.10, stem_len=0.05, apple_r=0.04)
    nom = nominal_spur_stem_apple_length_m(params)
    body_q = np.zeros((3, 7), dtype=np.float64)
    body_q[0, :3] = (0.0, 0.0, 0.0)
    body_q[1, :3] = (0.0, 0.0, -0.10)
    body_q[2, :3] = (0.0, 0.0, -0.30)  # extra sag vs straight nominal 0.19 m path
    reports = per_env_settle_quasi_static_reports(
        body_q,
        params_list=[params],
        spur_bodies=[0],
        stem_bodies=[1],
        apple_body=2,
        bodies_per_world=3,
    )
    assert len(reports) == 1
    assert reports[0].nominal_length_m == pytest.approx(nom)
    assert reports[0].path_length_m > nom
    assert reports[0].is_quasi_static is False


def test_count_non_quasi_static_envs():
    params_ok = _params(spur_len=0.10, stem_len=0.05, apple_r=0.04)
    params_bad = _params(spur_len=0.10, stem_len=0.05, apple_r=0.04)
    nom_ok = nominal_spur_stem_apple_length_m(params_ok)
    body_q = np.zeros((6, 7), dtype=np.float64)
    # world 0: straight rest path
    body_q[0, :3] = (0.0, 0.0, 0.0)
    body_q[1, :3] = (0.0, 0.0, -0.10)
    body_q[2, :3] = (0.0, 0.0, -0.15 - params_ok.apple_radius)
    # world 1: stretched path
    body_q[3, :3] = (0.0, 0.0, 0.0)
    body_q[4, :3] = (0.0, 0.0, -0.10)
    body_q[5, :3] = (0.0, 0.0, -0.40)
    count, reports = count_non_quasi_static_envs(
        body_q,
        params_list=[params_ok, params_bad],
        spur_bodies=[0],
        stem_bodies=[1],
        apple_body=2,
        bodies_per_world=3,
    )
    assert reports[0].path_length_m == pytest.approx(nom_ok, rel=1e-3)
    assert reports[0].is_quasi_static is True
    assert reports[1].is_quasi_static is False
    assert count == 1


def test_per_env_stability_reports_flags_motion_and_fall():
    params = _params(spur_len=0.10, stem_len=0.05, apple_r=0.04)
    nom = nominal_spur_stem_apple_length_m(params)
    body_q = np.zeros((3, 7), dtype=np.float64)
    body_q[0, :3] = (0.0, 0.0, 0.0)
    body_q[1, :3] = (0.0, 0.0, -0.10)
    body_q[2, :3] = (0.0, 0.0, -0.15 - params.apple_radius)
    body_qd = np.zeros((3, 6), dtype=np.float64)
    body_qd[2, 0] = 0.2  # apple still moving

    reports = per_env_settle_stability_reports(
        body_q,
        body_qd,
        params_list=[params],
        spur_bodies=[0],
        stem_bodies=[1],
        apple_body=2,
        bodies_per_world=3,
        max_branch_speed_m_s=0.05,
        apple_z_min_m=0.0,
    )
    assert len(reports) == 1
    assert reports[0].path_length_m == pytest.approx(nom, rel=1e-3)
    assert reports[0].is_quasi_static is True
    assert reports[0].apple_z_m == pytest.approx(-0.19)
    assert reports[0].max_branch_speed_m_s == pytest.approx(0.2)
    assert reports[0].is_stable is False
    assert "residual_motion" in reports[0].issues
    assert "apple_below_floor" in reports[0].issues


def test_print_settle_stability_report_includes_per_env_lines(capsys):
    params = _params(spur_len=0.10, stem_len=0.05, apple_r=0.04)
    body_q = np.zeros((6, 7), dtype=np.float64)
    body_q[0, :3] = (0.0, 0.0, 1.0)
    body_q[1, :3] = (0.0, 0.0, 0.90)
    body_q[2, :3] = (0.0, 0.0, 0.85)
    body_q[3, :3] = (0.0, 0.0, 1.0)
    body_q[4, :3] = (0.0, 0.0, 0.90)
    body_q[5, :3] = (0.0, 0.0, 0.40)
    body_qd = np.zeros((6, 6), dtype=np.float64)
    reports = per_env_settle_stability_reports(
        body_q,
        body_qd,
        params_list=[params, params],
        spur_bodies=[0],
        stem_bodies=[1],
        apple_body=2,
        bodies_per_world=3,
        apple_z_min_m=0.0,
    )
    print_settle_stability_report(reports)
    out = capsys.readouterr().out
    assert "env0: STABLE" in out
    assert "env1: UNSTABLE" in out
    assert "branch_path>nominal" in out
    assert "1/2 envs stable" in out


def test_count_apples_outside_envelope_all_inside():
    results = [(0.01, 0.01, True), (0.02, 0.02, True)]
    outside, pct = count_apples_outside_envelope(results)
    assert outside == 0
    assert pct == pytest.approx(0.0)


def test_count_apples_outside_envelope_some_outside():
    results = [(0.01, 0.01, True), (0.10, 0.10, False)]
    outside, pct = count_apples_outside_envelope(results)
    assert outside == 1
    assert pct == pytest.approx(50.0)


def test_count_apples_outside_envelope_all_outside():
    results = [(0.10, 0.10, False), (0.20, 0.20, False)]
    outside, pct = count_apples_outside_envelope(results)
    assert outside == 2
    assert pct == pytest.approx(100.0)


def test_count_apples_outside_envelope_empty():
    outside, pct = count_apples_outside_envelope([])
    assert outside == 0
    assert pct == pytest.approx(0.0)


def test_print_envelope_coverage_report_output(capsys):
    results = [(0.01, 0.01, True), (0.10, 0.10, False)]
    print_envelope_coverage_report(results)
    out = capsys.readouterr().out
    assert "working envelope" in out
    assert "env0: INSIDE" in out
    assert "env1: OUTSIDE" in out
    assert "50.0%" in out
    assert "1/2 envs outside envelope" in out


def test_print_envelope_coverage_report_includes_stability(capsys):
    params = _params(spur_len=0.10, stem_len=0.05, apple_r=0.04)
    body_q = np.zeros((6, 7), dtype=np.float64)
    body_q[0, :3] = (0.0, 0.0, 1.0)
    body_q[1, :3] = (0.0, 0.0, 0.90)
    body_q[2, :3] = (0.0, 0.0, 0.85)
    body_q[3, :3] = (0.0, 0.0, 1.0)
    body_q[4, :3] = (0.0, 0.0, 0.90)
    body_q[5, :3] = (0.0, 0.0, 0.40)
    body_qd = np.zeros((6, 6), dtype=np.float64)
    stability = per_env_settle_stability_reports(
        body_q,
        body_qd,
        params_list=[params, params],
        spur_bodies=[0],
        stem_bodies=[1],
        apple_body=2,
        bodies_per_world=3,
        apple_z_min_m=0.0,
    )
    ik_results = [(0.01, 0.01, True), (0.10, 0.10, False)]
    print_envelope_coverage_report(ik_results, stability_reports=stability)
    out = capsys.readouterr().out
    assert "stability" in out.lower()
    assert "env0: STABLE" in out
    assert "env1: UNSTABLE" in out
    assert "INSIDE" in out
    assert "OUTSIDE" in out
    assert "1/2 envs stable" in out
    assert "1/2 envs outside envelope" in out
