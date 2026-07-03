"""Tests for branch KE envelope decay settle diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from apple_pick_sim.coupled_fruiting.settle_ke_decay import (
    KeSample,
    SettleKeAnalysisConfig,
    _tail_samples,
    branch_linear_kinetic_energy_j,
    envelope_is_decaying,
    find_ke_peaks,
    per_env_settle_ke_decay_reports,
    print_settle_checkpoint_report,
)
from apple_pick_sim.coupled_fruiting.settle_quasi_static import SettleStabilityReport


def test_branch_linear_kinetic_energy_j_known_masses():
    body_qd = np.zeros((3, 6), dtype=np.float64)
    body_qd[0, 0] = 1.0  # 0.5 * 2 * 1^2 = 1 J
    body_qd[1, 1] = 2.0  # 0.5 * 1 * 4 = 2 J
    masses = np.array([2.0, 1.0, 0.5], dtype=np.float64)
    ke = branch_linear_kinetic_energy_j(body_qd, masses, [0, 1, 2])
    assert ke == pytest.approx(3.0)


def test_find_ke_peaks_detects_local_maxima():
    ke = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0], dtype=np.float64)
    peaks = find_ke_peaks(ke)
    assert peaks == [1, 3, 5]


def test_envelope_is_decaying_true_for_damped_sine():
    t = np.linspace(0.0, 10.0, 500)
    ke = 0.5 * np.exp(-0.3 * t) ** 2 * (1.0 + np.sin(4.0 * t))
    peaks_idx = find_ke_peaks(ke)
    peak_vals = [float(ke[i]) for i in peaks_idx]
    assert len(peak_vals) >= 3
    assert envelope_is_decaying(peak_vals, min_peaks=3, peak_decay_rtol=0.05)


def test_envelope_is_decaying_false_for_constant_amplitude():
    t = np.linspace(0.0, 10.0, 500)
    ke = 0.5 * (1.0 + np.sin(4.0 * t))
    peaks_idx = find_ke_peaks(ke)
    peak_vals = [float(ke[i]) for i in peaks_idx]
    assert len(peak_vals) >= 3
    assert not envelope_is_decaying(peak_vals, min_peaks=3, peak_decay_rtol=0.10)


def test_envelope_is_decaying_false_for_growing():
    peak_vals = [0.01, 0.012, 0.015, 0.018]
    assert not envelope_is_decaying(peak_vals, min_peaks=3, peak_decay_rtol=0.05)


def test_settle_ke_decay_report_strong_gate():
    # Decaying peaks, final peak below threshold
    n = 100
    substeps = np.arange(n, dtype=np.int64)
    sim_times = substeps * 0.001
    ke = 0.01 * np.exp(-0.05 * np.arange(n)) * (1.0 + np.sin(np.linspace(0, 8 * math.pi, n)))
    speed = np.sqrt(2.0 * ke / 0.5)  # pseudo speed for one 0.5 kg body
    reports = per_env_settle_ke_decay_reports(
        [(int(s), float(t), float(k), float(v)) for s, t, k, v in zip(substeps, sim_times, ke, speed)],
        world=0,
        branch_mass_kg=0.5,
        config=SettleKeAnalysisConfig(
            analysis_tail_fraction=1.0,
            min_peaks=3,
            peak_decay_rtol=0.05,
            speed_threshold_m_s=0.05,
            ke_peak_threshold_j=None,
        ),
    )
    assert len(reports) == 1
    report = reports[0]
    assert report.is_envelope_decaying
    assert report.is_ke_below_threshold
    assert report.is_ke_decay_stable


def test_settle_ke_decay_report_growing_envelope_fails():
    n = 100
    substeps = np.arange(n, dtype=np.int64)
    sim_times = substeps * 0.001
    ke = 0.005 * (1.0 + 0.5 * np.arange(n) / n) * (1.0 + np.sin(np.linspace(0, 8 * math.pi, n)))
    speed = np.full(n, 0.2)
    reports = per_env_settle_ke_decay_reports(
        [(int(s), float(t), float(k), float(v)) for s, t, k, v in zip(substeps, sim_times, ke, speed)],
        world=0,
        branch_mass_kg=0.5,
        config=SettleKeAnalysisConfig(
            analysis_tail_fraction=1.0,
            min_peaks=3,
            peak_decay_rtol=0.05,
            speed_threshold_m_s=0.05,
            ke_peak_threshold_j=1.0,
        ),
    )
    report = reports[0]
    assert not report.is_ke_decay_stable
    assert "envelope_growing" in report.issues or not report.is_envelope_decaying


def test_tail_samples_keeps_last_fraction():
    samples = [KeSample(i, float(i) * 0.001, float(i), 0.0) for i in range(100)]
    tail = _tail_samples(samples, analysis_tail_fraction=0.5)
    assert len(tail) == 50
    assert tail[0].substep == 50
    assert tail[-1].substep == 99


def test_analysis_tail_fraction_excludes_early_growing_peaks():
    """Tail window should omit early growing peaks present in the full series."""
    early = [
        KeSample(i, i * 0.001, (0.02 + 0.002 * i) * (1.0 + math.sin(i)), 0.1)
        for i in range(50)
    ]
    late = [
        KeSample(50 + i, (50 + i) * 0.001, 0.03 * math.exp(-0.08 * i) * (1.0 + math.sin(i)), 0.02)
        for i in range(50)
    ]
    samples = early + late
    cfg = SettleKeAnalysisConfig(
        analysis_tail_fraction=0.5,
        min_peaks=3,
        peak_decay_rtol=0.05,
        speed_threshold_m_s=0.05,
        ke_peak_threshold_j=0.5,
    )
    full = per_env_settle_ke_decay_reports(samples, world=0, branch_mass_kg=0.5, config=cfg)
    tail_only = per_env_settle_ke_decay_reports(
        _tail_samples(samples, analysis_tail_fraction=0.5),
        world=0,
        branch_mass_kg=0.5,
        config=SettleKeAnalysisConfig(
            analysis_tail_fraction=1.0,
            min_peaks=3,
            peak_decay_rtol=0.05,
            speed_threshold_m_s=0.05,
            ke_peak_threshold_j=0.5,
        ),
    )
    assert len(full[0].peak_ke_j) >= len(tail_only[0].peak_ke_j)
    assert tail_only[0].is_envelope_decaying


def test_per_env_branch_ke_j_uses_world_count_not_world_start_length():
    """``body_world_start`` has length ``world_count + 2`` in Newton; do not over-iterate."""
    from unittest.mock import MagicMock

    from apple_pick_sim.coupled_fruiting.settle_ke_decay import per_env_branch_ke_j_from_cable

    cable = MagicMock()
    cable.apple_body = 11
    cable.spur_bodies = [5, 6, 7]
    cable.stem_bodies = [8, 9, 10]
    cable.model.world_count = 4
    cable.model.body_world_start.numpy.return_value = np.array([0, 12, 24, 36, 48, 48])
    cable.state_0.body_qd.numpy.return_value = np.zeros((48, 6), dtype=np.float64)
    cable.model.body_mass.numpy.return_value = np.ones(48, dtype=np.float64)

    ke = per_env_branch_ke_j_from_cable(cable)
    assert len(ke) == 4


def test_print_settle_checkpoint_report_combined_ke(capsys):
    reports = [
        SettleStabilityReport(
            world=0,
            path_length_m=0.19,
            nominal_length_m=0.19,
            is_quasi_static=True,
            apple_z_m=0.45,
            apple_speed_m_s=0.1,
            max_branch_speed_m_s=0.1,
            issues=("residual_motion",),
            is_stable=False,
        )
    ]
    print_settle_checkpoint_report(
        reports,
        [0.0123],
        substep_idx=1000,
        sim_time_s=0.556,
    )
    out = capsys.readouterr().out
    assert "branch KE" in out
    assert "ke=0.0123 J" in out
    assert "peaks=" not in out
    assert "KE_DECAY" not in out
