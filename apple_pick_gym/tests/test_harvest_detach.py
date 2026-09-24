"""Elliptical (force + torque) fruit-detachment envelope at the spur-stem junction.

``(F/F_max)^2 + (tau/tau_max)^2 >= 1`` with ``F_max = 20 N`` and
``tau_max = 0.05 N*m`` (maintainer decision, 2026-09-24). The torque that enters
the envelope is the couple the junction itself carries -- the solver reports
torque about the *child body's COM*, which carries an extra ``r x F`` lever term
that is not junction load, so it is shifted back to the joint anchor first.
"""

from __future__ import annotations

import math

import pytest
import torch

from apple_pick_gym.batched_envs.harvest_detach import (
    DetachEnvelopeConfig,
    detach_index,
    detach_utilization,
    junction_wrench_at_anchor,
)


def _wrench(f, t):
    return torch.tensor([list(f) + list(t)], dtype=torch.float32)


def test_default_envelope_is_20n_and_0p05nm():
    cfg = DetachEnvelopeConfig()
    assert cfg.f_max_n == 20.0
    assert cfg.tau_max_nm == 0.05


def test_pure_force_detaches_exactly_at_f_max():
    cfg = DetachEnvelopeConfig()
    idx = detach_index(torch.cat([_wrench((0, 0, 20.0), (0, 0, 0)), _wrench((0, 0, 19.9), (0, 0, 0))]), cfg)
    assert idx[0] == pytest.approx(1.0)
    assert idx[1] < 1.0


def test_pure_torque_detaches_exactly_at_tau_max():
    cfg = DetachEnvelopeConfig()
    idx = detach_index(_wrench((0, 0, 0), (0.03, 0.04, 0.0)), cfg)  # |tau| = 0.05
    assert idx[0] == pytest.approx(1.0)


def test_twist_and_pull_synergy():
    """Torque weakens the pull resistance: 0.7*F_max alone holds, but with 0.75*tau_max it fails."""
    cfg = DetachEnvelopeConfig()
    pull_only = detach_index(_wrench((14.0, 0, 0), (0, 0, 0)), cfg)
    twist_and_pull = detach_index(_wrench((14.0, 0, 0), (0, 0, 0.0375)), cfg)
    assert pull_only[0] == pytest.approx(0.49)
    assert twist_and_pull[0] == pytest.approx(0.49 + 0.5625)
    assert twist_and_pull[0] >= 1.0 > pull_only[0]


def test_index_uses_vector_norms_not_components():
    cfg = DetachEnvelopeConfig()
    a = detach_index(_wrench((12.0, 16.0, 0.0), (0, 0, 0)), cfg)  # |F| = 20
    assert a[0] == pytest.approx(1.0)


def test_utilization_is_sqrt_of_index():
    cfg = DetachEnvelopeConfig()
    w = torch.cat([_wrench((10.0, 0, 0), (0, 0, 0)), _wrench((0, 0, 0), (0, 0, 0.1))])
    u = detach_utilization(w, cfg)
    torch.testing.assert_close(u, torch.tensor([0.5, 2.0]))


def test_rejects_non_positive_limits():
    with pytest.raises(ValueError):
        DetachEnvelopeConfig(f_max_n=0.0)
    with pytest.raises(ValueError):
        DetachEnvelopeConfig(tau_max_nm=-1.0)


def test_anchor_shift_removes_com_lever_term():
    """Child COM 1 cm below the anchor, 10 N sideways force applied at the anchor, no couple.

    Torque about the COM is (anchor - com) x F = (0,0,0.01) x (10,0,0) = (0, 0.1, 0);
    the junction itself carries no couple, so the anchor-frame torque is zero.
    """
    force = torch.tensor([[10.0, 0.0, 0.0]])
    anchor = torch.tensor([[0.0, 0.0, 0.0]])
    com = torch.tensor([[0.0, 0.0, -0.01]])
    torque_com = torch.cross(anchor - com, force, dim=-1)
    torch.testing.assert_close(torque_com, torch.tensor([[0.0, 0.1, 0.0]]))
    w = junction_wrench_at_anchor(torch.cat([force, torque_com], dim=-1), child_anchor=anchor, child_com=com)
    torch.testing.assert_close(w[:, :3], force)
    torch.testing.assert_close(w[:, 3:], torch.zeros(1, 3))


def test_anchor_shift_keeps_pure_couple():
    force = torch.zeros(1, 3)
    couple = torch.tensor([[0.0, 0.0, 0.02]])
    w = junction_wrench_at_anchor(
        torch.cat([force, couple], dim=-1),
        child_anchor=torch.tensor([[1.0, 2.0, 3.0]]),
        child_com=torch.tensor([[1.0, 2.0, 2.9]]),
    )
    torch.testing.assert_close(w[:, 3:], couple)


def test_anchor_shift_is_batched():
    n = 5
    g = torch.Generator().manual_seed(0)
    force = torch.randn(n, 3, generator=g)
    couple = torch.randn(n, 3, generator=g) * 0.01
    anchor = torch.randn(n, 3, generator=g)
    com = anchor + torch.randn(n, 3, generator=g) * 0.01
    torque_com = couple + torch.cross(anchor - com, force, dim=-1)
    w = junction_wrench_at_anchor(torch.cat([force, torque_com], dim=-1), child_anchor=anchor, child_com=com)
    torch.testing.assert_close(w[:, 3:], couple, atol=1e-6, rtol=1e-5)


def test_lever_term_matters_at_this_tau_max():
    """Why the anchor shift is not optional: a 2.5 mm COM offset at F_max is already tau_max."""
    cfg = DetachEnvelopeConfig()
    lever = 0.0025
    assert lever * cfg.f_max_n == pytest.approx(cfg.tau_max_nm)
    assert math.isclose(lever * 20.0, 0.05)


# ---------------------------------------------------------------- torsion / bending split
from apple_pick_gym.batched_envs.harvest_detach import split_torque  # noqa: E402


def test_split_torque_into_torsion_and_bending():
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    tors, bend = split_torque(torch.tensor([[0.3, 0.4, 0.02]]), axis)
    assert float(tors[0]) == pytest.approx(0.02)
    assert float(bend[0]) == pytest.approx(0.5)
    tors2, _ = split_torque(torch.tensor([[0.0, 0.0, -0.02]]), -axis)  # axis sign does not matter
    assert float(tors2[0]) == pytest.approx(0.02)


def test_total_mode_is_the_default_and_unchanged():
    cfg = DetachEnvelopeConfig()
    assert cfg.torque_mode == "total"
    w = _wrench((0, 0, 0), (0.5, 0, 0))  # pure bending, 10x tau_max
    assert detach_index(w, cfg)[0] == pytest.approx(100.0)


def test_split_mode_gives_bending_and_torsion_their_own_limits():
    cfg = DetachEnvelopeConfig(torque_mode="split", torsion_max_nm=0.05, bending_max_nm=0.9)
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    bend = _wrench((0, 0, 0), (0.45, 0, 0))
    twist = _wrench((0, 0, 0), (0, 0, 0.05))
    both = _wrench((10.0, 0, 0), (0.45, 0, 0.025))
    assert detach_index(bend, cfg, stem_axis=axis)[0] == pytest.approx(0.25)
    assert detach_index(twist, cfg, stem_axis=axis)[0] == pytest.approx(1.0)
    assert detach_index(both, cfg, stem_axis=axis)[0] == pytest.approx(0.25 + 0.25 + 0.25)


def test_split_mode_requires_the_stem_axis():
    with pytest.raises(ValueError, match="stem_axis"):
        detach_index(_wrench((0, 0, 1), (0, 0, 0)), DetachEnvelopeConfig(torque_mode="split"))


def test_split_limits_must_be_positive():
    with pytest.raises(ValueError):
        DetachEnvelopeConfig(torque_mode="split", bending_max_nm=0.0)


def test_wrench_source_defaults_to_the_stem_elastic_joint():
    """[D1] the rigid junction's constraint readout carries a +-0.03 N*m solver-noise floor; the
    first soft stem joint's elastic wrench, shifted to the junction by statics, has the same mean
    and ~65x less step-to-step noise (CPU, v1 worlds)."""
    assert DetachEnvelopeConfig().wrench_source == "stem_elastic"
    DetachEnvelopeConfig(wrench_source="junction_readout")
    with pytest.raises(ValueError):
        DetachEnvelopeConfig(wrench_source="bogus")


def test_stem_root_statics_shift():
    """Moment about the junction J of a wrench (F, M_A) acting at A: M_J = M_A + (A - J) x F."""
    from apple_pick_gym.batched_envs.harvest_detach import shift_moment

    F = torch.tensor([[0.0, 0.0, -5.0]])
    M_A = torch.tensor([[0.001, 0.0, 0.0]])
    A = torch.tensor([[0.0024, 0.0, 0.0]])
    J = torch.zeros(1, 3)
    torch.testing.assert_close(shift_moment(M_A, F, from_point=A, to_point=J), torch.tensor([[0.001, 0.012, 0.0]]))


# --- [D7] per-env envelope thresholds (F_max / tau_max are rough estimates)
import numpy as np  # noqa: E402

from apple_pick_gym.batched_envs.harvest_detach import (  # noqa: E402
    DetachEnvelopeConfig,
    detach_index,
    detach_utilization,
    envelope_thresholds,
)


def test_d7_no_ranges_gives_the_nominal_thresholds():
    cfg = DetachEnvelopeConfig()
    th = envelope_thresholds(cfg, 4, np.random.default_rng(0), device="cpu")
    assert th.shape == (4, 2)
    torch.testing.assert_close(th, torch.tensor([[20.0, 0.05]] * 4))


def test_d7_ranges_sample_per_env_within_bounds_uniform_force_log_uniform_torque():
    cfg = DetachEnvelopeConfig(f_max_range_n=(15.0, 25.0), tau_max_range_nm=(0.04, 0.12))
    th = envelope_thresholds(cfg, 20000, np.random.default_rng(0), device="cpu")
    f, t = th[:, 0], th[:, 1]
    assert float(f.min()) >= 15.0 and float(f.max()) <= 25.0
    assert float(t.min()) >= 0.04 and float(t.max()) <= 0.12
    assert abs(float(f.mean()) - 20.0) < 0.2  # uniform
    assert abs(float(torch.log(t).mean()) - 0.5 * (math.log(0.04) + math.log(0.12))) < 0.01  # log-uniform
    assert len(set(f[:10].tolist())) == 10  # per env


def test_d7_index_uses_per_env_thresholds():
    cfg = DetachEnvelopeConfig()
    w = torch.tensor([[20.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.05, 0.0, 0.0]])
    th = torch.tensor([[10.0, 0.05], [20.0, 0.1]])
    torch.testing.assert_close(detach_index(w, cfg, thresholds=th), torch.tensor([4.0, 0.25]))
    torch.testing.assert_close(detach_utilization(w, cfg, thresholds=th), torch.tensor([2.0, 0.5]))
    torch.testing.assert_close(detach_index(w, cfg), torch.tensor([1.0, 1.0]))  # nominal unchanged


def test_d7_ranges_are_validated():
    with pytest.raises(ValueError):
        DetachEnvelopeConfig(f_max_range_n=(25.0, 15.0))
    with pytest.raises(ValueError):
        DetachEnvelopeConfig(tau_max_range_nm=(0.0, 0.1))
    with pytest.raises(ValueError):  # the split envelope keeps fixed limits
        DetachEnvelopeConfig(torque_mode="split", f_max_range_n=(15.0, 25.0))
