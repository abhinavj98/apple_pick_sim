"""Sensor-realistic ft_wrist: causal EMA + per-env bias + per-step noise + slow drift.

Lives strictly on the gym observation path (docs/handbook-sysid-scoring.md's
"No sim EMA/LPF" rule governs the batched_sysid_v1 feature bags, not this).
"""

from __future__ import annotations

import math

import pytest
import torch

from apple_pick_gym.batched_envs.sensor_realism import FtSensorConfig, FtSensorModel


def _analytic_alpha(cutoff_hz: float, control_hz: float) -> float:
    return 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / control_hz)


def test_alpha_matches_documented_60hz_10hz_value():
    alpha = _analytic_alpha(cutoff_hz=10.0, control_hz=60.0)
    assert alpha == pytest.approx(0.65, abs=0.01)


def test_ema_passes_dc_unchanged():
    cfg = FtSensorConfig(control_hz=60.0, cutoff_hz=10.0, bias_std=0.0, noise_std=0.0, drift_std=0.0)
    model = FtSensorModel(num_envs=2, device="cpu", config=cfg)
    model.reset()
    raw = torch.full((2, 6), 5.0)
    out = raw
    for _ in range(200):
        out = model.step(raw)
    torch.testing.assert_close(out, raw, atol=1e-3, rtol=0)


def test_ema_attenuates_nyquist_frequency_by_analytic_factor():
    """Alternating +C/-C input (Nyquist, f = control_hz/2) settles to a steady-state
    peak-to-peak amplitude ratio matching the discrete EMA frequency response
    |H(pi)| = a / (2 - a)."""
    cfg = FtSensorConfig(control_hz=60.0, cutoff_hz=10.0, bias_std=0.0, noise_std=0.0, drift_std=0.0)
    model = FtSensorModel(num_envs=1, device="cpu", config=cfg)
    model.reset()
    c = 10.0
    outs = []
    for n in range(400):
        raw = torch.full((1, 6), c if n % 2 == 0 else -c)
        outs.append(model.step(raw)[0, 0].item())
    steady = outs[-20:]
    measured_amplitude = (max(steady) - min(steady)) / 2.0
    expected_ratio = model.alpha / (2.0 - model.alpha)
    expected_amplitude = c * expected_ratio
    assert abs(measured_amplitude - expected_amplitude) / expected_amplitude < 0.05


def test_bias_is_constant_within_episode_and_varies_per_env():
    cfg = FtSensorConfig(control_hz=60.0, cutoff_hz=10.0, bias_std=2.0, noise_std=0.0, drift_std=0.0)
    model = FtSensorModel(num_envs=8, device="cpu", config=cfg, generator=torch.Generator().manual_seed(0))
    model.reset()
    raw = torch.zeros((8, 6))
    out1 = model.step(raw)
    out2 = model.step(raw)
    # Same env, same step-to-step bias contribution (no noise/drift configured here).
    torch.testing.assert_close(out1, out2, atol=1e-6, rtol=0)
    # Different envs get different bias draws.
    assert not torch.allclose(out1[0], out1[1])


def test_bias_resamples_on_reset():
    cfg = FtSensorConfig(control_hz=60.0, cutoff_hz=10.0, bias_std=2.0, noise_std=0.0, drift_std=0.0)
    model = FtSensorModel(num_envs=4, device="cpu", config=cfg, generator=torch.Generator().manual_seed(1))
    model.reset()
    raw = torch.zeros((4, 6))
    before = model.step(raw).clone()
    model.reset()
    after = model.step(raw).clone()
    assert not torch.allclose(before, after)


def test_ema_state_resets_no_transient_ramp_from_zero():
    """A causal EMA seeded at 0 would ramp up slowly after reset; this model
    seeds the EMA with the first post-reset sample instead (no artificial
    warm-up transient)."""
    cfg = FtSensorConfig(control_hz=60.0, cutoff_hz=10.0, bias_std=0.0, noise_std=0.0, drift_std=0.0)
    model = FtSensorModel(num_envs=1, device="cpu", config=cfg)
    model.reset()
    raw = torch.full((1, 6), 7.0)
    first_out = model.step(raw)
    torch.testing.assert_close(first_out, raw, atol=1e-6, rtol=0)


def test_drift_resets_to_zero_and_accumulates_within_episode():
    cfg = FtSensorConfig(control_hz=60.0, cutoff_hz=10.0, bias_std=0.0, noise_std=0.0, drift_std=1.0)
    model = FtSensorModel(num_envs=1, device="cpu", config=cfg, generator=torch.Generator().manual_seed(2))
    model.reset()
    raw = torch.zeros((1, 6))
    first = model.step(raw).clone()
    for _ in range(50):
        model.step(raw)
    later = model.step(raw).clone()
    assert not torch.allclose(first, later), "drift must accumulate over the episode"
    model.reset()
    reset_out = model.step(raw)
    assert abs(reset_out[0, 0].item()) < abs(later[0, 0].item()), "drift must reset toward zero on reset()"


def test_masked_reset_only_affects_selected_envs():
    cfg = FtSensorConfig(control_hz=60.0, cutoff_hz=10.0, bias_std=3.0, noise_std=0.0, drift_std=0.0)
    model = FtSensorModel(num_envs=3, device="cpu", config=cfg, generator=torch.Generator().manual_seed(3))
    model.reset()
    raw = torch.zeros((3, 6))
    before = model.step(raw).clone()
    mask = torch.tensor([False, True, False])
    model.reset(env_mask=mask)
    after = model.step(raw)
    torch.testing.assert_close(after[0], before[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(after[2], before[2], atol=1e-6, rtol=0)
    assert not torch.allclose(after[1], before[1])


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])


def test_per_channel_std_scales_force_and_torque_separately():
    """Force (N) and torque (N*m) differ by ~2 orders of magnitude; one scalar std cannot
    model both, so every std/clip also accepts a 6-tuple [Fx,Fy,Fz,Tx,Ty,Tz]."""
    std = (2.0, 2.0, 2.0, 0.01, 0.01, 0.01)
    cfg = FtSensorConfig(bias_std=std, noise_std=0.0, drift_std=0.0)
    model = FtSensorModel(num_envs=4000, device="cpu", config=cfg, generator=torch.Generator().manual_seed(0))
    model.reset()
    out = model.step(torch.zeros(4000, 6))
    measured = out.std(dim=0)
    torch.testing.assert_close(measured, torch.tensor(std), rtol=0.1, atol=0.0)


def test_per_channel_drift_clip():
    clip = (1.0, 1.0, 1.0, 0.01, 0.01, 0.01)
    cfg = FtSensorConfig(bias_std=0.0, noise_std=0.0, drift_std=10.0, drift_clip=clip)
    model = FtSensorModel(num_envs=8, device="cpu", config=cfg, generator=torch.Generator().manual_seed(1))
    model.reset()
    for _ in range(20):
        out = model.step(torch.zeros(8, 6))
    assert torch.all(out.abs() <= torch.tensor(clip) + 1e-6)
    assert torch.all(out[:, 3:].abs().amax(dim=0) > 0.009)


def test_rl_training_preset_turns_sensor_dr_on():
    cfg = FtSensorConfig.rl_training()
    assert cfg.control_hz == 60.0 and cfg.cutoff_hz == 10.0
    for field in (cfg.bias_std, cfg.noise_std, cfg.drift_std, cfg.drift_clip):
        vals = torch.as_tensor(field, dtype=torch.float32)
        assert vals.shape == (6,) and bool(torch.all(vals > 0))


def test_rl_training_noise_matches_the_real_rig():
    """Per-channel noise measured on the real rig's quiet unloaded holds (s02, 32 segments,
    ft_wrist_raw, 60 Hz block mean): F ~0.10-0.12 N, Tx/Ty ~0.033-0.050 N*m, Tz ~0.004 N*m."""
    noise = torch.as_tensor(FtSensorConfig.rl_training().noise_std)
    torch.testing.assert_close(noise, torch.tensor([0.12, 0.11, 0.12, 0.04, 0.05, 0.005]))
    assert float(noise[3]) > 5 * float(noise[5])  # Tx/Ty are an order noisier than Tz
