"""Tests for FR3 keyboard teleop velocity helpers."""

from __future__ import annotations

import numpy as np

from apple_pick_sim.robot.fr3_robot.controllers.keyboard import (
    EEVelocity,
    add_gaussian_noise_to_ee_velocity,
)


def test_add_gaussian_noise_to_ee_velocity_is_deterministic():
    base = EEVelocity(linear=(0.5, 0.0, 0.0), angular=(0.1, 0.0, 0.0))
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    noisy_a = add_gaussian_noise_to_ee_velocity(base, rng=rng_a, std=0.02)
    noisy_b = add_gaussian_noise_to_ee_velocity(base, rng=rng_b, std=0.02)
    assert noisy_a == noisy_b
    assert noisy_a != base


def test_add_gaussian_noise_to_ee_velocity_per_call_differs():
    base = EEVelocity(linear=(1.0, 0.0, 0.0))
    rng = np.random.default_rng(0)
    first = add_gaussian_noise_to_ee_velocity(base, rng=rng, std=0.05)
    second = add_gaussian_noise_to_ee_velocity(base, rng=rng, std=0.05)
    assert first != second
