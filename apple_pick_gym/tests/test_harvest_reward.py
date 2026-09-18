"""Dense harvest reward shaping: progress, pull-out, and collateral-force penalties.

Ported from feature/rl-gym's harvest_reward.py (math unchanged) with
f_threshold_n updated to the maintainer's current placeholder (5 N, flagged
for a later revisit -- possibly a combined force+torque criterion). Reward
is privileged: it reads uncapped junction forces and raw ft_wrist from
``info``, not the sensor-realistic ``obs["ft_wrist"]`` the policy sees.
"""

from __future__ import annotations

import math

import torch

from apple_pick_gym.batched_envs.harvest_reward import (
    HarvestRewardConfig,
    compute_collateral_penalty,
    compute_dense_reward,
    compute_progress_reward,
    compute_pullout_penalty,
    quat_rotate_vector,
)


def test_default_threshold_is_5n():
    cfg = HarvestRewardConfig()
    assert cfg.f_threshold_n == 5.0


def test_progress_reward_clips_at_threshold():
    cfg = HarvestRewardConfig(f_threshold_n=5.0)
    force = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5], [0.0, 0.0, 10.0]])
    r = compute_progress_reward(force, cfg)
    torch.testing.assert_close(r, torch.tensor([0.0, 0.5, 1.0]))


def test_quat_rotate_vector_identity():
    ident = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # xyzw identity
    v = torch.tensor([[1.0, 2.0, 3.0]])
    out = quat_rotate_vector(ident, v)
    torch.testing.assert_close(out, v)


def test_quat_rotate_vector_quarter_turn_z():
    # 90 deg about world Z: xyzw = (0,0,sin(45),cos(45))
    s = math.sqrt(0.5)
    q = torch.tensor([[0.0, 0.0, s, s]])
    v = torch.tensor([[1.0, 0.0, 0.0]])
    out = quat_rotate_vector(q, v)
    torch.testing.assert_close(out, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6, rtol=0)


def test_pullout_penalty_zero_when_force_perpendicular_to_ee_z():
    ident = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    ft_wrist = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # force along world x, ee_z=world z
    p = compute_pullout_penalty(ft_wrist, ident)
    assert p.item() == 0.0


def test_pullout_penalty_positive_along_ee_z_clamped_at_zero_when_negative():
    ident = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    pull_out = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]])
    push_in = torch.tensor([[0.0, 0.0, -5.0, 0.0, 0.0, 0.0]])
    assert compute_pullout_penalty(pull_out, ident).item() == 5.0
    assert compute_pullout_penalty(push_in, ident).item() == 0.0


def test_collateral_penalty_sums_non_target_junctions():
    woody = {
        "spur_stem": torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),  # target, excluded
        "stem_apple": torch.tensor([[3.0, 4.0, 0.0, 0.0, 0.0, 0.0]]),  # norm 5
        "primary_spur": torch.tensor([[0.0, 0.0, 2.0, 0.0, 0.0, 0.0]]),  # norm 2
    }
    total = compute_collateral_penalty(woody, target_junction_name="spur_stem")
    torch.testing.assert_close(total, torch.tensor([7.0]))


def test_collateral_penalty_raises_when_only_target_junction_present():
    woody = {"spur_stem": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])}
    try:
        compute_collateral_penalty(woody, target_junction_name="spur_stem")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_dense_reward_combines_weighted_terms():
    cfg = HarvestRewardConfig(
        f_threshold_n=5.0, w_progress=1.0, w_pullout=0.5, w_collateral=0.1
    )
    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]])}
    info = {
        "target_junction_force": torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]]),  # progress=1.0
        "ft_wrist": torch.tensor([[0.0, 0.0, 2.0, 0.0, 0.0, 0.0]]),  # pullout=2.0
        "woody_part_force": {
            "spur_stem": torch.zeros(1, 6),
            "stem_apple": torch.tensor([[3.0, 4.0, 0.0, 0.0, 0.0, 0.0]]),  # collateral=5.0
        },
    }
    reward = compute_dense_reward(obs, info, target_junction_name="spur_stem", cfg=cfg)
    expected = 1.0 * 1.0 - 0.5 * 2.0 - 0.1 * 5.0
    assert reward.shape == (1, 1)
    torch.testing.assert_close(reward, torch.tensor([[expected]]))


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
