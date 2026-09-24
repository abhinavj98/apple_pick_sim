"""Dense harvest reward shaping: progress, pull-out, and collateral-force penalties.

Progress is the fraction of the spur-stem detach envelope in use
(``harvest_detach.py``: ``(F/20 N)^2 + (tau/0.05 N*m)^2 >= 1``), replacing the old
force-only threshold. Collateral is measured against each junction's rest load.
Reward is privileged: it reads uncapped junction wrenches and raw ft_wrist from
``info``, not the sensor-realistic ``obs["ft_wrist"]`` the policy sees.
"""

from __future__ import annotations

import math

import torch

from apple_pick_gym.batched_envs.harvest_detach import DetachEnvelopeConfig
from apple_pick_gym.batched_envs.harvest_reward import (
    HarvestRewardConfig,
    compute_collateral_penalty,
    compute_dense_reward,
    compute_progress_reward,
    compute_pullout_penalty,
    quat_rotate_vector,
)


def test_default_success_is_the_20n_0p05nm_envelope():
    cfg = HarvestRewardConfig()
    assert cfg.detach == DetachEnvelopeConfig(f_max_n=20.0, tau_max_nm=0.05)
    assert not hasattr(cfg, "f_threshold_n")


def test_default_progress_is_delta_and_slack_is_on():
    cfg = HarvestRewardConfig()
    assert cfg.progress_mode == "delta"
    assert cfg.w_slack > 0.0


def test_slack_is_a_constant_per_step_cost():
    from apple_pick_gym.batched_envs.harvest_reward import compute_dense_reward_terms, weight_dense_reward_terms

    cfg = HarvestRewardConfig(w_slack=0.02)
    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)}
    info = {
        "target_junction_wrench": torch.zeros(2, 6),
        "ft_wrist": torch.zeros(2, 6),
        "woody_part_force": {"spur_stem": torch.zeros(2, 6), "stem_apple": torch.zeros(2, 6)},
    }
    raw = compute_dense_reward_terms(obs, info, target_junction_name="spur_stem", cfg=cfg)
    torch.testing.assert_close(raw["slack"], torch.ones(2))
    torch.testing.assert_close(weight_dense_reward_terms(raw, cfg)["slack"], torch.full((2,), -0.02))


def test_progress_reward_is_envelope_utilization_clipped_at_one():
    cfg = HarvestRewardConfig()
    wrench = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],  # half of F_max
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.025],  # half of tau_max
            [0.0, 0.0, 40.0, 0.0, 0.0, 0.0],  # beyond the envelope
        ]
    )
    r = compute_progress_reward(wrench, cfg)
    torch.testing.assert_close(r, torch.tensor([0.0, 0.5, 0.5, 1.0]))


def test_progress_reward_rewards_twist_and_pull_over_pull_alone():
    cfg = HarvestRewardConfig()
    pull = torch.tensor([[12.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    twist_pull = torch.tensor([[12.0, 0.0, 0.0, 0.0, 0.0, 0.03]])
    assert compute_progress_reward(twist_pull, cfg) > compute_progress_reward(pull, cfg)


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


def test_collateral_penalty_counts_only_load_above_rest_baseline():
    woody = {
        "spur_stem": torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "stem_apple": torch.tensor([[0.0, 0.0, 9.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0, 0.0, 0.0]]),
        "primary_spur": torch.tensor([[0.0, 0.0, 6.0, 0.0, 0.0, 0.0], [0.0, 0.0, 6.0, 0.0, 0.0, 0.0]]),
    }
    woody["spur_stem"] = woody["spur_stem"].expand(2, 6)
    baseline = {"stem_apple": torch.tensor([4.0, 4.0]), "primary_spur": torch.tensor([6.0, 6.0])}
    total = compute_collateral_penalty(woody, target_junction_name="spur_stem", baseline_norm=baseline)
    # env0: stem_apple 9-4=5, primary 0 ; env1: unloading below rest is not rewarded -> 0
    torch.testing.assert_close(total, torch.tensor([5.0, 0.0]))


def test_collateral_penalty_raises_when_only_target_junction_present():
    woody = {"spur_stem": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])}
    try:
        compute_collateral_penalty(woody, target_junction_name="spur_stem")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_dense_reward_combines_weighted_terms():
    cfg = HarvestRewardConfig(w_progress=1.0, w_pullout=0.5, w_collateral=0.1, w_slack=0.01)
    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]])}
    info = {
        # anchor-frame target wrench, on the envelope -> progress=1.0
        "target_junction_wrench": torch.tensor([[0.0, 0.0, 20.0, 0.0, 0.0, 0.0]]),
        "ft_wrist": torch.tensor([[0.0, 0.0, 2.0, 0.0, 0.0, 0.0]]),  # pullout=2.0
        "woody_part_force": {
            "spur_stem": torch.zeros(1, 6),
            "stem_apple": torch.tensor([[3.0, 4.0, 0.0, 0.0, 0.0, 0.0]]),  # collateral=5.0
        },
    }
    reward = compute_dense_reward(obs, info, target_junction_name="spur_stem", cfg=cfg)
    expected = 1.0 * 1.0 - 0.5 * 2.0 - 0.1 * 5.0 - 0.01  # absolute-weighted progress; slack per step
    assert reward.shape == (1, 1)
    torch.testing.assert_close(reward, torch.tensor([[expected]]))


def test_dense_reward_uses_collateral_baseline_from_info():
    cfg = HarvestRewardConfig(w_progress=0.0, w_pullout=0.0, w_collateral=1.0, w_slack=0.0)
    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]])}
    info = {
        "target_junction_wrench": torch.zeros(1, 6),
        "ft_wrist": torch.zeros(1, 6),
        "woody_part_force": {
            "spur_stem": torch.zeros(1, 6),
            "stem_apple": torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]]),
        },
        "collateral_baseline_norm": {"stem_apple": torch.tensor([3.0])},
    }
    reward = compute_dense_reward(obs, info, target_junction_name="spur_stem", cfg=cfg)
    torch.testing.assert_close(reward, torch.tensor([[-2.0]]))


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])


def test_d7_progress_uses_per_env_thresholds_from_info():
    from apple_pick_gym.batched_envs.harvest_reward import compute_dense_reward_terms

    cfg = HarvestRewardConfig()
    w = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2)
    th = torch.tensor([[20.0, 0.05], [10.0, 0.05]])
    torch.testing.assert_close(compute_progress_reward(w, cfg, thresholds=th), torch.tensor([0.5, 1.0]))
    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2)}
    info = {
        "target_junction_wrench": w,
        "detach_thresholds": th,
        "ft_wrist": torch.zeros(2, 6),
        "woody_part_force": {"spur_stem": w, "other": torch.zeros(2, 6)},
    }
    terms = compute_dense_reward_terms(obs, info, target_junction_name="spur_stem", cfg=cfg)
    torch.testing.assert_close(terms["progress"], torch.tensor([0.5, 1.0]))


def test_d9_pullout_hinge_charges_only_force_above_the_grip_capacity():
    from apple_pick_gym.batched_envs.harvest_reward import compute_pullout_penalty

    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3)  # ee z = world z
    ft = torch.tensor([[0.0, 0.0, 0.1, 0, 0, 0], [0.0, 0.0, 8.0, 0, 0, 0], [0.0, 0.0, 14.0, 0, 0, 0]])
    torch.testing.assert_close(compute_pullout_penalty(ft, q), torch.tensor([0.1, 8.0, 14.0]))  # default: no hinge
    torch.testing.assert_close(compute_pullout_penalty(ft, q, threshold_n=10.0), torch.tensor([0.0, 0.0, 4.0]))


def test_d9_dense_terms_use_the_configured_pullout_hinge():
    from apple_pick_gym.batched_envs.harvest_reward import compute_dense_reward_terms

    cfg = HarvestRewardConfig(pullout_threshold_n=10.0)
    w = torch.zeros(1, 6)
    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]])}
    info = {
        "target_junction_wrench": w,
        "ft_wrist": torch.tensor([[0.0, 0.0, 12.0, 0.0, 0.0, 0.0]]),
        "woody_part_force": {"spur_stem": w, "other": torch.zeros(1, 6)},
    }
    terms = compute_dense_reward_terms(obs, info, target_junction_name="spur_stem", cfg=cfg)
    torch.testing.assert_close(terms["pullout"], torch.tensor([2.0]))
