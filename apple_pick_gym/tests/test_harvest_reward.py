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
        "detach_utilization": torch.zeros(2),
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
        "detach_utilization": torch.tensor([1.0]),
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
        "detach_utilization": torch.zeros(1),
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
        "detach_utilization": compute_progress_reward(w, cfg, thresholds=th),  # what the env publishes
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
        "detach_utilization": torch.zeros(1),
        "ft_wrist": torch.tensor([[0.0, 0.0, 12.0, 0.0, 0.0, 0.0]]),
        "woody_part_force": {"spur_stem": w, "other": torch.zeros(1, 6)},
    }
    terms = compute_dense_reward_terms(obs, info, target_junction_name="spur_stem", cfg=cfg)
    torch.testing.assert_close(terms["pullout"], torch.tensor([2.0]))


def test_d11_wrist_force_soft_cap_charges_total_wrist_force_above_it():
    # [D11] the bend-detach policy pushed laterally: wrist 23 N mean peak, 10% of envs over the 40 N
    # safety cliff, while pull-out (grip axis only) stayed ~free. Charge |F_wrist| above a soft cap.
    from apple_pick_gym.batched_envs.harvest_reward import compute_dense_reward_terms, weight_dense_reward_terms

    cfg = HarvestRewardConfig(wrist_force_soft_cap_n=25.0, w_wrist=0.5)
    w = torch.zeros(3, 6)
    ft = torch.zeros(3, 6)
    ft[0, 0], ft[1, 0], ft[2, 1] = 20.0, 30.0, 45.0  # lateral (x/y) pushes; ee z = world z
    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3)}
    info = {
        "target_junction_wrench": w,
        "detach_utilization": torch.zeros(3),
        "ft_wrist": ft,
        "woody_part_force": {"spur_stem": w, "other": torch.zeros(3, 6)},
    }
    raw = compute_dense_reward_terms(obs, info, target_junction_name="spur_stem", cfg=cfg)
    torch.testing.assert_close(raw["wrist"], torch.tensor([0.0, 5.0, 20.0]))
    torch.testing.assert_close(raw["pullout"], torch.zeros(3))  # lateral: not pull-out
    torch.testing.assert_close(weight_dense_reward_terms(raw, cfg)["wrist"], torch.tensor([0.0, -2.5, -10.0]))
    assert HarvestRewardConfig().w_wrist == 0.0  # library default: off


def test_d17_progress_term_is_the_env_published_utilization_clipped():
    # [D17] the env owns the detach criterion (envelope or bend-pull) and publishes
    # info["detach_utilization"]; the reward reads it rather than recomputing it from a wrench.
    from apple_pick_gym.batched_envs.harvest_reward import compute_dense_reward_terms

    cfg = HarvestRewardConfig()
    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 4)}
    info = {
        "target_junction_wrench": torch.zeros(4, 6),  # ignored by progress
        "detach_utilization": torch.tensor([-0.2, 0.0, 0.4, 1.7]),
        "ft_wrist": torch.zeros(4, 6),
        "woody_part_force": {"spur_stem": torch.zeros(4, 6), "other": torch.zeros(4, 6)},
    }
    terms = compute_dense_reward_terms(obs, info, target_junction_name="spur_stem", cfg=cfg)
    torch.testing.assert_close(terms["progress"], torch.tensor([0.0, 0.0, 0.4, 1.0]))


def test_d17_collateral_clip_bounds_the_weighted_term_but_not_the_raw_load():
    # [D17] an unbounded per-step collateral cost lets one bad episode dominate the value loss and
    # the value scaler; the weighted term is clipped, the raw load (peak-collateral metrics) is not.
    from apple_pick_gym.batched_envs.harvest_reward import weight_dense_reward_terms

    raw = {k: torch.zeros(3) for k in ("progress", "pullout", "wrist", "slack")}
    raw["collateral"] = torch.tensor([10.0, 60.0, 200.0])
    clipped = weight_dense_reward_terms(raw, HarvestRewardConfig(w_collateral=0.02, collateral_clip_n=50.0))
    torch.testing.assert_close(clipped["collateral"], torch.tensor([-0.2, -1.0, -1.0]))
    torch.testing.assert_close(raw["collateral"], torch.tensor([10.0, 60.0, 200.0]))
    unclipped = weight_dense_reward_terms(raw, HarvestRewardConfig(w_collateral=0.02))
    torch.testing.assert_close(unclipped["collateral"], torch.tensor([-0.2, -1.2, -4.0]))
    assert HarvestRewardConfig().collateral_clip_n is None  # library default: off


def test_d17m_deviation_collateral_charges_unloading_and_redirection():
    # [D17m] lifting the apple unloads the other junctions; the relu of the norm increase scored that
    # 0, but it is still robot effort on the tree. "deviation" charges |F - F_rest| as a vector.
    woody = {
        "spur_stem": torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4),  # target, excluded
        "primary_spur": torch.tensor(
            [
                [0.0, 0.0, -6.0, 0.0, 0.0, 0.0],  # at rest
                [0.0, 0.0, -2.0, 0.0, 0.0, 0.0],  # unloaded by 4 N
                [0.0, 6.0, 0.0, 0.0, 0.0, 0.0],  # same magnitude, turned 90 deg
                [0.0, 0.0, -9.0, 0.0, 0.0, 0.0],  # loaded by 3 N
            ]
        ),
    }
    rest = {"primary_spur": torch.tensor([[0.0, 0.0, -6.0]] * 4)}
    dev = compute_collateral_penalty(woody, target_junction_name="spur_stem", baseline_force=rest, mode="deviation")
    torch.testing.assert_close(dev, torch.tensor([0.0, 4.0, 6.0 * math.sqrt(2.0), 3.0]))
    rest_norm = {"primary_spur": torch.full((4,), 6.0)}
    relu = compute_collateral_penalty(woody, target_junction_name="spur_stem", baseline_norm=rest_norm)
    torch.testing.assert_close(relu, torch.tensor([0.0, 0.0, 0.0, 3.0]))  # the old blind spots


def test_d17m_deviation_collateral_needs_the_rest_vectors():
    woody = {"spur_stem": torch.zeros(1, 6), "primary_spur": torch.zeros(1, 6)}
    try:
        compute_collateral_penalty(woody, target_junction_name="spur_stem", mode="deviation")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_d17m_dense_terms_use_the_configured_collateral_mode():
    from apple_pick_gym.batched_envs.harvest_reward import compute_dense_reward_terms

    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]])}
    info = {
        "target_junction_wrench": torch.zeros(1, 6),
        "detach_utilization": torch.zeros(1),
        "ft_wrist": torch.zeros(1, 6),
        "woody_part_force": {"spur_stem": torch.zeros(1, 6), "primary_spur": torch.tensor([[0.0, 0.0, -2.0, 0, 0, 0]])},
        "collateral_baseline_norm": {"primary_spur": torch.tensor([6.0])},
        "collateral_baseline_force": {"primary_spur": torch.tensor([[0.0, 0.0, -6.0]])},
    }
    old = compute_dense_reward_terms(obs, info, target_junction_name="spur_stem", cfg=HarvestRewardConfig())
    new = compute_dense_reward_terms(
        obs, info, target_junction_name="spur_stem", cfg=HarvestRewardConfig(collateral_mode="deviation")
    )
    torch.testing.assert_close(old["collateral"], torch.tensor([0.0]))
    torch.testing.assert_close(new["collateral"], torch.tensor([4.0]))
    assert HarvestRewardConfig().collateral_mode == "relu_norm"  # library default: v1/v2b behaviour
    try:
        HarvestRewardConfig(collateral_mode="abs")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_d17n_collateral_can_exclude_named_junctions():
    # [D17n] stem_apple is the apple end of the same stem: it carries the stem tension the pick needs,
    # so charging it as collateral double-counts the pull. collateral_exclude drops it from the sum.
    from apple_pick_gym.batched_envs.harvest_reward import compute_dense_reward_terms

    woody = {
        "spur_stem": torch.zeros(1, 6),
        "primary_spur": torch.tensor([[0.0, 0.0, -9.0, 0, 0, 0]]),
        "stem_apple": torch.tensor([[0.0, 0.0, -20.0, 0, 0, 0]]),
    }
    rest = {"primary_spur": torch.tensor([[0.0, 0.0, -6.0]]), "stem_apple": torch.tensor([[0.0, 0.0, -3.0]])}
    both = compute_collateral_penalty(woody, target_junction_name="spur_stem", baseline_force=rest, mode="deviation")
    torch.testing.assert_close(both, torch.tensor([20.0]))
    spur_only = compute_collateral_penalty(
        woody, target_junction_name="spur_stem", baseline_force=rest, mode="deviation", exclude=("stem_apple",)
    )
    torch.testing.assert_close(spur_only, torch.tensor([3.0]))
    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]])}
    info = {
        "target_junction_wrench": torch.zeros(1, 6),
        "detach_utilization": torch.zeros(1),
        "ft_wrist": torch.zeros(1, 6),
        "woody_part_force": woody,
        "collateral_baseline_force": rest,
    }
    cfg = HarvestRewardConfig(collateral_mode="deviation", collateral_exclude=("stem_apple",))
    terms = compute_dense_reward_terms(obs, info, target_junction_name="spur_stem", cfg=cfg)
    torch.testing.assert_close(terms["collateral"], torch.tensor([3.0]))
    assert HarvestRewardConfig().collateral_exclude == ()


def test_d17q_signed_norm_collateral_is_zero_at_rest_and_pays_for_unloading():
    # [D17q] maintainer: lifting should pay, and the term should be centred on 0 at rest.
    # "signed_norm" charges sum_j (|F_j| - |F_j|_rest): loading costs, unloading earns the same amount,
    # turning the load at constant magnitude is free.
    woody = {
        "spur_stem": torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4),  # target, excluded
        "primary_spur": torch.tensor(
            [
                [0.0, 0.0, -6.0, 0.0, 0.0, 0.0],  # at rest
                [0.0, 0.0, -2.0, 0.0, 0.0, 0.0],  # unloaded by 4 N
                [0.0, 6.0, 0.0, 0.0, 0.0, 0.0],  # same magnitude, turned 90 deg
                [0.0, 0.0, -9.0, 0.0, 0.0, 0.0],  # loaded by 3 N
            ]
        ),
        "stem_apple": torch.tensor([[0.0, 0.0, -50.0, 0.0, 0.0, 0.0]] * 4),  # excluded by name
    }
    rest_norm = {"primary_spur": torch.full((4,), 6.0), "stem_apple": torch.full((4,), 3.0)}
    signed = compute_collateral_penalty(
        woody, target_junction_name="spur_stem", baseline_norm=rest_norm, mode="signed_norm", exclude=("stem_apple",)
    )
    torch.testing.assert_close(signed, torch.tensor([0.0, -4.0, 0.0, 3.0]))
    # at or above rest everywhere it is the relu_norm term
    relu = compute_collateral_penalty(woody, target_junction_name="spur_stem", baseline_norm=rest_norm, exclude=("stem_apple",))
    torch.testing.assert_close(signed[[0, 2, 3]], relu[[0, 2, 3]])


def test_d17q_signed_norm_sums_junctions_and_needs_the_rest_norms():
    woody = {
        "spur_stem": torch.zeros(1, 6),
        "primary_spur": torch.tensor([[0.0, 0.0, -9.0, 0, 0, 0]]),  # +3
        "primary_support_left": torch.tensor([[0.0, 0.0, -1.0, 0, 0, 0]]),  # -4
    }
    rest = {"primary_spur": torch.tensor([6.0]), "primary_support_left": torch.tensor([5.0])}
    out = compute_collateral_penalty(woody, target_junction_name="spur_stem", baseline_norm=rest, mode="signed_norm")
    torch.testing.assert_close(out, torch.tensor([-1.0]))  # signed: unloading one offsets loading another
    try:
        compute_collateral_penalty(woody, target_junction_name="spur_stem", mode="signed_norm")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_d17q_signed_norm_dense_reward_credits_unloading_and_clips_only_the_cost():
    from apple_pick_gym.batched_envs.harvest_reward import compute_dense_reward_terms, weight_dense_reward_terms

    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2)}
    info = {
        "target_junction_wrench": torch.zeros(2, 6),
        "detach_utilization": torch.zeros(2),
        "ft_wrist": torch.zeros(2, 6),
        "woody_part_force": {
            "spur_stem": torch.zeros(2, 6),
            "primary_spur": torch.tensor([[0.0, 0.0, -2.0, 0, 0, 0], [0.0, 0.0, -106.0, 0, 0, 0]]),
        },
        "collateral_baseline_norm": {"primary_spur": torch.tensor([6.0, 6.0])},
        "collateral_baseline_force": {"primary_spur": torch.tensor([[0.0, 0.0, -6.0]] * 2)},
    }
    cfg = HarvestRewardConfig(collateral_mode="signed_norm", w_collateral=0.5, collateral_clip_n=50.0)
    terms = compute_dense_reward_terms(obs, info, target_junction_name="spur_stem", cfg=cfg)
    torch.testing.assert_close(terms["collateral"], torch.tensor([-4.0, 100.0]))
    w = weight_dense_reward_terms(terms, cfg)
    torch.testing.assert_close(w["collateral"], torch.tensor([2.0, -25.0]))  # credit +0.5*4; cost capped at 50 N


def test_d17q_collateral_torque_term_adds_weighted_signed_moment_change():
    # [D17q] maintainer: collateral = force + weighted torque at primary_spur, both against rest. The torque
    # (N*m, at the joint anchor) is converted to newtons by collateral_torque_n_per_nm.
    woody = {
        "spur_stem": torch.zeros(3, 6),
        "primary_spur": torch.tensor([[0.0, 0.0, -6.0, 0, 0, 0], [0.0, 0.0, -8.0, 0, 0, 0], [0.0, 0.0, -6.0, 0, 0, 0]]),
        "primary_support_left": torch.tensor([[0.0, 0.0, -50.0, 0, 0, 0]] * 3),  # excluded
    }
    torque = {
        "primary_spur": torch.tensor([[0.0, 0.5, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 0.2]]),  # rest, +1 N*m, -0.3 N*m
        "primary_support_left": torch.tensor([[9.0, 0.0, 0.0]] * 3),
    }
    kw = dict(
        target_junction_name="spur_stem",
        baseline_norm={"primary_spur": torch.full((3,), 6.0), "primary_support_left": torch.full((3,), 5.0)},
        mode="signed_norm",
        exclude=("primary_support_left",),
        torque=torque,
        baseline_torque_norm={"primary_spur": torch.full((3,), 0.5), "primary_support_left": torch.full((3,), 1.0)},
    )
    no_t = compute_collateral_penalty(woody, **kw)  # torque given but weight 0: force only
    torch.testing.assert_close(no_t, torch.tensor([0.0, 2.0, 0.0]))
    out = compute_collateral_penalty(woody, torque_weight=10.0, **kw)
    torch.testing.assert_close(out, torch.tensor([0.0, 2.0 + 10.0, -3.0]))


def test_d17q_collateral_torque_needs_torques_and_is_refused_in_deviation_mode():
    woody = {"spur_stem": torch.zeros(1, 6), "primary_spur": torch.zeros(1, 6)}
    base = {"primary_spur": torch.zeros(1)}
    for kw in (
        dict(mode="signed_norm", baseline_norm=base, torque_weight=1.0),  # no torques
        dict(mode="signed_norm", baseline_norm=base, torque_weight=1.0, torque={"primary_spur": torch.zeros(1, 3)}),  # no rest
        dict(mode="deviation", baseline_force={"primary_spur": torch.zeros(1, 3)}, torque_weight=1.0,
             torque={"primary_spur": torch.zeros(1, 3)}, baseline_torque_norm=base),
    ):
        try:
            compute_collateral_penalty(woody, target_junction_name="spur_stem", **kw)
            assert False, f"expected ValueError for {kw}"
        except ValueError:
            pass


def test_d17q_dense_terms_read_the_anchor_torques_when_weighted():
    from apple_pick_gym.batched_envs.harvest_reward import compute_dense_reward_terms

    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]])}
    info = {
        "target_junction_wrench": torch.zeros(1, 6),
        "detach_utilization": torch.zeros(1),
        "ft_wrist": torch.zeros(1, 6),
        "woody_part_force": {"spur_stem": torch.zeros(1, 6), "primary_spur": torch.tensor([[0.0, 0.0, -6.0, 9, 9, 9]])},
        "woody_part_torque_anchor": {"primary_spur": torch.tensor([[0.0, 2.0, 0.0]])},
        "collateral_baseline_norm": {"primary_spur": torch.tensor([6.0])},
        "collateral_baseline_torque_norm": {"primary_spur": torch.tensor([0.5])},
    }
    cfg = HarvestRewardConfig(collateral_mode="signed_norm", collateral_torque_n_per_nm=10.0)
    terms = compute_dense_reward_terms(obs, info, target_junction_name="spur_stem", cfg=cfg)
    torch.testing.assert_close(terms["collateral"], torch.tensor([15.0]))  # the COM torque (9, 9, 9) is not used
    assert HarvestRewardConfig().collateral_torque_n_per_nm == 0.0
