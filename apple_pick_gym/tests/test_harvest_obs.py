"""Observation flattening: sensor-realistic actor vector + privileged critic vector.

The nested v3 obs dict (woody_part_start_pos/end_pos: dict[str, (N,3)]) must
flatten to a fixed-order (N, D) tensor, deterministically by sorted junction
name, so the layout is stable across runs and topologies with the same
junction set.
"""

from __future__ import annotations

import torch

from apple_pick_gym.batched_envs.harvest_obs import (
    actor_obs_layout,
    critic_obs_layout,
    flatten_actor_obs,
    flatten_critic_obs,
)

_JUNCTIONS = ["stem_apple", "primary_spur", "spur_stem", "support"]


def _make_obs(n: int, junctions: list[str]) -> dict:
    return {
        "tcp_pos": torch.randn(n, 3),
        "tcp_quat": torch.randn(n, 4),
        "tcp_velocity": torch.randn(n, 6),
        "ft_wrist": torch.randn(n, 6),
        "apple_pos": torch.randn(n, 3),
        "apple_quat": torch.randn(n, 4),
        "robot_joint_q": torch.randn(n, 7),
        "woody_part_start_pos": {j: torch.randn(n, 3) for j in junctions},
        "woody_part_end_pos": {j: torch.randn(n, 3) for j in junctions},
        "last_action": torch.randn(n, 13),
        "step_frac": torch.rand(n, 1),
    }


def _make_privileged(n: int) -> dict:
    return {
        "spur_flexural_modulus_pa": torch.rand(n, 1) * 1e8,
        "stem_flexural_modulus_pa": torch.rand(n, 1) * 1e8,
        "spur_youngs_modulus_pa": torch.rand(n, 1) * 1e8,
        "stem_youngs_modulus_pa": torch.rand(n, 1) * 1e8,
        "spur_damping_ratio": torch.rand(n, 1),
        "stem_damping_ratio": torch.rand(n, 1),
        "primary_density": torch.rand(n, 1) * 900,
        "support_kp": torch.rand(n, 1) * 1e6,
        "support_roll_kp": torch.rand(n, 1) * 7.5,
        "support_zeta": torch.rand(n, 1),
        "arm_armature": torch.rand(n, 7),
        "arm_friction": torch.rand(n, 7),
        "arm_joint_damping": torch.rand(n, 7),
        "arm_link_mass_scale": torch.rand(n, 1),
        "arm_link_inertia_scale": torch.rand(n, 1),
        "arm_ee_mass_scale": torch.rand(n, 1),
        "arm_ee_inertia_scale": torch.rand(n, 1),
    }


def _make_woody_force(n: int, junctions: list[str]) -> dict:
    return {j: torch.randn(n, 6) for j in junctions}


def test_actor_layout_has_documented_fixed_width():
    layout = actor_obs_layout(_JUNCTIONS)
    # 3+4+6+6+3+4+7 = 33 fixed fields, + 4 junctions * (3+3) + 13 (last_action) + 1 (step_frac)
    expected = 33 + len(_JUNCTIONS) * 6 + 13 + 1
    assert layout.total_width == expected


def test_actor_layout_order_is_junction_name_sorted_not_dict_insertion_order():
    unsorted_junctions = ["support", "stem_apple", "primary_spur", "spur_stem"]
    layout_a = actor_obs_layout(unsorted_junctions)
    layout_b = actor_obs_layout(list(reversed(unsorted_junctions)))
    assert layout_a.entries == layout_b.entries


def test_flatten_actor_obs_matches_layout_width():
    obs = _make_obs(5, _JUNCTIONS)
    out = flatten_actor_obs(obs)
    layout = actor_obs_layout(_JUNCTIONS)
    assert out.shape == (5, layout.total_width)


def test_flatten_actor_obs_is_deterministic_for_same_input():
    obs = _make_obs(3, _JUNCTIONS)
    a = flatten_actor_obs(obs)
    b = flatten_actor_obs(obs)
    torch.testing.assert_close(a, b)


def test_flatten_actor_obs_places_known_fields_at_documented_slices():
    obs = _make_obs(2, _JUNCTIONS)
    out = flatten_actor_obs(obs)
    layout = actor_obs_layout(_JUNCTIONS)
    torch.testing.assert_close(out[:, layout.slice_for("tcp_pos")], obs["tcp_pos"])
    torch.testing.assert_close(out[:, layout.slice_for("ft_wrist")], obs["ft_wrist"])
    torch.testing.assert_close(out[:, layout.slice_for("last_action")], obs["last_action"])
    torch.testing.assert_close(out[:, layout.slice_for("step_frac")], obs["step_frac"])
    for j in _JUNCTIONS:
        torch.testing.assert_close(
            out[:, layout.slice_for(f"woody_part_start_pos/{j}")],
            obs["woody_part_start_pos"][j],
        )


def test_critic_layout_is_actor_layout_plus_privileged_fields():
    actor_layout = actor_obs_layout(_JUNCTIONS)
    critic_layout = critic_obs_layout(_JUNCTIONS)
    assert critic_layout.total_width > actor_layout.total_width
    # Strict prefix: every actor entry appears at the same (start, width) in critic.
    actor_by_name = {e.name: (e.start, e.width) for e in actor_layout.entries}
    critic_by_name = {e.name: (e.start, e.width) for e in critic_layout.entries}
    for name, slice_ in actor_by_name.items():
        assert critic_by_name[name] == slice_


def test_flatten_critic_obs_has_actor_obs_as_exact_prefix():
    obs = _make_obs(4, _JUNCTIONS)
    privileged = _make_privileged(4)
    woody_force = _make_woody_force(4, _JUNCTIONS)
    actor_out = flatten_actor_obs(obs)
    critic_out = flatten_critic_obs(obs, privileged, woody_force)
    assert critic_out.shape[0] == 4
    assert critic_out.shape[1] > actor_out.shape[1]
    torch.testing.assert_close(critic_out[:, : actor_out.shape[1]], actor_out)


def test_no_privileged_field_name_appears_in_actor_layout():
    """Explicit by-name check: none of the privileged/force field names leak
    into the actor's layout entries."""
    actor_layout = actor_obs_layout(_JUNCTIONS)
    actor_names = {e.name for e in actor_layout.entries}
    privileged = _make_privileged(1)
    woody_force = _make_woody_force(1, _JUNCTIONS)
    for name in privileged:
        assert name not in actor_names, f"privileged field {name!r} leaked into actor layout"
    for j in _JUNCTIONS:
        assert f"woody_part_force/{j}" not in actor_names


def test_mutating_privileged_inputs_does_not_change_actor_output():
    """flatten_actor_obs never receives privileged data, so it cannot leak by
    construction; this test pins that contract."""
    obs = _make_obs(3, _JUNCTIONS)
    before = flatten_actor_obs(obs).clone()
    privileged = _make_privileged(3)
    woody_force = _make_woody_force(3, _JUNCTIONS)
    privileged["support_kp"] *= 1000.0
    woody_force["support"] *= 1000.0
    _ = flatten_critic_obs(obs, privileged, woody_force)
    after = flatten_actor_obs(obs)
    torch.testing.assert_close(before, after)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
