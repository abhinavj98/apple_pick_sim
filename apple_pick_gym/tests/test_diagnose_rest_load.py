"""Rest-load statics used by ``rl/diagnose_rest_load``: gravity wrench of the hanging subtree."""

from __future__ import annotations

import numpy as np
import torch

from apple_pick_gym.rl.diagnose_rest_load import descendants, gravity_wrench_about


def test_descendants_follows_joints_from_parent_to_child():
    parent = np.array([-1, 0, 1, 2, 1, 5])
    child = np.array([0, 1, 2, 3, 4, 6])
    assert sorted(descendants(1, parent, child)) == [1, 2, 3, 4]
    assert sorted(descendants(5, parent, child)) == [5, 6]


def test_gravity_wrench_of_a_point_mass_on_a_horizontal_lever():
    # 0.2 kg, 50 mm to the side of the junction: F = (0, 0, -1.962), tau = r x F = 0.0981 N*m about -y
    com = torch.tensor([[[0.05, 0.0, 0.0]]])
    mass = torch.tensor([[0.2]])
    f, t = gravity_wrench_about(com, mass, torch.zeros(1, 3), g=(0.0, 0.0, -9.81))
    torch.testing.assert_close(f, torch.tensor([[0.0, 0.0, -1.962]]))
    torch.testing.assert_close(t, torch.tensor([[0.0, 0.0981, 0.0]]))  # (0.05,0,0) x (0,0,-1.962) = (0, 0.0981, 0)


def test_gravity_moment_vanishes_when_hanging_straight_below():
    com = torch.tensor([[[0.0, 0.0, -0.05], [0.0, 0.0, -0.02]]])
    mass = torch.tensor([[0.2, 0.001]])
    _, t = gravity_wrench_about(com, mass, torch.zeros(1, 3), g=(0.0, 0.0, -9.81))
    torch.testing.assert_close(t, torch.zeros(1, 3))
