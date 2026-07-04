"""Tests for batched action twist upload and vectorized clipping."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

torch = pytest.importorskip("torch")

from apple_pick_sim.robot.fr3_robot.controllers.batched_action_twists import (  # noqa: E402
    clip_action_tensor,
    upload_batched_twists_from_actions,
)


def test_clip_action_tensor_scales_per_env_rows():
    actions = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    clipped = clip_action_tensor(actions, linear_speed=1.0, angular_speed=0.5)
    assert clipped[0, 0] == pytest.approx(1.0)
    assert clipped[1, 5] == pytest.approx(0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_upload_batched_twists_from_actions_stays_on_device():
    import warp as wp

    wp.init()
    actions = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.5]],
        dtype=torch.float32,
        device="cuda:0",
    )
    lin = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
    ang = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
    upload_batched_twists_from_actions(lin, ang, actions)
    lin_host = lin.numpy()
    ang_host = ang.numpy()
    assert lin_host[0, 0] == pytest.approx(1.0)
    assert ang_host[0, 2] == pytest.approx(0.5)
