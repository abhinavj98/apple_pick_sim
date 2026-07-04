"""Device upload of per-env EE twist actions for batched teleop controllers."""

from __future__ import annotations

import warp as wp


def upload_batched_twists_from_actions(
    lin_vels_wp: wp.array,
    ang_vels_wp: wp.array,
    actions,
    *,
    lock_angular: bool = False,
) -> None:
    """Copy ``(N, 6)`` float32 action rows into ``wp.vec3`` twist buffers without host round-trip."""
    import torch

    torch_device = wp.device_to_torch(lin_vels_wp.device)
    lin = actions[:, :3].to(device=torch_device, dtype=torch.float32).contiguous()
    if lock_angular:
        ang = torch.zeros(
            (int(actions.shape[0]), 3),
            device=torch_device,
            dtype=torch.float32,
        )
    else:
        ang = actions[:, 3:6].to(device=torch_device, dtype=torch.float32).contiguous()
    wp.copy(lin_vels_wp, wp.from_torch(lin, dtype=wp.vec3))
    wp.copy(ang_vels_wp, wp.from_torch(ang, dtype=wp.vec3))


def clip_action_tensor(actions, *, linear_speed: float, angular_speed: float):
    """Vectorized speed clamp for ``(N, 6)`` action tensor."""
    import torch

    clipped = actions.clone()
    lin = clipped[:, :3]
    lin_norm = torch.linalg.norm(lin, dim=1, keepdim=True)
    lin_scale = torch.where(
        (lin_norm > linear_speed) & (lin_norm > 0),
        linear_speed / lin_norm,
        torch.ones_like(lin_norm),
    )
    clipped[:, :3] = lin * lin_scale
    ang = clipped[:, 3:6]
    ang_norm = torch.linalg.norm(ang, dim=1, keepdim=True)
    ang_scale = torch.where(
        (ang_norm > angular_speed) & (ang_norm > 0),
        angular_speed / ang_norm,
        torch.ones_like(ang_norm),
    )
    clipped[:, 3:6] = ang * ang_scale
    return clipped
