"""Default coupled-scene placement constants shared by gym envs and tests."""

from __future__ import annotations

# Coupled cable scenes: nominal proxy within FR3 reach from explicit robot base below.
COUPLED_BASE_POS = (0.2, 0.2, 0.5)
# Matches ``placement_xform_for_proxy`` for a proxy near ``COUPLED_BASE_POS`` (z ≈ 0.5).
COUPLED_ROBOT_BASE_POS = (0.2, 0.2, -0.35)
