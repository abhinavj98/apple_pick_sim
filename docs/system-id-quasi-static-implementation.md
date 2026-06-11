# Quasi-static stepped stiffness mapping (M3.0 §2.1)

## Behavior summary

§2.1 drives the EE through Fibonacci-hemisphere push directions. For each direction the
trajectory executes **move_out → hold → return**:

1. **move_out** — constant linear velocity along the direction for `n_steps × (step_size / move_speed)` seconds (default 5 × 2 cm at 5 cm/s → 10 cm total).
2. **hold** — zero velocity for `hold_duration_s` (default 1.5 s) so transients decay; steady-state `ft_wrist` is logged.
3. **return** — same speed back to the grasp center.

`ApplePickSysIdEnv` extends VIC with `Box(6)` EE velocity actions, higher default linear stiffness (`vic_linear_k=3000 N/m`), excitation metadata obs, actual `tcp_pos` from `body_q` (not the VIC target), and a wrench safety guard on `‖ft_wrist[:3]‖`.

## Code map

| Module | Role |
|--------|------|
| `apple_pick_sim/system_id/fibonacci_hemisphere.py` | Golden-ratio Fibonacci lattice; forward hemisphere filter |
| `apple_pick_sim/system_id/quasi_static_trajectory.py` | Phase machine + `iter_frames()` generator |
| `apple_pick_sim/system_id/excitation_state.py` | `ExcitationContext` dataclass |
| `apple_pick_gym/envs/apple_pick_sysid_env.py` | Gym env (`ApplePickSysId-v0`) |
| `apple_pick_sim/system_id/run_quasi_static.py` | Headless smoke runner |

## Fibonacci hemisphere

Points on the unit sphere use the standard golden-angle lattice:

\[
\phi_i = \arccos\left(1 - \frac{2(i + \tfrac{1}{2})}{N}\right),\quad
\theta_i = \frac{2\pi(i + \tfrac{1}{2})}{\varphi}
\]

where \(\varphi = (1+\sqrt{5})/2\). Directions with \(\mathbf{d}_i \cdot \hat{s} \ge 0\) for stem direction \(\hat{s}\) are kept; if fewer than `n` survive, indices wrap with reflection duplicates.

## Wrench guard

`compute_terminated` returns `True` when `np.linalg.norm(obs["ft_wrist"][:3]) > max_tcp_force_n` (default 30 N). This is a safety stop, not a task success signal.

## Tests

- `apple_pick_sim/tests/test_quasi_static_sysid.py` — lattice geometry, trajectory phases, hold frame count, net displacement
- `apple_pick_gym/tests/test_sysid_env.py` — action/obs contract, `tcp_pos` source, wrench guard, excitation context round-trip

## How to verify

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_quasi_static_sysid.py \
  apple_pick_gym/tests/test_sysid_env.py -q
```

Smoke (headless, one direction):

```bash
uv run python apple_pick_sim/system_id/run_quasi_static.py --viewer null --n-directions 1
```

Full 10-direction run uses ~3364 env steps (auto `max_episode_steps`). Default wrench guard is 1000 N because post-grasp VIC transients on step 0 can exceed 500 N.
