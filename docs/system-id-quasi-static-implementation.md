# Quasi-static stepped stiffness mapping (M3.0 §2.1)

## Behavior summary

§2.1 drives the EE through Fibonacci-hemisphere push directions. For each direction the
trajectory repeats **fast move → hold** for each increment, then **return**:

1. **move_out** — fast linear burst along the direction for `movement_per_step_m / move_speed_mps` seconds (default 5 cm at 0.2 m/s ≈ 0.25 s).
2. **hold** — zero velocity for `hold_duration_s` (default 1.5 s) so transients decay; steady-state `ft_wrist` is logged at each amplitude.
3. Repeat steps 1–2 for `total_movement_m / movement_per_step_m` increments (default 2 × 5 cm → 10 cm total; must be an integer multiple — see `derive_n_steps()`).
4. **return** — one fast reverse over `total_movement_m` back to the grasp center.

Quasi-static behavior comes from **hold settling**, not slow crawl speed.

Default `QuasiStaticStepConfig`: `movement_per_step_m=0.05`, `total_movement_m=0.10`, `move_speed_mps=0.2`, `hold_duration_s=1.5`, `control_hz=60`.

`ApplePickSysIdEnv` extends VIC with `Box(6)` EE velocity actions, excitation metadata obs, actual `tcp_pos` from `body_q` (not the VIC target), and optional robot-facing weld placement. Default VIC stiffness is `vic_linear_k=2000` N/m (not the replay-env default). Stem force/torque caps are off by default (`stem_force_cap_n=None`).

**Robot-facing weld:** when `fix_to_apple=True` and `robot_facing_weld=True` (default), each `reset()` picks a Fibonacci-hemisphere weld direction toward the fixture robot base; successive resets cycle through `n_weld_hemisphere_samples` (default 10). Override via `reset(options={"weld_direction": (x, y, z)})`. `info["weld_direction"]` reports the unit vector used.

**Episode length:** `ApplePickSysIdEnv` defaults to `max_episode_steps=240`. A full multi-direction run needs `estimate_trajectory_frames(config, n_directions) + margin`. `gym.make(..., max_episode_steps=N)` only sets the `TimeLimit` wrapper — the env still truncates at its constructor default (240) unless you pass `max_episode_steps` into `ApplePickSysIdEnv(...)` directly.

**Wrench guard:** not implemented in the env yet. `compute_terminated` inherits the coupled-env stub (`False` always). Monitor `obs["ft_wrist"]` in the caller for now; see `docs/system_identification.md` §2 safety note.

## Code map

| Module | Role |
|--------|------|
| `apple_pick_sim/system_id/fibonacci_hemisphere.py` | Golden-ratio Fibonacci lattice; forward hemisphere filter |
| `apple_pick_sim/system_id/quasi_static_trajectory.py` | Phase machine, `derive_n_steps`, `estimate_trajectory_frames`, `iter_frames()` |
| `apple_pick_sim/system_id/excitation_state.py` | `ExcitationContext` dataclass |
| `apple_pick_gym/envs/apple_pick_sysid_env.py` | Gym env (`ApplePickSysId-v0`) |
| `apple_pick_gym/examples/example_gym_sysid.py` | Interactive demo: viewer, per-step logging, mean hold forces |
| `apple_pick_sim/system_id/run_quasi_static.py` | Headless smoke runner (no viewer) |

## Fibonacci hemisphere

Points on the unit sphere use the standard golden-angle lattice:

\[
\phi_i = \arccos\left(1 - \frac{2(i + \tfrac{1}{2})}{N}\right),\quad
\theta_i = \frac{2\pi(i + \tfrac{1}{2})}{\varphi}
\]

where \(\varphi = (1+\sqrt{5})/2\). Directions with \(\mathbf{d}_i \cdot \hat{s} \ge 0\) for stem direction \(\hat{s}\) are kept; if fewer than `n` survive, indices wrap with reflection duplicates.

Excitation push directions (TCP → apple) are sampled separately from weld directions (apple → robot base).

## Tests

- `apple_pick_sim/tests/test_quasi_static_sysid.py` — lattice geometry, per-increment phase sequence, hold counts, amplitude at holds, net displacement, `derive_n_steps` integer-multiple rule
- `apple_pick_gym/tests/test_sysid_env.py` — action/obs contract, `tcp_pos` source, weld direction cycling/override, excitation context round-trip, VIC defaults, no force termination

## How to verify

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_quasi_static_sysid.py \
  apple_pick_gym/tests/test_sysid_env.py -q
```

**Canonical demo** (one direction, 2 cm increments, 10 cm total, 0.2 m/s bursts; headless Linux auto-selects `--viewer null`):

```bash
uv run python apple_pick_gym/examples/example_gym_sysid.py \
  --n-directions 1 --movement-per-step-m 0.02 --total-movement-m 0.10 \
  --move-speed-mps 0.2
```

With Newton GL viewer (requires a display):

```bash
uv run python apple_pick_gym/examples/example_gym_sysid.py --viewer gl \
  --n-directions 1 --movement-per-step-m 0.02 --total-movement-m 0.10 \
  --move-speed-mps 0.2
```

Headless smoke (trajectory utilities only, no gym viewer):

```bash
uv run python apple_pick_sim/system_id/run_quasi_static.py --viewer null --n-directions 1
```

Default-config example (5 cm/step, 10 cm total):

```bash
uv run python apple_pick_gym/examples/example_gym_sysid.py --viewer null \
  --n-directions 1 --movement-per-step-m 0.05 --total-movement-m 0.10 \
  --move-speed-mps 0.2 --hold-duration-s 1.5
```

Full 10-direction run needs `max_episode_steps` on the env constructor ≥ `estimate_trajectory_frames(QuasiStaticStepConfig(), 10) + 64` (default 240 truncates early).
