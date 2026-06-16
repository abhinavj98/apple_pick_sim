# Gym observation contract (v1)

## Schema version

All `apple_pick_gym` envs include `info["obs_schema"] == "v1"` on `reset()` and `step()`.

`obs_schema` is a **metadata tag in `info`**, not an observation key. Dataset loaders and training code should read it to select parsers for observation dict keys and semantics. Bump the string (e.g. `"v2"`) when breaking observation layout or naming.

## Migration (pre-v1 replay logs)

`ApplePickReplayEnv` observation keys were renamed for parity with other envs:

| Pre-v1 | v1 |
|--------|-----|
| `woody_start` | `woody_part_start_pos` |
| `woody_end` | `woody_part_end_pos` |

Logs without `info["obs_schema"]` should be treated as pre-v1 replay layout.

## Shared keys

| Key | Shape | Units | Description |
|-----|-------|-------|-------------|
| `woody_part_start_pos` | `(N*3,)` | m | Parent-side fixed-joint anchor positions |
| `woody_part_end_pos` | `(N*3,)` | m | Child-side fixed-joint anchor positions |
| `tcp_velocity` | `(6,)` | m/s, rad/s | TCP spatial velocity `[v(3), omega(3)]` |

`N` is the number of woody fixed joints in the current scene (`info["n_woody_parts"]`).

## Env-specific keys

### `ApplePickCoupled-v0`

| Key | Shape | Source |
|-----|-------|--------|
| `woody_part_force` | `(N*6,)` | `measure_fruiting_forces` after substep |
| `apple_pos` | `(3,)` | Apple body COM |
| `tcp_force` | `(6,)` | Fresh TCP harvest (`proxy_forces`) |
| `tcp_velocity` | `(6,)` | `body_qd` at TCP |

### `ApplePickVic-v0` / `ApplePickSysId-v0`

Coupled keys plus:

| Key | Shape | Source |
|-----|-------|--------|
| `ft_wrist` | `(6,)` | Lagged plant wrench from `coupling_forces_cache` (F/T sensor proxy) |
| `tcp_pos` | `(3,)` | Actual TCP body position (SysId only) |
| `excitation_type` | scalar int | SysId excitation phase metadata (SysId only) |
| `excitation_f_inst` | scalar float | Instantaneous excitation frequency (SysId only) |
| `excitation_direction` | `(3,)` | Unit push direction (SysId only) |

### `ApplePickReplayEnv`

| Key | Shape | Source |
|-----|-------|--------|
| `ft_wrist` | `(6,)` | Fresh TCP harvest via `_end_effector_wrench()` |

**Note:** `ft_wrist` semantics differ between Vic/SysId (lagged cache) and Replay (live harvest). Consumers must check env class or document source when mixing datasets.

## Tests

- `apple_pick_gym/tests/test_apple_pick_coupled_env.py` — Coupled parity and Replay obs contract
- `apple_pick_gym/tests/test_sysid_env.py` — SysId excitation obs keys

Run:

```bash
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/ -q -m "not slow"
```
