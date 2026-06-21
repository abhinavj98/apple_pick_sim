# Gym observation contract (v3)

## Schema version

All `apple_pick_gym` envs include `info["obs_schema"] == "v3"` on `reset()` and `step()`.

`obs_schema` is a **metadata tag in `info`**, not an observation key. Dataset loaders and training code should read it to select parsers for observation dict keys and semantics. Bump the string (e.g. `"v4"`) when breaking observation layout or naming.

## Migration

### v2 → v3 (observation-only replay pose bundle)

SysID observations and observation-only replay logs now include a replay pose bundle:

| v2 | v3 |
|----|-----|
| TCP position/velocity only | Add `tcp_quat` for TCP orientation |
| Apple position only | Add `apple_quat` for apple orientation |
| Robot state restored from simulator snapshot or defaults | Add `robot_joint_q` for robot joint-position initialization |

These keys are observation-only replay inputs. They let `ApplePickReplayEnv` initialize pose from recorded observations without privileged simulator snapshots; they do not replace calibration metadata such as robot base, fruiting base, or weld fixture transforms.

### v1 → v2 (woody positions)

`woody_part_start_pos` and `woody_part_end_pos` changed from flat `(N*3,)` arrays to **dicts keyed by junction name**:

| v1 | v2 |
|----|-----|
| `obs["woody_part_start_pos"][i*3:(i+1)*3]` | `obs["woody_part_start_pos"][junction_names[i]]` |
| `obs["woody_part_end_pos"][i*3:(i+1)*3]` | `obs["woody_part_end_pos"][junction_names[i]]` |

Junction names match `env.junction_names` / `info["fruiting_link_forces"]` keys (e.g. `"stem_apple"`). `woody_part_force` remains a flat `(N*6,)` array in v2/v3.

### Pre-v1 replay logs

`ApplePickReplayEnv` still exposes `woody_start` / `woody_end` as flat `(N*3,)` arrays (not the coupled `woody_part_*` keys).

Logs without `info["obs_schema"]` should be treated as pre-v1 replay layout.

## Shared keys

| Key | Shape | Units | Description |
|-----|-------|-------|-------------|
| `woody_part_start_pos` | `dict[str, (3,)]` | m | Parent-side fixed-joint anchor positions keyed by junction name |
| `woody_part_end_pos` | `dict[str, (3,)]` | m | Child-side fixed-joint anchor positions, same keys |
| `tcp_velocity` | `(6,)` | m/s, rad/s | TCP spatial velocity `[v(3), omega(3)]` |

`N` is the number of woody fixed joints in the current scene (`info["n_woody_parts"]`). Dict keys match `env.junction_names`.

## Observation-only replay subset

M3.0.3 uses a reset-time subset of the observation contract to initialize replay without privileged simulator arrays. Collection stores these reset values in episode metadata separately from per-step Parquet frames, whose frame 0 is the observation after action 0. For real-world collection, these fields must be sensor-derived or calibration-derived in the same world frame used by the simulator:

| Field | Why replay needs it |
|-------|---------------------|
| `tcp_pos`, `tcp_velocity` | Rebuild the robot/TCP initial condition and drive transition features |
| `tcp_quat` | Rebuild the TCP orientation for the robot/TCP initial condition |
| `ft_wrist` | Compare interaction wrench and build MMD features; must be bias-corrected or accompanied by bias metadata |
| `apple_pos` | Place the fruit body and grasp/weld reference |
| `apple_quat` | Place the fruit body orientation and grasp/weld reference |
| `robot_joint_q` | Restore robot joint positions for observation-only replay initialization |
| `woody_part_start_pos`, `woody_part_end_pos` | Reconstruct or validate branch/stem geometry by junction name |
| `excitation_type`, `excitation_f_inst`, `excitation_direction` | Replay the same excitation context and feature labels |

Additional calibration metadata (`robot_base_pos`, `fruiting_base_pos`, camera/F/T transforms, grasp/weld transform, fixture identity) belongs in episode metadata or a named digital-twin fixture, not as per-step observation keys unless it changes during an episode. See `docs/observation-replay-digital-twin.md`.

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
| `tcp_quat` | `(4,)` | Actual TCP body quaternion `[x, y, z, w]` (SysId only) |
| `apple_quat` | `(4,)` | Apple body quaternion `[x, y, z, w]` (SysId only) |
| `robot_joint_q` | `(7,)` | Robot joint positions used for replay initialization (SysId only) |

### `ApplePickReplayEnv`

| Key | Shape | Source |
|-----|-------|--------|
| `ft_wrist` | `(6,)` | Fresh TCP harvest via `_end_effector_wrench()` |
| `woody_start` | `(N*3,)` | Parent-side anchors (flat; replay env only) |
| `woody_end` | `(N*3,)` | Child-side anchors (flat; replay env only) |
| `tcp_pos` | `(3,)` | Actual TCP body position |
| `tcp_quat` | `(4,)` | Actual TCP body quaternion `[x, y, z, w]` |
| `apple_pos` | `(3,)` | Apple body COM |
| `apple_quat` | `(4,)` | Apple body quaternion `[x, y, z, w]` |
| `robot_joint_q` | `(7,)` | Robot joint positions used for observation-only replay initialization |

**Note:** `ft_wrist` semantics differ between Vic/SysId (lagged cache) and Replay (live harvest). Consumers must check env class or document source when mixing datasets.

## Tests

- `apple_pick_gym/tests/test_apple_pick_coupled_env.py` — Coupled parity and Replay obs contract
- `apple_pick_gym/tests/test_sysid_env.py` — SysId excitation obs keys

Run:

```bash
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/ -q -m "not slow"
```
