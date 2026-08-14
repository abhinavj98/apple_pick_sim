# Real replay → parallel sys-ID plumbing (`vic_pose`)

| Field | Value |
| ----- | ----- |
| **Status** | **Slice 1 implemented** (2026-08-12); slice 2 **superseded** by `2026-08-13-real-sim-cma-feature-alignment-design.md` |
| **Date** | 2026-08-12 |
| **Roadmap** | M4.0 bit 3 (real `robot_replay` → grid / CMA) |
| **Extends** | `docs/superpowers/specs/2026-08-11-batched-real-replay-post-grasp-se3-design.md` (slice B), `docs/superpowers/specs/2026-08-07-real-to-batched-metadata-parity-design.md` (bit 3), `docs/superpowers/specs/2026-08-10-vic-pose-action-controller-design.md` |
| **Code (today)** | `real_batched_replay_build.py`, `robot_replay/example_replay_real_batched.py`, `example_youngs_modulus_sys_id.py`, `example_youngs_modulus_cmaes.py` (CMA wiring = slice 4) |

## Purpose

Reuse the working real-replay **build** (convert → `vic_pose` 19D drive → open-loop FR3 → logged TCP gripper → post-grasp SE(3)) inside the existing fused evaluator so **N envs share one converted geometry** and differ only in `(support_kp, E_spur, E_stem)`.

This is plumbing, not a new optimizer. Cartesian grid is the first consumer; CMA is the same builder plus `fit_youngs_modulus_structures`.

## Locked decisions

| Topic | Choice |
| --- | --- |
| Approach | Extract real-replay build; opt existing grid/CMA in. Do not add a second `robot_replay` grid script |
| Phenotype | Unchanged: \(\log_{10}(k_p^{\mathrm{support}}, E_{\mathrm{spur}}, E_{\mathrm{stem}})\); primary \(E\) fixed |
| Drive | Full 19D `vic_pose`: logged target pose **and** logged Kp/Kd |
| `control_hz` | Recorded episode / collection rate (no 60 Hz default) |
| Sim-sim defaults | Twist `vic` / `action_dim=6` unchanged for gym collect, MMD, default CMA |
| Ranking GT | Real logged bags, not fixture `fruiting_system_params` / `collection.sim_config` as a recoverable phenotype |
| Slice 1 acceptance | N parallel envs replay without blow-up; TCP moves; wrench-as-twist refused. Sinkhorn F/T is **not** an acceptance metric |
| Multi-structure fused real | Out of slice 1 (one converted episode = one structure; one `fruiting_base_pos`) |

## Slice order

```text
1. Shared real-replay build + fused plumbing   ← slice 1 Done
2. Feature transform before loss   ← **superseded** by `docs/superpowers/specs/2026-08-13-real-sim-cma-feature-alignment-design.md` (convert-time F/T rotate, no LPF, woody/hold/USD COM)
3. Cartesian grid ranking (Sinkhorn on transformed bags)
4. CMA (same builder + evaluator)
```

Convert keeps storing logged columns as-is. ~~Slice 2 maps **bags at score time**.~~ Feature alignment (2026-08-13 spec) maps F/T at **convert time** and woody/hold in convert + collector; see that spec.

## Slice 1 — shared build

**Module:** `apple_pick_gym/batched_envs/real_batched_replay_build.py`

`robot_replay` already imports gym replay helpers. The example becomes a thin CLI. Do not put the env factory in `apple_pick_sim/` (`ApplePickBatchedSysIdEnv` is gym). Do not put it under `robot_replay/` (grid/CMA would depend the wrong way).

| Helper | Role |
| --- | --- |
| `dataset_declares_vic_pose` | `collection.action_layout == "vic_pose_v1"` or `action_dim == 19` |
| `control_hz_from_episode_metadata` | Episode else collection; required; must be `> 0` |
| `fruiting_base_pos_from_episode_metadata` | Logged T-junction |
| `bootstrap_joint_q_from_episode_metadata` | Open-loop FR3 joints |
| `check_action_semantics` | Refuse wrench-as-twist unless legacy 6D hatch |
| `real_replay_sim_config` | Today’s `_test_sim_config` |
| `make_real_replay_build_env_fn` | Today’s `_build_env_fn`, but **honor** `gripper` / `per_env_grippers` |

`real_replay_sim_config` knobs (copy from the working example):

- `controller.mode="vic_pose"`, `action_dim=19`
- `per_env_ik=False`, `skip_ik_bootstrap=True`, `bootstrap_joint_q` from episode
- `scene.fruiting_base_pos` from episode
- `post_grasp_settle_substeps=500` (grid `build_sim_config` leaves this at 0)
- fixture `sim_build` joint/VIC knobs
- recorded `control_hz` on `runtime` **and** passed into `ApplePickBatchedSysIdEnv`

`make_real_replay_build_env_fn` gripper resolution:

1. If `per_env_grippers` is set, use it (fused path).
2. Else if `gripper` is set, broadcast to `num_envs` (scalar `replay_batched_sysid_structure`).
3. Else synthesize `gripper_proxy_for_real_batched_replay(episode_meta)`.
4. After `ApplePickBatchedSysIdEnv(...)`, apply batched post-grasp SE(3).

Raise if both `gripper` and `per_env_grippers` are set (same as sim-sim `_make_build_env_fn`).

### Grid opt-in

`example_youngs_modulus_sys_id.py` uses the real builder when `dataset_declares_vic_pose` or `--controller-mode vic_pose`. Twist `vic` remains the default for sim-sim datasets.

On real datasets:

- `--include-gt-candidate` is forced **off** (warn if the user passed true). Do not call `gt_support_kp_from_dataset` (real convert has no `collection.sim_config`).
- Tiny Cartesian product of `SupportKpYoungsCandidate` is enough for the smoke.
- `ranking.json` may be written as plumbing; F/T Sinkhorn is not acceptance.

## Slice 1 — evaluator holes

### Batched post-grasp SE(3)

`apply_logged_post_grasp_se3_to_cable` today writes `cable.apple_body` / `gripper_proxy_body` (template / env 0).

Add optional `layout: BatchedEnvLayout | None = None`. When `layout` is set and `num_envs > 1`, write the **same** logged apple SE(3) + TCP offset into every `layout.apple_body_indices[w]` / `layout.proxy_body_indices[w]`. Woody bodies stay settled. Then sync VBD rest / `body_q_prev` once.

Call **after** env construct (free settle → `seed_fix_to_apple_from_settled`). Do not snap apple to post-grasp before settle.

This is slice B of `2026-08-11-batched-real-replay-post-grasp-se3-design.md`, implemented on the shared `build_env_fn` rather than inside `seed_fix_to_apple_*`.

### Gripper on the fused path

`gripper_proxy_from_episode_metadata` has no TCP offset. `gripper_proxy_for_real_batched_replay` sets \(X_{\mathrm{offset}} = X_{\mathrm{apple}}^{-1} X_{\mathrm{tcp}}\).

When the dataset is `vic_pose_v1`:

- `prepare_youngs_modulus_structure` stores the real gripper on `ReplayStructureRequest`
- `replay_batched_sysid_structure` passes the real gripper as `gripper=`
- Shared `build_env_fn` honors those args

For one real structure × N material candidates, every env gets the **same** real gripper. Gripper config owns the joint offset; the SE(3) helper owns runtime apple/proxy `body_q`. Both are required.

### `action_dim`

Scalar `replay_batched_sysid_structure` already takes `action_dim` (example passes 19). `replay_candidates_for_structure` / `evaluate_youngs_modulus_candidates` must thread it (infer 19 from `dataset_declares_vic_pose`, else 6). Fused replay stacks `slot.recorded["action"]` as-is; the env must be 19D. `EnvDisableController` already freeze-holds 19D rows.

### No sim-oracle GT

`PreparedYoungsModulusStructure.gt_candidate` and `YoungsModulusEvaluation.gt_candidate` become `YoungsModulusCandidate | None`. On real datasets they are `None`; `is_gt` is always false. `use_oracle_params=True` still means “use episode `fruiting_system_params` for **geometry**,” not “this \(E\) is truth to recover.”

## Slice 2 — feature transform (superseded)

**Superseded by** `docs/superpowers/specs/2026-08-13-real-sim-cma-feature-alignment-design.md` (convert-time `R(tcp) @` F/T rotate, no LPF, no pose-only action). Do not implement the plan below.

Do **not** implement in slice 1.

Before Sinkhorn:

1. **F/T frame:** sim `ft_wrist` is world wrench at TCP COM; real collected F/T is a different sensor frame. Transform bags into one convention at score time (prefer: map sim → real sensor frame, or real → sim world; pick one in the slice-2 spec and test it).
2. **LPF:** real `ft_wrist` is low-pass filtered. Apply the **same** filter to sim replay `ft_wrist`. Do not invert the real filter.
3. **Action features:** Sinkhorn state uses `action[0:7]` (target pose) only. Drop `action[7:19]` (Kp/Kd) from the bag. Controller still consumes full 19D.

Keep convert parquet columns unchanged.

## Slice 3–4 (later)

- Slice 3: Cartesian ranking with transformed Sinkhorn bags; no GT insert.
- Slice 4: `example_youngs_modulus_cmaes.py` uses the same real `build_env_fn` + `action_dim` + optional GT. No new search vector.

## Tests (slice 1)

- Unit: post-grasp SE(3) on a 2-env layout stub (not only env 0); env 0 regression still passes.
- Unit: `dataset_declares_vic_pose`; `check_action_semantics` refuse wrench-as-twist (move with the helper).
- Unit: `make_real_replay_build_env_fn` uses `per_env_grippers` when provided; synthesizes real gripper when not.
- Unit: `prepare_youngs_modulus_structure` on a vic_pose stub dataset uses real gripper and `gt_candidate is None` without `collection.sim_config`.
- CLI: grid `--controller-mode` / auto-detect; sim-sim default still twist `vic`.
- Smoke (optional if GPU): convert `s02-d00` → grid `--viewer null` tiny candidate list.

## Non-goals (slice 1)

- F/T frame / LPF / pose-only Sinkhorn features
- pycma `ask`/`tell` / `cmaes_report.json`
- Multi-episode manifest
- Migrating gym collect / MMD / sim-sim CMA off twist `vic`
- Changing convert packing (still 19D pose+gains)
