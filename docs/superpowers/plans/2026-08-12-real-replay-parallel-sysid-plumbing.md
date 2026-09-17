# Real Replay Parallel Sys-ID Plumbing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach the existing fused Young's grid evaluator to rebuild the working real-replay scene N times with different `(support_kp, E_spur, E_stem)` on one converted `vic_pose_v1` episode.

**Architecture:** Extract `example_replay_real_batched.py` build helpers into `apple_pick_gym/batched_envs/real_batched_replay_build.py`. Batch post-grasp SE(3) across `layout.apple_body_indices`. Opt `example_youngs_modulus_sys_id.py` into that builder when the dataset declares `vic_pose_v1`. Do not change sim-sim twist `vic` defaults. F/T frame/LPF/Sinkhorn acceptance is a follow-up plan (spec slice 2).

**Tech Stack:** Python, existing `ApplePickBatchedSysIdEnv`, `replay_batched_sysid_structure` / fused multi-replay, `SupportKpYoungsCandidate`, pytest + `uv run`.

**Spec:** `docs/superpowers/specs/2026-08-12-real-replay-cmaes-plumbing-design.md`

## Global Constraints

- Slice 1 only: plumbing + parallel replay smoke. No F/T frame transform, no LPF, no pose-only Sinkhorn features, no pycma loop
- Drive remains full 19D `vic_pose` (logged pose **and** logged Kp/Kd)
- `control_hz` from episode/collection metadata only (no 60 Hz default)
- Gym collect / MMD / default sim-sim CMA stay twist `vic`, `action_dim=6`
- One converted episode = one structure; do not fuse multiple real `fruiting_base_pos` in one batch
- TDD: failing test before production code; run with `uv run --env-file pytest.env python -m pytest …`
- Work on a feature worktree if the workspace is `main` (see `.cursor/rules/worktree-feature-dev.mdc`)

## File map

| Path | Responsibility |
| --- | --- |
| `apple_pick_sim/system_id/batched_digital_twin_init.py` | Batched `apply_logged_post_grasp_se3_to_cable(..., layout=)` |
| `apple_pick_gym/batched_envs/real_batched_replay_build.py` | **Create.** Shared real sim config + `build_env_fn` |
| `robot_replay/example_replay_real_batched.py` | Thin CLI over the shared module |
| `apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py` | Thread `action_dim`; real gripper on scalar replay |
| `apple_pick_gym/batched_envs/batched_sysid_cmaes.py` | Real gripper; optional `gt_candidate`; thread `action_dim` |
| `apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py` | Opt into real builder when `vic_pose_v1` |
| Tests listed per task | Behavior gates |
| `docs/ROADMAP.md`, `robot_replay/README.md` | Commands + checklist |

---

### Task 1: Batched post-grasp SE(3)

**Files:**
- Modify: `apple_pick_sim/system_id/batched_digital_twin_init.py`
- Test: `apple_pick_sim/tests/test_batched_digital_twin_init.py`

**Interfaces:**
- Consumes: existing `apply_logged_post_grasp_se3_to_cable(cable, meta)`
- Produces: `apply_logged_post_grasp_se3_to_cable(cable, meta, *, layout: BatchedEnvLayout | None = None) -> None`

- [ ] **Step 1: Write the failing test**

Keep `test_apply_logged_post_grasp_se3_to_cable_sets_apple_and_proxy` unchanged (layout=None / env 0).

Add a 2-env stub: 4 bodies, apples at 0 and 2, proxies at 1 and 3. After apply with `layout`, **both** apples match logged pose and both proxies match logged TCP.

```python
def test_apply_logged_post_grasp_se3_writes_every_layout_world():
    from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
    from apple_pick_sim.system_id.real_post_grasp_plan import proxy_offset_from_apple_and_tcp

    apple_pos = (0.5, 0.6, 0.7)
    apple_quat = (0.0, 0.0, 0.0, 1.0)
    tcp_pos = (0.5, 0.55, 0.7)
    tcp_quat = (0.0, 0.0, 0.0, 1.0)
    offset = proxy_offset_from_apple_and_tcp(
        apple_pos=apple_pos,
        apple_quat_xyzw=apple_quat,
        tcp_pos=tcp_pos,
        tcp_quat_xyzw=tcp_quat,
    )
    # Reuse the _Arr/_State stub pattern from test_apply_logged_post_grasp_se3_to_cable_sets_apple_and_proxy
    # but body_q shape (4, 7). Template apple_body=0, gripper_proxy_body=1.
    # World 1 starts at a different pose so the test fails if only env 0 is written.
    layout = BatchedEnvLayout(
        num_envs=2,
        bodies_per_world=2,
        robot_bodies_per_world=1,
        joints_per_world=1,
        joint_coord_count_per_world=1,
        joint_dof_count_per_world=1,
        template_tcp_body=0,
        template_proxy_body=1,
        template_apple_body=0,
        tcp_body_indices=(0, 0),
        proxy_body_indices=(1, 3),
        apple_body_indices=(0, 2),
    )
    apply_logged_post_grasp_se3_to_cable(cable, meta, layout=layout)
    out = cable.state_0.body_q.numpy().reshape(-1, 7)
    for apple_id in (0, 2):
        np.testing.assert_allclose(out[apple_id, :3], apple_pos, atol=1e-6)
    for proxy_id in (1, 3):
        np.testing.assert_allclose(out[proxy_id, :3], tcp_pos, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_batched_digital_twin_init.py::test_apply_logged_post_grasp_se3_writes_every_layout_world -q
```

Expected: FAIL (`TypeError` unexpected kwarg `layout`, or env 1 still at the dummy pose).

- [ ] **Step 3: Implement**

In `apply_logged_post_grasp_se3_to_cable`, after resolving `apple_pos` / `apple_quat` / `offset`:

```python
if layout is not None and int(layout.num_envs) > 1:
    pairs = list(zip(layout.apple_body_indices, layout.proxy_body_indices, strict=True))
else:
    pairs = [(int(apple_id), int(proxy_id))]
for aid, pid in pairs:
    if int(aid) < 0 or int(pid) < 0:
        continue
    bq[int(aid), 0:3] = np.asarray(apple_pos, dtype=np.float32)
    bq[int(aid), 3:7] = np.asarray(apple_quat, dtype=np.float32)
    proxy_pos, proxy_quat = _proxy_world_pose_from_apple(bq[int(aid)], offset)
    bq[int(pid), 0:3] = proxy_pos
    bq[int(pid), 3:7] = proxy_quat
    bqd[int(aid)] = 0.0
    bqd[int(pid)] = 0.0
```

Keep the existing `assign` / `align_proxy_body_q_prev_for_vbd` / `sync_model_body_q_rest_from_state` once after the loop.

- [ ] **Step 4: Run tests**

```bash
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_batched_digital_twin_init.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apple_pick_sim/system_id/batched_digital_twin_init.py apple_pick_sim/tests/test_batched_digital_twin_init.py
git commit -m "$(cat <<'EOF'
Apply logged post-grasp SE(3) to every batched world.

EOF
)"
```

---

### Task 2: Shared real-replay build module

**Files:**
- Create: `apple_pick_gym/batched_envs/real_batched_replay_build.py`
- Test: `apple_pick_gym/tests/test_real_batched_replay_build.py`

**Interfaces:**
- Consumes: Task 1 `apply_logged_post_grasp_se3_to_cable(..., layout=)`; existing `gripper_proxy_for_real_batched_replay`
- Produces:
  - `dataset_declares_vic_pose(collection: Mapping[str, Any], episode_meta: Mapping[str, Any] | None = None) -> bool`
  - `control_hz_from_episode_metadata(meta, *, collection=None) -> float`
  - `fruiting_base_pos_from_episode_metadata(meta) -> tuple[float, float, float]`
  - `bootstrap_joint_q_from_episode_metadata(meta) -> tuple[float, ...]`
  - `check_action_semantics(*, controller_mode, collection, episode_meta, allow_wrench_as_twist) -> None`
  - `real_replay_sim_config(...) -> BatchedHeterogeneousCoupledSimConfig`
  - `make_real_replay_build_env_fn(...) -> Callable`

Move bodies from `robot_replay/example_replay_real_batched.py` (`_test_sim_config`, `_build_env_fn`, metadata helpers, `check_action_semantics`). Do not leave a second copy of the sim-config knobs in the example after Task 3.

- [ ] **Step 1: Write failing tests** (module does not exist yet)

```python
def test_dataset_declares_vic_pose_from_layout_or_dim():
    from apple_pick_gym.batched_envs.real_batched_replay_build import dataset_declares_vic_pose
    assert dataset_declares_vic_pose({"action_layout": "vic_pose_v1"}) is True
    assert dataset_declares_vic_pose({"action_dim": 19}) is True
    assert dataset_declares_vic_pose({"action_dim": 6}) is False

def test_check_action_semantics_refuses_wrench_as_twist_on_vic_pose():
    from apple_pick_gym.batched_envs.real_batched_replay_build import check_action_semantics
    with pytest.raises(SystemExit, match="legacy 6D"):
        check_action_semantics(
            controller_mode="vic",
            collection={"action_dim": 19, "action_layout": "vic_pose_v1"},
            episode_meta={},
            allow_wrench_as_twist=True,
        )

def test_make_real_replay_build_env_fn_honors_per_env_grippers(monkeypatch):
    """Factory must pass fused per_env_grippers into ApplePickBatchedSysIdEnv, not delete them."""
    from apple_pick_sim.fruiting_system.params import GripperProxyConfig
    captured = {}

    class _FakeEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._sim = SimpleNamespace(scene=SimpleNamespace(cable=None, layout=None))

    monkeypatch.setattr(
        "apple_pick_gym.batched_envs.real_batched_replay_build.ApplePickBatchedSysIdEnv",
        _FakeEnv,
    )
    # Also stub real_replay_sim_config / gripper_proxy if construct would load fixtures.
    g0 = GripperProxyConfig()
    g1 = GripperProxyConfig()
    meta = {
        "control_hz": 15.0,
        "fruiting_base_pos": [0.0, 0.5, 0.95],
        "initial_robot_joint_q": [0.0] * 7,
        "initial_apple_pos": [0.1, 0.2, 0.3],
        "initial_apple_quat": [0.0, 0.0, 0.0, 1.0],
        "initial_tcp_pos": [0.1, 0.15, 0.3],
        "initial_tcp_quat": [0.0, 0.0, 0.0, 1.0],
    }
    fn = make_real_replay_build_env_fn(
        ranges_path=Path("apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json"),
        ranges=load_ranges("apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json"),
        topology_seed=0,
        fruiting_base_pos=(0.0, 0.5, 0.95),
        episode_meta=meta,
        bootstrap_joint_q=(0.0,) * 7,
        controller_mode="vic_pose",
        control_hz=15.0,
    )
    fn(num_envs=2, per_env_params=[None, None], max_episode_steps=4, per_env_grippers=[g0, g1])
    assert captured["per_env_grippers"] == [g0, g1]
```

Port the existing `check_action_semantics` cases from `apple_pick_gym/tests/test_real_batched_replay_cli.py` onto this module (keep example tests passing via re-export in Task 3).

- [ ] **Step 2: Run tests — expect fail** (import error)

```bash
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_real_batched_replay_build.py -q
```

- [ ] **Step 3: Implement the module**

`make_real_replay_build_env_fn` gripper rules:

```python
if gripper is not None and per_env_grippers is not None:
    raise ValueError("scalar gripper and per_env_grippers cannot both be provided")
if per_env_grippers is not None:
    grippers = list(per_env_grippers)
elif gripper is not None:
    grippers = [gripper] * int(num_envs)
else:
    real_g = gripper_proxy_for_real_batched_replay(dict(episode_meta))
    grippers = [real_g] * int(num_envs)
```

After env construct:

```python
scene = env._sim.scene
layout = getattr(scene, "layout", None)
cable = getattr(scene, "cable", None)
if cable is not None:
    apply_logged_post_grasp_se3_to_cable(cable, dict(episode_meta), layout=layout)
```

`real_replay_sim_config` must set `controller.mode` / `action_dim=19` for `vic_pose`, `per_env_ik=False`, `bootstrap_joint_q`, `fruiting_base_pos`, `post_grasp_settle_substeps=500`, recorded `control_hz` on `runtime`. Pass the same `control_hz` into `ApplePickBatchedSysIdEnv(control_hz=...)`.

- [ ] **Step 4: Run tests — expect pass**

```bash
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_real_batched_replay_build.py -q
```

- [ ] **Step 5: Commit**

```bash
git add apple_pick_gym/batched_envs/real_batched_replay_build.py apple_pick_gym/tests/test_real_batched_replay_build.py
git commit -m "$(cat <<'EOF'
Extract shared real-replay env build for vic_pose datasets.

EOF
)"
```

---

### Task 3: Thin the real replay example

**Files:**
- Modify: `robot_replay/example_replay_real_batched.py`
- Test: `apple_pick_gym/tests/test_real_batched_replay_cli.py` (must stay green)

**Interfaces:**
- Consumes: Task 2 public helpers
- Produces: example re-exports `check_action_semantics` so existing CLI tests keep importing from the example module

- [ ] **Step 1: Write / adjust a test** that the example’s `_test_sim_config` / `_build_env_fn` are gone **or** that they are thin wrappers. Prefer: example imports `make_real_replay_build_env_fn` and `real_replay_sim_config` and calls them from `_run`. Re-export:

```python
from apple_pick_gym.batched_envs.real_batched_replay_build import (
    bootstrap_joint_q_from_episode_metadata,
    check_action_semantics,
    control_hz_from_episode_metadata,
    fruiting_base_pos_from_episode_metadata,
    make_real_replay_build_env_fn,
    real_replay_sim_config,
)
```

If a CLI test asserts `cfg.controller.mode == "vic_pose"` via `_test_sim_config`, point it at `real_replay_sim_config` or keep a one-line alias `_test_sim_config = real_replay_sim_config`.

- [ ] **Step 2: Run existing CLI tests (red if imports break)**

```bash
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_real_batched_replay_cli.py -q
```

- [ ] **Step 3: Wire `_run` to `make_real_replay_build_env_fn` / `real_replay_sim_config`** with the same CLI args as today (`settle_*`, `controller_mode`, `control_hz`, `bootstrap_joint_q`, `fruiting_base_pos`, `episode_meta`). Pass `action_dim=19 if controller_mode == "vic_pose" else 6` into `replay_batched_sysid_structure` as today.

- [ ] **Step 4: Re-run CLI tests**

```bash
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_real_batched_replay_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add robot_replay/example_replay_real_batched.py apple_pick_gym/tests/test_real_batched_replay_cli.py
git commit -m "$(cat <<'EOF'
Point real batched replay example at the shared build helper.

EOF
)"
```

---

### Task 4: Real gripper + optional GT in prepare / scalar replay

**Files:**
- Modify: `apple_pick_gym/batched_envs/batched_sysid_cmaes.py`
- Modify: `apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py`
- Test: `apple_pick_gym/tests/test_batched_sysid_cmaes_candidate.py` (or a new `test_real_replay_prepare.py`)

**Interfaces:**
- Consumes: `dataset_declares_vic_pose`, `gripper_proxy_for_real_batched_replay`
- Produces: `PreparedYoungsModulusStructure.gt_candidate: YoungsModulusCandidate | None`; real gripper on `ReplayStructureRequest`; `replay_batched_sysid_structure` / `replay_candidates_for_structure` accept and forward `action_dim`

- [ ] **Step 1: Failing tests**

```python
def test_prepare_vic_pose_dataset_uses_real_gripper_and_skips_gt(tmp_path):
    """Manifest without collection.sim_config must not raise; gripper has TCP offset."""
    # Minimal BatchedSysIdDataset stub or tiny converted-style manifest:
    # collection = {action_layout: vic_pose_v1, action_dim: 19, num_directions: 1, seed: 0}
    # episode meta with initial_apple_*, initial_tcp_*, fruiting_system_params JSON
    prepared = prepare_youngs_modulus_structure(
        dataset=dataset,
        structure_idx=0,
        candidates=(SupportKpYoungsCandidate(1e4, 1e9, 1e9),),
        num_directions=1,
        scoring=YoungsModulusScoringConfig(use_median=True, hold_id_onehot=False, pool_directions=True, n_holds=1, n_directions=1, device="cpu"),
    )
    assert prepared.gt_candidate is None
    assert prepared.replay_request.gripper.weld_proxy_offset_in_apple_frame is not None

def test_score_is_gt_false_when_gt_candidate_is_none():
    """is_gt must not call youngs_modulus_values_match(None)."""
    # Build a PreparedYoungsModulusStructure with gt_candidate=None and one
    # SupportKpYoungsCandidate. Feed replay_by_key with a copy of the recorded
    # episode so Sinkhorn can run (or monkeypatch score_candidate_wasserstein_complete
    # to return a finite aggregate). Assert scores[0].is_gt is False.
```

Also: `replay_candidates_for_structure` must pass `action_dim` through; a unit test that a 19-wide recorded action is accepted when `action_dim=19` already exists in `test_batched_sysid_replay.py` (`test_stacked_recorded_actions_accepts_action_dim_19`). Add forwarding so `evaluate_youngs_modulus_candidates` does not default to 6 on vic_pose datasets.

- [ ] **Step 2: Run — expect fail** (`gt_support_kp_from_dataset` ValueError on missing `sim_config`)

```bash
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_batched_sysid_cmaes_candidate.py -q -k vic_pose
```

- [ ] **Step 3: Implement**

In `prepare_youngs_modulus_structure`, after loading metadata:

```python
meta = dataset.load_episode_metadata(int(structure_idx), first_direction_idx)
collection = dataset.manifest.get("collection", {})
real = dataset_declares_vic_pose(collection, meta)
if real:
    gripper = gripper_proxy_for_real_batched_replay(meta)
    gt_candidate = None
else:
    gripper = gripper_proxy_from_episode_metadata(meta)
    if isinstance(candidate_list[0], SupportKpYoungsCandidate):
        gt_candidate = gt_support_kp_youngs_candidate_from_structure(dataset, int(structure_idx))
    else:
        gt_candidate = youngs_modulus_candidate_from_params(base_params)
```

Change dataclass fields:

```python
gt_candidate: YoungsModulusCandidate | None
```

In `score_prepared_youngs_modulus_structure`:

```python
is_gt=(
    prepared.gt_candidate is not None
    and youngs_modulus_values_match(candidate, prepared.gt_candidate)
),
```

In `replay_batched_sysid_structure`, when `dataset_declares_vic_pose(...)`, use `gripper_proxy_for_real_batched_replay(structure_meta)`.

Thread `action_dim` (default 6) through `replay_candidates_for_structure` into `replay_batched_sysid_structure`. In `evaluate_youngs_modulus_candidates`, set

```python
action_dim = 19 if dataset_declares_vic_pose(dataset.manifest.get("collection", {})) else 6
```

and pass it in. Do not change fused stacking (already uses recorded width).

- [ ] **Step 4: Run candidate + replay tests**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_batched_sysid_cmaes_candidate.py \
  apple_pick_gym/tests/test_batched_sysid_replay.py \
  apple_pick_gym/tests/test_batched_sysid_cmaes_loop.py -q
```

Expected: PASS. Sim-sim GT insert still works.

- [ ] **Step 5: Commit**

```bash
git add apple_pick_gym/batched_envs/batched_sysid_cmaes.py apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py apple_pick_gym/tests
git commit -m "$(cat <<'EOF'
Use real TCP-offset grippers and skip sim-oracle GT on vic_pose datasets.

EOF
)"
```

---

### Task 5: Grid CLI opt-in

**Files:**
- Modify: `apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py`
- Test: `apple_pick_gym/tests/test_example_youngs_modulus_sys_id_cli.py`

**Interfaces:**
- Consumes: `make_real_replay_build_env_fn`, `real_replay_sim_config`, `dataset_declares_vic_pose`, `check_action_semantics`
- Produces: `--controller-mode {vic,vic_pose}` (default auto from dataset)

- [ ] **Step 1: Failing CLI tests**

```python
def test_parser_accepts_controller_mode_vic_pose():
    module = _load_module()
    parser = module._make_parser()
    args = parser.parse_args(["--dataset", "/tmp/ds", "--output", "/tmp/out", "--controller-mode", "vic_pose"])
    assert args.controller_mode == "vic_pose"

def test_run_vic_pose_dataset_uses_real_builder_and_skips_gt(monkeypatch, tmp_path):
    """When collection.action_layout is vic_pose_v1, do not call gt_support_kp_from_dataset."""
    # Monkeypatch evaluate_youngs_modulus_structures to capture build_env_fn / skip GPU.
    # Manifest collection: action_layout=vic_pose_v1, action_dim=19, control_hz=15, num_directions=1
    # Assert include_gt_candidate effectively False.
    # Assert captured sim_config.controller.mode == "vic_pose" and action_dim == 19
```

Default parse without the flag: `controller_mode` is `None` (auto). Sim-sim tests that call `_run` with a fake dataset must still get twist `vic` `build_sim_config`.

- [ ] **Step 2: Run — expect fail** (`unrecognized arguments: --controller-mode`)

```bash
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_example_youngs_modulus_sys_id_cli.py::test_parser_accepts_controller_mode_vic_pose -q
```

- [ ] **Step 3: Implement `_run` branch**

```python
episode_meta = dataset.load_episode_metadata(structure_indices[0], 0)
collection = dataset.manifest.get("collection", {})
mode = args.controller_mode
if mode is None:
    mode = "vic_pose" if dataset_declares_vic_pose(collection, episode_meta) else "vic"
check_action_semantics(
    controller_mode=mode,
    collection=collection,
    episode_meta=episode_meta,
    allow_wrench_as_twist=False,
)
if mode == "vic_pose":
    if bool(args.include_gt_candidate):
        print("warning: --include-gt-candidate ignored for vic_pose_v1 (no sim-oracle GT)", file=sys.stderr)
    control_hz = control_hz_from_episode_metadata(episode_meta, collection=collection)
    fruiting_base_pos = fruiting_base_pos_from_episode_metadata(episode_meta)
    bootstrap_joint_q = bootstrap_joint_q_from_episode_metadata(episode_meta)
    build_env_fn = make_real_replay_build_env_fn(
        ranges_path=Path(ranges_path),
        ranges=ranges,
        topology_seed=int(collection.get("topology_seed", collection.get("seed", 0))),
        fruiting_base_pos=fruiting_base_pos,
        episode_meta=episode_meta,
        settle_substeps=settle_config.get("settle_substeps") or 5000,
        settle_quiet_every=settle_config.get("settle_quiet_every"),
        settle_gravity_ramp=bool(settle_config.get("settle_gravity_ramp")),
        post_grasp_settle_substeps=500,
        bootstrap_joint_q=bootstrap_joint_q,
        controller_mode="vic_pose",
        control_hz=control_hz,
    )
    replay_sim_config = real_replay_sim_config(
        num_envs=1,
        topology_seed=int(collection.get("topology_seed", collection.get("seed", 0))),
        fruiting_base_pos=fruiting_base_pos,
        ranges=ranges,
        bootstrap_joint_q=bootstrap_joint_q,
        controller_mode="vic_pose",
        control_hz=control_hz,
        post_grasp_settle_substeps=500,
    )
    include_gt = False
else:
    # existing _make_build_env_fn + build_sim_config path
    include_gt = bool(args.include_gt_candidate)
```

Pass `include_gt` into `_candidates_for_structure`. Do not call `gt_support_kp_youngs_candidate_from_structure` when `include_gt` is False **and** the dataset is vic_pose (that helper still requires `sim_config`). Change `_candidates_for_structure` to skip GT lookup when `include_gt` is False:

```python
candidates = candidates_from_support_kp_grid_cli(...)
if include_gt:
    gt = gt_support_kp_youngs_candidate_from_structure(dataset, int(structure_idx))
    candidates = maybe_include_gt_candidate(candidates, gt, include_gt=True)
```

Today it always loads GT first. That is the crash on real convert.

- [ ] **Step 4: Run grid CLI tests**

```bash
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_example_youngs_modulus_sys_id_cli.py -q
```

Expected: PASS, including existing sim-sim include-gt tests.

- [ ] **Step 5: Commit**

```bash
git add apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py apple_pick_gym/tests/test_example_youngs_modulus_sys_id_cli.py
git commit -m "$(cat <<'EOF'
Opt Young's grid into real vic_pose replay build.

EOF
)"
```

---

### Task 6: Docs + ROADMAP

**Files:**
- Modify: `docs/ROADMAP.md` (M4.0 checklist: shared path / grid opt-in; note slice 2 F/T transform next)
- Modify: `robot_replay/README.md` (convert → grid smoke command)
- Modify: `docs/superpowers/specs/2026-08-11-batched-real-replay-post-grasp-se3-design.md` (slice B: implemented via shared `build_env_fn` + batched helper)
- Modify: `docs/superpowers/specs/2026-08-12-real-replay-cmaes-plumbing-design.md` status → Slice 1 implemented (when code lands)

- [ ] **Step 1: Add README command** (from repo root)

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input robot_replay/s02-d00.parquet \
  --dataset-out /tmp/real_batched_s02_d00 --overwrite

uv run python apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py \
  --dataset /tmp/real_batched_s02_d00 \
  --output /tmp/real_kp_e_grid \
  --viewer null \
  --support-kp-values 1e3,1e4 \
  --log10-e-spur 9.0 \
  --log10-e-stem 9.0 \
  --no-include-gt-candidate \
  --overwrite
```

State explicitly: ranking F/T is **not** trusted until slice 2 (frame + LPF). Success is: envs build, 19D steps, no wrench-as-twist, no `sim_config` crash.

- [ ] **Step 2: Optional GPU smoke** if a converted dataset and CUDA are available; otherwise document skip.

```bash
uv run python apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py --help
```

Expected: `--controller-mode` in help.

- [ ] **Step 3: Commit**

```bash
git add docs/ROADMAP.md robot_replay/README.md docs/superpowers/specs/2026-08-11-batched-real-replay-post-grasp-se3-design.md docs/superpowers/specs/2026-08-12-real-replay-cmaes-plumbing-design.md
git commit -m "$(cat <<'EOF'
Document real vic_pose grid smoke and defer F/T feature transform.

EOF
)"
```

---

## Out of this plan (next plans)

- **Slice 2:** F/T frame alignment + LPF on sim `ft_wrist` + Sinkhorn `action[0:7]` only
- **Slice 3:** Trusted Cartesian ranking
- **Slice 4:** CMA `example_youngs_modulus_cmaes.py` on the same builder
