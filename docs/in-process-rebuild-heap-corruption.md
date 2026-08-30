# In-process rebuild heap corruption (Warp/Newton)

**Last updated:** 2026-08-27 (likely root cause identified upstream — see the OpenUSD section)

**Status:** Open. Reproducible but intermittent. Strong upstream match to an OpenUSD
`LoadUsdPhysicsFromRange` heap-corruption bug fixed in usd-core 26.5, which Newton's
dependency pin currently excludes. Mitigated in production by process-isolated CMA
evaluation waves.

**Scope:** Rebuilding `BatchedHeterogeneousCoupledSim` many times in a single process and stepping each scene. Hit primarily by Young's-modulus CMA with `--no-isolated-eval-waves`, and by `stress_plant_rebuild_loop.py --mode rebuild-replay`.

---

## Symptom

After one or more in-process scene rebuilds, the process dies with a memory-corruption symptom during simulation stepping. The observed failures are all different surface errors from one underlying cause:

| Where | Surface error |
|---|---|
| `ModelBuilder.color()` → `graph_coloring_get_groups` → `pack_arg` → `value.__ctype__()` | `TypeError: 'function' object is not subscriptable` |
| `env.step` → VBD → `body_hessian_al.zero_()` | `AttributeError: 'code' object has no attribute '_apic_ensure_tracked'` |
| `env.step` → VBD `_solve_rigid_body_iteration` → `wp.launch` → `context.py:10347` | SIGSEGV in `ctypes.addressof(x) for x in params` |
| `env.step` → VBD `_solve_rigid_body_iteration` → `pack_arg` → `runtime.get_device()` | SIGSEGV at function entry |

GPU memory and RSS stay flat, so this is host-side object corruption, not a VRAM or allocation leak.

## What the evidence establishes

**It is a dangling pointer, not a logic bug.** `ctypes.addressof(x)` and function-entry faults only occur when the referenced `PyObject` has been freed. Every crash site is a *victim*, not the culprit — chasing the crash line is a dead end.

**The dangling reference is held at C level, not from Python.** A live Python reference would keep the object alive via refcounting, so a dangling pointer implies something holds the address without owning a reference.

**The poison values identify the recycler.** In two captures the corrupted slots contained a `code` object and a `function` object. Those are exactly what Warp module compilation allocates in bulk. This fits a stale pointer into memory freed when the previous scene died, which the next rebuild's codegen then recycles into fresh `code`/`function` objects.

**The second scene is born broken.** In the reproduction, wave 0 ran all 250 steps cleanly and wave 1 died at step 11. This is cross-rebuild contamination, not gradual wear or resource exhaustion.

**Stepping is required.** A rebuild/teardown loop that never calls `env.step()` does not reproduce the bug. See the methodology warning below.

## What has been ruled out

- **CUDA graph capture** — not used anywhere in the CMA path (only in unrelated examples).
- **Resource exhaustion** — across the 11 generations before a crash, RSS went 1417 → 2303 MiB and GPU stayed flat at ~788 MiB.
- **Pymalloc buffer overrun** — `PYTHONMALLOC=debug` reports no bad-block or trailing-byte violation, so this is not an overrun of a Python-allocated block.
- **The replicated MuJoCo robot cache** — the bug reproduces with MuJoCo reuse disabled.
- **Other project-level globals** — `apple_pick_sim` has no module-level cache holding Warp or GPU objects; all module globals are constants.
- **Scalar retry** — previously suspected as an amplifier and since removed. It was unreachable in the single-structure configuration where the bug appears.

## Methodology warnings

**The bug is intermittent.** Observed crash points range from CMA generation 1 to generation 11, and in the harness from wave 1 step 11 to surviving well past that with identical settings. **A single passing run proves nothing.** Any A/B comparison must be a crash *rate* over repeated trials, not one run per arm.

**A harness that does not step is not a reproduction.** An earlier "corruption proof" ran `stress_plant_rebuild_loop.py --mode rebuild-replay` for 51 iterations under `PYTHONMALLOC=debug` and reported clean. Every iteration logged `replay_steps=0` and `step_s=0.000` — it never called `env.step()` once, so it exercised build and teardown only. The result carried no information. Always confirm `step_s` is non-zero before trusting a negative result.

## Reproduction

Runs at CMA scale (50 envs) and crashes far sooner than a full CMA search — minutes rather than ~60 of them. Intermittent, so repeat it.

```bash
PYTHONMALLOC=debug uv run python -X faulthandler \
  apple_pick_sim/examples/stress_plant_rebuild_loop.py \
  --mode rebuild-replay --dataset tmp/real_batched_s09_k_frame \
  --num-envs 50 --cycles 30 --resets-per-wave 1 \
  --replay-steps 250 --settle-substeps 1000 --post-grasp-settle-substeps 500 \
  --direction-indices 0 1 2 3 4 --reuse-replicated-mujoco
```

`--replay-steps` must be well above zero. Verify the log shows non-zero `step_s` per frame.

## Upstream match: OpenUSD `LoadUsdPhysicsFromRange` heap corruption

**This is now the leading explanation.** Newton issue
[#3655](https://github.com/newton-physics/newton/issues/3655), "Repeated USD visual-shape
imports corrupt the host heap", reports the same profile: repeated in-process
`ModelBuilder.add_usd` corrupts the host heap and surfaces as a segfault or `malloc()`
error at an unpredictable later point, with "the failure count varies with allocator
layout."

It was closed in favour of [#3293](https://github.com/newton-physics/newton/issues/3293),
which identifies the root cause as OpenUSD's `UsdPhysics.LoadUsdPhysicsFromRange`
heap-corrupting when a single rigid body owns many colliders. Crash probability rises with
shape count (that issue measures `rigid N=16` at 1/10, `N=45` at 10/10; static colliders
never crash). **Fixed in usd-core 26.5.**

This repo is exposed on all three counts:

- `newton/pyproject.toml` previously pinned `usd-core>=25.5,<26.5` (capped
  immediately below the fix); relaxed to `>=26.5,<27`.
- Pre-mitigation installs were on **usd-core 26.3**, inside the affected range.
- FR3 import now passes `load_visual_shapes=False` (was default `True`).

This reframes an earlier assumption. Rebuild *count* is not the trigger — a single import
can seed the corruption, which then surfaces much later during heavy allocation. That
matches the reproduction, where MuJoCo reuse was on (so only wave 0 imported USD) yet the
crash landed in wave 1. It also explains why disabling MuJoCo reuse did not help: that
*increases* the number of imports.

Not yet proven to be this repo's bug — the crashes land inside VBD stepping rather than
inside `add_usd` — but delayed surfacing is exactly what #3655 describes.

### Mitigations (applied / available)

1. **Upgrade usd-core to ≥26.5** — applied via Newton's pin (`>=26.5,<27`).
2. **`load_visual_shapes=False`** — applied in
   `apple_pick_sim/robot/fr3_robot/setup.py`. Operation is headless-only (the
   base env rejects any `render_mode` but `None`; CMA runs `--viewer null`), so
   visual shapes are wasted work.
3. **`PXR_WORK_THREAD_LIMIT=1`** before pxr initializes — optional env-only
   belt-and-suspenders; #3655 reported this alone took repeated visual-shape
   imports from corrupting to 32/32 clean.

## Earlier hypothesis (superseded)

Warp/Newton in-process model lifecycle, exposed by a rebuild pattern that upstream does not commonly exercise. With the robot cache and all project globals eliminated, the state surviving a rebuild is Warp/Newton internal: the module registry, loaded CUDA modules, the memory pool, and Newton's `module="unique"` kernels. Note that `apple_pick_gym/batched_envs/apple_pick_batched_base_env.py::close()` already documents a related known gap — Newton does not unregister contact `@wp.func` globals per model.

This is inference from the poison identity, not proof.

## Next steps

1. **Test the OpenUSD mitigations against a measured crash rate.** Run the reproduction with
   `PXR_WORK_THREAD_LIMIT=1`, and separately with `load_visual_shapes=False`, comparing crash
   *rates* over repeated trials against a baseline collected under identical parallelism.
2. **C-level backtrace at the fault.** Run the reproduction under `gdb --batch`. This distinguishes CPython dereferencing a dead `PyObject` (a Python-layer lifetime bug) from a fault inside `libwarp` (a native bug). Core dumps are currently unavailable on this host (`ulimit -c` is 0, `core_pattern` routes to apport), so gdb must attach directly.
2. **Measure a crash rate**, not a pass/fail, for any configuration comparison.
3. **Minimal Newton-only reproduction.** Rebuild and step a Newton model in a loop using only Newton APIs. If it still crashes, the issue is upstream and should be reported there.

## Mitigation

Process-isolated evaluation waves avoid the bug entirely by resetting process state between waves, and this is the CMA default. `--no-isolated-eval-waves` is a known-unsafe path and should not be used for production searches until this is root-caused.

## Related

- `docs/handbook-youngs-cma.md` — CMA evaluation waves and isolation flags.
- `apple_pick_sim/examples/stress_plant_rebuild_loop.py` — reproduction harness.
- `apple_pick_gym/batched_envs/apple_pick_batched_base_env.py` — `close()` teardown ordering (drain before free).
