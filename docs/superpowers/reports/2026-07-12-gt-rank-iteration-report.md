# GT rank iteration report (give-up)

**Date:** 2026-07-12  
**Worktree:** `apple_pick_sim-sysid-stable-collect` (`feature/sysid-stable-collect`)  
**Goal:** Make `scripts/collect_and_rank_sysid_gt.sh` show GT among the best candidates.  
**Outcome:** **Not achieved.** Stopped after fixture + disable/exclude fixes; ranking remains blocked or GT is not preferred.

## What we ran

| Run | Config | Result |
|-----|--------|--------|
| A | New fixture VIC; soft-disable on any `stable=False` | Trajectories truncated (~5–15 frames); every `(s,d)` excluded |
| B | Hard-disable only NaN/IK; exclude if unstable frac >25% | Full-length eps (~541 frames) but **46–70%** frames `force_cap`; all still excluded |
| C | Forced MSE rank with exclusions cleared (`--include-excluded`, structure 0) | All **10/10 candidates DISQUALIFIED**; viz GT ranks: pos=7, force=2, torque=7; `best_is_gt=False` |

Logs: `tmp/collect_and_rank_sysid_gt_loop.log`, `tmp/gt_rank_forced.log`.

## Root cause

Copied fixture gains are incompatible with the 30 N stem / stability force cap:

```json
"vic_gains": { "linear_k": 400.0, "linear_d": 15.0, "angular_k": 80.0, "angular_d": 5.0 }
```

vs `DEFAULT_STEM_FORCE_CAP_N = 30` and `StabilityThresholds.max_force_n = 30`.

Observed on GT collect (structure 0–1, 5 directions):

- Force median / p95 / max ≈ **30 N** on every episode  
- Unstable fraction ≈ **0.46–0.70** (almost all from `force_cap_exceeded`)  
- Wrist force is **saturated**, so force MSE cannot separate stiffness multipliers (0.1× / 1× / 10×)  
- Replay candidates also saturate → `unstable_fraction_all > 10%` → mass disqualification  

So this is not a small ranking-bug; the excitation is wrench-limited.

## Fixes already landed (useful, but not enough)

1. `EnvDisableController` + collect/replay soft-disable  
2. `hard_blowup_mask`: sticky-disable only on NaN/Inf or IK bootstrap (not force/speed caps) — commit `d4e5e3c`  
3. Offline exclude requires **>25%** unstable frames (not a single spike)  
4. Usable-direction remapping + `--include-excluded`  

These stop false truncations from one force spike, but do **not** restore discriminative force/pose ranking under continuous cap saturation.

## Why giving up now

Further iteration without **retuning the fixture / caps together** would be guessing:

- Lower `linear_k` (back toward ~10–50) **or**  
- Raise stem force/torque caps **and** `StabilityThresholds` in lockstep **or**  
- Score only unsaturated channels / drop force until caps are fixed  

That is the planned stability-monitor / fixture deep-dive, outside “quick GT-rank green” scope.

## Recommended next steps (human)

1. **Co-tune** VIC gains and `stem_force_cap_N` / monitor `max_force_n` so hold forces sit **below** the cap with headroom (e.g. p95 force ≪ cap).  
2. Re-collect with that fixture; confirm unstable frac ≪ 10% on GT.  
3. Re-run `collect_and_rank_sysid_gt.sh` and require `best_is_gt=True` (or `gt_rank_combined ≤ 2`) on ≥ most structures.  
4. Keep hard-blowup disable; treat force-cap as a **quality warning**, not episode death, until caps are consistent.  
5. Fix shell quirk: grid `SystemExit("…")` sometimes reported as `failed with exit 0` under retries — make failure non-zero and fail the script when no usable directions remain.

## Success criteria (not met)

- [ ] Default script path produces usable directions after exclude  
- [ ] Final hold rank: `best_is_gt=True` or GT among top-2 on most structures  
- [ ] Candidates not mass-disqualified from force-cap instability  
