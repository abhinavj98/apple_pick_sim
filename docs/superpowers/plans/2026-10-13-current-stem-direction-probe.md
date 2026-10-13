# Current Stem-Direction Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find out whether the v2f policies track the stem's *current* direction as it bends during the pull, and whether
their commanded rotation follows it. Every direction probe so far only tested *fixed per-episode* directions (as sampled,
or at rest at step 0).

**Architecture:** This is analysis tooling (a spike) under `runs/vic_harvest/analysis_v2f/`, which `.gitignore` excludes
(`runs/`). No product code changes.
1. Extend the deterministic probe rollout to log the stem chord and spur-tip orientation at every step.
2. Re-run the 8 rollouts (2 policies × 2 world families × train/holdout).
3. Fit ridge/MLP probes of the current direction from what the policy sees.
4. Measure whether the rotate action follows the current direction or the bend.

**Tech Stack:** Python 3.12 (uv), numpy, scikit-learn (RidgeCV, MLPRegressor), torch, and warp (sim state). The GPU runs
under the user-scoped CUDA MPS daemon (`~/.cache/apple_pick_mps`).

**Spec:** None (spike). The question was agreed in chat on 2026-10-13: "for rotating the stem, some identification would be
useful", and then "are you comparing against rest direction or actual direction?" → yes, probe the actual direction.

## Probes run so far, and why

Each probe asks one question: can a plant quantity be decoded from what the policy has seen? Each decode compares five
feature sets, so that "the policy learned it" can be told apart from "the inputs make it easy":
- `lstm`: the trained actor's [h, c] state.
- `random`: a random-weight twin LSTM fed the same observations.
- `obs_now`: the current observation.
- `obs_win`: the last 10 observations.
- `grasp`: gravity in the tool frame at the grasp, i.e. what is known before any interaction.

Fits use the train worlds and the R² is scored on separate holdout worlds. The behaviour probe asks a second question: does
the *action* depend on the quantity beyond the grasp?

| Round | Script | Question | Why it was asked | Finding |
|---|---|---|---|---|
| r7 (D15-D17) | `analysis_r7/probe_fit.py`, `probe_nonlinear.py`, `probe_lc.py` | Does run 7's LSTM hold plant parameters (support kp, bending moduli, arm DR) or the stem rest tilt? | Is the recurrent actor doing sys-ID or acting as a schedule? | Support kp: lstm 0.28 at t=60 vs random 0.51 (the inputs carry it; the policy keeps less). Stem tilt c1: lstm 0.54-0.59 vs grasp 0.42 vs random 0.55, which matches the random twin, so it is not learned. |
| r7 | `analysis_r7/probe_task.py` | What does the memory track? (tension, detach index u, stem bend, wrist force, progress) | Understand the controller | Task state is tracked (tension 0.95, wrist force 0.95, bend 0.60), but `obs_now` gives the same: it is a feedback readout, not memory. |
| r7 | `analysis_r7/feedback_tests.py` | Feedback controller or grasp-conditioned schedule? (open loop, F/T frozen, observation ablations) | Same | Mostly a schedule; F/T freezing costs about 1.5 pts. |
| D17o wide | `analysis_vvwide/vvwide_probe.py` (wide r2) | Does wider plant DR make the policy identify support stiffness? | Make identification pay | Readable from obs at 0.75-0.83 (MLP, t=60). The LSTM holds less (0.67) than the obs or the random twin (0.81). There is no behavioural adaptation. |
| D17p vv-wide | `analysis_v2e/vvwide_probe_v2e.py`, `lstm_sysid_probe_v2e.py` | Same on 1e5-1e11 Pa plants. Upper bound: how much the (obs, act) sequence reveals at all (a freshly trained supervised LSTM) | Is the information even there? | Upper bound by t=40: support kp 0.86, spur modulus 0.70, stem modulus 0.51. The policy's memory does not exceed the random twin. M1 (memory wiped every step) costs 9 pts on vv-wide plants: a short-horizon filter, not a persistent estimate. |
| D17p | `analysis_v2e/theta_probe_v2e.py` | Is the stem-only bend angle θ_b (now / at detach) readable? | Prerequisite for a rotate-to-unbend strategy | Weak: about 0.2-0.4 from any feature set, and lstm ≈ random. |
| D17p | `analysis_v2e/pivot_bend.py` | Does the policy pivot the apple (rotation about the apple centre) or drag it? | Requested stem-bend / rotation analysis | It drags: apple-centre travel / (r × rotation) is 6-8. More rotation goes with *less* bend (corr -0.37). No rotate-to-bend strategy. |
| v2f (today) | `analysis_v2f/probe_v2f.py` | With varied geometry (true / wide_true fixtures, 30° downward spur cone), does it identify materials, geometry or stem / spur direction? | The maintainer expected more identification from more varied plants | Materials: support kp 0.33-0.47 (lstm) ≈ random; stem modulus and roll kp ≈ 0. Geometry ≈ 0. Directions: the grasp alone gives everything that is decodable (spur x 0.94, stem rest z 0.70, stem sampled x 0.55); interaction adds nothing. Behaviour: the grasp predicts actions at 0.96-0.99; the true directions add ≈ 0 to rotate. |

**Gap this plan closes.** The v2f direction targets are all static: the stem direction as sampled (`per_env_params.stem.direction`,
before settling), the stem rest chord at reset (`_rest_chord_world`), the spur direction as sampled, and the stem tilt off
gravity derived from the rest chord. None of them is the direction *at step t* while the stem bends. The current direction
is the one a rotation should respond to, and it may be easier to read from the force direction than the rest direction is.
The rollouts store only the scalar bend angle θ_b, not its direction.

## Global Constraints

- Delete files only inside `/tmp`. Everything else needs the maintainer's explicit OK, and a denial must not be worked
  around (e.g. by deleting from Python). (AGENTS.md)
- Use `$CLAUDE_JOB_DIR/tmp` (`/home/abhinav/.claude/jobs/23182e4c/tmp`) for scratch files, not `/tmp`.
- Write the new rollouts to `runs/vic_harvest/analysis_v2f/probe_cur/`. Do not overwrite `analysis_v2f/probe/` (today's
  results depend on it).
- Use the checkpoints `runs/vic_harvest/v2f_true_s0/checkpoints/ckpt_000032000` and
  `runs/vic_harvest/v2f_widetrue_s0/checkpoints/ckpt_000032000` (the same ones probed today).
- Use the world sets `apple_pick_gym/world_sets/snapshots/harvest_worlds_{true,wide_true}_{train,holdout}`: 5000 train /
  1000 holdout worlds.
- Run one eval/rollout per process (the env-rebuild crash rule), at most 4 in parallel, with
  `CUDA_MPS_PIPE_DIRECTORY=/home/abhinav/.cache/apple_pick_mps/pipe CUDA_MPS_LOG_DIRECTORY=/home/abhinav/.cache/apple_pick_mps/log`.
- Frames: v2f uses `action_frame: "tool"`, so the rotate action (dims 3:6) is a rotation vector in the *current* TCP frame.
  `bend_frame: "spur"` means θ_b is measured against the rest chord carried with the spur tip.
- Quaternions are `[x, y, z, w]`. Use `rot(q, v)` from `analysis_r7/probe_fit.py` to rotate v by q; the inverse is the
  conjugate (negate xyz).

## Review Focus

1. **Off-by-one between the logged chord and the action.** The chord is logged *after* `wrapper.step` at step t, but
   `act[:, t]` was chosen *before* it, from `obs[:, t]`. The decode target for the observation at step t must be
   `chord[:, t-1]` (the state the observation reports on). Task 3 pins this, following `probe_task.py`'s convention.
2. **Frozen / finished envs.** After success or a trip, the env freezes and the chord stops changing. Use only rows with
   `live[:, t] & ~invalid`, as `probe_v2f.py` does. Task 3 pins this.
3. **Sign ambiguity of "rotate to unbend".** The bend axis is `rest × current`. Undoing the bend means rotating about
   `-(rest × current)`. Report the signed cosine and compare it against a world-shuffled baseline, so that a ±0 result is
   not read as an effect. Task 3 pins this.
4. **Rest-chord frame.** θ_b is measured in the spur-tip frame. Recomputing it from the logged chords must match the env's
   `theta` within about 1°, or the frames are wrong. Task 1's sanity check pins this.
5. **Tiny bends early on.** At t ≤ 4 the bend is under 1°, so its direction is noise. Report the bend-direction cosine
   only for rows with bend ≥ 2°, and print n. Task 3 pins this.

---

### Task 1: Log the per-step stem chord and spur-tip orientation; sanity-check frames on a smoke rollout

**Files:**
- Modify: `runs/vic_harvest/analysis_v2f/probe_rollout_v2f.py` (`POSE` dict, `log_pose`, `static`)
- Create: `runs/vic_harvest/analysis_v2f/check_current_chord.py`

**Interfaces:**
- Consumes: `env._target_stem_axis() -> (N,3)` unit world chord (spur-stem anchor → stem-apple anchor), and
  `env._target_parent_quat() -> (N,4)` spur-tip world quat, both in `apple_pick_vic_harvest_env.py:428,496`.
- Produces new npz keys:
  - `stem_chord_world (N,T,3)`: chord after step t.
  - `spur_tip_quat (N,T,4)`: spur-tip quat after step t.
  - `spur_tip_quat0 (N,4)`: spur-tip quat at reset.

The first two keys are already added (applied 2026-10-13, before this plan). This task adds `spur_tip_quat0` and verifies
all three.

- [ ] **Step 1: Add the reset spur-tip quat to `static`**

In `static = {...}`, after `"rest_chord_world": ...`, add:

```python
        "spur_tip_quat0": env._target_parent_quat().float().cpu().numpy(),  # spur tip at reset (bend_frame "spur")
```

- [ ] **Step 2: Write the sanity check**

```python
"""[v2f] Frame check for the per-step stem chord logged by probe_rollout_v2f.py.
1. After the first step the chord is still within a few degrees of the rest chord.
2. The stem-only bend recomputed from the chords (rest chord carried with the spur tip) matches the env's theta_b."""
import sys
import numpy as np
sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parents[1].joinpath("analysis_r7").as_posix())
from probe_fit import rot  # noqa: E402

z = np.load(sys.argv[1])
ok = ~z["invalid"]
c, rest = z["stem_chord_world"], z["rest_chord_world"] / np.linalg.norm(z["rest_chord_world"], axis=1, keepdims=True)
q0i = z["spur_tip_quat0"].copy(); q0i[:, :3] *= -1
rest_spur = rot(q0i, rest)                                     # rest chord in the spur-tip frame at reset
T = c.shape[1]
carried = rot(z["spur_tip_quat"].reshape(-1, 4), np.repeat(rest_spur, T, 0)).reshape(c.shape)  # carried with the tip
ang = np.degrees(np.arccos(np.clip((c * carried).sum(-1), -1, 1)))
a0 = np.degrees(np.arccos(np.clip((c[:, 0] * rest).sum(-1), -1, 1)))[ok]
live = z["live"] & ok[:, None]
err = np.abs(ang - z["theta"])[live]
print(f"step-0 drift from rest: median {np.median(a0):.2f} deg, p99 {np.percentile(a0, 99):.2f}")
print(f"|recomputed - env theta_b|: median {np.median(err):.3f} deg, p99 {np.percentile(err, 99):.3f}")
assert np.percentile(a0, 99) < 3.0, "chord after step 0 far from rest: wrong key or frame"
assert np.percentile(err, 99) < 1.0, "recomputed stem-only bend disagrees with theta_b: frame mismatch"
print("OK")
```

- [ ] **Step 3: Smoke rollout (64 worlds) and run the check**

```bash
cd /home/abhinav/codes/apple_pick_sim/.claude/worktrees/rl-skrl-ppo
export CUDA_MPS_PIPE_DIRECTORY=/home/abhinav/.cache/apple_pick_mps/pipe CUDA_MPS_LOG_DIRECTORY=/home/abhinav/.cache/apple_pick_mps/log
S=/home/abhinav/.claude/jobs/23182e4c/tmp/smoke_cur
uv run python runs/vic_harvest/analysis_v2f/probe_rollout_v2f.py \
  --checkpoint runs/vic_harvest/v2f_widetrue_s0/checkpoints/ckpt_000032000 \
  --world-dir apple_pick_gym/world_sets/snapshots/harvest_worlds_wide_true_holdout --num-envs 64 --max-step 120 --out $S.npz
uv run python runs/vic_harvest/analysis_v2f/check_current_chord.py $S.npz
```

Expected: `saved ...`, then two summary lines, then `OK`.
- If check 2 fails with an error of tens of degrees: the logged spur quat is the wrong body. Compare against
  `env._target_parent_body` and fix it before going on.
- If the error is small but nonzero: `theta` may be logged before vs after the step. Compare `ang[:, t]` against
  `theta[:, t]` and `theta[:, t+1]`, and use the matching index convention in Task 3.

### Task 2: Full rollouts (8 files)

**Files:**
- Use: `runs/vic_harvest/analysis_v2f/run_probes_cur.sh` (already created: a copy of `run_probes.sh` that writes to
  `probe_cur/`)

**Interfaces:**
- Produces: `runs/vic_harvest/analysis_v2f/probe_cur/{true,widetrue}_{true,widetrue}_{train,holdout}.npz` with the same keys
  as `probe/` plus the three new ones.

- [ ] **Step 1: Launch in the background**

```bash
cd /home/abhinav/codes/apple_pick_sim/.claude/worktrees/rl-skrl-ppo
nohup bash runs/vic_harvest/analysis_v2f/run_probes_cur.sh > runs/vic_harvest/analysis_v2f/probe_cur/run.log 2>&1 &
```

Expected: about 20-30 min. The log ends with `PROBE ROLLOUTS DONE`, and each line before it is a `saved ...` line.

- [ ] **Step 2: Verify**

```bash
for f in runs/vic_harvest/analysis_v2f/probe_cur/*_holdout.npz; do uv run python runs/vic_harvest/analysis_v2f/check_current_chord.py $f | tail -1; done
```

Expected: `OK` ×4. Also check that the holdout success rate printed in each `.log` matches today's `probe/` logs for the same
pair (the rollout is deterministic, and logging must not change behaviour).

### Task 3: Current-direction probes

**Files:**
- Create: `runs/vic_harvest/analysis_v2f/probe_cur_v2f.py`

**Interfaces:**
- Consumes: `features(z, si, t)`, `r2`, `rot` from `analysis_r7/probe_fit.py`, and `mlp_pred` (copy it from
  `probe_v2f.py:65`).
- Produces: `probe_cur_v2f_{run}_{set}.json` with:
  - `decode[target][t][fs]`
  - `behaviour[t][label][group]`
  - `align[t] = {cos_mean, cos_shuffled, n}`

Targets at step t, for the observation at step t, which reports the state after step t-1. All are in the *current* tool
frame, `qt = tcp_quat[:, t-1]`:
- `cur chord x/y/z`: `stem_chord_world[:, t-1]` rotated by `conj(qt)`.
- `bend vec x/y/z`: the current chord minus the rest chord carried with the spur tip (Task 1's `carried`), in the current
  tool frame. This is the stem-only bend displacement.
- `bend axis x/y/z`: `cross(carried, current)` in the current tool frame (magnitude ≈ sin θ_b).
- Control: `theta_b` = `theta[:, t-1]` (scalar magnitude), so that "rotates more when bent more" can be told apart from
  directional tracking.

- [ ] **Step 1: Write the script**

```python
"""[v2f] Does the policy track the stem's CURRENT direction, and does its rotation follow it?
Fit on probe_cur/<run>_<set>_train.npz, test on .._holdout.npz. The observation at step t reports the state after step t-1,
so targets use index t-1; act[:, t] is chosen from obs[:, t]. Frames: current tool frame (action_frame "tool").
1. decode: ridge (+ MLP for lstm / obs_win / grasp) R^2 of the current chord, the stem-only bend vector and the bend axis
   from lstm / random / obs_now / obs_win / grasp.
2. behaviour: R^2 of the action groups from grasp, grasp+theta_b, grasp+theta_b+current direction (chord, bend vec, axis).
3. align: cosine between the commanded rotation vector (act 3:6) and -bend axis (rotate to unbend), rows with theta_b >= 2 deg,
   vs the same with bend axes shuffled across worlds (baseline ~0).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent / "analysis_r7"))
from probe_fit import features, r2, rot  # noqa: E402

STEPS = [4, 10, 24, 40, 60, 90]
FS = ["lstm", "random", "obs_now", "obs_win", "grasp"]
MLP_FS = ["lstm", "obs_win", "grasp"]
ACT = {"translate": slice(0, 3), "rotate": slice(3, 6), "K linear": slice(6, 9), "K angular": slice(9, 12), "damping": slice(12, 13)}


def conj(q):
    q = q.copy(); q[..., :3] *= -1
    return q


def mlp_pred(xa, ya, xb):
    mu, sd = ya.mean(0), ya.std(0) + 1e-12
    m = MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=300, early_stopping=True,
                     n_iter_no_change=15, random_state=0).fit(xa, (ya - mu) / sd)
    return m.predict(xb).reshape(-1, *ya.shape[1:]) * sd + mu


def cur_targets(z, t):
    """Targets for the observation at step t (state after step t-1), in the current tool frame."""
    k = t - 1
    rest = z["rest_chord_world"] / np.linalg.norm(z["rest_chord_world"], axis=1, keepdims=True)
    rest_spur = rot(conj(z["spur_tip_quat0"]), rest)
    carried = rot(z["spur_tip_quat"][:, k], rest_spur)
    cur = z["stem_chord_world"][:, k]
    qi = conj(z["tcp_quat"][:, k])
    out = {}
    for name, v in (("cur chord", cur), ("bend vec", cur - carried), ("bend axis", np.cross(carried, cur))):
        vt = rot(qi, v)
        for i, ax in enumerate("xyz"):
            out[f"{name} {ax}"] = vt[:, i]
    out["theta_b"] = z["theta"][:, k]
    return out


def probe(run, wset):
    A, B = np.load(H / f"probe_cur/{run}_{wset}_train.npz"), np.load(H / f"probe_cur/{run}_{wset}_holdout.npz")
    steps = list(A["steps"])
    res = {"decode": {}, "behaviour": {}, "align": {}, "n": {}}
    rng = np.random.default_rng(0)
    for t in STEPS:
        si = steps.index(t)
        ma, mb = A["live"][:, t] & ~A["invalid"], B["live"][:, t] & ~B["invalid"]
        res["n"][t] = [int(ma.sum()), int(mb.sum())]
        TA, TB = cur_targets(A, t), cur_targets(B, t)
        keys = list(TA)
        Ya = np.column_stack([TA[k][ma] for k in keys]); Yb = np.column_stack([TB[k][mb] for k in keys])
        FA, FB = features(A, si, t), features(B, si, t)
        for fs in FS:
            sc = StandardScaler().fit(FA[fs][ma])
            xa, xb = sc.transform(FA[fs][ma]), sc.transform(FB[fs][mb])
            pr = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(xa, Ya).predict(xb)
            pm = mlp_pred(xa, Ya, xb) if fs in MLP_FS else None
            for j, k in enumerate(keys):
                row = res["decode"].setdefault(k, {}).setdefault(t, {})
                row[f"ridge:{fs}"] = round(float(r2(Yb[:, j], pr[:, j])), 3)
                if pm is not None:
                    row[f"mlp:{fs}"] = round(float(r2(Yb[:, j], pm[:, j])), 3)
        # behaviour: does the action depend on the current direction beyond grasp + bend magnitude?
        dirk = [k for k in keys if k != "theta_b"]
        g_a, g_b = FA["grasp"][ma], FB["grasp"][mb]
        th_a, th_b = TA["theta_b"][ma, None], TB["theta_b"][mb, None]
        d_a = np.column_stack([TA[k][ma] for k in dirk]); d_b = np.column_stack([TB[k][mb] for k in dirk])
        act_a, act_b = A["act"][ma, t].astype(np.float32), B["act"][mb, t].astype(np.float32)
        beh = {}
        for label, xa_raw, xb_raw in (("grasp", g_a, g_b), ("grasp+theta", np.hstack([g_a, th_a]), np.hstack([g_b, th_b])),
                                      ("grasp+theta+cur dir", np.hstack([g_a, th_a, d_a]), np.hstack([g_b, th_b, d_b]))):
            sc = StandardScaler().fit(xa_raw)
            p = mlp_pred(sc.transform(xa_raw), act_a, sc.transform(xb_raw))
            per = [float(r2(act_b[:, j], p[:, j])) if act_b[:, j].var() > 1e-10 else float("nan") for j in range(act_b.shape[1])]
            beh[label] = {g: round(float(np.nanmean(per[s])), 3) for g, s in ACT.items()}
        res["behaviour"][t] = beh
        # align: commanded rotation vs "rotate to unbend" (-bend axis), holdout rows bent >= 2 deg
        ax = np.column_stack([TB[f"bend axis {a}"] for a in "xyz"])[mb]
        rv = act_b[:, 3:6]
        keep = (TB["theta_b"][mb] >= 2.0) & (np.linalg.norm(rv, axis=1) > 1e-6)
        u = -ax[keep] / np.linalg.norm(ax[keep], axis=1, keepdims=True)
        w = rv[keep] / np.linalg.norm(rv[keep], axis=1, keepdims=True)
        cos = (u * w).sum(1)
        cos_sh = (u[rng.permutation(len(u))] * w).sum(1)
        res["align"][t] = {"cos_mean": round(float(cos.mean()), 3), "cos_median": round(float(np.median(cos)), 3),
                           "cos_shuffled": round(float(cos_sh.mean()), 3), "n": int(keep.sum())}
        print(f"[{run}@{wset}] t={t:3d} n={ma.sum()}/{mb.sum()} align {res['align'][t]}", flush=True)
    json.dump({f"{run}@{wset}": res}, open(H / f"probe_cur_v2f_{run}_{wset}.json", "w"), indent=1)
    return res


def show(key, res):
    print(f"\n===== {key}: decode R^2 (ridge lstm / random / obs_now / obs_win / grasp | mlp lstm / obs_win / grasp)")
    for k, rows in res["decode"].items():
        print(f"  {k}")
        for t, row in rows.items():
            print(f"     t={int(t):3d} " + " ".join(f"{row[f'ridge:{f}']:+.2f}" for f in FS)
                  + " | " + " ".join(f"{row.get(f'mlp:{f}', float('nan')):+.2f}" for f in MLP_FS))
    print("  -- behaviour: action R^2 (translate / rotate / K lin / K ang / damping)")
    for t, beh in res["behaviour"].items():
        for label, g in beh.items():
            print(f"     t={int(t):3d} {label:22s} " + " ".join(f"{v:+.2f}" for v in g.values()))


if __name__ == "__main__":
    pairs = [a.split("@") for a in sys.argv[1:]] or [("widetrue", "widetrue"), ("true", "widetrue"), ("true", "true"), ("widetrue", "true")]
    for run, wset in pairs:
        show(f"{run}@{wset}", probe(run, wset))
```

- [ ] **Step 2: Check the index convention on one pair**

Run: `uv run python runs/vic_harvest/analysis_v2f/probe_cur_v2f.py widetrue@widetrue 2>&1 | tee runs/vic_harvest/analysis_v2f/probe_cur_v2f_widetrue_widetrue.log`

Expected positive control: the `theta_b` decode from `obs_now` at t=24 is close to D17p's `theta_probe_v2e` (about
0.15-0.4). If it is ≈ 0 while the cur-chord targets are high, the t-1 indexing is wrong. Re-check it against Task 1's
finding on `theta` timing.

- [ ] **Step 3: Run the other three pairs (in parallel, CPU)**

```bash
cd /home/abhinav/codes/apple_pick_sim/.claude/worktrees/rl-skrl-ppo
for p in true@widetrue true@true widetrue@true; do
  uv run python runs/vic_harvest/analysis_v2f/probe_cur_v2f.py $p > runs/vic_harvest/analysis_v2f/probe_cur_v2f_${p/@/_}.log 2>&1 &
done; wait
```

Expected: 3 JSON files, plus logs that end with the behaviour tables.

### Task 4: Read-out and report

**Files:**
- Read: the 4 `probe_cur_v2f_*.json` / `.log` files.

- [ ] **Step 1: Answer three questions per pair, with numbers**
  1. *Is the current direction readable?* Compare `cur chord` and `bend vec` from `obs_now` / `obs_win` against `grasp`.
     A gain over grasp means the current direction is visible in what the policy sees (the force direction).
  2. *Does the policy hold it beyond its inputs?* `lstm` > `obs_win` and > `random` → learned tracking; `lstm` ≈ `random`
     → passive.
  3. *Does the rotation follow it?* Look at the behaviour gain of `grasp+theta+cur dir` over `grasp+theta` for `rotate`,
     and at `align.cos_mean` vs `cos_shuffled`. Clearly positive → it rotates to unbend; ≈ shuffled → no directional
     response.
- [ ] **Step 2: Report in chat.** Use one compact table per question (`v2f_widetrue@wide_true` in full, and the other pairs
  only where they differ), then a plain conclusion. If the rotation does follow the bend, say which steps. If it does not,
  recall the two options already raised: an oracle observation of the direction, or a grasp roll/azimuth independent of
  the stem.
- [ ] **Step 3:** Do not commit `runs/` (gitignored). Add a decision-log entry only if the maintainer asks.

## Cost

- GPU: about 20-30 min for 8 rollouts (4 in parallel), plus about 1 min for the smoke test.
- CPU: about 10-20 min for the probes (MLPs on 5000×512 features at 6 steps × 5 feature sets).
