# Sys-ID transition features (MMD / Wasserstein)

**Purpose:** document the observation state vector and transition-bag layout used by
biased MMD and Sinkhorn (Wasserstein) scoring on hold-phase quasi-static data.

**Code owners:**

| Role | Module |
| ---- | ------ |
| State + transition builders | `apple_pick_sim/system_id/mmd_features.py` |
| GT z-score + MMD² | `apple_pick_sim/system_id/mmd.py` |
| Sinkhorn bags + pooling | `apple_pick_sim/system_id/wasserstein.py` |
| CLI flags | `apple_pick_gym/batched_examples/example_batched_sysid_mmd_grid.py` |
| Named gates | `scripts/gate_sysid_gt_sinkhorn.sh` |

Related: `docs/system_identification.md` §3, `docs/specs/2026-07-08-wasserstein-sysid-ranking-design.md`,
`docs/sysid-mmd-grid-replay-alignment.md`.

## Document status

| Field | Value |
| ----- | ----- |
| **Last reviewed** | 2026-07-15 |
| **Scope** | Features as implemented for hold-phase MMD/Wasserstein (not chirp / full-traj damping bags) |

---

## 1. Per-frame state \(s_t\)

Built by `build_state_matrix` from `STATE_VECTOR_FIELDS` in `mmd_features.py`.
Columns are concatenated in this order:

| Field | Dim | Source / notes |
| ----- | --- | -------------- |
| `ft_wrist` | 6 | Wrist wrench (force 3 + torque 3), robot-facing |
| `tcp_velocity` | 6 | TCP linear (3) + angular (3) velocity |
| `action` | 6 | Recorded EE velocity command (replay drive signal) |
| `tcp_pos` | 3 | TCP position |
| `apple_pos` | 3 | Apple / fruit body position |
| `woody_part_start_pos` | \(3 N_j\) | Junction endpoint starts, flattened in `junction_names` order |
| `woody_part_end_pos` | \(3 N_j\) | Junction endpoint ends, same order |
| `woody_bending_angles` | \(N_j\) | Per-junction chord deflection (rad) from **frame-0 rest** |

\(N_j = \texttt{len(junction\_names)}\).

**Total state dim**

\[
D_s = 6+6+6+3+3 + 3N_j + 3N_j + N_j = 24 + 7 N_j
\]

### Woody geometry and bending angles

- Endpoints come from Sim observations (or real markers at rod start/end).
- For junction \(j\), chord direction at frame \(t\): \(\hat d_t = (p_{\mathrm{end}}-p_{\mathrm{start}}) / \|\cdot\|\).
- `woody_bending_angles[j,t] = \arccos(\mathrm{clip}(\hat d_t \cdot \hat d_0, -1, 1))`, with frame 0 forced to 0.
- Full ArUco marker orientations (twist about the chord) are **not** in the bag today; endpoint positions + angles are sufficient for bend stiffness ranking.

### Auxiliary arrays (not columns of \(s\), but required for bagging)

| Key | Role |
| --- | ---- |
| `phase` | Trajectory phase; hold scoring keeps `phase == 1` |
| `dir_idx` | Integer excitation direction identity |
| `excitation_direction` | 3D unit vector (recorded; bags key by `dir_idx`) |
| `excitation_type` | Trajectory family id |
| `junction_names` | Ordered woody labels |
| `stable` | Optional bool mask; unstable frames excluded from hold samples |
| `hold_number` | Optional 0-based hold index within a direction (`-1` if unknown) |

Phase integers (`trajectory_store.PHASE_TO_INT`): e.g. `hold = 1`. Current MMD/Wasserstein builders only keep **hold** frames for bags.

---

## 2. Transition row \(v\)

Markovian transition encoding (same for MMD and Sinkhorn):

\[
v = [s,\ \Delta s]
\]

so each sample has feature width \(D_v = 2 D_s\) before optional one-hots.

### Modes (`use_median` / CLI `--use-median`)

| Mode | How rows are built | Typical use |
| ---- | ------------------ | ----------- |
| **Frame→frame** (`use_median=False`) | On each contiguous **full** hold segment, after an in-hold `stable` mask: consecutive **kept** indices \(i\to j\): \(\Delta s = s_j-s_i\) (may skip unstable frames; not always adjacent times) | Dense bags; more samples for Sinkhorn |
| **Hold→hold median** (`use_median=True`; **CLI default**) | One median state per full hold over stable frames; consecutive keepable holds \(i \to i+1\): \(\Delta s = s_{i+1}^{\mathrm{med}}-s_i^{\mathrm{med}}\) | Stiffer staircase / equilibrium steps; `gate_median_hold` |

Python helpers (`build_transition_features_by_direction`, `prepare_gt_wasserstein_context`, …) default `use_median=False`; the batched grid CLI defaults `--use-median` **on**.

Hold segments come from `iter_kept_hold_segments`: contiguous runs with `phase == 1` and matching `dir_idx` only. `stable=False` does **not** split segments; `_stable_masked_segment` drops unstable frames when aggregating medians / frame→frame rows. There is **no** latter-half burn-in in these builders (older design notes / hold *diagnostics* may still mention burn-in — see Ambiguities).

Bags are produced per direction by `build_transition_features_by_direction` / `combine_transition_features`:

```text
dict[dir_idx, ndarray (N_transitions, D_v + optional one-hots)]
```

---

## 3. Optional one-hot extras

Appended **after** \([s, \Delta s]\) when enabled:

| Flag / kwargs | When | Width | Purpose |
| ------------- | ---- | ----- | ------- |
| `--hold-id-onehot` / `hold_id_onehot` | CLI default on | \(N_{\mathrm{holds}}\) | Tags which amplitude step the row’s **source** hold belongs to |
| `--pool-directions` / `pool_directions` | CLI default on (gate_pooled_dirs); forces `dir_id_onehot=True` in `wasserstein.py` | \(N_{\mathrm{dirs}}\) | Tags excitation direction after pooling |

The lower-level builder accepts `dir_id_onehot` without pooling; Sinkhorn prepare/score and the CLI only turn it on when `pool_directions` is set.

Pooled Sinkhorn path (`wasserstein.py`):

1. Build per-direction bags (with dir one-hot when pooling).
2. Concatenate all directions into one bag keyed by `POOLED_DIRECTION_KEY = -1`.
3. Fit one GT z-score on the pooled bag; score candidates with Sinkhorn.

Full pooled row (with both extras, as in `gate_pooled_dirs`):

\[
v = [s,\ \Delta s,\ \mathrm{hold\_id\ one\text{-}hot},\ \mathrm{dir\_idx\ one\text{-}hot}]
\]

### Batched grid CLI notes

- Defaults: `--use-median`, `--hold-id-onehot`, and `--pool-directions` on (disable with `--no-*`).
- Deprecated: `--mse-hold-aggregation {mean,median,none}` aliases median on/off; `--mse-hold-latter-half` is a **no-op** (full holds always).

### Named Sinkhorn gates

From `scripts/gate_sysid_gt_sinkhorn.sh` (`SCORE_EXTRA`). Script default
`GATE=gate_pooled_dirs` (same as the batched grid CLI defaults); override with
`--gate …`.

| Gate | Flags |
| ---- | ----- |
| `gate_median_hold` | `--use-median --no-hold-id-onehot --no-pool-directions` |
| `gate_hold_id` | `--use-median --hold-id-onehot --no-pool-directions` |
| `gate_pooled_dirs` (**script default**) | `--use-median --hold-id-onehot --pool-directions` |

---

## 4. Pre-processing before distance

1. **Hold-only** transitions as above (per direction, then optional pool).
2. **GT z-score** per bag: `fit_gt_normalization` / `apply_normalization` (`mmd.py`) — mean/std from GT features for that direction (or pooled bag); clamp tiny std to `eps`.
3. **Sinkhorn:** Geomloss `SamplesLoss("sinkhorn", p=2, blur=1.0)` on normalized bags (`wasserstein.py`).
4. **MMD (library):** biased MMD² with RBF bandwidth from GT median pairwise distance (`rbf_bandwidth_median`).

Aggregate Sinkhorn across directions is a **transition-count weighted** mean of per-direction distances (when not pooled into a single bag).

---

## 5. Role of each feature (stiffness vs damping)

Current shipped bags are built for **§2.1 quasi-static hold** scoring of **bend stiffness**. Damping identification needs move / chirp dynamics (full trajectory); that is not the default bag filter today.

| Feature | Stiffness (\(K\) / \(E\)) on holds | Damping (\(B\) / \(\zeta\)) if extended to full traj |
| ------- | ----------------------------------- | ----------------------------------------------------- |
| `ft_wrist` | Critical — equilibrium \(F \approx Kx\) | High — peaks / lag / decay during move→settle |
| `tcp_pos`, `apple_pos` | Critical — deflection | High — path, overshoot, settling |
| Woody start/end + bending angles | High (sim / marked rods) — segment shape | High — time series of chord deflection |
| `tcp_velocity` | Low on settled holds | Critical on move / early hold ring-down |
| `action` | Low–medium (shared under aligned replay) | Medium–high — same command, different response |
| `hold_id` one-hot | Medium — keeps amplitude steps distinct | Lower priority if phase/context labeled elsewhere |
| `dir_idx` one-hot (pooling) | Medium–high — preserves anisotropic \(K(\hat u)\) | Medium — avoid mixing directions in one OT bag |
| Hold medians / hold-only \(\Delta s\) | Strong for \(K(x)\) staircase | Weak for \(B\) — prefer frame \(\Delta s\) on move |

Real-world ArUco at woody start/end maps cleanly onto `woody_part_*_pos`; marker **orientation** (torsion) is optional and not required for the current feature set.

---

## 6. Ambiguities (code as written)

- **Hold diagnostics vs bags:** `batched_hold_quasi_static.py` still defaults to a latter-half metrics window for quasi-static *checks*. That path is separate from MMD/Wasserstein feature builders, which always use full hold segments.
- **\(N_{\mathrm{holds}}\) / \(N_{\mathrm{dirs}}\):** widths resolve from `n_holds` / `n_directions` when passed, else from `hold_number` max+1 / `dir_idx` max+1 (or per-direction segment count). Callers that pool must keep GT and candidate widths aligned.

---

## 7. Tests

| Tests | What they cover |
| ----- | --------------- |
| `apple_pick_sim/tests/test_mmd_features.py` | State layout, woody flatten order, bending angles, hold segments (no split on `stable`), median / one-hot transitions, no latter-half burn-in |
| `apple_pick_sim/tests/test_wasserstein.py` | Per-dir vs pooled context, Sinkhorn identical≈0, pooling + dir one-hot (`POOLED_DIRECTION_KEY`) |
| `apple_pick_sim/tests/test_mmd.py` | Normalization + biased MMD² |

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_mmd_features.py \
  apple_pick_sim/tests/test_wasserstein.py \
  apple_pick_sim/tests/test_mmd.py -q
```
