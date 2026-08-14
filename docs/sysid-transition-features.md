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
| Young's-modulus complete scoring | `apple_pick_gym/batched_envs/batched_sysid_cmaes.py` |
| Named gates | `scripts/gate_sysid_gt_sinkhorn.sh` |
| Young's multi-seed gate | `scripts/gate_youngs_modulus_sysid.sh` |

Related: `docs/system_identification.md` §3,
`docs/sysid-mmd-grid-replay-alignment.md`, and
`docs/youngs-modulus-sysid.md`.

## Document status

| Field | Value |
| ----- | ----- |
| **Last reviewed** | 2026-07-17 |
| **Scope** | Features as implemented for hold-phase MMD/Wasserstein (not chirp / full-traj damping bags) |

---

## 1. Per-frame state \(s_t\)

Built by `build_state_matrix` from `STATE_VECTOR_FIELDS` in `mmd_features.py`.
Columns are concatenated in this order:

| Field | Dim | Source / notes |
| ----- | --- | -------------- |
| `ft_wrist` | 6 | Wrist wrench (force 3 + torque 3), robot-facing |
| `tcp_velocity` | 6 | TCP linear (3) + angular (3) velocity |
| `tcp_pos` | 3 | TCP position |
| `apple_pos` | 3 | Apple / fruit body position |
| `woody_part_start_pos` | \(3 N_j\) | Junction endpoint starts, flattened in `junction_names` order |
| `woody_part_end_pos` | \(3 N_j\) | Junction endpoint ends, same order |
| `woody_bending_angles` | \(N_j\) | Per-junction chord deflection (rad) from **frame-0 rest** |

\(N_j = \texttt{len(junction\_names)}\).

**Total state dim**

\[
D_s = 6+6+3+3 + 3N_j + 3N_j + N_j = 18 + 7 N_j
\]

`action` is still required in episode bags (`REQUIRED_ARRAY_KEYS`) for replay
drive and collector validation, but it is **not** concatenated into score-time
`STATE_VECTOR` (see
`docs/superpowers/specs/2026-08-14-sinkhorn-fixed-scale-normalization-design.md`).

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
| `action` | Full replay command (e.g. 19D `vic_pose`); required in bags, excluded from `STATE_VECTOR` |
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
3. Fit one GT mean + fixed physical scale on the pooled bag; score candidates with Sinkhorn.

### Complete candidate scoring

`prepare_gt_wasserstein_scoring_context()` retains both a pooled context and
independently normalized physical-direction contexts.
`score_candidate_wasserstein_complete()` verifies that every expected physical
direction has a usable bag before producing pooled fitness. Missing or empty
bags return candidate-local invalid results rather than aborting a structure.

The pooled aggregate and per-direction diagnostics are intentionally different
quantities:

- pooled fitness uses one GT normalization fitted after concatenation and
  drives ranking/optimization;
- each physical-direction diagnostic uses its own GT normalization; and
- `POOLED_DIRECTION_KEY` is internal and must not appear in serialized
  `per_direction_sinkhorn`.

Non-finite internal scores serialize as JSON `null`. The exact empty-bag
disqualified shape may use a null aggregate, an empty per-direction map, and a
null rank. See `docs/youngs-modulus-sysid.md` for the report contract.

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
2. **GT mean + fixed physical scale** per bag: `fit_gt_normalization` / `apply_normalization` (`mmd.py`) — mean from GT features for that direction (or pooled bag); divide by fixed `STATE_VECTOR_PHYS_SCALE` (not GT std). See `docs/superpowers/specs/2026-08-14-sinkhorn-fixed-scale-normalization-design.md`. Hold/dir one-hots use scale 1 and are not mean-centered.
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
| `action` | Not in score-time `STATE_VECTOR` (replay drive only) | Medium–high — same command, different response |
| `hold_id` one-hot | Medium — keeps amplitude steps distinct | Lower priority if phase/context labeled elsewhere |
| `dir_idx` one-hot (pooling) | Medium–high — preserves anisotropic \(K(\hat u)\) | Medium — avoid mixing directions in one OT bag |
| Hold medians / hold-only \(\Delta s\) | Strong for \(K(x)\) staircase | Weak for \(B\) — prefer frame \(\Delta s\) on move |

Real-world ArUco at woody start/end maps cleanly onto `woody_part_*_pos`; marker **orientation** (torsion) is optional and not required for the current feature set.

---

## 6. Ranking expectation

Under healthy excitation / sampling, GT should rank first on the shipped hold
bags (median + hold-id + pooled directions). Occasional worse ranks from bad
sampling are allowed and should trigger fixture/cap/seed diagnosis rather than
an automatic feature change. The Young's-modulus operational gate uses a
strict majority per seed (three of five by default), with every configured seed
required to pass.

## 7. Ambiguities (code as written)

- **Hold diagnostics vs bags:** `batched_hold_quasi_static.py` still defaults to a latter-half metrics window for quasi-static *checks*. That path is separate from MMD/Wasserstein feature builders, which always use full hold segments.
- **\(N_{\mathrm{holds}}\) / \(N_{\mathrm{dirs}}\):** widths resolve from `n_holds` / `n_directions` when passed, else from `hold_number` max+1 / `dir_idx` max+1 (or per-direction segment count). Callers that pool must keep GT and candidate widths aligned.

---

## 8. Tests

| Tests | What they cover |
| ----- | --------------- |
| `apple_pick_sim/tests/test_mmd_features.py` | State layout, woody flatten order, bending angles, hold segments (no split on `stable`), median / one-hot transitions, no latter-half burn-in |
| `apple_pick_sim/tests/test_wasserstein.py` | Per-dir vs pooled context, completeness, sparse physical IDs, singleton bags, pooling + dir one-hot |
| `apple_pick_sim/tests/test_mmd.py` | Normalization + biased MMD² |
| `apple_pick_gym/tests/test_batched_sysid_youngs_grid.py` | Complete scorer integration and candidate-local disqualification |
| `apple_pick_gym/tests/test_youngs_modulus_gate_report.py` | Strict-JSON GT evidence and strict-majority gate |

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_mmd_features.py \
  apple_pick_sim/tests/test_wasserstein.py \
  apple_pick_sim/tests/test_mmd.py -q
```
