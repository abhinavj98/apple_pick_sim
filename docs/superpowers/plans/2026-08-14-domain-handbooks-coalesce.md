# Domain Handbooks Coalesce (Approach B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the scattered living-doc + dated-spec pile with five code-truth domain handbooks that agents can trust, each absorbing shipped content from related `docs/superpowers/specs/` (and legacy `docs/specs/`) with explicit archive links.

**Architecture:** Keep `docs/VISION.md` / `docs/ROADMAP.md` as intent+status. Rewrite (or heavily expand) five handbooks under `docs/`. Demote superseded living docs to short stubs that point at the handbook. Stamp every absorbed superpowers spec `Implemented` / `Superseded` / `Historical` and add a one-line “canonical living doc” pointer. Refresh `docs/CODEBASE_GUIDE.md` as the map into the five handbooks.

**Tech Stack:** Markdown docs only; verification = claim↔code grep + existing `uv run` validation commands from ROADMAP (no new runtime features).

## Global Constraints

- **Code is source of truth** — every contract claim must cite a module/symbol (or test name); if code and an old living doc disagree, rewrite the doc.
- **ROADMAP owns status** — handbooks must not invent slice IDs or “Done/Next” tables; link to `docs/ROADMAP.md` for sequencing.
- **Do not delete** dated specs/plans in the first pass — stamp + point, then optionally move to `docs/archive/` in a later cleanup task.
- **Preserve runnable commands** — every handbook “How to verify” section must use `uv run …` matching README/ROADMAP.
- **Layer vocabulary** must stay consistent across handbooks:
  - **Runtime obs** — gym / scene observation dicts (may still include `woody_part_end_pos`).
  - **Bag bag** — `batched_sysid_v1` parquet arrays (trajectory frames: no `woody_end`; `action` 6D or 19D).
  - **Score vector** — `STATE_VECTOR_FIELDS` / Sinkhorn features (`action` excluded; fixed `STATE_VECTOR_PHYS_SCALE`).
- **Current M4.0 contracts** (must appear in handbooks 3–5, not only in specs):
  - `vic_pose` / 19D `vic_pose_v1` packing
  - Convert-time `R(tcp) @` F/T; **no** sim EMA/LPF
  - Two CMA woody starts + `apple_pos`; drop `woody_end` from sys-ID bags / `STATE_VECTOR`
  - Scalar `hold_number` from `hold_index`; one-hot only at score time
  - `action` required in bags, **not** in `STATE_VECTOR`
  - GT mean + fixed physical scale (`STATE_VECTOR_PHYS_SCALE`), not GT-std z-score
- **Hard geometry / stability constants** (code wins; several specs/docs wrong — fix in Task 0 + H1/H2):
  - Stem wrench caps: `DEFAULT_STEM_FORCE_CAP_N = 40.0`, `DEFAULT_STEM_TORQUE_CAP_NM = 10.0` in `coupled_fruiting/scene.py` (stability monitor aliases these). **Not** 100/40 (retune spec) or 100/100 (gym-obs / sys-ID living docs / ROADMAP).
  - EE tool length: `EE_CYLINDER_HALF_HEIGHT = 0.09` → **180 mm** total (`fr3_robot/paths.py`). **Not** 140 mm / `hh=0.07` from `2026-08-04-tcp-tip-flange-geometry-design.md`.

---

## Target handbook set (5)

| # | New / primary path | Absorbs (living) | Primary code owners |
| - | ------------------ | ---------------- | ------------------- |
| H1 | `docs/handbook-coupled-simulation.md` | `handbook-coupled-simulation.md`, `handbook-coupled-simulation.md`, `handbook-coupled-simulation.md`, weld/payload/wrench satellites as sections or deep-links | `coupled_fruiting/`, `fruiting_system/`, examples |
| H2 | `docs/handbook-variable-impedance.md` | `handbook-variable-impedance.md` (+ wrench-cap content) | `vic_joint_torques*`, `ee_impedance*`, `vic_wrench.py` |
| H3 | `docs/handbook-sysid-scoring.md` | `handbook-sysid-scoring.md`, `handbook-sysid-scoring.md`, parts of `handbook-sysid-scoring.md`, `handbook-sysid-scoring.md` (legacy callout) | `system_id/mmd_features.py`, `mmd.py`, `wasserstein.py`, trajectory stores |
| H4 | `docs/handbook-real-replay.md` | `robot_replay/README.md` (keep thin CLI), `handbook-real-replay.md`, digital-twin replay init pieces | `robot_replay/`, `real_to_batched_sysid.py`, `batched_digital_twin_init.py`, `real_batched_replay_build.py` |
| H5 | `docs/handbook-youngs-cma.md` | `handbook-youngs-cma.md`, `handbook-youngs-cma.md`, support-\(k_p\) phenotype notes | `batched_sysid_cmaes.py`, Young’s examples, gates |

**Index:** `docs/CODEBASE_GUIDE.md` + optional short `docs/FEATURES.md` TOC that only lists the five handbooks (no duplicate contracts).

```text
VISION / ROADMAP
        │
        ▼
CODEBASE_GUIDE / FEATURES  ──►  H1 Coupled sim
                             ├►  H2 VIC / vic_pose
                             ├►  H3 Sys-ID scoring & bags
                             ├►  H4 Real replay pipeline
                             └►  H5 Young's / CMA phenotype
                                        │
                                        ▼
                         docs/superpowers/specs|plans  (archive)
```

---

## Cross-link matrix (must appear in each handbook header)

Every handbook opens with:

```markdown
## Document status
| Field | Value |
| Last reviewed | YYYY-MM-DD |
| Code owners | `path/…` |
| Status | Living handbook — defer sequencing to `docs/ROADMAP.md` |
| Related handbooks | H… |
| Archive specs | bullet list with status stamps |
```

| From → To | Why |
| --------- | --- |
| H1 → H2 | VIC runs inside coupled substep after weld |
| H1 → H3 | Batched collect/replay produces bags scored in H3 |
| H2 → H4 | Real drive uses `vic_pose` |
| H2 → H5 | Grid/CMA opt into `vic_pose` from dataset metadata |
| H3 → H4 | Real convert must emit H3 bag/score contract |
| H3 → H5 | Grid/CMA Sinkhorn uses H3 features + scales |
| H4 → H1 | Rebuild/settle/weld uses coupled builders |
| H4 → H5 | Shared `make_real_replay_build_env_fn` feeds grid/CMA |
| H5 → H3 | Phenotype search scores H3 bags |

---

## H1 — Coupled simulation handbook

**Create:** `docs/handbook-coupled-simulation.md`  
**Stub afterward:** `handbook-coupled-simulation.md`, `handbook-coupled-simulation.md`, `handbook-coupled-simulation.md` (keep filenames as redirects), optionally leave deep satellites (`WRENCH_READOUT.md`, `handbook-coupled-simulation.md`, `explicit-apple-load-tcp-harvest.md`, `gpu-coupling-optimization.md`, `damping-tuning.md`, `material-parameter-sampling.md`, `heterogeneous-batched-vectorization-audit.md`) as linked appendices — **do not** duplicate their full math unless a claim is wrong.

### TOC (required sections)

1. **Purpose & non-goals** — two-`Model` split reason (`SolverMuJoCo` vs `JointType.CABLE`).
2. **Ownership table** — Model A (arm) vs Model B (plant); proxy bodies; wrench exchange lag.
3. **Public API** — builders, `CoupledFruitingScene.coupled_substep`, FR3 requirements (absorb `handbook-coupled-simulation.md`).
4. **Settle → weld → teleop** — single-env and batched; settle cache default **off**; co-located physics vs viewer spacing.
5. **Homogeneous vs heterogeneous batches** — `replicate(N)`, per-env DR, layout.
6. **Wrench / payload / TCP harvest** — short summary + links to satellite docs; F/T sign contract as implemented.
7. **TCP / flange / post-grasp weld geometry** — absorb *implemented outcomes* from:
   - `2026-08-04-true-tcp-pose-weld-design.md`
   - `2026-08-04-tcp-tip-flange-geometry-design.md` — **geometry table superseded**: length **180 mm** (`hh=0.09`), not 140 mm; stamp Superseded by alignment slice-0 / code
   - `2026-08-07-pre-grasp-apple-orientation-design.md`
   - `2026-08-05-apple-position-only-post-grasp-weld-design.md` (superseded open issue note)
   - EE mass/COM/`I_ee` from alignment slice 0 (`2026-08-13-…`)
8. **GPU hot path** — pointer to `gpu-coupling-optimization.md` + batched audit; defaults.
9. **Code map** — modules/symbols.
10. **Tests & verify** — pytest modules + canonical example:
    `uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py …`

### Archive specs to stamp + link

| Spec | Stamp |
| ---- | ----- |
| `2026-07-03-batched-heterogeneous-*-design.md` (build/runtime/example) | Historical / Implemented |
| `2026-07-03-batched-gpu-hot-path-design.md` | Historical / Implemented |
| `2026-07-03-v32-close-out-design.md` | Historical / Implemented |
| `2026-07-03-pre-gym-scope-narrowing-design.md` | Historical / Implemented |
| `2026-07-04-batched-gym-base-env-design.md` | Historical / Implemented (gym pieces → also H5/H3 as needed) |
| Weld/TCP Aug specs above | Implemented (or Superseded where noted) |

### Related living docs (link, don’t merge wholesale)

- `docs/WRENCH_READOUT.md`, `docs/handbook-coupled-simulation.md`, `docs/explicit-apple-load-tcp-harvest.md`
- `docs/damping-tuning.md`, `docs/material-parameter-sampling.md`, `docs/real-world-proxy.md`
- `docs/gpu-coupling-optimization.md`, `docs/heterogeneous-batched-vectorization-audit.md`
- → H2 for controller details

---

## H2 — Variable impedance handbook

**Create:** `docs/handbook-variable-impedance.md` (rewrite of `handbook-variable-impedance.md` content into handbook form; leave old path as stub).

### TOC

1. **Total TCP wrench / joint-torque control** — kinematic vs dynamic gym envs.
2. **Twist mode (`vic`)** — 6D action, gains from fixture/`sim_build`, batched velocity buffers.
3. **Pose mode (`vic_pose`)** — 19D layout, anisotropic wrench kernel, soft-disable freeze semantics (absorb `2026-08-10-vic-pose-action-controller-design.md`).
4. **Wrench caps** — document **code** values 40 N / 10 N·m; stamp `2026-07-17-wrench-cap-retune-design.md` **Superseded by undocumented retunes** (do not copy 100/40). Correct any living-doc 100/100 claims when stubbing. Optional short “why” note in `damping-tuning.md` if commit history is the only rationale.
5. **Who uses which mode** — gym collect/MMD/default CMA = twist; real replay/grid opt-in = `vic_pose`.
6. **Code map + tests** — cite `test_ee_impedance_batched_pose_actions.py` (not the missing `test_vic_pose_actions.py`), soft-disable tests, ROADMAP verify commands.

### Archive specs

| Spec | Stamp |
| ---- | ----- |
| `2026-08-10-vic-pose-action-controller-design.md` | Implemented → H2 (fix verification test path) |
| `2026-07-17-wrench-cap-retune-design.md` | **Superseded** (undocumented retune → 40 N / 10 N·m) → H2 |
| `docs/specs/2026-07-10-vic-wrench-caps-design.md` | Historical → H2 |

### Related

- → H1 for substep placement; → H4 for real default; → H5 for `--controller-mode`

---

## H3 — Sys-ID scoring & bags handbook

**Create:** `docs/handbook-sysid-scoring.md`  
**Primary absorb:** `handbook-sysid-scoring.md`, `handbook-sysid-scoring.md`, alignment + Sinkhorn specs.  
**Stub:** those two living docs + update `handbook-sysid-scoring.md` header to **Legacy single-env only**.

### TOC

1. **Three layers** — runtime obs vs bag vs score vector (mandatory glossary).
2. **`batched_sysid_v1` layout** — manifest, episode meta, frame columns; **action dim 6 or 19**; `hold_number`; woody starts; **no trajectory `woody_end`** (pre-weld exception only).
3. **`STATE_VECTOR_FIELDS`** — exact order/dims for \(J=2\); `REQUIRED_ARRAY_KEYS` still includes `action`.
4. **Feature alignment contract** (absorb `2026-08-13-…`):
   - Convert `R(tcp) @` F/T; score converted `ft_wrist`
   - Two woody starts + `apple_pos`; `CMA_WOODY_JUNCTIONS`
   - Scalar `hold_number` from `hold_index`
   - Explicit: **no** sim EMA/LPF; **no** scoring of `action`
5. **Normalization** — GT mean + `STATE_VECTOR_PHYS_SCALE` / `transition_feature_scale` (absorb `2026-08-14-…`); one-hots uncentered scale 1.
6. **Transition bags** — \([s,\Delta s]\) + hold/dir one-hots; pooling; median-hold (absorb `2026-07-14-median-hold-features-design.md`).
7. **Scorers** — Sinkhorn (`wasserstein.py`) is production; `biased_mmd2` / `batched_sysid_mmd_grid.py` marked **stale path**.
8. **Replay alignment notes** — absorb still-valid parts of `handbook-sysid-scoring.md` (pre-weld strip, structure weld, `--infer-params`).
9. **Legacy single-env Parquet** — short section + link to stubbed `handbook-sysid-scoring.md`.
10. **Code map + tests** — `test_mmd.py`, `test_mmd_features.py`, convert/collector tests.

### Archive specs

| Spec | Stamp |
| ---- | ----- |
| `2026-08-13-real-sim-cma-feature-alignment-design.md` | Implemented → H3 (also H4 convert) |
| `2026-08-14-sinkhorn-fixed-scale-normalization-design.md` | Implemented → H3 |
| `2026-07-14-median-hold-features-design.md` | Implemented → H3 |
| `2026-07-06-batched-sysid-mmd-grid-design.md` | Historical (MMD path stale) |
| `2026-06-22-mmd-grid-diagnostic-design.md` | Historical |
| `2026-07-04-batched-sysid-collection-design.md` | Historical → H3/H5 |
| `2026-06-22-sysid-dashboard-design.md` | Historical if dashboard still exists |

### Related

- → H4 convert must emit this contract; → H5 scores it; → H2 for action semantics only

---

## H4 — Real replay pipeline handbook

**Create:** `docs/handbook-real-replay.md`  
**Keep:** `robot_replay/README.md` as **CLI / folder layout cheat-sheet** (≤ ~150 lines) pointing here for contracts.  
**Stub:** `handbook-real-replay.md` after absorption.

### TOC

1. **End-to-end flow** — real parquet → convert → `batched_sysid_v1` → twin init → settle/weld → `vic_pose` replay → (grid/CMA in H5).
2. **Episode inputs** — `robot_replay/new_data/…` layout, tracking vs robot parquet, manifest.
3. **Convert contract** — `convert_real_to_batched_sysid_metadata.py` / `real_to_batched_sysid.export_real_episode_to_batched_dataset`:
   - 19D `vic_pose_v1` packing
   - F/T rotate at convert time
   - woody/hold packing per H3
4. **Pre-grasp vs post-grasp** — non-bending rebuild vs settled weld; absorb viewer specs’ *shipped* behavior:
   - `2026-07-24-real-pre-grasp-settle-viewer-design.md`
   - `2026-07-24-real-post-grasp-viewer-design.md`
   - `2026-08-07-real-to-batched-metadata-parity-design.md` (bits 1–2 Done; bit 3 open pieces → H5)
5. **Post-grasp SE(3)** — `apply_logged_post_grasp_se3_to_cable` / shared build (absorb `2026-08-11-…`).
6. **GL replay + camera + MP4** — absorb `2026-08-10-real-batched-gl-replay-design.md` (**Drive Done via vic_pose** — fix stale Pending), `2026-08-12-real-camera-gl-viewer-design.md`, `2026-08-12-gl-video-record-design.md`.
7. **Shared gym builder** — `real_batched_replay_build.py` / `make_real_replay_build_env_fn` (absorb plumbing slice 1 from `2026-08-12-real-replay-cmaes-plumbing-design.md`).
8. **Digital twin geometry** — link `docs/digital-twin.md`; note obs-vs-bag layering.
9. **CLI cheat-sheet** — defer detailed flags to `robot_replay/README.md`.
10. **Tests & verify** — `robot_replay/tests/…`, ROADMAP M4.0 commands (convert → grid smoke noted as parquet-local).

### Archive specs

| Spec | Stamp |
| ---- | ----- |
| Parity / GL / camera / video / SE3 / plumbing (Aug 07–12) | Implemented or Partial (plumbing slice 3–4 open → H5) |
| `2026-08-10-real-batched-gl-replay-design.md` | **Must** update Drive → Done before/while writing H4 |

### Related

- → H1 settle/weld; → H2 `vic_pose`; → H3 bag contract; → H5 ranking/CMA

---

## H5 — Young's / CMA phenotype handbook

**Create:** `docs/handbook-youngs-cma.md`  
**Stub:** `handbook-youngs-cma.md`, `handbook-youngs-cma.md`.

### TOC

1. **Phenotype** — support-joint \(k_p\) × spur/stem \(\log_{10} E\); primary \(E\) fixed (absorb `2026-08-04-support-joint-kp-sysid-design.md`).
2. **Cartesian / fused grid** — `example_youngs_modulus_sys_id.py`; real opt-in `vic_pose` via dataset metadata / `--controller-mode`.
3. **CMA-ES loop** — counters, soft-disable, reports, gates (absorb `2026-07-16-youngs-modulus-cmaes-loop-design.md` + current implementation doc).
4. **Scoring handoff** — **does not restate** H3 math; links to H3 for `STATE_VECTOR` / fixed-scale / no-action score.
5. **Real-data path** — shared builder from H4; GT bags without sim-oracle phenotype; `--include-gt-candidate` forced off; remaining open: trusted ranking smoke + CMA slice 4 (point to ROADMAP, don’t claim Done).
6. **Stability / exclusion** — link `batched-stability-monitor-design.md`.
7. **Commands** — collect → grid → CMA → gate scripts from README/ROADMAP.
8. **Tests** — gate scripts + unit/CLI controller-mode tests.

### Archive specs

| Spec | Stamp |
| ---- | ----- |
| `2026-08-04-support-joint-kp-sysid-design.md` | Implemented → H5 |
| `2026-07-16-youngs-modulus-cmaes-loop-design.md` | Implemented → H5 |
| Plumbing slices 3–4 in `2026-08-12-real-replay-cmaes-plumbing-design.md` | Partial — open work tracked in ROADMAP |

### Related

- → H3 scoring; → H4 builder; → H2 controller mode; `docs/system_identification.md` becomes protocol overview stub pointing here + H3

---

## What happens to other living docs

| Doc | Fate |
| --- | ---- |
| `VISION.md`, `ROADMAP.md` | Unchanged role; ROADMAP links handbooks instead of long spec lists where possible |
| `CODEBASE_GUIDE.md` | Rewrite Document index around H1–H5; Known gaps refreshed from ROADMAP |
| `FEATURES.md` (new, optional short) | One-page feature → handbook → code entry |
| `system_identification.md` | Trim to M3 protocol intent + pointers to H3/H5 (stop duplicating feature tables) |
| `gym-observation-contract.md` | Keep; add banner distinguishing obs vs bag vs score |
| `digital-twin.md` | Keep; cross-link H4; clarify woody_end in **obs/rebuild**, not score bags |
| `batched-stability-monitor-design.md` | Keep as satellite of H5 (or fold summary into H5 §6) |
| `real-world-proxy.md`, material/damping | Keep as H1 satellites |
| All `docs/superpowers/plans/*.md` | Untouched except optional footer “handbook: Hx” after related handbook ships |
| `.superpowers/sdd/*` | Execution artifacts — do not promote to living docs |

---

## File structure (create / modify)

| Path | Responsibility |
| ---- | -------------- |
| `docs/handbook-coupled-simulation.md` | H1 living handbook |
| `docs/handbook-variable-impedance.md` | H2 |
| `docs/handbook-sysid-scoring.md` | H3 |
| `docs/handbook-real-replay.md` | H4 |
| `docs/handbook-youngs-cma.md` | H5 |
| `docs/FEATURES.md` | Optional one-page index |
| `docs/CODEBASE_GUIDE.md` | Point at handbooks |
| Stubs for replaced living docs | 10–20 line redirect + “last full content in git history / handbook” |
| Listed superpowers specs | Status stamp + `Canonical: docs/handbook-….md` |
| `robot_replay/README.md` | Slim CLI; link H4 |
| `docs/ROADMAP.md` | Replace long “Build on / Specs:” bullets with handbook links where helpful |

---

### Task 0: Spec stamp pass + P0 living-doc corrections (unblock accurate writing)

**Files:**
- Modify: every `docs/superpowers/specs/*.md` status table (and the three `docs/specs/*.md`)
- Especially fix: `2026-08-10-real-batched-gl-replay-design.md` Drive → Done
- **P0 living docs** (correct numbers/contracts *before* handbooks, or the first handbook draft will copy lies):
  - `docs/system_identification.md` §3.1 / §3.3 / §4 (state vector, fixed-scale norm, support-\(k_p\) phenotype) — highest blast radius
  - Cap numbers → **40 N / 10 N·m**: `batched-stability-monitor-design.md`, `ROADMAP.md` (and distinguish legacy SysId 100/100 vs batched in `gym-observation-contract.md` / trajectory-storage)
  - `docs/handbook-sysid-scoring.md` schema (no frame `woody_end`, 6-or-19D action, `hold_number`, `action_*` metadata)
  - `robot_replay/README.md`: close broken bash fence; point commands at `new_data/`; Bit-3 prose must not promise LPF / pose-only action; note `--record-video` / `camera_to_base_4x4`

- [ ] **Step 1:** For each of the 30 specs, set Status to one of `Implemented` / `Partial` / `Superseded` / `Historical` using ROADMAP + code (not prose tense). Priority CONFLICT stamps:
  - `2026-07-17-wrench-cap-retune` → Superseded (code 40 N / 10 N·m)
  - `2026-06-22-mmd-grid-diagnostic` → Historical / superseded by H3 + 08-13/08-14 specs
  - `2026-08-04-tcp-tip-flange` → Superseded length (180 mm); keep radius/test notes
  - `2026-07-16-youngs-cmaes-loop` → Superseded phenotype-only by support-\(k_p\) spec
- [ ] **Step 2:** Add `Canonical living doc:` field (Hx path) even if handbook not written yet (planned path OK). Fix broken verify paths (`test_vic_pose_actions.py` → `test_ee_impedance_batched_pose_actions.py`; MMD grid test module name; drop missing `task-8-report.md` cite or note absent).
- [ ] **Step 3:** Apply P0 living-doc corrections listed above (system_identification, caps, batched schema, robot_replay README + `new_data/` + fence). Sync root `README.md` real-replay paths if they still cite missing parquet names.
- [ ] **Step 4:** Commit stamps + P0 doc corrections.

```bash
git add docs/superpowers/specs docs/specs
git commit -m "$(cat <<'EOF'
docs: stamp superpowers specs with handbook canonical targets

EOF
)"
```

---

### Task 1: Write H3 (Sys-ID scoring) first

**Why first:** Highest CRIT discrepancy density; H4/H5 depend on its contracts.

**Files:**
- Create: `docs/handbook-sysid-scoring.md`
- Modify: stub `docs/handbook-sysid-scoring.md`, `docs/handbook-sysid-scoring.md`
- Modify: banner on `docs/handbook-sysid-scoring.md`, `docs/gym-observation-contract.md`

- [ ] **Step 1:** Draft H3 TOC sections 1–5 from code:
  - `STATE_VECTOR_FIELDS`, `STATE_VECTOR_PHYS_SCALE` in `mmd_features.py`
  - `fit_gt_normalization` in `mmd.py`
  - convert woody/F-T/hold in `real_to_batched_sysid.py`
  - bag write rules in `batched_trajectory_store.py`
- [ ] **Step 2:** Grep living docs for stale phrases and ensure H3 contradicts none of code:
  ```bash
  rg -n "GT z-score|woody_end__|action.*f32×6|std\(GT\)" docs/handbook-sysid-scoring.md
  ```
  Expected: zero stale claims (historical callouts OK if labeled Legacy).
- [ ] **Step 3:** Stub old living docs with pointer to H3.
- [ ] **Step 4:** Commit.

```bash
git add docs/handbook-sysid-scoring.md docs/handbook-sysid-scoring.md docs/handbook-sysid-scoring.md docs/handbook-sysid-scoring.md docs/gym-observation-contract.md
git commit -m "$(cat <<'EOF'
docs: add sys-ID scoring handbook from code-truth contracts

EOF
)"
```

---

### Task 2: Write H2 (VIC / vic_pose)

**Files:**
- Create: `docs/handbook-variable-impedance.md`
- Stub: `docs/handbook-variable-impedance.md`

- [ ] **Step 1:** Port accurate sections from current `handbook-variable-impedance.md`; expand `vic_pose` from code + `2026-08-10` spec.
- [ ] **Step 2:** Verify soft-disable / aniso kernel claims against tests listed in TOC.
- [ ] **Step 3:** Stub old path; commit.

---

### Task 3: Write H4 (Real replay)

**Files:**
- Create: `docs/handbook-real-replay.md`
- Modify: `robot_replay/README.md` (slim + link)
- Stub: `docs/handbook-real-replay.md`
- Modify: GL replay spec status if not done in Task 0

- [ ] **Step 1:** Document convert → replay → shared builder using `robot_replay/` + `real_batched_replay_build.py`.
- [ ] **Step 2:** Explicitly state Drive = `vic_pose` Done; ranking/CMA open → H5/ROADMAP.
- [ ] **Step 3:** Cross-link H1/H2/H3; commit.

---

### Task 4: Write H5 (Young's / CMA)

**Files:**
- Create: `docs/handbook-youngs-cma.md`
- Stub: `docs/handbook-youngs-cma.md`, `docs/handbook-youngs-cma.md`
- Trim: `docs/system_identification.md` to protocol + H3/H5 pointers (after Task 0 patched the wrong tables — highest blast-radius stale doc)

- [ ] **Step 1:** Port grid/CMA commands and phenotype from existing Young’s docs + support-\(k_p\) spec.
- [ ] **Step 2:** Do **not** duplicate H3 feature math — link only.
- [ ] **Step 3:** Mark open M4.0 ranking/CMA items as ROADMAP-owned; commit.

---

### Task 5: Write H1 (Coupled simulation)

**Files:**
- Create: `docs/handbook-coupled-simulation.md`
- Stub: `handbook-coupled-simulation.md`, `handbook-coupled-simulation.md`, `handbook-coupled-simulation.md`

- [ ] **Step 1:** Merge architecture + API + batched flow; link satellites instead of inlining all wrench math.
- [ ] **Step 2:** Absorb weld/TCP *outcomes* (not full design history).
- [ ] **Step 3:** Stub three primaries; commit.

---

### Task 6: Index + ROADMAP pointer refresh

**Files:**
- Modify: `docs/CODEBASE_GUIDE.md`
- Create (optional): `docs/FEATURES.md`
- Modify: `docs/ROADMAP.md` (Current focus “Build on / Specs” → handbook links)
- Modify: `AGENTS.md` if it lists doc read order — insert handbooks after ROADMAP

- [ ] **Step 1:** Rewrite CODEBASE_GUIDE Document index to H1–H5 first; archive section for superpowers.
- [ ] **Step 2:** Refresh Known gaps from ROADMAP (remove “draft viewer”, fix M4.0 next).
- [ ] **Step 3:** Grep for stale owner-table claims:
  ```bash
  rg -n "GT z-score|Drive pending|wrench ≠ twist|Last reviewed.*2026-07" docs/*.md
  ```
- [ ] **Step 4:** Commit.

---

### Task 7: Verification gate (docs-only)

- [ ] **Step 1:** For each handbook, confirm every “Code map” symbol exists:
  ```bash
  rg -n "STATE_VECTOR_PHYS_SCALE|make_real_replay_build_env_fn|CMA_WOODY_JUNCTIONS|unpack_pose_action" apple_pick_sim apple_pick_gym robot_replay
  ```
- [ ] **Step 2:** Run a cheap test subset cited by H3 (scoring contract):
  ```bash
  uv run --env-file pytest.env python -m pytest \
    apple_pick_sim/tests/test_mmd_features.py apple_pick_sim/tests/test_mmd.py -q -p no:launch_testing
  ```
  Expected: PASS
- [ ] **Step 3:** Update audit canvas or add `docs/superpowers/reports/2026-08-14-handbook-coalesce-report.md` listing stubs + remaining satellite docs.
- [ ] **Step 4:** Final commit.

---

## Self-review (plan vs Approach B)

| Requirement | Task coverage |
| ----------- | ------------- |
| Five domain handbooks absorbing specs | Tasks 1–5 |
| Related-doc links | Cross-link matrix + each TOC “Related” |
| Specs demoted / stamped | Task 0 + per-handbook archive tables |
| Code-truth for M4.0 scoring/replay | H3 then H4/H5 order |
| Index refresh | Task 6 |
| No placeholder “TBD” sections in handbooks | Each TOC lists concrete sections; open work deferred to ROADMAP by name |

**Out of scope for this plan:** moving files into `docs/archive/` filesystem; rewriting satellite math docs; implementing ranking/CMA product work.

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-08-14-domain-handbooks-coalesce.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per handbook task (0→1→3→4→5→2→6 or as ordered), review between tasks  
2. **Inline Execution** — same session, checkpoint after each handbook

**Which approach?**
