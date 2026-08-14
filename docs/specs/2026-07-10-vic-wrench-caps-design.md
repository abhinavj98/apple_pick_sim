# VIC wrench caps (FR3-aligned)

**Date:** 2026-07-10  
**Status:** Superseded — current scene caps are 40 N / 10 N·m
**Canonical living doc:** `docs/handbook-variable-impedance.md`
**Branch:** `feature/batched-sysid-mmd`

## Problem

VIC impedance can command arbitrarily large TCP wrenches when pose error or gains are large (`F = K e + D (v_des − v_act)`). Stem harvest already clamps plant→robot feedback (`stem_force_cap_N` / `stem_torque_cap_Nm`), but the **applied VIC wrench** itself is uncapped. That is unrealistic for an FR3-class arm and can destabilize the coupled sim.

## Goals

- Clamp VIC force and torque norms independently after the impedance law (same pattern as stem harvest).
- Defaults sit slightly below stem caps (~80%).
- Raise stem defaults to **50 N** / **20 N·m**; VIC defaults to **40 N** / **16 N·m**.
- Config lives on `ControllerConfig` only (not fixture `sim_build` in this slice).
- Cover CPU `compute_applied_wrench` and the batched Warp VIC wrench kernel.

## Non-goals

- Soft (tanh) saturation.
- Joint-torque or joint-limit caps.
- Putting caps in ranges JSON `sim_build`.
- Changing how stem harvest caps are applied (only their default values).

## Defaults

| Knob | New default | Notes |
| ---- | ----------- | ----- |
| `DEFAULT_STEM_FORCE_CAP_N` | `50.0` | Was `30.0` |
| `DEFAULT_STEM_TORQUE_CAP_NM` | `20.0` | Was `10.0` |
| `ControllerConfig.vic_force_cap_N` | `40.0` | ~80% of stem force |
| `ControllerConfig.vic_torque_cap_Nm` | `16.0` | ~80% of stem torque |

`None` or `≤ 0` on a VIC cap field disables that axis (uncapped).

## Clamp law

After computing impedance force \(F\) and torque \(\tau\):

\[
F \leftarrow F \cdot \min\!\left(1,\ \frac{F_{\max}}{\|F\|+\varepsilon}\right),\quad
\tau \leftarrow \tau \cdot \min\!\left(1,\ \frac{\tau_{\max}}{\|\tau\|+\varepsilon}\right)
\]

Independent scaling (does **not** scale the full 6D wrench as one vector). Match stem harvest’s existing style in `proxy_coupling.py`.

## API / wiring

1. Add `vic_force_cap_N: float | None = 40.0` and `vic_torque_cap_Nm: float | None = 16.0` to `ControllerConfig` in `batched_heterogeneous_config.py`.
2. Prefer a small shared helper (e.g. `clip_spatial_wrench_force_torque`) used by:
   - `Fr3EEImpedanceController.compute_applied_wrench` (and any thin wrapper),
   - batched VIC Warp kernel / launcher in `vic_joint_torques_batched.py`.
3. When configuring VIC on `BatchedHeterogeneousCoupledSim` / scene, copy caps from `config.controller` onto scene fields (or pass into launchers) so the hot path does not re-read the frozen config each substep if that is already the pattern for gains.
4. Update `DEFAULT_STEM_FORCE_CAP_N` / `DEFAULT_STEM_TORQUE_CAP_NM` in `coupled_fruiting/scene.py` (and any tests/docs that hard-code 30/10).

## Tests

- Helper: below cap unchanged; above cap scales to exact limit; zero/None disables.
- `compute_applied_wrench` with large \(K\cdot e\) → \(\|F\|\le 40\), \(\|\tau\|\le 16\).
- Batched path parity or kernel-level assertion that caps are applied.
- Config defaults: stem 50/20, VIC 40/16.
- Update existing tests that assert stem defaults of 30/10.

## Docs

- `docs/variable-impedance-teleop.md` — note VIC caps and defaults.
- Fix stale “100 N / 100 N·m” stem-cap prose where it still appears if touched; at least update damping/sys-id notes that cite the old 30/10 if present in the same edit set.

## Success criteria

- With high VIC gains and large TCP error, applied VIC force/torque norms never exceed 40 N / 16 N·m when caps are enabled.
- Stem harvest defaults are 50 N / 20 N·m.
- Disabling VIC caps (`None`) restores uncapped impedance wrench behavior.
