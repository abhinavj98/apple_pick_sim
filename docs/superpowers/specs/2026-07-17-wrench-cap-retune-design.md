# Wrench-Cap Retune Design

**Date:** 2026-07-17  
**Status:** Superseded — current scene defaults are 40 N / 10 N·m
**Canonical living doc:** `docs/handbook-variable-impedance.md`

## Purpose

Raise the default stem-feedback force cap from 50 N to 100 N and the torque cap
from 20 N·m to 40 N·m. Replay instability detection must continue to use these
same shared defaults, so applied-wrench clamping and instability classification
remain aligned.

## Scope

- Change `DEFAULT_STEM_FORCE_CAP_N` to `100.0`.
- Change `DEFAULT_STEM_TORQUE_CAP_NM` to `40.0`.
- Preserve `StabilityThresholds` defaults as references to those shared
  constants.
- Update tests and documentation that state the old limits.
- Keep the unstable-frame disqualification threshold unchanged at strictly
  greater than 25%.

## Testing

First update the default-value assertions to require 100 N and 40 N·m and
confirm they fail against the old constants. Then update the production
constants and run the focused simulation configuration and batched stability
monitor tests.

## Non-goals

- Adding separate replay-only limits.
- Adding CLI overrides.
- Changing speed, NaN/Inf, IK, or unstable-fraction policies.
