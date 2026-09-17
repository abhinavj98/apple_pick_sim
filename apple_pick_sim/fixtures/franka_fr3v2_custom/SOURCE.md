# Franka FR3 custom calibrated dynamics fixtures

Calibrated joint dynamics for FCI/MuJoCo use; inertials match official FR3 v2.1.

- `inertials.yaml` — link0–7 mass, COM, inertia tensor (from fr3v2_1)
- `dynamics.yaml` — joint motor inertia, gear ratio, per-joint `mu_viscous` / `mu_coulomb`
  (also mirrors `damping` / `armature` for engine-native fields)

Used at VIC configure time by `apple_pick_sim.robot.fr3_robot.fr3_v21_props`.
