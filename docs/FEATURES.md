# Feature map

Use this page to find the living handbook and first code entry for a feature.
The handbooks own subsystem contracts; `docs/ROADMAP.md` owns status and
sequencing. This page intentionally does not repeat either.

| Feature | Living handbook | Start in code |
| --- | --- | --- |
| Coupled MuJoCo arm + VBD plant | [H1: Coupled simulation](handbook-coupled-simulation.md) | `apple_pick_sim/coupled_fruiting/scene.py::CoupledFruitingScene` |
| Batched heterogeneous build, settle, and weld | [H1: Coupled simulation](handbook-coupled-simulation.md) | `apple_pick_sim/coupled_fruiting/batched_heterogeneous_build.py::build_batched_heterogeneous_scene` |
| Fruiting geometry and material sampling | [H1: Coupled simulation](handbook-coupled-simulation.md) | `apple_pick_sim/fruiting_system/params.py`, `build.py` |
| `vic` twist and `vic_pose` control | [H2: Variable impedance](handbook-variable-impedance.md) | `apple_pick_sim/coupled_fruiting/vic_joint_torques.py`, `ee_impedance_batched.py` |
| TCP wrench limits and soft-disable | [H2: Variable impedance](handbook-variable-impedance.md) | `apple_pick_sim/coupled_fruiting/vic_wrench.py`, `apple_pick_gym/batched_envs/` |
| Batched sys-ID datasets and feature bags | [H3: Sys-ID scoring](handbook-sysid-scoring.md) | `apple_pick_sim/system_id/batched_trajectory_store.py`, `mmd_features.py` |
| Sinkhorn/Wasserstein and MMD scoring | [H3: Sys-ID scoring](handbook-sysid-scoring.md) | `apple_pick_sim/system_id/wasserstein.py`, `mmd.py` |
| Real Parquet conversion and replay | [H4: Real replay](handbook-real-replay.md) | `robot_replay/`, `apple_pick_sim/system_id/real_to_batched_sysid.py` |
| Real pre-grasp rebuild and post-grasp weld | [H4: Real replay](handbook-real-replay.md) | `apple_pick_sim/system_id/batched_digital_twin_init.py`, `real_post_grasp_plan.py` |
| Shared real grid/CMA builder | [H4: Real replay](handbook-real-replay.md) | `apple_pick_gym/batched_envs/real_batched_replay_build.py` |
| Young's/support-\(k_p\) Cartesian grid | [H5: Young's and CMA](handbook-youngs-cma.md) | `apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py` |
| CMA-ES calibration and gates | [H5: Young's and CMA](handbook-youngs-cma.md) | `apple_pick_gym/batched_envs/batched_sysid_cmaes.py`, `apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py` |

For the complete repository and documentation map, see
`docs/CODEBASE_GUIDE.md`. Dated files under `docs/superpowers/{specs,plans}/`
are design archives, not additional living contracts.
