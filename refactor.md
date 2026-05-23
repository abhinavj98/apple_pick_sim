# Refactor backlog (maintainer-owned)

**Status (2026-05-22):** Staggered **coupling + force transfer** behavior is **accepted** — refactors should preserve apply → MuJoCo → sync → VBD → harvest unless a slice explicitly changes physics with tests. **Abhinav** will expand this file and **`docs/ROADMAP.md` Current focus** with ordered tasks; agents should read the latest version before large structural PRs.

---

## Draft tasks (initial notes — not authoritative until maintainer revises)

apple_pick_sim/fruiting_system/
  __init__.py          # re-export current public API (README/tests unchanged)
  params.py            # ranges, validate, sample, RodParams, FruitingSystemParams, fingerprints on params
  build.py             # _FruitingChainArtifacts, chain, rods, proxy add, finalize, pin, collision filters
  scene.py             # FruitingSystemScene, generate_scene, run_rollout, measure_*, solver helpers
  coupled.py           # GripperProxyConfig, CoupledCableScene, generate_coupled_cable_scene, coupled fingerprint


For apple_pick_sim/coupled_fruiting.py
  Rename 
  sync_proxy_state: extract_forces_vbd_to_mujoco

  harvest_proxy_wrenches_velocity_delta_kernel: compute_mujoco_return_wrench

  Other launch and coupled sync functions similarly need to be renamed. 

  IMPORTANT: There are some CPU functions that need to be moved as a warp kernel. I want end-to-end GPU simulation

