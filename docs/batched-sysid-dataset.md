# Batched sys-ID dataset

> Redirect: the `batched_sysid_v1` storage and scoring contracts now live in
> the [Sys-ID scoring and bags handbook](handbook-sysid-scoring.md).

Use the handbook for:

- manifest, episode metadata, and frame-table layout;
- 6D `vic` and 19D `vic_pose_v1` actions;
- scalar `hold_number` and CMA woody starts;
- the trajectory rule excluding `woody_end`; and
- the boundary between replay bags and score vectors.

Implementation entry points remain
`apple_pick_sim.system_id.batched_trajectory_store.BatchedEpisodeWriter` and
`BatchedSysIdDataset`. The last full version is available in git history.
Roadmap sequencing remains in [`ROADMAP.md`](ROADMAP.md).
