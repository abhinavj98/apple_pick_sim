"""System identification excitation utilities for M3.0."""

from apple_pick_sim.system_id.excitation_state import ExcitationContext
from apple_pick_sim.system_id.fibonacci_hemisphere import (
    sample_fibonacci_hemisphere,
    sample_robot_facing_pull_directions,
    stem_perpendicular_robot_pole,
)
from apple_pick_sim.system_id.episode_meta import EpisodeMeta
from apple_pick_sim.system_id.parquet_init import (
    digital_twin_obs_from_episode,
    initialize_env_from_parquet,
    observation_reset_options_from_parquet,
)
from apple_pick_sim.system_id.quasi_static_trajectory import (
    QuasiStaticStepConfig,
    QuasiStaticTrajectory,
    derive_n_steps,
    estimate_trajectory_frames,
)
from apple_pick_sim.system_id.batched_digital_twin_init import (
    digital_twin_obs_from_batched_episode,
    infer_base_params_for_structure,
)
from apple_pick_sim.system_id.batched_trajectory_store import (
    BatchedEpisodeWriter,
    BatchedSysIdDataset,
    batched_dataset_exists,
    materialize_legacy_episode_dir,
    resolve_batched_dataset_output_dir,
    write_manifest,
)
from apple_pick_sim.system_id.trajectory_store import (
    TrajectoryDataset,
    TrajectoryWriter,
    grasp_snapshot_from_env,
    load_grasp_snapshot_into_env,
    target_tf_from_array,
    target_tf_to_array,
)

__all__ = [
    "BatchedEpisodeWriter",
    "BatchedSysIdDataset",
    "batched_dataset_exists",
    "EpisodeMeta",
    "ExcitationContext",
    "QuasiStaticStepConfig",
    "QuasiStaticTrajectory",
    "TrajectoryDataset",
    "TrajectoryWriter",
    "digital_twin_obs_from_batched_episode",
    "digital_twin_obs_from_episode",
    "infer_base_params_for_structure",
    "derive_n_steps",
    "estimate_trajectory_frames",
    "grasp_snapshot_from_env",
    "initialize_env_from_parquet",
    "load_grasp_snapshot_into_env",
    "materialize_legacy_episode_dir",
    "observation_reset_options_from_parquet",
    "resolve_batched_dataset_output_dir",
    "sample_fibonacci_hemisphere",
    "sample_robot_facing_pull_directions",
    "stem_perpendicular_robot_pole",
    "target_tf_from_array",
    "target_tf_to_array",
    "write_manifest",
]
