"""Open-loop trajectory replay env for sim-to-real calibration rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from apple_pick_gym.envs.apple_pick_sysid_env import ApplePickSysIdEnv
from apple_pick_sim.system_id.trajectory_store import (
    TrajectoryDataset,
    load_grasp_snapshot_into_env,
)
from apple_pick_sim.system_id.parquet_init import (
    initialize_env_from_parquet,
    observation_reset_options_from_parquet,
)


class ApplePickReplayEnv(ApplePickSysIdEnv):
    """Replay pre-recorded EE velocity trajectories in the coupled sim.

  Observations (all ``float32``):

  - ``ft_wrist``: ``(6,)`` plant wrench at TCP ``[F(3), tau(3)]`` [N, N·m]
  - ``woody_start``: ``(N*3,)`` parent-side fixed-joint anchor positions [m]
  - ``woody_end``: ``(N*3,)`` child-side fixed-joint anchor positions [m]
  - ``tcp_velocity``: ``(6,)`` TCP spatial velocity ``[v(3), omega(3)]`` [m/s, rad/s]
  - ``tcp_pos``: ``(3,)`` actual TCP body position [m]
  - ``apple_pos``: ``(3,)`` apple body CoM [m]

  Action contract: ``Box(6)`` EE velocity per step. During dataset replay the
  passed action is ignored; stored actions from the dataset are applied instead.

  Use ``reset(options={'params': FruitingSystemParams(...)})`` to inject tuned
  candidate parameters for CEM rollouts.
  """

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        max_episode_steps: int = 240,
        enable_self_collisions: bool = False,
        fix_to_apple: bool = True,
        fix_to_apple_warmup_substeps: int = 0,
        max_woody_parts: int = 64,
        mujoco_solver_kwargs: dict[str, Any] | None = None,
        control_hz: float = 60.0,
        dataset_dir: str | Path | None = None,
        robot_facing_weld: bool = False,
        device: str | None = None,
    ) -> None:
        self._dataset_dir = Path(dataset_dir) if dataset_dir is not None else None
        self._dataset: TrajectoryDataset | None = None
        self._episode_id: str | None = None
        self._episode_meta: dict[str, Any] | None = None
        self._replay_actions: np.ndarray | None = None
        self._replay_step_idx = 0
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            enable_self_collisions=enable_self_collisions,
            fix_to_apple=fix_to_apple,
            fix_to_apple_warmup_substeps=fix_to_apple_warmup_substeps,
            max_woody_parts=max_woody_parts,
            mujoco_solver_kwargs=mujoco_solver_kwargs,
            control_hz=control_hz,
            robot_facing_weld=robot_facing_weld,
            device=device,
        )

    @staticmethod
    def _observation_space_for(n_woody: int) -> spaces.Dict:
        return spaces.Dict(
            {
                "ft_wrist": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
                "woody_start": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_woody * 3,), dtype=np.float32
                ),
                "woody_end": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_woody * 3,), dtype=np.float32
                ),
                "tcp_velocity": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                ),
                "tcp_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "tcp_quat": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
                ),
                "apple_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "apple_quat": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
                ),
                "robot_joint_q": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
            }
        )

    def _setup_action_space(self) -> None:
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.observation_space = self._observation_space_for(self._cfg.max_woody_parts)

    def _setup_observation_space(self) -> None:
        self.observation_space = self._observation_space_for(self._n_woody_parts)

    def _make_obs(self) -> dict[str, Any]:
        start_pos, end_pos = self._woody_start_end_pos()
        return {
            "ft_wrist": self._ft_wrist(),
            "woody_start": start_pos,
            "woody_end": end_pos,
            "tcp_velocity": self._tcp_velocity(),
            "tcp_pos": self._tcp_pos(),
            "tcp_quat": self._tcp_quat(),
            "apple_pos": self._apple_pos(),
            "apple_quat": self._apple_quat(),
            "robot_joint_q": self._robot_joint_q(),
        }

    def load_dataset(
        self,
        dataset_dir: str | Path,
        *,
        episode_id: str | None = None,
    ) -> str:
        """Load a trajectory dataset directory and select an episode."""
        self._dataset_dir = Path(dataset_dir)
        self._dataset = TrajectoryDataset(self._dataset_dir)
        episode_ids = self._dataset.episode_ids()
        if not episode_ids:
            raise ValueError(f"dataset has no episodes: {self._dataset_dir}")
        if episode_id is None:
            episode_id = episode_ids[0]
        elif episode_id not in episode_ids:
            raise KeyError(f"episode_id {episode_id!r} not found in dataset")
        self._episode_id = episode_id
        self._episode_meta = self._dataset.load_episode_meta(episode_id)
        self._replay_actions = self._dataset.load_episode_actions(episode_id)
        return episode_id

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        options = dict(options or {})
        use_snapshot = bool(options.pop("use_snapshot", True))
        dataset_dir = options.pop("dataset_dir", None)
        if dataset_dir is not None:
            self.load_dataset(dataset_dir, episode_id=options.get("episode_id"))
        elif self._dataset is None and self._dataset_dir is not None:
            self.load_dataset(self._dataset_dir, episode_id=options.get("episode_id"))

        snap = None
        if use_snapshot and self._dataset is not None and self._episode_id is not None:
            snap = self._dataset.load_initial_state(self._episode_id)

        if snap is not None and "weld_direction" in snap:
            options["weld_direction"] = tuple(float(x) for x in snap["weld_direction"])
            if "weld_reference_pos" in snap:
                options["weld_reference_pos"] = tuple(
                    float(x) for x in snap["weld_reference_pos"]
                )
            if "weld_reference_quat" in snap:
                options["weld_reference_quat"] = tuple(
                    float(x) for x in snap["weld_reference_quat"]
                )
        elif self._episode_meta is not None:
            if self._dataset is not None and self._episode_id is not None:
                derived_options = observation_reset_options_from_parquet(
                    self._dataset,
                    self._episode_id,
                )
                for key, value in derived_options.items():
                    options.setdefault(key, value)
            weld = self._episode_meta.get("weld_direction")
            if weld is not None:
                options.setdefault("weld_direction", weld)

        if self._episode_meta is not None:
            control_hz = self._episode_meta.get("control_hz")
            if control_hz is not None:
                self._cfg.control_hz = float(control_hz)
            n_frames = 0 if self._replay_actions is None else int(self._replay_actions.shape[0])
            if n_frames > 0:
                self._cfg.max_episode_steps = n_frames

        obs, info = super().reset(seed=seed, options=options)
        self._replay_step_idx = 0
        if self._episode_id is not None:
            info["replay_episode_id"] = self._episode_id

        if snap is not None:
            if self._cfg.fix_to_apple_warmup_substeps == 0:
                load_grasp_snapshot_into_env(self, snap)
                obs = self._make_obs()
                info["initial_state_restored"] = True
                info["observation_init"] = False
            else:
                info["initial_state_restored"] = False
                info["observation_init"] = False
        elif self._dataset is not None and self._episode_id is not None:
            initialize_env_from_parquet(self, self._dataset, self._episode_id)
            obs = self._make_obs()
            info["initial_state_restored"] = False
            info["observation_init"] = True

        return obs, info

    def step(self, action):
        del action
        if self._replay_actions is None or self._replay_actions.shape[0] == 0:
            raise RuntimeError("load_dataset() or reset(options={'dataset_dir': ...}) before step()")

        if self._replay_step_idx >= self._replay_actions.shape[0]:
            obs = self._make_obs()
            info = self._make_info()
            info["replay_action"] = np.zeros((6,), dtype=np.float32)
            return obs, 0.0, False, True, info

        replay_action = self._replay_actions[self._replay_step_idx]
        frame_idx = int(self._replay_step_idx)
        self._replay_step_idx += 1
        obs, reward, terminated, truncated, info = super().step(replay_action)
        info["replay_action"] = np.asarray(replay_action, dtype=np.float32)
        info["replay_frame_idx"] = frame_idx
        info["vic_use_joint_torques"] = bool(
            self._scene is not None and getattr(self._scene, "vic_use_joint_torques", False)
        )
        if self._replay_step_idx >= self._replay_actions.shape[0]:
            truncated = True
        return obs, reward, terminated, truncated, info

    def compute_reward(self, obs: dict[str, Any], info: dict[str, Any]) -> float:
        del obs, info
        return 0.0

    def compute_terminated(self, obs: dict[str, Any], info: dict[str, Any]) -> bool:
        del obs, info
        return False
