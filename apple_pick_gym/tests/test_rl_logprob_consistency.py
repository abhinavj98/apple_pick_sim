"""Rollout vs update log-probs: recomputing the stored rollout must give the same log-probs.

GPU (D11 run, 250-step episodes): KL spikes of 0.2-9 at LR 3e-5, which a real policy change can't
produce. This pins where the mismatch can and cannot come from:

- the recurrent path (stored LSTM states, resets at episode ends mid-sequence): must match exactly
  when the observation / state scalers are frozen;
- skrl's RunningStandardScaler updates its statistics inside update epoch 0 (``train=not epoch``), so
  one extreme row (a solver blow-up) shifts every row's normalised input and hence the log-probs.
"""

from __future__ import annotations

import dataclasses

import torch

from apple_pick_gym.rl import trainer as trainer_mod
from apple_pick_gym.rl.config import EnvConfig, PPOConfig, TrainConfig
from apple_pick_gym.rl.models import RecurrentNetConfig

_NET = RecurrentNetConfig(pre_mlp=(32,), lstm_hidden=16, post_mlp=(32,), sequence_length=8)


def _recompute_log_prob(agent) -> tuple[torch.Tensor, torch.Tensor]:
    """``(stored, recomputed)`` log-probs over the whole memory, in update order, scalers frozen."""
    mem = agent.memory
    idx = mem.all_sequence_indexes
    view = lambda name: mem.tensors_view[name][idx]
    rnn = [view(n).transpose(0, 1) for n in agent._rnn_tensors_names if "policy" in n]
    inputs = {
        "observations": agent._observation_preprocessor(view("observations")),
        "states": agent._state_preprocessor(view("states")),
        "taken_actions": view("actions"),
        "rnn": rnn,
        "terminated": view("terminated"),
        "truncated": view("truncated"),
    }
    with torch.no_grad():
        _, outputs = agent.policy.act(inputs, role="policy")
    return view("log_prob").flatten(), outputs["log_prob"].flatten()


def test_recurrent_log_probs_match_the_rollout_when_episodes_end_mid_sequence(tmp_path, monkeypatch):
    seen = {}
    base_factory = trainer_mod._ppo_rnn_class

    def factory():
        base = base_factory()

        class Checked(base):
            def update(self, *, timestep: int, timesteps: int) -> None:
                if "stored" not in seen:
                    seen["stored"], seen["recomputed"] = _recompute_log_prob(self)
                super().update(timestep=timestep, timesteps=timesteps)

        return Checked

    monkeypatch.setattr(trainer_mod, "_ppo_rnn_class", factory)
    # 13-step episodes vs 8-step sequences: auto-resets land mid-sequence at varying offsets
    cfg = TrainConfig(
        env=EnvConfig(kind="surrogate", num_envs=4, max_episode_steps=13, device="cpu"),
        ppo=PPOConfig(rollouts=32, mini_batches=2, learning_epochs=1),
        actor=_NET,
        critic=_NET,
        timesteps=32,
        checkpoint_every_updates=100,
        run_dir=str(tmp_path / "run"),
    )
    trainer_mod.run_training(cfg)
    stored, recomputed = seen["stored"], seen["recomputed"]
    assert stored.shape == recomputed.shape and stored.numel() == 32 * 4
    # float32: batched vs step-by-step LSTM differ by ~1e-3 on a ~-7 log-prob (KL ~1e-7)
    torch.testing.assert_close(recomputed, stored, atol=2e-3, rtol=5e-4)


def test_scaler_update_inside_the_update_shifts_every_row():
    # skrl's RunningStandardScaler: stats update on the batch, then every row is renormalised with them
    from skrl.resources.preprocessors.torch import RunningStandardScaler

    scaler = RunningStandardScaler(size=3, device="cpu")
    normal = torch.randn(512, 3)
    scaler(normal, train=True)
    before = scaler(normal[:4]).clone()
    batch = torch.cat([normal, torch.full((1, 3), 1900.0)])  # one blow-up row (N), as on GPU
    scaler(batch, train=True)
    after = scaler(normal[:4])
    assert float((after - before).abs().max()) > 0.5  # every normal row moved


def test_debug_kl_logs_pre_update_kl_and_scaler_shift(tmp_path):
    # opt-in GPU diagnostic: pre-update KL with scalers frozen (~0 if stored data is consistent) and
    # how far the obs scaler's mean moves during the update (in its own std units)
    import json

    cfg = TrainConfig(
        env=EnvConfig(kind="surrogate", num_envs=4, max_episode_steps=13, device="cpu"),
        ppo=PPOConfig(rollouts=16, mini_batches=2, learning_epochs=1, debug_kl=True),
        actor=_NET,
        critic=_NET,
        timesteps=32,
        checkpoint_every_updates=100,
        run_dir=str(tmp_path / "run"),
    )
    trainer_mod.run_training(cfg)
    rows = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    ups = [r for r in rows if r["kind"] == "update"]
    assert len(ups) == 2
    for r in ups:
        assert r["Debug / pre-update KL (frozen scalers)"] < 1e-5
        assert r["Debug / obs scaler mean shift (max, std units)"] >= 0.0
