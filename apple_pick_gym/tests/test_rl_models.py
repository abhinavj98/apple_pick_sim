"""LSTM actor / privileged LSTM critic for skrl 2.1 PPO_RNN (CPU, fast)."""

from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch

from apple_pick_gym.rl.models import LstmGaussianActor, LstmValueCritic, RecurrentNetConfig

OBS, STATE, ACT, N, L = 7, 11, 3, 4, 5


def _spaces():
    box = lambda d: gymnasium.spaces.Box(-np.inf, np.inf, (d,), np.float32)
    act = gymnasium.spaces.Box(-1.0, 1.0, (ACT,), np.float32)
    return box(OBS), box(STATE), act


def _cfg(**kw):
    return RecurrentNetConfig(pre_mlp=(16,), lstm_hidden=8, lstm_layers=1, post_mlp=(12,), sequence_length=L, **kw)


def _models():
    torch.manual_seed(0)
    o, s, a = _spaces()
    actor = LstmGaussianActor(observation_space=o, state_space=s, action_space=a, device="cpu", num_envs=N, cfg=_cfg())
    critic = LstmValueCritic(observation_space=o, state_space=s, action_space=a, device="cpu", num_envs=N, cfg=_cfg())
    return actor, critic


def _zero_rnn(model, batch):
    spec = model.get_specification()["rnn"]
    return [torch.zeros(size[0], batch, size[2]) for size in spec["sizes"]]


def test_specification_shapes():
    actor, critic = _models()
    for m in (actor, critic):
        spec = m.get_specification()["rnn"]
        assert spec["sequence_length"] == L
        assert spec["sizes"] == [(1, N, 8), (1, N, 8)]


def test_rollout_step_returns_actions_and_next_state():
    actor, critic = _models()
    actor.eval(), critic.eval()  # skrl collects rollouts with training mode off
    obs, states = torch.randn(N, OBS), torch.randn(N, STATE)
    actions, out = actor.act({"observations": obs, "states": states, "rnn": _zero_rnn(actor, N)}, role="policy")
    assert actions.shape == (N, ACT)
    assert out["log_prob"].shape == (N, 1)
    assert [t.shape for t in out["rnn"]] == [(1, N, 8), (1, N, 8)]
    assert float(actions.abs().max()) <= 1.0  # clipped to the action box
    values, vout = critic.act({"observations": obs, "states": states, "rnn": _zero_rnn(critic, N)}, role="value")
    assert values.shape == (N, 1)
    assert [t.shape for t in vout["rnn"]] == [(1, N, 8), (1, N, 8)]


def _stepwise(model, x_seq, key, done_seq, role):
    """Roll out step by step like skrl's collection loop, zeroing state after done."""
    rnn = _zero_rnn(model, x_seq.shape[0])
    outs, stored = [], []
    for t in range(x_seq.shape[1]):
        stored.append([h.clone() for h in rnn])
        inputs = {"observations": torch.zeros(x_seq.shape[0], OBS), "states": torch.zeros(x_seq.shape[0], STATE), "rnn": rnn}
        inputs[key] = x_seq[:, t]
        out, extra = model.compute(inputs, role=role)
        outs.append(out)
        rnn = [h.clone() for h in extra["rnn"]]
        ended = done_seq[:, t]
        for h in rnn:
            h[:, ended] = 0.0
    return torch.stack(outs, dim=1), stored, rnn


@pytest.mark.parametrize("which", ["actor", "critic"])
def test_sequence_mode_matches_stepwise_rollout_with_resets(which):
    """The training-time sequence forward must reproduce the rollout, including hidden-state
    resets at terminated/truncated steps in the middle of a stored sequence."""
    actor, critic = _models()
    model, key, role = (actor, "observations", "policy") if which == "actor" else (critic, "states", "value")
    width = OBS if which == "actor" else STATE
    g = torch.Generator().manual_seed(1)
    x = torch.randn(N, L, width, generator=g)
    terminated = torch.zeros(N, L, dtype=torch.bool)
    truncated = torch.zeros(N, L, dtype=torch.bool)
    terminated[1, 1] = True
    terminated[2, 3] = True
    truncated[:, 2] = False
    truncated[3, 2] = True

    model.eval()
    ref, stored, _ = _stepwise(model, x, key, terminated | truncated, role)

    # skrl training layout: (N*L, .) sequence-major per env, rnn from memory per step
    model.train()
    flat = x.reshape(N * L, width)
    rnn_mem = [torch.stack([s[i] for s in stored], dim=2).reshape(1, N * L, 8) for i in range(2)]
    inputs = {
        "observations": torch.zeros(N * L, OBS),
        "states": torch.zeros(N * L, STATE),
        "rnn": rnn_mem,
        "terminated": terminated.reshape(N * L, 1),
        "truncated": truncated.reshape(N * L, 1),
    }
    inputs[key] = flat
    out, _ = model.compute(inputs, role=role)
    torch.testing.assert_close(out.reshape(N, L, -1), ref, atol=1e-5, rtol=1e-5)


def test_gradients_reach_both_lstms():
    actor, critic = _models()
    actor.train(), critic.train()
    obs, states = torch.randn(N * L, OBS), torch.randn(N * L, STATE)
    common = {"terminated": torch.zeros(N * L, 1, dtype=torch.bool), "truncated": torch.zeros(N * L, 1, dtype=torch.bool)}
    _, a_out = actor.act({"observations": obs, "states": states, "rnn": _zero_rnn(actor, N * L), **common}, role="policy")
    v, _ = critic.act({"observations": obs, "states": states, "rnn": _zero_rnn(critic, N * L), **common}, role="value")
    (a_out["mean_actions"].sum() + v.sum()).backward()
    for m in (actor, critic):
        g = m.lstm.weight_ih_l0.grad
        assert g is not None and float(g.abs().sum()) > 0.0


def test_actor_ignores_critic_states():
    actor, _ = _models()
    actor.eval()
    obs = torch.randn(N, OBS)
    rnn = _zero_rnn(actor, N)
    a1, _ = actor.compute({"observations": obs, "states": torch.randn(N, STATE), "rnn": rnn}, role="policy")
    a2, _ = actor.compute({"observations": obs, "states": torch.full((N, STATE), 1e6), "rnn": rnn}, role="policy")
    torch.testing.assert_close(a1, a2)


def test_critic_reads_states_not_observations():
    _, critic = _models()
    critic.eval()
    states = torch.randn(N, STATE)
    rnn = _zero_rnn(critic, N)
    v1, _ = critic.compute({"observations": torch.randn(N, OBS), "states": states, "rnn": rnn}, role="value")
    v2, _ = critic.compute({"observations": torch.full((N, OBS), 1e6), "states": states, "rnn": rnn}, role="value")
    torch.testing.assert_close(v1, v2)


def test_initial_log_std_is_configurable_and_clipped():
    o, s, a = _spaces()
    actor = LstmGaussianActor(
        observation_space=o, state_space=s, action_space=a, device="cpu", num_envs=N, cfg=_cfg(initial_log_std=-0.5)
    )
    torch.testing.assert_close(actor.log_std_parameter.detach(), torch.full((ACT,), -0.5))
