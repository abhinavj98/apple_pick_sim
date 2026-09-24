"""LSTM actor and privileged LSTM critic for skrl 2.1 ``PPO_RNN``.

- :class:`LstmGaussianActor` reads **only** ``inputs["observations"]`` (the 40-D
  proprioception + F/T actor vector) -- it never touches the critic state, so
  privileged information cannot reach the deployed policy.
- :class:`LstmValueCritic` reads **only** ``inputs["states"]`` (the privileged
  critic vector, which already contains the actor vector as a prefix).

Both are ``pre-MLP -> LSTM -> post-MLP -> head``. They are separate networks
(not shared) because their inputs differ.

**skrl 2.1 recurrent contract** (read from ``skrl/agents/torch/ppo/ppo_rnn.py``):

- rollout (``model.training == False``): one step per env; ``inputs["rnn"]`` is
  ``[h, c]`` each ``(layers, N, H)``; the returned ``outputs["rnn"]`` becomes the
  next step's state (the agent zeroes envs whose episode ended);
- update (``model.training == True``): the minibatch is sequence-major per env,
  ``(B*L, F)``; ``inputs["rnn"]`` holds the *stored* per-step initial states
  ``(layers, B*L, H)`` of which only each sequence's first step is used, and
  ``inputs["terminated"]`` / ``["truncated"]`` mark episode ends *inside* the
  sequence, after which the state must be zeroed exactly as during the rollout.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch import nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


@dataclasses.dataclass(frozen=True)
class RecurrentNetConfig:
    """Sizes of one recurrent tower plus the actor's initial exploration std."""

    pre_mlp: tuple[int, ...] = (256,)
    lstm_hidden: int = 256
    lstm_layers: int = 1
    post_mlp: tuple[int, ...] = (256, 128)
    sequence_length: int = 32
    initial_log_std: float = -0.7
    min_log_std: float = -5.0
    max_log_std: float = 0.5
    # [D16b] actor mean = mean_bound * tanh(x / mean_bound): bounded just beyond the [-1, 1] box
    mean_bound: float | None = 1.5  # None: unbounded (D15 behaviour)


def _mlp(in_dim: int, sizes: tuple[int, ...]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    d = in_dim
    for h in sizes:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    return nn.Sequential(*layers), d


class _RecurrentTower(nn.Module):
    """pre-MLP -> LSTM -> post-MLP with skrl's sequence / rollout handling."""

    def __init__(self, in_dim: int, cfg: RecurrentNetConfig, num_envs: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.pre, d = _mlp(in_dim, cfg.pre_mlp)
        self.lstm = nn.LSTM(input_size=d, hidden_size=cfg.lstm_hidden, num_layers=cfg.lstm_layers, batch_first=True)
        self.post, self.out_dim = _mlp(cfg.lstm_hidden, cfg.post_mlp)

    def rnn_sizes(self) -> list[tuple[int, int, int]]:
        size = (self.cfg.lstm_layers, self.num_envs, self.cfg.lstm_hidden)
        return [size, size]

    def forward(self, x: torch.Tensor, inputs: dict[str, Any], training: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
        feats = self.pre(x)
        h, c = inputs["rnn"][0], inputs["rnn"][1]
        if not training:
            out, (h, c) = self.lstm(feats.unsqueeze(1), (h, c))
            return self.post(out.squeeze(1)), [h, c]

        seq = self.cfg.sequence_length
        feats = feats.view(-1, seq, feats.shape[-1])
        n_seq = feats.shape[0]
        h = h.view(self.cfg.lstm_layers, n_seq, seq, h.shape[-1])[:, :, 0, :].contiguous()
        c = c.view(self.cfg.lstm_layers, n_seq, seq, c.shape[-1])[:, :, 0, :].contiguous()

        ended = None
        for key in ("terminated", "truncated"):
            flag = inputs.get(key)
            if flag is not None:
                flag = flag.view(n_seq, seq).bool()
                ended = flag if ended is None else (ended | flag)

        if ended is None or not bool(ended[:, :-1].any()):
            out, (h, c) = self.lstm(feats, (h, c))
        else:
            # Split the sequence after every step at which any env's episode ended and zero
            # those envs' state -- the same reset the rollout applied between those steps.
            cuts = (ended[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
            bounds = [0] + cuts + [seq]
            outs = []
            for i0, i1 in zip(bounds[:-1], bounds[1:]):
                o, (h, c) = self.lstm(feats[:, i0:i1, :], (h, c))
                outs.append(o)
                if i1 < seq:
                    keep = (~ended[:, i1 - 1]).to(h.dtype).view(1, n_seq, 1)
                    h, c = h * keep, c * keep
            out = torch.cat(outs, dim=1)
        return self.post(out.reshape(-1, out.shape[-1])), [h, c]


class LstmGaussianActor(GaussianMixin, Model):
    """Recurrent Gaussian policy over ``inputs["observations"]``; state-independent log std."""

    def __init__(self, *, observation_space, state_space, action_space, device, num_envs: int, cfg: RecurrentNetConfig):
        Model.__init__(
            self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device
        )
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_log_std=True,
            min_log_std=cfg.min_log_std,
            max_log_std=cfg.max_log_std,
            reduction="sum",
        )
        self.cfg = cfg
        self.tower = _RecurrentTower(self.num_observations, cfg, num_envs)
        self.mean_head = nn.Linear(self.tower.out_dim, self.num_actions)
        with torch.no_grad():  # start near the action-box centre with a small spread
            self.mean_head.weight.mul_(0.01)
            self.mean_head.bias.zero_()
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), float(cfg.initial_log_std)))

    @property
    def lstm(self) -> nn.LSTM:
        return self.tower.lstm

    def get_specification(self) -> dict[str, Any]:
        return {"rnn": {"sequence_length": self.cfg.sequence_length, "sizes": self.tower.rnn_sizes()}}

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        feats, rnn = self.tower(inputs["observations"], inputs, self.training)
        # [D16/D16b] bounded mean. Unbounded, a mean far outside the [-1, 1] box put clipped actions
        # deep in the Gaussian tail, where log-probs swing by nats for tiny parameter steps (GPU KL
        # spikes of 8-7080 with exact stored data). A bound of exactly 1 (plain tanh) needs a
        # saturated tanh to command full-rate actions -- the D16 run drifted passive. 1.5 keeps edge
        # actions easy to sample and the clipped action within ~1.1 std of the mean.
        raw = self.mean_head(feats)
        if self.cfg.mean_bound is None:
            mean = raw
        else:
            b = float(self.cfg.mean_bound)
            mean = b * torch.tanh(raw / b)
        return mean, {"log_std": self.log_std_parameter.expand_as(mean), "rnn": rnn}


class LstmValueCritic(DeterministicMixin, Model):
    """Recurrent value function over the privileged ``inputs["states"]``."""

    def __init__(self, *, observation_space, state_space, action_space, device, num_envs: int, cfg: RecurrentNetConfig):
        Model.__init__(
            self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device
        )
        DeterministicMixin.__init__(self, clip_actions=False)
        self.cfg = cfg
        self.tower = _RecurrentTower(self.num_states, cfg, num_envs)
        self.value_head = nn.Linear(self.tower.out_dim, 1)

    @property
    def lstm(self) -> nn.LSTM:
        return self.tower.lstm

    def get_specification(self) -> dict[str, Any]:
        return {"rnn": {"sequence_length": self.cfg.sequence_length, "sizes": self.tower.rnn_sizes()}}

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        feats, rnn = self.tower(inputs["states"], inputs, self.training)
        return self.value_head(feats), {"rnn": rnn}
