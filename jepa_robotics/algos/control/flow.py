"""Reusable rectified-flow networks for action-chunk control."""
from __future__ import annotations

import math

import torch
from torch import nn

from jepa_robotics.models import MLP


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    return torch.cat([torch.sin(t * freqs[None]), torch.cos(t * freqs[None])], dim=-1)


class FlowNet(nn.Module):
    """Conditional velocity field over flattened action chunks."""

    def __init__(self, chunk_dim: int, cond_dim: int, hidden: int = 512, t_dim: int = 128) -> None:
        super().__init__()
        self.t_dim = t_dim
        self.net = MLP([chunk_dim + t_dim + cond_dim, hidden, hidden, hidden, chunk_dim], layer_norm=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, timestep_embedding(t, self.t_dim), cond], dim=-1))

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, chunk_dim: int, n_steps: int = 10) -> torch.Tensor:
        x = torch.randn(cond.shape[0], chunk_dim, device=cond.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((cond.shape[0], 1), i * dt, device=cond.device)
            x = x + self(x, t, cond) * dt
        return x


class ChunkPolicy(nn.Module):
    """Deterministic, tanh-bounded action-chunk baseline."""

    def __init__(self, in_dim: int, action_dim: int, chunk: int, hidden: int) -> None:
        super().__init__()
        self.chunk = chunk
        self.action_dim = action_dim
        self.net = MLP([in_dim, hidden, hidden, action_dim * chunk], layer_norm=True)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(latent)).reshape(-1, self.chunk, self.action_dim)
