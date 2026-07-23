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


class _ConditionalResidualBlock(nn.Module):
    """FiLM-modulated residual block for strongly state-dependent flow modes."""

    def __init__(self, hidden: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.modulation = nn.Linear(cond_dim, 2 * hidden)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1.0 + scale) + shift
        return x + self.net(h)


class ResidualFlowNet(nn.Module):
    """Condition-modulated residual flow for heterogeneous behavior manifolds.

    A single velocity field can represent qualitatively different action modes
    when they are separated by state (for example ordinary locomotion and
    self-righting). FiLM modulation injects the full state condition at every
    residual block instead of only at the input, avoiding a hand-written mode
    selector while retaining one shared action distribution.
    """

    def __init__(
        self,
        chunk_dim: int,
        cond_dim: int,
        hidden: int = 512,
        t_dim: int = 128,
        n_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.t_dim = t_dim
        self.cond_encoder = MLP([cond_dim, hidden, hidden], layer_norm=True)
        self.in_proj = nn.Linear(chunk_dim + t_dim, hidden)
        self.blocks = nn.ModuleList(
            _ConditionalResidualBlock(hidden, hidden) for _ in range(max(1, n_blocks))
        )
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, chunk_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        c = self.cond_encoder(cond)
        h = self.in_proj(torch.cat([x, timestep_embedding(t, self.t_dim)], dim=-1))
        for block in self.blocks:
            h = block(h, c)
        return self.out(torch.nn.functional.silu(self.out_norm(h)))

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, chunk_dim: int, n_steps: int = 10) -> torch.Tensor:
        x = torch.randn(cond.shape[0], chunk_dim, device=cond.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((cond.shape[0], 1), i * dt, device=cond.device)
            x = x + self(x, t, cond) * dt
        return x


def build_flow_net(
    chunk_dim: int,
    cond_dim: int,
    hidden: int,
    *,
    architecture: str = "mlp",
    n_blocks: int = 4,
) -> nn.Module:
    """Build a flow network while keeping old checkpoints loadable."""
    if architecture == "mlp":
        return FlowNet(chunk_dim, cond_dim, hidden)
    if architecture == "residual":
        return ResidualFlowNet(chunk_dim, cond_dim, hidden, n_blocks=n_blocks)
    raise ValueError(f"Unknown flow architecture: {architecture}")


class ChunkPolicy(nn.Module):
    """Deterministic, tanh-bounded action-chunk baseline."""

    def __init__(self, in_dim: int, action_dim: int, chunk: int, hidden: int) -> None:
        super().__init__()
        self.chunk = chunk
        self.action_dim = action_dim
        self.net = MLP([in_dim, hidden, hidden, action_dim * chunk], layer_norm=True)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(latent)).reshape(-1, self.chunk, self.action_dim)
