"""Reusable action-prior networks for SSL latent control."""
from __future__ import annotations

import math

import torch
from torch import nn


class InversePrior(nn.Module):
    """MLP inverse model mapping future-conditioned latents to action chunks."""

    def __init__(self, cond_dim: int, chunk_dim: int, hidden: int, n_blocks: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(cond_dim, hidden), nn.SiLU()]
        for _ in range(n_blocks - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, chunk_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class EpsNet(nn.Module):
    """Diffusion/flow action-chunk network conditioned on latent features."""

    def __init__(self, chunk_dim, cond_dim, hidden=512, t_dim=128, n_blocks=4):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.t_dim = t_dim
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.in_proj = nn.Linear(chunk_dim, hidden)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(hidden),
                        "cond": nn.Linear(2 * hidden, hidden),
                        "ff": nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)),
                    }
                )
            )
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, chunk_dim))

    def forward(self, a_noisy, t, cond):
        temb = self.t_mlp(sinusoidal_embedding(t, self.t_dim))
        cemb = self.cond_mlp(cond)
        ctx = torch.cat([temb, cemb], dim=-1)
        h = self.in_proj(a_noisy)
        for blk in self.blocks:
            x = blk["norm"](h)
            x = x + blk["cond"](ctx)
            h = h + blk["ff"](x)
        return self.out(h)


def make_ddpm(T, device):
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, 0)
    return {"betas": betas, "alphas": alphas, "abar": abar, "T": T}
