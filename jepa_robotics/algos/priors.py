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


@torch.no_grad()
def sample_action_chunks(
    net: nn.Module,
    ddpm: dict,
    cond: torch.Tensor,
    chunk_dim: int,
    device,
    *,
    objective: str = "diffusion",
    flow_steps: int = 16,
    cfg_weight: float = 1.0,
    init_noise_scale: float = 1.0,
    warm_init: torch.Tensor | None = None,
    warm_tau: float = 0.0,
) -> torch.Tensor:
    """Sample action chunks from a diffusion or rectified-flow prior.

    The same sampler is used for primitive action priors and residual refiners.
    ``objective="flow"`` integrates a velocity field from noise to data;
    ``objective="diffusion"`` runs the DDPM reverse process.

    ``warm_init`` (flow objective only) warm-starts sampling from a previous
    plan for receding-horizon temporal coherence: integration begins at time
    ``warm_tau`` in (0, 1) from the rectified-flow interpolation
    ``(1 - warm_tau) * noise + warm_tau * warm_init``, so samples stay near
    the shifted previous chunk while re-noising enough to adapt.
    """

    B = cond.shape[0]
    guided = cfg_weight != 1.0
    null = torch.zeros_like(cond) if guided else None

    def predict(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = net(x, t, cond)
        if guided:
            out_u = net(x, t, null)
            out = out_u + cfg_weight * (out - out_u)
        return out

    if objective == "flow":
        T = ddpm["T"]
        noise = torch.randn(B, chunk_dim, device=device) * init_noise_scale
        if warm_init is not None and warm_tau > 0.0:
            tau0 = min(max(float(warm_tau), 0.0), 0.999)
            x = (1.0 - tau0) * noise + tau0 * warm_init
        else:
            tau0 = 0.0
            x = noise
        n_steps = max(1, flow_steps)
        dt = (1.0 - tau0) / float(n_steps)
        for i in range(n_steps):
            tau = torch.full((B,), tau0 + i * dt, device=device)
            x = x + dt * predict(x, tau * T)
        return x

    betas, alphas, abar, T = ddpm["betas"], ddpm["alphas"], ddpm["abar"], ddpm["T"]
    a = torch.randn(B, chunk_dim, device=device) * init_noise_scale
    for t in reversed(range(int(T))):
        tt = torch.full((B,), t, device=device, dtype=torch.long)
        eps = predict(a, tt)
        mean = (a - betas[t] / torch.sqrt(1 - abar[t]) * eps) / torch.sqrt(alphas[t])
        a = mean + torch.sqrt(betas[t]) * torch.randn_like(a) if t > 0 else mean
    return a
