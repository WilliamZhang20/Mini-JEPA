"""Reusable action-prior networks for SSL latent control."""
from __future__ import annotations

import math

import torch
from torch import nn


def parse_horizons(value: str) -> list[int]:
    """Parse a comma-separated positive horizon set for training CLIs."""
    horizons = sorted({int(v) for v in value.split(",") if v.strip()})
    if not horizons or min(horizons) < 1:
        raise ValueError("future horizons must be positive, e.g. 4,8,12,16")
    return horizons


def append_emphasis(parts: list, state: torch.Tensor, checkpoint: dict) -> None:
    """Append a checkpoint-declared repeated live-state feature slice."""
    dims = checkpoint.get("emphasis_dims")
    repeat = int(checkpoint.get("emphasis_repeat", 0) or 0)
    if dims and repeat > 0:
        lo, hi = (int(x) for x in dims.split(","))
        parts.append(state[:, lo:hi].repeat(1, repeat))


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


class FlowChunkActor(nn.Module):
    """Rectified-flow prior over future-conditioned action chunks.

    A deterministic inverse actor collapses a multimodal chunk distribution to
    its conditional mean, which is exactly what fails on in-hand reorientation
    (see the pen ledger entries). Sampling this prior yields genuinely diverse
    on-manifold candidates for the JEPA predictor to rank, rather than one
    proposal plus Gaussian jitter. The condition stays ``(z_t, z_goal,
    horizon)`` only — no hand-supplied structure.
    """

    def __init__(
        self,
        cond_dim: int,
        chunk_dim: int,
        hidden: int = 512,
        n_blocks: int = 4,
        flow_T: int = 1000,
        flow_steps: int = 16,
    ) -> None:
        super().__init__()
        self.net = EpsNet(chunk_dim, cond_dim, hidden=hidden, n_blocks=n_blocks)
        self.chunk_dim = chunk_dim
        self.flow_T = flow_T
        self.flow_steps = flow_steps

    def loss(self, chunk: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Rectified-flow matching loss on one batch of (chunk, cond) rows."""
        noise = torch.randn_like(chunk)
        tau = torch.rand(chunk.shape[0], device=chunk.device)
        x = (1.0 - tau[:, None]) * noise + tau[:, None] * chunk
        velocity = self.net(x, tau * self.flow_T, cond)
        return nn.functional.mse_loss(velocity, chunk - noise)

    @torch.no_grad()
    def sample(
        self, cond: torch.Tensor, n: int = 1, init_noise_scale: float = 1.0
    ) -> torch.Tensor:
        """Draw ``n`` chunks per cond row; returns ``[rows * n, chunk_dim]``.

        ``init_noise_scale`` is a sampling temperature: 1.0 covers the full
        conditional distribution, lower values concentrate samples near the
        mode. Precision-critical tasks (measured: hammer) collapse under
        full-noise samples but still benefit from mild diversity.
        """
        rep = cond.repeat_interleave(n, dim=0)
        return sample_action_chunks(
            self.net,
            {"T": self.flow_T},
            rep,
            self.chunk_dim,
            rep.device,
            objective="flow",
            flow_steps=self.flow_steps,
            init_noise_scale=init_noise_scale,
        )


class PredictorGuidedRefiner(nn.Module):
    """Amortized chunk correction driven by the predictor's own rollout error.

    Ranking lets the world model *select* among proposals but never improve
    them; free gradient descent improves them but exploits predictor error
    off-manifold (measured on hammer: it "beats" ground-truth chunks that
    reality cannot beat). This head is the middle path: the chunk is rolled
    through the frozen predictor, the latent goal error is fed back in, and
    the head outputs a correction *trained to land on the demonstrated
    chunk*. The world model thereby generates actions from its own
    evaluation, but the update direction comes from the demo manifold rather
    than from cost gradients, so iterating it cannot chase predictor error.
    """

    def __init__(
        self, latent_dim: int, chunk_dim: int, hidden: int = 512, n_blocks: int = 4
    ) -> None:
        super().__init__()
        in_dim = 3 * latent_dim + 1 + chunk_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_blocks - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, chunk_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        z_goal: torch.Tensor,
        z_err: torch.Tensor,
        h_token: torch.Tensor,
        chunk_flat: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([z, z_goal, z_err, h_token, chunk_flat], dim=-1))

    @torch.no_grad()
    def refine(
        self,
        wm,
        z: torch.Tensor,
        z_goal: torch.Tensor,
        h_token: torch.Tensor,
        chunk: torch.Tensor,
        steps: int = 3,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ) -> torch.Tensor:
        """Iteratively correct ``chunk [n, k, act]`` toward ``z_goal``, feeding
        the world model's own rollout endpoint back in at every step."""
        n, k, _ = chunk.shape
        for _ in range(steps):
            z_end = wm.rollout_heads(z, chunk, k).mean(dim=0)[:, -1]
            delta = self.forward(z, z_goal, z_goal - z_end, h_token, chunk.reshape(n, -1))
            chunk = (chunk + delta.view_as(chunk)).clamp(action_low, action_high)
        return chunk


def sample_chunk(
    net: nn.Module,
    ddpm: dict,
    cond: torch.Tensor,
    chunk_dim: int,
    device,
    objective: str = "diffusion",
    flow_steps: int = 16,
    cfg_weight: float = 1.0,
    init_noise_scale: float = 1.0,
) -> torch.Tensor:
    """Compatibility spelling for the shared action-chunk sampler.

    Unlike the old evaluator-local implementation, this lives beside the
    network and diffusion schedule used by both training and evaluation.
    """
    return sample_action_chunks(
        net,
        ddpm,
        cond,
        chunk_dim,
        device,
        objective=objective,
        flow_steps=flow_steps,
        cfg_weight=cfg_weight,
        init_noise_scale=init_noise_scale,
    )
