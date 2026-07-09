"""Contact-trace models and scoring for contact-aware SSL planning."""
from __future__ import annotations

import torch
from torch import nn

from jepa_robotics.models import MLP


def possession_trust_barrier(
    contact_trace: torch.Tensor,
    target_trace: torch.Tensor,
    *,
    palm_ball: float,
    ball_target: float,
    contact_threshold: float = 0.06,
    keep_threshold: float = 0.09,
    target_threshold: float = 0.06,
    target_slack: float = 0.10,
    barrier: float = 10.0,
) -> torch.Tensor:
    """Mode-aware trust-region barrier over predicted contact traces.

    Candidate chunks are scored with a large barrier proportional to how far
    their predicted contact evolution exits the trust region of the current
    contact mode, instead of a weak additive distance term:

    - possession (palm-ball in contact, ball not yet at target): predicted
      possession loss anywhere in the chunk is barred, as is a predicted
      ball-target regression beyond ``target_slack``;
    - reach (no contact yet): chunks predicted to end farther from contact
      than the current palm-ball distance are barred;
    - placed (ball within ``target_threshold`` of target): no barrier, only
      the caller's settle scoring applies.

    ``contact_trace`` and ``target_trace`` are ``[N, H]`` predicted palm-ball
    and ball-target distances. Returns a ``[N]`` score to add (lower better).
    Because the barrier is proportional to violation, ranking degrades
    gracefully to "least violating" when every candidate violates.
    """

    if ball_target <= target_threshold:
        return torch.zeros(contact_trace.shape[0], dtype=contact_trace.dtype, device=contact_trace.device)
    if palm_ball <= contact_threshold:
        drop = torch.relu(contact_trace.max(dim=1).values - keep_threshold)
        fling = torch.relu(target_trace.max(dim=1).values - (ball_target + target_slack))
        return barrier * (drop + fling)
    approach = torch.relu(contact_trace[:, -1] - palm_ball)
    return barrier * approach


class ContactTraceCVAE(nn.Module):
    """Conditional VAE for multimodal contact traces.

    The condition is usually ``[z_t, raw_t, action_chunk]`` and the target is a
    flattened trace such as ``[palm_ball_1, ball_target_1, ...]``.
    """

    def __init__(self, cond_dim: int, trace_dim: int, latent_dim: int, hidden: int, n_blocks: int = 3) -> None:
        super().__init__()
        enc_sizes = [cond_dim + trace_dim] + [hidden] * max(1, n_blocks) + [2 * latent_dim]
        dec_sizes = [cond_dim + latent_dim] + [hidden] * max(1, n_blocks) + [trace_dim]
        self.encoder = MLP(enc_sizes, layer_norm=True)
        self.decoder = MLP(dec_sizes, layer_norm=True)
        self.latent_dim = latent_dim

    def encode(self, cond: torch.Tensor, trace: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.encoder(torch.cat([cond, trace], dim=-1))
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, torch.clamp(logvar, -8.0, 6.0)

    def decode(self, cond: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.decoder(torch.cat([cond, latent], dim=-1)), min=0.0)

    def forward(self, cond: torch.Tensor, trace: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(cond, trace)
        std = torch.exp(0.5 * logvar)
        latent = mu + std * torch.randn_like(std)
        return self.decode(cond, latent), mu, logvar

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n: int = 8) -> torch.Tensor:
        cond_rep = cond.repeat_interleave(n, dim=0)
        latent = torch.randn(cond_rep.shape[0], self.latent_dim, device=cond.device, dtype=cond.dtype)
        return self.decode(cond_rep, latent).view(cond.shape[0], n, -1)
