"""Transformer architectures strong enough for Shadow Dexterous Hand control.

The Adroit Shadow Hand tasks are 24-30 DoF, contact-rich, and multimodal (many
finger configurations realize the same subgoal). The shallow MLP
``ActionConditionedJEPA`` and MLP flow prior under-fit that regime: an MLP mixes
all joints/object/contact dims through dense layers with no relational inductive
bias, and the MLP flow prior mode-averages multimodal finger actions.

This module provides two drop-in-compatible upgrades:

* ``DexterousJEPA`` — a *tokenized* action-conditioned JEPA. Each state
  dimension is its own token, so self-attention explicitly models
  joint<->joint<->object<->contact interactions (the structure that matters for
  in-hand manipulation). A causal transformer carries the latent through the
  action chunk (better long-range contact credit than a GRU), an optional
  ensemble of dynamics heads gives an epistemic-disagreement signal for
  contact-uncertain regions, and a contact-consistency head (DexWM-style) forces
  the latent to retain fingertip/relative-geometry detail. It exposes the same
  interface as ``ActionConditionedJEPA`` (encode / encode_target / update_target
  / reset_target / predict_rollout / predict / disagreement / state_probe) so it
  loads through ``load_jepa_artifact`` and runs under the existing planners.

* ``DexterousFlowPrior`` — a DiT (diffusion/flow transformer) action-chunk
  prior. Each action step is a token; conditioning ``(z_t, z_future)`` + flow
  time modulate every block via AdaLN-Zero. This models the multimodal
  distribution over dexterous action chunks instead of averaging it, and matches
  the ``EpsNet(x, t, cond) -> velocity`` signature so it slots into the shared
  ``sample_action_chunks`` sampler.
"""
from __future__ import annotations

import math
from copy import deepcopy

import torch
from torch import nn


def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(1, half - 1))
    args = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:  # odd dim pad
        emb = torch.cat([emb, torch.zeros(emb.shape[0], dim - emb.shape[-1], device=t.device)], dim=-1)
    return emb


class _TokenEncoder(nn.Module):
    """Per-dimension token transformer: state scalars -> latent.

    Each of the D state dims becomes a token (shared value projection + a learned
    per-dim embedding that carries which joint/object/contact channel it is), plus
    a learned [LATENT] summary token whose output is projected to the latent.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        d_model: int,
        depth: int,
        heads: int,
        dropout: float,
        token_groups: tuple[tuple[int, int], ...] | None = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.token_groups = tuple(token_groups or ())
        grouped = {
            index
            for lo, hi in self.token_groups
            for index in range(lo, hi)
        }
        if any(lo < 0 or hi > state_dim or lo >= hi for lo, hi in self.token_groups):
            raise ValueError(f"Invalid token_groups={self.token_groups} for state_dim={state_dim}")
        if sum(hi - lo for lo, hi in self.token_groups) != len(grouped):
            raise ValueError(f"token_groups must not overlap: {self.token_groups}")
        self.scalar_indices = tuple(index for index in range(state_dim) if index not in grouped)
        self.value = nn.Linear(1, d_model)
        self.dim_emb = nn.Parameter(torch.randn(len(self.scalar_indices), d_model) * 0.02)
        if self.token_groups:
            self.group_value = nn.ModuleList(
                nn.Linear(hi - lo, d_model) for lo, hi in self.token_groups
            )
            self.group_emb = nn.Parameter(torch.randn(len(self.token_groups), d_model) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, latent_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        b = state.shape[0]
        scalar = state[:, self.scalar_indices]
        tok = self.value(scalar.unsqueeze(-1)) + self.dim_emb.unsqueeze(0)
        if self.token_groups:
            group_tok = torch.stack(
                [
                    projection(state[:, lo:hi])
                    for projection, (lo, hi) in zip(self.group_value, self.token_groups)
                ],
                dim=1,
            )
            tok = torch.cat([tok, group_tok + self.group_emb.unsqueeze(0)], dim=1)
        tok = torch.cat([self.cls.expand(b, -1, -1), tok], dim=1)          # [b, 1+D, d]
        out = self.tr(tok)
        return self.head(self.norm(out[:, 0]))


class _DynamicsHead(nn.Module):
    """Causal transformer latent dynamics over an action chunk.

    Sequence = [latent token, action_0, ..., action_{H-1}]; a causal mask makes
    the output at action step i depend only on the latent and a_0..a_i, so it
    predicts the latent AFTER applying a_0..a_i. Returns residual per-step
    latents; the caller adds them to z0 for absolute latents.
    """

    def __init__(self, latent_dim: int, action_dim: int, d_model: int, depth: int, heads: int,
                 max_horizon: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.z_in = nn.Linear(latent_dim, d_model)
        self.a_in = nn.Linear(action_dim, d_model)
        self.step_emb = nn.Parameter(torch.randn(max_horizon + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, latent_dim)

    def forward(self, z: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        b, h, _ = action_seq.shape
        z_tok = self.z_in(z).unsqueeze(1)                      # [b,1,d]
        a_tok = self.a_in(action_seq)                          # [b,h,d]
        seq = torch.cat([z_tok, a_tok], dim=1) + self.step_emb[: h + 1].unsqueeze(0)
        mask = torch.triu(torch.ones(h + 1, h + 1, device=z.device, dtype=torch.bool), diagonal=1)
        out = self.tr(seq, mask=mask)                          # [b,1+h,d]
        deltas = self.head(self.norm(out[:, 1:]))              # [b,h,latent] residuals
        return z.unsqueeze(1) + deltas


class DexterousJEPA(nn.Module):
    """Tokenized transformer action-conditioned JEPA for dexterous hands.

    Interface-compatible with ``ActionConditionedJEPA``.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        d_model: int = 256,
        enc_depth: int = 4,
        dyn_depth: int = 4,
        heads: int = 8,
        max_horizon: int = 16,
        ensemble_heads: int = 1,
        dropout: float = 0.0,
        contact_dims: tuple[int, int] | None = None,
        token_groups: tuple[tuple[int, int], ...] | None = None,
        hidden_dim: int | None = None,  # accepted for config compatibility; unused
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.max_horizon = max_horizon
        self.ensemble_heads = max(1, ensemble_heads)
        self.contact_dims = contact_dims
        self.token_groups = tuple(token_groups or ())

        self.encoder = _TokenEncoder(
            state_dim, latent_dim, d_model, enc_depth, heads, dropout, self.token_groups
        )
        self.target_encoder = deepcopy(self.encoder)
        self.dyn_heads = nn.ModuleList(
            _DynamicsHead(latent_dim, action_dim, d_model, dyn_depth, heads, max_horizon, dropout)
            for _ in range(self.ensemble_heads)
        )
        # State + contact-consistency decoders (DexWM-style): full state, plus an
        # emphasized contact/relative-geometry slice so the latent keeps the
        # fingertip/object detail contact control depends on.
        self.state_probe = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim), nn.GELU(), nn.Linear(2 * latent_dim, state_dim),
        )
        self.distance_probe = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, 1))
        if contact_dims is not None:
            lo, hi = contact_dims
            self.contact_probe = nn.Sequential(
                nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, hi - lo),
            )
        self.reset_target()

    # --- encoder interface ---
    def reset_target(self) -> None:
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_target(self, ema: float) -> None:
        for o, t in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            t.data.mul_(ema).add_(o.data, alpha=1.0 - ema)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    @torch.no_grad()
    def encode_target(self, state: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(state)

    # --- dynamics interface ---
    def rollout_heads(self, z: torch.Tensor, action_seq: torch.Tensor, horizon: int) -> torch.Tensor:
        a = action_seq[:, :horizon]
        return torch.stack([head(z, a) for head in self.dyn_heads], dim=0)  # [K,b,h,latent]

    def predict_rollout(self, z: torch.Tensor, action_seq: torch.Tensor, horizon: int) -> torch.Tensor:
        return self.rollout_heads(z, action_seq, horizon).mean(dim=0)

    def predict(self, z: torch.Tensor, action_seq: torch.Tensor, horizon: int) -> torch.Tensor:
        return self.predict_rollout(z, action_seq, horizon)[:, -1]

    def disagreement(self, z: torch.Tensor, action_seq: torch.Tensor, horizon: int) -> torch.Tensor:
        if self.ensemble_heads <= 1:
            return torch.zeros((), device=z.device, dtype=z.dtype)
        return self.rollout_heads(z, action_seq, horizon).var(dim=0).mean()

    def contact_consistency(self, z: torch.Tensor) -> torch.Tensor | None:
        return self.contact_probe(z) if self.contact_dims is not None else None


# ---------------------------------------------------------------------------
# DiT action-chunk flow prior
# ---------------------------------------------------------------------------
class _AdaLNBlock(nn.Module):
    """DiT block with AdaLN-Zero conditioning (attention + MLP, gated by cond)."""

    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)  # AdaLN-Zero

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        sa1, sb1, g1, sa2, sb2, g2 = self.ada(c).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + sa1) + sb1
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1 * h
        h = self.norm2(x) * (1 + sa2) + sb2
        return x + g2 * self.mlp(h)


class DexterousFlowPrior(nn.Module):
    """DiT flow prior over action chunks; ``forward(x, t, cond) -> velocity``.

    ``x`` is a flattened chunk ``[B, H*action_dim]`` (matches ``EpsNet``); each
    action step is a token, conditioned on ``(z_t, z_future)`` + flow time via
    AdaLN-Zero, so multimodal dexterous chunks are modeled rather than averaged.
    """

    def __init__(self, chunk_dim: int, cond_dim: int, hidden: int = 384, n_blocks: int = 6,
                 heads: int = 6, action_dim: int = 30, t_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        assert chunk_dim % action_dim == 0, "chunk_dim must be H*action_dim"
        self.H = chunk_dim // action_dim
        self.action_dim = action_dim
        self.d = hidden
        self.a_in = nn.Linear(action_dim, hidden)
        self.pos = nn.Parameter(torch.randn(self.H, hidden) * 0.02)
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList(_AdaLNBlock(hidden, heads, dropout) for _ in range(n_blocks))
        self.norm_out = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        nn.init.zeros_(self.ada_out[-1].weight); nn.init.zeros_(self.ada_out[-1].bias)
        self.out = nn.Linear(hidden, action_dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        tok = self.a_in(x.view(b, self.H, self.action_dim)) + self.pos.unsqueeze(0)
        c = self.t_mlp(_sinusoidal(t, self.t_dim)) + self.cond_mlp(cond)
        for blk in self.blocks:
            tok = blk(tok, c)
        sa, sb = self.ada_out(c).unsqueeze(1).chunk(2, dim=-1)
        tok = self.norm_out(tok) * (1 + sa) + sb
        return self.out(tok).reshape(b, self.H * self.action_dim)
