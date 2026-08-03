"""Learned high level: predict the desirable next latent instead of scheduling it.

The phase-inverse controllers supply their temporal structure by hand. The
number of phases is a flag, the phase index is ``floor(step / max_steps *
n_phases)`` — a wall-clock schedule that knows nothing about what the hand is
actually doing — and the subgoal is a nearest-neighbour lookup into a stored
bank of demo states, kept monotone by an explicit rule.

This module learns that structure instead. A small network reads the current
JEPA latent and emits

* ``z_goal``  — the latent the demonstrations move to next, ``h`` steps ahead;
* ``state_goal`` — its decoded state, which grounds the planner's geometry term;
* ``progress`` — where in the task this latent sits, supervised only by the
  demo's own normalized time index.

Nothing here is task-specific: the only supervision is "this is what the
demonstrations did next", which is the same evidence the phase schedule was
approximating. ``progress`` replaces the hand-set phase counter, and ``z_goal``
replaces the demo bank plus its monotonicity rule.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..models import MLP


class LatentSubgoalNet(nn.Module):
    """``z_t -> (z_{t+h}, decoded state, progress)`` for a horizon-conditioned h."""

    def __init__(
        self,
        *,
        latent_dim: int,
        state_dim: int,
        hidden: int = 512,
        n_blocks: int = 3,
        max_horizon: int = 16,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.state_dim = int(state_dim)
        self.max_horizon = int(max_horizon)
        self.stem = nn.Sequential(nn.Linear(latent_dim + 1, hidden), nn.SiLU())
        self.blocks = nn.ModuleList(
            MLP([hidden, hidden, hidden], layer_norm=True) for _ in range(n_blocks)
        )
        self.to_latent = nn.Linear(hidden, latent_dim)
        self.to_state = nn.Linear(hidden, state_dim)
        self.to_progress = nn.Linear(hidden, 1)

    def forward(self, z: torch.Tensor, horizon: torch.Tensor):
        h = self.stem(torch.cat([z, horizon / float(self.max_horizon)], dim=-1))
        for block in self.blocks:
            h = h + block(h)
        return self.to_latent(h), self.to_state(h), self.to_progress(h)

    @torch.no_grad()
    def subgoal(self, z: torch.Tensor, horizon: int):
        """Convenience wrapper returning ``(z_goal, state_goal, progress)`` for a scalar horizon."""
        token = torch.full((z.shape[0], 1), float(horizon), device=z.device, dtype=z.dtype)
        z_goal, state_goal, progress = self.forward(z, token)
        return z_goal, state_goal, torch.sigmoid(progress).squeeze(-1)


def subgoal_loss(
    net: LatentSubgoalNet,
    z: torch.Tensor,
    horizon: torch.Tensor,
    z_target: torch.Tensor,
    state_target: torch.Tensor,
    progress_target: torch.Tensor,
    *,
    state_weight: float = 1.0,
    progress_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    z_pred, state_pred, progress_pred = net(z, horizon)
    loss_latent = F.mse_loss(F.normalize(z_pred, dim=-1), F.normalize(z_target, dim=-1))
    loss_state = F.mse_loss(state_pred, state_target)
    loss_progress = F.binary_cross_entropy_with_logits(
        progress_pred.squeeze(-1), progress_target
    )
    total = loss_latent + state_weight * loss_state + progress_weight * loss_progress
    return total, {
        "loss": float(total.detach()),
        "latent": float(loss_latent.detach()),
        "state": float(loss_state.detach()),
        "progress": float(loss_progress.detach()),
    }


def build_subgoal_windows(episodes, normalizer, horizons: list[int]):
    """Flatten demos into ``(state, horizon, future_state, progress)`` training rows."""
    states, horizon_col, futures, progress = [], [], [], []
    for ep in episodes:
        raw = np.asarray(ep.states, dtype=np.float32)
        n = len(raw)
        if n < max(horizons) + 2:
            continue
        norm = normalizer.encode(raw)
        for h in horizons:
            idx = np.arange(n - h, dtype=np.int64)
            states.append(norm[idx])
            futures.append(norm[idx + h])
            horizon_col.append(np.full(len(idx), float(h), dtype=np.float32))
            progress.append((idx.astype(np.float32) / float(max(1, n - 1))))
    return (
        np.concatenate(states),
        np.concatenate(horizon_col)[:, None],
        np.concatenate(futures),
        np.concatenate(progress),
    )
