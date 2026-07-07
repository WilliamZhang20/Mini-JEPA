"""Same-latent hierarchical world model components.

This is the HWM-style interface: a high-level model predicts future states in the
same latent space as the frozen low-level JEPA, conditioned on learned
macro-actions encoded from primitive action chunks.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from jepa_robotics.models import MLP


class MacroActionEncoder(nn.Module):
    """Compress a primitive action chunk into one continuous macro-action."""

    def __init__(self, action_dim: int, macro_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.gru = nn.GRU(action_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, macro_dim)

    def forward(self, chunk: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(chunk)
        return self.head(h[-1])


class LatentMacroPredictor(nn.Module):
    """Residual high-level dynamics in the frozen low-level JEPA latent space."""

    def __init__(self, latent_dim: int, macro_dim: int, hidden: int = 512, n_blocks: int = 3) -> None:
        super().__init__()
        layers: list[int] = [latent_dim + macro_dim]
        layers.extend([hidden] * max(1, n_blocks))
        layers.append(latent_dim)
        self.net = MLP(layers, layer_norm=True)

    def forward(self, z: torch.Tensor, macro_action: torch.Tensor) -> torch.Tensor:
        return z + self.net(torch.cat([z, macro_action], dim=-1))


def sample_macro_dataset(episodes, stride: int, overlap: int = 2):
    step = max(1, stride // max(1, overlap))
    states, futures, chunks, starts, finals = [], [], [], [], []
    for ep in episodes:
        states_ep = ep.states.astype(np.float32)
        actions_ep = ep.actions.astype(np.float32)
        T = len(actions_ep)
        if T < stride:
            continue
        starts.append(states_ep[0])
        finals.append(states_ep[-1])
        for t in range(0, T - stride + 1, step):
            states.append(states_ep[t])
            futures.append(states_ep[t + stride])
            chunks.append(actions_ep[t : t + stride])
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(futures, dtype=np.float32),
        np.asarray(chunks, dtype=np.float32),
        np.asarray(starts, dtype=np.float32),
        np.asarray(finals, dtype=np.float32),
    )
