"""Learned task-completion predicates for hierarchical control."""
from __future__ import annotations

import torch
from torch import nn


class LatentCompletionProbe(nn.Module):
    """Predict monotone subtask-completion predicates from a frozen JEPA latent.

    The probe is intentionally small.  It turns demonstration progression into
    the high-level switch signal, replacing privileged environment completion
    events at runtime while leaving the JEPA encoder frozen.
    """

    def __init__(self, input_dim: int, num_tasks: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_tasks),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
