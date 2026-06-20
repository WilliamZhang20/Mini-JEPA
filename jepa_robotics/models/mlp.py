from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """A simple multi-layer perceptron with optional LayerNorm and SiLU activations."""

    def __init__(self, sizes: list[int], layer_norm: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                if layer_norm:
                    layers.append(nn.LayerNorm(sizes[i + 1]))
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
