from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .evaluate import load_jepa_artifact


class JEPALatentExtractor(BaseFeaturesExtractor):
    """Frozen JEPA encoder as an SB3 feature extractor for goal-conditioned RL.

    SB3/HER keeps the original Dict observation space. This extractor flattens
    ``observation``, ``achieved_goal`` and ``desired_goal`` the same way the JEPA
    code does, applies the saved normalizer, and feeds the result through the
    frozen JEPA encoder. The RL actor/critic are still trained with real
    environment rewards, so this removes the BC/scripted-reference ceiling while
    reusing the JEPA representation.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        model_path: str | Path,
        device: str = "auto",
        layer_norm: bool = False,
    ) -> None:
        load_device = torch.device("cpu")
        model, normalizer, _spec, config = load_jepa_artifact(Path(model_path), load_device)
        features_dim = int(config["latent_dim"])
        super().__init__(observation_space, features_dim=features_dim)

        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.register_buffer("mean", torch.as_tensor(normalizer.mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(normalizer.std, dtype=torch.float32))
        self.requested_device = device
        self.layer_norm = layer_norm

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for key in ("observation", "achieved_goal", "desired_goal"):
            value = observations[key]
            parts.append(value.float().reshape(value.shape[0], -1))
        state = torch.cat(parts, dim=-1)
        state = (state - self.mean.to(state.device)) / self.std.to(state.device)
        with torch.no_grad():
            z = self.model.encode(state)
            if self.layer_norm:
                z = F.layer_norm(z, z.shape[-1:])
            return z

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["requested_device"] = "auto"
        return state
