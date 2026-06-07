from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
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


class ActionConditionedJEPA(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int,
        max_horizon: int,
        predictor_mode: str = "direct",
        residual_prediction: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.max_horizon = max_horizon
        self.predictor_mode = predictor_mode
        self.residual_prediction = residual_prediction

        self.encoder = MLP([state_dim, hidden_dim, hidden_dim, latent_dim], layer_norm=True)
        self.target_encoder = deepcopy(self.encoder)
        if predictor_mode == "direct":
            self.predictor = MLP(
                [latent_dim + max_horizon * action_dim + 1, hidden_dim, hidden_dim, latent_dim],
                layer_norm=True,
            )
        elif predictor_mode == "rollout":
            self.action_encoder = MLP([action_dim, hidden_dim, hidden_dim], layer_norm=True)
            self.transition = MLP(
                [latent_dim + hidden_dim + 1, hidden_dim, hidden_dim, latent_dim],
                layer_norm=True,
            )
        else:
            raise ValueError(f"Unknown predictor_mode: {predictor_mode}")
        self.state_probe = MLP([latent_dim, hidden_dim, state_dim])
        self.distance_probe = MLP([latent_dim, hidden_dim, 1])
        self.reset_target()

    def reset_target(self) -> None:
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update_target(self, ema: float) -> None:
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(ema).add_(online.data, alpha=1.0 - ema)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    @torch.no_grad()
    def encode_target(self, state: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(state)

    def predict(self, z: torch.Tensor, action_seq: torch.Tensor, horizon: int) -> torch.Tensor:
        if self.predictor_mode == "rollout":
            pred = z
            for i in range(horizon):
                step = torch.full(
                    (action_seq.shape[0], 1),
                    float(i + 1) / float(self.max_horizon),
                    dtype=action_seq.dtype,
                    device=action_seq.device,
                )
                action_emb = self.action_encoder(action_seq[:, i])
                pred = pred + self.transition(torch.cat([pred, action_emb, step], dim=-1))
            return pred

        batch = action_seq.shape[0]
        padded = torch.zeros(
            batch,
            self.max_horizon,
            self.action_dim,
            dtype=action_seq.dtype,
            device=action_seq.device,
        )
        padded[:, :horizon] = action_seq[:, :horizon]
        horizon_token = torch.full(
            (batch, 1),
            float(horizon) / float(self.max_horizon),
            dtype=action_seq.dtype,
            device=action_seq.device,
        )
        pred = self.predictor(torch.cat([z, padded.flatten(1), horizon_token], dim=-1))
        if self.residual_prediction:
            pred = z + pred
        return pred


def normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return F.mse_loss(pred, target)


def variance_regularizer(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))
