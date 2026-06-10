from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F


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


class ActionConditionedJEPA(nn.Module):
    """Action-conditioned JEPA world model: encodes states to a latent and predicts future latents given actions."""

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
        elif predictor_mode == "recurrent":
            # Recurrent latent dynamics (Dreamer / DINO-WM style). A GRU cell
            # carries the latent through the rollout, which keeps many-step
            # predictions stable and lets the planner score every intermediate
            # state. A residual MLP refines each gated update.
            self.action_encoder = MLP([action_dim, hidden_dim, hidden_dim], layer_norm=True)
            self.gru = nn.GRUCell(hidden_dim + 1, latent_dim)
            self.transition = MLP(
                [latent_dim + hidden_dim + 1, hidden_dim, latent_dim],
                layer_norm=True,
            )
        else:
            raise ValueError(f"Unknown predictor_mode: {predictor_mode}")
        # A wider, two-hidden-layer state decoder: accurate geometry (gripper +
        # object positions) is what the manipulation-aware planner relies on.
        self.state_probe = MLP([latent_dim, hidden_dim, hidden_dim, state_dim], layer_norm=True)
        self.distance_probe = MLP([latent_dim, hidden_dim, 1])
        self.reset_target()

    def reset_target(self) -> None:
        """Copy the online encoder weights into the (frozen) target encoder."""
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update_target(self, ema: float) -> None:
        """Exponential-moving-average update of the target encoder toward the online encoder."""
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(ema).add_(online.data, alpha=1.0 - ema)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Map a (normalized) state to its online latent representation."""
        return self.encoder(state)

    @torch.no_grad()
    def encode_target(self, state: torch.Tensor) -> torch.Tensor:
        """Map a state to its latent using the frozen EMA target encoder (used for prediction targets)."""
        return self.target_encoder(state)

    def predict_rollout(self, z: torch.Tensor, action_seq: torch.Tensor, horizon: int) -> torch.Tensor:
        """Roll the latent dynamics forward and return every intermediate latent.

        Returns a tensor of shape ``[batch, horizon, latent_dim]``. Only valid
        for the ``rollout`` and ``recurrent`` predictor modes.
        """
        preds = []
        pred = z
        for i in range(horizon):
            step = torch.full(
                (action_seq.shape[0], 1),
                float(i + 1) / float(self.max_horizon),
                dtype=action_seq.dtype,
                device=action_seq.device,
            )
            action_emb = self.action_encoder(action_seq[:, i])
            if self.predictor_mode == "recurrent":
                gru_in = torch.cat([action_emb, step], dim=-1)
                pred = self.gru(gru_in, pred)
                pred = pred + self.transition(torch.cat([pred, action_emb, step], dim=-1))
            else:  # rollout
                pred = pred + self.transition(torch.cat([pred, action_emb, step], dim=-1))
            preds.append(pred)
        return torch.stack(preds, dim=1)

    def predict(self, z: torch.Tensor, action_seq: torch.Tensor, horizon: int) -> torch.Tensor:
        """Predict the latent ``horizon`` steps ahead given the action sequence.

        Uses the last step of a latent rollout for the ``rollout``/``recurrent``
        modes, or a single padded forward pass for the ``direct`` mode.
        """
        if self.predictor_mode in ("rollout", "recurrent"):
            return self.predict_rollout(z, action_seq, horizon)[:, -1]

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


class GoalConditionedPolicy(nn.Module):
    """A small action prior learned on the JEPA latent representation.

    The JEPA encoder maps the full observation (which already includes the
    desired goal) to a latent ``z``; this policy maps ``z`` to an action. It is
    trained by behaviour cloning on the collected trajectories, i.e. purely
    self-supervised from the same data the world model uses. It is not meant to
    be the final controller on its own - it is the *proposal* that the
    world-model MPC refines, which is what makes precise contact skills like
    grasping reliable.
    """

    def __init__(self, *, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = MLP([latent_dim, hidden_dim, hidden_dim, action_dim], layer_norm=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(z))


def normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between L2-normalized prediction and target latents (cosine-style JEPA loss)."""
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return F.mse_loss(pred, target)


def variance_regularizer(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Hinge penalty pushing each latent dimension's batch std toward 1, to prevent representation collapse."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))
