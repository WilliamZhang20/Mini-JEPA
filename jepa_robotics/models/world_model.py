from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from .mlp import MLP


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
        transition_depth: int = 1,
        ensemble_heads: int = 1,
        inverse_dynamics: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.max_horizon = max_horizon
        self.predictor_mode = predictor_mode
        self.residual_prediction = residual_prediction
        self.transition_depth = max(1, transition_depth)
        self.ensemble_heads = max(1, ensemble_heads)

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
            # state. ``transition_depth`` stacked residual MLP blocks then refine
            # the gated update; depth > 1 adds capacity for the strongly
            # nonlinear, ballistic dynamics of contact/striking tasks (e.g.
            # FetchSlide, where the puck coasts long after the strike).
            self.action_encoder = MLP([action_dim, hidden_dim, hidden_dim], layer_norm=True)
            self.gru = nn.GRUCell(hidden_dim + 1, latent_dim)
            self.transition_blocks = nn.ModuleList(
                MLP([latent_dim + hidden_dim + 1, hidden_dim, latent_dim], layer_norm=True)
                for _ in range(self.transition_depth)
            )
            # Roadmap A (item 2): ensemble dynamics. Extra independent recurrent
            # heads (sharing only the action encoder) give an inter-head
            # *disagreement* signal that flags where the model is uncertain --
            # the known fix for model-exploitation in planning, and an
            # exploration signal for data collection. K=1 keeps the original
            # single-head parameter layout so existing checkpoints still load.
            if self.ensemble_heads > 1:
                self.ensemble_grus = nn.ModuleList(
                    nn.GRUCell(hidden_dim + 1, latent_dim)
                    for _ in range(self.ensemble_heads - 1)
                )
                self.ensemble_blocks = nn.ModuleList(
                    nn.ModuleList(
                        MLP([latent_dim + hidden_dim + 1, hidden_dim, latent_dim], layer_norm=True)
                        for _ in range(self.transition_depth)
                    )
                    for _ in range(self.ensemble_heads - 1)
                )
        else:
            raise ValueError(f"Unknown predictor_mode: {predictor_mode}")
        # A wider, two-hidden-layer state decoder: accurate geometry (gripper +
        # object positions) is what the manipulation-aware planner relies on.
        self.state_probe = MLP([latent_dim, hidden_dim, hidden_dim, state_dim], layer_norm=True)
        self.distance_probe = MLP([latent_dim, hidden_dim, 1])
        # Inverse-dynamics head a_t = g(z_t, z_{t+1}): predicting the action from a
        # latent transition forces the encoder to RETAIN the fine, control-relevant
        # (action-discriminative) detail that VICReg/prediction smoothing otherwise
        # discards — making the latent useful for contact-rich manipulation control,
        # not just abstract/navigation tasks.
        self.inverse_dynamics = inverse_dynamics
        if inverse_dynamics:
            self.inverse_head = MLP([2 * latent_dim, hidden_dim, hidden_dim, action_dim], layer_norm=True)
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

    def _recurrent_head(self, z, action_seq, horizon, gru, blocks):
        """Roll one recurrent dynamics head; returns ``[batch, horizon, latent]``."""
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
            gru_in = torch.cat([action_emb, step], dim=-1)
            pred = gru(gru_in, pred)
            for block in blocks:
                pred = pred + block(torch.cat([pred, action_emb, step], dim=-1))
            preds.append(pred)
        return torch.stack(preds, dim=1)

    def _heads(self):
        """Iterate (gru, transition_blocks) over the primary head and any ensemble heads."""
        yield self.gru, self.transition_blocks
        if self.ensemble_heads > 1:
            for gru, blocks in zip(self.ensemble_grus, self.ensemble_blocks):
                yield gru, blocks

    def rollout_heads(self, z: torch.Tensor, action_seq: torch.Tensor, horizon: int) -> torch.Tensor:
        """Per-head recurrent rollouts, shape ``[num_heads, batch, horizon, latent]``."""
        return torch.stack(
            [self._recurrent_head(z, action_seq, horizon, g, b) for g, b in self._heads()],
            dim=0,
        )

    def disagreement(self, z: torch.Tensor, action_seq: torch.Tensor, horizon: int) -> torch.Tensor:
        """Mean across-head latent variance over the rollout (epistemic uncertainty).

        Zero unless this is an ensemble recurrent model. Used as an MPC cost term
        (avoid model-exploitation) and an exploration signal (Roadmap A, item 2).
        """
        if self.predictor_mode != "recurrent" or self.ensemble_heads <= 1:
            return torch.zeros((), device=z.device, dtype=z.dtype)
        return self.rollout_heads(z, action_seq, horizon).var(dim=0).mean()

    def predict_rollout(self, z: torch.Tensor, action_seq: torch.Tensor, horizon: int) -> torch.Tensor:
        """Roll the latent dynamics forward and return every intermediate latent.

        Returns a tensor of shape ``[batch, horizon, latent_dim]``. For an
        ensemble recurrent model this is the across-head mean. Only valid for the
        ``rollout`` and ``recurrent`` predictor modes.
        """
        if self.predictor_mode == "recurrent":
            if self.ensemble_heads > 1:
                return self.rollout_heads(z, action_seq, horizon).mean(dim=0)
            return self._recurrent_head(z, action_seq, horizon, self.gru, self.transition_blocks)
        # rollout mode: single shared transition (no ensemble support)
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
