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
        inverse_horizon: int = 1,
        latent_norm: bool = False,
        per_head_action_encoder: bool = False,
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
        self.inverse_horizon = max(1, inverse_horizon)

        # Latent-space normalization. A planner rolls the predictor many steps
        # without re-encoding a real observation, so per-step drift in latent
        # *scale* compounds and silently miscalibrates the subgoal distance it
        # minimizes. A parameter-free LayerNorm applied to both encoded and
        # predicted latents pins every rollout step onto the same shell, which
        # is what makes a step-16 latent cost comparable to the step-1 costs the
        # model was trained on. Opt-in, so existing checkpoints still load.
        self.latent_norm = bool(latent_norm)
        self.latent_ln = (
            nn.LayerNorm(latent_dim, elementwise_affine=False) if latent_norm else nn.Identity()
        )
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
            # heads give an inter-head *disagreement* signal that flags where the
            # model is uncertain -- the known fix for model-exploitation in
            # planning, and an exploration signal for data collection. K=1 keeps
            # the original single-head parameter layout so existing checkpoints
            # still load.
            #
            # ``per_head_action_encoder`` matters more than it looks. With one
            # shared action encoder the heads see an *identical* embedding of an
            # action, so they cannot disagree about an action chunk they have
            # never seen -- which is precisely the disagreement a planner needs,
            # since it is the actions, not the states, that the optimizer pushes
            # off-distribution. Measured on hammer: with a shared encoder the
            # disagreement penalty fails to stop gradient planning from
            # "beating" ground-truth chunks at any weight up to 10.
            self.per_head_action_encoder = bool(per_head_action_encoder)
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
                if self.per_head_action_encoder:
                    self.ensemble_action_encoders = nn.ModuleList(
                        MLP([action_dim, hidden_dim, hidden_dim], layer_norm=True)
                        for _ in range(self.ensemble_heads - 1)
                    )
        else:
            raise ValueError(f"Unknown predictor_mode: {predictor_mode}")
        # A wider, two-hidden-layer state decoder: accurate geometry (gripper +
        # object positions) is what the manipulation-aware planner relies on.
        self.state_probe = MLP([latent_dim, hidden_dim, hidden_dim, state_dim], layer_norm=True)
        self.distance_probe = MLP([latent_dim, hidden_dim, 1])
        # Inverse-dynamics head a_{t:t+k-1} = g(z_t, z_{t+k}): predicting the
        # action chunk from a latent transition forces the encoder to retain
        # control-relevant detail. k=1 preserves the original single-step head.
        self.inverse_dynamics = inverse_dynamics
        if inverse_dynamics:
            self.inverse_head = MLP(
                [2 * latent_dim, hidden_dim, hidden_dim, action_dim * self.inverse_horizon],
                layer_norm=True,
            )
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
        return self.latent_ln(self.encoder(state))

    @torch.no_grad()
    def encode_target(self, state: torch.Tensor) -> torch.Tensor:
        """Map a state to its latent using the frozen EMA target encoder (used for prediction targets)."""
        return self.latent_ln(self.target_encoder(state))

    def _recurrent_head(self, z, action_seq, horizon, gru, blocks, action_encoder=None):
        """Roll one recurrent dynamics head; returns ``[batch, horizon, latent]``."""
        encode_action = action_encoder or self.action_encoder
        preds = []
        pred = z
        for i in range(horizon):
            step = torch.full(
                (action_seq.shape[0], 1),
                float(i + 1) / float(self.max_horizon),
                dtype=action_seq.dtype,
                device=action_seq.device,
            )
            action_emb = encode_action(action_seq[:, i])
            gru_in = torch.cat([action_emb, step], dim=-1)
            pred = gru(gru_in, pred)
            for block in blocks:
                pred = pred + block(torch.cat([pred, action_emb, step], dim=-1))
            # Re-normalize each rollout step so a long rollout stays on the same
            # latent shell as the encoder's own outputs (see ``latent_norm``).
            pred = self.latent_ln(pred)
            preds.append(pred)
        return torch.stack(preds, dim=1)

    def _heads(self):
        """Iterate (gru, transition_blocks, action_encoder) over every dynamics head."""
        yield self.gru, self.transition_blocks, self.action_encoder
        if self.ensemble_heads > 1:
            per_head = getattr(self, "ensemble_action_encoders", None)
            for i, (gru, blocks) in enumerate(zip(self.ensemble_grus, self.ensemble_blocks)):
                yield gru, blocks, (per_head[i] if per_head is not None else self.action_encoder)

    def rollout_heads(self, z: torch.Tensor, action_seq: torch.Tensor, horizon: int) -> torch.Tensor:
        """Per-head recurrent rollouts, shape ``[num_heads, batch, horizon, latent]``."""
        return torch.stack(
            [self._recurrent_head(z, action_seq, horizon, g, b, a) for g, b, a in self._heads()],
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
