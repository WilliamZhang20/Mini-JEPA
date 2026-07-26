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
    """Anatomical token transformer: normalized state -> structured latent slots.

    Scalar channels not covered by ``token_groups`` retain one token each. Local
    hand/object/tactile groups get one learned projection each. Optional SE(3)
    goal-error features add a relation token computed from the *raw* achieved and
    desired poses. Multiple learned latent queries let object, contact and hand
    information survive as separate slots instead of forcing the entire hand
    through one summary token before dynamics sees an action.
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
        latent_slots: int = 1,
        pose_relation_dims: tuple[int, int] | None = None,
        state_mean: torch.Tensor | None = None,
        state_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.latent_slots = max(1, int(latent_slots))
        if latent_dim % self.latent_slots:
            raise ValueError(
                f"latent_dim={latent_dim} must be divisible by latent_slots={self.latent_slots}"
            )
        self.slot_dim = latent_dim // self.latent_slots
        self.token_groups = tuple(token_groups or ())
        self.pose_relation_dims = pose_relation_dims
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
        if pose_relation_dims is not None:
            achieved, desired = pose_relation_dims
            if achieved < 0 or desired < 0 or achieved + 7 > state_dim or desired + 7 > state_dim:
                raise ValueError(
                    f"Invalid pose_relation_dims={pose_relation_dims} for state_dim={state_dim}"
                )
            if state_mean is None or state_std is None:
                raise ValueError("state_mean/std are required with pose_relation_dims")
            self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
            self.register_buffer("state_std", torch.as_tensor(state_std, dtype=torch.float32))
            self.relation_value = nn.Linear(6, d_model)
            self.relation_emb = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, self.latent_slots, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.slot_dim)

    def _pose_relation(self, state: torch.Tensor) -> torch.Tensor:
        """Goal-relative translation and SO(3) log vector from normalized state."""
        achieved, desired = self.pose_relation_dims
        raw = state * self.state_std + self.state_mean
        p_cur, p_goal = raw[:, achieved:achieved + 3], raw[:, desired:desired + 3]
        q_cur = raw[:, achieved + 3:achieved + 7]
        q_goal = raw[:, desired + 3:desired + 7]
        q_cur = q_cur / q_cur.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        q_goal = q_goal / q_goal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # q_err = q_goal * conjugate(q_cur), in MuJoCo's (w,x,y,z) order.
        aw, av = q_goal[:, :1], q_goal[:, 1:]
        bw, bv = q_cur[:, :1], -q_cur[:, 1:]
        err_w = aw * bw - (av * bv).sum(dim=-1, keepdim=True)
        err_v = aw * bv + bw * av + torch.cross(av, bv, dim=-1)
        # Canonical hemisphere makes q and -q identical before the log map.
        sign = torch.where(err_w < 0, -torch.ones_like(err_w), torch.ones_like(err_w))
        err_w, err_v = err_w * sign, err_v * sign
        sin_half = err_v.norm(dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(sin_half, err_w.clamp_min(1e-7))
        rotvec = err_v * (angle / sin_half.clamp_min(1e-7))
        # Five centimetres is the natural HandManipulate target-position scale.
        return torch.cat([(p_goal - p_cur) / 0.05, rotvec], dim=-1)

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
        if self.pose_relation_dims is not None:
            relation = self.relation_value(self._pose_relation(state)).unsqueeze(1)
            tok = torch.cat([tok, relation + self.relation_emb.unsqueeze(0)], dim=1)
        tok = torch.cat([self.cls.expand(b, -1, -1), tok], dim=1)
        out = self.tr(tok)
        slots = self.head(self.norm(out[:, :self.latent_slots]))
        return slots.reshape(b, self.latent_dim)


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


class _SlotDynamicsHead(nn.Module):
    """Recurrent object/contact-slot dynamics conditioned on each hand command.

    At every environment step the current latent slots and one full actuator
    command attend to each other, then every slot receives a residual update.
    This preserves structured hand/object/contact memory across a regrasp cycle;
    the old single-token dynamics remains available for existing checkpoints.
    """

    def __init__(
        self,
        latent_dim: int,
        latent_slots: int,
        action_dim: int,
        d_model: int,
        depth: int,
        heads: int,
        max_horizon: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if latent_dim % latent_slots:
            raise ValueError("latent_dim must be divisible by latent_slots")
        self.latent_dim = latent_dim
        self.latent_slots = latent_slots
        self.slot_dim = latent_dim // latent_slots
        self.slot_in = nn.Linear(self.slot_dim, d_model)
        self.action_in = nn.Linear(action_dim, d_model)
        self.slot_emb = nn.Parameter(torch.randn(latent_slots, d_model) * 0.02)
        self.step_emb = nn.Parameter(torch.randn(max_horizon, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model,
            heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.slot_dim)

    def forward(self, z: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        b, horizon, _ = action_seq.shape
        slots = z.reshape(b, self.latent_slots, self.slot_dim)
        predictions = []
        for step in range(horizon):
            slot_tokens = self.slot_in(slots) + self.slot_emb.unsqueeze(0)
            action_token = (
                self.action_in(action_seq[:, step]) + self.step_emb[step]
            ).unsqueeze(1)
            out = self.tr(torch.cat([slot_tokens, action_token], dim=1))
            delta = self.head(self.norm(out[:, :self.latent_slots]))
            slots = slots + delta
            predictions.append(slots.reshape(b, self.latent_dim))
        return torch.stack(predictions, dim=1)


class _ContextualSlotDynamicsHead(nn.Module):
    """Short-history latent dynamics with per-block action conditioning.

    A single observation is ambiguous in dexterous manipulation: object and
    finger velocity, incipient slip, and contact loading are only observable
    from a short temporal window.  This predictor consumes ``context_len``
    encoded states and the actions between them.  For every imagined step, all
    state-history slots attend jointly while the complete action window
    modulates *every* transformer block through AdaLN.  The predicted state and
    candidate action are then shifted into the context for recurrent rollout.

    This follows the empirically strongest JEPA-WM recipe for low-dimensional
    control: W=3 context, short recurrent prediction, and action conditioning in
    every block.  Learned temporal embeddings are used here because the context
    is deliberately tiny; recent ablations find the much larger gains come from
    history and AdaLN, with only task-dependent gains from RoPE.
    """

    def __init__(
        self,
        latent_dim: int,
        latent_slots: int,
        action_dim: int,
        d_model: int,
        depth: int,
        heads: int,
        context_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if latent_dim % latent_slots:
            raise ValueError("latent_dim must be divisible by latent_slots")
        if context_len < 2:
            raise ValueError("context_len must be at least 2")
        self.latent_dim = latent_dim
        self.latent_slots = latent_slots
        self.slot_dim = latent_dim // latent_slots
        self.context_len = context_len
        self.action_dim = action_dim
        self.slot_in = nn.Linear(self.slot_dim, d_model)
        self.slot_emb = nn.Parameter(torch.randn(latent_slots, d_model) * 0.02)
        self.time_emb = nn.Parameter(torch.randn(context_len, d_model) * 0.02)
        self.action_condition = nn.Sequential(
            nn.Linear(context_len * action_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList(
            _AdaLNBlock(d_model, heads, dropout) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.slot_dim)

    def forward(
        self,
        z_history: torch.Tensor,
        action_history: torch.Tensor,
        action_seq: torch.Tensor,
        *,
        detach_intermediate: bool = False,
    ) -> torch.Tensor:
        """Predict a rollout from state/action history.

        Args:
            z_history: ``[B, W, latent_dim]`` ending at the current state.
            action_history: ``[B, W-1, action_dim]`` actions connecting those
                states.
            action_seq: ``[B, H, action_dim]`` candidate future commands.
            detach_intermediate: truncate the gradient after every non-terminal
                predicted step.  This implements last-gradient-only rollout
                training while retaining recurrent predictions as context.
        """
        if z_history.ndim != 3 or z_history.shape[1] != self.context_len:
            raise ValueError(
                f"z_history must be [B,{self.context_len},{self.latent_dim}], "
                f"got {tuple(z_history.shape)}"
            )
        if action_history.shape[1:] != (self.context_len - 1, self.action_dim):
            raise ValueError(
                "action_history must be "
                f"[B,{self.context_len - 1},{self.action_dim}], "
                f"got {tuple(action_history.shape)}"
            )
        batch, horizon, _ = action_seq.shape
        history = z_history
        past_actions = action_history
        predictions = []
        for step in range(horizon):
            slots = history.reshape(
                batch, self.context_len, self.latent_slots, self.slot_dim
            )
            tokens = self.slot_in(slots)
            tokens = (
                tokens
                + self.time_emb.view(1, self.context_len, 1, -1)
                + self.slot_emb.view(1, 1, self.latent_slots, -1)
            ).reshape(batch, self.context_len * self.latent_slots, -1)
            action_window = torch.cat(
                [past_actions, action_seq[:, step : step + 1]], dim=1
            )
            condition = self.action_condition(action_window.flatten(1))
            for block in self.blocks:
                tokens = block(tokens, condition)
            newest = tokens.reshape(
                batch, self.context_len, self.latent_slots, -1
            )[:, -1]
            delta = self.head(self.norm(newest))
            current = history[:, -1].reshape(
                batch, self.latent_slots, self.slot_dim
            )
            prediction = (current + delta).reshape(batch, self.latent_dim)
            predictions.append(prediction)

            recurrent = (
                prediction.detach()
                if detach_intermediate and step < horizon - 1
                else prediction
            )
            history = torch.cat([history[:, 1:], recurrent.unsqueeze(1)], dim=1)
            past_actions = torch.cat(
                [past_actions[:, 1:], action_seq[:, step : step + 1]], dim=1
            )
        return torch.stack(predictions, dim=1)


class _LatentDifferenceActionDecoder(nn.Module):
    """Recover each actuator command from one latent displacement.

    Action queries attend to the structured latent-slot differences.  Because
    the decoder never receives either endpoint, state-specific shortcuts cannot
    solve the task: the transition geometry itself must retain action
    information (the LDAD principle from Delta-JEPA).
    """

    def __init__(
        self,
        latent_dim: int,
        latent_slots: int,
        action_dim: int,
        d_model: int,
        depth: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if latent_dim % latent_slots:
            raise ValueError("latent_dim must be divisible by latent_slots")
        slot_dim = latent_dim // latent_slots
        self.latent_slots = latent_slots
        self.action_dim = action_dim
        self.delta_in = nn.Linear(slot_dim, d_model)
        self.slot_emb = nn.Parameter(torch.randn(latent_slots, d_model) * 0.02)
        self.action_queries = nn.Parameter(
            torch.randn(action_dim, d_model) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model,
            heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, 1)

    def forward(self, latent_delta: torch.Tensor) -> torch.Tensor:
        batch = latent_delta.shape[0]
        slots = latent_delta.reshape(batch, self.latent_slots, -1)
        delta_tokens = self.delta_in(slots) + self.slot_emb.unsqueeze(0)
        action_tokens = self.action_queries.unsqueeze(0).expand(batch, -1, -1)
        tokens = self.transformer(torch.cat([delta_tokens, action_tokens], dim=1))
        action_tokens = tokens[:, self.latent_slots :]
        return self.output(self.norm(action_tokens)).squeeze(-1)


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
        latent_slots: int = 1,
        pose_relation_dims: tuple[int, int] | None = None,
        state_mean: torch.Tensor | None = None,
        state_std: torch.Tensor | None = None,
        object_probe_dims: tuple[int, int] | None = None,
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
        self.latent_slots = max(1, int(latent_slots))
        self.pose_relation_dims = pose_relation_dims
        self.object_probe_dims = object_probe_dims
        if latent_dim % self.latent_slots:
            raise ValueError("latent_dim must be divisible by latent_slots")
        self.slot_dim = latent_dim // self.latent_slots

        self.encoder = _TokenEncoder(
            state_dim,
            latent_dim,
            d_model,
            enc_depth,
            heads,
            dropout,
            self.token_groups,
            self.latent_slots,
            pose_relation_dims,
            state_mean,
            state_std,
        )
        self.target_encoder = deepcopy(self.encoder)
        dynamics_type = _SlotDynamicsHead if self.latent_slots > 1 else _DynamicsHead
        if self.latent_slots > 1:
            self.dyn_heads = nn.ModuleList(
                dynamics_type(
                    latent_dim,
                    self.latent_slots,
                    action_dim,
                    d_model,
                    dyn_depth,
                    heads,
                    max_horizon,
                    dropout,
                )
                for _ in range(self.ensemble_heads)
            )
        else:
            self.dyn_heads = nn.ModuleList(
                dynamics_type(
                    latent_dim, action_dim, d_model, dyn_depth, heads, max_horizon, dropout
                )
                for _ in range(self.ensemble_heads)
            )
        # State + contact-consistency decoders (DexWM-style): full state, plus an
        # emphasized contact/relative-geometry slice so the latent keeps the
        # fingertip/object detail contact control depends on.
        self.state_probe = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim), nn.GELU(), nn.Linear(2 * latent_dim, state_dim),
        )
        self.distance_probe = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, 1))
        if object_probe_dims is not None:
            lo, hi = object_probe_dims
            # The first latent query is explicitly object-supervised. Other slots
            # remain available for hand configuration, contact and global context.
            self.object_probe = nn.Sequential(
                nn.Linear(self.slot_dim, 2 * self.slot_dim),
                nn.GELU(),
                nn.Linear(2 * self.slot_dim, hi - lo),
            )
        if contact_dims is not None:
            lo, hi = contact_dims
            contact_in = self.slot_dim if self.latent_slots > 1 else latent_dim
            contact_hidden = 2 * contact_in if self.latent_slots > 1 else contact_in
            self.contact_probe = nn.Sequential(
                nn.Linear(contact_in, contact_hidden),
                nn.GELU(),
                nn.Linear(contact_hidden, hi - lo),
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
        if self.contact_dims is None:
            return None
        contact_z = (
            z.reshape(*z.shape[:-1], self.latent_slots, self.slot_dim)[..., 1, :]
            if self.latent_slots > 1 else z
        )
        return self.contact_probe(contact_z)

    def predict_object(self, z: torch.Tensor) -> torch.Tensor | None:
        """Decode the normalized achieved object pose from the object slot."""
        if self.object_probe_dims is None:
            return None
        object_z = z.reshape(*z.shape[:-1], self.latent_slots, self.slot_dim)[..., 0, :]
        return self.object_probe(object_z)


class ContextualDexterousJEPA(DexterousJEPA):
    """DexterousJEPA with short state/action context and AdaLN dynamics.

    The encoders and diagnostic probes are intentionally identical to
    :class:`DexterousJEPA`, allowing a representation learned from reward-free
    play to be frozen while a context-aware predictor is fit solely with a JEPA
    latent prediction loss.
    """

    def __init__(
        self,
        *,
        context_len: int = 3,
        latent_difference_actions: bool = False,
        action_decoder_depth: int = 3,
        full_object_probe: bool = False,
        explicit_object_slot: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.context_len = int(context_len)
        self.latent_difference_actions = bool(latent_difference_actions)
        self.full_object_probe = bool(full_object_probe)
        self.explicit_object_slot = bool(explicit_object_slot)
        if self.explicit_object_slot:
            if self.object_probe_dims is None:
                raise ValueError("explicit_object_slot requires object_probe_dims")
            lo, hi = self.object_probe_dims
            if hi - lo > self.slot_dim:
                raise ValueError("Object fields do not fit in the first latent slot")
        d_model = int(kwargs.get("d_model", 256))
        dyn_depth = int(kwargs.get("dyn_depth", 4))
        heads = int(kwargs.get("heads", 8))
        dropout = float(kwargs.get("dropout", 0.0))
        self.dyn_heads = nn.ModuleList(
            _ContextualSlotDynamicsHead(
                self.latent_dim,
                self.latent_slots,
                self.action_dim,
                d_model,
                dyn_depth,
                heads,
                self.context_len,
                dropout,
            )
            for _ in range(self.ensemble_heads)
        )
        if self.latent_difference_actions:
            self.action_decoder = _LatentDifferenceActionDecoder(
                self.latent_dim,
                self.latent_slots,
                self.action_dim,
                d_model,
                action_decoder_depth,
                heads,
                dropout,
            )
        if self.full_object_probe:
            if self.object_probe_dims is None:
                raise ValueError("full_object_probe requires object_probe_dims")
            lo, hi = self.object_probe_dims
            self.object_probe = nn.Sequential(
                nn.Linear(self.latent_dim, 2 * self.latent_dim),
                nn.GELU(),
                nn.Linear(2 * self.latent_dim, hi - lo),
            )

    def decode_latent_action(
        self, current_z: torch.Tensor, next_z: torch.Tensor
    ) -> torch.Tensor:
        if not self.latent_difference_actions:
            raise RuntimeError("This model has no latent-difference action decoder")
        return self.action_decoder(next_z - current_z)

    def _inject_explicit_object(
        self, latent: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        if not self.explicit_object_slot:
            return latent
        lo, hi = self.object_probe_dims
        # Clone avoids mutating an encoder output needed elsewhere in autograd.
        latent = latent.clone()
        latent[..., : hi - lo] = state[..., lo:hi]
        return latent

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self._inject_explicit_object(super().encode(state), state)

    @torch.no_grad()
    def encode_target(self, state: torch.Tensor) -> torch.Tensor:
        latent = self.target_encoder(state)
        return self._inject_explicit_object(latent, state)

    def predict_object(self, z: torch.Tensor) -> torch.Tensor | None:
        if self.object_probe_dims is None:
            return None
        if self.explicit_object_slot:
            lo, hi = self.object_probe_dims
            return z[..., : hi - lo]
        if self.full_object_probe:
            return self.object_probe(z)
        return super().predict_object(z)

    def rollout_heads_context(
        self,
        z_history: torch.Tensor,
        action_history: torch.Tensor,
        action_seq: torch.Tensor,
        horizon: int,
        *,
        detach_intermediate: bool = False,
    ) -> torch.Tensor:
        actions = action_seq[:, :horizon]
        return torch.stack(
            [
                head(
                    z_history,
                    action_history,
                    actions,
                    detach_intermediate=detach_intermediate,
                )
                for head in self.dyn_heads
            ],
            dim=0,
        )

    def predict_rollout_context(
        self,
        z_history: torch.Tensor,
        action_history: torch.Tensor,
        action_seq: torch.Tensor,
        horizon: int,
        *,
        detach_intermediate: bool = False,
    ) -> torch.Tensor:
        return self.rollout_heads_context(
            z_history,
            action_history,
            action_seq,
            horizon,
            detach_intermediate=detach_intermediate,
        ).mean(dim=0)

    def predict_rollout(
        self, z: torch.Tensor, action_seq: torch.Tensor, horizon: int
    ) -> torch.Tensor:
        """Compatibility fallback for callers without an observation history."""
        z_history = z.unsqueeze(1).expand(-1, self.context_len, -1)
        action_history = torch.zeros(
            z.shape[0],
            self.context_len - 1,
            self.action_dim,
            device=z.device,
            dtype=z.dtype,
        )
        return self.predict_rollout_context(
            z_history, action_history, action_seq, horizon
        )

    def rollout_heads(
        self, z: torch.Tensor, action_seq: torch.Tensor, horizon: int
    ) -> torch.Tensor:
        z_history = z.unsqueeze(1).expand(-1, self.context_len, -1)
        action_history = torch.zeros(
            z.shape[0],
            self.context_len - 1,
            self.action_dim,
            device=z.device,
            dtype=z.dtype,
        )
        return self.rollout_heads_context(
            z_history, action_history, action_seq, horizon
        )


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


class DexterousInverseDynamics(nn.Module):
    """Multi-hypothesis temporal decoder for contact-rich inverse dynamics.

    Each learned mode emits a full action-increment chunk.  Best-of-mode
    reconstruction prevents incompatible finger gaits from being averaged,
    while JEPA forward consistency can select and train hypotheses by their
    predicted object effect.
    """

    def __init__(
        self,
        condition_dim: int,
        horizon: int,
        action_dim: int,
        *,
        hidden: int = 256,
        n_blocks: int = 4,
        heads: int = 8,
        modes: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.modes = modes
        self.condition = nn.Sequential(
            nn.Linear(condition_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.time_queries = nn.Parameter(
            torch.randn(horizon, hidden) * 0.02
        )
        self.mode_queries = nn.Parameter(
            torch.randn(modes, hidden) * 0.02
        )
        self.blocks = nn.ModuleList(
            _AdaLNBlock(hidden, heads, dropout) for _ in range(n_blocks)
        )
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, action_dim)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        batch = len(condition)
        context = self.condition(condition)
        tokens = (
            self.time_queries.view(1, 1, self.horizon, -1)
            + self.mode_queries.view(1, self.modes, 1, -1)
        ).expand(batch, -1, -1, -1)
        tokens = tokens.reshape(batch * self.modes, self.horizon, -1)
        repeated_context = (
            context[:, None, :]
            .expand(-1, self.modes, -1)
            .reshape(batch * self.modes, -1)
        )
        for block in self.blocks:
            tokens = block(tokens, repeated_context)
        output = self.output(self.norm(tokens))
        return output.reshape(
            batch, self.modes, self.horizon, self.action_dim
        )
