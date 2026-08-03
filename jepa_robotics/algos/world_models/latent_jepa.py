"""Dense-rollout training for the action-conditioned JEPA world model.

``jepa_robotics.train`` supervises the predictor only at the horizons named in
``--horizons``, and it re-rolls the dynamics once per horizon. That is enough
for a model whose job is to *score* a chunk, but it is the wrong objective for
a model whose job is to be *planned through*: a planner queries every
intermediate step of the rollout, and it queries them for action chunks that no
demonstration ever executed.

This module trains the same ``ActionConditionedJEPA`` with the objective a
planner actually needs:

* one rollout of length ``H``, supervised at **every** step against the EMA
  target encoder (dense multi-step, no teacher forcing);
* the state probe decoded at every rollout step, so latent distance stays tied
  to physical geometry instead of drifting into an unanchored embedding;
* per-head bootstrap masks, so the ensemble's inter-head disagreement is a
  usable epistemic signal rather than K copies of the same function;
* a chunk inverse-dynamics head, which keeps the encoder control-aware.

Data lives on the GPU as one flat tensor with per-episode window bounds, so a
training step is an index-gather rather than a DataLoader round trip.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ...models import covariance_regularizer, variance_regularizer


@dataclass
class RolloutWindows:
    """GPU-resident (state, action-chunk, per-step future) window sampler.

    ``states``/``actions`` are the concatenation of every episode. ``starts``
    holds only those flat indices ``i`` for which ``i .. i+horizon`` stays
    inside one episode, so a sampled window never crosses an episode boundary.
    """

    states: torch.Tensor  # [N, state_dim] normalized
    actions: torch.Tensor  # [N, action_dim] (row i is the action taken at state i)
    starts: torch.Tensor  # [M] valid window start indices
    horizon: int

    @classmethod
    def from_episodes(
        cls,
        episodes,
        normalizer,
        horizon: int,
        device: torch.device,
    ) -> "RolloutWindows":
        state_blocks, action_blocks, start_blocks = [], [], []
        cursor = 0
        for ep in episodes:
            states = np.asarray(ep.states, dtype=np.float32)
            actions = np.asarray(ep.actions, dtype=np.float32)
            n_steps = min(len(states) - 1, len(actions))
            if n_steps < horizon:
                continue
            states = states[: n_steps + 1]
            actions = actions[:n_steps]
            # Pad the action array to the state length so both share one index
            # space; the pad row is never sampled because it lies past the last
            # valid window start.
            padded = np.concatenate([actions, np.zeros((1, actions.shape[1]), np.float32)], axis=0)
            state_blocks.append(normalizer.encode(states))
            action_blocks.append(padded)
            start_blocks.append(cursor + np.arange(n_steps - horizon + 1, dtype=np.int64))
            cursor += len(states)
        if not start_blocks:
            raise ValueError(f"no episode is long enough for horizon {horizon}")
        return cls(
            states=torch.from_numpy(np.concatenate(state_blocks)).to(device),
            actions=torch.from_numpy(np.concatenate(action_blocks)).to(device),
            starts=torch.from_numpy(np.concatenate(start_blocks)).to(device),
            horizon=int(horizon),
        )

    def __len__(self) -> int:
        return int(self.starts.numel())

    def sample(self, batch_size: int, generator: torch.Generator | None = None):
        pick = torch.randint(
            0, self.starts.numel(), (batch_size,), device=self.starts.device, generator=generator
        )
        return self.gather(self.starts[pick])

    def gather(self, idx: torch.Tensor):
        """Return ``(state, actions[B,H,A], futures[B,H,S])`` for window starts ``idx``."""
        steps = torch.arange(self.horizon, device=idx.device)
        act_idx = idx[:, None] + steps[None, :]
        fut_idx = act_idx + 1
        return self.states[idx], self.actions[act_idx], self.states[fut_idx]


def dense_rollout_loss(
    model,
    state: torch.Tensor,
    actions: torch.Tensor,
    futures: torch.Tensor,
    *,
    weights: dict[str, float],
    bootstrap: float = 0.8,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dense multi-step JEPA loss over one rollout of the full action chunk."""
    batch, horizon, _ = actions.shape
    z = model.encode(state)
    per_head = model.rollout_heads(z, actions, horizon)  # [K, B, H, L]
    n_heads = per_head.shape[0]

    with torch.no_grad():
        target = model.encode_target(futures.reshape(batch * horizon, -1))
        target = target.view(batch, horizon, -1)
        target_n = F.normalize(target, dim=-1)

    pred_n = F.normalize(per_head, dim=-1)
    # [K, B]: mean squared latent error per head per sample.
    per_sample = (pred_n - target_n.unsqueeze(0)).pow(2).mean(dim=(2, 3))
    if n_heads > 1 and 0.0 < bootstrap < 1.0:
        # Bootstrap masks decorrelate the heads. Without them every head sees
        # the same gradient on the same data and the disagreement signal the
        # planner relies on collapses to numerical noise.
        mask = (
            torch.rand(per_sample.shape, device=per_sample.device, generator=generator) < bootstrap
        ).float()
        mask = mask + (mask.sum(dim=1, keepdim=True) == 0).float()  # never drop a whole head
        loss_pred = (per_sample * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        loss_pred = per_sample.mean()

    rollout = per_head.mean(dim=0)  # [B, H, L]
    loss_pred_probe = F.mse_loss(model.state_probe(rollout), futures)
    loss_probe = F.mse_loss(model.state_probe(z), state)
    loss_var = variance_regularizer(z) + variance_regularizer(rollout[:, -1])
    loss_cov = covariance_regularizer(z) + covariance_regularizer(rollout[:, -1])

    if getattr(model, "inverse_dynamics", False):
        k = min(int(getattr(model, "inverse_horizon", 1)), horizon)
        z_next = model.encode(futures[:, k - 1])
        inv_pred = model.inverse_head(torch.cat([z, z_next], dim=-1))
        loss_inverse = F.mse_loss(inv_pred, actions[:, :k].reshape(batch, -1))
    else:
        loss_inverse = torch.zeros((), device=z.device, dtype=z.dtype)

    total = (
        loss_pred
        + weights["pred_probe"] * loss_pred_probe
        + weights["probe"] * loss_probe
        + weights["var"] * loss_var
        + weights["cov"] * loss_cov
        + weights["inverse"] * loss_inverse
    )
    metrics = {
        "loss": float(total.detach()),
        "pred": float(loss_pred.detach()),
        "pred_probe": float(loss_pred_probe.detach()),
        "probe": float(loss_probe.detach()),
        "var": float(loss_var.detach()),
        "cov": float(loss_cov.detach()),
        "inverse": float(loss_inverse.detach()),
    }
    if n_heads > 1:
        metrics["disagreement"] = float(per_head.var(dim=0).mean().detach())
    return total, metrics


@torch.no_grad()
def rollout_accuracy(model, windows: RolloutWindows, *, samples: int = 4096, seed: int = 0) -> dict:
    """Open-loop rollout error against a static-latent baseline.

    ``ratio_static`` > 1 means the predictor beats "assume nothing moves" —
    the minimum bar for a model worth planning through. It is reported per
    step because contact tasks typically degrade sharply after impact.
    """
    model.eval()
    generator = torch.Generator(device=windows.states.device).manual_seed(seed)
    pick = torch.randint(
        0, windows.starts.numel(), (min(samples, len(windows)),),
        device=windows.states.device, generator=generator,
    )
    state, actions, futures = windows.gather(windows.starts[pick])
    z = model.encode(state)
    rollout = model.predict_rollout(z, actions, actions.shape[1])
    decoded = model.state_probe(rollout)
    err = torch.linalg.norm(decoded - futures, dim=-1)
    static = torch.linalg.norm(state.unsqueeze(1) - futures, dim=-1)
    model.train()
    return {
        "state_rmse_per_step": [float(v) for v in err.mean(dim=0)],
        "static_rmse_per_step": [float(v) for v in static.mean(dim=0)],
        "ratio_static_per_step": [float(v) for v in (static.mean(dim=0) / err.mean(dim=0).clamp_min(1e-8))],
        "state_rmse": float(err.mean()),
        "static_rmse": float(static.mean()),
        "ratio_static": float(static.mean() / err.mean().clamp_min(1e-8)),
    }
