"""Latent-space CEM planning through the JEPA predictor.

This is the piece the Adroit controllers were missing. The phase-inverse
controllers route through ``encoder`` only: they map ``(z_t, z_future)`` to an
action chunk and execute it, so the JEPA *predictor* is never queried and the
world model is a representation, not a world model. Here the predictor chooses:
candidate action chunks are rolled forward in latent space and ranked by how
close the predicted future lands to the subgoal latent.

The search is iCEM-style (Pinneri et al.): colored noise gives temporally
correlated candidates, which matters at 26 actuators where white noise
averages to near-zero net motion, and elites are refit across a few iterations.
Two terms keep the optimizer honest about model error:

* an ensemble-disagreement penalty, so candidates that only look good because
  the heads disagree about them are not selected (model exploitation);
* a trust region around the warm-started previous plan, so the search stays in
  the region where the dense-rollout training actually constrained the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def colored_noise(
    shape: tuple[int, ...],
    beta: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``[..., horizon, action_dim]`` noise with power spectrum ``1/f^beta``.

    ``beta=0`` is white noise. Larger beta yields smoother, temporally
    correlated action sequences — the regime where a high-DoF hand actually
    produces net motion instead of jittering in place.
    """
    *lead, horizon, action_dim = shape
    if beta <= 0:
        return torch.randn(shape, device=device, generator=generator)
    freqs = torch.fft.rfftfreq(horizon, device=device)
    scale = torch.ones_like(freqs)
    scale[1:] = freqs[1:] ** (-beta / 2.0)
    spectrum = torch.complex(
        torch.randn((*lead, action_dim, freqs.numel()), device=device, generator=generator),
        torch.randn((*lead, action_dim, freqs.numel()), device=device, generator=generator),
    ) * scale
    noise = torch.fft.irfft(spectrum, n=horizon, dim=-1)
    noise = noise.transpose(-1, -2)
    std = noise.std(dim=-2, keepdim=True).clamp_min(1e-6)
    return noise / std


@dataclass
class LatentCEMConfig:
    # "cem"  - sampled shooting in raw action space
    # "grad" - backprop through the predictor
    # "rank" - score a caller-supplied set of on-manifold candidates
    method: str = "cem"
    horizon: int = 8
    candidates: int = 256
    iterations: int = 4
    grad_restarts: int = 32
    grad_iters: int = 60
    grad_lr: float = 0.1
    grad_init_std: float = 0.1
    elite_frac: float = 0.1
    init_std: float = 0.6
    min_std: float = 0.05
    noise_beta: float = 2.0
    momentum: float = 0.1
    keep_elite_frac: float = 0.3
    end_weight: float = 1.0
    path_weight: float = 0.25
    state_weight: float = 0.0
    disagreement_weight: float = 0.0
    smooth_weight: float = 0.0
    trust_region: float = 0.0
    action_low: float = -1.0
    action_high: float = 1.0
    seed: int = 0
    extras: dict = field(default_factory=dict)


class LatentCEMPlanner:
    """Rank/refine action chunks by rolling them through the JEPA predictor."""

    def __init__(self, model, cfg: LatentCEMConfig, device: torch.device, normalizer=None) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.normalizer = normalizer
        self.generator = torch.Generator(device=device).manual_seed(cfg.seed)
        self.action_dim = int(model.action_dim)
        self.prev_mean: torch.Tensor | None = None
        self.last_diagnostics: dict[str, float] = {}

    def reset(self) -> None:
        self.prev_mean = None

    @torch.no_grad()
    def inverse_proposal(self, z: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor | None:
        """Seed chunk from the world model's own inverse-dynamics head.

        This head is an auxiliary of the JEPA training objective (it is what
        keeps the encoder control-aware), not a separately trained policy. It
        is only a *starting point*: whether it survives contact with the
        predictor is what the ablations measure.
        """
        if not getattr(self.model, "inverse_dynamics", False):
            return None
        # The head emits exactly ``inverse_horizon`` actions; reshape by that,
        # then trim or tile to the planning horizon.
        k = int(getattr(self.model, "inverse_horizon", 1))
        chunk = self.model.inverse_head(torch.cat([z, z_goal], dim=-1))
        chunk = chunk.view(1, k, self.action_dim)
        if k < self.cfg.horizon:
            chunk = chunk.repeat(1, (self.cfg.horizon + k - 1) // k, 1)
        return chunk[0, : self.cfg.horizon].clamp(self.cfg.action_low, self.cfg.action_high)

    def _cost(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        z_goal: torch.Tensor,
        goal_state: torch.Tensor | None,
        prev_action: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        n = actions.shape[0]
        z_rep = z.expand(n, -1)
        per_head = self.model.rollout_heads(z_rep, actions, cfg.horizon)  # [K, N, H, L]
        rollout = per_head.mean(dim=0)

        goal_n = F.normalize(z_goal, dim=-1).view(1, 1, -1)
        dist = (F.normalize(rollout, dim=-1) - goal_n).pow(2).sum(dim=-1)  # [N, H]
        cost = cfg.end_weight * dist[:, -1] + cfg.path_weight * dist.mean(dim=1)
        terms = {"latent_end": dist[:, -1], "latent_path": dist.mean(dim=1)}

        if cfg.state_weight > 0 and goal_state is not None:
            decoded = self.model.state_probe(rollout)
            state_dist = torch.linalg.norm(decoded - goal_state.view(1, 1, -1), dim=-1)
            cost = cost + cfg.state_weight * (state_dist[:, -1] + 0.25 * state_dist.mean(dim=1))
            terms["state_end"] = state_dist[:, -1]

        if cfg.disagreement_weight > 0 and per_head.shape[0] > 1:
            disagree = per_head.var(dim=0).mean(dim=(1, 2))
            cost = cost + cfg.disagreement_weight * disagree
            terms["disagreement"] = disagree

        if cfg.smooth_weight > 0:
            head = actions[:, :1] if prev_action is None else actions[:, :1] - prev_action.view(1, 1, -1)
            delta = torch.cat([head, actions[:, 1:] - actions[:, :-1]], dim=1)
            cost = cost + cfg.smooth_weight * delta.pow(2).mean(dim=(1, 2))

        return cost, terms

    def _grad_plan(
        self,
        z: torch.Tensor,
        z_goal: torch.Tensor,
        goal_state: torch.Tensor | None,
        prev_action: torch.Tensor | None,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        """Optimize the chunk by backprop through the latent rollout.

        Sampling cannot search a 26-actuator chunk: measured on hammer demos,
        CEM never finds a chunk the model scores as well as the ground-truth
        one, while gradient descent does so easily. Actions are parameterized
        through ``tanh`` so the bound is respected by construction, and several
        restarts are optimized in parallel because the landscape is not convex.
        """
        cfg = self.cfg
        h, a_dim = cfg.horizon, self.action_dim
        base = torch.atanh(anchor.clamp(-0.99, 0.99))
        raw = base.unsqueeze(0).repeat(cfg.grad_restarts, 1, 1)
        raw = raw + torch.randn(
            raw.shape, device=self.device, generator=self.generator
        ) * cfg.grad_init_std
        raw[0] = base
        raw = raw.detach().requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=cfg.grad_lr)

        def decode(x: torch.Tensor) -> torch.Tensor:
            a = torch.tanh(x)
            if cfg.trust_region > 0:
                a = anchor.unsqueeze(0) + (a - anchor.unsqueeze(0)).clamp(
                    -cfg.trust_region, cfg.trust_region
                )
            return a.clamp(cfg.action_low, cfg.action_high)

        with torch.enable_grad():
            for _ in range(cfg.grad_iters):
                cost, _ = self._cost(z, decode(raw), z_goal, goal_state, prev_action)
                opt.zero_grad(set_to_none=True)
                cost.sum().backward()
                opt.step()
        with torch.no_grad():
            actions = decode(raw)
            cost, terms = self._cost(z, actions, z_goal, goal_state, prev_action)
            best = int(torch.argmin(cost))
            self.last_diagnostics = {k: float(v[best]) for k, v in terms.items()}
            self.last_diagnostics["cost"] = float(cost[best])
        return actions[best].detach()

    @torch.no_grad()
    def rank(
        self,
        z: torch.Tensor,
        z_goal: torch.Tensor,
        candidates: torch.Tensor,
        *,
        goal_state: torch.Tensor | None = None,
        prev_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pick the best of a caller-supplied candidate set by predictor rollout.

        The measured failure of ``grad``/``cem`` on this task is that the
        optimizer leaves the region where the model is valid. Ranking sidesteps
        that: every candidate comes from a learned prior evaluated at a
        different subgoal (plus small perturbations), so the whole set is
        on-manifold and the predictor is only asked the question it can answer
        — *which of these plausible chunks gets closest to the subgoal* — which
        is the same role it plays in the Fetch push/pick controllers.
        """
        cost, terms = self._cost(z, candidates, z_goal, goal_state, prev_action)
        best = int(torch.argmin(cost))
        self.last_diagnostics = {k: float(v[best]) for k, v in terms.items()}
        self.last_diagnostics["cost"] = float(cost[best])
        self.last_diagnostics["n_candidates"] = float(candidates.shape[0])
        self.last_diagnostics["chosen"] = float(best)
        self.last_diagnostics["cost_spread"] = float(cost.max() - cost.min())
        return candidates[best]

    @torch.no_grad()
    def plan(
        self,
        z: torch.Tensor,
        z_goal: torch.Tensor,
        *,
        goal_state: torch.Tensor | None = None,
        prev_action: torch.Tensor | None = None,
        proposal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the best ``[horizon, action_dim]`` chunk for reaching ``z_goal``.

        ``proposal`` optionally seeds the search (e.g. from the world model's
        own inverse-dynamics head). The planner still selects among
        predictor-scored candidates, so a seeded run and an unseeded run differ
        only in where the search starts.
        """
        cfg = self.cfg
        if cfg.method == "grad":
            if proposal is not None:
                anchor = proposal.view(cfg.horizon, self.action_dim)
            elif self.prev_mean is not None:
                anchor = torch.cat([self.prev_mean[1:], self.prev_mean[-1:]], dim=0)
            else:
                anchor = torch.zeros(cfg.horizon, self.action_dim, device=self.device)
            best = self._grad_plan(z, z_goal, goal_state, prev_action, anchor)
            self.prev_mean = best.clone()
            return best
        h, a_dim = cfg.horizon, self.action_dim
        n_elite = max(2, int(cfg.candidates * cfg.elite_frac))

        if proposal is not None:
            mean = proposal.view(h, a_dim).clone()
        elif self.prev_mean is not None:
            # Shift the previous plan one step and repeat its tail: receding
            # horizon warm start.
            mean = torch.cat([self.prev_mean[1:], self.prev_mean[-1:]], dim=0)
        else:
            mean = torch.zeros(h, a_dim, device=self.device)
        std = torch.full((h, a_dim), cfg.init_std, device=self.device)
        anchor = mean.clone()

        elites: torch.Tensor | None = None
        best_actions = mean
        best_cost = torch.tensor(float("inf"), device=self.device)
        for it in range(cfg.iterations):
            noise = colored_noise(
                (cfg.candidates, h, a_dim), cfg.noise_beta, self.device, self.generator
            )
            actions = mean.unsqueeze(0) + noise * std.unsqueeze(0)
            if elites is not None and cfg.keep_elite_frac > 0:
                keep = max(1, int(n_elite * cfg.keep_elite_frac))
                actions[:keep] = elites[:keep]
            actions[-1] = mean  # always evaluate the current mean itself
            if cfg.trust_region > 0:
                actions = anchor.unsqueeze(0) + (actions - anchor.unsqueeze(0)).clamp(
                    -cfg.trust_region, cfg.trust_region
                )
            actions = actions.clamp(cfg.action_low, cfg.action_high)

            cost, terms = self._cost(z, actions, z_goal, goal_state, prev_action)
            order = torch.argsort(cost)
            elites = actions[order[:n_elite]]
            if cost[order[0]] < best_cost:
                best_cost = cost[order[0]]
                best_actions = actions[order[0]].clone()
                self.last_diagnostics = {
                    k: float(v[order[0]]) for k, v in terms.items()
                }
            new_mean = elites.mean(dim=0)
            mean = (1 - cfg.momentum) * new_mean + cfg.momentum * mean
            std = elites.std(dim=0).clamp_min(cfg.min_std)

        self.prev_mean = best_actions.clone()
        self.last_diagnostics["cost"] = float(best_cost)
        return best_actions
