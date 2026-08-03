"""Evaluate the JEPA latent-planning controller (predictor in the loop).

The controller is:

    z_t      = encoder(o_t)
    z_goal   = high level (learned subgoal net, or a demo-retrieved future)
    a_{t:t+H-1} = argmin over candidates of || predict_rollout(z_t, a) - z_goal ||

so every executed action is chosen by the JEPA *predictor*, not emitted by an
inverse model. ``--ablate`` re-randomizes one component at load time, which is
how the load-bearing claim is checked rather than asserted:

* ``predictor``  — random-init dynamics, trained encoder/probe. If success
  survives this, the planner was not really planning.
* ``encoder``    — random-init encoder, trained dynamics.
* ``subgoal``    — random-init high level.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.algos.futures import DemoLockedFutureIndex
from jepa_robotics.algos.latent_subgoal import LatentSubgoalNet
from jepa_robotics.algos.planning.latent_mpc import LatentCEMConfig, LatentCEMPlanner
from jepa_robotics.algos.priors import FlowChunkActor, InversePrior, PredictorGuidedRefiner
from jepa_robotics.data import load_episodes_npz
from jepa_robotics.envs import flatten_obs, make_env
from jepa_robotics.evaluate import load_jepa_artifact, rollout_policy
from jepa_robotics.tasks import resolve_task


def reinit_module(module: torch.nn.Module, seed: int) -> None:
    """Re-randomize every parameter of ``module`` in place."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for param in module.parameters():
        if param.dim() >= 2:
            weight = torch.empty(param.shape)
            torch.nn.init.kaiming_uniform_(weight, a=5 ** 0.5, generator=generator)
            param.data.copy_(weight.to(param.device))
        else:
            param.data.copy_(
                (torch.rand(param.shape, generator=generator) * 0.2 - 0.1).to(param.device)
            )


class LatentPlanPolicy:
    """Receding-horizon JEPA latent planner with a learned or retrieved subgoal."""

    def __init__(
        self,
        *,
        wm,
        normalizer,
        planner: LatentCEMPlanner,
        subgoal_net: LatentSubgoalNet | None,
        demo_index: DemoLockedFutureIndex | None,
        device,
        horizon: int,
        exec_k: int,
        action_scale: float,
        proposal_mode: str,
        actor: InversePrior | None,
        actor_ckpt: dict | None,
        rank_horizons: list[int],
        rank_noise_std: float,
        rank_noise_copies: int,
        name: str,
        refiner: PredictorGuidedRefiner | None = None,
        refiner_ckpt: dict | None = None,
        refine_steps: int = 3,
        flow_temp: float = 1.0,
    ) -> None:
        self.flow_temp = flow_temp
        self.proposal_mode = proposal_mode
        self.actor = actor
        self.actor_ckpt = actor_ckpt
        self.refiner = refiner
        self.refiner_ckpt = refiner_ckpt
        self.refine_steps = refine_steps
        self.rank_horizons = rank_horizons
        self.rank_noise_std = rank_noise_std
        self.rank_noise_copies = rank_noise_copies
        self.name = name
        self.wm = wm
        self.normalizer = normalizer
        self.planner = planner
        self.subgoal_net = subgoal_net
        self.demo_index = demo_index
        self.device = device
        self.horizon = horizon
        self.exec_k = max(1, exec_k)
        self.action_scale = action_scale
        self.cached: list[np.ndarray] = []
        self.prev_action: torch.Tensor | None = None
        self.diagnostics: list[dict] = []

    def reset(self) -> None:
        self.cached = []
        self.prev_action = None
        self.planner.reset()
        if self.demo_index is not None:
            self.demo_index.reset()

    @torch.no_grad()
    def _chunks_for(
        self, z: torch.Tensor, z_goal: torch.Tensor, horizon: int, n: int = 1
    ) -> torch.Tensor:
        """``n`` action chunks reaching ``z_goal``, shape ``[n, horizon, action]``.

        Uses the dedicated actor if one was loaded, else the world model's
        auxiliary inverse head. ``n > 1`` is meaningful only for a stochastic
        (flow) actor; the deterministic paths return a single chunk.
        """
        if self.actor is None:
            return self.planner.inverse_proposal(z, z_goal).unsqueeze(0)
        h_token = torch.full(
            (z.shape[0], 1), float(horizon) / float(self.actor_ckpt["max_horizon"]),
            device=z.device, dtype=z.dtype,
        )
        cond = torch.cat([z, z_goal, h_token], dim=-1)
        if isinstance(self.actor, FlowChunkActor):
            flat = self.actor.sample(cond, n, init_noise_scale=self.flow_temp)
        else:
            flat = self.actor(cond)
        chunk = flat.view(
            flat.shape[0], int(self.actor_ckpt["chunk"]), int(self.actor_ckpt["action_dim"])
        )
        k = chunk.shape[1]
        if k < self.horizon:
            chunk = chunk.repeat(1, (self.horizon + k - 1) // k, 1)
        return chunk[:, : self.horizon].clamp(
            self.planner.cfg.action_low, self.planner.cfg.action_high
        )

    def _chunk_for(self, z: torch.Tensor, z_goal: torch.Tensor, horizon: int) -> torch.Tensor:
        return self._chunks_for(z, z_goal, horizon, n=1)[0]

    @torch.no_grad()
    def _candidates(self, z: torch.Tensor) -> torch.Tensor:
        """On-manifold candidate chunks: one per subgoal lookahead, plus copies.

        Every candidate is the actor answering "how do I get to *this*
        subgoal", for several subgoals the learned high level proposes at
        different lookaheads. Diversity therefore comes from the high level —
        and, for a flow actor, from its own samples — rather than from
        perturbing actions off-distribution, which is what made free
        optimization exploit the predictor. The Gaussian jitter copies exist
        only for the deterministic actor; flow samples are already diverse.
        """
        stochastic = isinstance(self.actor, FlowChunkActor)
        chunks, goals, hs = [], [], []
        for h in self.rank_horizons:
            z_h, _state_h, _p = self.subgoal_net.subgoal(z, h)
            n = (1 + self.rank_noise_copies) if stochastic else 1
            c = self._chunks_for(z, z_h, h, n=n)
            chunks.append(c)
            goals.append(z_h.expand(c.shape[0], -1))
            hs.append(torch.full((c.shape[0], 1), float(h), device=z.device, dtype=z.dtype))
        if not chunks:
            zero = torch.zeros(1, self.horizon, self.wm.action_dim, device=self.device)
            return zero, None, None
        base = torch.cat(chunks, dim=0)
        goal_t = torch.cat(goals, dim=0)
        h_t = torch.cat(hs, dim=0)
        if not stochastic and self.rank_noise_std > 0 and self.rank_noise_copies > 0:
            jitter = base.repeat(self.rank_noise_copies, 1, 1)
            jitter = jitter + torch.randn_like(jitter) * self.rank_noise_std
            base = torch.cat([base, jitter], dim=0)
            goal_t = torch.cat([goal_t, goal_t.repeat(self.rank_noise_copies, 1)], dim=0)
            h_t = torch.cat([h_t, h_t.repeat(self.rank_noise_copies, 1)], dim=0)
        base = base.clamp(self.planner.cfg.action_low, self.planner.cfg.action_high)
        return base, goal_t, h_t

    @torch.no_grad()
    def _apply_refiner(
        self,
        z: torch.Tensor,
        candidates: torch.Tensor,
        cand_goals: torch.Tensor,
        cand_hs: torch.Tensor,
    ) -> torch.Tensor:
        """Predictor-feedback correction of every candidate before ranking.

        This is where the world model *generates* rather than only selects:
        each candidate is rolled through the predictor, the latent goal error
        is fed back, and the refiner head moves the chunk toward the demo
        manifold's answer. Each candidate is refined toward the subgoal that
        generated it — refining everything toward one goal was measured to
        destroy the lookahead diversity ranking needs (hammer 0.833 -> 0.633).
        Ranking afterwards is only the final pick.
        """
        n = candidates.shape[0]
        k = min(int(self.refiner_ckpt["chunk"]), candidates.shape[1])
        h_token = cand_hs / float(self.refiner_ckpt["max_horizon"])
        refined = self.refiner.refine(
            self.wm, z.expand(n, -1), cand_goals, h_token,
            candidates[:, :k], steps=self.refine_steps,
            action_low=self.planner.cfg.action_low,
            action_high=self.planner.cfg.action_high,
        )
        return torch.cat([refined, candidates[:, k:]], dim=1)

    @torch.no_grad()
    def _plan(self, obs) -> np.ndarray:
        raw = flatten_obs(obs)
        state = torch.from_numpy(self.normalizer.encode(raw)).unsqueeze(0).to(self.device)
        z = self.wm.encode(state)
        if self.subgoal_net is not None:
            z_goal, state_goal, progress = self.subgoal_net.subgoal(z, self.horizon)
            self.diagnostics.append({"progress": float(progress[0])})
        else:
            target_raw = self.demo_index.query(raw, self.normalizer)
            state_goal = torch.from_numpy(
                self.normalizer.encode(np.asarray(target_raw, dtype=np.float32))
            ).unsqueeze(0).to(self.device)
            z_goal = self.wm.encode_target(state_goal)
        proposal = (
            self._chunk_for(z, z_goal, self.horizon)
            if self.proposal_mode == "inverse"
            else None
        )
        if self.proposal_mode == "inverse-only":
            plan = self._chunk_for(z, z_goal, self.horizon)
        elif self.planner.cfg.method == "rank":
            candidates, cand_goals, cand_hs = self._candidates(z)
            if self.refiner is not None and cand_goals is not None:
                candidates = self._apply_refiner(z, candidates, cand_goals, cand_hs)
            plan = self.planner.rank(
                z, z_goal, candidates,
                goal_state=state_goal, prev_action=self.prev_action,
            )
        else:
            plan = self.planner.plan(
                z, z_goal, goal_state=state_goal,
                prev_action=self.prev_action, proposal=proposal,
            )
        return (plan * self.action_scale).detach().cpu().numpy().astype(np.float32)

    def act(self, obs, env):
        if not self.cached:
            plan = self._plan(obs)
            self.cached = [plan[i].copy() for i in range(min(self.exec_k, len(plan)))]
        action = np.clip(self.cached.pop(0), env.action_space.low, env.action_space.high)
        self.prev_action = torch.from_numpy(action).to(self.device)
        return action.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--subgoal-path", type=Path, default=None,
                   help="Learned high level. Omit and pass --demo-npz to use demo retrieval instead.")
    p.add_argument("--actor-path", type=Path, default=None,
                   help="Dedicated future-conditioned actor (train_latent_actor.py). "
                        "Without it the world model's auxiliary inverse head is used.")
    p.add_argument("--demo-npz", type=Path, default=None)
    p.add_argument("--demo-episodes", type=int, default=400)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=64000)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--method", choices=["cem", "grad", "rank"], default="rank")
    p.add_argument("--rank-horizons", default="2,4,8,16",
                   help="Subgoal lookaheads the high level is queried at; each yields one "
                        "candidate chunk that the predictor then ranks.")
    p.add_argument("--rank-noise-std", type=float, default=0.05)
    p.add_argument("--rank-noise-copies", type=int, default=8)
    p.add_argument("--proposal", choices=["none", "inverse", "inverse-only"], default="inverse",
                   help="'inverse-only' executes the world model's inverse head with no planning "
                        "at all -- the reference point for whether the predictor adds anything.")
    p.add_argument("--grad-restarts", type=int, default=32)
    p.add_argument("--grad-iters", type=int, default=60)
    p.add_argument("--grad-lr", type=float, default=0.1)
    p.add_argument("--grad-init-std", type=float, default=0.1)
    p.add_argument("--candidates", type=int, default=256)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--elite-frac", type=float, default=0.1)
    p.add_argument("--init-std", type=float, default=0.6)
    p.add_argument("--min-std", type=float, default=0.05)
    p.add_argument("--noise-beta", type=float, default=2.0)
    p.add_argument("--momentum", type=float, default=0.1)
    p.add_argument("--end-weight", type=float, default=1.0)
    p.add_argument("--path-weight", type=float, default=0.25)
    p.add_argument("--state-weight", type=float, default=0.0)
    p.add_argument("--disagreement-weight", type=float, default=0.0)
    p.add_argument("--smooth-weight", type=float, default=0.0)
    p.add_argument("--trust-region", type=float, default=0.0)
    p.add_argument("--exec-k", type=int, default=2)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--refiner-path", type=Path, default=None,
                   help="optional PredictorGuidedRefiner checkpoint: every rank candidate "
                        "is corrected from the predictor's own rollout error before ranking")
    p.add_argument("--refine-steps", type=int, default=3)
    p.add_argument("--flow-temp", type=float, default=1.0,
                   help="flow-actor sampling temperature (init noise scale); lower "
                        "concentrates samples near the mode for precision tasks")
    p.add_argument("--ablate", choices=["none", "predictor", "encoder", "subgoal", "actor", "refiner"],
                   default="none")
    p.add_argument("--ablate-seed", type=int, default=1234)
    p.add_argument("--tag", default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--video-out", type=Path, default=None)
    p.add_argument("--video-episodes", type=int, default=1)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    task = resolve_task(args.task, None)
    wm, normalizer, spec, cfg = load_jepa_artifact(args.model_path, device)

    subgoal_net = None
    demo_index = None
    if args.subgoal_path is not None:
        ckpt = torch.load(args.subgoal_path, map_location=device, weights_only=False)
        subgoal_net = LatentSubgoalNet(
            latent_dim=int(ckpt["latent_dim"]),
            state_dim=int(ckpt["state_dim"]),
            hidden=int(ckpt["hidden"]),
            n_blocks=int(ckpt["n_blocks"]),
            max_horizon=int(ckpt["max_horizon"]),
        ).to(device)
        subgoal_net.load_state_dict(ckpt["state_dict"])
        subgoal_net.eval()
    elif args.demo_npz is not None:
        episodes = load_episodes_npz(args.demo_npz)[: args.demo_episodes]
        demo_index = DemoLockedFutureIndex(
            [ep.states for ep in episodes], normalizer, horizon=args.horizon
        )
    else:
        raise SystemExit("pass --subgoal-path (learned high level) or --demo-npz (retrieval)")

    actor = actor_ckpt = None
    if args.actor_path is not None:
        actor_ckpt = torch.load(args.actor_path, map_location=device, weights_only=False)
        if actor_ckpt.get("actor_type", "inverse") == "flow":
            actor = FlowChunkActor(
                int(actor_ckpt["cond_dim"]), int(actor_ckpt["chunk_dim"]),
                int(actor_ckpt["hidden"]), int(actor_ckpt["n_blocks"]),
                flow_steps=int(actor_ckpt.get("flow_steps", 16)),
            ).to(device)
        else:
            actor = InversePrior(
                int(actor_ckpt["cond_dim"]), int(actor_ckpt["chunk_dim"]),
                int(actor_ckpt["hidden"]), int(actor_ckpt["n_blocks"]),
            ).to(device)
        actor.load_state_dict(actor_ckpt["state_dict"])
        actor.eval()

    refiner = refiner_ckpt = None
    if args.refiner_path is not None:
        refiner_ckpt = torch.load(args.refiner_path, map_location=device, weights_only=False)
        refiner = PredictorGuidedRefiner(
            int(refiner_ckpt["latent_dim"]), int(refiner_ckpt["chunk_dim"]),
            int(refiner_ckpt["hidden"]), int(refiner_ckpt["n_blocks"]),
        ).to(device)
        refiner.load_state_dict(refiner_ckpt["state_dict"])
        refiner.eval()

    if args.ablate == "predictor":
        reinit_module(wm.gru, args.ablate_seed)
        reinit_module(wm.transition_blocks, args.ablate_seed + 1)
        if getattr(wm, "ensemble_heads", 1) > 1:
            reinit_module(wm.ensemble_grus, args.ablate_seed + 2)
            reinit_module(wm.ensemble_blocks, args.ablate_seed + 3)
        reinit_module(wm.action_encoder, args.ablate_seed + 4)
    elif args.ablate == "encoder":
        reinit_module(wm.encoder, args.ablate_seed)
        wm.reset_target()
    elif args.ablate == "subgoal" and subgoal_net is not None:
        reinit_module(subgoal_net, args.ablate_seed)
    elif args.ablate == "actor" and actor is not None:
        reinit_module(actor, args.ablate_seed)
    elif args.ablate == "refiner" and refiner is not None:
        reinit_module(refiner, args.ablate_seed)

    env = make_env(
        task.env_id,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps or task.max_episode_steps,
        render_mode="rgb_array" if args.video_out is not None else None,
        width=args.width if args.video_out is not None else None,
        height=args.height if args.video_out is not None else None,
    )
    low = float(np.min(env.action_space.low))
    high = float(np.max(env.action_space.high))
    planner = LatentCEMPlanner(
        wm,
        LatentCEMConfig(
            method=args.method,
            horizon=args.horizon,
            candidates=args.candidates,
            iterations=args.iterations,
            grad_restarts=args.grad_restarts,
            grad_iters=args.grad_iters,
            grad_lr=args.grad_lr,
            grad_init_std=args.grad_init_std,
            elite_frac=args.elite_frac,
            init_std=args.init_std,
            min_std=args.min_std,
            noise_beta=args.noise_beta,
            momentum=args.momentum,
            end_weight=args.end_weight,
            path_weight=args.path_weight,
            state_weight=args.state_weight,
            disagreement_weight=args.disagreement_weight,
            smooth_weight=args.smooth_weight,
            trust_region=args.trust_region,
            action_low=low,
            action_high=high,
            seed=args.seed,
        ),
        device,
        normalizer=normalizer,
    )
    source = "learned" if subgoal_net is not None else "demo"
    name = args.tag or f"jepa_latent_{args.method}_{source}_{args.proposal}"
    if args.ablate != "none":
        name = f"{name}_ablate_{args.ablate}"
    policy = LatentPlanPolicy(
        wm=wm,
        normalizer=normalizer,
        planner=planner,
        subgoal_net=subgoal_net,
        demo_index=demo_index,
        device=device,
        horizon=args.horizon,
        exec_k=args.exec_k,
        action_scale=args.action_scale,
        proposal_mode=args.proposal,
        actor=actor,
        actor_ckpt=actor_ckpt,
        rank_horizons=[int(x) for x in args.rank_horizons.split(",") if x.strip()],
        rank_noise_std=args.rank_noise_std,
        rank_noise_copies=args.rank_noise_copies,
        name=name,
        refiner=refiner,
        refiner_ckpt=refiner_ckpt,
        refine_steps=args.refine_steps,
        flow_temp=args.flow_temp,
    )
    metrics = rollout_policy(
        env,
        policy,
        episodes=args.episodes,
        seed=args.seed,
        video_path=args.video_out,
        video_episodes=min(args.video_episodes, args.episodes),
        fps=args.fps,
    )
    env.close()
    row = {
        "event": "latent_mpc_eval",
        "task": task.name,
        "model_path": str(args.model_path),
        "subgoal_path": str(args.subgoal_path) if args.subgoal_path else None,
        "actor_path": str(args.actor_path) if args.actor_path else None,
        "refiner_path": str(args.refiner_path) if args.refiner_path else None,
        "refine_steps": int(args.refine_steps) if args.refiner_path else None,
        "demo_npz": str(args.demo_npz) if args.demo_npz else None,
        "subgoal_source": source,
        "ablate": args.ablate,
        "seed": args.seed,
        "planner": {
            "method": args.method, "proposal": args.proposal,
            "rank_horizons": args.rank_horizons,
            "rank_noise_std": args.rank_noise_std,
            "rank_noise_copies": args.rank_noise_copies,
            "horizon": args.horizon, "candidates": args.candidates,
            "iterations": args.iterations, "exec_k": args.exec_k,
            "grad_restarts": args.grad_restarts, "grad_iters": args.grad_iters,
            "grad_lr": args.grad_lr,
            "end_weight": args.end_weight, "path_weight": args.path_weight,
            "state_weight": args.state_weight,
            "disagreement_weight": args.disagreement_weight,
            "smooth_weight": args.smooth_weight, "trust_region": args.trust_region,
            "noise_beta": args.noise_beta, "init_std": args.init_std,
        },
        **metrics,
    }
    print(json.dumps(row, default=str), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")


if __name__ == "__main__":
    main()
