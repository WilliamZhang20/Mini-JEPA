"""Model-based control on Adroit: BC proposes, JEPA world-model MPC refines.

This is P1 — finally using the JEPA *predictor* (predict_rollout) for control, not
just the encoder. At each step: encode obs -> z; the BC policy proposes an action;
we sample candidate action sequences (BC action + Gaussian noise, a few CEM
iters), roll each through the JEPA dynamics, score by the cumulative latent reward
head, and execute the best first action. The pure-BC sequence is always a
candidate, so MPC degrades gracefully to BC if the world model is unreliable
(guards against the model-exploitation that broke slide MPC).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.envs import make_env, flatten_obs
from jepa_robotics.evaluate import load_jepa_artifact, load_policy_artifact
from jepa_robotics.models import MLP
from jepa_robotics.tasks import resolve_task


def load_reward_head(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    head = MLP([ck["latent_dim"], ck["hidden"], ck["hidden"], 1]).to(device)
    head.load_state_dict(ck["state_dict"]); head.eval()
    return head


@torch.no_grad()
def mpc_action(wm, policy, head, z, action_dim, low, high, *, horizon, candidates,
               cem_iters, std0, elite_frac, rng, device, disagree_weight=0.0):
    """Return the first action of the best BC-seeded action sequence under the WM+reward head.

    With ``disagree_weight`` > 0 and an ensemble WM, subtract the inter-head
    rollout disagreement (epistemic uncertainty) from the score — Roadmap A #2's
    anti-model-exploitation: avoid actions the world model is uncertain about.
    """
    a_bc = policy(z)[0].cpu().numpy()  # BC proposal
    mean = np.tile(a_bc, (horizon, 1)).astype(np.float32)  # [H, A]
    std = np.full((horizon, action_dim), std0, dtype=np.float32)
    best_seq = mean.copy()
    ensemble = getattr(wm, "ensemble_heads", 1) > 1 and disagree_weight > 0.0
    for _ in range(cem_iters):
        samples = rng.normal(mean, std, size=(candidates, horizon, action_dim)).astype(np.float32)
        samples[0] = mean  # always keep the (pure-BC / current-mean) candidate
        samples = np.clip(samples, low, high)
        seqs = torch.from_numpy(samples).to(device)
        z_rep = z.repeat(candidates, 1)
        traj = wm.predict_rollout(z_rep, seqs, horizon)              # [K, H, latent]
        scores = head(traj).squeeze(-1).sum(dim=1)                   # cumulative predicted reward
        if ensemble:
            heads = wm.rollout_heads(z_rep, seqs, horizon)           # [n_heads, K, H, latent]
            disagree = heads.var(dim=0).mean(dim=(1, 2))             # [K] epistemic uncertainty
            scores = scores - disagree_weight * disagree
        elites = torch.topk(scores, max(1, int(candidates * elite_frac))).indices.cpu().numpy()
        mean = samples[elites].mean(0)
        std = samples[elites].std(0) + 1e-3
        best_seq = samples[int(torch.argmax(scores).cpu())]
    return np.clip(best_seq[0], low, high).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="adroit_pen")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--policy-path", type=Path, required=True)
    p.add_argument("--reward-head", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--candidates", type=int, default=200)
    p.add_argument("--cem-iters", type=int, default=2)
    p.add_argument("--std", type=float, default=0.3)
    p.add_argument("--elite-frac", type=float, default=0.1)
    p.add_argument("--disagree-weight", type=float, default=0.0,
                   help="Penalize ensemble inter-head disagreement in the MPC score (anti-exploitation).")
    p.add_argument("--also-bc", action="store_true", help="Also eval pure BC for a head-to-head.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, device)
    pol, _ = load_policy_artifact(args.policy_path, device)
    head = load_reward_head(args.reward_head, device)
    wm.eval(); pol.eval()
    H = min(args.horizon, int(cfg["max_horizon"]))

    def rollout(use_mpc):
        rng = np.random.default_rng(args.seed)
        succ = []
        for ep in range(args.episodes):
            env = make_env(task.env_id, seed=args.seed + ep, max_episode_steps=task.max_episode_steps)
            low, high = env.action_space.low, env.action_space.high
            obs, _ = env.reset(seed=args.seed + ep)
            term = trunc = False; info = {}
            while not (term or trunc):
              with torch.no_grad():
                z = wm.encode(torch.from_numpy(norm.encode(flatten_obs(obs))).unsqueeze(0).to(device))
                if use_mpc:
                    a = mpc_action(wm, pol, head, z, spec.action_dim, low, high,
                                   horizon=H, candidates=args.candidates, cem_iters=args.cem_iters,
                                   std0=args.std, elite_frac=args.elite_frac, rng=rng, device=device,
                                   disagree_weight=args.disagree_weight)
                else:
                    a = np.clip(pol(z)[0].cpu().numpy(), low, high).astype(np.float32)
                obs, _, term, trunc, info = env.step(a)
            succ.append(float(info.get("is_success", info.get("success", 0.0))))
            env.close()
        return float(np.mean(succ))

    if args.also_bc:
        bc = rollout(use_mpc=False)
        print(f'{{"task": "{task.name}", "policy": "BC", "episodes": {args.episodes}, "success_rate": {bc:.4f}}}', flush=True)
    mpc = rollout(use_mpc=True)
    print(f'{{"task": "{task.name}", "policy": "BC+WM-MPC", "horizon": {H}, "candidates": {args.candidates}, '
          f'"episodes": {args.episodes}, "success_rate": {mpc:.4f}}}', flush=True)


if __name__ == "__main__":
    main()
