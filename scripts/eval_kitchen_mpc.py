"""Pure model-based control on FrankaKitchen using the JEPA *world model* — no BC.

This finally uses JEPA as a world model on kitchen: at each step we encode obs->z
and run CEM planning *entirely through the JEPA predictor* (predict_rollout),
scoring imagined latent trajectories by a learned reward head (predicted task-count)
and executing the best first action (receding horizon). There is no behaviour-cloned
proposal — the dynamics predictor + reward head select the action from scratch.

An ensemble-disagreement penalty (Roadmap A #2) optionally down-weights action
sequences the world model is uncertain about — the anti-model-exploitation guard,
which matters most on contact-rich dynamics like kitchen.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.envs import make_env, flatten_obs
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import MLP
from jepa_robotics.tasks import resolve_task


def load_reward_head(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    head = MLP([ck["latent_dim"], ck["hidden"], ck["hidden"], 1]).to(device)
    head.load_state_dict(ck["state_dict"]); head.eval()
    return head, ck.get("r_mean", 0.0), ck.get("r_std", 1.0)


@torch.no_grad()
def mpc_action(wm, head, z, action_dim, low, high, *, horizon, candidates, cem_iters,
               std0, elite_frac, rng, device, prev_mean=None, disagree_weight=0.0):
    """Plan a horizon-H action sequence from scratch (no BC) via CEM through the WM."""
    if prev_mean is None:
        mean = np.zeros((horizon, action_dim), dtype=np.float32)
    else:
        # warm-start: shift previous plan forward one step (receding horizon)
        mean = np.vstack([prev_mean[1:], prev_mean[-1:]]).astype(np.float32)
    std = np.full((horizon, action_dim), std0, dtype=np.float32)
    ensemble = getattr(wm, "ensemble_heads", 1) > 1 and disagree_weight > 0.0
    best_seq = mean.copy()
    for _ in range(cem_iters):
        samples = rng.normal(mean, std, size=(candidates, horizon, action_dim)).astype(np.float32)
        samples[0] = mean
        samples = np.clip(samples, low, high).astype(np.float32)  # clip vs float64 bounds promotes -> recast
        seqs = torch.from_numpy(samples).to(device)
        z_rep = z.repeat(candidates, 1)
        traj = wm.predict_rollout(z_rep, seqs, horizon)        # [K, H, latent]
        scores = head(traj).squeeze(-1).sum(dim=1)             # cumulative predicted task-count
        if ensemble:
            heads = wm.rollout_heads(z_rep, seqs, horizon)     # [n_heads, K, H, latent]
            disagree = heads.var(dim=0).mean(dim=(1, 2))
            scores = scores - disagree_weight * disagree
        k = max(1, int(candidates * elite_frac))
        elites = torch.topk(scores, k).indices.cpu().numpy()
        mean = samples[elites].mean(0)
        std = samples[elites].std(0) + 1e-3
        best_seq = samples[int(torch.argmax(scores).cpu())]
    return np.clip(best_seq[0], low, high).astype(np.float32), mean


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="franka_kitchen")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--reward-head", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--candidates", type=int, default=512)
    p.add_argument("--cem-iters", type=int, default=3)
    p.add_argument("--std", type=float, default=0.5)
    p.add_argument("--elite-frac", type=float, default=0.1)
    p.add_argument("--disagree-weight", type=float, default=0.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, device)
    head, _, _ = load_reward_head(args.reward_head, device)
    wm.eval()
    H = min(args.horizon, int(cfg["max_horizon"]))

    rng = np.random.default_rng(args.seed)
    tasks = []
    for ep in range(args.episodes):
        env = make_env(task.env_id, seed=args.seed + ep, max_episode_steps=task.max_episode_steps)
        low, high = env.action_space.low, env.action_space.high
        obs, _ = env.reset(seed=args.seed + ep)
        term = trunc = False; info = {}; prev_mean = None
        while not (term or trunc):
            with torch.no_grad():
                z = wm.encode(torch.from_numpy(norm.encode(flatten_obs(obs))).unsqueeze(0).to(device))
            a, prev_mean = mpc_action(wm, head, z, spec.action_dim, low, high,
                                      horizon=H, candidates=args.candidates, cem_iters=args.cem_iters,
                                      std0=args.std, elite_frac=args.elite_frac, rng=rng, device=device,
                                      prev_mean=prev_mean, disagree_weight=args.disagree_weight)
            obs, _, term, trunc, info = env.step(a)
        tasks.append(int(info.get("tasks_done", 0)))
        env.close()
    tasks = np.array(tasks)
    print(f'RESULT kitchen-WM-MPC(no-BC) horizon={H} candidates={args.candidates} '
          f'mean_tasks={tasks.mean():.2f}/4 full4={np.mean(tasks>=4):.2f} '
          f'>=1={np.mean(tasks>=1):.2f} >=2={np.mean(tasks>=2):.2f} >=3={np.mean(tasks>=3):.2f}', flush=True)


if __name__ == "__main__":
    main()
