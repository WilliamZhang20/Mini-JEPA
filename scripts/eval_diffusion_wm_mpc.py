"""Diffusion proposes, JEPA world-model selects — FrankaKitchen.

The project thesis is "the policy proposes actions and the world-model MPC refines
them." Earlier the BC proposer was too weak for refinement to matter. Here the
proposer is the action-chunked DIFFUSION policy (multimodal, sequenced), and the
JEPA WORLD MODEL does the selection:

  encode obs -> z; sample N candidate action chunks from the diffusion policy
  (conditioned on z); roll EACH chunk through the ensemble latent dynamics from z;
  score = sum reward_head(predicted latents) - disagree_coef * ensemble_disagreement;
  execute the best chunk's first exec-k actions; replan (receding horizon).

This uses the full JEPA world model: encoder (conditioning) + latent dynamics
(rollout) + reward head (scoring) + ensemble (anti-exploitation). Set --candidates 1
to fall back to the plain diffusion policy for a head-to-head.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.envs import make_env, flatten_obs
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import MLP
from jepa_robotics.tasks import resolve_task
from scripts.eval_diffusion_policy import DEFAULT_TASKS, Scheduler
from jepa_robotics.algos.priors import EpsNet, make_ddpm
from scripts.train_latent_dynamics import EnsembleLatentDynamics, LatentDynamics


@torch.no_grad()
def sample_chunks(net, ddpm, cond, n, chunk_dim, device, objective="diffusion", flow_steps=16):
    """Sample n action chunks conditioned on the single latent cond [1, L]."""
    c = cond.repeat(n, 1)
    if objective == "flow":
        T = ddpm["T"]
        x = torch.randn(n, chunk_dim, device=device)
        dt = 1.0 / flow_steps
        for i in range(flow_steps):
            tau = torch.full((n,), i * dt, device=device)
            x = x + dt * net(x, tau * T, c)
        return x
    betas, alphas, abar, T = ddpm["betas"], ddpm["alphas"], ddpm["abar"], ddpm["T"]
    a = torch.randn(n, chunk_dim, device=device)
    for t in reversed(range(T)):
        tt = torch.full((n,), t, device=device, dtype=torch.long)
        eps = net(a, tt, c)
        mean = (a - betas[t] / torch.sqrt(1 - abar[t]) * eps) / torch.sqrt(alphas[t])
        a = mean + torch.sqrt(betas[t]) * torch.randn_like(a) if t > 0 else mean
    return a  # [n, chunk_dim]


@torch.no_grad()
def score_chunks(dyn, rhead, z, chunks, H, A_dim, alow, ahigh, disagree_coef):
    """Roll each chunk through the ensemble latent dynamics from z; return scores [n]."""
    n = chunks.shape[0]
    acts = chunks.view(n, H, A_dim).clamp(alow, ahigh)
    zc = z.repeat(n, 1)
    score = torch.zeros(n, device=z.device)
    ensemble = hasattr(dyn, "disagreement")
    for k in range(H):
        a = acts[:, k]
        if ensemble and disagree_coef > 0:
            score = score - disagree_coef * dyn.disagreement(zc, a)
        zc = dyn(zc, a)
        score = score + rhead(zc).squeeze(-1)
    return score


@torch.no_grad()
def score_chunks_jepa(wm, rhead, z, chunks, H, A_dim, alow, ahigh, disagree_coef):
    n = chunks.shape[0]
    acts = chunks.view(n, H, A_dim).clamp(alow, ahigh)
    z_rep = z.repeat(n, 1)
    traj = wm.predict_rollout(z_rep, acts, H)
    scores = rhead(traj).squeeze(-1).sum(dim=1)
    if getattr(wm, "ensemble_heads", 1) > 1 and disagree_coef > 0.0:
        heads = wm.rollout_heads(z_rep, acts, H)
        scores = scores - disagree_coef * heads.var(dim=0).mean(dim=(1, 2))
    return scores


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="franka_kitchen")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--dynamics", type=Path, default=None,
                   help="Optional separate latent dynamics. Omit to score candidates with the JEPA rollout head.")
    p.add_argument("--reward-head", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--candidates", type=int, default=16, help="diffusion chunks proposed per replan (1 = plain diffusion)")
    p.add_argument("--exec-k", type=int, default=8)
    p.add_argument("--disagree-coef", type=float, default=1.0)
    p.add_argument("--task-order", default=",".join(DEFAULT_TASKS),
                   help="Comma-separated one-hot order used by a subtask-conditioned skill checkpoint.")
    p.add_argument("--subtask-timeout", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()
    tasks_order = [t.strip() for t in args.task_order.split(",") if t.strip()]
    if len(tasks_order) != 4:
        raise ValueError(f"--task-order must contain exactly 4 task names, got {tasks_order}")

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    else:
        dev = torch.device(args.device)
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    L = int(cfg["latent_dim"])

    ck = torch.load(args.policy, map_location=dev, weights_only=False)
    net = EpsNet(ck["chunk_dim"], ck["cond_dim"], ck["hidden"], n_blocks=ck["n_blocks"]).to(dev)
    net.load_state_dict(ck["ema"]); net.eval()
    ddpm = make_ddpm(ck["diffusion_steps"], dev)
    H, A_dim, chunk_dim = ck["H"], ck["action_dim"], ck["chunk_dim"]
    HH = int(ck.get("obs_hist", 1))
    objective = ck.get("objective", "diffusion")
    raw_obs = bool(ck.get("raw_obs", False))
    concat_raw = bool(ck.get("concat_raw", False))
    progress_cond = bool(ck.get("progress_cond", False))
    subtask_cond = bool(ck.get("subtask_cond", False))

    dyn = None
    if args.dynamics is not None:
        dck = torch.load(args.dynamics, map_location=dev, weights_only=False)
        nh = int(dck.get("n_heads", 1))
        dyn = (EnsembleLatentDynamics(L, A_dim, dck["hidden"], nh) if nh > 1
               else LatentDynamics(L, A_dim, dck["hidden"])).to(dev)
        dyn.load_state_dict(dck["state_dict"]); dyn.eval()
    rck = torch.load(args.reward_head, map_location=dev, weights_only=False)
    rhead = MLP([rck["latent_dim"], rck["hidden"], rck["hidden"], 1]).to(dev)
    rhead.load_state_dict(rck["state_dict"]); rhead.eval()

    env0 = make_env(task.env_id, seed=0, max_episode_steps=task.max_episode_steps)
    alow = torch.as_tensor(env0.action_space.low, dtype=torch.float32, device=dev)
    ahigh = torch.as_tensor(env0.action_space.high, dtype=torch.float32, device=dev)
    env0.close()

    def encode_feature(obs, progress=0.0, target=None):
        x = torch.from_numpy(norm.encode(flatten_obs(obs))).unsqueeze(0).to(dev)
        if raw_obs:
            feat = x
        else:
            z = wm.encode(x)
            feat = torch.cat([x, z], dim=-1) if concat_raw else z
        if progress_cond:
            feat = torch.cat([feat, torch.full((1, 1), float(progress), device=dev)], dim=-1)
        if subtask_cond:
            oh = torch.zeros(1, 4, device=dev)
            if target is not None:
                oh[0, target] = 1.0
            feat = torch.cat([feat, oh], dim=-1)
        return feat

    tasks = []
    for ep in range(args.episodes):
        env = make_env(task.env_id, seed=args.seed + ep, max_episode_steps=task.max_episode_steps)
        low, high = env.action_space.low, env.action_space.high
        obs, _ = env.reset(seed=args.seed + ep)
        progress = 0.0; done_tasks = set(); sched = Scheduler(tasks_order, args.subtask_timeout); target = sched.update(done_tasks)
        hist = deque([encode_feature(obs, progress, target)] * HH, maxlen=HH)
        term = trunc = False; info = {}
        while not (term or trunc):
            x = torch.from_numpy(norm.encode(flatten_obs(obs))).unsqueeze(0).to(dev)
            z = wm.encode(x)
            cond = torch.cat(list(hist), dim=-1)
            chunks = sample_chunks(net, ddpm, cond, args.candidates, chunk_dim, dev, objective)
            if args.candidates > 1:
                if dyn is not None:
                    scores = score_chunks(dyn, rhead, z, chunks, H, A_dim, alow, ahigh, args.disagree_coef)
                else:
                    scores = score_chunks_jepa(wm, rhead, z, chunks, H, A_dim, alow, ahigh, args.disagree_coef)
                best = chunks[int(torch.argmax(scores))]
            else:
                best = chunks[0]
            chunk = best.cpu().numpy().reshape(H, A_dim)
            for j in range(min(args.exec_k, H)):
                if term or trunc:
                    break
                obs, _, term, trunc, info = env.step(np.clip(chunk[j], low, high).astype(np.float32))
                done_tasks |= set(info.get("step_task_completions", []))
                target = sched.update(done_tasks)
                progress = float(info.get("tasks_done", 0)) / 4.0
                hist.append(encode_feature(obs, progress, target))
        tasks.append(int(info.get("tasks_done", 0)))
        env.close()
    tasks = np.array(tasks)
    tag = f"diffusion+WM-select(N={args.candidates})" if args.candidates > 1 else "diffusion-only"
    print(f"RESULT {tag} mean_tasks={tasks.mean():.2f}/4 full4={np.mean(tasks >= 4):.2f} "
          f">=1={np.mean(tasks >= 1):.2f} >=2={np.mean(tasks >= 2):.2f} >=3={np.mean(tasks >= 3):.2f}", flush=True)


if __name__ == "__main__":
    main()
