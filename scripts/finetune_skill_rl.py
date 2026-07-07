"""Advantage-Weighted Regression (AWR) online RL fine-tuning of the flow skill.

Self-imitation only *keeps* full-4 successes (binary). A real RL objective uses the
full reward signal with partial credit and a baseline: maximise E[return] by
weighting each transition's flow-matching loss by exp(beta * (return - baseline)),
where return = #subtasks completed in that episode. Unlike self-imitation it can
push *down* low-return behaviour and *up* near-misses, so it can improve beyond the
demonstrated/self-collected distribution (the RPL online-RL phase, in spirit).

On-policy AWR loop: collect K episodes with the current policy (recording per-episode
return), advantage-weight every transition, fine-tune the flow skill for M steps,
update the baseline, repeat. Warm-started from the best self-imitation policy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MINARI_DATASETS_PATH", "/u5/w223zhan/jepa-mini/.cache/minari")

from jepa_robotics.envs import make_env, flatten_obs
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task
from jepa_robotics.algos.priors import EpsNet, make_ddpm
from scripts.eval_diffusion_policy import DEFAULT_TASKS, sample_chunk, Scheduler  # type: ignore

TASKS = DEFAULT_TASKS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--init-from", type=Path, required=True, help="warm-start policy (its config defines conditioning)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--collect-eps", type=int, default=80, help="episodes collected per AWR iteration")
    p.add_argument("--finetune-steps", type=int, default=8000, help="weighted-regression steps per iteration")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--beta", type=float, default=2.0, help="AWR temperature (advantage weight = exp(beta*adv))")
    p.add_argument("--wmax", type=float, default=20.0, help="max advantage weight (clip)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--exec-k", type=int, default=4)
    p.add_argument("--seed", type=int, default=300000)
    p.add_argument("--eval-eps", type=int, default=50)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task("franka_kitchen", None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    L = int(cfg["latent_dim"])

    ck = torch.load(args.init_from, map_location=dev, weights_only=False)
    H, A_dim, chunk_dim = ck["H"], ck["action_dim"], ck["chunk_dim"]
    HH = int(ck["obs_hist"]); cond_dim = ck["cond_dim"]
    concat_raw = bool(ck.get("concat_raw", False)); subtask_cond = bool(ck.get("subtask_cond", False))
    progress_cond = bool(ck.get("progress_cond", False)); objective = ck.get("objective", "flow")
    net = EpsNet(chunk_dim, cond_dim, ck["hidden"], n_blocks=ck["n_blocks"]).to(dev)
    net.load_state_dict(ck["ema"])
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    ddpm = make_ddpm(ck["diffusion_steps"], dev); Tsteps = ck["diffusion_steps"]

    @torch.no_grad()
    def featurize_obs(obs, progress, target):
        x = torch.from_numpy(norm.encode(flatten_obs(obs))).unsqueeze(0).to(dev)
        zz = wm.encode(x)
        f = torch.cat([x, zz], dim=-1) if concat_raw else zz
        if progress_cond:
            f = torch.cat([f, torch.full((1, 1), float(progress), device=dev)], dim=-1)
        if subtask_cond:
            oh = torch.zeros(1, 4, device=dev); oh[0, target] = 1.0
            f = torch.cat([f, oh], dim=-1)
        return f

    @torch.no_grad()
    def rollout(seed, policy_net, record=True):
        """One episode; returns (cond_seq[T,cond_dim], chunks[T,H*A], return) and tasks_done."""
        from collections import deque
        env = make_env(task.env_id, seed=seed, max_episode_steps=task.max_episode_steps)
        low, high = env.action_space.low, env.action_space.high
        obs, _ = env.reset(seed=seed)
        prog = 0.0; done = set(); sched = Scheduler(TASKS, 0); tgt = sched.update(done)
        f = featurize_obs(obs, prog, tgt); hist = deque([f] * HH, maxlen=HH)
        term = trunc = False; info = {}; step_i = 0; chunk = None; j = 0
        conds, acts = [], []
        while not (term or trunc):
            cond = torch.cat(list(hist), dim=-1)
            if step_i % args.exec_k == 0:
                chunk = sample_chunk(policy_net, ddpm, cond, chunk_dim, dev, objective)[0].cpu().numpy().reshape(H, A_dim)
                j = 0
                if record:  # record (cond, sampled chunk) at the sampling step — AWR reweights the policy's own actions
                    conds.append(cond[0].cpu().numpy()); acts.append(chunk.reshape(-1).astype(np.float32))
            a = np.clip(chunk[min(j, H - 1)], low, high).astype(np.float32)
            obs, _, term, trunc, info = env.step(a)
            done |= set(info.get("step_task_completions", [])); tgt = sched.update(done)
            prog = float(info.get("tasks_done", 0)) / 4.0
            hist.append(featurize_obs(obs, prog, tgt)); step_i += 1; j += 1
        env.close()
        nt = int(info.get("tasks_done", 0))
        return (np.asarray(conds, np.float32), np.asarray(acts, np.float32), nt) if conds else (None, None, nt)

    baseline = 2.5
    seed_ctr = args.seed
    for it in range(1, args.iterations + 1):
        # --- collect on-policy ---
        C, Ac, W, rets = [], [], [], []
        for _ in range(args.collect_eps):
            c, a, nt = rollout(seed_ctr, net); seed_ctr += 1
            rets.append(nt)
            if c is None:
                continue
            adv = nt - baseline
            w = float(np.clip(np.exp(args.beta * adv), 0.0, args.wmax))
            C.append(c); Ac.append(a); W.append(np.full(len(c), w, np.float32))
        meanret = float(np.mean(rets)); full4 = float(np.mean(np.array(rets) >= 4))
        baseline = 0.9 * baseline + 0.1 * meanret
        Cc = torch.from_numpy(np.concatenate(C)).to(dev); Aa = torch.from_numpy(np.concatenate(Ac)).to(dev)
        Ww = torch.from_numpy(np.concatenate(W)).to(dev); Nn = len(Cc)
        # --- advantage-weighted flow-matching fine-tune ---
        net.train()
        for step in range(args.finetune_steps):
            idx = torch.randint(0, Nn, (args.batch_size,), device=dev)
            a0 = Aa[idx]; c = Cc[idx]; w = Ww[idx]
            tau = torch.rand(a0.shape[0], device=dev)
            x0 = torch.randn_like(a0); x_t = (1 - tau)[:, None] * x0 + tau[:, None] * a0
            pred = net(x_t, tau * Tsteps, c)
            loss = (w * ((pred - (a0 - x0)) ** 2).mean(-1)).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            with torch.no_grad():
                for k, v in net.state_dict().items():
                    ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
        net.eval()
        print(json.dumps({"event": "awr_iter", "iter": it, "collect_meanret": round(meanret, 3),
                          "collect_full4": round(full4, 3), "baseline": round(baseline, 3),
                          "transitions": Nn, "w_mean": round(float(Ww.mean()), 2)}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ema": ema, "state_dict": net.state_dict(), "chunk_dim": chunk_dim, "cond_dim": cond_dim,
                "action_dim": A_dim, "H": H, "obs_hist": HH, "latent_dim": L, "hidden": ck["hidden"],
                "n_blocks": ck["n_blocks"], "diffusion_steps": Tsteps, "objective": objective,
                "concat_raw": concat_raw, "subtask_cond": subtask_cond, "progress_cond": progress_cond,
                "raw_obs": False}, args.out)
    print(json.dumps({"event": "awr_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
