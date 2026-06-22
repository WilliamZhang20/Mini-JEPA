"""Behaviour analysis for a kitchen skill policy: per-subtask completion rates,
completion-order, and the bottleneck (where episodes stall). Helps interpret what
RL/self-imitation actually changed."""

from __future__ import annotations

import argparse
import os
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.envs import make_env, flatten_obs
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task
from scripts.train_diffusion_policy import EpsNet, make_ddpm
from scripts.eval_diffusion_policy import sample_chunk, Scheduler  # type: ignore

TASKS = ["microwave", "kettle", "light switch", "slide cabinet"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--exec-k", type=int, default=4)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task("franka_kitchen", None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev); wm.eval()
    ck = torch.load(args.policy, map_location=dev, weights_only=False)
    net = EpsNet(ck["chunk_dim"], ck["cond_dim"], ck["hidden"], n_blocks=ck["n_blocks"]).to(dev)
    net.load_state_dict(ck["ema"]); net.eval()
    ddpm = make_ddpm(ck["diffusion_steps"], dev)
    H, A_dim, chunk_dim = ck["H"], ck["action_dim"], ck["chunk_dim"]
    HH = int(ck.get("obs_hist", 1)); objective = ck.get("objective", "flow")
    concat_raw = bool(ck.get("concat_raw", False)); subtask_cond = bool(ck.get("subtask_cond", False))
    progress_cond = bool(ck.get("progress_cond", False))

    def enc(obs, prog, tgt):
        x = torch.from_numpy(norm.encode(flatten_obs(obs))).unsqueeze(0).to(dev)
        f = torch.cat([x, wm.encode(x)], dim=-1) if concat_raw else wm.encode(x)
        if progress_cond:
            f = torch.cat([f, torch.full((1, 1), float(prog), device=dev)], dim=-1)
        if subtask_cond:
            oh = torch.zeros(1, 4, device=dev); oh[0, tgt] = 1.0; f = torch.cat([f, oh], dim=-1)
        return f

    per_task = Counter(); counts = Counter(); orders = Counter(); first_missing = Counter()
    n = args.episodes
    for ep in range(n):
        env = make_env(task.env_id, seed=args.seed + ep, max_episode_steps=task.max_episode_steps)
        low, high = env.action_space.low, env.action_space.high
        obs, _ = env.reset(seed=args.seed + ep)
        prog = 0.0; done = set(); order = []; sched = Scheduler(0); tgt = sched.update(done)
        f = enc(obs, prog, tgt); hist = deque([f] * HH, maxlen=HH)
        term = trunc = False; info = {}; step_i = 0; chunk = None; j = 0
        with torch.no_grad():
            while not (term or trunc):
                if step_i % args.exec_k == 0:
                    cond = torch.cat(list(hist), dim=-1)
                    chunk = sample_chunk(net, ddpm, cond, chunk_dim, dev, objective)[0].cpu().numpy().reshape(H, A_dim)
                    j = 0
                a = np.clip(chunk[min(j, H - 1)], low, high).astype(np.float32)
                obs, _, term, trunc, info = env.step(a)
                for t in info.get("step_task_completions", []):
                    if t in TASKS and t not in done:
                        done.add(t); order.append(TASKS.index(t))
                tgt = sched.update(done); prog = float(info.get("tasks_done", 0)) / 4.0
                hist.append(enc(obs, prog, tgt)); step_i += 1; j += 1
        for t in done:
            per_task[t] += 1
        counts[len(done)] += 1
        orders[tuple(order)] += 1
        missing = [TASKS[i] for i in range(4) if TASKS[i] not in done]
        if missing:
            first_missing[missing[0]] += 1   # first canonical task left undone (the bottleneck)
        env.close()

    print(f"\n=== behaviour analysis: {args.policy.name} ({n} eps) ===")
    print("per-subtask completion rate:")
    for t in TASKS:
        print(f"   {t:14s}: {per_task[t]/n:.2f}")
    print("tasks-completed distribution:", {k: counts[k] for k in sorted(counts)})
    print("most common completion orders (task indices):")
    for o, c in orders.most_common(5):
        print(f"   {'->'.join(TASKS[i] for i in o) or '(none)'}: {c}")
    print("bottleneck — first canonical task left undone (over non-perfect eps):")
    for t, c in first_missing.most_common():
        print(f"   {t:14s}: {c}")


if __name__ == "__main__":
    main()
