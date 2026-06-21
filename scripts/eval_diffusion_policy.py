"""Evaluate the action-chunked diffusion policy on FrankaKitchen.

Receding-horizon control: encode obs -> z (JEPA latent), sample an H-action chunk by
DDPM reverse diffusion conditioned on z, execute the first ``exec-k`` actions, then
replan. Uses the EMA weights (diffusion policies sample much better from EMA).
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
from jepa_robotics.tasks import resolve_task
from scripts.train_diffusion_policy import EpsNet, make_ddpm


@torch.no_grad()
def sample_chunk(net, ddpm, cond, chunk_dim, device, objective="diffusion", flow_steps=16):
    """Sample one action chunk conditioned on cond [B, condL].
    diffusion -> DDPM reverse; flow -> Euler ODE integration of the velocity field."""
    B = cond.shape[0]
    if objective == "flow":
        T = ddpm["T"]
        x = torch.randn(B, chunk_dim, device=device)
        dt = 1.0 / flow_steps
        for i in range(flow_steps):
            tau = torch.full((B,), i * dt, device=device)
            v = net(x, tau * T, cond)            # same time scaling as training
            x = x + dt * v
        return x
    betas, alphas, abar, T = ddpm["betas"], ddpm["alphas"], ddpm["abar"], ddpm["T"]
    a = torch.randn(B, chunk_dim, device=device)
    for t in reversed(range(T)):
        tt = torch.full((B,), t, device=device, dtype=torch.long)
        eps = net(a, tt, cond)
        mean = (a - betas[t] / torch.sqrt(1 - abar[t]) * eps) / torch.sqrt(alphas[t])
        if t > 0:
            a = mean + torch.sqrt(betas[t]) * torch.randn_like(a)
        else:
            a = mean
    return a


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="franka_kitchen")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--exec-k", type=int, default=4, help="actions executed per replan (receding horizon)")
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--video-out", type=Path, default=None, help="record an mp4 of the policy")
    p.add_argument("--video-episodes", type=int, default=6)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    ck = torch.load(args.policy, map_location=dev, weights_only=False)
    net = EpsNet(ck["chunk_dim"], ck["cond_dim"], ck["hidden"], n_blocks=ck["n_blocks"]).to(dev)
    net.load_state_dict(ck["ema"] if args.use_ema else ck["state_dict"])
    net.eval()
    ddpm = make_ddpm(ck["diffusion_steps"], dev)
    H, A_dim, chunk_dim = ck["H"], ck["action_dim"], ck["chunk_dim"]
    HH = int(ck.get("obs_hist", 1))
    objective = ck.get("objective", "diffusion")
    raw_obs = bool(ck.get("raw_obs", False))
    concat_raw = bool(ck.get("concat_raw", False))

    def encode(obs):
        x = torch.from_numpy(norm.encode(flatten_obs(obs))).unsqueeze(0).to(dev)
        if raw_obs:
            return x
        z = wm.encode(x)
        return torch.cat([x, z], dim=-1) if concat_raw else z   # [raw++latent] or latent

    from collections import deque
    tasks = []
    for ep in range(args.episodes):
        env = make_env(task.env_id, seed=args.seed + ep, max_episode_steps=task.max_episode_steps)
        low, high = env.action_space.low, env.action_space.high
        obs, _ = env.reset(seed=args.seed + ep)
        z = encode(obs)
        hist = deque([z] * HH, maxlen=HH)             # consecutive latent history
        term = trunc = False; info = {}; step_i = 0; chunk = None; j = 0
        while not (term or trunc):
            if step_i % args.exec_k == 0:
                cond = torch.cat(list(hist), dim=-1)  # [1, HH*L]
                chunk = sample_chunk(net, ddpm, cond, chunk_dim, dev, objective)[0].cpu().numpy().reshape(H, A_dim)
                j = 0
            a = np.clip(chunk[min(j, H - 1)], low, high).astype(np.float32)
            obs, _, term, trunc, info = env.step(a)
            hist.append(encode(obs))
            step_i += 1; j += 1
        tasks.append(int(info.get("tasks_done", 0)))
        env.close()
    tasks = np.array(tasks)
    print(f"RESULT diffusion-policy(exec_k={args.exec_k}) mean_tasks={tasks.mean():.2f}/4 "
          f"full4={np.mean(tasks >= 4):.2f} >=1={np.mean(tasks >= 1):.2f} "
          f">=2={np.mean(tasks >= 2):.2f} >=3={np.mean(tasks >= 3):.2f}", flush=True)

    if args.video_out is not None:
        import imageio.v2 as imageio
        args.video_out.parent.mkdir(parents=True, exist_ok=True)
        frames, vtasks = [], []
        for ep in range(args.video_episodes):
            env = make_env(task.env_id, seed=args.seed + 5000 + ep, max_episode_steps=task.max_episode_steps,
                           render_mode="rgb_array", width=args.width, height=args.height)
            low, high = env.action_space.low, env.action_space.high
            obs, _ = env.reset(seed=args.seed + 5000 + ep)
            z = encode(obs); hist = deque([z] * HH, maxlen=HH)
            term = trunc = False; info = {}; step_i = 0; chunk = None; j = 0
            f = env.render()
            if f is not None: frames.append(f)
            while not (term or trunc):
                if step_i % args.exec_k == 0:
                    cond = torch.cat(list(hist), dim=-1)
                    chunk = sample_chunk(net, ddpm, cond, chunk_dim, dev, objective)[0].cpu().numpy().reshape(H, A_dim)
                    j = 0
                a = np.clip(chunk[min(j, H - 1)], low, high).astype(np.float32)
                obs, _, term, trunc, info = env.step(a)
                hist.append(encode(obs)); step_i += 1; j += 1
                f = env.render()
                if f is not None: frames.append(f)
            vtasks.append(int(info.get("tasks_done", 0)))
            env.close()
        imageio.mimsave(args.video_out, frames, fps=args.fps, format="FFMPEG")
        print(f'{{"event": "recorded", "video": "{args.video_out}", '
              f'"episodes": {args.video_episodes}, "mean_tasks": {float(np.mean(vtasks)):.2f}}}', flush=True)


if __name__ == "__main__":
    main()
