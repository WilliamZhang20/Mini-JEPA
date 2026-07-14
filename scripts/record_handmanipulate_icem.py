"""Record the long-horizon iCEM controller on HandManipulate to an mp4."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.envs import flatten_obs, make_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task
from scripts.eval_handmanipulate_icem import ICEM


def qgeo(a, b):
    a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.degrees(2 * np.arccos(min(1.0, abs(float(a @ b))))))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="handmanipulate_block_rotate_z")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--seed", type=int, default=60000)
    p.add_argument("--max-episode-steps", type=int, default=250)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--candidates", type=int, default=256)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--exec-k", type=int, default=2)
    p.add_argument("--beta", type=float, default=2.5)
    p.add_argument("--disagree-weight", type=float, default=-0.5)
    p.add_argument("--fine-deg", type=float, default=12.0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    dev = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    H = min(args.horizon, int(cfg.get("max_horizon", args.horizon)))
    env = make_env(task.env_id, seed=args.seed, max_episode_steps=args.max_episode_steps,
                   render_mode="rgb_array", width=480, height=480)
    lo = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=dev)
    hi = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=dev)
    mpc = ICEM(wm, norm, spec, dev, H=H, N=args.candidates, iters=args.iters, elite_frac=0.1,
               init_std=0.5, beta=args.beta, keep_frac=0.3, exec_k=args.exec_k,
               disagree_w=args.disagree_weight, reset_w=0.0, path_w=0.25,
               fine_deg=args.fine_deg, fine_H=2, fine_N=128)
    mpc.fine_kappa = 0.0
    ag, dgo = spec.obs_dim, spec.obs_dim + spec.goal_dim

    frames, results = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        mpc.reset()
        s = flatten_obs(obs)
        dg_q = torch.as_tensor(s[dgo + 3:dgo + 7] / (np.linalg.norm(s[dgo + 3:dgo + 7]) + 1e-9),
                               dtype=torch.float32, device=dev)
        start_gap = qgeo(s[ag + 3:ag + 7], s[dgo + 3:dgo + 7])
        term = trunc = False; info = {}
        f = env.render()
        if f is not None:
            frames += [f] * 8  # brief hold on the start frame
        while not (term or trunc):
            obs, _, term, trunc, info = env.step(mpc.act(obs, env, dg_q, lo, hi))
            s = flatten_obs(obs)
            f = env.render()
            if f is not None:
                frames.append(f)
        succ = int(info.get("is_success", 0.0))
        gap = qgeo(s[ag + 3:ag + 7], s[dgo + 3:dgo + 7])
        results.append({"ep": ep, "start_gap_deg": round(start_gap, 1), "final_gap_deg": round(gap, 1), "success": succ})
        if frames:
            frames += [frames[-1]] * 8  # brief hold on the end frame
        print(json.dumps({"event": "record_ep", **results[-1]}), flush=True)
    env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps, format="FFMPEG")
    print(json.dumps({"event": "recorded", "out": str(args.out), "frames": len(frames),
                      "success_rate": round(np.mean([r["success"] for r in results]), 3)}), flush=True)


if __name__ == "__main__":
    main()
