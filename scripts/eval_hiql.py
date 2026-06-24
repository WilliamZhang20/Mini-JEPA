"""Evaluate / record a trained HIQL agent on AntMaze (self-contained hierarchy:
pi_high proposes a subgoal every k steps, pi_low reaches it)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MINARI_DATASETS_PATH", "/u5/w223zhan/jepa-mini/.cache/minari")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa_robotics.envs import flatten_obs, make_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task
from scripts.train_hiql import PiHigh, PiLow, VNet


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--hiql", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--record-out", type=Path, default=None)
    p.add_argument("--record-tries", type=int, default=15)
    p.add_argument("--record-n", type=int, default=1, help="number of successful episodes to concatenate")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    task = resolve_task(args.task, None)
    dev = torch.device(args.device)
    wm, norm, spec, _ = load_jepa_artifact(args.model_path, dev)
    art = torch.load(args.hiql, map_location=dev, weights_only=False); c = art["config"]
    raw_only = bool(c.get("raw_only", False))
    pil = PiLow(c["rep_dim"], c["goal_dim"], c["action_dim"], c["hidden"]).to(dev); pil.load_state_dict(art["pi_low"]); pil.eval()
    pih = PiHigh(c["rep_dim"], c["goal_dim"], c["hidden"]).to(dev); pih.load_state_dict(art["pi_high"]); pih.eval()
    k = c["subgoal_k"]
    env0 = make_env(task.env_id, seed=0, max_episode_steps=task.max_episode_steps)
    a_lo = env0.action_space.low; a_hi = env0.action_space.high; env0.close()

    @torch.no_grad()
    def run(seed, render=False):
        rmode = "rgb_array" if render else None
        e = make_env(task.env_id, seed=seed, max_episode_steps=task.max_episode_steps, render_mode=rmode)
        obs, _ = e.reset(seed=seed)
        goal = torch.as_tensor(obs["desired_goal"], dtype=torch.float32, device=dev).unsqueeze(0)
        term = trunc = False; info = {}; t = 0; sg = goal; frames = []
        if render:
            f = e.render();  frames.append(f) if f is not None else None
        while not (term or trunc):
            r = torch.from_numpy(norm.encode(flatten_obs(obs))).unsqueeze(0).to(dev)
            rep = r if raw_only else torch.cat([r, wm.encode(r)], dim=1)
            cur = torch.as_tensor(obs["achieved_goal"], dtype=torch.float32, device=dev).unsqueeze(0)
            if t % k == 0:
                sg = cur + pih(rep, goal)
            a = pil(rep, sg)[0].cpu().numpy()
            a = np.clip(a_lo + (a + 1) * 0.5 * (a_hi - a_lo), a_lo, a_hi).astype(np.float32)
            obs, _, term, trunc, info = e.step(a); t += 1
            if render:
                f = e.render();  frames.append(f) if f is not None else None
        e.close()
        return float(info.get("is_success", info.get("success", 0.0))), frames

    succ = [run(args.seed + i)[0] for i in range(args.episodes)]
    print(json.dumps({"task": task.name, "policy": "HIQL", "episodes": args.episodes,
                      "success_rate": round(float(np.mean(succ)), 4)}), flush=True)

    if args.record_out is not None:
        import imageio.v2 as imageio
        clip, got = [], 0
        for i in range(args.record_tries):
            s, frames = run(args.seed + 1000 + i, render=True)
            if s > 0.5 and frames:
                clip.extend(frames[::2]); got += 1
                if got >= args.record_n:
                    break
        if clip:
            args.record_out.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(args.record_out, clip, fps=30, format="FFMPEG")
            print(json.dumps({"event": "recorded", "path": str(args.record_out),
                              "episodes": got, "frames": len(clip)}), flush=True)
        else:
            print(json.dumps({"event": "no_success_to_record"}), flush=True)


if __name__ == "__main__":
    main()
