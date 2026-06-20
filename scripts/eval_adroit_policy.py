"""Evaluate (and optionally record) a behaviour-cloned latent policy on a flat,
non-goal env like Adroit Door, where the goal-env baselines in evaluate.py do not
apply. Rolls the GoalConditionedPolicy on the frozen JEPA latent and reports the
``is_success`` rate.
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
from jepa_robotics.tasks import resolve_task


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="adroit_door")
    p.add_argument("--model-path", type=Path, required=True, help="JEPA world-model .pt")
    p.add_argument("--policy-path", type=Path, required=True, help="BC GoalConditionedPolicy .pt")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--video-out", type=Path, default=None)
    p.add_argument("--video-episodes", type=int, default=6)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    device = torch.device(args.device)
    task = resolve_task(args.task, None)
    model, normalizer, spec, _ = load_jepa_artifact(args.model_path, device)
    policy, _ = load_policy_artifact(args.policy_path, device)
    model.eval(); policy.eval()

    low = None
    successes = []
    for ep in range(args.episodes):
        env = make_env(task.env_id, seed=args.seed + ep, max_episode_steps=task.max_episode_steps)
        if low is None:
            low = env.action_space.low; high = env.action_space.high
        obs, _ = env.reset(seed=args.seed + ep)
        term = trunc = False
        info = {}
        while not (term or trunc):
            state = torch.from_numpy(normalizer.encode(flatten_obs(obs))).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy(model.encode(state))[0].cpu().numpy()
            action = np.clip(action, low, high).astype(np.float32)
            obs, _, term, trunc, info = env.step(action)
        successes.append(float(info.get("is_success", info.get("success", 0.0))))
        env.close()
    rate = float(np.mean(successes))
    print(f'{{"task": "{task.name}", "policy": "{args.policy_path.name}", '
          f'"episodes": {args.episodes}, "success_rate": {rate:.4f}}}', flush=True)

    if args.video_out is not None:
        import imageio.v2 as imageio

        args.video_out.parent.mkdir(parents=True, exist_ok=True)
        frames, vsucc = [], []
        for ep in range(args.video_episodes):
            env = make_env(task.env_id, seed=args.seed + 1000 + ep, max_episode_steps=task.max_episode_steps,
                           render_mode="rgb_array", width=args.width, height=args.height)
            obs, _ = env.reset(seed=args.seed + 1000 + ep)
            f = env.render();  frames.append(f) if f is not None else None
            term = trunc = False; info = {}
            while not (term or trunc):
                state = torch.from_numpy(normalizer.encode(flatten_obs(obs))).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = np.clip(policy(model.encode(state))[0].cpu().numpy(), low, high).astype(np.float32)
                obs, _, term, trunc, info = env.step(action)
                f = env.render();  frames.append(f) if f is not None else None
            vsucc.append(float(info.get("is_success", info.get("success", 0.0))))
            env.close()
        imageio.mimsave(args.video_out, frames, fps=args.fps, format="FFMPEG")
        print(f'{{"event": "recorded", "video": "{args.video_out}", '
              f'"episodes": {args.video_episodes}, "success_rate": {float(np.mean(vsucc)):.3f}}}', flush=True)


if __name__ == "__main__":
    main()
