"""Collect Adroit trajectories from a retained latent BC policy.

Use this only as a data-generation bridge for SSL replacement experiments. The
runtime controller trained from the output should still be future-conditioned
latent planning, not this BC policy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.envs import flatten_obs, make_env
from jepa_robotics.evaluate import load_jepa_artifact, load_policy_artifact
from jepa_robotics.tasks import resolve_task


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="adroit_relocate")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--policy-path", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=72000)
    p.add_argument("--keep-success-only", action="store_true", default=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    task = resolve_task(args.task, None)
    wm, norm, _spec, _cfg = load_jepa_artifact(args.model_path, dev)
    policy, _ = load_policy_artifact(args.policy_path, dev)
    wm.eval()
    policy.eval()

    states_out, actions_out, successes = [], [], []
    for ep in range(args.episodes):
        env = make_env(task.env_id, seed=args.seed + ep, max_episode_steps=task.max_episode_steps)
        low, high = env.action_space.low, env.action_space.high
        obs, _ = env.reset(seed=args.seed + ep)
        states = [flatten_obs(obs).copy()]
        actions = []
        term = trunc = False
        info = {}
        while not (term or trunc):
            state = torch.from_numpy(norm.encode(flatten_obs(obs))).unsqueeze(0).to(dev)
            with torch.no_grad():
                action = policy(wm.encode(state))[0].cpu().numpy()
            action = np.clip(action, low, high).astype(np.float32)
            obs, _, term, trunc, info = env.step(action)
            states.append(flatten_obs(obs).copy())
            actions.append(action.copy())
        env.close()
        success = float(info.get("is_success", info.get("success", 0.0)))
        successes.append(success)
        keep = (not args.keep_success_only) or success > 0
        if keep:
            states_out.append(np.asarray(states, dtype=np.float32))
            actions_out.append(np.asarray(actions, dtype=np.float32))
        print(json.dumps({"event": "collect_bc_ep", "episode": ep, "success": success, "kept": int(keep)}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        states=np.array(states_out, dtype=object),
        actions=np.array(actions_out, dtype=object),
        rewards=np.array([np.ones(len(a), dtype=np.float32) for a in actions_out], dtype=object),
    )
    print(
        json.dumps(
            {
                "event": "collect_bc_saved",
                "out": str(args.out),
                "episodes": int(args.episodes),
                "kept": int(len(states_out)),
                "success_rate": float(np.mean(successes) if successes else 0.0),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
