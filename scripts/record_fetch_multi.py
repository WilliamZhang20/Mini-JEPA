"""Record longer multi-episode videos of the UNIFIED Fetch controller.

One JEPA world model + one policy (Roadmap B), run through the canonical adapter
on each sub-task (reach/push/pick) for several episodes, concatenated into one
MP4 per task. Goals are randomized per episode (and pick alternates table /
mid-air targets) so the clip shows variety.

    PYTHONNOUSERSITE=1 MUJOCO_GL=egl python scripts/record_fetch_multi.py \
        --model-path runs/fetch_multi/fetch_multi_model.pt \
        --policy-path runs/fetch_multi/fetch_multi_policy.pt \
        --episodes 6 --out-dir runs/fetch_multi
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa_robotics.envs import CANONICAL_OBJECT_PRESENT_IDX, FETCH_MULTI_SUBTASKS, make_env
from jepa_robotics.evaluate import JEPAMPCPolicy, load_jepa_artifact, load_policy_artifact

SUBTASK_REACH_WEIGHT = {"reach": 0.0, "push": 0.0, "pick": 0.1}
SUBTASK_ACTION_STD = {"reach": 0.5, "push": 0.3, "pick": 0.5}
TABLE_Z = 0.425


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--policy-path", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--mpc-candidates", type=int, default=128)
    p.add_argument("--mpc-horizon", type=int, default=12)
    p.add_argument("--cem-iters", type=int, default=4)
    p.add_argument("--policy-proposal-fraction", type=float, default=0.5)
    p.add_argument("--out-dir", type=Path, default=Path("runs/fetch_multi"))
    p.add_argument("--subtasks", default="reach,push,pick")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    model, normalizer, spec, _ = load_jepa_artifact(args.model_path, device)
    policy_net, _ = load_policy_artifact(args.policy_path, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    want = set(args.subtasks.split(","))

    for st in FETCH_MULTI_SUBTASKS:
        name, env_id, controller = st["name"], st["env_id"], st["controller"]
        if name not in want:
            continue
        max_steps = st["max_episode_steps"]
        env = make_env(env_id, seed=args.seed, max_episode_steps=max_steps,
                       canonical_task=name, render_mode="rgb_array")
        unwrapped = env.unwrapped
        controller_obj = JEPAMPCPolicy(
            model=model, normalizer=normalizer, spec=spec, device=device,
            candidates=args.mpc_candidates, horizon=args.mpc_horizon, seed=args.seed + 10_000,
            method="cem", score_mode="manip", cem_iters=args.cem_iters,
            action_std=SUBTASK_ACTION_STD[name],
            manip_reach_weight=SUBTASK_REACH_WEIGHT[name], manip_path_weight=0.3,
            scripted_controller=controller, policy_net=policy_net,
            policy_proposal_fraction=args.policy_proposal_fraction,
            object_present_idx=CANONICAL_OBJECT_PRESENT_IDX,
        )
        frames, successes = [], []
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            controller_obj.reset()
            # Pick: force a mid-air goal on alternate episodes to show lifting.
            if name == "pick" and ep % 2 == 1:
                g = np.asarray(unwrapped.goal, dtype=np.float64).copy()
                g[2] = TABLE_Z + float(np.random.default_rng(args.seed + ep).uniform(0.13, 0.30))
                unwrapped.goal = g
                obs["desired_goal"] = g.astype(np.float32)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            terminated = truncated = False
            info: dict = {}
            while not (terminated or truncated):
                action = controller_obj.act(obs, env)
                obs, _, terminated, truncated, info = env.step(action)
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            successes.append(float(info.get("is_success", 0.0)))
        env.close()
        out = args.out_dir / f"fetch_multi_{name}_agent.mp4"
        imageio.mimsave(out, frames, fps=args.fps, format="FFMPEG")
        print(json.dumps({"event": "recorded", "subtask": name, "path": str(out),
                          "episodes": args.episodes, "success": float(np.mean(successes)),
                          "frames": len(frames)}), flush=True)


if __name__ == "__main__":
    main()
