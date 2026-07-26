"""Build a goal-conditioned action memory from successful AntMaze prefixes."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jepa_robotics.data import load_episodes_npz
from jepa_robotics.envs import make_env, obs_spec_from_env
from jepa_robotics.tasks import resolve_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=1000)
    parser.add_argument("--success-radius", type=float, default=0.5)
    parser.add_argument("--xy-weight", type=float, default=4.0)
    parser.add_argument("--goal-weight", type=float, default=8.0)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--route-start", type=float, nargs=2, default=None)
    parser.add_argument("--route-goal", type=float, nargs=2, default=None)
    parser.add_argument("--route-radius", type=float, default=1.5)
    args = parser.parse_args()

    task = resolve_task(args.task, None)
    env = make_env(task.env_id, seed=0, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(env)
    env.close()
    achieved = slice(spec.obs_dim, spec.obs_dim + spec.goal_dim)
    desired = slice(spec.obs_dim + spec.goal_dim, spec.obs_dim + 2 * spec.goal_dim)

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    successful = 0
    episodes = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    for episode in episodes:
        ep_states = np.asarray(episode.states, dtype=np.float32)
        ep_actions = np.asarray(episode.actions, dtype=np.float32)
        goal = ep_states[0, desired]
        if args.route_start is not None or args.route_goal is not None:
            if args.route_start is None or args.route_goal is None:
                parser.error("--route-start and --route-goal must be provided together")
            if (
                np.linalg.norm(
                    ep_states[0, achieved] - np.asarray(args.route_start)
                ) > args.route_radius
                or np.linalg.norm(goal - np.asarray(args.route_goal))
                > args.route_radius
            ):
                continue
        distances = np.linalg.norm(ep_states[:, achieved] - goal[None], axis=1)
        hits = np.flatnonzero(distances <= args.success_radius)
        if len(hits) == 0:
            continue
        successful += 1
        stop = min(len(ep_actions), int(hits[0]) + 1)
        for t in range(0, max(0, stop - args.chunk + 1)):
            states.append(ep_states[t : t + 1])
            actions.append(ep_actions[t : t + args.chunk][None])

    raw = np.concatenate(states, axis=0)
    action_array = np.concatenate(actions, axis=0)
    mean = raw.mean(axis=0)
    std = np.maximum(raw.std(axis=0), 0.05)
    weight = np.ones(raw.shape[1], dtype=np.float32)
    weight[achieved] = args.xy_weight
    weight[desired] = args.goal_weight
    features = ((raw - mean) / std * weight).astype(np.float32)
    artifact = {
        "features": features,
        "actions": action_array.astype(np.float32),
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "feature_weight": weight,
        "config": {
            "architecture": "goal_conditioned_action_memory_v1",
            "task": task.name,
            "source_episodes": len(episodes),
            "successful_episodes": successful,
            "entries": len(features),
            "success_radius": args.success_radius,
            "xy_weight": args.xy_weight,
            "goal_weight": args.goal_weight,
            "chunk": args.chunk,
            "route_start": args.route_start,
            "route_goal": args.route_goal,
            "route_radius": args.route_radius,
            "episodes_npz": str(args.episodes_npz),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.out)
    print(
        json.dumps(
            {
                "event": "action_memory_saved",
                "path": str(args.out),
                "entries": len(features),
                "successful_episodes": successful,
                "source_episodes": len(episodes),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
