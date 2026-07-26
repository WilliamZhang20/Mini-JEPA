"""Build a goal-conditioned local waypoint memory from successful demos."""
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
    parser.add_argument("--hwm", type=Path, required=True)
    parser.add_argument("--episodes-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=1000)
    parser.add_argument("--success-radius", type=float, default=0.5)
    parser.add_argument("--waypoint-horizon", type=int, default=8)
    parser.add_argument("--min-displacement", type=float, default=0.25)
    parser.add_argument("--sample-step", type=int, default=None)
    args = parser.parse_args()

    task = resolve_task(args.task, None)
    env = make_env(task.env_id, seed=0, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(env)
    env.close()
    achieved = slice(spec.obs_dim, spec.obs_dim + spec.goal_dim)
    desired = slice(spec.obs_dim + spec.goal_dim, spec.obs_dim + 2 * spec.goal_dim)
    hwm = torch.load(args.hwm, map_location="cpu", weights_only=False)
    stride = int(hwm["config"]["stride"])
    waypoint_horizon = int(args.waypoint_horizon)
    sample_step = args.sample_step or max(1, waypoint_horizon // 2)

    current: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    waypoints: list[np.ndarray] = []
    successful_episodes = 0
    rejected_stationary = 0
    episodes = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    for episode in episodes:
        states = np.asarray(episode.states, dtype=np.float32)
        if len(states) <= waypoint_horizon:
            continue
        goal = states[0, desired]
        if float(np.linalg.norm(states[:, achieved] - goal[None], axis=1).min()) > args.success_radius:
            continue
        successful_episodes += 1
        for t in range(0, len(states) - waypoint_horizon, sample_step):
            here = states[t, achieved]
            there = states[t + waypoint_horizon, achieved]
            if float(np.linalg.norm(there - here)) < args.min_displacement:
                rejected_stationary += 1
                continue
            current.append(here.copy())
            goals.append(goal.copy())
            waypoints.append(there.copy())

    if not current:
        raise RuntimeError("No successful non-stationary waypoint transitions found")
    artifact = {
        "current": np.asarray(current, dtype=np.float32),
        "goals": np.asarray(goals, dtype=np.float32),
        "waypoints": np.asarray(waypoints, dtype=np.float32),
        "config": {
            "architecture": "goal_conditioned_waypoint_memory_v1",
            "task": task.name,
            "stride": stride,
            "waypoint_horizon": waypoint_horizon,
            "sample_step": sample_step,
            "success_radius": args.success_radius,
            "min_displacement": args.min_displacement,
            "source_episodes": len(episodes),
            "successful_episodes": successful_episodes,
            "episodes_npz": str(args.episodes_npz),
            "hwm": str(args.hwm),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.out)
    print(
        json.dumps(
            {
                "event": "waypoint_memory_saved",
                "path": str(args.out),
                "entries": len(current),
                "successful_episodes": successful_episodes,
                "source_episodes": len(episodes),
                "rejected_stationary": rejected_stationary,
                "stride": stride,
                "waypoint_horizon": waypoint_horizon,
                "sample_step": sample_step,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
