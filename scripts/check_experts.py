"""Measure the success rate of the scripted reference controllers.

These controllers (``scripted_pick_place_action`` / ``scripted_push_action`` in
``jepa_robotics.data``) are used both to collect training data and as the
"conventional method" baseline in evaluation, so their quality matters a lot.
Run this after any change to them:

    PYTHONNOUSERSITE=1 MUJOCO_GL=egl python scripts/check_experts.py
    PYTHONNOUSERSITE=1 MUJOCO_GL=egl python scripts/check_experts.py --episodes 200 --gain 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa_robotics.data import scripted_action
from jepa_robotics.envs import make_env

TASKS = [
    ("FetchPickAndPlace-v4", "pick_place", 100),
    ("FetchPush-v4", "push", 100),
]


def evaluate(env_id: str, controller: str, gain: float, episodes: int, max_steps: int) -> dict:
    env = make_env(env_id, seed=0, max_episode_steps=max_steps)
    rng = np.random.default_rng(0)
    successes, final_dists = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            action = scripted_action(obs, env.action_space.shape[0], gain, controller, env, rng)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, _, terminated, truncated, info = env.step(action)
        successes.append(float(info.get("is_success", 0.0)))
        final_dists.append(float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])))
    env.close()
    return {
        "success": float(np.mean(successes)),
        "final_dist": float(np.mean(final_dists)),
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--gain", type=float, default=10.0)
    args = parser.parse_args()
    for env_id, controller, max_steps in TASKS:
        r = evaluate(env_id, controller, args.gain, args.episodes, max_steps)
        print(
            f"{env_id:24s} gain={args.gain:<5g} "
            f"success={r['success']:.3f} final_dist={r['final_dist']:.4f} "
            f"(n={r['episodes']})"
        )


if __name__ == "__main__":
    main()
