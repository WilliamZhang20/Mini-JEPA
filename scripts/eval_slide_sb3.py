"""Standalone deterministic confirmation eval for a JEPA-latent SB3 (TQC) checkpoint.

Loads an SB3 .zip controller that runs on JEPA latent observations and reports the
success rate over a large batch of episodes. Reports both a single fixed-seed batch
and the pooled rate across several seed offsets so we can tell a genuine high-water
mark from a lucky 10-episode EvalCallback window.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.envs import make_env
from jepa_robotics.tasks import resolve_task


def eval_checkpoint(model, env_id, max_steps, base_seed, episodes, deterministic):
    successes = []
    for ep in range(episodes):
        env = make_env(env_id, seed=base_seed + ep, max_episode_steps=max_steps)
        obs, _ = env.reset(seed=base_seed + ep)
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
        successes.append(float(info.get("is_success", 0.0)))
        env.close()
    return np.array(successes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="fetch_slide")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--stochastic", action="store_true")
    args = p.parse_args()

    from sb3_contrib import TQC

    task = resolve_task(args.task, None)
    env_id = task.env_id
    max_steps = task.max_episode_steps

    probe = make_env(env_id, seed=args.seed, max_episode_steps=max_steps)
    model = TQC.load(str(args.checkpoint), env=probe, device=args.device)

    succ = eval_checkpoint(
        model, env_id, max_steps, args.seed, args.episodes, deterministic=not args.stochastic
    )
    probe.close()
    print(
        f'{{"checkpoint": "{args.checkpoint.name}", "episodes": {args.episodes}, '
        f'"deterministic": {str(not args.stochastic).lower()}, '
        f'"success_rate": {succ.mean():.4f}, "n_success": {int(succ.sum())}}}',
        flush=True,
    )


if __name__ == "__main__":
    main()
