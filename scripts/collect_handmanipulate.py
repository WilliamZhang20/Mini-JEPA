"""Collect self-supervised exploration data for the Shadow Hand HandManipulate
suite (Block/Egg/Pen). No demos exist for in-hand reorientation, and the
DexterousJEPA world model is self-supervised, so any dynamics-rich trajectories
suffice. White-noise actions drop the object almost immediately; temporally
CORRELATED actions (Ornstein-Uhlenbeck) keep the fingers moving coherently and
the object in play far longer, giving the world model real contact/rotation
dynamics to model. Saves an episodes npz consumed by train_dexterous_jepa.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.data import Episode, save_episodes_npz
from jepa_robotics.envs import flatten_obs, make_env, obs_spec_from_env
from jepa_robotics.tasks import resolve_task


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-steps", type=int, default=200000)
    p.add_argument("--ep-len", type=int, default=100)
    p.add_argument("--ou-theta", type=float, default=0.15, help="OU mean-reversion (lower = more temporally correlated)")
    p.add_argument("--ou-sigma", type=float, default=0.3, help="OU noise scale")
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument(
        "--action-slew-limit",
        type=float,
        default=0.0,
        help="Maximum per-step change per actuator after OU sampling (0 disables).",
    )
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    task = resolve_task(args.task, None)
    env = make_env(task.env_id, seed=args.seed, max_episode_steps=args.ep_len)
    spec = obs_spec_from_env(env)
    lo, hi = env.action_space.low, env.action_space.high
    rng = np.random.default_rng(args.seed)

    episodes: list[Episode] = []
    action_delta_sq: list[float] = []
    total = 0
    ep_i = 0
    while total < args.num_steps:
        obs, _ = env.reset(seed=args.seed + ep_i)
        states = [flatten_obs(obs)]
        actions = []
        a = np.zeros(spec.action_dim, dtype=np.float32)  # OU state
        prev_act = np.zeros(spec.action_dim, dtype=np.float32)
        term = trunc = False
        while not (term or trunc) and total < args.num_steps:
            # Ornstein-Uhlenbeck correlated exploration
            a = a + args.ou_theta * (-a) + args.ou_sigma * rng.standard_normal(spec.action_dim).astype(np.float32)
            act = np.clip(a * args.action_scale, lo, hi).astype(np.float32)
            if args.action_slew_limit > 0:
                delta = np.clip(
                    act - prev_act,
                    -args.action_slew_limit,
                    args.action_slew_limit,
                )
                act = np.clip(prev_act + delta, lo, hi).astype(np.float32)
            action_delta_sq.append(float(np.mean(np.square(act - prev_act))))
            prev_act = act
            obs, _, term, trunc, _ = env.step(act)
            actions.append(act)
            states.append(flatten_obs(obs))
            total += 1
        if len(actions) >= 2:
            episodes.append(Episode(states=np.asarray(states, np.float32), actions=np.asarray(actions, np.float32)))
        ep_i += 1
        if ep_i % 200 == 0:
            print(json.dumps({"event": "collect", "steps": total, "target": args.num_steps, "episodes": len(episodes)}), flush=True)
    env.close()
    save_episodes_npz(args.out, episodes, spec)
    print(json.dumps({
        "event": "collected",
        "task": args.task,
        "episodes": len(episodes),
        "steps": total,
        "action_delta_rms": float(np.sqrt(np.mean(action_delta_sq))),
        "action_slew_limit": args.action_slew_limit,
        "out": str(args.out),
    }), flush=True)


if __name__ == "__main__":
    main()
