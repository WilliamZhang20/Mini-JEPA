"""Targeted flip-recovery trial collection for the AntMaze walker.

Natural recoveries in plain self-trials are rare (~1.2% of flipped 8-step
windows), so the first recovery flow was trained on only ~500 accidental
flail-recoveries. This collector goes hunting: it drives episodes with the
canonical gait walker toward random subgoals (which flips the ant roughly half
the time), and while the torso is tipped (uprightness < --enter) it switches to
OU-correlated exploration until upright again (> --exit) or the episode ends.
Everything is recorded in the standard episode npz format; the recovery miner
(train_recovery_walker.py) then keeps whichever chunks actually raised
uprightness — self-supervised trial data for the one skill the demos cannot
teach.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.algos.maze_low_level import LowLevelFlow
from jepa_robotics.data import Episode, save_episodes_npz
from jepa_robotics.envs import flatten_obs, make_env
from jepa_robotics.tasks import resolve_task
from scripts.train_recovery_walker import uprightness


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--jepa-model", type=Path, required=True)
    p.add_argument("--walker", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=70000)
    p.add_argument("--enter", type=float, default=0.5)
    p.add_argument("--exit", dest="exit_u", type=float, default=0.8)
    p.add_argument("--ou-theta", type=float, default=0.15)
    p.add_argument("--ou-sigma", type=float, default=0.6)
    p.add_argument("--subgoal-h", type=int, default=60,
                   help="Steps between random subgoal refreshes for the gait walker.")
    p.add_argument("--subgoal-radius", type=float, default=4.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    task = resolve_task(args.task, None)
    env = make_env(task.env_id, seed=args.seed, max_episode_steps=task.max_episode_steps)
    low = LowLevelFlow(args.jepa_model, args.walker, env.action_space.low, env.action_space.high,
                       device=args.device)
    rng = np.random.default_rng(args.seed)
    a_dim = env.action_space.shape[0]

    episodes, n_flip_steps, n_recoveries = [], 0, 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        states, actions = [flatten_obs(obs)], []
        term = trunc = False
        sg = np.asarray(obs["achieved_goal"], np.float32)
        ou = np.zeros(a_dim, np.float32)
        t = 0
        flipped = False
        while not (term or trunc):
            u = float(uprightness(np.asarray(states[-1], np.float32)[None])[0])
            was = flipped
            flipped = u < (args.exit_u if flipped else args.enter)
            if flipped and not was:
                ou = np.zeros(a_dim, np.float32)
            if not flipped and was:
                n_recoveries += 1
            if flipped:
                n_flip_steps += 1
                ou = ou - args.ou_theta * ou + args.ou_sigma * rng.standard_normal(a_dim).astype(np.float32)
                a = np.clip(ou, env.action_space.low, env.action_space.high).astype(np.float32)
            else:
                if t % args.subgoal_h == 0:
                    ag = np.asarray(obs["achieved_goal"], np.float32)
                    ang = rng.uniform(0, 2 * np.pi)
                    sg = ag + args.subgoal_radius * np.array([np.cos(ang), np.sin(ang)], np.float32)
                a = low.act(obs, sg)
            obs, _, term, trunc, _ = env.step(a)
            states.append(flatten_obs(obs)); actions.append(a); t += 1
        episodes.append(Episode(states=np.asarray(states, np.float32),
                                actions=np.asarray(actions, np.float32)))
        if (ep + 1) % 20 == 0:
            print(json.dumps({"event": "collect", "episodes": ep + 1,
                              "flip_steps": n_flip_steps, "ou_recoveries": n_recoveries}), flush=True)
    env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_episodes_npz(args.out, episodes)
    print(json.dumps({"event": "recovery_trials_saved", "path": str(args.out),
                      "episodes": len(episodes), "flip_steps": n_flip_steps,
                      "ou_recoveries": n_recoveries}), flush=True)


if __name__ == "__main__":
    main()
