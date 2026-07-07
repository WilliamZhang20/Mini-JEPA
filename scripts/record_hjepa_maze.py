"""Record a video of the H-JEPA maze agent (subgoal graph + low-level) reaching
the goal, for the AntMaze writeup. Saves the first successful episode found."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MINARI_DATASETS_PATH", "/u5/w223zhan/jepa-mini/.cache/minari")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa_robotics.data import collect_episodes, load_episodes_npz
from jepa_robotics.envs import make_env, obs_spec_from_env
from jepa_robotics.tasks import resolve_task
from scripts.eval_hjepa_maze import LowLevelBC, LowLevelInverse, build_subgoal_graph, dijkstra_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--low-type", default="bc", choices=["bc", "inverse"])
    p.add_argument("--bc-policy", type=Path, default=None)
    p.add_argument("--inverse-policy", type=Path, default=None)
    p.add_argument("--jepa-model", type=Path, required=True)
    p.add_argument("--graph-npz", type=Path, default=None)
    p.add_argument("--graph-steps", type=int, default=120000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--tries", type=int, default=20)
    p.add_argument("--landmarks", type=int, default=150)
    p.add_argument("--k-reach", type=int, default=40)
    p.add_argument("--reach-radius", type=float, default=2.5)
    p.add_argument("--subgoal-timeout", type=int, default=60)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    task = resolve_task(args.task, None)
    genv = make_env(task.env_id, seed=args.seed + 7, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(genv); genv.close()
    if args.graph_npz is not None:
        episodes = load_episodes_npz(args.graph_npz)
    else:
        genv = make_env(task.env_id, seed=args.seed + 7, max_episode_steps=task.max_episode_steps)
        episodes, _ = collect_episodes(
            genv,
            num_steps=args.graph_steps,
            seed=args.seed + 7,
            scripted_fraction=0.5,
            controller_gain=5.0,
            action_noise=0.3,
            controller=task.controller,
            log_every=0,
        )
        genv.close()
    landmarks, adj = build_subgoal_graph(episodes, spec, args.landmarks, args.k_reach, seed=args.seed)

    env = make_env(task.env_id, seed=args.seed, max_episode_steps=task.max_episode_steps)
    if args.low_type == "inverse":
        if args.inverse_policy is None:
            raise ValueError("--low-type inverse requires --inverse-policy")
        low = LowLevelInverse(args.jepa_model, args.inverse_policy, env.action_space.low, env.action_space.high, device=args.device)
    else:
        if args.bc_policy is None:
            raise ValueError("--low-type bc requires --bc-policy")
        low = LowLevelBC(args.jepa_model, args.bc_policy, env.action_space.low, env.action_space.high, device=args.device)
    env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.tries):
        env = make_env(task.env_id, seed=args.seed + ep, max_episode_steps=task.max_episode_steps,
                       render_mode="rgb_array", width=args.width, height=args.height)
        obs, _ = env.reset(seed=args.seed + ep)
        goal = np.asarray(obs["desired_goal"], np.float32)
        cur = np.asarray(obs["achieved_goal"], np.float32)
        s = int(np.argmin(np.linalg.norm(landmarks - cur, axis=1)))
        d = int(np.argmin(np.linalg.norm(landmarks - goal, axis=1)))
        path = dijkstra_path(adj, s, d)
        wps = ([landmarks[i] for i in path] if path else []) + [goal]
        wi = 0; since = 0; term = trunc = False; info = {}; frames = []
        f = env.render()
        if f is not None:
            frames.append(f)
        while not (term or trunc):
            sg = wps[min(wi, len(wps) - 1)]
            a = low.act(obs, sg)
            obs, _, term, trunc, info = env.step(a); since += 1
            f = env.render()
            if f is not None:
                frames.append(f)
            ag = np.asarray(obs["achieved_goal"], np.float32)
            if wi < len(wps) - 1 and (np.linalg.norm(ag - sg) < args.reach_radius or since > args.subgoal_timeout):
                wi += 1; since = 0
        env.close()
        success = float(info.get("is_success", info.get("success", 0.0)))
        print(f'{{"event":"episode","ep":{ep},"success":{success},"frames":{len(frames)}}}', flush=True)
        if success > 0.5:
            imageio.mimsave(args.out, frames[::2], fps=args.fps, format="FFMPEG")  # subsample frames (long episodes)
            print(f'{{"event":"recorded","path":"{args.out}","ep":{ep}}}', flush=True)
            return
    print('{"event":"no_success_to_record"}', flush=True)


if __name__ == "__main__":
    main()
