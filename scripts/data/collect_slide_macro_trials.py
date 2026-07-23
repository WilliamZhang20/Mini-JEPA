"""Collect self-supervised FetchSlide event-to-event strike trials."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.algos.world_models.ballistic import fetch_slide_ready, slide_macro_action
from jepa_robotics.data import scripted_slide_action
from jepa_robotics.envs import flatten_obs, make_env, obs_spec_from_env
from jepa_robotics.tasks import resolve_task


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="fetch_slide")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--trials", type=int, default=2000)
    p.add_argument("--seed", type=int, default=13000)
    p.add_argument("--max-approach-steps", type=int, default=50)
    p.add_argument("--min-angle-deg", type=float, default=-50.0)
    p.add_argument("--max-angle-deg", type=float, default=50.0)
    p.add_argument("--min-amplitude", type=float, default=0.15)
    p.add_argument("--max-amplitude", type=float, default=1.0)
    p.add_argument("--min-duration", type=int, default=2)
    p.add_argument("--max-duration", type=int, default=10)
    args = p.parse_args()

    task = resolve_task(args.task, None)
    env = make_env(task.env_id, seed=args.seed, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(env)
    rng = np.random.default_rng(args.seed)
    pre_states, final_states, macros, durations = [], [], [], []
    successes, distances, approach_steps = [], [], []
    skipped = 0
    for trial in range(args.trials):
        obs, _ = env.reset(seed=args.seed + trial)
        ready = False
        steps = 0
        while steps < args.max_approach_steps:
            raw = flatten_obs(obs)
            if fetch_slide_ready(raw, spec.obs_dim, spec.goal_dim):
                ready = True
                break
            action = scripted_slide_action(obs, spec.action_dim, 12.0)
            obs, _, term, trunc, _ = env.step(action)
            steps += 1
            if term or trunc:
                break
        if not ready:
            skipped += 1
            continue

        pre = flatten_obs(obs).copy()
        approach_count = steps
        obj = pre[spec.obs_dim : spec.obs_dim + spec.goal_dim]
        goal = pre[spec.obs_dim + spec.goal_dim : spec.obs_dim + 2 * spec.goal_dim]
        base = float(np.arctan2(goal[1] - obj[1], goal[0] - obj[0]))
        angle = base + np.deg2rad(rng.uniform(args.min_angle_deg, args.max_angle_deg))
        amplitude = float(rng.uniform(args.min_amplitude, args.max_amplitude))
        duration = int(rng.integers(args.min_duration, args.max_duration + 1))
        macro = np.array([np.cos(angle), np.sin(angle), amplitude,
                          duration / args.max_duration], np.float32)
        info = {}
        term = trunc = False
        for _ in range(duration):
            action = slide_macro_action(macro, flatten_obs(obs), spec.obs_dim,
                                        spec.goal_dim, spec.action_dim)
            obs, _, term, trunc, info = env.step(action)
            steps += 1
            if term or trunc:
                break
        # Lift away after impact, then enter an absorbing no-control coast.
        for _ in range(4):
            if term or trunc:
                break
            action = np.zeros(spec.action_dim, np.float32)
            action[2] = 1.0
            if spec.action_dim >= 4:
                action[3] = -1.0
            obs, _, term, trunc, info = env.step(action)
            steps += 1
        while not (term or trunc):
            action = np.zeros(spec.action_dim, np.float32)
            if spec.action_dim >= 4:
                action[3] = -1.0
            obs, _, term, trunc, info = env.step(action)
            steps += 1

        final = flatten_obs(obs).copy()
        achieved = final[spec.obs_dim : spec.obs_dim + spec.goal_dim]
        desired = final[spec.obs_dim + spec.goal_dim : spec.obs_dim + 2 * spec.goal_dim]
        pre_states.append(pre)
        final_states.append(final)
        macros.append(macro)
        durations.append(duration)
        distances.append(float(np.linalg.norm(achieved - desired)))
        successes.append(float(info.get("is_success", 0.0)))
        approach_steps.append(approach_count)
        if (trial + 1) % 100 == 0:
            print(json.dumps({"event": "slide_macro_collect", "trial": trial + 1,
                              "kept": len(pre_states), "skipped": skipped,
                              "success_rate": float(np.mean(successes))}), flush=True)

    env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        pre_states=np.asarray(pre_states, np.float32),
        final_states=np.asarray(final_states, np.float32),
        macros=np.asarray(macros, np.float32),
        durations=np.asarray(durations, np.int64),
        successes=np.asarray(successes, np.float32),
        final_distances=np.asarray(distances, np.float32),
        approach_steps=np.asarray(approach_steps, np.int64),
        obs_dim=np.asarray(spec.obs_dim), goal_dim=np.asarray(spec.goal_dim),
        action_dim=np.asarray(spec.action_dim), max_duration=np.asarray(args.max_duration),
    )
    print(json.dumps({"event": "slide_macro_saved", "path": str(args.out),
                      "trials": len(pre_states), "skipped": skipped,
                      "success_rate": float(np.mean(successes)),
                      "mean_final_distance": float(np.mean(distances))}), flush=True)


if __name__ == "__main__":
    main()
