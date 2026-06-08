"""Record an MP4 of the scripted reference controller solving a Fetch task.

    PYTHONNOUSERSITE=1 MUJOCO_GL=egl python scripts/record_expert.py \
        --task fetch_pick_place --episodes 3 --gain 12 \
        --out runs/fetch_pick_place/videos/reference_fetch_pick_place.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa_robotics.data import scripted_action
from jepa_robotics.envs import make_env
from jepa_robotics.tasks import resolve_task


# Table-surface height for the Fetch object; mid-air goals sit above it.
TABLE_Z = 0.425


def varied_goal(rng: np.random.Generator, obj_xy: np.ndarray, force_air: bool, table_only: bool) -> np.ndarray:
    """Sample a goal that is clearly different each episode.

    x/y are spread across the reachable workspace (kept a little away from the
    object so the motion is visible); z is on the table or, for pick-and-place,
    lifted well into the air so the mid-air targets are obvious on camera.
    """
    center = np.array([1.34, 0.75], dtype=np.float32)
    xy = center + rng.uniform(-0.13, 0.13, size=2).astype(np.float32)
    # nudge the goal away from the object so there is a real distance to cover
    if np.linalg.norm(xy - obj_xy) < 0.12:
        xy = obj_xy + (xy - obj_xy) / (np.linalg.norm(xy - obj_xy) + 1e-6) * 0.18
    if table_only or not force_air:
        z = TABLE_Z
    else:
        z = TABLE_Z + float(rng.uniform(0.13, 0.32))
    return np.array([xy[0], xy[1], z], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="fetch_pick_place")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--gain", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--vary-goal",
        action="store_true",
        help="Override each episode's goal so targets clearly differ, alternating "
        "table and mid-air placements (pick-and-place only for mid-air).",
    )
    args = parser.parse_args()

    task = resolve_task(args.task, None)
    suffix = "_varied" if args.vary_goal else ""
    out = args.out or Path(f"runs/{task.slug}/videos/reference_{task.slug}{suffix}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    table_only = task.controller != "pick_place"  # push/reach goals stay on the surface

    env = make_env(
        task.env_id, seed=args.seed, max_episode_steps=task.max_episode_steps, render_mode="rgb_array"
    )
    unwrapped = env.unwrapped
    rng = np.random.default_rng(0)
    goal_rng = np.random.default_rng(args.seed)
    frames = []
    successes = []
    goals = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        if args.vary_goal:
            # alternate mid-air / table so the recording shows both clearly
            force_air = (ep % 2 == 1)
            g = varied_goal(goal_rng, np.asarray(obs["achieved_goal"][:2], dtype=np.float32),
                            force_air, table_only)
            unwrapped.goal = g
            obs["desired_goal"] = g.astype(np.float32)
            goals.append(g)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            action = scripted_action(obs, env.action_space.shape[0], args.gain, task.controller, env, rng)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, _, terminated, truncated, info = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        successes.append(float(info.get("is_success", 0.0)))
    env.close()

    imageio.mimsave(out, frames, fps=args.fps, format="FFMPEG")
    msg = f"wrote {out}  episodes={args.episodes}  success={np.mean(successes):.2f}  frames={len(frames)}"
    if goals:
        heights = ", ".join("air" if g[2] > TABLE_Z + 0.02 else "table" for g in goals)
        msg += f"  goal_heights=[{heights}]"
    print(msg)


if __name__ == "__main__":
    main()
