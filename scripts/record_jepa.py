"""Record an MP4 of the JEPA agent (learned policy + world-model MPC) solving a task.

Loads a trained JEPA world model and its goal-conditioned policy, runs the
policy-seeded MPC controller for several episodes, and writes a video. Supports
the same ``--vary-goal`` showcase mode as ``record_expert.py`` so the targets
differ each episode (alternating table / mid-air for pick-and-place).

    PYTHONNOUSERSITE=1 MUJOCO_GL=egl python scripts/record_jepa.py \
        --task fetch_pick_place --vary-goal --episodes 6 \
        --model-path runs/fetch_pick_place/checkpoints/pickplace_v2_model.pt \
        --policy-path runs/fetch_pick_place/checkpoints/pickplace_v2_policy.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa_robotics.envs import make_env
from jepa_robotics.evaluate import JEPAMPCPolicy, load_jepa_artifact, load_policy_artifact
from jepa_robotics.tasks import resolve_task

TABLE_Z = 0.425


def varied_goal(rng, obj_xy, force_air, table_only):
    center = np.array([1.34, 0.75], dtype=np.float32)
    xy = center + rng.uniform(-0.13, 0.13, size=2).astype(np.float32)
    if np.linalg.norm(xy - obj_xy) < 0.12:
        xy = obj_xy + (xy - obj_xy) / (np.linalg.norm(xy - obj_xy) + 1e-6) * 0.18
    z = TABLE_Z if (table_only or not force_air) else TABLE_Z + float(rng.uniform(0.13, 0.30))
    return np.array([xy[0], xy[1], z], dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="fetch_pick_place")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--policy-path", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--vary-goal", action="store_true")
    p.add_argument("--policy-only", action="store_true",
                   help="Execute the learned policy directly with no MPC refinement.")
    p.add_argument("--mpc-candidates", type=int, default=128)
    p.add_argument("--mpc-horizon", type=int, default=12)
    p.add_argument("--cem-iters", type=int, default=4)
    p.add_argument("--action-std", type=float, default=0.5)
    p.add_argument("--policy-proposal-fraction", type=float, default=0.5)
    p.add_argument("--manip-reach-weight", type=float, default=0.1)
    p.add_argument("--manip-path-weight", type=float, default=0.3)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    task = resolve_task(args.task, None)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    from jepa_robotics.evaluate import LearnedPolicyOnly

    model, normalizer, spec, _ = load_jepa_artifact(args.model_path, device)
    policy_net, _ = load_policy_artifact(args.policy_path, device)

    suffix = ("_policyonly" if args.policy_only else "") + ("_varied" if args.vary_goal else "")
    out = args.out or Path(f"runs/{task.slug}/videos/jepa_agent_{task.slug}{suffix}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    table_only = task.controller != "pick_place"

    env = make_env(task.env_id, seed=args.seed, max_episode_steps=task.max_episode_steps, render_mode="rgb_array")
    unwrapped = env.unwrapped
    if args.policy_only:
        controller = LearnedPolicyOnly(
            model=model, policy_net=policy_net, normalizer=normalizer, spec=spec, device=device
        )
    else:
        controller = JEPAMPCPolicy(
            model=model, normalizer=normalizer, spec=spec, device=device,
            candidates=args.mpc_candidates, horizon=args.mpc_horizon, seed=args.seed + 10_000,
            method="cem", score_mode="manip", cem_iters=args.cem_iters, action_std=args.action_std,
            manip_reach_weight=args.manip_reach_weight, manip_path_weight=args.manip_path_weight,
            scripted_controller=task.controller,
            policy_net=policy_net, policy_proposal_fraction=args.policy_proposal_fraction,
        )

    goal_rng = np.random.default_rng(args.seed)
    frames, successes, goals = [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        controller.prev_action = np.zeros(spec.action_dim, dtype=np.float32)
        controller.prev_plan = None
        if args.vary_goal:
            g = varied_goal(goal_rng, np.asarray(obs["achieved_goal"][:2], dtype=np.float32),
                            force_air=(ep % 2 == 1), table_only=table_only)
            unwrapped.goal = g
            obs["desired_goal"] = g.astype(np.float32)
            goals.append(g)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            action = controller.act(obs, env)
            obs, _, terminated, truncated, info = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        successes.append(float(info.get("is_success", 0.0)))

    env.close()
    imageio.mimsave(out, frames, fps=args.fps, format="FFMPEG")
    msg = f"wrote {out}  episodes={args.episodes}  success={np.mean(successes):.2f}  frames={len(frames)}"
    if goals:
        msg += "  goal_heights=[" + ", ".join("air" if g[2] > TABLE_Z + 0.02 else "table" for g in goals) + "]"
    print(msg)


if __name__ == "__main__":
    main()
