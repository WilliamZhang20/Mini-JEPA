"""Hierarchical SSL controller for in-hand reorientation (no reward, no demos).

The abstract level plans in SO(3): it splits the (large) rotation from the current
object orientation to the goal into a chain of small subgoals along the geodesic,
each within the low-level's proven ~20-30 deg controllable band. The low-level is
the geodesic-cost CEM over the DexterousJEPA world model (the same primitive the
controllability probe validated). Closed-loop: every env step the abstract level
re-reads the achieved orientation and places the next subgoal a fixed angle ahead
toward the goal (a moving carrot), so drift and regrasps are absorbed.

This is the maze H-JEPA recipe (plan abstract, compose a locally-reachable
primitive) ported to orientation space. Flat MPC to the full goal saturates at the
single-shot ceiling; the chain is what recovers the rest.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.envs import flatten_obs, make_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task
from scripts.eval_subgoal_controllability import Controller, quat_geodesic, quat_mul


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def step_toward(q_cur, q_goal, step_rad):
    """Subgoal = rotate q_cur toward q_goal by min(step_rad, full angle) along
    their relative axis. Works for any axis (generalizes past Z)."""
    qc = q_cur / (np.linalg.norm(q_cur) + 1e-9)
    qg = q_goal / (np.linalg.norm(q_goal) + 1e-9)
    q_rel = quat_mul(qg, quat_conj(qc))
    q_rel = q_rel / (np.linalg.norm(q_rel) + 1e-9)
    if q_rel[0] < 0:  # shortest arc (double cover)
        q_rel = -q_rel
    angle = 2.0 * np.arccos(min(1.0, abs(float(q_rel[0]))))
    if angle < 1e-4:
        return qg
    axis = q_rel[1:] / (np.linalg.norm(q_rel[1:]) + 1e-9)
    s = min(step_rad, angle)
    q_step = np.array([np.cos(s / 2.0), *(axis * np.sin(s / 2.0))], dtype=np.float32)
    return quat_mul(q_step, qc)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="handmanipulate_block_rotate_z")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=60000)
    p.add_argument("--max-episode-steps", type=int, default=150)
    p.add_argument("--step-deg", type=float, default=20.0, help="subgoal spacing along the geodesic")
    p.add_argument("--inner-thr-deg", type=float, default=12.0, help="advance the subgoal once within this")
    p.add_argument("--max-hold", type=int, default=24, help="max steps to servo one subgoal before advancing")
    p.add_argument("--planner", default="mppi", choices=["mppi", "cem"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--init-std", type=float, default=0.5)
    p.add_argument("--exec-k", type=int, default=0, help="actions executed per plan (0 = full horizon)")
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--candidates", type=int, default=512)
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--torch-seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    dev = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(args.torch_seed)
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    env = make_env(task.env_id, seed=args.seed, max_episode_steps=args.max_episode_steps)
    lo = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=dev)
    hi = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=dev)
    exec_k = args.exec_k if args.exec_k > 0 else args.horizon  # 0 -> execute the full plan (validated regime)
    ctrl = Controller(wm, norm, spec, dev, args.horizon, args.candidates, args.iters, 0.1,
                      args.init_std, planner=args.planner, exec_k=exec_k, temperature=args.temperature)
    ag, dgo = spec.obs_dim, spec.obs_dim + spec.goal_dim
    step_rad = np.radians(args.step_deg)

    successes, final_gaps = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        s = flatten_obs(obs)
        q_goal = s[dgo + 3:dgo + 7].copy()
        term = trunc = False
        info = {}
        inner_thr = np.radians(args.inner_thr_deg)
        # Discrete subgoal: hold one fixed while the low-level servos it (the
        # validated primitive regime), advance it a step along the geodesic only
        # once reached. The final subgoal shrinks to the goal itself.
        subgoal_q = step_toward(s[ag + 3:ag + 7], q_goal, step_rad)
        ctrl.reset()
        held = 0
        while not (term or trunc):
            q_cur = s[ag + 3:ag + 7]
            if (quat_geodesic(q_cur, subgoal_q) < inner_thr or held >= args.max_hold) \
                    and quat_geodesic(q_cur, q_goal) > 1e-3:
                subgoal_q = step_toward(q_cur, q_goal, step_rad)     # advance
                held = 0
            subgoal_pose = np.concatenate([s[ag:ag + 3], subgoal_q]).astype(np.float32)
            act = ctrl.act(obs, subgoal_pose, lo, hi)
            obs, _, term, trunc, info = env.step(act)
            s = flatten_obs(obs)
            held += 1
        successes.append(float(info.get("is_success", 0.0)))
        final_gaps.append(np.degrees(quat_geodesic(s[ag + 3:ag + 7], q_goal)))
    env.close()
    print(json.dumps({"event": "hierarchical_reorient_eval", "task": task.name,
                      "model_path": str(args.model_path), "episodes": args.episodes,
                      "success_rate": round(float(np.mean(successes)), 3),
                      "median_final_gap_deg": round(float(np.median(final_gaps)), 1),
                      "step_deg": args.step_deg, "max_episode_steps": args.max_episode_steps}), flush=True)


if __name__ == "__main__":
    main()
