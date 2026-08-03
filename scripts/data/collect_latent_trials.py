"""Collect on-policy transition trials for the JEPA world model.

A world model trained only on demonstrations is accurate only on the
demonstration manifold — but a planner's whole job is to perturb *off* that
manifold looking for something better, so the region it queries is exactly the
region the model never saw. That mismatch is the documented reason latent MPC
has failed on the contact-rich tasks in this repo.

This collects the missing evidence: rollouts of the planner itself under
exploration noise (``--policy planner``) plus temporally-correlated random
motion for raw coverage (``--policy colored``). No reward or success signal is
used for filtering by default — every transition is equally valid evidence
about dynamics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.algos.latent_subgoal import LatentSubgoalNet
from jepa_robotics.algos.planning.latent_mpc import LatentCEMConfig, LatentCEMPlanner, colored_noise
from jepa_robotics.data import Episode, save_episodes_npz
from jepa_robotics.envs import flatten_obs, make_env, obs_spec_from_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--policy", choices=["colored", "planner"], default="planner")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=900000)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--subgoal-path", type=Path, default=None)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--method", choices=["cem", "grad"], default="grad")
    p.add_argument("--proposal", choices=["none", "inverse"], default="inverse")
    p.add_argument("--grad-restarts", type=int, default=16)
    p.add_argument("--grad-iters", type=int, default=40)
    p.add_argument("--trust-region", type=float, default=0.0)
    p.add_argument("--candidates", type=int, default=128)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--exec-k", type=int, default=4)
    p.add_argument("--explore-std", type=float, default=0.2,
                   help="Gaussian noise added to executed actions, so trials cover the "
                        "neighbourhood of the plan rather than only the plan itself.")
    p.add_argument("--colored-beta", type=float, default=2.0)
    p.add_argument("--colored-scale", type=float, default=0.8)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    task = resolve_task(args.task, None)
    max_steps = args.max_episode_steps or task.max_episode_steps
    env = make_env(task.env_id, seed=args.seed, max_episode_steps=max_steps)
    spec = obs_spec_from_env(env)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    wm = normalizer = subgoal_net = planner = None
    if args.policy == "planner":
        if args.model_path is None or args.subgoal_path is None:
            raise SystemExit("--policy planner needs --model-path and --subgoal-path")
        wm, normalizer, _spec, _cfg = load_jepa_artifact(args.model_path, device)
        ckpt = torch.load(args.subgoal_path, map_location=device, weights_only=False)
        subgoal_net = LatentSubgoalNet(
            latent_dim=int(ckpt["latent_dim"]), state_dim=int(ckpt["state_dim"]),
            hidden=int(ckpt["hidden"]), n_blocks=int(ckpt["n_blocks"]),
            max_horizon=int(ckpt["max_horizon"]),
        ).to(device)
        subgoal_net.load_state_dict(ckpt["state_dict"])
        subgoal_net.eval()
        planner = LatentCEMPlanner(
            wm,
            LatentCEMConfig(
                method=args.method, horizon=args.horizon, candidates=args.candidates,
                iterations=args.iterations, seed=args.seed,
                grad_restarts=args.grad_restarts, grad_iters=args.grad_iters,
                trust_region=args.trust_region,
                action_low=float(np.min(env.action_space.low)),
                action_high=float(np.max(env.action_space.high)),
            ),
            device,
        )

    episodes: list[Episode] = []
    successes = 0
    for ep_idx in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep_idx)
        if planner is not None:
            planner.reset()
        states = [flatten_obs(obs)]
        actions: list[np.ndarray] = []
        cached: list[np.ndarray] = []
        noise_seq, noise_at = None, 0
        terminated = truncated = False
        last_info: dict = {}
        while not (terminated or truncated):
            if args.policy == "colored":
                if noise_seq is None or noise_at >= len(noise_seq):
                    noise_seq = (
                        colored_noise((1, 16, spec.action_dim), args.colored_beta, device, generator)[0]
                        .cpu().numpy() * args.colored_scale
                    )
                    noise_at = 0
                action = noise_seq[noise_at]
                noise_at += 1
            else:
                if not cached:
                    with torch.no_grad():
                        raw = flatten_obs(obs)
                        s = torch.from_numpy(normalizer.encode(raw)).unsqueeze(0).to(device)
                        z = wm.encode(s)
                        z_goal, state_goal, _ = subgoal_net.subgoal(z, args.horizon)
                        proposal = (
                            planner.inverse_proposal(z, z_goal)
                            if args.proposal == "inverse" else None
                        )
                        plan = planner.plan(
                            z, z_goal, goal_state=state_goal, proposal=proposal
                        )
                    plan_np = plan.cpu().numpy()
                    cached = [plan_np[i].copy() for i in range(min(args.exec_k, len(plan_np)))]
                action = cached.pop(0)
                if args.explore_std > 0:
                    action = action + rng.normal(0.0, args.explore_std, size=action.shape)
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            obs, _reward, terminated, truncated, last_info = env.step(action)
            states.append(flatten_obs(obs))
            actions.append(action)
        successes += int(bool(last_info.get("is_success", last_info.get("success", 0.0))))
        episodes.append(
            Episode(states=np.asarray(states, np.float32), actions=np.asarray(actions, np.float32))
        )
        if (ep_idx + 1) % 25 == 0:
            print(json.dumps({"event": "collect", "episodes": ep_idx + 1,
                              "success_frac": successes / (ep_idx + 1)}), flush=True)
    env.close()
    save_episodes_npz(args.out, episodes, spec)
    print(json.dumps({"event": "collected", "out": str(args.out), "policy": args.policy,
                      "episodes": len(episodes), "success_frac": successes / max(1, len(episodes)),
                      "steps": int(sum(len(e.actions) for e in episodes))}), flush=True)


if __name__ == "__main__":
    main()
