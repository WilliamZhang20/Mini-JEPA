"""Train a goal-conditioned action prior on top of a trained JEPA world model.

This is the "amortized controller" half of a world-model agent. The JEPA
encoder (frozen) gives a latent representation of the observation (which
already contains the desired goal); we behaviour-clone the scripted experts'
actions as a function of that latent. The resulting policy is used at planning
time as the MPC proposal, which the world model then refines/verifies -- the
combination is what makes precise contact skills (grasping) reliable, where
sampling-only MPC fails.

    PYTHONNOUSERSITE=1 MUJOCO_GL=egl python -m jepa_robotics.train_policy \
        --task fetch_pick_place \
        --model-path runs/fetch_pick_place/checkpoints/pickplace_v2_model.pt \
        --out runs/fetch_pick_place/checkpoints/pickplace_v2_policy.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import collect_episodes
from .envs import make_env, obs_spec_from_env
from .evaluate import load_jepa_artifact
from .models import GoalConditionedPolicy
from .tasks import resolve_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None)
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--collect-steps", type=int, default=300_000)
    parser.add_argument("--scripted-fraction", type=float, default=0.97)
    parser.add_argument("--controller-gain", type=float, default=12.0)
    parser.add_argument("--action-noise", type=float, default=0.1)
    parser.add_argument(
        "--episodes-npz",
        type=Path,
        default=None,
        help="Train from pre-collected demonstration trajectories instead of a scripted expert.",
    )
    parser.add_argument(
        "--future-goal-relabel-frac",
        type=float,
        default=0.0,
        help="Fraction of states whose desired_goal is relabeled to a future achieved_goal "
             "(hindsight). Makes the BC policy a general nearby-goal reacher for H-JEPA low levels.",
    )
    parser.add_argument("--train-steps", type=int, default=40_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    task = resolve_task(args.task, args.env_id)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, normalizer, spec, config = load_jepa_artifact(args.model_path, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(json.dumps({"event": "policy_config", "task": task.name, **vars(args)}, default=str), flush=True)

    if args.episodes_npz is not None:
        # Train directly from pre-collected demonstrations (Adroit) or the
        # canonical multi-task Fetch union. Prefer the
        # spec saved in the npz (canonical 35-D state) over any single env's spec.
        from .data import load_episodes_npz, load_spec_npz

        env = make_env(task.env_id, seed=args.seed, max_episode_steps=task.max_episode_steps)
        spec = load_spec_npz(args.episodes_npz) or obs_spec_from_env(env)
        env.close()
        episodes = load_episodes_npz(args.episodes_npz)
        print(json.dumps({"event": "loaded_episodes_npz", "path": str(args.episodes_npz),
                          "episodes": len(episodes), "state_dim": spec.state_dim}), flush=True)
    else:
        env = make_env(task.env_id, seed=args.seed, max_episode_steps=task.max_episode_steps)
        episodes, _ = collect_episodes(
            env,
            num_steps=args.collect_steps,
            seed=args.seed,
            scripted_fraction=args.scripted_fraction,
            controller_gain=args.controller_gain,
            action_noise=args.action_noise,
            controller=task.controller,
            log_every=args.collect_steps // 5 if args.collect_steps else 0,
        )
        env.close()

    # Self-supervised future-goal relabeling for the low-level: replace each state's
    # desired_goal with a FUTURE achieved_goal so the policy learns to reach
    # arbitrary nearby positions (subgoals), not just the demos' final goals.
    # Essential for using the BC policy as an H-JEPA low-level on big mazes.
    if args.future_goal_relabel_frac > 0 and spec.is_goal_env and spec.goal_dim > 0:
        rng_h = np.random.default_rng(args.seed + 99)
        gs, ge = spec.obs_dim, spec.obs_dim + spec.goal_dim
        ds, de = spec.obs_dim + spec.goal_dim, spec.obs_dim + 2 * spec.goal_dim
        relabeled = 0
        for ep in episodes:
            S = ep.states
            T = len(ep.actions)
            for t in range(T):
                if rng_h.random() < args.future_goal_relabel_frac:
                    tf = int(rng_h.integers(t + 1, len(S)))
                    S[t, ds:de] = S[tf, gs:ge]
                    relabeled += 1
        print(json.dumps({
            "event": "future_goal_relabel",
            "frac": args.future_goal_relabel_frac,
            "relabeled": relabeled,
        }), flush=True)

    # Flatten to (state, action) pairs; encode states with the frozen JEPA encoder.
    states = np.concatenate([ep.states[:-1] for ep in episodes], axis=0)
    actions = np.concatenate([ep.actions for ep in episodes], axis=0)
    states = normalizer.encode(states)
    states_t = torch.from_numpy(states).to(device)
    actions_t = torch.from_numpy(actions).to(device)
    with torch.no_grad():
        latents = []
        for i in range(0, states_t.shape[0], 8192):
            latents.append(model.encode(states_t[i : i + 8192]))
        latents = torch.cat(latents, dim=0)
    print(json.dumps({"event": "policy_dataset", "pairs": int(latents.shape[0])}), flush=True)

    policy = GoalConditionedPolicy(
        latent_dim=int(config["latent_dim"]),
        action_dim=spec.action_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    n = latents.shape[0]
    step = 0
    while step < args.train_steps:
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        pred = policy(latents[idx])
        loss = F.smooth_l1_loss(pred, actions_t[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1
        if step == 1 or step % 2000 == 0:
            print(json.dumps({"event": "policy_train", "step": step, "bc_loss": float(loss.detach().cpu())}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": policy.state_dict(),
            "config": {
                "latent_dim": int(config["latent_dim"]),
                "action_dim": spec.action_dim,
                "hidden_dim": args.hidden_dim,
                "model_path": str(args.model_path),
                "task": task.name,
            },
        },
        args.out,
    )
    print(json.dumps({"event": "policy_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
