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

from .data import Episode, collect_episodes
from .envs import flatten_obs, make_env, obs_spec_from_env
from .evaluate import SB3Policy, load_jepa_artifact
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
    parser.add_argument("--teacher-sb3-path", type=Path, default=None)
    parser.add_argument(
        "--teacher-time-feature",
        action="store_true",
        help="Use SB3's TimeFeatureWrapper for teacher actions, then strip it before JEPA encoding.",
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

    if args.teacher_sb3_path is None:
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
    else:
        from sb3_contrib.common.wrappers import TimeFeatureWrapper

        env = make_env(task.env_id, seed=args.seed, max_episode_steps=task.max_episode_steps)
        action_low = env.action_space.low
        action_high = env.action_space.high
        spec = obs_spec_from_env(env)
        teacher_env = TimeFeatureWrapper(env) if args.teacher_time_feature else env
        teacher = SB3Policy(args.teacher_sb3_path, name="teacher", env=teacher_env)
        episodes = []
        total_steps = 0
        episode_idx = 0
        rng = np.random.default_rng(args.seed)

        def strip_time_feature(obs):
            if not args.teacher_time_feature:
                return obs
            stripped = {k: np.asarray(v).copy() for k, v in obs.items()}
            obs_vec = stripped["observation"].reshape(-1)
            if obs_vec.shape[0] > spec.obs_dim:
                stripped["observation"] = obs_vec[: spec.obs_dim].astype(np.float32)
            return stripped

        while total_steps < args.collect_steps:
            obs, _ = teacher_env.reset(seed=args.seed + episode_idx)
            states = [flatten_obs(strip_time_feature(obs))]
            actions = []
            terminated = truncated = False
            while not (terminated or truncated) and total_steps < args.collect_steps:
                action = teacher.act(obs, teacher_env)
                if args.action_noise > 0:
                    action = action + rng.normal(0.0, args.action_noise, size=spec.action_dim).astype(np.float32)
                action = np.clip(action, action_low, action_high).astype(np.float32)
                obs, _, terminated, truncated, _info = teacher_env.step(action)
                states.append(flatten_obs(strip_time_feature(obs)))
                actions.append(action)
                total_steps += 1
                if args.collect_steps and args.collect_steps // 5 and total_steps % (args.collect_steps // 5) == 0:
                    print(json.dumps({"event": "teacher_collect", "steps": total_steps, "target_steps": args.collect_steps}), flush=True)
            episodes.append(Episode(states=np.asarray(states, dtype=np.float32), actions=np.asarray(actions, dtype=np.float32)))
            episode_idx += 1
        teacher_env.close()

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
