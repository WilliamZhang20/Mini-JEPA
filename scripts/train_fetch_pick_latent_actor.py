"""Train a Fetch push/pick actor through the JEPA dynamics, not action labels.

Demos define latent futures through a ``*_latent_subgoals.pt`` artifact. This
script trains an actor by rolling its actions through the frozen JEPA predictor
and minimizing distance to those latent futures. It never minimizes an
action-label / behaviour-cloning loss.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa_robotics.data import collect_episodes
from jepa_robotics.envs import make_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import LatentSubgoalActor
from jepa_robotics.subgoals import load_subgoal_artifact, make_latent_subgoal_target_state
from jepa_robotics.tasks import resolve_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="fetch_pick_place")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--subgoal-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--collect-steps", type=int, default=80_000)
    parser.add_argument("--scripted-fraction", type=float, default=0.9)
    parser.add_argument("--controller-gain", type=float, default=12.0)
    parser.add_argument("--action-noise", type=float, default=0.15)
    parser.add_argument("--train-steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--rollout-horizon", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--action-l2", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    task = resolve_task(args.task, None)
    if task.name not in {"fetch_pick_place", "fetch_push"}:
        raise ValueError("This actor trainer is scoped to FetchPickAndPlace and FetchPush.")
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, normalizer, spec, config = load_jepa_artifact(args.model_path, device)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    subgoals = load_subgoal_artifact(args.subgoal_path)

    env = make_env(task.env_id, seed=args.seed, max_episode_steps=task.max_episode_steps)
    episodes, env_spec = collect_episodes(
        env,
        num_steps=args.collect_steps,
        seed=args.seed,
        scripted_fraction=args.scripted_fraction,
        controller_gain=args.controller_gain,
        action_noise=args.action_noise,
        controller=task.controller,
        log_every=max(1, args.collect_steps // 5),
    )
    env.close()
    if env_spec != spec:
        raise ValueError(f"Model spec {spec} does not match collected env spec {env_spec}.")

    raw_states = np.concatenate([ep.states[:-1] for ep in episodes], axis=0).astype(np.float32)
    target_states = np.stack(
        [make_latent_subgoal_target_state(s, spec, subgoals)[1] for s in raw_states]
    ).astype(np.float32)
    states_t = torch.from_numpy(normalizer.encode(raw_states)).to(device)
    targets_t = torch.from_numpy(normalizer.encode(target_states)).to(device)
    latent_std = torch.as_tensor(subgoals["latent_std"], dtype=torch.float32, device=device).view(1, -1)
    with torch.no_grad():
        z_states, z_targets = [], []
        for i in range(0, states_t.shape[0], 8192):
            z_states.append(model.encode(states_t[i : i + 8192]).detach())
            z_targets.append(model.encode(targets_t[i : i + 8192]).detach())
        z_states = torch.cat(z_states, dim=0)
        z_targets = torch.cat(z_targets, dim=0)

    actor = LatentSubgoalActor(
        latent_dim=int(config["latent_dim"]),
        action_dim=spec.action_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=args.lr, weight_decay=1e-4)

    n = z_states.shape[0]
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        z = z_states[idx]
        z_goal = z_targets[idx]
        preds = []
        actions = []
        for _ in range(args.rollout_horizon):
            action = actor(z, z_goal)
            actions.append(action)
            z = model.predict_rollout(z, action.unsqueeze(1), 1)[:, -1]
            preds.append(z)
        traj = torch.stack(preds, dim=1)
        scale = latent_std.clamp_min(1e-4).unsqueeze(1)
        latent_dist = torch.linalg.norm((traj - z_goal.unsqueeze(1)) / scale, dim=-1)
        action_tensor = torch.stack(actions, dim=1)
        loss = latent_dist[:, -1].mean() + 0.15 * latent_dist.mean() + args.action_l2 * action_tensor.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 1000 == 0:
            print(json.dumps({"event": "latent_actor_train", "step": step, "loss": float(loss.detach().cpu())}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "config": {
                "kind": "latent_subgoal_actor",
                "latent_dim": int(config["latent_dim"]),
                "action_dim": spec.action_dim,
                "hidden_dim": args.hidden_dim,
                "model_path": str(args.model_path),
                "subgoal_path": str(args.subgoal_path),
                "task": task.name,
                "rollout_horizon": args.rollout_horizon,
            },
        },
        args.out,
    )
    print(json.dumps({"event": "latent_actor_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
