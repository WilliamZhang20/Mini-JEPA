"""Train a deterministic, goal-directed AntMaze action-chunk specialist."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jepa_robotics.algos.task_families.maze import build_directed_chunks
from jepa_robotics.data import load_episodes_npz
from jepa_robotics.envs import make_env, obs_spec_from_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import MLP
from jepa_robotics.tasks import resolve_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--episodes-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--route-start", type=float, nargs=2, required=True)
    parser.add_argument("--route-goal", type=float, nargs=2, required=True)
    parser.add_argument("--route-radius", type=float, default=1.5)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--max-relabel-h", type=int, default=60)
    parser.add_argument("--min-progress", type=float, default=0.15)
    parser.add_argument(
        "--condition-final-goal",
        action="store_true",
        help="Train successful prefixes against the episode's final desired goal.",
    )
    parser.add_argument("--input-noise", type=float, default=0.03)
    parser.add_argument("--emphasis-repeat", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=261100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    _, normalizer, _, _ = load_jepa_artifact(args.model_path, device)
    env = make_env(task.env_id, seed=0, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(env)
    env.close()
    gs, ge = spec.obs_dim, spec.obs_dim + spec.goal_dim
    ds, de = ge, ge + spec.goal_dim
    start = np.asarray(args.route_start, dtype=np.float32)
    goal = np.asarray(args.route_goal, dtype=np.float32)
    episodes = [
        episode
        for episode in load_episodes_npz(args.episodes_npz)
        if np.linalg.norm(episode.states[0, gs:ge] - start) <= args.route_radius
        and np.linalg.norm(episode.states[0, ds:de] - goal) <= args.route_radius
    ]
    if args.condition_final_goal:
        raw_states: list[np.ndarray] = []
        raw_chunks: list[np.ndarray] = []
        for episode in episodes:
            distance = np.linalg.norm(
                episode.states[:, gs:ge] - episode.states[0, ds:de][None],
                axis=1,
            )
            hits = np.flatnonzero(distance <= 0.5)
            if not len(hits):
                continue
            stop = min(len(episode.actions), int(hits[0]) + 1)
            for t in range(stop):
                chunk = episode.actions[t : t + args.chunk]
                if len(chunk) < args.chunk:
                    chunk = np.concatenate(
                        [chunk, np.repeat(chunk[-1:], args.chunk - len(chunk), axis=0)]
                    )
                raw_states.append(episode.states[t])
                raw_chunks.append(chunk)
        states = normalizer.encode(np.asarray(raw_states, dtype=np.float32))
        chunks = np.asarray(raw_chunks, dtype=np.float32)
    else:
        states, chunks, _ = build_directed_chunks(
            episodes,
            spec,
            normalizer,
            args.chunk,
            np.random.default_rng(args.seed),
            max_relabel_h=args.max_relabel_h,
            min_progress=args.min_progress,
        )
    target_np = chunks.reshape(len(chunks), -1).astype(np.float32)
    state_tensor = torch.from_numpy(states).to(device)
    target = torch.from_numpy(target_np).to(device)
    condition_dim = states.shape[1] + args.emphasis_repeat * spec.goal_dim
    net = MLP(
        [
            condition_dim,
            args.hidden,
            args.hidden,
            args.hidden,
            target.shape[1],
        ],
        layer_norm=True,
    ).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    for step in range(1, args.steps + 1):
        index = torch.randint(0, len(target), (args.batch_size,), device=device)
        state_batch = state_tensor[index].clone()
        if args.input_noise > 0:
            state_batch[:, gs:de] += (
                torch.randn_like(state_batch[:, gs:de]) * args.input_noise
            )
        delta = state_batch[:, ds:de] - state_batch[:, gs:ge]
        condition = torch.cat(
            [state_batch, delta.repeat(1, args.emphasis_repeat)], dim=1
        )
        loss = nn.functional.smooth_l1_loss(net(condition), target[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 5000 == 0:
            print(json.dumps({"event": "chunk_bc_train", "step": step, "loss": float(loss.detach())}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "config": {
                "architecture": "directed_raw_chunk_bc_v1",
                "chunk": args.chunk,
                "action_dim": spec.action_dim,
                "cond_dim": condition_dim,
                "hidden": args.hidden,
                "emphasis_repeat": args.emphasis_repeat,
                "agent_dims": [gs, ge],
                "goal_dims": [ds, de],
                "route_start": args.route_start,
                "route_goal": args.route_goal,
                "route_radius": args.route_radius,
                "episodes": len(episodes),
                "pairs": len(target),
                "condition_final_goal": args.condition_final_goal,
                "input_noise": args.input_noise,
            },
        },
        args.out,
    )
    print(json.dumps({"event": "chunk_bc_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
