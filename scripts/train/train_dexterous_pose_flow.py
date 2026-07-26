"""Train a hindsight SE(3)-conditioned flow prior for Shadow Hand action chunks.

Every target is an actually reached future from reward-free exploration:

    condition = [JEPA(s_t), delta_SE3(object_t, object_{t+H})]
    target    = action[t:t+H]

The flow therefore learns inverse dynamics from transition evidence rather than
expert action labels or reward optimization. At runtime the same six-dimensional
pose error can describe a small hierarchical step toward any requested pose.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.algos.task_families.dexterous import relative_pose_features
from jepa_robotics.data import load_episodes_npz
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import DexterousFlowPrior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--episodes-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--max-episodes", type=int, default=2000)
    parser.add_argument("--max-pairs", type=int, default=200000)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--flow-steps", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--position-scale", type=float, default=0.05)
    parser.add_argument("--motion-weight", type=float, default=0.5)
    parser.add_argument(
        "--action-representation",
        choices=["absolute", "delta"],
        default="absolute",
        help="Model absolute commands or standardized increments from the previous command.",
    )
    parser.add_argument(
        "--condition-prev-action",
        action="store_true",
        help="Append the action immediately before each chunk to the flow condition.",
    )
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    )
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    world_model, normalizer, spec, model_config = load_jepa_artifact(
        args.model_path, device
    )
    world_model.eval()
    for parameter in world_model.parameters():
        parameter.requires_grad_(False)

    condition_states: list[np.ndarray] = []
    pose_deltas: list[np.ndarray] = []
    chunks: list[np.ndarray] = []
    previous_actions: list[np.ndarray] = []
    motion: list[float] = []
    episodes = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    achieved = spec.obs_dim
    desired = spec.obs_dim + spec.goal_dim
    horizon = int(args.chunk)
    for episode in episodes:
        if len(episode.actions) < horizon:
            continue
        states = episode.states.astype(np.float32)
        for t in range(len(episode.actions) - horizon + 1):
            future_pose = states[t + horizon, achieved:achieved + 7]
            delta = relative_pose_features(
                states[t, achieved:achieved + 7],
                future_pose,
                position_scale=args.position_scale,
            )
            # The JEPA itself may carry a goal-relation token. Align that token
            # with this hindsight transition instead of leaving the episode's
            # unrelated environment goal in the encoded current state.
            conditioned_state = states[t].copy()
            conditioned_state[desired:desired + 7] = future_pose
            condition_states.append(conditioned_state)
            pose_deltas.append(delta)
            chunks.append(
                episode.actions[t:t + horizon].astype(np.float32)
            )
            previous_actions.append(
                episode.actions[t - 1].astype(np.float32)
                if t > 0
                else np.zeros(spec.action_dim, dtype=np.float32)
            )
            motion.append(float(np.linalg.norm(delta)))

    if len(condition_states) > args.max_pairs:
        keep = rng.choice(len(condition_states), args.max_pairs, replace=False)
        condition_states = [condition_states[index] for index in keep]
        pose_deltas = [pose_deltas[index] for index in keep]
        chunks = [chunks[index] for index in keep]
        previous_actions = [previous_actions[index] for index in keep]
        motion = [motion[index] for index in keep]
    normalized_states = normalizer.encode(
        np.asarray(condition_states, dtype=np.float32)
    ).astype(np.float32)
    latent_batches = []
    with torch.no_grad():
        for start in range(0, len(normalized_states), 8192):
            latent_batches.append(
                world_model.encode(
                    torch.from_numpy(normalized_states[start:start + 8192]).to(device)
                )
            )
    latents = torch.cat(latent_batches)
    pose_delta = torch.from_numpy(
        np.asarray(pose_deltas, dtype=np.float32)
    ).to(device)
    condition_parts = [latents, pose_delta]
    previous_action_array = np.asarray(previous_actions, dtype=np.float32)
    if args.condition_prev_action:
        condition_parts.append(
            torch.from_numpy(previous_action_array).to(device)
        )
    condition = torch.cat(condition_parts, dim=-1)
    action_array = np.asarray(chunks, dtype=np.float32)
    action_delta_mean = np.zeros(spec.action_dim, dtype=np.float32)
    action_delta_std = np.ones(spec.action_dim, dtype=np.float32)
    if args.action_representation == "delta":
        increments = np.diff(
            np.concatenate(
                [previous_action_array[:, None, :], action_array], axis=1
            ),
            axis=1,
        )
        flat_increments = increments.reshape(-1, spec.action_dim)
        action_delta_mean = flat_increments.mean(0).astype(np.float32)
        action_delta_std = flat_increments.std(0).clip(1e-3).astype(np.float32)
        action_array = (
            increments - action_delta_mean[None, None, :]
        ) / action_delta_std[None, None, :]
    action_chunk = torch.from_numpy(
        action_array.reshape(len(action_array), -1).astype(np.float32)
    ).to(device)
    sample_weight = torch.as_tensor(motion, dtype=torch.float32, device=device)
    sample_weight = (1.0 + args.motion_weight * sample_weight).clamp_min(1e-6)
    sample_weight /= sample_weight.sum()

    flow = DexterousFlowPrior(
        int(action_chunk.shape[1]),
        int(condition.shape[1]),
        hidden=args.hidden,
        n_blocks=args.blocks,
        heads=args.heads,
        action_dim=spec.action_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = {key: value.detach().clone() for key, value in flow.state_dict().items()}
    print(
        json.dumps(
            {
                "event": "dexterous_pose_flow_data",
                "pairs": len(condition),
                "condition_dim": int(condition.shape[1]),
                "chunk": horizon,
                "parameters_M": round(sum(p.numel() for p in flow.parameters()) / 1e6, 2),
                "motion_p50": round(float(np.percentile(motion, 50)), 3),
                "motion_p90": round(float(np.percentile(motion, 90)), 3),
            }
        ),
        flush=True,
    )
    for step in range(1, args.steps + 1):
        index = torch.multinomial(sample_weight, args.batch_size, replacement=True)
        target = action_chunk[index]
        cond = condition[index]
        noise = torch.randn_like(target)
        tau = torch.rand(target.shape[0], device=device)
        noisy = (1.0 - tau[:, None]) * noise + tau[:, None] * target
        velocity = flow(noisy, tau * args.flow_steps, cond)
        loss = nn.functional.mse_loss(velocity, target - noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for key, value in flow.state_dict().items():
                ema[key].mul_(0.999).add_(value.detach(), alpha=0.001)
        if step == 1 or step % 2000 == 0:
            print(
                json.dumps(
                    {
                        "event": "dexterous_pose_flow_train",
                        "step": step,
                        "loss": round(float(loss.detach()), 5),
                    }
                ),
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ema": ema,
            "state_dict": flow.state_dict(),
            "config": {
                "architecture": "dexterous_pose_flow",
                "chunk_dim": int(action_chunk.shape[1]),
                "condition_dim": int(condition.shape[1]),
                "action_dim": int(spec.action_dim),
                "chunk": horizon,
                "hidden": int(args.hidden),
                "blocks": int(args.blocks),
                "heads": int(args.heads),
                "flow_steps": int(args.flow_steps),
                "position_scale": float(args.position_scale),
                "model_path": str(args.model_path),
                "episodes_npz": str(args.episodes_npz),
                "model_latent_dim": int(model_config["latent_dim"]),
                "aligned_model_goal": True,
                "action_representation": args.action_representation,
                "condition_prev_action": bool(args.condition_prev_action),
                "action_delta_mean": action_delta_mean.tolist(),
                "action_delta_std": action_delta_std.tolist(),
            },
        },
        args.out,
    )
    print(
        json.dumps({"event": "dexterous_pose_flow_saved", "path": str(args.out)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
