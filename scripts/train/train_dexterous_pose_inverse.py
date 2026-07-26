"""Train multi-hypothesis JEPA inverse dynamics from reward-free transitions."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.algos.task_families.dexterous import relative_pose_features
from jepa_robotics.data import load_episodes_npz
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import DexterousInverseDynamics


def quaternion_geodesic(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    prediction = prediction / torch.linalg.vector_norm(
        prediction, dim=-1, keepdim=True
    ).clamp_min(1e-6)
    target = target / torch.linalg.vector_norm(
        target, dim=-1, keepdim=True
    ).clamp_min(1e-6)
    return 2.0 * torch.acos(
        (prediction * target).sum(-1).abs().clamp(max=1.0)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--episodes-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--motion-weight", type=float, default=3.0)
    parser.add_argument("--lambda-forward", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=941)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(
        args.device
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    world_model, normalizer, spec, model_config = load_jepa_artifact(
        args.model_path, device
    )
    world_model.eval()
    for parameter in world_model.parameters():
        parameter.requires_grad_(False)

    states, future_objects, pose_deltas, chunks, previous_actions, motion = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    achieved = spec.obs_dim
    desired = spec.obs_dim + spec.goal_dim
    horizon = args.horizon
    for episode in load_episodes_npz(args.episodes_npz):
        for index in range(len(episode.actions) - horizon + 1):
            state = episode.states[index].astype(np.float32).copy()
            future = episode.states[
                index + horizon, achieved:achieved + 7
            ].astype(np.float32)
            delta = relative_pose_features(
                state[achieved:achieved + 7],
                future,
                position_scale=0.05,
            )
            state[desired:desired + 7] = future
            states.append(state)
            future_objects.append(future)
            pose_deltas.append(delta)
            chunks.append(
                episode.actions[
                    index:index + horizon
                ].astype(np.float32)
            )
            previous_actions.append(
                episode.actions[index - 1].astype(np.float32)
                if index > 0
                else np.zeros(spec.action_dim, dtype=np.float32)
            )
            motion.append(float(np.linalg.norm(delta)))
    if len(states) > args.max_pairs:
        keep = rng.choice(len(states), args.max_pairs, replace=False)
        states = [states[index] for index in keep]
        future_objects = [future_objects[index] for index in keep]
        pose_deltas = [pose_deltas[index] for index in keep]
        chunks = [chunks[index] for index in keep]
        previous_actions = [previous_actions[index] for index in keep]
        motion = [motion[index] for index in keep]

    states_array = np.asarray(states, dtype=np.float32)
    normalized = normalizer.encode(states_array).astype(np.float32)
    latent_batches = []
    with torch.no_grad():
        for start in range(0, len(normalized), 8192):
            latent_batches.append(
                world_model.encode(
                    torch.from_numpy(
                        normalized[start:start + 8192]
                    ).to(device)
                )
            )
    latents = torch.cat(latent_batches)
    pose_delta = torch.from_numpy(
        np.asarray(pose_deltas, dtype=np.float32)
    ).to(device)
    previous = np.asarray(previous_actions, dtype=np.float32)
    previous_tensor = torch.from_numpy(previous).to(device)
    condition = torch.cat([latents, pose_delta, previous_tensor], dim=-1)

    action_array = np.asarray(chunks, dtype=np.float32)
    increments = np.diff(
        np.concatenate([previous[:, None, :], action_array], axis=1),
        axis=1,
    )
    flat_increments = increments.reshape(-1, spec.action_dim)
    increment_mean = flat_increments.mean(0).astype(np.float32)
    increment_std = flat_increments.std(0).clip(1e-3).astype(np.float32)
    increment_target = torch.from_numpy(
        (
            (increments - increment_mean[None, None, :])
            / increment_std[None, None, :]
        ).astype(np.float32)
    ).to(device)
    future_object = torch.from_numpy(
        np.asarray(future_objects, dtype=np.float32)
    ).to(device)
    sample_weight = torch.as_tensor(
        motion, dtype=torch.float32, device=device
    )
    sample_weight = 1.0 + args.motion_weight * sample_weight
    sample_weight /= sample_weight.sum()
    increment_mean_t = torch.from_numpy(increment_mean).to(device)
    increment_std_t = torch.from_numpy(increment_std).to(device)
    object_mean = torch.as_tensor(
        normalizer.mean[achieved:achieved + 7],
        dtype=torch.float32,
        device=device,
    )
    object_std = torch.as_tensor(
        normalizer.std[achieved:achieved + 7],
        dtype=torch.float32,
        device=device,
    )

    inverse = DexterousInverseDynamics(
        int(condition.shape[1]),
        horizon,
        spec.action_dim,
        hidden=args.hidden,
        n_blocks=args.blocks,
        heads=args.heads,
        modes=args.modes,
    ).to(device)
    optimizer = torch.optim.AdamW(
        inverse.parameters(), lr=args.lr, weight_decay=1e-4
    )
    print(
        json.dumps(
            {
                "event": "dexterous_pose_inverse_data",
                "pairs": len(condition),
                "condition_dim": int(condition.shape[1]),
                "horizon": horizon,
                "modes": args.modes,
                "parameters_M": round(
                    sum(p.numel() for p in inverse.parameters()) / 1e6, 2
                ),
            }
        ),
        flush=True,
    )
    for step in range(1, args.steps + 1):
        index = torch.multinomial(
            sample_weight, args.batch_size, replacement=True
        )
        prediction = inverse(condition[index])
        reconstruction = torch.square(
            prediction - increment_target[index, None]
        ).mean(dim=(2, 3))
        winner = reconstruction.argmin(1)
        batch_index = torch.arange(args.batch_size, device=device)
        selected = prediction[batch_index, winner]
        reconstruction_loss = reconstruction[batch_index, winner].mean()
        decoded_increments = (
            selected * increment_std_t + increment_mean_t
        )
        predicted_actions = (
            previous_tensor[index, None]
            + torch.cumsum(decoded_increments, dim=1)
        ).clamp(-1.0, 1.0)
        rollout = world_model.predict_rollout(
            latents[index], predicted_actions, horizon
        )
        predicted_object_norm = world_model.predict_object(rollout[:, -1])
        predicted_object = (
            predicted_object_norm * object_std + object_mean
        )
        position_loss = torch.square(
            (
                predicted_object[:, :3]
                - future_object[index, :3]
            )
            / 0.05
        ).mean()
        rotation_loss = torch.square(
            quaternion_geodesic(
                predicted_object[:, 3:],
                future_object[index, 3:],
            )
        ).mean()
        forward_loss = position_loss + rotation_loss
        loss = reconstruction_loss + args.lambda_forward * forward_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(inverse.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 2000 == 0:
            print(
                json.dumps(
                    {
                        "event": "dexterous_pose_inverse_train",
                        "step": step,
                        "loss": round(float(loss.detach()), 5),
                        "reconstruction": round(
                            float(reconstruction_loss.detach()), 5
                        ),
                        "forward": round(float(forward_loss.detach()), 5),
                    }
                ),
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": inverse.state_dict(),
            "config": {
                "architecture": "dexterous_pose_inverse",
                "condition_dim": int(condition.shape[1]),
                "action_dim": int(spec.action_dim),
                "horizon": horizon,
                "hidden": args.hidden,
                "blocks": args.blocks,
                "heads": args.heads,
                "modes": args.modes,
                "model_path": str(args.model_path),
                "episodes_npz": str(args.episodes_npz),
                "model_latent_dim": int(model_config["latent_dim"]),
                "position_scale": 0.05,
                "condition_prev_action": True,
                "aligned_model_goal": True,
                "action_representation": "delta",
                "action_delta_mean": increment_mean.tolist(),
                "action_delta_std": increment_std.tolist(),
                "lambda_forward": args.lambda_forward,
            },
        },
        args.out,
    )
    print(
        json.dumps(
            {
                "event": "dexterous_pose_inverse_saved",
                "path": str(args.out),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
