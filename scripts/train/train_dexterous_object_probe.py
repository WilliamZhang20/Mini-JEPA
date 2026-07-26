"""Calibrate an SE(3) readout on a frozen dexterous JEPA world model.

The probe is trained from observation fields already present in reward-free
play.  It sees both real encoded latents and imagined latents produced by the
frozen action-conditioned predictor, so MPC can score long rollouts in physical
object-pose geometry without changing the JEPA dynamics or learning a reward.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jepa_robotics.data import load_episodes_npz
from jepa_robotics.evaluate import load_jepa_artifact


def quaternion_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target, dim=-1)
    return (1.0 - (prediction * target).sum(-1).abs()).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--episodes-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout-weight", type=float, default=2.0)
    parser.add_argument("--rotation-weight", type=float, default=5.0)
    parser.add_argument("--max-episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2081)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(
        args.device
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model, normalizer, spec, config = load_jepa_artifact(
        args.model_path, device
    )
    if not hasattr(model, "predict_rollout_context"):
        raise ValueError("Object-probe calibration requires a contextual model")
    object_dims = config.get("object_dims")
    if not object_dims:
        raise ValueError("Checkpoint does not declare object_dims")
    object_lo, object_hi = (int(v) for v in object_dims)
    if object_hi - object_lo != 7:
        raise ValueError("Object probe expects position3 + quaternion4")

    # Replace the stale slot-only probe with a fresh full-latent readout. Delta
    # shaping may distribute transition information across hand/contact/object
    # slots, while the complete embedding still retains the physical pose.
    model.full_object_probe = True
    model.object_probe = nn.Sequential(
        nn.Linear(model.latent_dim, 2 * model.latent_dim),
        nn.GELU(),
        nn.Linear(2 * model.latent_dim, 7),
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.object_probe.parameters():
        parameter.requires_grad_(True)
    model.eval()
    model.object_probe.train()

    episodes = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    context_len = int(model.context_len)
    for episode in episodes:
        if len(episode.actions) < context_len + args.horizon:
            continue
        states.append(normalizer.encode(episode.states.astype(np.float32)))
        actions.append(episode.actions.astype(np.float32))
    if not states:
        raise ValueError("No episode is long enough for this calibration")

    object_mean = torch.as_tensor(
        normalizer.mean[object_lo:object_hi],
        dtype=torch.float32,
        device=device,
    )
    object_std = torch.as_tensor(
        normalizer.std[object_lo:object_hi],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.object_probe.parameters(), lr=args.lr, weight_decay=1e-4
    )
    print(
        json.dumps(
            {
                "event": "dexterous_object_probe_data",
                "episodes": len(states),
                "horizon": args.horizon,
                "context_len": context_len,
                "model": str(args.model_path),
            }
        ),
        flush=True,
    )

    def sample_batch():
        state_history, action_history, future_actions, future_states = [], [], [], []
        for _ in range(args.batch_size):
            episode_id = int(rng.integers(len(states)))
            t = int(
                rng.integers(
                    context_len - 1,
                    len(actions[episode_id]) - args.horizon + 1,
                )
            )
            state_history.append(
                states[episode_id][t - context_len + 1 : t + 1]
            )
            action_history.append(
                actions[episode_id][t - context_len + 1 : t]
            )
            future_actions.append(
                actions[episode_id][t : t + args.horizon]
            )
            future_states.append(
                states[episode_id][t + 1 : t + args.horizon + 1]
            )
        return (
            torch.from_numpy(np.stack(state_history)).to(device),
            torch.from_numpy(np.stack(action_history)).to(device),
            torch.from_numpy(np.stack(future_actions)).to(device),
            torch.from_numpy(np.stack(future_states)).to(device),
        )

    def pose_loss(latents: torch.Tensor, target_state: torch.Tensor):
        prediction_norm = model.predict_object(latents)
        target_norm = target_state[..., object_lo:object_hi]
        position = F.mse_loss(prediction_norm[..., :3], target_norm[..., :3])
        prediction_raw = prediction_norm * object_std + object_mean
        target_raw = target_norm * object_std + object_mean
        rotation = quaternion_loss(
            prediction_raw[..., 3:], target_raw[..., 3:]
        )
        return position + args.rotation_weight * rotation, rotation

    for step in range(1, args.steps + 1):
        state_history, action_history, future_actions, future_states = sample_batch()
        batch = state_history.shape[0]
        with torch.no_grad():
            z_history = model.encode(
                state_history.reshape(batch * context_len, -1)
            ).reshape(batch, context_len, -1)
            encoded_future = model.encode(
                future_states.reshape(batch * args.horizon, -1)
            ).reshape(batch, args.horizon, -1)
            rollout = model.predict_rollout_context(
                z_history,
                action_history,
                future_actions,
                args.horizon,
            )
        encoded_loss, encoded_rotation = pose_loss(
            encoded_future, future_states
        )
        rollout_loss, rollout_rotation = pose_loss(rollout, future_states)
        loss = encoded_loss + args.rollout_weight * rollout_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.object_probe.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 500 == 0:
            print(
                json.dumps(
                    {
                        "event": "dexterous_object_probe_train",
                        "step": step,
                        "loss": round(float(loss.detach()), 5),
                        "encoded_rotation_deg": round(
                            float(torch.rad2deg(2.0 * torch.acos(
                                (1.0 - encoded_rotation.detach()).clamp(0.0, 1.0)
                            ))),
                            3,
                        ),
                        "rollout_rotation_deg": round(
                            float(torch.rad2deg(2.0 * torch.acos(
                                (1.0 - rollout_rotation.detach()).clamp(0.0, 1.0)
                            ))),
                            3,
                        ),
                    }
                ),
                flush=True,
            )

    artifact = torch.load(
        args.model_path, map_location="cpu", weights_only=False
    )
    output_config = dict(artifact["config"])
    output_config.update(
        {
            "full_object_probe": True,
            "object_probe_calibration_horizon": args.horizon,
            "object_probe_rollout_weight": args.rollout_weight,
            "object_probe_rotation_weight": args.rotation_weight,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "normalizer": artifact["normalizer"],
            "spec": artifact["spec"],
            "config": output_config,
        },
        args.out,
    )
    print(
        json.dumps(
            {"event": "dexterous_object_probe_saved", "path": str(args.out)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
