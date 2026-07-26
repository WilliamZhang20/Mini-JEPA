"""Train a reward-free action-chunk prior on dexterous play trajectories.

The prior models smooth action increments conditioned on a short JEPA
state/action history. It is not goal-conditioned and is never optimized by a
task reward; MPC uses it only to keep candidate plans on the data manifold,
while the world-model goal energy selects among sampled play continuations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jepa_robotics.data import load_episodes_npz
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import DexterousFlowPrior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--episodes-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--flow-steps", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-episodes", type=int, default=2000)
    parser.add_argument("--max-pairs", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=2351)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(
        args.device
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model, normalizer, spec, model_config = load_jepa_artifact(
        args.model_path, device
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    context_len = int(getattr(model, "context_len", 1))

    raw_histories: list[np.ndarray] = []
    action_histories: list[np.ndarray] = []
    chunks: list[np.ndarray] = []
    episodes = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    for episode in episodes:
        if len(episode.actions) < context_len + args.chunk:
            continue
        normalized = normalizer.encode(episode.states.astype(np.float32))
        for t in range(context_len - 1, len(episode.actions) - args.chunk + 1):
            raw_histories.append(
                normalized[t - context_len + 1 : t + 1]
            )
            action_histories.append(
                episode.actions[t - context_len + 1 : t].astype(np.float32)
            )
            chunks.append(
                episode.actions[t : t + args.chunk].astype(np.float32)
            )
    if len(chunks) > args.max_pairs:
        keep = rng.choice(len(chunks), args.max_pairs, replace=False)
        raw_histories = [raw_histories[i] for i in keep]
        action_histories = [action_histories[i] for i in keep]
        chunks = [chunks[i] for i in keep]

    state_history = torch.from_numpy(np.stack(raw_histories))
    encoded_batches = []
    with torch.no_grad():
        for start in range(0, len(state_history), 4096):
            batch = state_history[start : start + 4096].to(device)
            encoded_batches.append(
                model.encode(batch.reshape(-1, spec.state_dim))
                .reshape(len(batch), context_len, -1)
                .cpu()
            )
    encoded_history = torch.cat(encoded_batches)
    action_history_np = np.stack(action_histories).astype(np.float32)
    condition = torch.cat(
        [
            encoded_history.flatten(1),
            torch.from_numpy(action_history_np).flatten(1),
        ],
        dim=-1,
    ).to(device)

    action_chunks = np.stack(chunks).astype(np.float32)
    previous = (
        action_history_np[:, -1]
        if context_len > 1
        else np.zeros((len(chunks), spec.action_dim), dtype=np.float32)
    )
    increments = np.diff(
        np.concatenate([previous[:, None], action_chunks], axis=1), axis=1
    )
    increment_mean = increments.reshape(-1, spec.action_dim).mean(0).astype(np.float32)
    increment_std = (
        increments.reshape(-1, spec.action_dim).std(0).clip(1e-3).astype(np.float32)
    )
    standardized = (
        increments - increment_mean[None, None]
    ) / increment_std[None, None]
    target = torch.from_numpy(standardized.reshape(len(chunks), -1)).to(device)

    flow = DexterousFlowPrior(
        target.shape[1],
        condition.shape[1],
        hidden=args.hidden,
        n_blocks=args.blocks,
        heads=args.heads,
        action_dim=spec.action_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = {
        key: value.detach().clone() for key, value in flow.state_dict().items()
    }
    print(
        json.dumps(
            {
                "event": "dexterous_play_prior_data",
                "pairs": len(target),
                "context_len": context_len,
                "condition_dim": condition.shape[1],
                "chunk": args.chunk,
                "parameters_M": round(
                    sum(p.numel() for p in flow.parameters()) / 1e6, 3
                ),
                "increment_rms": float(np.sqrt(np.mean(increments ** 2))),
            }
        ),
        flush=True,
    )
    for step in range(1, args.steps + 1):
        index = torch.randint(len(target), (args.batch_size,), device=device)
        clean = target[index]
        cond = condition[index]
        noise = torch.randn_like(clean)
        tau = torch.rand(len(clean), device=device)
        noisy = (1.0 - tau[:, None]) * noise + tau[:, None] * clean
        velocity = flow(noisy, tau * args.flow_steps, cond)
        loss = nn.functional.mse_loss(velocity, clean - noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for key, value in flow.state_dict().items():
                ema[key].mul_(0.999).add_(value.detach(), alpha=0.001)
        if step == 1 or step % 1000 == 0:
            print(
                json.dumps(
                    {
                        "event": "dexterous_play_prior_train",
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
                "architecture": "dexterous_play_flow",
                "chunk_dim": int(target.shape[1]),
                "condition_dim": int(condition.shape[1]),
                "action_dim": int(spec.action_dim),
                "chunk": args.chunk,
                "context_len": context_len,
                "hidden": args.hidden,
                "blocks": args.blocks,
                "heads": args.heads,
                "flow_steps": args.flow_steps,
                "model_path": str(args.model_path),
                "episodes_npz": str(args.episodes_npz),
                "increment_mean": increment_mean.tolist(),
                "increment_std": increment_std.tolist(),
            },
        },
        args.out,
    )
    print(
        json.dumps(
            {"event": "dexterous_play_prior_saved", "path": str(args.out)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
