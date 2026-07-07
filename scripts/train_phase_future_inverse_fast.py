"""Vectorized phase-conditioned inverse prior training for flat demo tasks.

This is the same objective as ``train_phase_future_inverse.py`` but caches all
episode latents in large batches first. It avoids per-episode encoder overhead,
which is the bottleneck on large Adroit datasets.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.algos.phase import batch_phase_features
from jepa_robotics.algos.priors import InversePrior
from jepa_robotics.data import load_episodes_npz
from jepa_robotics.evaluate import load_jepa_artifact
from scripts.train_fetch_flow_prior import parse_horizons


@torch.no_grad()
def encode_batches(wm, arr: np.ndarray, device, batch: int = 32768, target: bool = False) -> torch.Tensor:
    outs = []
    for i in range(0, len(arr), batch):
        x = torch.from_numpy(arr[i : i + batch]).to(device)
        outs.append(wm.encode_target(x) if target else wm.encode(x))
    return torch.cat(outs, dim=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--future-horizons", type=parse_horizons, default=None)
    p.add_argument("--n-phases", type=int, default=4)
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument("--max-bank", type=int, default=60000)
    p.add_argument("--train-steps", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=137)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for param in wm.parameters():
        param.requires_grad_(False)

    episodes = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    H = args.chunk
    future_horizons = args.future_horizons or [H]
    max_future = max(max(future_horizons), H)

    all_states, all_actions, offsets, lengths = [], [], [], []
    cursor = 0
    for ep in episodes:
        states = ep.states.astype(np.float32)
        actions = ep.actions.astype(np.float32)
        if len(actions) < max_future:
            continue
        all_states.append(states)
        all_actions.append(actions)
        offsets.append(cursor)
        lengths.append(len(actions))
        cursor += len(states)
    states_np = np.concatenate(all_states, axis=0).astype(np.float32)
    norm_states_np = norm.encode(states_np).astype(np.float32)
    z_online = encode_batches(wm, norm_states_np, dev, target=False)
    z_target = encode_batches(wm, norm_states_np, dev, target=True)

    conds, chunks = [], []
    bank_states, bank_futures, bank_phase, bank_target_phase, bank_progress, bank_target_progress = [], [], [], [], [], []
    for actions, off, T in zip(all_actions, offsets, lengths):
        n = T - max_future + 1
        ts = np.arange(n, dtype=np.int64)
        cur_progress = ts.astype(np.float32) / float(max(1, T))
        cur_phase = np.clip(np.floor(cur_progress * args.n_phases), 0, args.n_phases - 1).astype(np.int64)
        for future_h in future_horizons:
            fts = ts + int(future_h)
            target_progress = fts.astype(np.float32) / float(max(1, T))
            target_phase = np.clip(np.floor(target_progress * args.n_phases), 0, args.n_phases - 1).astype(np.int64)
            h_token = torch.full((n, 1), float(future_h) / float(max(future_horizons)), dtype=z_online.dtype, device=dev)
            phase_feat = torch.from_numpy(
                batch_phase_features(cur_phase, target_phase, cur_progress, target_progress, args.n_phases)
            ).to(dev)
            conds.append(torch.cat([z_online[off + ts], z_target[off + fts], h_token, phase_feat], dim=-1))
            chunks.append(torch.from_numpy(np.stack([actions[t : t + H].reshape(-1) for t in ts])).to(dev))
        fts = ts + max(future_horizons)
        bank_states.append(states_np[off + ts])
        bank_futures.append(states_np[off + fts])
        bank_phase.append(cur_phase)
        bank_target_phase.append(np.clip(np.floor((fts.astype(np.float32) / float(max(1, T))) * args.n_phases), 0, args.n_phases - 1).astype(np.int64))
        bank_progress.append(cur_progress)
        bank_target_progress.append(fts.astype(np.float32) / float(max(1, T)))

    Cond = torch.cat(conds, dim=0)
    Chunk = torch.cat(chunks, dim=0)
    bank_states_np = np.concatenate(bank_states, axis=0).astype(np.float32)
    bank_futures_np = np.concatenate(bank_futures, axis=0).astype(np.float32)
    bank_phase_np = np.concatenate(bank_phase, axis=0).astype(np.int64)
    bank_target_phase_np = np.concatenate(bank_target_phase, axis=0).astype(np.int64)
    bank_progress_np = np.concatenate(bank_progress, axis=0).astype(np.float32)
    bank_target_progress_np = np.concatenate(bank_target_progress, axis=0).astype(np.float32)
    if len(bank_states_np) > args.max_bank:
        idx = rng.choice(len(bank_states_np), size=args.max_bank, replace=False)
        bank_states_np = bank_states_np[idx]
        bank_futures_np = bank_futures_np[idx]
        bank_phase_np = bank_phase_np[idx]
        bank_target_phase_np = bank_target_phase_np[idx]
        bank_progress_np = bank_progress_np[idx]
        bank_target_progress_np = bank_target_progress_np[idx]

    prior = InversePrior(Cond.shape[1], Chunk.shape[1], args.hidden, args.n_blocks).to(dev)
    opt = torch.optim.AdamW(prior.parameters(), lr=args.lr, weight_decay=1e-4)
    print(
        json.dumps(
            {
                "event": "phase_inverse_fast_data",
                "pairs": int(Cond.shape[0]),
                "bank": int(len(bank_states_np)),
                "chunk": int(H),
                "cond_dim": int(Cond.shape[1]),
                "chunk_dim": int(Chunk.shape[1]),
                "n_phases": int(args.n_phases),
            }
        ),
        flush=True,
    )
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, Cond.shape[0], (args.batch_size,), device=dev)
        pred = prior(Cond[idx])
        loss = nn.functional.mse_loss(pred, Chunk[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 1000 == 0:
            print(json.dumps({"event": "phase_inverse_fast_train", "step": int(step), "loss": float(loss.detach().cpu())}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": prior.state_dict(),
            "cond_dim": int(Cond.shape[1]),
            "chunk_dim": int(Chunk.shape[1]),
            "action_dim": int(spec.action_dim),
            "H": int(H),
            "latent_dim": int(cfg["latent_dim"]),
            "hidden": int(args.hidden),
            "n_blocks": int(args.n_blocks),
            "future_horizons": future_horizons,
            "n_phases": int(args.n_phases),
            "concat_raw": False,
            "bank_states": bank_states_np,
            "bank_futures": bank_futures_np,
            "bank_phase": bank_phase_np,
            "bank_target_phase": bank_target_phase_np,
            "bank_progress": bank_progress_np,
            "bank_target_progress": bank_target_progress_np,
            "model_path": str(args.model_path),
            "episodes_npz": str(args.episodes_npz),
        },
        args.out,
    )
    print(json.dumps({"event": "phase_inverse_fast_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
