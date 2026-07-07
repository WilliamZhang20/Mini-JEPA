"""Train a future-conditioned inverse chunk prior for flat offline-demo tasks.

For non-goal tasks such as Adroit, demos supply desirable futures but there is no
explicit desired_goal field. Training uses self-supervised transition tuples:

    z_t = encoder(s_t)
    z_future = target_encoder(s_{t+h})
    inverse(z_t, z_future, h) -> a_{t:t+H-1}

The saved demo future bank lets evaluation retrieve a nearby demonstrated future
state without copying its action label.
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

from jepa_robotics.data import load_episodes_npz
from jepa_robotics.evaluate import load_jepa_artifact
from scripts.train_fetch_flow_prior import parse_horizons
from jepa_robotics.algos.priors import InversePrior


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--future-horizons", type=parse_horizons, default=None)
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument("--max-bank", type=int, default=50000)
    p.add_argument("--train-steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=83)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
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
    conds, chunks = [], []
    bank_states, bank_futures = [], []
    with torch.no_grad():
        for ep in episodes:
            S = ep.states.astype(np.float32)
            A = ep.actions.astype(np.float32)
            if len(A) < max_future:
                continue
            Sn = torch.from_numpy(norm.encode(S)).to(dev)
            z_online = wm.encode(Sn)
            z_target = wm.encode_target(Sn)
            for t in range(len(A) - max_future + 1):
                for future_h in future_horizons:
                    h_token = torch.tensor([float(future_h) / float(max(future_horizons))], dtype=z_online.dtype, device=dev)
                    conds.append(torch.cat([z_online[t], z_target[t + future_h], h_token], dim=-1))
                    chunks.append(torch.from_numpy(A[t : t + H].reshape(-1)).to(dev))
                bank_states.append(S[t])
                bank_futures.append(S[t + max(future_horizons)])

    Cond = torch.stack(conds)
    Chunk = torch.stack(chunks)
    bank_states_np = np.asarray(bank_states, dtype=np.float32)
    bank_futures_np = np.asarray(bank_futures, dtype=np.float32)
    if len(bank_states_np) > args.max_bank:
        idx = rng.choice(len(bank_states_np), size=args.max_bank, replace=False)
        bank_states_np = bank_states_np[idx]
        bank_futures_np = bank_futures_np[idx]

    prior = InversePrior(Cond.shape[1], Chunk.shape[1], args.hidden, args.n_blocks).to(dev)
    opt = torch.optim.AdamW(prior.parameters(), lr=args.lr, weight_decay=1e-4)
    print(json.dumps({"event": "flat_inverse_data", "pairs": int(Cond.shape[0]), "bank": int(len(bank_states_np)), "chunk": H, "cond_dim": int(Cond.shape[1]), "chunk_dim": int(Chunk.shape[1])}), flush=True)
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, Cond.shape[0], (args.batch_size,), device=dev)
        pred = prior(Cond[idx])
        loss = nn.functional.mse_loss(pred, Chunk[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 2000 == 0:
            print(json.dumps({"event": "flat_inverse_train", "step": step, "loss": float(loss.detach().cpu())}), flush=True)

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
            "bank_states": bank_states_np,
            "bank_futures": bank_futures_np,
            "model_path": str(args.model_path),
            "episodes_npz": str(args.episodes_npz),
        },
        args.out,
    )
    print(json.dumps({"event": "flat_inverse_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
