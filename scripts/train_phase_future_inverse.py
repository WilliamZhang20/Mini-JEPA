"""Train a phase-conditioned future inverse prior for flat demo tasks.

This is the hierarchical variant of the flat Adroit inverse prior. Demonstrations
are split into self-supervised temporal phases, then each training tuple uses:

    z_t = encoder(s_t)
    z_future = target_encoder(s_{t+h})
    inverse(z_t, z_future, h, phase_t, phase_future) -> a_{t:t+H-1}

The phases are not action labels. They are progress/subgoal structure induced
from trajectories so demos specify desirable future manifolds while trials/demos
still teach which action chunks realize those futures.
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
from jepa_robotics.algos.phase import batch_phase_features
from jepa_robotics.algos.priors import parse_horizons
from jepa_robotics.algos.priors import InversePrior


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--future-horizons", type=parse_horizons, default=None)
    p.add_argument("--n-phases", type=int, default=4)
    p.add_argument("--concat-raw", action="store_true",
                   help="Append normalized current/future state to the latent+phase condition.")
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument("--max-bank", type=int, default=50000)
    p.add_argument("--train-steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=131)
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
    conds, chunks = [], []
    bank_states, bank_futures, bank_phase, bank_target_phase, bank_progress, bank_target_progress = [], [], [], [], [], []
    with torch.no_grad():
        for ep in episodes:
            states = ep.states.astype(np.float32)
            actions = ep.actions.astype(np.float32)
            T = len(actions)
            if T < max_future:
                continue
            norm_states_np = norm.encode(states).astype(np.float32)
            norm_states = torch.from_numpy(norm_states_np).to(dev)
            z_online = wm.encode(norm_states)
            z_target = wm.encode_target(norm_states)
            n = T - max_future + 1
            ts = np.arange(n, dtype=np.int64)
            cur_progress = (ts.astype(np.float32) / float(max(1, T))).astype(np.float32)
            cur_phase = np.clip(np.floor(cur_progress * args.n_phases), 0, args.n_phases - 1).astype(np.int64)
            for future_h in future_horizons:
                fts = ts + int(future_h)
                target_progress = (fts.astype(np.float32) / float(max(1, T))).astype(np.float32)
                target_phase = np.clip(np.floor(target_progress * args.n_phases), 0, args.n_phases - 1).astype(np.int64)
                h_token = torch.full(
                    (n, 1),
                    float(future_h) / float(max(future_horizons)),
                    dtype=z_online.dtype,
                    device=dev,
                )
                phase_feat = torch.from_numpy(
                    batch_phase_features(cur_phase, target_phase, cur_progress, target_progress, args.n_phases)
                ).to(dev)
                parts = [z_online[ts], z_target[fts], h_token, phase_feat]
                if args.concat_raw:
                    parts.extend(
                        [
                            torch.from_numpy(norm_states_np[ts]).to(dev),
                            torch.from_numpy(norm_states_np[fts]).to(dev),
                        ]
                    )
                conds.append(torch.cat(parts, dim=-1))
                chunks.append(torch.from_numpy(np.stack([actions[t : t + H].reshape(-1) for t in ts])).to(dev))
            fts = ts + max(future_horizons)
            bank_states.append(states[ts])
            bank_futures.append(states[fts])
            bank_phase.append(cur_phase)
            bank_target_phase.append(np.clip(np.floor((fts.astype(np.float32) / float(max(1, T))) * args.n_phases), 0, args.n_phases - 1).astype(np.int64))
            bank_progress.append(cur_progress)
            bank_target_progress.append((fts.astype(np.float32) / float(max(1, T))).astype(np.float32))

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
                "event": "phase_inverse_data",
                "pairs": int(Cond.shape[0]),
                "bank": int(len(bank_states_np)),
                "chunk": H,
                "cond_dim": int(Cond.shape[1]),
                "chunk_dim": int(Chunk.shape[1]),
                "n_phases": int(args.n_phases),
                "concat_raw": bool(args.concat_raw),
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
        if step == 1 or step % 2000 == 0:
            print(json.dumps({"event": "phase_inverse_train", "step": step, "loss": float(loss.detach().cpu())}), flush=True)

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
            "concat_raw": bool(args.concat_raw),
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
    print(json.dumps({"event": "phase_inverse_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
