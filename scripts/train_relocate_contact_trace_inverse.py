"""Train a Relocate inverse prior conditioned on a demo contact trace.

Terminal future latents are too coarse for Relocate. This trainer adds an
explicit self-supervised contact trajectory to the condition:

    contact_trace[i] = [||palm-ball||, ||ball-target||] at s_{t+i}

Demos/trials still define desirable futures and causal action chunks; runtime
does not copy action labels, it retrieves a contact trace/future and predicts
the action chunk from current latent, future latent, raw states, and trace.
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

from jepa_robotics.algos.priors import InversePrior
from jepa_robotics.data import load_episodes_npz
from jepa_robotics.evaluate import load_jepa_artifact


def contact_trace(states: np.ndarray, start: int, horizon: int) -> np.ndarray:
    future = states[start + 1 : start + horizon + 1]
    palm_ball = np.linalg.norm(future[:, 30:33], axis=-1)
    ball_target = np.linalg.norm(future[:, 36:39], axis=-1)
    return np.stack([palm_ball, ball_target], axis=-1).reshape(-1).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument("--max-bank", type=int, default=50000)
    p.add_argument("--train-steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=331)
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

    H = args.chunk
    conds, chunks = [], []
    bank_states, bank_futures, bank_traces = [], [], []
    with torch.no_grad():
        for ep in load_episodes_npz(args.episodes_npz)[: args.max_episodes]:
            states = ep.states.astype(np.float32)
            actions = ep.actions.astype(np.float32)
            if len(actions) < H or states.shape[-1] < 39:
                continue
            norm_states_np = norm.encode(states).astype(np.float32)
            norm_states = torch.from_numpy(norm_states_np).to(dev)
            z_online = wm.encode(norm_states)
            z_target = wm.encode_target(norm_states)
            for t in range(len(actions) - H + 1):
                trace = contact_trace(states, t, H)
                trace_t = torch.from_numpy(trace).to(dev)
                h_token = torch.tensor([1.0], dtype=z_online.dtype, device=dev)
                parts = [
                    z_online[t],
                    z_target[t + H],
                    h_token,
                    torch.from_numpy(norm_states_np[t]).to(dev),
                    torch.from_numpy(norm_states_np[t + H]).to(dev),
                    trace_t,
                ]
                conds.append(torch.cat(parts, dim=-1))
                chunks.append(torch.from_numpy(actions[t : t + H].reshape(-1)).to(dev))
                bank_states.append(states[t])
                bank_futures.append(states[t + H])
                bank_traces.append(trace)

    Cond = torch.stack(conds)
    Chunk = torch.stack(chunks)
    bank_states_np = np.asarray(bank_states, dtype=np.float32)
    bank_futures_np = np.asarray(bank_futures, dtype=np.float32)
    bank_traces_np = np.asarray(bank_traces, dtype=np.float32)
    if len(bank_states_np) > args.max_bank:
        idx = rng.choice(len(bank_states_np), size=args.max_bank, replace=False)
        bank_states_np = bank_states_np[idx]
        bank_futures_np = bank_futures_np[idx]
        bank_traces_np = bank_traces_np[idx]

    prior = InversePrior(int(Cond.shape[1]), int(Chunk.shape[1]), args.hidden, args.n_blocks).to(dev)
    opt = torch.optim.AdamW(prior.parameters(), lr=args.lr, weight_decay=1e-4)
    print(
        json.dumps(
            {
                "event": "relocate_trace_inverse_data",
                "pairs": int(Cond.shape[0]),
                "bank": int(len(bank_states_np)),
                "chunk": int(H),
                "cond_dim": int(Cond.shape[1]),
                "chunk_dim": int(Chunk.shape[1]),
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
            print(json.dumps({"event": "relocate_trace_inverse_train", "step": step, "loss": float(loss.detach().cpu())}), flush=True)

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
            "trace_dim": int(2 * H),
            "bank_states": bank_states_np,
            "bank_futures": bank_futures_np,
            "bank_traces": bank_traces_np,
            "model_path": str(args.model_path),
            "episodes_npz": str(args.episodes_npz),
        },
        args.out,
    )
    print(json.dumps({"event": "relocate_trace_inverse_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
