"""Train a flat-task future-conditioned flow prior from offline demonstrations.

The training tuples are self-supervised transitions:

    z_t = encoder(s_t)
    z_future = target_encoder(s_{t+h})
    flow(a_{t:t+H-1} | z_t, z_future, h)

Demos define reachable/desirable futures. The saved future bank is used at
evaluation to retrieve a local demo future; action labels are not copied at
runtime.
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
from jepa_robotics.algos.priors import EpsNet
from scripts.train_fetch_flow_prior import parse_horizons


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--future-horizons", type=parse_horizons, default=None)
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument("--max-bank", type=int, default=50000)
    p.add_argument("--train-steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--flow-steps", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--concat-raw", action="store_true",
                   help="Append normalized current/future states to the latent condition.")
    p.add_argument("--require-possession", choices=["none", "held", "free"], default="none",
                   help="Segment-pure flow specialist: 'free' keeps only pairs whose current frame is NOT in possession, 'held' keeps only pairs held at both frames. Splitting the bimodal contact regime de-blurs the expert manifold so flow stops mode-averaging across regimes.")
    p.add_argument("--possession-dims", default="30,33",
                   help="Raw-state slice (lo,hi) whose vector norm defines the possession predicate.")
    p.add_argument("--possession-threshold", type=float, default=0.06)
    p.add_argument("--emphasis-dims", default=None,
                   help="Raw-state slice (lo,hi) of the CURRENT state to duplicate --emphasis-repeat extra times in the flow conditioning, servoing samples to the live contact geometry (e.g. palm-ball 30,33) without blurring action targets.")
    p.add_argument("--emphasis-repeat", type=int, default=0)
    p.add_argument("--seed", type=int, default=97)
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
    pred_lo, pred_hi = (int(x) for x in args.possession_dims.split(","))
    emph_lo = emph_hi = None
    if args.emphasis_dims is not None and args.emphasis_repeat > 0:
        emph_lo, emph_hi = (int(x) for x in args.emphasis_dims.split(","))
    conds, chunks = [], []
    bank_states, bank_futures = [], []
    with torch.no_grad():
        for ep in episodes:
            states = ep.states.astype(np.float32)
            actions = ep.actions.astype(np.float32)
            if len(actions) < max_future:
                continue
            held = np.linalg.norm(states[:, pred_lo:pred_hi], axis=-1) < args.possession_threshold
            norm_states_np = norm.encode(states).astype(np.float32)
            norm_states = torch.from_numpy(norm_states_np).to(dev)
            z_online = wm.encode(norm_states)
            z_target = wm.encode_target(norm_states)
            for t in range(len(actions) - max_future + 1):
                for future_h in future_horizons:
                    if args.require_possession == "held" and not (held[t] and held[t + future_h]):
                        continue
                    if args.require_possession == "free" and held[t]:
                        continue
                    h_token = torch.tensor(
                        [float(future_h) / float(max(future_horizons))],
                        dtype=z_online.dtype,
                        device=dev,
                    )
                    parts = [z_online[t], z_target[t + future_h], h_token]
                    if args.concat_raw:
                        parts.extend(
                            [
                                torch.from_numpy(norm_states_np[t]).to(dev),
                                torch.from_numpy(norm_states_np[t + future_h]).to(dev),
                            ]
                        )
                    if emph_lo is not None:
                        parts.append(torch.from_numpy(norm_states_np[t, emph_lo:emph_hi]).to(dev).repeat(args.emphasis_repeat))
                    conds.append(torch.cat(parts, dim=-1))
                    chunks.append(torch.from_numpy(actions[t : t + H].reshape(-1)).to(dev))
                bank_states.append(states[t])
                bank_futures.append(states[t + max(future_horizons)])

    Cond = torch.stack(conds)
    Chunk = torch.stack(chunks)
    bank_states_np = np.asarray(bank_states, dtype=np.float32)
    bank_futures_np = np.asarray(bank_futures, dtype=np.float32)
    if len(bank_states_np) > args.max_bank:
        idx = rng.choice(len(bank_states_np), size=args.max_bank, replace=False)
        bank_states_np = bank_states_np[idx]
        bank_futures_np = bank_futures_np[idx]

    net = EpsNet(Chunk.shape[1], Cond.shape[1], args.hidden, n_blocks=args.n_blocks).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    ema_decay = 0.999
    print(
        json.dumps(
            {
                "event": "flat_flow_data",
                "pairs": int(Cond.shape[0]),
                "bank": int(len(bank_states_np)),
                "chunk": H,
                "cond_dim": int(Cond.shape[1]),
                "chunk_dim": int(Chunk.shape[1]),
                "concat_raw": bool(args.concat_raw),
            }
        ),
        flush=True,
    )
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, Cond.shape[0], (args.batch_size,), device=dev)
        x1 = Chunk[idx]
        cond = Cond[idx]
        x0 = torch.randn_like(x1)
        tau = torch.rand(x1.shape[0], device=dev)
        xt = (1.0 - tau)[:, None] * x0 + tau[:, None] * x1
        pred_v = net(xt, tau * args.flow_steps, cond)
        loss = nn.functional.mse_loss(pred_v, x1 - x0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for key, value in net.state_dict().items():
                ema[key].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
        if step == 1 or step % 2000 == 0:
            print(json.dumps({"event": "flat_flow_train", "step": step, "loss": float(loss.detach().cpu())}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ema": ema,
            "state_dict": net.state_dict(),
            "chunk_dim": int(Chunk.shape[1]),
            "cond_dim": int(Cond.shape[1]),
            "action_dim": int(spec.action_dim),
            "H": int(H),
            "latent_dim": int(cfg["latent_dim"]),
            "hidden": int(args.hidden),
            "n_blocks": int(args.n_blocks),
            "diffusion_steps": int(args.flow_steps),
            "objective": "flow",
            "conditioning": "z_t_z_future",
            "future_horizons": future_horizons,
            "concat_raw": bool(args.concat_raw),
            "require_possession": args.require_possession,
            "possession_threshold": float(args.possession_threshold),
            "emphasis_dims": args.emphasis_dims if emph_lo is not None else None,
            "emphasis_repeat": int(args.emphasis_repeat) if emph_lo is not None else 0,
            "bank_states": bank_states_np,
            "bank_futures": bank_futures_np,
            "model_path": str(args.model_path),
            "episodes_npz": str(args.episodes_npz),
        },
        args.out,
    )
    print(json.dumps({"event": "flat_flow_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
