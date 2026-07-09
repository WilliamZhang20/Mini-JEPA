"""Train a residual diffusion/flow refiner around a future-conditioned flow prior.

The flow prior remains the fast global proposal:

    a_flow ~ flow(a_chunk | z_t, z_future)

The refiner learns the fine correction back to the demonstrated transition:

    residual = a_demo - a_flow
    residual_prior(residual | z_t, z_future, a_flow)

This keeps the control path self-supervised: demos/trials provide transition
chunks and desirable futures, but evaluation does not copy action labels.
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

from jepa_robotics.algos.priors import EpsNet, make_ddpm, sample_action_chunks
from jepa_robotics.data import load_episodes_npz
from jepa_robotics.evaluate import load_jepa_artifact
from scripts.train_fetch_flow_prior import parse_horizons


def build_future_pairs(args, wm, norm, dev):
    episodes = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    H = args.chunk
    future_horizons = args.future_horizons or [H]
    max_future = max(max(future_horizons), H)
    conds, chunks = [], []
    with torch.no_grad():
        for ep in episodes:
            states = ep.states.astype(np.float32)
            actions = ep.actions.astype(np.float32)
            if len(actions) < max_future:
                continue
            norm_states_np = norm.encode(states).astype(np.float32)
            norm_states = torch.from_numpy(norm_states_np).to(dev)
            z_online = wm.encode(norm_states)
            z_target = wm.encode_target(norm_states)
            for t in range(len(actions) - max_future + 1):
                for future_h in future_horizons:
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
                    conds.append(torch.cat(parts, dim=-1))
                    chunks.append(torch.from_numpy(actions[t : t + H].reshape(-1)).to(dev))
    Cond = torch.stack(conds)
    Chunk = torch.stack(chunks)
    if args.max_pairs > 0 and Cond.shape[0] > args.max_pairs:
        idx = torch.randperm(Cond.shape[0], device=dev)[: args.max_pairs]
        Cond = Cond[idx]
        Chunk = Chunk[idx]
    return Cond, Chunk, future_horizons


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--flow-path", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=None)
    p.add_argument("--future-horizons", type=parse_horizons, default=None)
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument("--max-pairs", type=int, default=120000)
    p.add_argument("--proposal-batch", type=int, default=2048)
    p.add_argument("--proposal-mode", choices=["flow", "noisy_data", "mixed"], default="mixed")
    p.add_argument("--proposal-noise-std", type=float, default=0.10)
    p.add_argument("--flow-steps", type=int, default=16)
    p.add_argument("--flow-init-noise-scale", type=float, default=1.0)
    p.add_argument("--concat-raw", action="store_true",
                   help="Append normalized current/future states to latent condition; defaults to the flow checkpoint setting.")
    p.add_argument("--train-steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--diffusion-steps", type=int, default=50)
    p.add_argument("--objective", choices=["diffusion", "flow"], default="diffusion")
    p.add_argument("--cfg-dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=191)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    torch.manual_seed(args.seed)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for param in wm.parameters():
        param.requires_grad_(False)

    flow_ckpt = torch.load(args.flow_path, map_location=dev, weights_only=False)
    if args.chunk is None:
        args.chunk = int(flow_ckpt["H"])
    if args.future_horizons is None:
        args.future_horizons = list(flow_ckpt.get("future_horizons", [args.chunk]))
    if bool(flow_ckpt.get("concat_raw", False)) != bool(args.concat_raw):
        # argparse default is absent; mirror checkpoint unless explicitly passed below.
        args.concat_raw = bool(flow_ckpt.get("concat_raw", False))

    Cond, Chunk, future_horizons = build_future_pairs(args, wm, norm, dev)
    if Cond.shape[1] != int(flow_ckpt["cond_dim"]):
        raise ValueError(f"condition dim mismatch: data {Cond.shape[1]} vs flow ckpt {flow_ckpt['cond_dim']}")
    flow = EpsNet(
        int(flow_ckpt["chunk_dim"]),
        int(flow_ckpt["cond_dim"]),
        int(flow_ckpt["hidden"]),
        n_blocks=int(flow_ckpt["n_blocks"]),
    ).to(dev)
    flow.load_state_dict(flow_ckpt["ema"])
    flow.eval()
    flow_ddpm = make_ddpm(int(flow_ckpt["diffusion_steps"]), dev)

    proposals = []
    with torch.no_grad():
        for start in range(0, Cond.shape[0], args.proposal_batch):
            cond = Cond[start : start + args.proposal_batch]
            if args.proposal_mode == "noisy_data":
                prop = Chunk[start : start + args.proposal_batch] + args.proposal_noise_std * torch.randn_like(Chunk[start : start + args.proposal_batch])
            else:
                prop = sample_action_chunks(
                    flow,
                    flow_ddpm,
                    cond,
                    int(flow_ckpt["chunk_dim"]),
                    dev,
                    objective=str(flow_ckpt.get("objective", "flow")),
                    flow_steps=args.flow_steps,
                    init_noise_scale=args.flow_init_noise_scale,
                )
                if args.proposal_mode == "mixed":
                    mix = torch.rand(prop.shape[0], 1, device=dev) < 0.5
                    noisy = Chunk[start : start + args.proposal_batch] + args.proposal_noise_std * torch.randn_like(prop)
                    prop = torch.where(mix, prop, noisy)
            proposals.append(prop)
    Proposal = torch.cat(proposals, dim=0)
    Residual = Chunk - Proposal
    RefCond = torch.cat([Cond, Proposal], dim=-1)

    net = EpsNet(Residual.shape[1], RefCond.shape[1], args.hidden, n_blocks=args.n_blocks).to(dev)
    ddpm = make_ddpm(args.diffusion_steps, dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    ema_decay = 0.999
    print(
        json.dumps(
            {
                "event": "residual_refiner_data",
                "pairs": int(Cond.shape[0]),
                "cond_dim": int(RefCond.shape[1]),
                "chunk_dim": int(Residual.shape[1]),
                "proposal_mode": args.proposal_mode,
                "residual_l2": float(torch.linalg.norm(Residual, dim=-1).mean().detach().cpu()),
                "flow_path": str(args.flow_path),
            }
        ),
        flush=True,
    )
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, RefCond.shape[0], (args.batch_size,), device=dev)
        x1 = Residual[idx]
        cond = RefCond[idx]
        if args.cfg_dropout > 0:
            keep = (torch.rand(cond.shape[0], 1, device=dev) >= args.cfg_dropout).float()
            cond = cond * keep
        if args.objective == "diffusion":
            t = torch.randint(0, args.diffusion_steps, (args.batch_size,), device=dev)
            noise = torch.randn_like(x1)
            ab = ddpm["abar"][t][:, None]
            x_t = torch.sqrt(ab) * x1 + torch.sqrt(1 - ab) * noise
            pred = net(x_t, t, cond)
            loss = nn.functional.mse_loss(pred, noise)
        else:
            tau = torch.rand(args.batch_size, device=dev)
            x0 = torch.randn_like(x1)
            x_t = (1.0 - tau)[:, None] * x0 + tau[:, None] * x1
            pred = net(x_t, tau * args.diffusion_steps, cond)
            loss = nn.functional.mse_loss(pred, x1 - x0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for key, value in net.state_dict().items():
                ema[key].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
        if step == 1 or step % 2000 == 0:
            print(json.dumps({"event": "residual_refiner_train", "step": step, "loss": float(loss.detach().cpu())}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ema": ema,
            "state_dict": net.state_dict(),
            "chunk_dim": int(Residual.shape[1]),
            "cond_dim": int(RefCond.shape[1]),
            "base_cond_dim": int(Cond.shape[1]),
            "action_dim": int(spec.action_dim),
            "H": int(args.chunk),
            "latent_dim": int(cfg["latent_dim"]),
            "hidden": int(args.hidden),
            "n_blocks": int(args.n_blocks),
            "diffusion_steps": int(args.diffusion_steps),
            "objective": args.objective,
            "conditioning": "z_t_z_future_a_flow",
            "future_horizons": future_horizons,
            "concat_raw": bool(args.concat_raw),
            "flow_path": str(args.flow_path),
            "model_path": str(args.model_path),
            "episodes_npz": str(args.episodes_npz),
            "proposal_mode": args.proposal_mode,
        },
        args.out,
    )
    print(json.dumps({"event": "residual_refiner_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
