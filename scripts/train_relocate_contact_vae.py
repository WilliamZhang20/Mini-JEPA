"""Train a conditional VAE over Relocate contact traces.

This models multimodal contact outcomes:

    p(contact_trace | z_t, raw_t, action_chunk)

The VAE is used as a contact-mode scorer for action candidates, not as an action
policy. It is self-supervised from rollout geometry.
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

from jepa_robotics.algos.contact import ContactTraceCVAE
from jepa_robotics.data import load_episodes_npz
from jepa_robotics.evaluate import load_jepa_artifact
from scripts.train_relocate_contact_dynamics import make_targets


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--max-episodes", type=int, default=2000)
    p.add_argument("--max-pairs", type=int, default=200000)
    p.add_argument("--train-steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--vae-latent-dim", type=int, default=16)
    p.add_argument("--beta", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=367)
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
    states0, actions, targets = [], [], []
    for ep in load_episodes_npz(args.episodes_npz)[: args.max_episodes]:
        s = ep.states.astype(np.float32)
        a = ep.actions.astype(np.float32)
        if len(a) < H or s.shape[-1] < 39:
            continue
        for t in range(len(a) - H + 1):
            states0.append(s[t])
            actions.append(a[t : t + H].reshape(-1))
            targets.append(make_targets(s, t, H))
    if not states0:
        raise RuntimeError("no valid contact VAE pairs")
    S0 = np.asarray(states0, dtype=np.float32)
    A = np.asarray(actions, dtype=np.float32)
    Y = np.asarray(targets, dtype=np.float32)
    if args.max_pairs > 0 and len(S0) > args.max_pairs:
        idx = rng.choice(len(S0), size=args.max_pairs, replace=False)
        S0 = S0[idx]
        A = A[idx]
        Y = Y[idx]

    with torch.no_grad():
        zs, raws = [], []
        for i in range(0, len(S0), 16384):
            raw = torch.from_numpy(norm.encode(S0[i : i + 16384])).to(dev)
            raws.append(raw)
            zs.append(wm.encode(raw).detach())
        Z = torch.cat(zs, dim=0)
        Raw = torch.cat(raws, dim=0)
    Cond = torch.cat([Z, Raw, torch.from_numpy(A).to(dev)], dim=-1)
    Trace = torch.from_numpy(Y).to(dev)

    model = ContactTraceCVAE(
        int(Cond.shape[1]),
        int(Trace.shape[1]),
        args.vae_latent_dim,
        args.hidden,
        args.n_blocks,
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(
        json.dumps(
            {
                "event": "relocate_contact_vae_data",
                "pairs": int(len(Cond)),
                "cond_dim": int(Cond.shape[1]),
                "trace_dim": int(Trace.shape[1]),
                "vae_latent_dim": int(args.vae_latent_dim),
                "chunk": int(H),
            }
        ),
        flush=True,
    )
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, len(Cond), (args.batch_size,), device=dev)
        recon, mu, logvar = model(Cond[idx], Trace[idx])
        recon_loss = nn.functional.smooth_l1_loss(recon, Trace[idx])
        kl = -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())
        loss = recon_loss + args.beta * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 2000 == 0:
            print(
                json.dumps(
                    {
                        "event": "relocate_contact_vae_train",
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "recon": float(recon_loss.detach().cpu()),
                        "kl": float(kl.detach().cpu()),
                    }
                ),
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cond_dim": int(Cond.shape[1]),
            "trace_dim": int(Trace.shape[1]),
            "vae_latent_dim": int(args.vae_latent_dim),
            "hidden": int(args.hidden),
            "n_blocks": int(args.n_blocks),
            "latent_dim": int(cfg["latent_dim"]),
            "state_dim": int(spec.state_dim),
            "action_dim": int(spec.action_dim),
            "H": int(H),
            "model_path": str(args.model_path),
            "episodes_npz": str(args.episodes_npz),
            "beta": float(args.beta),
        },
        args.out,
    )
    print(json.dumps({"event": "relocate_contact_vae_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
